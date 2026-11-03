"""
Domain totals, the printed summary, Desroziers diagnostics and the plots of an SLV inversion.

Each function takes the solved :class:`~fips.Posterior` (``post``) and, where the whole
state is needed, the :class:`~fips.Problem` :func:`~slv.inversion.pipelines.build_problem`
gave (``problem``): with the coverage filter on, ``post.problem`` holds only the kept
cells.
"""

from collections.abc import Hashable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fips import Posterior, Problem, Vector
from lair import inventories

from slv.inversion import viz
from slv.inversion.config import InversionConfig


def fluxes_as_inventory(
    config: InversionConfig, fluxes: pd.Series
) -> inventories.Inventory:
    """Converts a flux vector to an inventory format for easier analysis."""
    ds = fluxes.to_xarray().to_dataset()

    from slv.inversion.config import FLUX_FREQ_TIME_STEPS

    time_step = FLUX_FREQ_TIME_STEPS[config.flux_freq]

    return inventories.Inventory(
        ds, pollutant="CH4", src_units="umol/m2/s", time_step=time_step
    )


def total_units(config: InversionConfig) -> str:
    """
    Unit of :func:`calculate_total_flux`: it integrates each flux interval over its
    cells and its duration, so the mass part of ``output_units`` per interval.
    """
    units = config.output_units or "umol/m2/s"
    return f"{units.split('/')[0]} per {config.flux_freq} interval"


def calculate_total_flux(
    config: InversionConfig, fluxes: pd.Series, units=None
) -> pd.Series:
    """
    Domain-total emission per flux interval.

    The flux field is integrated over its cells' areas and each interval's duration
    (lair ``absolute_emissions``), so the result is a mass per interval, e.g. Gg per
    ``MS`` interval for ``units="Gg/m2/s"``.

    Parameters
    ----------
    config : InversionConfig
        The inversion (its ``flux_freq``).
    fluxes : pd.Series
        Flux in umol/m2/s indexed by (time, lat, lon); its ``name`` must be set.
    units : str, optional
        Convert to these flux units before integrating (e.g. ``"Gg/m2/s"``).

    Returns
    -------
    pd.Series
        Total per flux time.
    """
    inventory = fluxes_as_inventory(config, fluxes)
    if units:
        inventory = inventory.convert_units(units)
    return inventory.absolute_emissions[fluxes.name].sum(dim=("lat", "lon")).to_series()


def desroziers_diagnostic(
    post: Posterior,
    groupby: str | list[str] | None = "obs_location",
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Compare Desroziers-diagnosed vs specified observation error variances.

    Parameters
    ----------
    post : Posterior
        The solved inversion.
    groupby : str, list of str, or None
        Index level(s) to aggregate by. Default groups by site.
        Use None for per-observation values.
    freq : str, optional
        Temporal resampling frequency (e.g. 'MS', 'QS', 'YS').
        Groups obs_time into bins at this frequency in addition to
        any levels in ``groupby``.

    Returns
    -------
    pd.DataFrame
        Columns: diagnosed, specified (the obs error variances the solve used, so
        divided by ``gamma``), ratio (diagnosed / specified).
    """
    groupers: list = []
    if groupby is not None:
        groupers.extend([groupby] if isinstance(groupby, str) else groupby)
    if freq is not None:
        groupers.append(pd.Grouper(level="obs_time", freq=freq))
    return post.desroziers(by=groupers or None)


def _totals(
    config: InversionConfig, post: Posterior, problem: Problem
) -> tuple[pd.Series, pd.Series]:
    """
    Prior and posterior domain totals over every cell of ``problem``.

    With the coverage filter on, the cells left out of ``post.problem`` are at their
    prior in the posterior.
    """
    estimate = post.estimate.combine_first(problem.prior)
    prior = problem.prior.to_series("flux").rename("prior")
    posterior = estimate.to_series("flux").rename("posterior")
    return (
        calculate_total_flux(config, prior, units=config.output_units),
        calculate_total_flux(config, posterior, units=config.output_units),
    )


def _obs_and_background(problem: Problem) -> tuple[pd.Series, pd.Series | float]:
    """The observed concentration of ``problem`` and its background (0 if it has none)."""
    obs = problem.obs.to_series("concentration")
    if problem.offset is None:
        return obs, 0.0
    return obs, problem.offset.to_series("concentration")


def fit_metrics(post: Posterior) -> pd.DataFrame:
    r"""
    The fit of the prior and the posterior to the observations, one row each.

    The one place the sweep table (:func:`~slv.inversion.sweep.collect_metrics`) and
    the printed report (:func:`summarize`) take their fit numbers from, so the two
    always agree. The modeled concentration is :math:`H x + c`, with :math:`c` the
    background (the problem's offset), and the residual is the modeled minus the
    observed concentration. Let :math:`z` be the observed concentration and
    :math:`e = z - c` the observed enhancement.

    Parameters
    ----------
    post :
        The solved :class:`~fips.Posterior`; ``post.problem`` supplies the
        observations, the background and the prior.

    Returns
    -------
    pandas.DataFrame
        Index ``prior`` and ``posterior``; columns, in the units of the observations
        except the two R²:

        ``conc_rmse``
            The root mean square of the residual.
        ``conc_mean_bias``
            The mean of the residual, **model minus observed**: positive when the
            model is above the observations.
        ``enh_r2``
            The R² of the enhancement variance,
            :math:`1 - \sum (e - m)^2 / \sum (e - \bar{e})^2`, with :math:`m` the
            modeled enhancement :math:`H x`. It penalizes a bias or a wrong scale
            (a model twice too large everywhere scores far below 1), and is negative
            when the model fits worse than the mean enhancement. Paper 1 quotes this
            one.
        ``enh_corr2``
            The squared correlation of :math:`e` and :math:`m`. It ignores a bias
            and a scale (a model twice too large everywhere scores 1). For the
            posterior it is fips's :attr:`~fips.Posterior.r2`.

        Both R² are taken on the enhancement, not on the concentration: the
        background follows the observations from day to day, so an R² of the
        concentration scores the background's fit, not the state's. Both are
        ``nan`` when the observed enhancement is constant.

    See Also
    --------
    summarize : Prints these numbers.
    slv.inversion.sweep.collect_metrics : Puts them in the sweep table.

    Examples
    --------
    A model that is 1 above the observations everywhere has a mean bias of +1, and
    an enhancement R² below its squared correlation of 1:

    >>> import fips
    >>> import numpy as np
    >>> import pandas as pd
    >>> from slv.inversion.report import fit_metrics
    >>> index = pd.MultiIndex.from_product(
    ...     [["wbb"], pd.date_range("2020-01-01", periods=4)],
    ...     names=["obs_location", "obs_time"],
    ... )
    >>> x0 = fips.Vector.from_series(
    ...     pd.Series([1.0], index=pd.Index([0], name="cell")), block="flux"
    ... )
    >>> obs = fips.Vector.from_series(
    ...     pd.Series([1.0, 2.0, 3.0, 4.0], index=index), block="concentration"
    ... )
    >>> H = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> problem = fips.Problem(
    ...     prior=x0,
    ...     prior_cov=fips.Independent().with_std(0.1),
    ...     obs=obs,
    ...     obs_cov=fips.Independent().with_std(1.0),
    ...     forward=fips.Operator(H, obs.space, x0.space),
    ...     offset=1.0,  # the model is H x + 1: 1 above these observations
    ... )
    >>> post = fips.solve(problem)
    >>> fit_metrics(post).loc["prior", ["conc_mean_bias", "enh_corr2"]].round(
    ...     2
    ... ).tolist()
    [1.0, 1.0]
    """
    problem = post.problem
    obs, background = _obs_and_background(problem)
    enhancement = obs - background
    ss_tot = float(((enhancement - enhancement.mean()) ** 2).sum())

    modeled = {
        "prior": problem.predict(problem.prior),
        "posterior": post.modeled_obs,
    }
    rows = {}
    for state, vector in modeled.items():
        model = vector.to_series("concentration")
        residual = model - obs
        ss_res = float((residual**2).sum())
        rows[state] = {
            "conc_rmse": float(np.sqrt((residual**2).mean())),
            "conc_mean_bias": float(residual.mean()),
            "enh_r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "enh_corr2": float(enhancement.corr(model - background) ** 2),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _print_fit(post: Posterior) -> None:
    """The fit of the solve, its fluxes and the modeled concentrations, printed."""
    from scipy import stats

    problem = post.problem
    obs, background = _obs_and_background(problem)
    enhancement = obs - background
    prior_flux = problem.prior.to_series("flux")
    post_flux = post.estimate.to_series("flux")

    # --- Atmospheric metrics ---
    mean_enhancement = np.mean(enhancement)
    fit = fit_metrics(post)
    prior_fit, post_fit = fit.loc["prior"], fit.loc["posterior"]

    # --- Flux and Bayesian metrics ---
    n = len(prior_flux)
    m = len(obs)
    dofs = post.dofs
    n_negative = int((post_flux < 0).sum())
    prior_by_time = prior_flux.groupby(level="time").mean()
    post_by_time = post_flux.groupby(level="time").mean()
    change_pct = ((post_by_time / prior_by_time) - 1) * 100

    print("==================================================")
    print("          FLUX INVERSION SUMMARY REPORT           ")
    print("==================================================")
    print(f"State Vector Size (n):       {n}")
    print(f"Observation Vector Size (m): {m}")
    print(f"Degrees of Freedom (DOFS):   {dofs:.2f} ({(dofs / n) * 100:.1f}% of state)")
    print(f"Reduced Chi-Square:          {post.reduced_chi2:.3f}")
    print(f"Bayesian RMSE:               {post.rmse:.3f}")
    print(f"Uncertainty Reduction:       {post.uncertainty_reduction * 100:.1f}%")
    print("--------------------------------------------------")
    print("FLUX ESTIMATES (per grid cell):")
    print(f"  Mean Prior Flux:     {prior_flux.mean():.4f}")
    print(f"  Mean Posterior Flux: {post_flux.mean():.4f}")
    print(f"  Negative Cells:     {n_negative} / {n} ({n_negative / n * 100:.1f}%)")
    print("--------------------------------------------------")
    print("ATMOSPHERIC CONCENTRATIONS (Model vs Obs):")
    print(f"  Mean Obs Enhancement: {mean_enhancement:.4f}")
    print(f"  RMSE (Prior):         {prior_fit.conc_rmse:.4f}")
    print(f"  RMSE (Posterior):     {post_fit.conc_rmse:.4f}")
    print(f"  Mean Bias, model - obs (Prior):     {prior_fit.conc_mean_bias:.4f}")
    print(f"  Mean Bias, model - obs (Posterior): {post_fit.conc_mean_bias:.4f}")
    print(f"  Enhancement R^2, 1 - SS_res/SS_tot (Prior):     {prior_fit.enh_r2:.3f}")
    print(f"  Enhancement R^2, 1 - SS_res/SS_tot (Posterior): {post_fit.enh_r2:.3f}")
    print(
        f"  Enhancement corr^2 (Prior):                     {prior_fit.enh_corr2:.3f}"
    )
    print(f"  Enhancement corr^2 (Posterior):                 {post_fit.enh_corr2:.3f}")
    print("--------------------------------------------------")
    print("MEAN FLUX BY TIMESTEP:")
    timestep_df = pd.DataFrame(
        {"prior": prior_by_time, "posterior": post_by_time, "change_%": change_pct}
    )
    print(timestep_df.to_string(float_format="%.4f"))
    if len(post_by_time) > 2:
        t = pd.DatetimeIndex(post_by_time.index)
        t_years = t.year + (t.day_of_year - 1) / 365.25
        slope, _, r_value, p_value, _ = stats.linregress(t_years, post_by_time.values)
        print("--------------------------------------------------")
        print("POSTERIOR FLUX TREND:")
        print(
            f"  Slope:   {slope:+.4f} /yr\n"
            f"  R^2:     {r_value**2:.3f}\n"
            f"  p-value: {p_value:.3g}"
        )
    print("==================================================")


def summarize(
    config: InversionConfig, post: Posterior, problem: Problem | None = None
) -> None:
    """
    Print the fit of the solve, then prior and posterior domain totals.

    The fit against the observed concentrations (RMSE, mean bias as model minus
    observed, and the two R² of the enhancement) is :func:`fit_metrics`, the same
    numbers the sweep table holds.

    Totals are per flux interval (:func:`calculate_total_flux`) in
    ``config.output_units``, over every cell of ``problem`` (``post.problem`` by
    default): with the coverage filter on, the cells left out of the solve are at their
    prior, so both totals cover the full domain. With more than two intervals a linear
    trend of the posterior is printed.
    """
    from scipy import stats

    problem = post.problem if problem is None else problem
    _print_fit(post)
    total_prior, total_post = _totals(config, post, problem)
    units = total_units(config)

    print("--------------------------------------------------")
    print(f"DOMAIN TOTAL EMISSIONS [{units}]:")
    summary = pd.DataFrame(
        {
            "prior": total_prior,
            "posterior": total_post,
            "change_%": ((total_post / total_prior) - 1) * 100,
        }
    )
    print(summary.to_string(float_format="%.2f"))
    print(f"  Mean Prior:     {total_prior.mean():.2f}")
    print(f"  Mean Posterior: {total_post.mean():.2f}")
    print(
        f"  Mean Change:    {((total_post.mean() / total_prior.mean()) - 1) * 100:+.1f}%"
    )

    if len(total_post) > 2:
        t = pd.DatetimeIndex(total_post.index)
        t_years = t.year + (t.day_of_year - 1) / 365.25
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            t_years, total_post.values
        )
        print(
            f"  Trend:          {slope:+.2f} [{units}]/yr (R^2={r_value**2:.3f}, p={p_value:.3g})"
        )
    print("==================================================")


def _cells(vector: Vector) -> set:
    """The (lat, lon) cells of a vector's ``flux`` block."""
    flux = vector.space.select("flux")
    return set(
        zip(flux.get_level_values("lat"), flux.get_level_values("lon"), strict=True)
    )


def plot_obs(
    post: Posterior,
    location: Hashable | list | None = None,
    location_level: str = "obs_location",
    rolling_window: str | int = "30D",
    gap_threshold: str = "7D",
    show_raw: bool = True,
) -> np.ndarray:
    """
    Plot the obs against what the prior and the posterior model, one figure per location.

    Three series over time: the observed concentrations, the posterior's modeled ones
    (``post.modeled_obs``) and the prior's (``problem.predict(problem.prior)``), each as
    points and as a rolling mean. A gap in a location's times longer than
    ``gap_threshold`` is shaded, and the rolling means break there.

    Parameters
    ----------
    post : Posterior
        The solved inversion.
    location : label or list of labels, optional
        The location (or locations) to plot. By default, every location.
    location_level : str, default 'obs_location'
        The level of the obs' labels that holds the locations.
    rolling_window : str or int, default '30D'
        The window of the rolling means, as ``DataFrame.rolling`` takes it.
    gap_threshold : str, default '7D'
        The shortest gap between two times that is shaded as missing data.
    show_raw : bool, default True
        Whether to draw the points behind the rolling means.

    Returns
    -------
    numpy.ndarray
        The axes, one per location.
    """
    from matplotlib.artist import Artist
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    problem = post.problem
    data = pd.concat(
        [
            problem.obs.to_series("concentration").rename("observed"),
            post.modeled_obs.to_series("concentration").rename("posterior"),
            problem.predict(problem.prior).to_series("concentration").rename("prior"),
        ],
        axis=1,
    )
    if location is None:
        locations = list(data.index.get_level_values(location_level).unique())
    elif isinstance(location, list):
        locations = location
    else:
        locations = [location]

    colors = ["black", "red", "blue"]
    gap = pd.Timedelta(gap_threshold)
    axes = []
    for loc in locations:
        df = data.xs(loc, level=location_level).sort_index()
        df.columns.name = None
        fig, ax = plt.subplots()
        if show_raw:
            df.plot(
                ax=ax,
                style=".",
                alpha=0.3,
                markersize=4,
                color=colors,
                markeredgecolor="None",
                legend=False,
            )

        # The gaps in the times; the rolling means are NaN from the middle of each gap
        # to its end, which breaks the lines
        times = df.index
        gap_ends = times[times.to_series().diff() > gap]
        smoothed = df.rolling(window=rolling_window, center=True, min_periods=1).mean()
        for gap_end in gap_ends:
            gap_start = times[times < gap_end][-1]
            smoothed.loc[gap_start + (gap_end - gap_start) / 2 : gap_end] = np.nan
        smoothed.plot(ax=ax, linewidth=1.5, color=colors, legend=False)
        for gap_end in gap_ends:
            gap_start = times[times < gap_end][-1]
            ax.axvspan(gap_start, gap_end, color="grey", alpha=0.15)

        # The legend shows the rolling means only
        handles: list[Artist] = [
            Line2D([], [], color=c, linewidth=1.5, label=label)
            for c, label in zip(colors, df.columns, strict=False)
        ]
        if len(gap_ends) > 0:
            handles.append(Patch(facecolor="grey", alpha=0.15, label="no data"))
        ax.legend(handles=handles)

        # y-limits from the 1st and 99th percentiles, so outliers do not flatten the rest
        values = df.to_numpy().ravel()
        values = values[~np.isnan(values)]
        if len(values) > 0:
            p01, p99 = np.percentile(values, [1, 99])
            margin = (p99 - p01) * 0.1
            ax.set_ylim(p01 - margin, p99 + margin)

        ax.set(title=f"concentration at {loc}", ylabel="concentration", xlabel="time")
        fig.autofmt_xdate()
        axes.append(ax)
    return np.array(axes)


def plot_inputs(config: InversionConfig, problem: Problem, solved: Problem) -> None:
    """
    Plot the grid and sites, the obs time series and the prior fluxes.

    ``solved`` is the problem the solve gets: ``problem`` itself, or, with the coverage
    filter on, its restriction to the covered cells (the others are shaded).
    """
    viz.plot_grid(
        config.grid,
        extent=config.map_extent,
        tiler=config.tiler,
        zoom=config.tiler_zoom,
        sites=config.sites,
        site_config=config.site_config,
    )
    viz.plot_concentrations(problem.obs.to_series("concentration"))

    prior = problem.prior.to_series("flux").rename("prior").to_xarray()
    if len(solved.prior) < len(problem.prior):
        viz.plot_prior_with_coverage(
            prior,
            _cells(solved.prior),
            _cells(problem.prior),
            dx=config.dx,
            dy=config.dy,
            extent=config.map_extent,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
        )
    else:
        viz.plot_inventory(
            prior, extent=config.map_extent, tiler=config.tiler, zoom=config.tiler_zoom
        )
    plt.show()


def plot_results(config: InversionConfig, post: Posterior, problem: Problem) -> None:
    """
    Plot the posterior fluxes, the full-domain posterior (with the coverage filter on),
    domain totals over time, modelled vs observed concentrations, residuals, and
    background and bias.
    """
    viz.plot_fluxes(
        post,
        tiler=config.tiler,
        zoom=config.tiler_zoom,
        add_sites=True,
        sites=config.sites,
        site_config=config.site_config,
    )

    if len(post.problem.prior) < len(problem.prior):
        estimate = post.estimate.combine_first(problem.prior)
        viz.plot_reconstructed_posterior(
            estimate.to_series("flux").rename("posterior").to_xarray(),
            _cells(post.problem.prior),
            _cells(problem.prior),
            dx=config.dx,
            dy=config.dy,
            extent=config.map_extent,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
            sites=config.sites,
            site_config=config.site_config,
        )

    total_prior, total_posterior = _totals(config, post, problem)
    viz.plot_total_fluxes_over_time(
        total_prior, total_posterior, units=total_units(config)
    )
    plot_obs(post)
    viz.plot_residuals(post)
    viz.plot_background_and_bias(post)
    plt.show()


def plot_diagnostics(
    config: InversionConfig, post: Posterior, problem: Problem
) -> None:
    """
    Plot fluxes per time step, the Desroziers diagnostics and, with the coverage
    filter on, the left-out cells' contribution to the obs.
    """
    viz.plot_fluxes_by_timestep(
        post,
        extent=config.map_extent,
        tiler=config.tiler,
        zoom=config.tiler_zoom,
        add_sites=True,
        sites=config.sites,
        site_config=config.site_config,
    )
    viz.plot_desroziers(
        by_site=desroziers_diagnostic(post),
        per_obs=desroziers_diagnostic(post, groupby=None),
        timeseries=desroziers_diagnostic(post, freq=config.flux_freq),
    )

    solved = post.problem
    filtered = len(solved.prior) < len(problem.prior)
    if filtered and solved.offset is not None and problem.offset is not None:
        offset = solved.offset.to_series("concentration")
        removed = offset - problem.offset.to_series("concentration")
        viz.plot_removed_contribution(removed, offset)
    plt.show()
