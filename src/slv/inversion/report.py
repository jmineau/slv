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


def _print_fit(post: Posterior) -> None:
    """The fit of the solve, its fluxes and the modeled concentrations, printed."""
    from scipy import stats

    problem = post.problem
    obs = problem.obs.to_series("concentration")
    background = 0.0
    if problem.offset is not None:
        background = problem.offset.to_series("concentration")
    enhancement = obs - background
    prior_flux = problem.prior.to_series("flux")
    post_flux = post.estimate.to_series("flux")
    prior_obs = problem.predict(problem.prior).to_series("concentration")
    post_obs = post.modeled_obs.to_series("concentration")

    # --- Atmospheric metrics ---
    mean_enhancement = np.mean(enhancement)
    rmse_prior = np.sqrt(((prior_obs - obs) ** 2).mean())
    rmse_post = np.sqrt(((post_obs - obs) ** 2).mean())
    mb_prior = np.mean(prior_obs - obs)
    mb_post = np.mean(post_obs - obs)
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    r2_prior = 1 - (((obs - prior_obs) ** 2).sum() / ss_tot)
    r2_post = 1 - (((obs - post_obs) ** 2).sum() / ss_tot)

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
    print(f"Bayesian R^2:                {post.r2:.3f}")
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
    print(f"  RMSE (Prior):         {rmse_prior:.4f}")
    print(f"  RMSE (Posterior):     {rmse_post:.4f}")
    print(f"  Mean Bias (Prior):    {mb_prior:.4f}")
    print(f"  Mean Bias (Posterior):{mb_post:.4f}")
    print(f"  R^2 (Prior):          {r2_prior:.3f}")
    print(f"  R^2 (Posterior):      {r2_post:.3f}")
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
