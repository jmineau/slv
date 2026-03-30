import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lair.geo import PC, add_latlon_ticks

from slv.emissions.point_sources import plot_point_sources


def plot_sites(ax, sites, site_config, color="black"):
    for site in sites:
        ax.scatter(
            site_config.at[site, "longitude"],
            site_config.at[site, "latitude"],
            transform=PC,
            label=site.upper(),
            c=color,
            marker="o",
            s=100,
            edgecolor="white",
            linewidth=1.5,
        )

    return ax


def plot_grid(
    grid,
    extent,
    tiler,
    zoom,
    add_sites=True,
    sites=None,
    site_config=None,
    site_color="black",
    add_point_sources=None,
    subplot_kwargs=None,
):
    if add_point_sources is None:
        add_point_sources = {
            "landfill": "yellow",
            "refinery": "cyan",
            "powerplant": "orange",
            "industrial": "purple",
            "wastewater": "blue",
            "unknown": "pink",
        }
    if subplot_kwargs is None:
        subplot_kwargs = {"figsize": (6, 6)}

    fig, ax = plt.subplots(subplot_kw={"projection": tiler.crs}, **subplot_kwargs)

    ax.set_extent(extent, crs=PC)
    ax.add_image(tiler, zoom)

    grid.plot(ax=ax, transform=PC, add_colorbar=False, fc="None", ec="black")

    if add_point_sources:
        for ps_type, color in add_point_sources.items():
            plot_point_sources(ax=ax, kind=ps_type, color=color)
    if add_sites and sites is not None and site_config is not None:
        plot_sites(ax, sites=sites, site_config=site_config, color=site_color)

    add_latlon_ticks(ax, extent, x_rotation=45)

    return fig, ax


def plot_concentrations(obs):
    fig, ax = plt.subplots()

    obs.unstack(level="obs_location").plot(ax=ax, alpha=0.7)

    ax.set(
        title="Valid CH$_4$ Observations",
        xlabel="Time [UTC]",
        ylabel="CH$_4$ [ppm]",
    )
    fig.autofmt_xdate()

    return fig, ax


def plot_inventory(inventory, extent, tiler, zoom, subplot_kwargs=None):
    if subplot_kwargs is None:
        subplot_kwargs = {"figsize": (6, 6)}
    fig, ax = plt.subplots(subplot_kw={"projection": tiler.crs}, **subplot_kwargs)

    ax.set_extent(extent, crs=PC)
    ax.add_image(tiler, zoom)

    inventory.mean(dim="time").plot(
        ax=ax,
        transform=PC,
        add_colorbar=True,
        cmap="Reds",
        alpha=0.55,
        cbar_kwargs={"label": "CH$_4$ Flux [umol/m$^2$/s]"},
    )

    add_latlon_ticks(ax, extent, x_rotation=45)

    ax.set(title="Prior Flux Inventory (mean over time)")

    return fig, ax


def plot_fluxes(
    problem,
    tiler,
    zoom,
    add_sites=True,
    sites=None,
    site_config=None,
    site_color="black",
    add_point_sources=None,
):
    fig, axes = problem.plot.fluxes(tiler=tiler, tiler_zoom=zoom)

    if add_point_sources is None:
        add_point_sources = {
            "landfill": "yellow",
            "refinery": "cyan",
        }

    for ax in axes:
        if add_point_sources:
            for ps_type, color in add_point_sources.items():
                plot_point_sources(ax=ax, kind=ps_type, color=color)
        if add_sites and sites is not None and site_config is not None:
            plot_sites(ax, sites=sites, site_config=site_config, color=site_color)


def plot_fluxes_by_timestep(
    problem,
    extent,
    tiler,
    zoom,
    add_sites=True,
    sites=None,
    site_config=None,
    site_color="black",
    add_point_sources=None,
):
    facet = (
        problem.posterior_fluxes.to_xarray()
        .astype(float)
        .plot(
            col="time",
            col_wrap=8,
            subplot_kws={"projection": tiler.crs},
            transform=PC,
            alpha=0.6,
            cmap="coolwarm",
        )
    )
    for ax in facet.axes.flatten():
        ax.set_extent(extent, crs=PC)
        ax.add_image(tiler, zoom)
        if add_point_sources:
            for ps_type, color in add_point_sources.items():
                plot_point_sources(ax=ax, kind=ps_type, color=color)
        if add_sites and sites is not None and site_config is not None:
            plot_sites(ax, sites=sites, site_config=site_config, color=site_color)

    return facet


def plot_total_fluxes_over_time(*total_fluxes: pd.Series):
    fig, ax = plt.subplots()

    for flux in total_fluxes:
        flux.plot(ax=ax, label=flux.name)

    ax.set(
        title="Total Emissions for Inversion Domain",
        xlabel="Time",
        ylabel="Total CH$_4$ Flux [g/s]",
    )
    fig.autofmt_xdate()

    ax.legend()

    return fig, ax


def plot_mdm_components(components: dict[str, pd.DataFrame]):
    """Plot the total error magnitude for each MDM error component.

    Parameters
    ----------
    components : dict of str to pd.DataFrame
        Dictionary where keys are component names and values are covariance matrices
        as DataFrames with symmetric indexes.

    Notes
    -----
    The total error is computed using the Frobenius norm of the covariance matrix,
    which includes contributions from off-diagonal elements (correlations).
    """
    names = []
    total_errors = []

    for name, cov_matrix in components.items():
        names.append(name)

        # Compute Frobenius norm: sqrt(sum of all squared elements)
        # This includes off-diagonal covariances
        total_error = np.linalg.norm(cov_matrix.values, "fro")
        total_errors.append(total_error)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, total_errors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Color bars with a gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    for bar, color in zip(bars, colors, strict=False):
        bar.set_color(color)

    ax.set_ylabel("Total Error Magnitude [ppm]", fontsize=12)
    ax.set_xlabel("MDM Error Component", fontsize=12)
    ax.set_title("Model-Data Mismatch Error Components", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x-axis labels if needed
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    return fig, ax


def plot_desroziers(
    by_site: pd.DataFrame, per_obs: pd.DataFrame, timeseries: pd.DataFrame
):
    """Plot Desroziers diagnostic: variance comparison, box-whisker, and timeseries.

    Parameters
    ----------
    by_site : pd.DataFrame
        Site-aggregated Desroziers diagnostic (default groupby) with
        'diagnosed', 'specified', and 'ratio' columns.
    per_obs : pd.DataFrame
        Per-observation Desroziers diagnostic (groupby=None) with
        'diagnosed', 'specified', and 'ratio' columns. Index should have
        'obs_location' and 'obs_time' levels.
    timeseries : pd.DataFrame
        Temporally-aggregated Desroziers diagnostic with 'ratio' column.
        Index should have 'obs_location' and 'obs_time' levels.
    """
    import seaborn as sns

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 13))

    # --- Panel 1: side-by-side variance bars by site ---
    x = np.arange(len(by_site))
    width = 0.35
    labels = [str(idx) for idx in by_site.index]

    ax1.bar(
        x - width / 2,
        by_site["specified"],
        width,
        label="Specified (MDM)",
        color="steelblue",
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.bar(
        x + width / 2,
        by_site["diagnosed"],
        width,
        label="Diagnosed (Desroziers)",
        color="coral",
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_ylabel("Mean Variance [ppm$^2$]")
    ax1.set_title(
        "Desroziers Diagnostic: Specified vs Diagnosed Error", fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Panel 2: box-whisker of ratio by site ---
    plot_data = per_obs.reset_index()
    upper = plot_data["ratio"].quantile(0.95)
    sns.boxplot(
        data=plot_data,
        x="obs_location",
        y="ratio",
        ax=ax2,
        showfliers=False,
        palette="Set2",
    )
    ax2.set_ylim(top=max(upper * 1.1, 1.5))
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_ylabel("Ratio (Diagnosed / Specified)")
    ax2.set_xlabel("")
    ax2.set_title("Desroziers Ratio Distribution by Site", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: ratio timeseries per site ---
    ts = timeseries.reset_index()
    for site, group in ts.groupby("obs_location"):
        ax3.plot(
            group["obs_time"], group["ratio"], marker="o", markersize=3, label=site
        )
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax3.set_ylabel("Ratio (Diagnosed / Specified)")
    ax3.set_xlabel("Time")
    ax3.set_title("Desroziers Ratio Over Time", fontweight="bold")
    ax3.legend(title="Site")
    ax3.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    plt.tight_layout()

    return fig, (ax1, ax2, ax3)


def plot_residuals(
    problem,
    rolling_window="30d",
    gap_threshold="7d",
    show_raw=True,
    location_dim="obs_location",
):
    """Plot posterior - observed concentration residuals for all sites.

    Parameters
    ----------
    problem : FluxProblem
        Solved inversion problem.
    rolling_window : str or int, default '30d'
        Rolling window for smoothed lines.
    gap_threshold : str, default '7d'
        Minimum gap duration to highlight as missing data.
    show_raw : bool, default True
        Whether to show raw residual points behind the smoothed lines.
    location_dim : str, default 'obs_location'
        Name of the location dimension in the data.
    """
    obs = problem.concentrations
    posterior = problem.posterior_concentrations
    residuals = (posterior - obs).dropna()

    locations = residuals.index.get_level_values(location_dim).unique()
    gap_td = pd.Timedelta(gap_threshold)

    fig, ax = plt.subplots()

    for location in locations:
        loc_resid = residuals.loc[location]
        color = ax._get_lines.get_next_color()

        if show_raw:
            ax.scatter(
                loc_resid.index,
                loc_resid.values,
                s=4,
                alpha=0.15,
                color=color,
            )

        # Detect gaps
        times = loc_resid.index.sort_values()
        diffs = times.to_series().diff()
        gap_starts = times[diffs > gap_td]

        # Smoothed line with gaps broken
        smoothed = loc_resid.rolling(
            window=rolling_window, center=True, min_periods=1
        ).mean()
        for gap_start in gap_starts:
            gap_begin = times[times < gap_start][-1]
            mid = gap_begin + (gap_start - gap_begin) / 2
            smoothed.loc[mid:gap_start] = np.nan

        ax.plot(
            smoothed.index,
            smoothed.values,
            linewidth=1.5,
            color=color,
            label=location,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()

    # Percentile-based y-limits
    all_vals = residuals.values
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) > 0:
        p01, p99 = np.percentile(all_vals, [1, 99])
        margin = (p99 - p01) * 0.1
        ax.set_ylim(p01 - margin, p99 + margin)

    ax.set(
        title="Concentration Residuals (posterior $-$ observed)",
        ylabel="Residual [ppm]",
        xlabel="Time",
    )
    fig.autofmt_xdate()

    return fig, ax


def plot_background_and_bias(problem):
    """Plot background concentration and bias corrections.

    Parameters
    ----------
    problem : FluxProblem
        Solved inversion problem containing prior, posterior, and constant.

    Notes
    -----
    - If bias exists, creates two subplots: background and bias
    - If no bias, only plots background
    - Background: automatically detects if all sites share the same regional
      background (plots single line) or have site-specific backgrounds (plots
      separate lines)
    - Bias is shown as prior (initial) and posterior (estimated) values
    - For grouped bias (site/site_group), each group gets its own line
    """
    # Check if bias exists
    has_bias = False
    if hasattr(problem, "prior") and hasattr(problem.prior, "data"):
        # Check if prior has a "block" level in its index
        prior_index = problem.prior.data.index
        if isinstance(prior_index, pd.MultiIndex) and "block" in prior_index.names:
            # Check if "bias" is one of the block values
            has_bias = "bias" in prior_index.get_level_values("block").unique()

    # Set up subplots
    n_plots = 2 if has_bias else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Plot background
    if hasattr(problem, "constant") and problem.constant is not None:
        bg_data = problem.constant["concentration"]

        # Get unique sites
        if isinstance(bg_data.index, pd.MultiIndex):
            sites = bg_data.index.get_level_values("obs_location").unique()

            # Check if all sites have identical background values
            site_series = {
                site: bg_data.xs(site, level="obs_location") for site in sites
            }

            # Compare all site series to the first one
            first_site = list(site_series.keys())[0]
            first_series = site_series[first_site]
            all_identical = all(
                site_series[site].equals(first_series) for site in sites
            )

            if all_identical:
                # All sites share the same background - plot just one line
                axes[0].plot(
                    first_series.index,
                    first_series.values,
                    color="blue",
                    linewidth=2,
                    label="Regional Background (all sites)",
                )
            else:
                # Sites have different backgrounds - plot each separately
                for site in sites:
                    site_data = site_series[site]
                    axes[0].plot(
                        site_data.index,
                        site_data.values,
                        label=site,
                        alpha=0.7,
                        linewidth=1.5,
                    )
        else:
            axes[0].plot(
                bg_data.index,
                bg_data.values,
                color="blue",
                linewidth=2,
                label="Regional Background",
            )

        axes[0].set_ylabel("Background [ppm]", fontsize=12)
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_title("Background Concentration", fontsize=14, fontweight="bold")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0].grid(True, alpha=0.3)

    # Plot bias if it exists
    if has_bias:
        prior_bias = problem.prior["bias"]
        posterior_bias = problem.posterior["bias"]

        # Check if bias has grouping (MultiIndex)
        if isinstance(prior_bias.index, pd.MultiIndex):
            # Grouped bias (site or site_group)
            group_level = prior_bias.index.names[
                1
            ]  # Should be 'obs_location' or 'site_group'
            groups = prior_bias.index.get_level_values(group_level).unique()

            # Get the default color cycle
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for i, group in enumerate(groups):
                prior_group = prior_bias.xs(group, level=group_level)
                posterior_group = posterior_bias.xs(group, level=group_level)

                # Use same color for prior and posterior of this group
                color = colors[i % len(colors)]

                # Plot prior as dashed, posterior as solid
                axes[1].plot(
                    prior_group.index,
                    prior_group.values,
                    "--",
                    label=f"{group} (prior)",
                    color=color,
                    alpha=0.6,
                    linewidth=1.5,
                )
                axes[1].plot(
                    posterior_group.index,
                    posterior_group.values,
                    "-",
                    label=f"{group} (posterior)",
                    color=color,
                    linewidth=2,
                )
        else:
            # Time-only bias
            axes[1].plot(
                prior_bias.index,
                prior_bias.values,
                "--",
                label="Prior",
                color="gray",
                alpha=0.6,
                linewidth=1.5,
            )
            axes[1].plot(
                posterior_bias.index,
                posterior_bias.values,
                "-",
                label="Posterior",
                color="red",
                linewidth=2,
            )

        axes[1].axhline(y=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        axes[1].set_ylabel("Bias [ppm]", fontsize=12)
        axes[1].set_xlabel("Time", fontsize=12)
        axes[1].set_title("Bias Correction", fontsize=14, fontweight="bold")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes
