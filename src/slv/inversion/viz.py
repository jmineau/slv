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
    add_point_sources={
        "landfill": "yellow",
        "refinery": "cyan",
        "powerplant": "orange",
        "industrial": "purple",
        "wastewater": "blue",
        "unknown": "pink",
    },
    subplot_kwargs={"figsize": (6, 6)},
):
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

    obs.unstack(level="obs_location").resample("h").mean().plot(ax=ax, alpha=0.7)

    ax.set(
        title="Hourly Averaged Valid CH$_4$ Observations",
        xlabel="Time [UTC]",
        ylabel="CH$_4$ [ppm]",
    )
    fig.autofmt_xdate()

    return fig, ax


def plot_inventory(inventory, extent, tiler, zoom, subplot_kwargs={"figsize": (6, 6)}):
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
