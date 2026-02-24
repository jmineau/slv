import matplotlib.pyplot as plt
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

    inventory.mean(dim="time").plot(ax=ax, transform=PC, add_colorbar=True, cmap="Reds")

    add_latlon_ticks(ax, extent, x_rotation=45)

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
        .astyle(float)
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

    return fig, ax
