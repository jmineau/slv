"""Tests for plot_point_sources."""

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from slv.emissions.point_sources import _load_points, plot_point_sources  # noqa: E402


def test_plots_one_marker_per_source():
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    try:
        plot_point_sources("landfill", ax)
        n_landfills = (_load_points()["category"] == "landfill").sum()
        assert len(ax.collections) == n_landfills
    finally:
        plt.close(fig)
