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


def test_every_category_has_its_own_marker():
    from slv.emissions.point_sources import markers

    categories = set(_load_points()["category"])
    assert categories <= set(markers)  # lpg used to fall back to the default dot
    assert len({markers[c] for c in categories}) == len(categories)


def test_names_are_spelled_right():
    names = _load_points()["name"]
    assert "Northrop Grumman Industry" in set(names)
    assert not names.str.contains("Grummand|Northrup").any()
