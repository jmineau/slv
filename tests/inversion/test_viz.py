"""Tests for slv.inversion.viz on synthetic inputs (Agg backend, no map tiles)."""

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from lair.geo import PC

from slv.inversion import viz

SITE_CONFIG = pd.DataFrame(
    {
        "type": ["stationary", "stationary", "mobile"],
        "latitude": [40.76, 40.60, 40.70],
        "longitude": [-111.85, -111.90, -111.95],
    },
    index=pd.Index(["wbb", "hdp", "trx01"], name="stid"),
)
TIMES = pd.date_range("2020-01-01", periods=3, freq="MS")
EXTENT = (-112.1, -111.7, 40.4, 40.9)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def tiler(monkeypatch):
    """A stand-in tile source: its CRS only; add_image is recorded, never fetched."""
    calls = []
    monkeypatch.setattr(GeoAxes, "add_image", lambda self, *a, **k: calls.append(a))
    return SimpleNamespace(crs=PC, calls=calls)


def flux_field(values=None):
    lat = [40.525, 40.575]
    lon = [-111.975, -111.925]
    data = np.ones((len(TIMES), 2, 2)) if values is None else values
    return xr.DataArray(
        data,
        coords={"time": TIMES, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
    )


def obs_series(values_by_site, times):
    index = pd.MultiIndex.from_product(
        [list(values_by_site), pd.DatetimeIndex(times)],
        names=["obs_location", "obs_time"],
    )
    return pd.Series(np.concatenate(list(values_by_site.values())), index=index)


# --------------------------------------------------------------------------- sites


def test_plot_sites_one_handle_per_kind():
    fig, ax = plt.subplots(subplot_kw={"projection": PC})
    handles, labels = viz.plot_sites(ax, ["wbb", "hdp", "trx01"], SITE_CONFIG)
    assert labels == ["Stationary Site", "Mobile Site"]
    assert len(handles[0].get_offsets()) == 2 and len(handles[1].get_offsets()) == 1


def test_plot_sites_stationary_only():
    fig, ax = plt.subplots(subplot_kw={"projection": PC})
    _, labels = viz.plot_sites(ax, ["wbb"], SITE_CONFIG)
    assert labels == ["Stationary Site"]


# --------------------------------------------------------------------------- maps


def test_plot_grid_draws_cells_sites_and_point_sources(tiler):
    grid = flux_field().isel(time=0, drop=True) * 0
    fig, ax = viz.plot_grid(
        grid, EXTENT, tiler, 10, sites=["wbb", "trx01"], site_config=SITE_CONFIG
    )
    assert tiler.calls == [(tiler, 10)]
    labels = ax.get_legend_handles_labels()[1]
    assert "Stationary Site" in labels and "Mobile Site" in labels
    legend_text = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Sites" in legend_text and "Point Sources" in legend_text


def test_plot_inventory_maps_the_time_mean(tiler):
    values = np.stack([np.full((2, 2), v) for v in (1.0, 2.0, 6.0)])
    fig, ax = viz.plot_inventory(flux_field(values), EXTENT, tiler, 10)
    mesh = ax.collections[0]
    np.testing.assert_allclose(mesh.get_array().compressed(), 3.0)
    assert fig.axes[-1].get_ylabel() == "CH$_4$ Flux [umol/m$^2$/s]"


def test_shade_removed_cells_hatches_only_removed():
    fig, ax = plt.subplots(subplot_kw={"projection": PC})
    all_cells = {(40.525, -111.975), (40.525, -111.925), (40.575, -111.975)}
    viz.shade_removed_cells(ax, {(40.525, -111.975)}, all_cells, 0.05, 0.05)
    corners = sorted(p.get_xy() for p in ax.patches)
    # lower-left corners of the two removed cells
    np.testing.assert_allclose(corners, [(-112.0, 40.55), (-111.95, 40.5)])
    assert all(p.get_hatch() == "///" for p in ax.patches)


@pytest.mark.parametrize(
    ("func", "title"),
    [
        (viz.plot_prior_with_coverage, "Prior"),
        (viz.plot_reconstructed_posterior, "Posterior"),
    ],
)
def test_coverage_maps_shade_the_removed_cells(tiler, func, title):
    all_cells = {(40.525, -111.975), (40.525, -111.925)}
    fig, ax = func(
        flux_field(), {(40.525, -111.975)}, all_cells, 0.05, 0.05, EXTENT, tiler, 10
    )
    assert title in ax.get_title()
    assert len(ax.patches) == 1


def test_plot_fluxes_by_timestep_one_panel_per_time(tiler):
    problem = SimpleNamespace(posterior_fluxes=flux_field().to_series())
    facet = viz.plot_fluxes_by_timestep(problem, EXTENT, tiler, 10, add_sites=False)
    assert facet.axs.size >= len(TIMES)
    assert len([ax for ax in facet.axs.flat if ax.has_data()]) == len(TIMES)
    assert len(tiler.calls) == len(facet.axs.flat)


def test_plot_fluxes_adds_sites_and_point_sources_to_each_panel(tiler):
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": PC})
    problem = SimpleNamespace(
        plot=SimpleNamespace(fluxes=lambda **kwargs: (fig, list(axes)))
    )
    viz.plot_fluxes(problem, tiler, 10, sites=["wbb"], site_config=SITE_CONFIG)
    for ax in axes:
        labels = ax.get_legend_handles_labels()[1]
        assert "Stationary Site" in labels
        assert len(ax.collections) > 1  # the landfills and refineries too


# --------------------------------------------------------------------------- series


def test_plot_concentrations_one_line_per_location():
    obs = obs_series({"wbb": [2.0, 2.1], "hdp": [2.2, 2.3]}, TIMES[:2])
    fig, ax = viz.plot_concentrations(obs)
    assert sorted(line.get_label() for line in ax.get_lines()) == ["hdp", "wbb"]
    assert ax.get_ylabel() == "CH$_4$ [ppm]"


def test_plot_total_fluxes_over_time_labels_units():
    prior = pd.Series([1.0, 2.0, 3.0], index=TIMES, name="prior")
    post = pd.Series([1.5, 2.5, 3.5], index=TIMES, name="posterior")
    fig, ax = viz.plot_total_fluxes_over_time(prior, post, units="Gg per MS interval")
    assert [line.get_label() for line in ax.get_lines()] == ["prior", "posterior"]
    assert ax.get_ylabel() == "Total CH$_4$ emissions [Gg per MS interval]"
    fig, ax = viz.plot_total_fluxes_over_time(prior)
    assert ax.get_ylabel() == "Total CH$_4$ emissions"


def test_plot_removed_contribution_two_panels():
    background = obs_series({"wbb": [2.0, 2.1, 2.2]}, TIMES)
    removed = background * 0.01
    fig, axes = viz.plot_removed_contribution(removed, background)
    np.testing.assert_allclose(axes[0].get_lines()[0].get_ydata(), [2.0, 2.1, 2.2])
    np.testing.assert_allclose(
        axes[1].get_lines()[0].get_ydata(), [0.020, 0.021, 0.022]
    )


def test_plot_mdm_components_bars_are_frobenius_norms():
    index = pd.RangeIndex(2)
    comps = {
        "instr": pd.DataFrame(np.diag([3.0, 4.0]), index, index),
        "bg": pd.DataFrame(np.ones((2, 2)), index, index),
    }
    fig, ax = viz.plot_mdm_components(comps)
    heights = [bar.get_height() for bar in ax.patches]
    np.testing.assert_allclose(heights, [5.0, 2.0])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["instr", "bg"]


def test_plot_desroziers_three_panels():
    by_site = pd.DataFrame(
        {"diagnosed": [2.0, 1.0], "specified": [1.0, 2.0], "ratio": [2.0, 0.5]},
        index=pd.Index(["wbb", "hdp"], name="obs_location"),
    )
    per_obs = pd.DataFrame(
        {"ratio": [1.0, 3.0, 0.5, 0.5]},
        index=obs_series({"wbb": [0, 0], "hdp": [0, 0]}, TIMES[:2]).index,
    )
    fig, (ax1, ax2, ax3) = viz.plot_desroziers(by_site, per_obs, per_obs)
    heights = [bar.get_height() for bar in ax1.patches]
    assert heights == [1.0, 2.0, 2.0, 1.0]  # specified (wbb, hdp), then diagnosed
    sites = [ln.get_label() for ln in ax3.get_lines() if ln.get_label()[0] != "_"]
    assert sites == ["hdp", "wbb"]


def test_plot_residuals_breaks_the_smoothed_line_at_gaps():
    times = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-02-01", "2020-02-02"]
    )
    obs = obs_series({"wbb": np.full(5, 2.0)}, times)
    post = obs + np.array([0.1, 0.2, 0.3, -0.1, -0.2])
    problem = SimpleNamespace(concentrations=obs, posterior_concentrations=post)
    fig, ax = viz.plot_residuals(problem, rolling_window=1, show_raw=False)
    (line,) = [ln for ln in ax.get_lines() if ln.get_label() == "wbb"]
    y = np.asarray(line.get_ydata(), dtype=float)
    np.testing.assert_allclose(y[:3], [0.1, 0.2, 0.3])
    assert np.isnan(y[3])  # first point after a 29-day gap is blanked
    np.testing.assert_allclose(y[4], -0.2)


def _problem(bias=False, same_background=True):
    obs_index = obs_series({"wbb": [0, 0], "hdp": [0, 0]}, TIMES[:2]).index
    bg = pd.Series([2.0, 2.1, 2.0, 2.1 if same_background else 2.3], index=obs_index)
    prior_index = [("flux", t) for t in TIMES]
    blocks = {"concentration": bg}
    problem = SimpleNamespace(constant=blocks)
    if bias:
        prior_index += [("bias", t) for t in TIMES[:2]]
        bias_prior = pd.Series([0.0, 0.0], index=TIMES[:2])
        bias_post = pd.Series([0.1, -0.1], index=TIMES[:2])
        prior_data = pd.Series(
            0.0, index=pd.MultiIndex.from_tuples(prior_index, names=["block", "t"])
        )
        problem.prior = _Blocks(prior_data, {"bias": bias_prior})
        problem.posterior = _Blocks(None, {"bias": bias_post})
    return problem


class _Blocks:
    """``obj.data`` plus ``obj["name"]`` block access, like a fips Vector."""

    def __init__(self, data, blocks):
        self.data = data
        self._blocks = blocks

    def __getitem__(self, key):
        return self._blocks[key]


def test_background_shared_by_all_sites_is_one_line():
    fig, axes = viz.plot_background_and_bias(_problem())
    assert len(axes) == 1
    assert [ln.get_label() for ln in axes[0].get_lines()] == [
        "Regional Background (all sites)"
    ]


def test_site_backgrounds_are_one_line_each():
    fig, axes = viz.plot_background_and_bias(_problem(same_background=False))
    assert sorted(ln.get_label() for ln in axes[0].get_lines()) == ["hdp", "wbb"]


def test_bias_panel_shows_prior_and_posterior():
    fig, axes = viz.plot_background_and_bias(_problem(bias=True))
    assert len(axes) == 2
    lines = {ln.get_label(): ln for ln in axes[1].get_lines()}
    np.testing.assert_allclose(lines["Posterior"].get_ydata(), [0.1, -0.1])
    np.testing.assert_allclose(lines["Prior"].get_ydata(), [0.0, 0.0])
