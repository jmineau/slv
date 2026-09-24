"""Tests for the network transect builder on a synthetic two-line network."""

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from slv.measurements.mobile.network import UTM12
from slv.measurements.mobile.transects import (
    assign_lines,
    build_network_transects,
    load_network_transects,
)

gpd = pytest.importorskip("geopandas")
pytest.importorskip("scipy")
pytest.importorskip("xarray")

X0, Y0 = 420000.0, 4508000.0
TO_LONLAT = Transformer.from_crs(UTM12, "EPSG:4326", always_xy=True)


def _network():
    """Red: 100 points north-south along x = X0 (y from Y0 southward). Green shares the
    first 20 (northern) points, then branches east. ``s`` increases southward on Red and
    along the branch on Green, like the real UTA lines."""
    spacing = 50.0
    red_y = Y0 - spacing * np.arange(100)
    red = pd.DataFrame({"x": X0, "y": red_y, "lines": "R"})
    red.loc[:19, "lines"] = "GR"
    green_x = X0 + spacing * np.arange(1, 61)
    green = pd.DataFrame({"x": green_x, "y": red_y[19], "lines": "G"})
    pts = pd.concat([red, green], ignore_index=True)
    s_R = np.r_[spacing * np.arange(100), np.full(60, np.nan)]
    s_G = np.r_[
        spacing * np.arange(20), np.full(80, np.nan), spacing * (20 + np.arange(1, 61))
    ]
    g = gpd.GeoDataFrame(
        {
            "point": np.arange(len(pts)),
            "lines": pts["lines"].to_numpy(),
            "segment": np.arange(len(pts)) // 40,
            "s_R": s_R,
            "s_G": s_G,
        },
        geometry=gpd.points_from_xy(pts["x"], pts["y"]),
        crs=UTM12,
    )
    return g


def _obs(points, t0="2024-06-01 12:00:00", speed=10.0, gap_min=30):
    """A Red out-and-back (north end -> south end, dwell, back) then, after a gap, a Green
    run out along the branch. 1-s samples; CH4 2.0 ppm with a +0.5 ppm source at Red
    point 60 and +0.3 at Green point 130."""
    r = points[points.s_R.notna()].sort_values("s_R")
    g = points[points.s_G.notna()].sort_values("s_G")

    def along(df, n_per_pt):
        xs = np.repeat(df.geometry.x.to_numpy(), n_per_pt)
        ys = np.repeat(df.geometry.y.to_numpy(), n_per_pt)
        pid = np.repeat(df["point"].to_numpy(), n_per_pt)
        return xs, ys, pid

    n = int(50 / speed)
    x1, y1, p1 = along(r, n)
    dwell = 90
    xd, yd, pd_ = np.full(dwell, x1[-1]), np.full(dwell, y1[-1]), np.full(dwell, p1[-1])
    x2, y2, p2 = x1[::-1], y1[::-1], p1[::-1]
    x3, y3, p3 = along(g, n)
    x = np.r_[x1, xd, x2, x3]
    y = np.r_[y1, yd, y2, y3]
    pid = np.r_[p1, pd_, p2, p3]
    t = np.arange(len(x), dtype=float)
    t[len(x1) + dwell + len(x2) :] += gap_min * 60
    ch4 = np.full(len(x), 2.0)
    ch4[pid == 60] += 0.5
    ch4[pid == 130] += 0.3
    lon, lat = TO_LONLAT.transform(x, y)
    return pd.DataFrame(
        {
            "Time_UTC": pd.Timestamp(t0) + pd.to_timedelta(t, unit="s"),
            "Longitude_deg": lon,
            "Latitude_deg": lat,
            "CH4_ppm": ch4,
            "cal_source": "pipeline",
            "low_pressure": False,
        }
    )


def test_assign_lines_votes_among_the_points_own_letters():
    # Green-only, then the shared trunk, then Red/Blue shared, then Red-only: the trunk
    # samples go to the nearer exclusive line and the BR samples can only be B or R
    pl = np.array(["G", "G", "BGR", "BGR", "BR", "BR", "R", "R", "", "BR"], dtype=str)
    t = np.arange(10) * 60.0
    hit = np.ones(10, bool)
    hit[8] = False
    out = assign_lines(pl, t, hit, window_s=150)
    assert list(out) == ["G", "G", "G", "G", "R", "R", "R", "R", "", "R"]
    # a BR sample with no exclusive point within the window is unresolved
    pl2 = np.array(["BR", "BR", "BR"], dtype=str)
    assert list(
        assign_lines(pl2, np.arange(3) * 60.0, np.ones(3, bool), window_s=150)
    ) == ["", "", ""]


def test_build_network_transects_two_lines(tmp_path):
    pts = _network()
    obs = _obs(pts)
    ds = build_network_transects(obs, pts, lag=None, max_dwell_s=30)
    assert ds.sizes["point"] == 160
    assert list(ds.line.values) == ["G", "R", "R"] or list(ds.line.values) == [
        "R",
        "R",
        "G",
    ]
    red = ds.sel(transect=ds.line == "R")
    green = ds.sel(transect=ds.line == "G")
    assert list(red.direction.values) == [1, -1]
    assert list(red.heading.values) == ["S", "N"]
    assert green.sizes["transect"] == 1 and green.heading.values[0] == "E"
    # the sources land on their points in every transit that passes them
    assert np.allclose(red.obs.values[:, 60], 2.5)
    assert np.allclose(red.obs.values[:, 10], 2.0)
    assert green.obs.values[0, 130] == pytest.approx(2.3)
    # the shared trunk is filled for both lines; the Red branch is empty on the Green run
    assert np.isfinite(green.obs.values[0, :20]).all()
    assert np.isnan(green.obs.values[0, 20:100]).all()
    # the dwell at the south end is trimmed
    assert red.n.values[:, 99].max() <= 32
    assert (ds.frac_uncalibrated.values == 0).all()
    assert ds.attrs["n_off_network"] == 0


def test_lag_moves_the_source_back_along_the_track():
    pts = _network()
    obs = _obs(pts)
    # a 10-s lag at 10 m/s is 100 m = 2 points: uncorrected, the source reads 2 points
    # further along in each direction; with the lag applied both directions agree
    ds0 = build_network_transects(obs, pts, lag=None)
    red0 = ds0.sel(transect=ds0.line == "R")
    ds1 = build_network_transects(obs, pts, lag=10.0)
    red1 = ds1.sel(transect=ds1.line == "R")
    peaks0 = np.nanargmax(red0.obs.values[:, :100], axis=1)
    peaks1 = np.nanargmax(red1.obs.values[:, :100], axis=1)
    assert list(peaks0) == [60, 60]  # the synthetic source is placed by point, no lag
    # shifting the samples back moves the apparent source against the direction of travel
    assert list(peaks1) == [58, 62]
    assert (ds1.lag_s.values == 10).all() and (
        ds1.lag_source.values == "constant"
    ).all()


def test_roundtrip_netcdf(tmp_path):
    pts = _network()
    ds = build_network_transects(_obs(pts), pts)
    d = tmp_path / "network"
    d.mkdir()
    ds.attrs["year"] = 2024
    ds.to_netcdf(d / "trx01_CH4_2024.nc")
    ds2 = ds.copy()
    ds2.attrs["year"] = 2025
    ds2.to_netcdf(d / "trx01_CH4_2025.nc")
    back = load_network_transects(transects_dir=tmp_path)
    assert back.sizes["transect"] == 2 * ds.sizes["transect"]
    assert list(back.transect.values) == list(range(back.sizes["transect"]))
    only = load_network_transects(years=[2025], transects_dir=tmp_path)
    assert only.sizes["transect"] == ds.sizes["transect"]
    assert str(only.t_start.dtype).startswith("datetime64")
