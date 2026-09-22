"""Tests for the TRAX network loaders on a synthetic two-line network (no group data).

Red runs 4 km east; Green shares its first 2 km, then turns 2 km north. Built in UTM
12 N, so along-route distances are plain x / y offsets.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from slv.measurements.mobile import network, receptors
from slv.measurements.mobile.network import UTM12, get_geodf, storage_locations

X0, Y0 = 420000.0, 4508000.0


def write_lines(group):
    lines = gpd.GeoDataFrame(
        {"line": ["R", "G"]},
        geometry=[
            LineString([(X0, Y0), (X0 + 4000, Y0)]),
            LineString([(X0, Y0), (X0 + 2000, Y0), (X0 + 2000, Y0 + 2000)]),
        ],
        crs=UTM12,
    ).to_crs("EPSG:4326")
    d = group / "spatial/transportation/light_rail"
    d.mkdir(parents=True)
    lines.to_file(d / "UTA_TRAX.geojson")


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """Group data with the lines, and a user dir holding the generated point caches."""
    root = tmp_path_factory.mktemp("trax")
    write_lines(root / "group")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LINGROUP_DATA_DIR", str(root / "group"))
        mp.setenv("SLV_USER_DATA_DIR", str(root / "user"))  # no trax/ dir yet
        points = receptors.load_trax_network_points(meters=True)
    return root, points


@pytest.fixture
def env(roots, monkeypatch):
    root, _ = roots
    monkeypatch.setenv("LINGROUP_DATA_DIR", str(root / "group"))
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(root / "user"))
    return root


# --------------------------------------------------------------------------- roots


def test_data_roots_resolve_on_use(monkeypatch, tmp_path):
    monkeypatch.delenv("SLV_USER_DATA_DIR", raising=False)
    with pytest.raises(OSError, match="SLV_USER_DATA_DIR"):
        network.user_dir()
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LINGROUP_DATA_DIR", str(tmp_path / "group"))
    assert tmp_path == network.USER_DIR
    assert tmp_path / "group" == network.GROUP_DIR
    with pytest.raises(AttributeError):
        network.NOT_A_DIR  # noqa: B018


# --------------------------------------------------------------------------- get_geodf


def test_get_geodf_inputs(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
    assert get_geodf(gdf) is gdf
    gdf.to_file(tmp_path / "x.geojson")
    assert len(get_geodf(tmp_path / "x.geojson")) == 1
    assert len(get_geodf(str(tmp_path / "x.geojson"))) == 1
    assert get_geodf(None) is None and get_geodf(False) is None
    assert get_geodf(storage_locations["JRRSC"]).geom_type.iloc[0] == "Polygon"
    with pytest.raises(ValueError, match="True"):
        get_geodf(True)
    with pytest.raises(ValueError, match="Unsupported"):
        get_geodf(3)


# --------------------------------------------------------------------------- lines, points


def test_load_trax_lines_in_metres(env):
    lines = network.load_trax_lines(meters=True).set_index("line")
    np.testing.assert_allclose(lines.length.loc[["R", "G"]], [4000, 4000], atol=0.5)
    assert network.load_trax_lines().crs.to_epsg() == 4326


def test_trax_points_one_set_on_shared_track(env):
    pts = network.load_trax_points(2000, meters=True)
    xy = {
        (round(p.x - X0), round(p.y - Y0)): lines
        for p, lines in zip(pts.geometry, pts["lines"], strict=True)
    }
    # every 2 km along the network, the shared stretch once, tagged with both lines
    assert xy == {(0, 0): "GR", (2000, 0): "GR", (4000, 0): "R", (2000, 2000): "G"}


def test_trax_points_read_their_cache(env, monkeypatch):
    network.load_trax_points(2000)  # written by the fixture's build
    monkeypatch.setattr(network, "points_along_line", _no_rebuild)
    assert len(network.load_trax_points(2000)) == 4


def test_trax_points_lonlat_rounded(env):
    pts = network.load_trax_points(2000)
    assert pts.crs.to_epsg() == 4326
    np.testing.assert_array_equal(pts.geometry.x, pts.geometry.x.round(5))


def _no_rebuild(*args, **kwargs):
    raise AssertionError("the cache should have been used")


# --------------------------------------------------------------------------- 50-m network


def test_network_points_distance_along_each_line(roots):
    _, pts = roots
    x = pts.geometry.x.to_numpy() - X0
    y = pts.geometry.y.to_numpy() - Y0
    on_r = pts["lines"].str.contains("R").to_numpy()
    on_g = pts["lines"].str.contains("G").to_numpy()
    # Red is straight east: s_R is x. Green turns north at 2 km: s_G is x, then 2000 + y
    np.testing.assert_allclose(pts["s_R"][on_r], x[on_r], atol=0.5)
    np.testing.assert_allclose(
        pts["s_G"][on_g], np.where(y[on_g] > 1, 2000 + y[on_g], x[on_g]), atol=0.5
    )
    assert pts["s_R"][~on_r].isna().all() and pts["s_G"][~on_g].isna().all()
    assert set(pts["lines"]) == {"GR", "R", "G"}


def test_network_points_belong_to_the_nearest_2km_point(roots, env):
    _, pts = roots
    segs = gpd.GeoSeries(network.load_trax_points(2000, meters=True).geometry.values)
    nearest = [int(segs.distance(p).argmin()) for p in pts.geometry]
    assert pts["segment"].tolist() == nearest
    assert (pts["segment_dist_m"] <= 1000).all()


def test_network_points_cache_and_crs(env, monkeypatch):
    monkeypatch.setattr(receptors, "load_trax_points", _no_rebuild)
    pts = receptors.load_trax_network_points()
    assert pts.crs.to_epsg() == 4326
    assert {"point", "lines", "segment", "s_R", "s_G"} <= set(pts.columns)


# --------------------------------------------------------------------------- fixes


def test_load_trax_fixes_filters_dedups_and_sorts(tmp_path):
    t = pd.to_datetime(
        [
            "2024-06-01 12:00:02",
            "2024-06-01 12:00:00",
            "2024-06-01 12:00:01",
            "2024-06-01 12:00:01",
            "2024-06-01 12:00:03",
        ]
    )
    df = pd.DataFrame(
        {
            "Time_UTC": t,
            "Latitude_deg": [40.1, 40.0, 40.05, 40.05, 40.2],
            "Longitude_deg": -111.9,
            "CH4_ppm": 2.0,  # not read
            "state": ["route", "stopped", "line", "line", "depot"],
        }
    )
    path = tmp_path / "obs.parquet"
    df.to_parquet(path)

    fixes = receptors.load_trax_fixes(path)  # on_track: route, line, stopped
    assert list(fixes.columns) == ["Time_UTC", "Latitude_deg", "Longitude_deg"]
    assert fixes.Time_UTC.tolist() == sorted(t.unique()[:3])
    assert fixes.Latitude_deg.tolist() == [40.0, 40.05, 40.1]
    assert len(receptors.load_trax_fixes(path, location="all")) == 4
    assert len(receptors.load_trax_fixes(path, location=("depot",))) == 1
