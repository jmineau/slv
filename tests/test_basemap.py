"""Tests for slv.basemap: loaders on synthetic files, layers on synthetic data (no network)."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import cartopy
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from cartopy.io.img_tiles import StadiaMapsTiles
from cartopy.mpl.feature_artist import FeatureArtist
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.lines import Line2D
from shapely.geometry import LineString, box

from slv import basemap
from slv.basemap import PC, SaltLake

BBOX = (-112.2, 40.5, -111.7, 40.9)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    spatial, user = tmp_path / "spatial", tmp_path / "user"
    monkeypatch.setenv("SLV_SPATIAL_DIR", str(spatial))
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(user))
    return spatial, user


def write(gdf, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path)


# ----- loaders -----


def test_load_population_density_over_land(data_dirs):
    spatial, user = data_dirs
    bg = gpd.GeoDataFrame(
        {"GEOID": ["490351001001", "490351001002"], "ALAND": [2e6, 0]},
        geometry=[box(-112.0, 40.6, -111.9, 40.7), box(-111.9, 40.6, -111.8, 40.7)],
        crs="EPSG:4326",
    )
    write(bg, spatial / "census/block_groups/utah/cb_2022_49_bg_500k.shp")
    (user / "census/raw").mkdir(parents=True)
    pd.DataFrame(
        {"GEOID": ["490351001001", "490351001002"], "population": [3000, 50]}
    ).to_csv(user / "census/raw/acs5_2022_utah_bg.csv", index=False)

    pop = basemap.load_population(BBOX).set_index("GEOID")
    assert pop.loc["490351001001", "density_km2"] == pytest.approx(1500)
    assert np.isnan(pop.loc["490351001002", "density_km2"])  # water-only block group


def test_load_interstates_matches_the_string_cartocode(data_dirs):
    # CARTOCODE is a string column; the old isin([1, ..., 6]) matched nothing
    spatial, _ = data_dirs
    roads = gpd.GeoDataFrame(
        {"CARTOCODE": ["1", "2", "15"]},
        geometry=[
            LineString([(-112.0, 40.6 + i / 10), (-111.8, 40.6)]) for i in range(3)
        ],
        crs="EPSG:4326",
    )
    write(roads, spatial / "transportation/roads/Roads.shp")
    assert basemap.load_interstates(BBOX)["CARTOCODE"].tolist() == ["1"]


def test_load_mesowest_filters(data_dirs):
    _, user = data_dirs
    (user / "mesowest").mkdir(parents=True)
    pd.DataFrame(
        {
            "Station ID": ["A", "B", "C"],
            "Latitude": [40.7, 40.8, 40.6],
            "Longitude": [-111.9, -111.8, -112.0],
            "Mesonet": ["UUNET", "UUNET", "AIRU"],
            "Status": ["ACTIVE", "INACTIVE", "ACTIVE"],
        }
    ).to_csv(user / "mesowest/MesoWest_Utah_stations_20221017.csv", index=False)
    assert basemap.load_mesowest()["Station ID"].tolist() == ["A"]
    assert len(basemap.load_mesowest(status=None, networks=None)) == 3


def test_load_borders_rejects_unknown_level():
    with pytest.raises(ValueError, match="level"):
        basemap.load_borders("city")


def test_stadia_tiles_need_a_key(monkeypatch):
    monkeypatch.delenv("STADIA_API_KEY", raising=False)
    with pytest.raises(OSError, match="STADIA_API_KEY"):
        SaltLake(tiles="terrain")


def test_stadia_tiles_survive_a_race_for_the_cache_folder(tmp_path, monkeypatch):
    """
    Another process makes cartopy's tile cache folder between cartopy's check for it
    and its own ``mkdir``: several jobs on one node, each drawing its first terrain
    map. Building the tiler fetches nothing, so the key need not be a real one.
    """
    cache_dir = tmp_path / "cartopy_cache"
    monkeypatch.setitem(cartopy.config, "cache_dir", cache_dir)
    monkeypatch.setenv("STADIA_API_KEY", "not-a-key")
    real_mkdir = Path.mkdir
    raced = []

    def mkdir_after_the_other_process(self, *args, **kwargs):
        if not raced:
            raced.append(self)
            real_mkdir(self, parents=True, exist_ok=True)  # the other process wins
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir_after_the_other_process)

    tiler = basemap.stadia_tiles()

    # The folder cartopy lost the race for was its tile cache, so the race was run.
    assert raced and raced[0].is_relative_to(cache_dir)
    assert isinstance(tiler, StadiaMapsTiles)
    assert tiler.cache_path == cache_dir
    assert (tiler.apikey, tiler.style) == ("not-a-key", "stamen_terrain")


# ----- layers (synthetic data) -----


def population():
    return gpd.GeoDataFrame(
        {"density_km2": [50.0, 1500.0, 3500.0]},
        geometry=[box(-112.1 + i / 10, 40.6, -112.0 + i / 10, 40.7) for i in range(3)],
        crs="EPSG:4326",
    )


def trax():
    return gpd.GeoDataFrame(
        {"line": ["R", "G", "B"]},
        geometry=[
            LineString([(-111.9, 40.6 + i / 20), (-111.85, 40.8)]) for i in range(3)
        ],
        crs="EPSG:4326",
    )


def stations():
    return gpd.GeoDataFrame(
        {"Station ID": ["A", "B"]},
        geometry=gpd.points_from_xy([-111.95, -111.8], [40.7, 40.75]),
        crs="EPSG:4326",
    )


def full_map(**kwargs):
    return (
        SaltLake(BBOX, **kwargs)
        .add_population(population())
        .add_trax("RG", trax=trax())
        .add_sites(["wbb", "hdp"], labels={"wbb": "UOU"})
        .add_mesowest(stations())
        .add_legend()
        .add_locator_map()
        .add_north_arrow()
    )


def test_layers_chain_and_legend_follows_their_order():
    m = full_map()
    assert isinstance(m, SaltLake)
    labels = [t.get_text() for t in m.ax.get_legend().get_texts()]
    assert labels == ["TRAX", "stationary", "MesoWest"]


def test_trax_legend_entry_stacks_the_line_colours():
    m = full_map()
    m.fig.canvas.draw()
    colors = {
        a.get_color()
        for a in m.ax.get_legend().findobj(Line2D)
        if a.get_linewidth() > 3
    }
    assert {"red", "green"} <= colors


def test_population_hides_sparse_block_groups_and_adds_a_colorbar():
    m = SaltLake(BBOX).add_population(population(), min_density=100)
    assert len(m.ax.collections[-1].get_paths()) == 2  # the 50 km^-2 block group is out
    (panel,) = m.ax.child_axes  # the white panel inside the map...
    (cax,) = panel.child_axes  # ...holding the colorbar
    assert cax.get_ylabel() == "Population km$^{-2}$"


def test_site_labels_use_the_given_names():
    m = SaltLake(BBOX).add_sites(["wbb", "hdp"], labels={"wbb": "UOU"})
    assert sorted(t.get_text() for t in m.ax.texts) == ["HDP", "UOU"]


def test_unknown_site_and_trax_line_raise():
    with pytest.raises(ValueError, match="Unknown sites"):
        SaltLake(BBOX).add_sites(["nope"])
    with pytest.raises(ValueError, match="No TRAX line"):
        SaltLake(BBOX).add_trax("S", trax=trax())


class _NoFeature:
    """Stands in for cartopy's ``OCEAN``, ``LAND`` and ``STATES``: draws nothing, notes the scale."""

    def __init__(self, scales):
        self.scales = scales

    def with_scale(self, scale):
        self.scales.append(scale)
        return cfeature.ShapelyFeature([], PC)


@pytest.fixture
def scales(monkeypatch):
    """
    The Natural Earth scales the locator map asks for, with nothing downloaded.

    ``full_map`` draws the real features (cartopy downloads and caches them); the
    tests that look at the box and the place do not need them.
    """
    asked: list[str] = []
    for name in ("OCEAN", "LAND", "STATES"):
        monkeypatch.setattr(cfeature, name, _NoFeature(asked))
    return asked


def box_style(m):
    """The (facecolor, edgecolor) the locator map draws the main map's box with."""
    artist = m.locator_map.collections[-1]  # the box is the last layer drawn on it
    assert isinstance(artist, FeatureArtist)
    return artist._feature._kwargs["facecolor"], artist._feature._kwargs["edgecolor"]


def test_locator_map_is_a_map_axis_inside_the_main_axes(scales):
    m = SaltLake(BBOX)
    assert m.add_locator_map() is m
    assert isinstance(m.locator_map, GeoAxes)
    assert m.locator_map in m.ax.child_axes
    assert scales == ["50m"] * 3  # lair's default scale, for ocean, land and states


def test_locator_map_outlines_the_bbox_and_fill_fills_it(scales):
    outlined = SaltLake(BBOX).add_locator_map()
    assert box_style(outlined) == ("none", "red")
    filled = SaltLake(BBOX).add_locator_map(fill=True, color="blue", scale="110m")
    assert box_style(filled) == ("blue", "blue")
    assert scales[-3:] == ["110m"] * 3


def test_locator_map_is_placed_by_bounds_in_the_main_axes_and_moves_with_them(scales):
    left, bottom, width, height = bounds = (0.1, 0.2, 0.3, 0.4)
    m = SaltLake(BBOX).add_locator_map(bounds=bounds)
    for position in [(0.1, 0.1, 0.8, 0.8), (0.2, 0.3, 0.5, 0.4)]:
        m.ax.set_position(position)
        m.fig.canvas.draw()
        main = m.ax.get_position()
        placed = m.locator_map.get_position()
        # The map keeps its aspect, so it fills the box ``bounds`` asks for only in one
        # direction and is centered in the other.
        want = (
            main.x0 + left * main.width,
            main.y0 + bottom * main.height,
            width * main.width,
            height * main.height,
        )
        assert placed.width <= want[2] + 1e-9 and placed.height <= want[3] + 1e-9
        assert placed.x0 + placed.width / 2 == pytest.approx(want[0] + want[2] / 2)
        assert placed.y0 + placed.height / 2 == pytest.approx(want[1] + want[3] / 2)


def test_a_plain_axis_is_refused():
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="GeoAxes"):
        SaltLake(BBOX, ax=ax)


def test_draws_on_a_given_axis():
    import cartopy.crs as ccrs

    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    m = SaltLake(BBOX, ax=axes[1]).add_trax("R", trax=trax()).add_legend()
    assert m.ax is axes[1]
    assert m.fig is fig
    assert axes[0].get_legend() is None
