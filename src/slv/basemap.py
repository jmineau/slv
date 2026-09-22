"""
Salt Lake Valley basemaps.

:class:`SaltLake` makes a cartopy map of the valley and layers features onto it. Every
``add_*`` method returns the map, so calls chain::

    from slv.basemap import SaltLake

    m = (
        SaltLake(tiles="terrain")
        .add_population()
        .add_trax(lines="RG")
        .add_sites(["wbb", "ldf", "hdp"], labels={"wbb": "UOU"})
        .add_mesowest()
        .add_legend()
        .add_inset()
        .add_north_arrow()
    )
    m.fig.savefig("slv.png", dpi=300)

The ``load_*`` functions read the layers' data (group data via ``$SLV_SPATIAL_DIR``, slv
user data via ``$SLV_USER_DATA_DIR``) and return lon/lat GeoDataFrames; the ``add_*``
methods only plot, so any of them also takes data you pass in.
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from lair.geo import add_latlon_ticks, bbox2extent
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from shapely.geometry import box

from slv import get_data_dir
from slv.domain import MAP_BBOX

PC = ccrs.PlateCarree()

#: Colour of each TRAX line, keyed by its ``line`` letter in UTA_TRAX.geojson.
TRAX_COLORS = {"R": "red", "G": "green", "B": "blue", "S": "gray"}

#: Stadia Maps credit, required wherever its tiles are shown.
STADIA_ATTRIBUTION = (
    "© Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors"
)


# ----- data -----


def load_population(bbox=MAP_BBOX) -> gpd.GeoDataFrame:
    """Block-group population and density (people km⁻², over land area).

    ACS 2022 5-year population (``$SLV_USER_DATA_DIR/census/raw/acs5_2022_utah_bg.csv``)
    joined on GEOID to the 2022 cartographic block groups
    (``$SLV_SPATIAL_DIR/census/block_groups/utah``). Block groups with no land area get a
    NaN density.
    """
    shp = (
        get_data_dir("SLV_SPATIAL_DIR")
        / "census/block_groups/utah/cb_2022_49_bg_500k.shp"
    )
    csv = get_data_dir("SLV_USER_DATA_DIR") / "census/raw/acs5_2022_utah_bg.csv"
    bg = gpd.read_file(shp, bbox=bbox)
    pop = pd.read_csv(csv, dtype={"GEOID": str})
    bg = bg.merge(pop[["GEOID", "population"]], on="GEOID", how="left")
    land_km2 = bg["ALAND"].where(bg["ALAND"] > 0) / 1e6
    bg["density_km2"] = bg["population"] / land_km2
    return bg.to_crs("EPSG:4326")


def load_mesowest(status="ACTIVE", networks=("UUNET",)) -> gpd.GeoDataFrame:
    """MesoWest stations (``$SLV_USER_DATA_DIR/mesowest``), optionally filtered by
    ``Status`` and ``Mesonet``; ``None`` keeps all."""
    csv = (
        get_data_dir("SLV_USER_DATA_DIR")
        / "mesowest/MesoWest_Utah_stations_20221017.csv"
    )
    df = pd.read_csv(csv)
    if status is not None:
        df = df[df["Status"] == status.upper()]
    if networks is not None:
        df = df[df["Mesonet"].isin(networks)]
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )


def load_interstates(bbox=MAP_BBOX) -> gpd.GeoDataFrame:
    """Interstates (UGRC road ``CARTOCODE`` "1") from
    ``$SLV_SPATIAL_DIR/transportation/roads``."""
    shp = get_data_dir("SLV_SPATIAL_DIR") / "transportation/roads/Roads.shp"
    roads = gpd.read_file(shp, bbox=bbox, columns=["CARTOCODE"])
    return roads[roads["CARTOCODE"] == "1"].to_crs("EPSG:4326")


def load_borders(level="county", bbox=MAP_BBOX) -> gpd.GeoDataFrame:
    """2022 Census state or county boundaries from
    ``$SLV_SPATIAL_DIR/administrative``."""
    files = {
        "state": "states/cb_2022_us_state_500k.shp",
        "county": "counties/cb_2022_us_county_500k.shp",
    }
    if level not in files:
        raise ValueError(f"level={level!r}; expected one of {list(files)}")
    shp = get_data_dir("SLV_SPATIAL_DIR") / "administrative" / files[level]
    return gpd.read_file(shp, bbox=bbox).to_crs("EPSG:4326")


def stadia_tiles(style="stamen_terrain") -> cimgt.StadiaMapsTiles:
    """A Stadia Maps tiler (``$STADIA_API_KEY``), cached on disk by cartopy.

    Fetched from tiles.stadiamaps.com when the map is drawn. Their terms ask for
    :data:`STADIA_ATTRIBUTION` on the map, which :class:`SaltLake` adds by default.
    """
    key = os.environ.get("STADIA_API_KEY")
    if not key:
        raise OSError("Set $STADIA_API_KEY to use Stadia Maps tiles.")
    return cimgt.StadiaMapsTiles(apikey=key, style=style, cache=True)


# ----- legend -----


class _StackedLines:
    """Legend handle for several lines drawn as one entry (e.g. the TRAX lines)."""

    def __init__(self, colors, linewidth):
        self.colors = list(colors)
        self.linewidth = linewidth


class _StackedLinesHandler(HandlerBase):
    """Draws a :class:`_StackedLines` handle as horizontal lines stacked top to bottom."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        n = len(orig_handle.colors)
        ys = np.linspace(height, 0, n + 2)[1:-1] - ydescent
        return [
            Line2D(
                [-xdescent, width - xdescent],
                [y, y],
                color=color,
                linewidth=orig_handle.linewidth,
                transform=trans,
            )
            for color, y in zip(orig_handle.colors, ys, strict=True)
        ]


def _thousands(x, pos=None):
    return "0" if x == 0 else f"{x / 1000:g}k"


# ----- map -----


class SaltLake:
    """A cartopy map of the Salt Lake Valley.

    Parameters
    ----------
    bbox : tuple
        (W, S, E, N) of the map in degrees; defaults to :data:`slv.domain.MAP_BBOX`.
    ax : cartopy GeoAxes, optional
        Draw on this map axis (e.g. a subplot) instead of a new figure.
    tiles : "terrain", cartopy tiler or None
        Background tiles; "terrain" is Stadia's ``stamen_terrain``
        (:func:`stadia_tiles`). A new figure then uses the tiler's projection.
    zoom : int
        Tile zoom level.
    figsize : tuple
        Size of a new figure.
    latlon_ticks : bool
        Label the axes in degrees.
    attribution : bool
        Credit the tile source in the corner of the map.
    """

    def __init__(
        self,
        bbox=MAP_BBOX,
        ax=None,
        tiles=None,
        zoom=11,
        figsize=(6, 7),
        latlon_ticks=False,
        attribution=True,
    ):
        tiler = stadia_tiles() if tiles == "terrain" else tiles
        if ax is None:
            projection = tiler.crs if tiler is not None else PC
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": projection}
            )
        self.ax = ax
        self.bbox = tuple(bbox)
        self.extent = bbox2extent(list(bbox))
        self._legend: list[tuple[object, str]] = []

        ax.set_extent(self.extent, crs=PC)
        if tiler is not None:
            ax.add_image(tiler, zoom, zorder=0)
            if attribution and isinstance(tiler, cimgt.StadiaMapsTiles):
                ax.text(
                    0.995,
                    0.003,
                    STADIA_ATTRIBUTION,
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=4,
                    zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
                )
        if latlon_ticks:
            add_latlon_ticks(ax, self.extent, x_rotation=30)

    @property
    def fig(self):
        """The figure holding the map."""
        return self.ax.figure

    def __repr__(self):
        labels = [label for _, label in self._legend]
        return f"SaltLake(bbox={self.bbox}, legend={labels})"

    def _add_to_legend(self, handle, label):
        if label:
            self._legend.append((handle, label))

    # ----- layers -----

    def add_population(
        self,
        population=None,
        cmap="pink_r",
        vmax=4000,
        min_density=100,
        alpha=0.6,
        colorbar=True,
    ):
        """Block-group population density (people km⁻²) with a colorbar panel.

        ``population`` defaults to :func:`load_population`; block groups below
        ``min_density`` are left unshaded.
        """
        gdf = load_population(self.bbox) if population is None else population
        gdf = gdf[gdf["density_km2"] >= min_density]
        gdf.plot(
            ax=self.ax,
            column="density_km2",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            alpha=alpha,
            edgecolor="none",
            transform=PC,
            zorder=1,
        )
        if colorbar:
            mappable = ScalarMappable(norm=Normalize(0, vmax), cmap=plt.get_cmap(cmap))
            self._colorbar_panel(mappable, "Population km$^{-2}$")
        return self

    def _colorbar_panel(self, mappable, label, bounds=(0.01, 0.01, 0.22, 0.4)):
        """A white panel in the lower-left corner holding a vertical colorbar."""
        panel = self.ax.inset_axes(bounds, zorder=6)
        panel.set_xticks([])
        panel.set_yticks([])
        cax = panel.inset_axes((0.12, 0.05, 0.16, 0.9))
        cbar = self.fig.colorbar(mappable, cax=cax, format=FuncFormatter(_thousands))
        cbar.set_label(label)
        return cbar

    def add_inventory(self, inventory, **kwargs):
        """A lair inventory (``Inventory.plot``; e.g. ``time=``, ``sector=``, ``alpha=``)."""
        kwargs.setdefault("zorder", 1)
        inventory.plot(ax=self.ax, **kwargs)
        return self

    def add_borders(self, level="county", borders=None, **kwargs):
        """State or county boundaries (:func:`load_borders`)."""
        gdf = load_borders(level, self.bbox) if borders is None else borders
        style = {"facecolor": "none", "edgecolor": "black", "linewidth": 0.8}
        gdf.plot(ax=self.ax, transform=PC, zorder=2, **(style | kwargs))
        return self

    def add_interstates(self, interstates=None, **kwargs):
        """Interstate highways (:func:`load_interstates`)."""
        gdf = load_interstates(self.bbox) if interstates is None else interstates
        style = {"color": "dimgray", "linewidth": 1.5}
        gdf.plot(ax=self.ax, transform=PC, zorder=2, **(style | kwargs))
        return self

    def add_trax(self, lines="RG", trax=None, colors=None, linewidth=5, label="TRAX"):
        """TRAX lines by their letters (R, G, B, S), in UTA's line colours.

        ``trax`` defaults to :func:`slv.measurements.mobile.load_trax_lines`.
        """
        if trax is None:
            from slv.measurements.mobile import load_trax_lines

            trax = load_trax_lines()
        colors = TRAX_COLORS | (colors or {})
        used = []
        for letter in lines:
            line = trax[trax["line"] == letter]
            if line.empty:
                raise ValueError(f"No TRAX line {letter!r}; have {list(trax['line'])}")
            line.plot(
                ax=self.ax,
                color=colors[letter],
                linewidth=linewidth,
                transform=PC,
                zorder=3,
                capstyle="round",
            )
            used.append(colors[letter])
        self._add_to_legend(_StackedLines(used, linewidth * 0.8), label)
        return self

    def add_points(
        self, points, marker="o", size=60, color="black", label=None, **kwargs
    ):
        """Point markers from a GeoDataFrame or a frame with longitude / latitude columns."""
        if isinstance(points, gpd.GeoDataFrame):
            lon, lat = points.geometry.x, points.geometry.y
        else:
            lon, lat = points["longitude"], points["latitude"]
        self.ax.scatter(
            lon,
            lat,
            s=size,
            marker=marker,
            color=color,
            transform=PC,
            zorder=5,
            **kwargs,
        )
        handle = Line2D(
            [], [], linestyle="none", marker=marker, color=color, markersize=size**0.5
        )
        self._add_to_legend(handle, label)
        return self

    def add_sites(
        self,
        sites,
        site_config=None,
        labels=False,
        label_offset=(10, 10),
        size=250,
        label="stationary",
    ):
        """Measurement sites (``site_config`` rows) as open circles.

        ``labels=True`` writes each site's ID in capitals beside it; a dict maps IDs to
        the text to write instead (e.g. ``{"wbb": "UOU"}``). ``label_offset`` is in
        points from the site, one (x, y) for all or a dict of them by site.
        """
        if site_config is None:
            from slv.measurements.sites import load_site_config

            site_config = load_site_config()
        missing = [s for s in sites if s not in site_config.index]
        if missing:
            raise ValueError(f"Unknown sites {missing}")
        rows = site_config.loc[list(sites)]
        rows = rows[["longitude", "latitude"]].astype(float)
        self.ax.scatter(
            rows["longitude"],
            rows["latitude"],
            s=size,
            facecolors="none",
            edgecolors="black",
            linewidths=3,
            transform=PC,
            zorder=5,
        )
        handle = Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=3,
            markersize=size**0.5,
        )
        self._add_to_legend(handle, label)
        if labels:
            names = labels if isinstance(labels, dict) else {}
            offsets = label_offset if isinstance(label_offset, dict) else {}
            for site, (lon, lat) in rows.iterrows():
                self.ax.annotate(
                    names.get(site, site.upper()),
                    (lon, lat),
                    xycoords=PC._as_mpl_transform(self.ax),
                    xytext=offsets.get(site, (10, 10) if offsets else label_offset),
                    textcoords="offset points",
                    fontsize=14,
                    fontweight="bold",
                    zorder=7,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                )
        return self

    def add_mesowest(self, stations=None, label="MesoWest", **kwargs):
        """MesoWest stations as crosses; ``stations`` defaults to active UUNET stations
        (:func:`load_mesowest`)."""
        gdf = load_mesowest() if stations is None else stations
        style = {"marker": "x", "size": 60, "linewidths": 2}
        return self.add_points(gdf, label=label, **(style | kwargs))

    # ----- furniture -----

    def add_legend(self, loc="upper left", **kwargs):
        """A legend of the layers added so far, in the order they were added."""
        handles = [h for h, _ in self._legend]
        labels = [label for _, label in self._legend]
        style = {
            "loc": loc,
            "handlelength": 2.5,
            "handleheight": 2,
            "labelspacing": 0.6,
            "handler_map": {_StackedLines: _StackedLinesHandler()},
        }
        legend = self.ax.legend(handles, labels, **(style | kwargs))
        legend.set_zorder(8)
        return self

    def add_inset(
        self,
        extent=(-125, -105, 30, 50),
        bounds=(0.7, 0.7, 0.29, 0.29),
        color="red",
        projection=None,
    ):
        """A locator map of the western US with the map's area marked.

        Uses cartopy's Natural Earth 50 m land, ocean and states (cached after the
        first download).
        """
        projection = projection or ccrs.AlbersEqualArea(central_longitude=-111)
        inset = self.ax.inset_axes(bounds, projection=projection, zorder=8)
        inset.set_extent(extent, crs=PC)
        inset.add_feature(cfeature.OCEAN.with_scale("50m"))
        inset.add_feature(cfeature.LAND.with_scale("50m"))
        inset.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.8)
        inset.add_geometries(
            [box(*self.bbox)], crs=PC, facecolor=color, edgecolor=color, linewidth=2
        )
        self.inset = inset
        return self

    def add_north_arrow(self, xy=(0.95, 0.07), fontsize=16):
        """A north arrow (▲ N) in axes coordinates."""
        self.ax.text(
            *xy,
            "▲\nN",
            transform=self.ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        return self
