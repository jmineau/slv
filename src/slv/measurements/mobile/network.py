"""The TRAX network and its yards: line geometry, staged points, storage polygons,
shed footprints, and the route-buffer filter used by the GPS merge.

Group-data source: ``UTA_TRAX.geojson`` (the lines) under
``$LINGROUP_DATA_DIR/spatial/transportation/light_rail``. Packaged here: the yard
polygons ``jrrsc.geojson`` and ``mrsc.geojson`` (:data:`storage_locations`) and the
shed footprints ``trax_depots.geojson`` (both data-derived except JRRSC's yard, which
is hand-drawn). Staged points along the lines are cached
under ``$SLV_USER_DATA_DIR/trax``.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
from lair.geo import points_along_line
from shapely import Point

from slv import get_data_dir


def group_dir() -> Path:
    """``$LINGROUP_DATA_DIR``, the lin-group data library."""
    return Path(get_data_dir("LINGROUP_DATA_DIR"))


def user_dir() -> Path:
    """``$SLV_USER_DATA_DIR``, where slv keeps derived data and caches."""
    return Path(get_data_dir("SLV_USER_DATA_DIR"))


def __getattr__(name: str) -> Path:
    # GROUP_DIR / USER_DIR resolve when first used, not at import, so the package
    # imports without the CHPC data roots set (CI, docs builds); scripts that
    # `from slv.measurements.mobile.network import USER_DIR` keep working.
    if name == "GROUP_DIR":
        return group_dir()
    if name == "USER_DIR":
        return user_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: UTM zone 12 N, the metric CRS used for distances and buffers.
UTM12 = "EPSG:32612"

#: Storage yards where the trains sleep, name -> packaged geojson (one polygon each).
#: JRRSC: Jordan River Rail Service Center (trx01/trx02; hand-drawn, copied from the
#: group spatial dir). MRSC: Midvale Rail Service Center (trx03, the old-train yard;
#: hull of trx03 parked positions, 2025 — see :mod:`.location`).
storage_locations = {
    "JRRSC": files(__package__).joinpath("jrrsc.geojson"),
    "MRSC": files(__package__).joinpath("mrsc.geojson"),
}


def get_geodf(
    obj: str | Path | gpd.GeoDataFrame | bool | None,
) -> gpd.GeoDataFrame | None:
    if isinstance(obj, gpd.GeoDataFrame):
        return obj
    elif isinstance(obj, (str, Path)):
        return gpd.read_file(obj)
    elif obj is False or obj is None:
        return None
    elif hasattr(obj, "open"):  # importlib.resources Traversable (packaged file)
        with obj.open("r") as f:
            return gpd.read_file(f)
    elif obj is True:
        raise ValueError(
            "Boolean value True is not a valid geodataframe input. Please provide a file path, GeoDataFrame, or set to False/None."
        )
    else:
        raise ValueError(f"Unsupported type for geodataframe: {type(obj)}")


def load_trax_lines(meters=False) -> gpd.GeoDataFrame:
    lines = gpd.read_file(
        group_dir() / "spatial/transportation/light_rail/UTA_TRAX.geojson"
    )
    if meters:
        # Convert to UTM zone 12 for meter units
        lines = lines.to_crs(ccrs.UTM(12).proj4_init)
    return lines


def load_trax_points(
    spacing=2000, meters=False, resolution_factor=None
) -> gpd.GeoDataFrame:
    points_geojson = user_dir() / f"trax/points_{spacing}m.geojson"

    if points_geojson.exists():
        print(f"Loading cached TRAX points from {points_geojson}")
        points_df = gpd.read_file(points_geojson)
    else:
        print(f"Generating TRAX points with {spacing}m spacing...")
        # Generate points along TRAX lines at specified spacing
        lines = load_trax_lines(meters=True)
        points = points_along_line(
            lines.geometry, spacing=spacing, resolution_factor=resolution_factor
        )
        # Round UTM coordinates to whole meters
        points = [Point(round(p.x), round(p.y)) for p in points]
        points_df = gpd.GeoDataFrame(geometry=points, crs=ccrs.UTM(12).proj4_init)

        # Determine which TRAX lines pass through each buffered point
        buffered = points_df.copy()
        buffered.geometry = points_df.buffer(spacing / 2)
        joined = gpd.sjoin(
            buffered, lines[["line", "geometry"]], how="left", predicate="intersects"
        )
        points_df["lines"] = joined.groupby(joined.index)["line"].apply(
            lambda x: "".join(sorted(x))
        )
        points_df.to_file(points_geojson, index=False)

    if meters:
        return points_df
    else:
        points_df = points_df.to_crs("EPSG:4326")
        points_df.geometry = gpd.points_from_xy(
            # Round lat/lon to 5 decimal places (~1 meter) to avoid excessive precision in GeoJSON
            points_df.geometry.x.round(5),
            points_df.geometry.y.round(5),
        )
        return points_df


def filter_near_routes(
    gps: gpd.GeoDataFrame, routes: gpd.GeoDataFrame, buffer: float
) -> gpd.GeoDataFrame:
    """Keep GPS points within ``buffer`` (in ``routes``' CRS units) of any route.

    The per-line buffers are dissolved into one geometry first: where lines share
    track (the red/blue/green downtown trunk) a point falls inside several buffers and
    a plain ``sjoin`` duplicated it — the old builder more than doubled rows there.
    """
    routes_buff = gpd.GeoDataFrame(
        geometry=[routes.buffer(buffer).union_all()], crs=routes.crs
    ).to_crs(gps.crs)
    return gpd.sjoin(gps, routes_buff, how="inner", predicate="within").drop(
        columns=["index_right"]
    )


def load_depot_footprint(meters: bool = False) -> gpd.GeoDataFrame:
    """Packaged shed footprints at the service centers (``trax_depots.geojson``).

    One feature per yard (``name``: JRRSC, MRSC). Data-derived: the 5–95 % box of the
    per-minute median positions of degraded fixes (trx01 at JRRSC, trx03 at MRSC,
    Jan–Aug 2025), padded 15 m. Roughly 120 × 180 m and 220 × 230 m.
    """
    with files(__package__).joinpath("trax_depots.geojson").open("r") as f:
        gdf = gpd.read_file(f)
    return gdf.to_crs(UTM12) if meters else gdf


def load_storage_polygons(meters: bool = False) -> gpd.GeoDataFrame:
    """All storage yards in :data:`storage_locations`, one row each with a ``name``."""
    rows = []
    for name, src in storage_locations.items():
        g = get_geodf(src).to_crs(UTM12)  # pyright: ignore[reportOptionalMemberAccess]
        rows.append(
            gpd.GeoDataFrame(
                {"name": [name]}, geometry=[g.geometry.union_all()], crs=UTM12
            )
        )
    gdf = pd.concat(rows, ignore_index=True)
    return gdf if meters else gdf.to_crs("EPSG:4326")
