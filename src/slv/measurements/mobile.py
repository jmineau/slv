from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import uataq
from lair.geo import points_along_line
from shapely import Point

from slv import get_data_dir

GROUP_DIR = Path(get_data_dir("LINGROUP_DATA_DIR"))
USER_DIR = Path(get_data_dir("SLV_USER_DATA_DIR"))

storage_locations = {
    "JRRSC": GROUP_DIR
    / "spatial/transportation/light_rail/JRRSC.geojson",  # Jordan River Rail Service Center
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
    elif obj is True:
        raise ValueError(
            "Boolean value True is not a valid geodataframe input. Please provide a file path, GeoDataFrame, or set to False/None."
        )
    else:
        raise ValueError(f"Unsupported type for geodataframe: {type(obj)}")


def load_trax_lines(meters=False) -> gpd.GeoDataFrame:
    lines = gpd.read_file(
        GROUP_DIR / "spatial/transportation/light_rail/UTA_TRAX.geojson"
    )
    if meters:
        # Convert to UTM zone 12 for meter units
        lines = lines.to_crs(ccrs.UTM(12).proj4_init)
    return lines


def load_trax_points(
    spacing=2000, meters=False, resolution_factor=None
) -> gpd.GeoDataFrame:
    points_geojson = USER_DIR / f"trax/points_{spacing}m.geojson"

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


def merge_with_gps(
    site,
    org,
    obs,
    time_range=None,
    num_processes=1,
    routes=None,
    route_buffer=None,
    storage_polygon=None,
):
    # Get routes and storage polygon defaults if not provided
    # Can be set to False to skip these filters, or provide custom geodataframes/paths
    if routes is None:  # noqa: SIM102
        if site.startswith("trx"):
            routes = load_trax_lines(meters=True)

    if route_buffer is None:  # noqa: SIM102
        if site.startswith("trx"):
            route_buffer = 50  # meters

    if storage_polygon is None:  # noqa: SIM102
        if site.startswith("trx"):
            storage_polygon = storage_locations["JRRSC"]

    print("Reading GPS data...")

    if org == "UATAQ":
        gps = uataq.read_data(
            "trx01",
            instruments="gps",
            lvl="final",
            time_range=time_range,
            num_processes=num_processes,
        )["gps"]
        gps = gpd.GeoDataFrame(
            gps,
            geometry=gpd.points_from_xy(gps.Longitude_deg, gps.Latitude_deg),
            crs="EPSG:4326",
        )
    else:
        raise ValueError(f"Organization {org} not supported for GPS loading.")

    # Trim altitude outliers
    gps = gps[
        (gps.Altitude_msl > gps.Altitude_msl.quantile(0.01))
        & (gps.Altitude_msl < gps.Altitude_msl.quantile(0.99))
    ]

    # Filter to locations within buffer of routes
    routes = get_geodf(routes)
    if routes is not None and route_buffer is not None:
        print("Filtering GPS points near routes...")
        routes_buff = gpd.GeoDataFrame(
            geometry=routes.buffer(route_buffer), crs=routes.crs
        )
        routes_buff = routes_buff.to_crs("EPSG:4326")
        gps = gpd.sjoin(gps, routes_buff, how="inner", predicate="within").drop(
            columns=["index_right"]
        )

    # Remove gps points within storage polygon
    storage_polygon = get_geodf(storage_polygon)
    if storage_polygon is not None:
        print("Removing GPS points within storage polygon...")
        gps = gpd.sjoin(gps, storage_polygon, how="left", predicate="within")
        gps = gps[gps.index_right.isnull()].drop(columns=["index_right"])

    gps = gps.drop(columns=["geometry"])

    # Merge obs and GPS data
    print(f"Merging {site} obs and GPS data...")
    obs = obs.set_index(
        "Time_UTC"
    )  # Ensure obs is indexed by time for merging with GPS data

    if org == "UATAQ":
        # For UATAQ (specifically lin group), the pi's time is not trustworthy,
        # so we need to get the UTC time from the GPS data
        # TODO if group == 'horel', this will need to be changed
        on = "Pi_Time"
        obs = obs.rename_axis(
            "Pi_Time", axis=0
        )  # Rename Time_UTC to Pi_Time for merging
    else:
        on = "Time_UTC"

    data = uataq.sites.MobileSite.merge_gps(obs, gps, on=on).reset_index()

    if "Pi_Time" in data.columns:
        data = data.drop(columns=["Pi_Time"])

    return data
