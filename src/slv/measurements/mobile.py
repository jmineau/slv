from importlib.resources import files
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import uataq
from lair.geo import points_along_line
from shapely import Point

from slv import get_data_dir

GROUP_DIR = Path(get_data_dir("LINGROUP_DATA_DIR"))
USER_DIR = Path(get_data_dir("SLV_USER_DATA_DIR"))

storage_locations = {
    # Jordan River Rail Service Center - where trx01/trx02 sleep
    "JRRSC": GROUP_DIR / "spatial/transportation/light_rail/JRRSC.geojson",
    # Midvale Rail Service Center - the old-train yard, where trx03 sleeps
    # (data-derived hull of trx03 parked positions; see trax_location)
    "MRSC": files(__package__).joinpath("mrsc.geojson"),
}


def get_geodf(
    obj: str | Path | gpd.GeoDataFrame | bool | None,
) -> gpd.GeoDataFrame | None:
    if isinstance(obj, gpd.GeoDataFrame):
        return obj
    elif isinstance(obj, (str, Path)):
        return gpd.read_file(obj)
    elif hasattr(obj, "open"):  # importlib.resources Traversable (packaged file)
        with obj.open("r") as f:
            return gpd.read_file(f)
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
        gps = filter_near_routes(gps, routes, route_buffer)

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


# ---------------------------------------------------------------------------
# TRAX CH4 observations (calibrated + uncalibrated windows), cached to parquet
# ---------------------------------------------------------------------------

CAL_SOURCES = ("pipeline", "manual_cal", "uncalibrated")
"""Provenance tag carried by every TRAX observation in ``cal_source``:

- ``pipeline``: pipeline-calibrated value (``CH4d_ppm_cal`` from the ``calibrated`` level).
- ``manual_cal``: the ``lgr_ugga_manual_cal`` instrument (no on-board tank since Nov 2023);
  the pipeline applies no calibration, so this is the analyzer's raw ``CH4d_ppm``.
- ``uncalibrated``: raw ``CH4d_ppm`` from the ``lgr_ugga`` qaqc level inside a window listed in
  ``trax_uncalibrated_windows.csv`` (tank empty, no valid reference). Same treatment as
  ``manual_cal``; the LGR's gain was within 0.5% of unity on either side of every window.
"""


def load_trax_uncalibrated_windows(
    path: str | Path | None = None, enabled_only: bool = True
) -> pd.DataFrame:
    """Windows where the LGR ran without a valid reference tank but the raw data are good.

    Packaged in ``trax_uncalibrated_windows.csv`` (columns: start, end, reason, enabled).
    Set ``enabled`` to false to drop a window without deleting the row.
    """
    if path is None:
        with (
            files(__package__).joinpath("trax_uncalibrated_windows.csv").open("r") as f
        ):
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["enabled"] = df["enabled"].astype(str).str.lower().isin(("true", "1", "yes"))
    if enabled_only:
        df = df.loc[df["enabled"].to_numpy()]
    return pd.DataFrame(df).reset_index(drop=True)


def select_uncalibrated(
    qaqc: pd.DataFrame, windows: pd.DataFrame, exclude_times: pd.Index | None = None
) -> pd.DataFrame:
    """Rows of a qaqc-level LGR frame that fall inside the uncalibrated windows.

    ``qaqc`` must have a ``Time_UTC`` column (or a datetime index) and a ``CH4`` column that
    has already passed :func:`slv.measurements.pollutants.normalize_pollutant`. Rows whose
    time is in ``exclude_times`` (e.g. times that do have a pipeline calibration) are dropped.
    Returns ``Time_UTC``, ``CH4`` and ``cal_source == "uncalibrated"``.
    """
    df = qaqc if "Time_UTC" in qaqc.columns else qaqc.reset_index()
    t = pd.to_datetime(df["Time_UTC"])
    mask = pd.Series(False, index=df.index)
    for start, end in zip(windows["start"], windows["end"], strict=True):
        mask |= (t >= start) & (t < end)
    out = df.loc[mask & df["CH4"].notna(), ["Time_UTC", "CH4"]].copy()
    if exclude_times is not None and len(exclude_times):
        out = out[~out["Time_UTC"].isin(exclude_times)]
    out["cal_source"] = "uncalibrated"
    return pd.DataFrame(out).reset_index(drop=True)


def filter_cal_source(
    df: pd.DataFrame, include_uncalibrated: bool = True
) -> pd.DataFrame:
    """Drop the ``uncalibrated`` rows when ``include_uncalibrated`` is False.

    ``manual_cal`` rows are always kept: they are the only post-Nov-2023 data.
    """
    if include_uncalibrated or "cal_source" not in df.columns:
        return df
    return pd.DataFrame(df.loc[(df["cal_source"] != "uncalibrated").to_numpy()])


#: Named sets of location states kept by :func:`load_trax_obs`.
#: ``on_track`` reproduces the old route-buffer behaviour (minus shed multipath ejecta,
#: plus pass-bys at the yards); ``outdoor`` adds minutes parked outside in a yard;
#: ``all`` keeps everything, including indoor (shed) and unknown minutes.
LOCATION_SETS: dict[str, tuple[str, ...] | None] = {
    "on_track": ("route", "line", "stopped"),
    "outdoor": ("route", "line", "stopped", "yard"),
    "all": None,
}


def filter_location(
    df: pd.DataFrame, location: str | tuple[str, ...] | None = "on_track"
) -> pd.DataFrame:
    """Keep rows whose ``state`` is in ``location`` (a :data:`LOCATION_SETS` name or a
    tuple of states). ``None``/``"all"`` keeps every row."""
    if location is None or "state" not in df.columns:
        return df
    states = LOCATION_SETS[location] if isinstance(location, str) else tuple(location)
    if states is None:
        return df
    return df[df["state"].isin(states)]


def label_trax_location(
    obs: pd.DataFrame,
    site: str = "trx01",
    time_range=None,
    states: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``state``, ``indoor`` and ``yard_name`` to a georeferenced TRAX obs frame.

    Uses :mod:`slv.measurements.trax_location`: per-minute states from the best GPS
    source for the era (``states`` may be passed in to reuse a classification).
    Observations are matched on ``Time_UTC`` floored to the minute; minutes without a
    classification come back ``unknown`` / NA.
    """
    from slv.measurements.trax_location import (
        classify_location,
        label_observations,
        location_features,
        read_trax_gps,
    )

    if states is None:
        if time_range is None:
            t = pd.to_datetime(obs["Time_UTC"])
            time_range = (t.min().floor("D"), t.max().ceil("D"))
        gps = read_trax_gps(time_range, site=site)
        if len(gps) == 0:
            states = None
        else:
            feat = location_features(gps, cr1000=gps)
            states = classify_location(feat)
    out = obs.copy()
    if states is None or len(states) == 0:
        out["state"] = "unknown"
        out["indoor"] = pd.array([pd.NA] * len(out), dtype="boolean")
        out["yard_name"] = None
        return out
    out["state"] = (
        label_observations(out, states).astype(object).fillna("unknown").values
    )
    minute = pd.DatetimeIndex(pd.to_datetime(out["Time_UTC"])).floor("min")
    out["indoor"] = states["indoor"].reindex(minute).values
    out["yard_name"] = (
        states["yard_name"].reindex(minute).values
        if "yard_name" in states.columns
        else None
    )
    return out


def _read_lgr(
    site, instrument, lvl, value_col, time_range, num_processes
) -> pd.DataFrame:
    """Read one LGR level via uataq, validate CH4, return Time_UTC + CH4 (+ index reset)."""
    from slv.measurements.pollutants import normalize_pollutant

    df = uataq.read_data(
        site,
        instruments=instrument,
        lvl=lvl,
        time_range=time_range,
        num_processes=num_processes,
    )[instrument]
    if "Time_UTC" not in df.columns:
        df = df.reset_index()
    df = df.rename(columns={value_col: "CH4"})
    df["CH4"] = normalize_pollutant(df, "CH4")
    return pd.DataFrame(df[["Time_UTC", "CH4"]])


def build_trax_obs(
    site: str = "trx01",
    time_range=None,
    num_processes: int = 1,
    windows: pd.DataFrame | None = None,
    classify: bool = True,
    **gps_kwargs,
) -> gpd.GeoDataFrame:
    """Build the georeferenced TRAX CH4 record from the pipeline levels.

    Sources, each tagged in ``cal_source`` (see :data:`CAL_SOURCES`):
    ``lgr_ugga`` calibrated → ``pipeline``; ``lgr_ugga_manual_cal`` qaqc → ``manual_cal``;
    ``lgr_ugga`` qaqc inside the uncalibrated windows (default: the packaged table) →
    ``uncalibrated``. QC via :func:`normalize_pollutant` (flags {0,1,2,-64,-140}, ID −10,
    valid range). Then merged with *every* GPS fix by :func:`merge_with_gps` (no route
    buffer, no storage-yard removal) and, with ``classify``, labelled per minute by
    :func:`label_trax_location` (``state``, ``indoor``, ``yard_name``) so that the
    location filter is applied at load time (:func:`load_trax_obs`). Pass ``routes`` /
    ``storage_polygon`` in ``gps_kwargs`` to restore the old pre-filtering. Heavy: reads
    the full pipeline archive — run on a compute node.
    """
    if windows is None:
        windows = load_trax_uncalibrated_windows()

    empty = pd.DataFrame(
        {"Time_UTC": pd.to_datetime([]), "CH4": pd.Series(dtype=float)}
    )

    print("Reading calibrated LGR data...")
    try:
        cal = _read_lgr(
            site, "lgr_ugga", "calibrated", "CH4d_ppm_cal", time_range, num_processes
        )
        cal = cal[cal.CH4.notna()].assign(cal_source="pipeline")
    except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
        cal = empty.assign(
            cal_source="pipeline"
        )  # e.g. post-Nov-2023 ranges: manual cal only

    print("Reading manual-cal LGR data...")
    try:
        man = _read_lgr(
            site, "lgr_ugga_manual_cal", "qaqc", "CH4d_ppm", time_range, num_processes
        )
        man = man[man.CH4.notna()].assign(cal_source="manual_cal")
    except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
        man = empty.assign(cal_source="manual_cal")

    parts = [cal, man]
    for i, (start, end) in enumerate(
        zip(windows["start"], windows["end"], strict=True)
    ):
        print(f"Reading uncalibrated window {start.date()} -> {end.date()} ...")
        q = _read_lgr(site, "lgr_ugga", "qaqc", "CH4d_ppm", (start, end), num_processes)
        parts.append(
            select_uncalibrated(q, windows.iloc[[i]], exclude_times=cal.Time_UTC)
        )

    obs = pd.concat(parts, ignore_index=True).sort_values("Time_UTC")
    obs = obs.drop_duplicates("Time_UTC", keep="first").rename(
        columns={"CH4": "CH4_ppm"}
    )

    gps_kwargs.setdefault("routes", False)
    gps_kwargs.setdefault("storage_polygon", False)
    data = merge_with_gps(
        site,
        "UATAQ",
        obs,
        time_range=time_range,
        num_processes=num_processes,
        **gps_kwargs,
    )
    if classify:
        print("Classifying location (indoor / yard / line) ...")
        data = label_trax_location(data, site=site, time_range=time_range)
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude_deg, data.Latitude_deg),
        crs="EPSG:4326",
    )


def load_trax_obs(
    cache: str | Path | None = None,
    include_uncalibrated: bool = True,
    location: str | tuple[str, ...] | None = "on_track",
    rebuild: bool = False,
    **build_kwargs,
) -> gpd.GeoDataFrame:
    """Load the cached TRAX CH4 record (``$SLV_USER_DATA_DIR/trax/obs.parquet``), building it if needed.

    ``include_uncalibrated=False`` drops the tank-out windows so their effect can be tested;
    the ``cal_source`` column is always present for finer filtering. ``location`` selects
    where the train was (:data:`LOCATION_SETS`): ``"on_track"`` (default) keeps
    route / line / stopped, ``"outdoor"`` also keeps yard-parked minutes, ``"all"`` keeps
    everything (indoor shed air and untrusted positions included); the ``state``,
    ``indoor`` and ``yard_name`` columns are always present.
    """
    cache = USER_DIR / "trax" / "obs.parquet" if cache is None else Path(cache)
    if cache.exists() and not rebuild:
        data = pd.read_parquet(cache)
        if "cal_source" not in data.columns or "state" not in data.columns:
            raise ValueError(
                f"{cache} predates cal_source / location tagging; call with rebuild=True"
            )
        data = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data.Longitude_deg, data.Latitude_deg),
            crs="EPSG:4326",
        )
    else:
        data = build_trax_obs(**build_kwargs)
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching TRAX obs to {cache}")
        pd.DataFrame(data.drop(columns="geometry")).to_parquet(cache)
    return gpd.GeoDataFrame(filter_cal_source(data, include_uncalibrated))


# ---------------------------------------------------------------------------
# TRAX transect archive (transect x point matrices per line and month)
# ---------------------------------------------------------------------------

TRANSECT_LINES = {"r": "Red", "g": "Green", "b": "Blue"}


def load_transects(line: str, months=None, transects_dir: str | Path | None = None):
    """Concatenate the archived transect matrices for one TRAX line into an xarray Dataset.

    Files: ``$SLV_USER_DATA_DIR/trax/transects/trx01_CH4_<line>_YYYY-MM.nc`` (dims
    ``transect`` x ``point``; variables ``obs`` [ppm], ``time`` [POSIX s], ``n``; point
    coords ``lat``/``lon``). ``months`` is an optional iterable of ``"YYYY-MM"`` strings.
    Returns the Dataset with a ``month`` coordinate on the transect dimension.
    """
    import xarray as xr

    d = (
        USER_DIR / "trax" / "transects"
        if transects_dir is None
        else Path(transects_dir)
    )
    files = sorted(d.glob(f"trx01_CH4_{line}_*.nc"))
    if months is not None:
        want = set(months)
        files = [f for f in files if f.stem.split("_")[-1] in want]
    if not files:
        raise FileNotFoundError(f"no transect files for line {line!r} in {d}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.assign_coords(
            month=("transect", [f.stem.split("_")[-1]] * ds.sizes["transect"])
        )
        parts.append(ds)
    out = xr.concat(parts, dim="transect", combine_attrs="drop_conflicts")
    return out.assign_coords(transect=np.arange(out.sizes["transect"]))
