"""The georeferenced TRAX CH4 observation record: build it from the pipeline levels,
tag every row with calibration provenance and location, cache it, and load it with a
location filter.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import uataq

from slv.measurements.mobile.calibration import (
    filter_cal_source,
    load_trax_uncalibrated_windows,
    select_uncalibrated,
)
from slv.measurements.mobile.gps import merge_with_gps
from slv.measurements.mobile.network import USER_DIR

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

    Uses :mod:`slv.measurements.mobile.location`: per-minute states from the best GPS
    source for the era (``states`` may be passed in to reuse a classification).
    Observations are matched on ``Time_UTC`` floored to the minute; minutes without a
    classification come back ``unknown`` / NA.
    """
    from slv.measurements.mobile.gps import read_trax_gps
    from slv.measurements.mobile.location import (
        classify_location,
        label_observations,
        location_features,
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
