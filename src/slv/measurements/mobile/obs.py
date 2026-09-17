"""The georeferenced TRAX CH4 observation record: build it from the pipeline levels,
tag every row with calibration provenance and location, cache it, and load it with a
location filter.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
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


#: Cavity-pressure band (torr) inside which manual-cal rows flagged −63 (pressure outside the
#: pipeline's 135–145 window) are kept and tagged ``low_pressure``. LGR 13-0221 ran at 131–134 torr
#: in Jan–Mar 2025 and 102–110 torr from Jul 2026 with CH4 within 0.02 ppm of its normal-pressure
#: behaviour (measurements/trax/record/outputs/pressure_flag_check.md); the 5–30 torr pressure
#: collapses of 2016/2017/2019 stay excluded.
LOW_PRESSURE_BAND: tuple[float, float] = (100.0, 145.0)


def _read_lgr(
    site,
    instrument,
    lvl,
    value_col,
    time_range,
    num_processes,
    keep: tuple[str, ...] = (),
    valid_flags: set | None = None,
) -> pd.DataFrame:
    """Read one LGR level via uataq, validate CH4, return Time_UTC + CH4 (+ ``keep`` columns).

    Only the needed columns are retained, so the full-width uataq frame is released as soon
    as this returns."""
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
    # uataq names the cavity pressure Internal_P_torr; the pipeline files say Cavity_P_torr
    df = df.rename(columns={value_col: "CH4", "Internal_P_torr": "Cavity_P_torr"})
    df["CH4"] = normalize_pollutant(df, "CH4", valid_flags=valid_flags)
    cols = ["Time_UTC", "CH4", *[c for c in keep if c in df.columns]]
    return pd.DataFrame(df[cols]).reset_index(drop=True)


def apply_low_pressure_rule(
    df: pd.DataFrame, band: tuple[float, float] | None = LOW_PRESSURE_BAND
) -> pd.DataFrame:
    """Keep rows flagged −63 whose ``Cavity_P_torr`` lies inside ``band`` and tag them.

    ``df`` must carry ``CH4`` (already validated with −63 among the accepted flags),
    ``QAQC_Flag`` and ``Cavity_P_torr``. Rows flagged −63 outside the band (or with no
    pressure) get ``CH4 = NaN``; every row gets a boolean ``low_pressure`` column
    (True for the kept −63 rows). ``band=None`` drops all −63 rows.
    """
    out = df.copy()
    is63 = out["QAQC_Flag"] == -63
    if band is None:
        inside = pd.Series(False, index=out.index)
    else:
        p = pd.to_numeric(out["Cavity_P_torr"], errors="coerce")
        inside = p.between(band[0], band[1])
    out.loc[is63 & ~inside, "CH4"] = np.nan
    out["low_pressure"] = (is63 & inside).to_numpy()
    return out


def _build_chunk(
    site, t0, t1, num_processes, windows, classify, low_pressure, gps_kwargs
):
    """One time chunk of :func:`build_trax_obs` (see there); returns a plain DataFrame."""
    from slv.measurements.pollutants import defaults

    time_range = (t0, t1)
    empty = pd.DataFrame(
        {"Time_UTC": pd.to_datetime([]), "CH4": pd.Series(dtype=float)}
    )
    print(f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] calibrated LGR ...", flush=True)
    try:
        cal = _read_lgr(
            site, "lgr_ugga", "calibrated", "CH4d_ppm_cal", time_range, num_processes
        )
        cal = cal[cal.CH4.notna()].assign(cal_source="pipeline")
    except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
        cal = empty.assign(cal_source="pipeline")

    print(f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] manual-cal LGR ...", flush=True)
    flags = {0, 1, 2, *defaults["CH4"]["valid_flags"]}
    if low_pressure is not None:
        flags.add(-63)
    try:
        man = _read_lgr(
            site,
            "lgr_ugga_manual_cal",
            "qaqc",
            "CH4d_ppm",
            time_range,
            num_processes,
            keep=("QAQC_Flag", "Cavity_P_torr"),
            valid_flags=flags,
        )
        if "QAQC_Flag" in man.columns and "Cavity_P_torr" in man.columns:
            man = apply_low_pressure_rule(man, low_pressure)
        man = man[man.CH4.notna()].assign(cal_source="manual_cal")
        man = man[
            [
                "Time_UTC",
                "CH4",
                "cal_source",
                *(["low_pressure"] if "low_pressure" in man else []),
            ]
        ]
    except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
        man = empty.assign(cal_source="manual_cal")

    parts = [cal, man]
    for i, (start, end) in enumerate(
        zip(windows["start"], windows["end"], strict=True)
    ):
        ws, we = max(start, t0), min(end, t1)
        if ws >= we:
            continue
        print(
            f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] uncalibrated window {ws.date()} -> {we.date()} ...",
            flush=True,
        )
        q = _read_lgr(site, "lgr_ugga", "qaqc", "CH4d_ppm", (ws, we), num_processes)
        parts.append(
            select_uncalibrated(q, windows.iloc[[i]], exclude_times=cal.Time_UTC)
        )

    obs = pd.concat(parts, ignore_index=True).sort_values("Time_UTC")
    obs = obs.drop_duplicates("Time_UTC", keep="first").rename(
        columns={"CH4": "CH4_ppm"}
    )
    if "low_pressure" in obs.columns:
        obs["low_pressure"] = obs["low_pressure"].fillna(False).astype(bool)
    else:
        obs["low_pressure"] = False
    del parts, cal, man
    if obs.empty:
        return obs

    data = merge_with_gps(
        site,
        "UATAQ",
        obs,
        time_range=time_range,
        num_processes=num_processes,
        **gps_kwargs,
    )
    if classify:
        print(f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] classifying location ...", flush=True)
        data = label_trax_location(data, site=site, time_range=time_range)
    return data


def build_trax_obs(
    site: str = "trx01",
    time_range=None,
    num_processes: int = 1,
    windows: pd.DataFrame | None = None,
    classify: bool = True,
    low_pressure: tuple[float, float] | None = LOW_PRESSURE_BAND,
    chunk: str = "YS",
    **gps_kwargs,
) -> gpd.GeoDataFrame:
    """Build the georeferenced TRAX CH4 record from the pipeline levels.

    Sources, each tagged in ``cal_source`` (see :data:`CAL_SOURCES`):
    ``lgr_ugga`` calibrated → ``pipeline``; ``lgr_ugga_manual_cal`` qaqc → ``manual_cal``;
    ``lgr_ugga`` qaqc inside the uncalibrated windows (default: the packaged table) →
    ``uncalibrated``. QC via :func:`normalize_pollutant` (flags {0,1,2,-64,-140}, ID −10,
    valid range). Manual-cal rows flagged −63 (cavity pressure outside 135–145 torr) are kept
    when the pressure is inside ``low_pressure`` (default :data:`LOW_PRESSURE_BAND`) and
    tagged ``low_pressure=True`` (:func:`apply_low_pressure_rule`); ``low_pressure=None``
    drops them. Then merged with *every* GPS fix by :func:`merge_with_gps` (no route buffer,
    no storage-yard removal) and, with ``classify``, labelled per minute by
    :func:`label_trax_location` (``state``, ``indoor``, ``yard_name``) so that the location
    filter is applied at load time (:func:`load_trax_obs`). Pass ``routes`` /
    ``storage_polygon`` in ``gps_kwargs`` to restore the old pre-filtering.

    The archive is processed in ``chunk``-sized pieces (pandas offset alias, default one
    year) so that only one chunk of full-width pipeline files is in memory at a time (the
    whole-archive read peaked at 250 GB). Still heavy — run on a compute node.
    """
    if windows is None:
        windows = load_trax_uncalibrated_windows()
    gps_kwargs.setdefault("routes", False)
    gps_kwargs.setdefault("storage_polygon", False)

    if time_range is None:
        t0, t1 = pd.Timestamp("2014-12-01"), pd.Timestamp.now().ceil("D")
    else:
        t0, t1 = (pd.Timestamp(t) for t in time_range)
    edges = pd.date_range(t0, t1, freq=chunk)
    edges = pd.DatetimeIndex([t0, *edges[(edges > t0) & (edges < t1)], t1])

    parts = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        part = _build_chunk(
            site, a, b, num_processes, windows, classify, low_pressure, gps_kwargs
        )
        if len(part):
            parts.append(pd.DataFrame(part))
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude_deg, data.Latitude_deg)
        if len(data)
        else None,
        crs="EPSG:4326",
    )


def load_trax_obs(
    cache: str | Path | None = None,
    include_uncalibrated: bool = True,
    include_low_pressure: bool = True,
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
    ``indoor`` and ``yard_name`` columns are always present. ``include_low_pressure=False``
    drops the manual-cal rows kept under the low-cavity-pressure rule (``low_pressure`` column,
    see :func:`build_trax_obs`).
    """
    cache = USER_DIR / "trax" / "obs.parquet" if cache is None else Path(cache)
    if cache.exists() and not rebuild:
        data = pd.read_parquet(cache)
        if any(c not in data.columns for c in ("cal_source", "state", "low_pressure")):
            raise ValueError(
                f"{cache} predates cal_source / location / low_pressure tagging; call with rebuild=True"
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
    data = filter_cal_source(data, include_uncalibrated)
    if not include_low_pressure and "low_pressure" in data.columns:
        data = data[~data["low_pressure"].astype(bool)]
    return gpd.GeoDataFrame(data)
