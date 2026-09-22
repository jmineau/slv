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
    apply_epoch_offset,
    epoch_offset_column,
    filter_cal_source,
    load_trax_uncalibrated_windows,
    select_uncalibrated,
)
from slv.measurements.mobile.gps import merge_with_gps
from slv.measurements.mobile.network import user_dir

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


#: Cavity-pressure band (torr) the LGR is specified for; the pipeline's own ``-63`` flag uses the
#: same bounds. It has to be re-checked in the loader because ``lgr_ugga_qaqc.r`` assigns flags in
#: sequence, so ``-64`` (cavity T out of range) OVERWRITES ``-63`` on a row that fails both — and
#: ``-64`` is an accepted flag. 360 on-track minutes at 100-135 torr reached obs.parquet that way
#: (2016-12-30/31 and 2017-01-08, the sample line starving; low tail -0.10 ppm).
PRESSURE_BAND: tuple[float, float] = (135.0, 145.0)

#: An atmosphere row below this (ppm) is a CH4 laser dropout, not air: the global background is
#: ~1.85-2.0 ppm and the pipeline's valid range starts at 1.70, so the shoulder of a dropout passes
#: QC. Measured bias of the surviving rows: -0.29 ppm one minute out, -0.22 within 8 min, gone by 15.
DROPOUT_CH4_PPM: float = 1.70

#: Minutes either side of a dropout row that are dropped with it (:func:`apply_dropout_rule`).
DROPOUT_WINDOW_MIN: int = 8

#: Calibrated rows whose slope ``CH4d_m`` deviates more than this fraction from the day's median
#: slope are dropped. One bad reference period (a restart with air still in the line, a dying
#: tank) is interpolated over the next hour by the pipeline's single-tank calibration; the
#: ±100 ppm tolerances never catch it (measurements/trax/record/outputs/trax_cal_audit.txt).
SLOPE_TOL: float | None = 0.05


def apply_slope_guard(df: pd.DataFrame, tol: float | None = SLOPE_TOL) -> pd.DataFrame:
    """Set ``CH4`` to NaN where ``CH4d_m`` differs from the same UTC day's median slope by more
    than ``tol`` (fraction). ``df`` needs ``Time_UTC``, ``CH4`` and ``CH4d_m``; ``tol=None`` is a
    no-op. Days with fewer than 100 sloped rows are left alone (no robust median)."""
    if tol is None or "CH4d_m" not in df.columns:
        return df
    out = df.copy()
    # force plain float64 whatever uataq returned (object / Arrow strings with multiprocess reads)
    m = pd.Series(
        pd.to_numeric(out["CH4d_m"].astype(object), errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        ),
        index=out.index,
    )
    day = pd.to_datetime(out["Time_UTC"]).dt.floor("D")
    med = m.groupby(day).transform("median")
    n = m.notna().groupby(day).transform("sum")
    bad = m.notna() & (n >= 100) & ((m / med - 1).abs() > tol)
    out.loc[bad, "CH4"] = np.nan
    return out


def dropout_times(
    support: pd.DataFrame, threshold: float = DROPOUT_CH4_PPM
) -> pd.Series:
    """Times of atmosphere rows whose *unvalidated* CH4 is below ``threshold`` — laser dropouts.

    ``support`` is a :func:`_read_support` frame (``Time_UTC``, ``CH4_raw``, ``ID_CH4``).
    Rows at or above the pipeline's 1.70 ppm valid minimum are the ones that survive QC, so the
    dropouts themselves have to be found before validation and used to mask their neighbours
    (:func:`apply_dropout_rule`).
    """
    if support.empty:
        return pd.Series(dtype="datetime64[ns]")
    atm = support["ID_CH4"].isna() | (support["ID_CH4"] == -10)
    bad = atm & support["CH4_raw"].lt(threshold)
    return pd.Series(pd.to_datetime(support.loc[bad, "Time_UTC"]).to_numpy())


def apply_dropout_rule(
    df: pd.DataFrame, dropouts: pd.Series, window_min: int = DROPOUT_WINDOW_MIN
) -> pd.DataFrame:
    """Drop rows within ``window_min`` minutes of a CH4 laser dropout.

    The analyzer does not fail cleanly: on the way into (and out of) a dropout it reads a few
    per cent to 25 % low for minutes to hours, and everything it reports above 1.70 ppm passes
    QC. ``dropouts`` comes from :func:`dropout_times`.
    """
    if df.empty or dropouts is None or len(dropouts) == 0:
        return df
    t = pd.to_datetime(df["Time_UTC"]).to_numpy("datetime64[m]").astype("int64")
    d = np.unique(pd.to_datetime(dropouts).to_numpy("datetime64[m]").astype("int64"))
    i = np.searchsorted(d, t)
    prev = np.abs(t - d[np.clip(i - 1, 0, len(d) - 1)])
    nxt = np.abs(d[np.clip(i, 0, len(d) - 1)] - t)
    return pd.DataFrame(df.loc[np.minimum(prev, nxt) > window_min])


def apply_pressure_rule(
    df: pd.DataFrame,
    pressure: pd.Series,
    band: tuple[float, float] | None = PRESSURE_BAND,
    sources: tuple[str, ...] = ("pipeline", "uncalibrated"),
) -> pd.DataFrame:
    """Drop ``sources`` rows whose cavity pressure is outside ``band`` (see :data:`PRESSURE_BAND`).

    ``pressure`` is indexed by ``Time_UTC``. Rows with no pressure reading are kept. The
    ``manual_cal`` rows are left alone by default: they are governed by
    :func:`apply_low_pressure_rule`, which deliberately keeps 100-135 torr for the analyzers that
    ran there (validated against UOU).
    """
    if band is None or df.empty or pressure is None or len(pressure) == 0:
        return df
    p = pressure[~pressure.index.duplicated()]
    val = pd.to_numeric(
        p.reindex(pd.to_datetime(df["Time_UTC"])), errors="coerce"
    ).to_numpy()
    in_band = np.isnan(val) | ((val >= band[0]) & (val <= band[1]))
    keep = in_band | ~df["cal_source"].isin(sources).to_numpy()
    return pd.DataFrame(df.loc[keep])


def recover_hot_rows(
    cal: pd.DataFrame, support: pd.DataFrame, flag: int = -64
) -> pd.DataFrame:
    """Calibrate the rows the pipeline flagged ``-64`` (cavity T outside 5-45 C) itself.

    ``lgr_ugga_calibrate`` fits the slope for these rows (``CH4d_m`` is there) but blanks
    ``CH4d_ppm_cal``, so ~645 on-track hours of summer afternoons — the best-mixed hours of the
    record — never reach obs.parquet. The value is simply the qaqc reading over the fitted slope.
    Hot-cavity CH4 is unbiased: on the same days, the 10th percentile above 45 C differs from
    below by -0.002 ppm (IQR -0.008..+0.010, 65 days; measurements/trax/audit).

    Returns the recovered rows (``Time_UTC``, ``CH4``), which the caller appends to the
    calibrated ones.
    """
    need = {"Time_UTC", "CH4d_m", "QAQC_Flag"}
    if cal.empty or support.empty or not need.issubset(cal.columns):
        return pd.DataFrame(
            {"Time_UTC": pd.to_datetime([]), "CH4": pd.Series(dtype=float)}
        )
    hot = cal[(cal["QAQC_Flag"] == flag) & cal["CH4"].isna()]
    if hot.empty:
        return pd.DataFrame(
            {"Time_UTC": pd.to_datetime([]), "CH4": pd.Series(dtype=float)}
        )
    m = pd.to_numeric(
        pd.Series(hot["CH4d_m"].to_numpy(), dtype=object), errors="coerce"
    )
    raw = support.drop_duplicates("Time_UTC").set_index("Time_UTC")["CH4_raw"]
    val = pd.to_numeric(raw.reindex(pd.to_datetime(hot["Time_UTC"])), errors="coerce")
    out = pd.DataFrame(
        {
            "Time_UTC": pd.to_datetime(hot["Time_UTC"]).to_numpy(),
            "CH4": val.to_numpy() / m.to_numpy(),
        }
    )
    return out[out["CH4"].between(1.7, 300)].reset_index(drop=True)


def _read_support(site, instrument, time_range, num_processes) -> pd.DataFrame:
    """Unvalidated qaqc columns the rules need: CH4 as reported, flag, cavity pressure, ID.

    Deliberately skips :func:`~slv.measurements.pollutants.normalize_pollutant`: the dropout rule
    has to see the sub-1.70 ppm rows that validation removes, and the pressure rule has to see
    rows whose ``-63`` was overwritten by ``-64``.
    """
    try:
        df = uataq.read_data(
            site,
            instruments=instrument,
            lvl="qaqc",
            time_range=time_range,
            num_processes=num_processes,
        )[instrument]
    except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
        return pd.DataFrame(
            {
                "Time_UTC": pd.to_datetime([]),
                "CH4_raw": pd.Series(dtype=float),
                "QAQC_Flag": pd.Series(dtype=float),
                "Cavity_P_torr": pd.Series(dtype=float),
                "ID_CH4": pd.Series(dtype=float),
            }
        )
    if "Time_UTC" not in df.columns:
        df = df.reset_index()
    df = df.rename(columns={"Internal_P_torr": "Cavity_P_torr", "CH4d_ppm": "CH4_raw"})
    for c in ("CH4_raw", "QAQC_Flag", "Cavity_P_torr", "ID_CH4"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return pd.DataFrame(
        df[["Time_UTC", "CH4_raw", "QAQC_Flag", "Cavity_P_torr", "ID_CH4"]]
    ).reset_index(drop=True)


def _pressure_series(support: pd.DataFrame) -> pd.Series:
    """Cavity pressure indexed by ``Time_UTC`` (for :func:`apply_pressure_rule`)."""
    if support.empty:
        return pd.Series(dtype=float)
    s = support.dropna(subset=["Cavity_P_torr"]).drop_duplicates("Time_UTC")
    return pd.Series(
        s["Cavity_P_torr"].to_numpy(), index=pd.to_datetime(s["Time_UTC"]).to_numpy()
    )


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
    site,
    t0,
    t1,
    num_processes,
    windows,
    classify,
    low_pressure,
    slope_tol,
    gps_kwargs,
    last=True,
    pressure_band=PRESSURE_BAND,
    dropout_window=DROPOUT_WINDOW_MIN,
    recover_hot=True,
):
    """One time chunk of :func:`build_trax_obs` (see there); returns a plain DataFrame.

    uataq time ranges include both ends, so unless this is the ``last`` chunk, LGR rows at
    exactly ``t1`` are left to the next chunk.
    """
    from slv.measurements.pollutants import defaults

    time_range = (t0, t1)
    empty = pd.DataFrame(
        {"Time_UTC": pd.to_datetime([]), "CH4": pd.Series(dtype=float)}
    )
    print(
        f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] qaqc support (dropouts, pressure) ...",
        flush=True,
    )
    support = pd.concat(
        [
            _read_support(site, "lgr_ugga", time_range, num_processes),
            _read_support(site, "lgr_ugga_manual_cal", time_range, num_processes),
        ],
        ignore_index=True,
    )
    drops = dropout_times(support) if dropout_window is not None else None

    print(f"[{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}] calibrated LGR ...", flush=True)
    try:
        cal = _read_lgr(
            site,
            "lgr_ugga",
            "calibrated",
            "CH4d_ppm_cal",
            time_range,
            num_processes,
            keep=("CH4d_m", "QAQC_Flag"),
        )
        cal = apply_slope_guard(cal, slope_tol)
        hot = recover_hot_rows(cal, support) if recover_hot else empty
        cal = pd.concat(
            [cal[cal.CH4.notna()][["Time_UTC", "CH4"]], hot[["Time_UTC", "CH4"]]],
            ignore_index=True,
        ).assign(cal_source="pipeline")
        if len(hot):
            print(f"  recovered {len(hot):,} flag -64 rows (hot cavity)", flush=True)
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
        try:
            q = _read_lgr(site, "lgr_ugga", "qaqc", "CH4d_ppm", (ws, we), num_processes)
        except (FileNotFoundError, KeyError, ValueError, uataq.errors.ReaderError):
            continue
        parts.append(
            select_uncalibrated(q, windows.iloc[[i]], exclude_times=cal.Time_UTC)
        )

    # stable sort: of rows sharing a time the first kept is the higher-priority source, in
    # the order of ``parts`` (pipeline > manual_cal > uncalibrated)
    obs = pd.concat(parts, ignore_index=True).sort_values("Time_UTC", kind="stable")
    obs = obs.drop_duplicates("Time_UTC", keep="first").rename(
        columns={"CH4": "CH4_ppm"}
    )
    if "low_pressure" in obs.columns:
        obs["low_pressure"] = obs["low_pressure"].fillna(False).astype(bool)
    else:
        obs["low_pressure"] = False
    if not last:
        obs = obs[obs["Time_UTC"] < t1]

    n0 = len(obs)
    obs = apply_pressure_rule(obs, _pressure_series(support), pressure_band)
    n1 = len(obs)
    if dropout_window is not None:
        obs = apply_dropout_rule(obs, drops, dropout_window)
    if n0:
        print(
            f"  pressure rule dropped {n0 - n1:,}; dropout rule dropped {n1 - len(obs):,}",
            flush=True,
        )
    obs["epoch_offset_ppm"] = epoch_offset_column(obs["Time_UTC"]).to_numpy()
    del parts, cal, man, support
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
    slope_tol: float | None = SLOPE_TOL,
    pressure_band: tuple[float, float] | None = PRESSURE_BAND,
    dropout_window: int | None = DROPOUT_WINDOW_MIN,
    recover_hot: bool = True,
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
    drops them. Pipeline-calibrated rows whose slope deviates more than ``slope_tol`` from the
    day's median slope are dropped (:func:`apply_slope_guard`; one bad reference period
    otherwise miscalibrates the next hour by 20–100 %). Then merged with *every* GPS fix by :func:`merge_with_gps` (no route buffer,
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
            site,
            a,
            b,
            num_processes,
            windows,
            classify,
            low_pressure,
            slope_tol,
            gps_kwargs,
            last=b == edges[-1],
            pressure_band=pressure_band,
            dropout_window=dropout_window,
            recover_hot=recover_hot,
        )
        if len(part):
            parts.append(pd.DataFrame(part))
    data = pd.concat(parts, ignore_index=True) if parts else _empty_obs(classify)
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude_deg, data.Latitude_deg),
        crs="EPSG:4326",
    )


def _empty_obs(classify: bool) -> pd.DataFrame:
    """The columns :func:`build_trax_obs` always returns, with no rows."""
    cols = {
        "Time_UTC": "datetime64[ns]",
        "CH4_ppm": float,
        "cal_source": object,
        "low_pressure": bool,
        "epoch_offset_ppm": float,
        "Latitude_deg": float,
        "Longitude_deg": float,
    }
    if classify:
        cols |= {"state": object, "indoor": "boolean", "yard_name": object}
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in cols.items()})


def load_trax_obs(
    cache: str | Path | None = None,
    include_uncalibrated: bool = True,
    include_low_pressure: bool = True,
    epoch_offset: bool = True,
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
    cache = user_dir() / "trax" / "obs.parquet" if cache is None else Path(cache)
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
    data = apply_epoch_offset(data, epoch_offset)
    data = filter_cal_source(data, include_uncalibrated)
    if not include_low_pressure and "low_pressure" in data.columns:
        data = data[~data["low_pressure"].astype(bool)]
    data = filter_location(data, location)
    return gpd.GeoDataFrame(data)
