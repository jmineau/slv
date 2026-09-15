"""Where is the TRAX train: on the line, parked in the JRRSC yard, or inside the depot?

The trx01 train spends most nights at the Jordan River Rail Service Center (JRRSC).
Sometimes it is parked outside on the yard loop, sometimes inside the maintenance
depot, and the green line runs right past the yard's north edge. The pipeline's
storage flag (GPS ``QAQC_Flag == 20``) and the slv loader's storage polygon lump all
three together. This module separates them from the GPS behaviour, at one-minute
resolution:

* **depot** — stationary and the GPS fix is degraded: the per-minute position
  scatter is metres (multipath under the roof, the "crazy" GPS) and the satellite
  count is low. Site visits (which happen inside the depot) show 2.5–7 m scatter
  and 6–7 satellites; parked outside the scatter is < 0.3 m with 8–12 satellites.
  The scattered fixes' medians cluster inside the building footprint
  (:data:`DEPOT_FOOTPRINT`, derived from Jan–Aug 2025 data).
* **yard** — stationary (or creeping) with a clean fix, inside the storage polygon
  but not on the line. Includes the loop track within 50 m of the green line.
* **line** — moving (max speed ≥ :data:`MOVING_SPEED` m/s) within :data:`LINE_DISTANCE` m of
  the green line, whether or not inside the storage polygon (a pass-by).
* **route** — moving anywhere else (normal operation; the route-buffer test in
  :func:`slv.measurements.mobile.merge_with_gps` handles the rest).
* **stopped** — stationary away from the yard (a station stop, a siding).
* **unknown** — no usable GPS in the minute, or a position that cannot be trusted:
  more than :data:`OFF_TRACK` m from any TRAX track (GPS junk, or a siding missing
  from the line geojson), or within :data:`NEAR_YARD` m of the storage polygon but
  neither on the line nor inside the yard buffer (multipath ejecta from the depot).

Supporting evidence that is *not* used by the classifier but is worth plotting:
inside the heated depot the roof temperature sits at 22–23 °C with low RH, ozone
goes to ~0, CO2 climbs; none of these are season-independent on their own. Battery
voltage (cr1000) separates *powered* from *train power off* (< :data:`POWER_OFF_V`),
not depot from yard, so it is returned as a separate ``powered`` flag.

Typical use::

    feat = location_features(gps, cr1000=logger)  # 1-min feature table
    states = classify_location(feat)  # per-minute state
    intervals = state_intervals(states)  # start/end/state table

``gps`` needs a datetime index and ``Latitude_deg``, ``Longitude_deg``;
``Speed_m_s`` and ``N_Sat`` are optional. Speed is missing wherever only GPGGA was
logged (lin-group GPS Dec 2015 – 19 Jan 2018; horel pilot logger Nov 2014 – Nov 2018):
there ``moving`` comes from the position-derived ``speed_est`` instead. Either the lin-group GPS (``uataq.read_data
('trx01', 'gps', lvl='qaqc')``) or the horel-group logger GPS (:func:`read_horel_cr1000`)
works; :func:`read_trax_gps` picks by era (lin GPS for the pilot years, horel logger
from 19 Nov 2018, which runs on its own battery and keeps recording when train power
is off).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from slv import get_data_dir
from slv.measurements.mobile import get_geodf, load_trax_lines, storage_locations

UTM12 = "EPSG:32612"

#: Speed (m/s, per-minute maximum) at or above which the train counts as moving.
#: Depot multipath produces spurious speeds up to ~1 m/s; a real move exceeds 2.
MOVING_SPEED = 2.0
#: Threshold (m/s) for the position-derived speed estimate used when no speed was
#: recorded (GPGGA-only eras): displacement between the minute medians one minute
#: before and after, over 120 s. Reproduces the speed rule on 98.9 % of 2025 minutes.
MOVING_SPEED_EST = 1.5
#: Distance (m) from the green line within which a moving train is "on the line".
LINE_DISTANCE = 50.0
#: Per-minute position std (m, max of x/y) above which the fix is degraded.
SCATTER_M = 1.0
#: Satellite count at or below which the fix is degraded (only if ``N_Sat`` present).
LOW_NSAT = 6
#: Buffer (m) around the storage polygon: scattered depot fixes leak past its edge.
YARD_BUFFER = 30.0
#: Radius (m) around the storage polygon inside which an off-line, off-yard minute is
#: multipath ejecta from the depot rather than a real stop → ``unknown``.
NEAR_YARD = 300.0
#: Distance (m) from any TRAX track beyond which a minute is ``unknown`` (a train is
#: never that far off the mapped lines; catches GPS junk and unmapped sidings).
OFF_TRACK = 100.0
#: Logger battery voltage below which train power is off (charging level is ~13–14 V).
POWER_OFF_V = 12.5
#: Centred window (minutes) for the majority vote that removes single-minute flips.
SMOOTH_MIN = 15

STATES = ("depot", "yard", "line", "route", "stopped", "unknown")

HOREL_CR1000_DIRS = (
    Path("/uufs/chpc.utah.edu/common/home/horel-group/uutrax/cr1000"),  # Nov 2018 on
    Path(
        "/uufs/chpc.utah.edu/common/home/horel-group/uutrax_pilot/cr1000"
    ),  # Nov 2014 - Nov 2018
)  # TODO move into uataq once its horel GPS reader keeps NSAT/RSTS


def load_depot_footprint(meters: bool = False) -> gpd.GeoDataFrame:
    """Packaged JRRSC depot building footprint (``jrrsc_depot.geojson``).

    Data-derived: the 5–95 % box of the per-minute median positions of scattered
    (degraded) fixes, Jan–Aug 2025, padded 15 m. Roughly 120 × 180 m.
    """
    with files(__package__).joinpath("jrrsc_depot.geojson").open("r") as f:
        gdf = gpd.read_file(f)
    return gdf.to_crs(UTM12) if meters else gdf


def read_horel_cr1000(time_range, site: str = "trx01") -> pd.DataFrame:
    """Horel-group CR1000 logger record (5 s): GPS + battery voltage + roof T/RH.

    Reads the monthly ``TRX01_YYYY_MM_cr1000.h5`` files directly because the uataq
    horel reader drops the satellite count and RMC status. Columns are renamed to
    the uataq convention (``Latitude_deg``, ``Speed_m_s``, ``N_Sat``, ``Status``,
    ``Battery_Voltage_V``, ``Logger_T_C``, ``Ambient_T_C``, ``Ambient_RH_pct``).
    Covers Nov 2014 on (pilot files first, then the post-pilot tree). The pilot-phase
    files (to 19 Nov 2018) are a different setup: one fix per minute (so the scatter
    feature is undefined), a receiver that reports 3–8 satellites, and no speed or RMC
    status. Do not classify from them — use :func:`read_lin_gps` for the pilot era.
    ``time_range`` is any pair pandas can parse.
    """
    import tables

    start, end = (pd.Timestamp(t) for t in time_range)
    rename = {
        "GLAT": "Latitude_deg",
        "GLON": "Longitude_deg",
        "GELV": "Altitude_msl",
        "RDIR": "Course_deg",
        "NSAT": "N_Sat",
        "RSTS": "Status",
        "VOLT": "Battery_Voltage_V",
        "TICC": "Logger_T_C",
        "TRNT": "Ambient_T_C",
        "TRNR": "Ambient_RH_pct",
    }
    parts = []
    for m in pd.period_range(start, end, freq="M"):
        name = f"{site.upper()}_{m.year}_{m.month:02d}_cr1000.h5"
        for d in HOREL_CR1000_DIRS:
            if (d / name).exists():
                with tables.open_file(d / name) as h5:
                    parts.append(pd.DataFrame(h5.root["obsdata/observations"].read()))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).replace(-9999.0, np.nan)
    df.index = pd.to_datetime(df.pop("EPOCHTIME"), unit="s").rename("Time_UTC")
    if "RSPD" in df.columns:
        df["Speed_m_s"] = df.pop("RSPD") * 0.514444  # knots -> m/s
    else:
        df["Speed_m_s"] = np.nan
    df = df.drop(columns=["GTIM", "PRES"], errors="ignore").rename(columns=rename)
    df = df[~df.index.duplicated()].sort_index()
    return df.loc[start:end]


#: Post-pilot start of the horel logger (5-s GPS with speed and RMC status).
HOREL_POST_PILOT = pd.Timestamp("2018-11-19T20:04")

LIN_GPS_DIR = Path(get_data_dir("LINGROUP_MEASUREMENTS_DIR")) / "data"
#: GPS QAQC flags dropped before classifying: bad fix quality, < 4 satellites,
#: invalid RMC status, pi-clock overlap. Flag 20 (storage box) and 0 are kept.
LIN_GPS_DROP_FLAGS = (-21, -22, -23, -200)


def read_lin_gps(time_range, site: str = "trx01", lvl: str = "qaqc") -> pd.DataFrame:
    """Lin-group (air-trend) GPS at 1 s from the pipeline ``qaqc`` level, Dec 2014 on.

    Read directly from the monthly ``YYYY_MM_qaqc.dat`` files (uataq's reader drops
    ``N_Sat``). Rows with :data:`LIN_GPS_DROP_FLAGS` are removed. ``Speed_m_s`` is NA
    from Dec 2015 to 19 Jan 2018 (GPGGA only), which :func:`classify_location`
    handles via ``speed_est``. Indexed by GPS ``Time_UTC``; ``Pi_Time`` is kept for
    merging with the LGR. Only records while the Pi has train power.
    """
    start, end = (pd.Timestamp(t) for t in time_range)
    cols = [
        "Time_UTC",
        "Pi_Time",
        "Latitude_deg",
        "Longitude_deg",
        "Altitude_msl",
        "Speed_m_s",
        "Course_deg",
        "N_Sat",
        "Fix_Quality",
        "QAQC_Flag",
    ]
    parts = []
    for m in pd.period_range(start, end, freq="M"):
        path = LIN_GPS_DIR / site / "gps" / lvl / f"{m.year}_{m.month:02d}_{lvl}.dat"
        if not path.exists():
            continue
        df = pd.read_csv(
            path, usecols=lambda c: c in cols, na_values="NA", low_memory=False
        )
        df["Time_UTC"] = pd.to_datetime(df.Time_UTC, errors="coerce")
        df = df.dropna(subset=["Time_UTC"])
        if "QAQC_Flag" in df.columns:
            df = df[~df.QAQC_Flag.isin(LIN_GPS_DROP_FLAGS)]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).set_index("Time_UTC").sort_index()
    return df.loc[start:end]


def read_trax_gps(time_range, site: str = "trx01") -> pd.DataFrame:
    """Best GPS source for classifying, by era: lin-group GPS before
    :data:`HOREL_POST_PILOT`, horel logger (with battery/T/RH) after."""
    start, end = (pd.Timestamp(t) for t in time_range)
    parts = []
    if start < HOREL_POST_PILOT:
        parts.append(read_lin_gps((start, min(end, HOREL_POST_PILOT)), site))
    if end >= HOREL_POST_PILOT:
        parts.append(read_horel_cr1000((max(start, HOREL_POST_PILOT), end), site))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def _prep_gps(gps: pd.DataFrame) -> gpd.GeoDataFrame:
    g = gps.dropna(subset=["Latitude_deg", "Longitude_deg"])
    g = g[g.Latitude_deg.between(40.3, 41.2) & g.Longitude_deg.between(-112.3, -111.5)]
    pts = gpd.GeoSeries(
        gpd.points_from_xy(g.Longitude_deg, g.Latitude_deg), crs="EPSG:4326"
    ).to_crs(UTM12)
    return gpd.GeoDataFrame(g, geometry=pts.values, crs=UTM12)


def location_features(
    gps: pd.DataFrame,
    cr1000: pd.DataFrame | None = None,
    freq: str = "1min",
    storage_polygon=None,
    line: str = "G",
) -> pd.DataFrame:
    """Per-``freq`` GPS features that the classifier needs (plus power, if available).

    Columns: ``n_gps``, ``x``/``y`` (UTM median), ``scatter`` (max of x/y std, m),
    ``speed`` (median m/s), ``speed_max``, ``speed_est`` (position-derived, see
    :data:`MOVING_SPEED_EST`), ``d_line``, ``d_track`` (median distance to any
    TRAX track, m) (median distance to the
    ``line`` track, m), ``d_yard`` (median distance to the storage polygon, 0 inside),
    ``in_yard`` (fraction of fixes inside the polygon), ``in_depot`` (median position
    inside :func:`load_depot_footprint`), ``nsat`` (median, if present), and from
    ``cr1000``: ``volt`` (median), ``volt_min``, ``amb_T``, ``amb_RH``.
    """
    g = _prep_gps(gps)
    if "Speed_m_s" not in g.columns:
        g["Speed_m_s"] = np.nan
    yard = get_geodf(storage_polygon or storage_locations["JRRSC"]).to_crs(UTM12)
    yard_geom = yard.geometry.union_all()
    lines = load_trax_lines(meters=True)
    line_geom = lines[lines.line == line].geometry.union_all()
    track_geom = lines.geometry.union_all()

    g["x"] = g.geometry.x
    g["y"] = g.geometry.y
    g["d_line"] = g.geometry.distance(line_geom)
    g["d_track"] = g.geometry.distance(track_geom)
    g["d_yard"] = g.geometry.distance(yard_geom)
    g["in_yard"] = g.geometry.within(yard_geom)

    r = g.drop(columns="geometry").resample(freq)
    f = pd.DataFrame(
        {
            "n_gps": r.size(),
            "x": r.x.median(),
            "y": r.y.median(),
            "scatter": pd.concat([r.x.std(), r.y.std()], axis=1).max(axis=1),
            "speed": r.Speed_m_s.median(),
            "speed_max": r.Speed_m_s.max(),
            "d_line": r.d_line.median(),
            "d_track": r.d_track.median(),
            "d_yard": r.d_yard.median(),
            "in_yard": r.in_yard.mean(),
        }
    )
    # position-derived speed for minutes/eras without a recorded speed:
    # displacement between the minute medians one minute before and after, over 120 s
    f["speed_est"] = (
        np.hypot(f.x.shift(-1) - f.x.shift(1), f.y.shift(-1) - f.y.shift(1)) / 120
    )
    if "N_Sat" in g.columns:
        f["nsat"] = r.N_Sat.median()

    depot = load_depot_footprint(meters=True).geometry.union_all()
    med = gpd.GeoSeries(gpd.points_from_xy(f.x, f.y), crs=UTM12, index=f.index)
    f["in_depot"] = med.within(depot) & f.x.notna()

    if cr1000 is not None and len(cr1000) and "Battery_Voltage_V" in cr1000.columns:
        c = cr1000.resample(freq)
        f["volt"] = c.Battery_Voltage_V.median()
        f["volt_min"] = c.Battery_Voltage_V.min()
        f["amb_T"] = c.Ambient_T_C.median()
        f["amb_RH"] = c.Ambient_RH_pct.median()
    return f


def _smooth_bool(s: pd.Series, window: int) -> pd.Series:
    """Centred majority vote over ``window`` samples (NaN counts as 0)."""
    return s.astype(float).rolling(window, center=True, min_periods=1).mean() > 0.5


def classify_location(
    feat: pd.DataFrame,
    moving_speed: float = MOVING_SPEED,
    moving_speed_est: float = MOVING_SPEED_EST,
    line_distance: float = LINE_DISTANCE,
    scatter_m: float = SCATTER_M,
    low_nsat: int = LOW_NSAT,
    yard_buffer: float = YARD_BUFFER,
    near_yard_m: float = NEAR_YARD,
    off_track_m: float = OFF_TRACK,
    smooth_min: int = SMOOTH_MIN,
    use_footprint: bool = False,
) -> pd.DataFrame:
    """Classify each row of :func:`location_features` into one of :data:`STATES`.

    Rules, per minute:

    1. no fixes → ``unknown``
    2. ``speed_max >= moving_speed`` (or, where no speed was recorded,
       ``speed_est >= moving_speed_est``): ``d_line < line_distance`` → ``line``, else
       ``yard`` if within ``yard_buffer`` m of the storage polygon, else ``route``.
    3. stationary near the yard: degraded fix (``scatter > scatter_m`` or
       ``nsat <= low_nsat``, majority-voted over ``smooth_min`` minutes) → ``depot``;
       with ``use_footprint`` a clean fix whose median sits inside the depot
       footprint is also ``depot``. Off by default: the yard loop runs through the
       padded footprint, so it mislabels clean yard nights (e.g. 2025-02-16).
       Otherwise ``yard``.
    4. stationary elsewhere → ``stopped``.
    5. overrides to ``unknown``: ``d_track > off_track_m`` (unless inside the yard
       buffer), or within ``near_yard_m`` of the storage polygon while neither on the
       line nor inside the yard buffer.

    Returns a frame with ``state`` (categorical), ``degraded`` (raw per-minute
    flag), ``degraded_smooth`` and, when battery voltage is present, ``powered``
    (``volt_min >= POWER_OFF_V``).
    """
    out = pd.DataFrame(index=feat.index)
    has_fix = feat.n_gps.fillna(0) > 0
    moving = feat.speed_max >= moving_speed
    if "speed_est" in feat.columns:  # GPGGA-only eras: no recorded speed
        no_speed = feat.speed_max.isna()
        moving = moving | (no_speed & (feat.speed_est >= moving_speed_est))
    near_yard = feat.d_yard <= yard_buffer
    on_line = feat.d_line < line_distance

    degraded = feat.scatter > scatter_m
    if "nsat" in feat.columns:
        degraded |= feat.nsat <= low_nsat
    degraded &= has_fix & ~moving
    out["degraded"] = degraded
    # vote only among stationary near-yard minutes so passes/gaps don't dilute it
    stat_yard = has_fix & ~moving & near_yard
    vote = degraded.where(stat_yard)
    out["degraded_smooth"] = (
        _smooth_bool(vote.ffill(limit=2).fillna(False), smooth_min) & stat_yard
    )

    depot = out.degraded_smooth.copy()
    if use_footprint and "in_depot" in feat.columns:
        depot |= stat_yard & feat.in_depot.fillna(False)

    state = pd.Series("unknown", index=feat.index, dtype=object)
    state[has_fix & moving & on_line] = "line"
    state[has_fix & moving & ~on_line & near_yard] = "yard"
    state[has_fix & moving & ~on_line & ~near_yard] = "route"
    state[stat_yard] = "yard"
    state[depot] = "depot"
    state[has_fix & ~moving & ~near_yard] = "stopped"
    # positions that cannot be trusted
    untrusted = has_fix & ~near_yard & (feat.d_yard <= near_yard_m) & ~on_line
    if "d_track" in feat.columns:
        untrusted |= has_fix & ~near_yard & (feat.d_track > off_track_m)
    state[untrusted] = "unknown"
    out["state"] = pd.Categorical(state, categories=STATES)

    if "volt_min" in feat.columns:
        out["powered"] = feat.volt_min >= POWER_OFF_V
    return out


def state_intervals(
    states: pd.Series | pd.DataFrame, min_minutes: int = 0
) -> pd.DataFrame:
    """Run-length table of a per-minute state series: ``start``, ``end``, ``state``, ``minutes``.

    ``end`` is the last minute of the run (inclusive). Runs shorter than
    ``min_minutes`` are dropped (not merged) — useful to list depot stays only.
    """
    s = states["state"] if isinstance(states, pd.DataFrame) else states
    s = s.astype(object)
    run = (s != s.shift()).cumsum()
    grp = s.groupby(run)
    iv = pd.DataFrame(
        {
            "start": grp.apply(lambda x: x.index[0]),
            "end": grp.apply(lambda x: x.index[-1]),
            "state": grp.first(),
            "minutes": grp.size(),
        }
    ).reset_index(drop=True)
    return iv[iv.minutes >= min_minutes].reset_index(drop=True)


def label_observations(
    obs: pd.DataFrame, states: pd.DataFrame, time_col: str = "Time_UTC"
) -> pd.Series:
    """Per-observation ``state`` from the per-minute table (time floored to the minute).

    Times outside the classified range come back as NaN.
    """
    t = (
        pd.to_datetime(obs[time_col])
        if time_col in obs.columns
        else pd.Series(obs.index)
    )
    minute = pd.DatetimeIndex(t).floor("min")
    return pd.Series(
        states["state"].reindex(minute).values, index=obs.index, name="state"
    )
