"""GPS records for the mobile platforms and the observation/GPS merge.

Two GPS sources exist for the TRAX trains:

* **lin-group (air-trend) GPS**, 1 s, Dec 2014 on, in the UATAQ pipeline
  (``qaqc``/``final`` levels; ``final`` is what observations are merged with). Speed
  and RMC status come from GPRMC, which was not logged Dec 2015 – 19 Jan 2018.
  Records only while the Pi has train power.
* **horel-group CR1000 logger**, 5 s from 19 Nov 2018 (pilot files before that are
  1-min and unusable for classification), with battery voltage and roof T/RH; runs
  on its own battery so it keeps recording when train power is off.

:func:`read_trax_gps` picks by era; :func:`merge_with_gps` attaches positions to an
observation frame (the pipeline's ``final`` GPS on the Pi clock).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import uataq

from slv.measurements.mobile.network import (
    filter_near_routes,
    get_geodf,
    load_trax_lines,
    storage_locations,
)

#: GPS columns forced to float64 on read. A year read on its own can hand back an all-NA
#: column (Speed_m_s / Course_deg in the GPGGA-only years) as Arrow strings, which breaks the
#: per-minute medians in :mod:`slv.measurements.mobile.location`.
GPS_NUMERIC = (
    "Latitude_deg",
    "Longitude_deg",
    "Altitude_msl",
    "Speed_m_s",
    "Course_deg",
    "N_Sat",
    "Fix_Quality",
    "QAQC_Flag",
    "Battery_Voltage_V",
    "Logger_T_C",
    "Ambient_T_C",
    "Ambient_RH_pct",
)


def _coerce_numeric(df: pd.DataFrame, cols=GPS_NUMERIC) -> pd.DataFrame:
    """Cast the listed columns (where present) to plain float64, NaN for anything unparsable."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(object), errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
    return df


#: Plausible GPS altitude (m MSL) for mobile platforms around Salt Lake: the Great Salt
#: Lake shore is ~1280 m and the highest paved roads in reach (the Uinta passes on the
#: Mirror Lake Highway, ~3290 m; Guardsman Pass ~2980 m) stay under the top. Fixes outside
#: are GPS junk.
ALTITUDE_RANGE_MSL = (1000.0, 3500.0)

#: Post-pilot start of the horel logger (5-s GPS with speed and RMC status).
HOREL_POST_PILOT = pd.Timestamp("2018-11-19T20:04")

#: Lin-group GPS QAQC flags dropped before classifying: bad fix quality, < 4
#: satellites, invalid RMC status, pi-clock overlap. Flag 20 (storage box) and 0 stay.
LIN_GPS_DROP_FLAGS = (-21, -22, -23, -200)


def read_horel_cr1000(time_range, site: str = "trx01") -> pd.DataFrame:
    """Horel-group CR1000 logger record (5 s): GPS + battery voltage + roof T/RH.

    ``uataq`` ``raw`` level of the ``gps`` and ``cr1000`` instruments (the monthly
    ``TRXnn_YYYY_MM_cr1000.h5`` files), joined on time. The ``raw`` level is the one to
    use: the horel ``qaqc``/``final`` CSVs drop the satellite count and RMC status.
    Columns follow the uataq convention with ``N_Satellites`` renamed to ``N_Sat``
    (``Latitude_deg``, ``Speed_m_s``, ``N_Sat``, ``Status``, ``Battery_Voltage_V``,
    ``Logger_T_C``, ``Ambient_T_C``, ``Ambient_RH_pct``).

    Covers Nov 2014 on (pilot files, then the post-pilot tree). The pilot-phase files
    (to :data:`HOREL_POST_PILOT`) are a different setup: one fix per minute (so the
    scatter feature is undefined), a receiver that reports 3–8 satellites, and no
    speed or RMC status. Do not classify from them — use :func:`read_lin_gps` for the
    pilot era. ``time_range`` is any pair pandas can parse.
    """
    try:
        d = uataq.read_data(
            site,
            instruments=["gps", "cr1000"],
            group="horel",
            lvl="raw",
            time_range=time_range,
        )
    except uataq.errors.ReaderError:
        return pd.DataFrame()
    gps = d.get("gps", pd.DataFrame())
    cr = d.get("cr1000", pd.DataFrame())
    df = (
        gps.join(cr, how="outer") if len(gps) and len(cr) else (gps if len(gps) else cr)
    )
    if not len(df):
        return pd.DataFrame()
    df = _coerce_numeric(
        df.rename(columns={"N_Satellites": "N_Sat"}).drop(
            columns=["Instrument_Time"], errors="ignore"
        )
    )
    if "Speed_m_s" not in df.columns:
        df["Speed_m_s"] = np.nan
    df.index.name = "Time_UTC"
    return df[~df.index.duplicated()].sort_index()


def read_lin_gps(time_range, site: str = "trx01", lvl: str = "qaqc") -> pd.DataFrame:
    """Lin-group (air-trend) GPS at 1 s from the pipeline ``qaqc`` level, Dec 2014 on.

    ``uataq`` read with ``N_Satellites`` renamed to ``N_Sat``; rows with
    :data:`LIN_GPS_DROP_FLAGS` are removed. ``Speed_m_s`` is NA from Dec 2015 to
    19 Jan 2018 (GPGGA only), which :func:`~.location.classify_location` handles via
    ``speed_est``. Indexed by GPS ``Time_UTC``; ``Pi_Time`` is kept for merging with
    the LGR. Only records while the Pi has train power.
    """
    try:
        df = uataq.read_data(site, instruments="gps", lvl=lvl, time_range=time_range)[
            "gps"
        ]
    except uataq.errors.ReaderError:
        return pd.DataFrame()
    df = _coerce_numeric(df.rename(columns={"N_Satellites": "N_Sat"}))
    if "QAQC_Flag" in df.columns:
        df = df[~df.QAQC_Flag.isin(LIN_GPS_DROP_FLAGS)]
    df.index.name = "Time_UTC"
    return df.sort_index()


def read_trax_gps(
    time_range, site: str = "trx01", fill_gaps: bool = True
) -> pd.DataFrame:
    """Best GPS source for classifying, by era: lin-group GPS before
    :data:`HOREL_POST_PILOT`, horel logger (with battery/T/RH) after.

    With ``fill_gaps`` (default), minutes the horel logger did not record are filled from the
    lin GPS. The horel logger goes down for days at a time (its battery drained 2024-07-23 to
    08-02; the logger was offline around 2025-09-02) and without this the classifier calls every
    such minute ``unknown`` — ~235 on-track hours of a normally running train, mostly Jul 2024,
    Jan 2025, Sep 2025 and Jul 2026. Filled rows carry no ``Battery_Voltage_V``, so the
    depot/yard rules fall back to their GPS-only form.
    """
    start, end = (pd.Timestamp(t) for t in time_range)
    parts = []
    if start < HOREL_POST_PILOT:
        parts.append(read_lin_gps((start, min(end, HOREL_POST_PILOT)), site))
    if end >= HOREL_POST_PILOT:
        post = (max(start, HOREL_POST_PILOT), end)
        horel = read_horel_cr1000(post, site)
        parts.append(horel)
        if fill_gaps:
            lin = read_lin_gps(post, site)
            if len(lin):
                covered = (
                    pd.DatetimeIndex(horel.index).floor("min").unique()
                    if len(horel)
                    else pd.DatetimeIndex([])
                )
                gap = lin[~pd.DatetimeIndex(lin.index).floor("min").isin(covered)]
                if len(gap):
                    print(f"Filling {len(gap):,} lin-GPS rows into horel gaps")
                    parts.append(gap)
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


def lin_clock_offset_by_day(gps: pd.DataFrame) -> pd.Series:
    """Median GPS-minus-Pi clock offset (s) per UTC day, from a lin GPS frame.

    The Pi had no working real-time clock for most of the record, so its time can be hours to
    days off. A horel fix can only georeference an LGR row (which carries Pi time) while the two
    clocks agree, which this measures: see :func:`fill_gps_gaps_with_horel`.
    """
    if not len(gps) or "Pi_Time" not in gps.columns:
        return pd.Series(dtype=float)
    t = pd.to_datetime(
        gps["Time_UTC"] if "Time_UTC" in gps.columns else gps.index, errors="coerce"
    )
    off = (t - pd.to_datetime(gps["Pi_Time"], errors="coerce")).dt.total_seconds()
    return off.groupby(pd.DatetimeIndex(t).floor("D")).median().dropna()


def clock_checked_days(
    offsets: pd.Series,
    days,
    max_clock_offset_s: float = 3.0,
    neighbour_days: int = 3,
) -> set:
    """Days whose Pi clock can be trusted to ``max_clock_offset_s``.

    A day with its own lin-GPS offset is judged on that; a day with no lin GPS at all is
    judged on the nearest days within ``neighbour_days`` on each side (the Pi clock drifts
    over days, not minutes, so a bracketed gap is safe — and Aug-Oct 2018, when the clock was
    days off, fails this test).
    """
    ok = set()
    for d in pd.DatetimeIndex(days):
        if d in offsets.index:
            if abs(offsets[d]) <= max_clock_offset_s:
                ok.add(d)
            continue
        near = offsets[
            (offsets.index >= d - pd.Timedelta(days=neighbour_days))
            & (offsets.index <= d + pd.Timedelta(days=neighbour_days))
        ]
        if len(near) and near.abs().max() <= max_clock_offset_s:
            ok.add(d)
    return ok


def fill_gps_gaps_with_horel(
    data: pd.DataFrame,
    site: str,
    lin_gps: pd.DataFrame,
    max_clock_offset_s: float = 3.0,
    tolerance: str = "15s",
) -> pd.DataFrame:
    """Georeference the rows the lin GPS could not, using the horel logger's 5-s fixes.

    136 days have LGR data but no lin GPS file at all (it was never archived) and many more have
    partial outages — ~754 on-track hours, most of it 2019-2023. The horel CR1000 logger recorded
    position throughout, in true UTC, so it can stand in wherever the Pi clock is trustworthy
    (:func:`clock_checked_days`). Adds ``gps_source`` (``"lin"`` / ``"horel"``) so the filled rows
    can be dropped or compared at load time.
    """
    if "gps_source" not in data.columns:
        data = data.copy()
        data["gps_source"] = np.where(data["Latitude_deg"].notna(), "lin", None)
    missing = data["Latitude_deg"].isna()
    if not missing.any():
        return data
    t = pd.to_datetime(data.loc[missing, "Time_UTC"])
    horel = read_horel_cr1000(
        (t.min() - pd.Timedelta("1h"), t.max() + pd.Timedelta("1h")), site
    )
    if not len(horel) or "Latitude_deg" not in horel.columns:
        return data

    ok_days = clock_checked_days(
        lin_clock_offset_by_day(lin_gps),
        pd.DatetimeIndex(t).floor("D").unique(),
        max_clock_offset_s,
    )
    if not ok_days:
        return data
    day = pd.DatetimeIndex(pd.to_datetime(data["Time_UTC"])).floor("D")
    use = missing & day.isin(list(ok_days))
    if not use.any():
        return data

    cols = [
        c
        for c in (
            "Latitude_deg",
            "Longitude_deg",
            "Altitude_msl",
            "Speed_m_s",
            "Course_deg",
        )
        if c in horel.columns
    ]
    h = horel[cols].dropna(subset=["Latitude_deg"]).sort_index()
    h.index = pd.DatetimeIndex(h.index).as_unit("ns")
    h.index.name = "Time_UTC"
    left = data.loc[use, ["Time_UTC"]].sort_values("Time_UTC")
    # the two records can carry different datetime resolutions (us from parquet, s from the
    # logger h5); merge_asof requires them to match exactly
    left["Time_UTC"] = pd.DatetimeIndex(pd.to_datetime(left["Time_UTC"])).as_unit("ns")
    filled = pd.merge_asof(
        left,
        h.reset_index(),
        on="Time_UTC",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    filled.index = left.index
    got = filled["Latitude_deg"].notna()
    for c in cols:
        data.loc[filled.index[got], c] = filled.loc[got, c].to_numpy()
    data.loc[filled.index[got], "gps_source"] = "horel"
    print(f"Georeferenced {int(got.sum()):,} rows from the horel logger")
    return data


def merge_with_gps(
    site,
    org,
    obs,
    time_range=None,
    num_processes=1,
    routes=None,
    route_buffer=None,
    storage_polygon=None,
    altitude_range=ALTITUDE_RANGE_MSL,
    horel_fallback=True,
):
    """Attach GPS positions to a mobile site's concentration records.

    Reads the site's final-level GPS through uataq (UATAQ sites only), drops the
    fixes with an altitude outside ``altitude_range`` (m MSL; fixes without one are kept,
    ``None`` keeps all), the fixes farther than
    ``route_buffer`` m from ``routes`` and those inside ``storage_polygon``, then joins
    on the GPS clock (the Pi clock of the lin loggers is not trusted). For ``trx*``
    sites the defaults are the TRAX lines, 50 m and the JRRSC yard; pass ``False`` to
    skip a filter.

    Parameters
    ----------
    site, org : str
        Site name and its organization (only ``"UATAQ"`` is supported).
    obs : pd.DataFrame
        Records with a ``Time_UTC`` column.
    time_range : optional
        Passed to ``uataq.read_data``.
    num_processes : int
        Passed to ``uataq.read_data``.
    routes, storage_polygon : GeoDataFrame, str or False, optional
        See :func:`~slv.measurements.mobile.network.get_geodf`.
    route_buffer : float, optional
        Metres, in the routes' CRS.
    altitude_range : (float, float), optional
        Plausible altitudes, default :data:`ALTITUDE_RANGE_MSL`.
    horel_fallback : bool
        Georeference the rows the lin GPS never covered from the horel logger, where the Pi
        clock checks out (:func:`fill_gps_gaps_with_horel`). Every row is tagged ``gps_source``.

    Returns
    -------
    pd.DataFrame
        ``obs`` with the GPS columns (``Latitude_deg``, ``Longitude_deg``,
        ``Altitude_msl``, ...).
    """
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
            site,
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

    # Drop GPS junk by altitude. A fixed range, not per-chunk quantiles: those dropped 2 %
    # of every chunk (the high end of a line along with it) and every fix when the altitude
    # was constant or missing.
    if altitude_range is not None:
        alt = pd.to_numeric(gps.Altitude_msl, errors="coerce")
        gps = gps[alt.isna() | alt.between(*altitude_range)]

    # Filter to locations within buffer of routes
    routes = get_geodf(routes)
    if routes is not None and route_buffer is not None:
        print("Filtering GPS points near routes...")
        gps = filter_near_routes(gps, routes, route_buffer)

    # Remove gps points within storage polygon
    storage_polygon = get_geodf(storage_polygon)
    if storage_polygon is not None:
        print("Removing GPS points within storage polygon...")
        storage_polygon = storage_polygon.to_crs(gps.crs)  # a UTM yard matched nothing
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
    data["gps_source"] = "lin"

    # The merge is an inner join, so rows the lin GPS never covered are gone. Offer them to the
    # horel logger, which recorded position through those outages (see fill_gps_gaps_with_horel).
    if horel_fallback and site.startswith("trx") and on == "Pi_Time":
        matched = (
            set(pd.to_datetime(data["Pi_Time"]).to_numpy())
            if "Pi_Time" in data
            else set()
        )
        pi = pd.DatetimeIndex(obs.index).floor("s")
        gap = obs.loc[~pi.isin(matched)]
        if len(gap):
            gap = gap.reset_index().rename(columns={"Pi_Time": "Time_UTC"})
            for c in (
                "Latitude_deg",
                "Longitude_deg",
                "Altitude_msl",
                "Speed_m_s",
                "Course_deg",
            ):
                if c not in gap.columns:
                    gap[c] = np.nan
            gap = fill_gps_gaps_with_horel(gap, site, gps)
            gap = gap[gap["Latitude_deg"].notna()]
            if len(gap):
                data = pd.concat([data, gap], ignore_index=True).sort_values("Time_UTC")

    if "Pi_Time" in data.columns:
        data = data.drop(columns=["Pi_Time"])

    return data
