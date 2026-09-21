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
