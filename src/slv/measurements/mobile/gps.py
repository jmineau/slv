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

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import uataq

from slv import get_data_dir
from slv.measurements.mobile.network import (
    filter_near_routes,
    get_geodf,
    load_trax_lines,
    storage_locations,
)

HOREL_CR1000_DIRS = (
    Path("/uufs/chpc.utah.edu/common/home/horel-group/uutrax/cr1000"),  # Nov 2018 on
    Path(
        "/uufs/chpc.utah.edu/common/home/horel-group/uutrax_pilot/cr1000"
    ),  # Nov 2014 - Nov 2018
)  # TODO move into uataq once its horel GPS reader keeps NSAT/RSTS


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
