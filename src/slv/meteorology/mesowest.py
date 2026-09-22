"""MesoWest surface observations for the SLV sites.

Reads the hourly station archive staged under ``$SLV_USER_DATA_DIR/mesowest``
(a link to the meteorology workspace's ``john_data`` pull). Each station is one
parquet file of hourly values with a ``Time`` column in UTC; the columns vary by
station, but wind speed, wind direction and air temperature are near-universal.

The archive is a fixed pull, not a live feed -- see :func:`station_hourly` for
the span it covers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slv import get_data_dir

#: Columns renamed on read, so callers do not carry MesoWest's ``_set_1`` suffixes.
_RENAME = {
    "wind_speed_set_1": "wind_speed",
    "wind_direction_set_1": "wind_direction",
    "air_temp_set_1": "air_temp",
    "relative_humidity_set_1": "relative_humidity",
    "pressure_set_1": "pressure",
    "solar_radiation_set_1": "solar_radiation",
    "ozone_concentration_set_1": "ozone",
    "PM_25_concentration_set_1": "pm25",
}


def mesowest_dir():
    """The staged MesoWest directory, ``$SLV_USER_DATA_DIR/mesowest``."""
    return get_data_dir("SLV_USER_DATA_DIR") / "mesowest"


def load_station_metadata() -> pd.DataFrame:
    """Station code, name, latitude, longitude and elevation for the archived stations.

    Indexed by ``station_code``. Only stations with an hourly file are returned, so
    the table can be used directly to pick a station to read.
    """
    meta = pd.read_csv(mesowest_dir() / "stations_metadata.csv")
    meta = meta.rename(
        columns={
            "LAT": "latitude",
            "LON": "longitude",
            "ELEVATION.ft": "elevation_ft",
        }
    )
    available = {
        p.name.replace("_hourly.parquet", "")
        for p in (mesowest_dir() / "hourly").glob("*_hourly.parquet")
    }
    meta = meta[meta["station_code"].isin(available)]
    return meta.set_index("station_code").sort_index()


def nearest_station(
    latitude: float, longitude: float, metadata: pd.DataFrame | None = None
):
    """The archived station closest to a point, as ``(station_code, distance_km)``.

    Distance is a local flat-earth approximation, which is well under a percent
    of error at the scale of the Salt Lake Valley.
    """
    meta = load_station_metadata() if metadata is None else metadata
    dx = (meta["longitude"] - longitude) * 111.32 * np.cos(np.radians(latitude))
    dy = (meta["latitude"] - latitude) * 111.32
    distance = np.hypot(dx, dy)
    code = distance.idxmin()
    return code, float(distance[code])


def station_hourly(
    station_code: str,
    columns: list[str] | None = None,
    time_range: tuple | None = None,
) -> pd.DataFrame:
    """Hourly observations for one station, indexed by ``Time_UTC``.

    Columns are renamed out of MesoWest's ``_set_1`` convention where a standard
    name exists (see ``_RENAME``); anything else keeps its archive name. ``columns``
    selects a subset by the *renamed* name, dropping any the station does not carry.
    Wind direction of exactly 0 with zero speed is left alone -- calms are a real
    state, and callers that bin by direction should drop them explicitly.
    """
    path = mesowest_dir() / "hourly" / f"{station_code}_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No hourly archive for station {station_code!r} at {path}"
        )

    df = pd.read_parquet(path).rename(columns=_RENAME)
    df["Time_UTC"] = pd.to_datetime(df.pop("Time"), utc=True).dt.tz_localize(None)
    df = df.set_index("Time_UTC").sort_index()

    if time_range is not None:
        df = df.loc[slice(*time_range)]
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


def wind_sector(direction: pd.Series, n_sectors: int = 16) -> pd.Series:
    """Compass sector label (N, NNE, ...) for a wind direction in degrees.

    ``n_sectors`` must be 4, 8 or 16. Directions are binned centred on each
    label, so N covers 348.75-11.25 degrees for 16 sectors.
    """
    labels = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
    }[n_sectors]
    width = 360 / n_sectors
    index = (
        np.floor(((direction % 360) + width / 2) / width).astype("Int64") % n_sectors
    )
    return pd.Series(
        pd.Categorical(
            [labels[i] if pd.notna(i) else None for i in index], categories=labels
        ),
        index=direction.index,
    )
