"""Wyoming mobile lab: Aeris CH4/C2H6 and vehicle met readers, enhancements and
C2/C1 ratios, and a wind-barb map of a drive."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from lair.air import wind_components, wind_direction
from lair.background import rolling_baseline
from lair.clock import UTC2MTN
from lair.geo import add_latlon_ticks

PC = ccrs.PlateCarree()


def read_aeris(file):
    """Read an Aeris analyzer CSV: ``CH4 (ppb)`` (converted from ppm), ``C2H6 (ppb)``,
    the analyzer's ``R`` and ``C2/C1``, indexed by ``Time_UTC``."""
    aeris = pd.read_csv(file)
    aeris["Time_UTC"] = pd.to_datetime(
        aeris["Time Stamp"], errors="coerce", format="%m/%d/%Y %H:%M:%S.%f"
    )
    aeris = aeris.dropna(subset=["Time_UTC"]).set_index("Time_UTC").sort_index()
    aeris = aeris[["CH4 (ppm)", "C2H6 (ppb)", "R", "C2/C1"]]
    aeris["CH4 (ppm)"] *= 1000
    aeris.rename(columns={"CH4 (ppm)": "CH4 (ppb)"}, inplace=True)
    return aeris


def read_met(file):
    """Read a vehicle met-station file: position (``latitude``, ``longitude``,
    ``Altitude (m)``), temperature, humidity, pressure, GPS-corrected wind and vehicle
    speed, indexed by ``Time_UTC``, plus the source ``filename``."""
    # last column is vehicle speed (not sure of units per Zak)
    cols = [
        "PC",
        "UTC hhmmss",
        "UTC Year",
        "UTC Month",
        "UTC Day",
        "Latitude (DD.ddd +N)",
        "Longitude (DDD.ddd -W)",
        "GPS Quality",
        "Altitude (m)",
        "Air Temperature (C)",
        "RH(%)",
        "Dew Point (C)",
        "Wind Direction (Deg True)",
        "Wind Direction (Deg Mag)",
        "Wind Speed (m/s)",
        "Pressure (bar)",
        "PCB1 Temperature (C)",
        "PCB2 Temperature (C)",
        "Supply Voltage (VDC)",
        "Heading(deg)",
        "GPSCorWindDirTrue (deg)",
        "GPSCorWindDirMag (deg)",
        "GPSCorWindSpeed (kts)",
        "GPSCorWindSpeed (m/s)",
        "VehicleSpeed",
    ]
    met = pd.read_csv(
        file,
        header=3,
        names=cols,
        index_col=False,
        skipinitialspace=True,
        on_bad_lines="skip",
        dtype={"UTC hhmmss": str},
    )
    met_time = dict(
        year=met["UTC Year"],
        month=met["UTC Month"],
        day=met["UTC Day"],
        hour=met["UTC hhmmss"].str.slice(0, 2),
        minute=met["UTC hhmmss"].str.slice(2, 4),
        second=met["UTC hhmmss"].str.slice(4, None),
    )
    # A garbled line survives on_bad_lines="skip" with the right field count but
    # nonsense values (a year of 3258, a second of 61). pandas assembles the
    # components before errors="coerce" can act, so out-of-range ones raise
    # instead of coercing -- one bad row otherwise loses the whole drive. Blank
    # out anything outside its calendar range first.
    limits = {
        "year": (1990, 2100),
        "month": (1, 12),
        "day": (1, 31),
        "hour": (0, 23),
        "minute": (0, 59),
        "second": (0, 60),
    }
    met_time = {k: pd.to_numeric(v, errors="coerce") for k, v in met_time.items()}
    valid = pd.Series(True, index=met.index)
    for name, (low, high) in limits.items():
        valid &= met_time[name].between(low, high)
    # Assemble only the good rows: passing NaN components through to_datetime
    # works but makes pandas warn on every cast.
    times = pd.Series(pd.NaT, index=met.index, dtype="datetime64[ns]")
    if valid.any():
        times[valid] = pd.to_datetime(
            {k: v[valid] for k, v in met_time.items()}, errors="coerce"
        )
    met["Time_UTC"] = times
    met = met.dropna(subset="Time_UTC").set_index("Time_UTC").sort_index()
    met = met[
        [
            "Latitude (DD.ddd +N)",
            "Longitude (DDD.ddd -W)",
            "GPS Quality",
            "Altitude (m)",
            "Air Temperature (C)",
            "RH(%)",
            "Dew Point (C)",
            "Pressure (bar)",
            "GPSCorWindDirTrue (deg)",
            "GPSCorWindSpeed (kts)",
            "VehicleSpeed",
        ]
    ]
    met = met.rename(
        columns={
            "Latitude (DD.ddd +N)": "latitude",
            "Longitude (DDD.ddd -W)": "longitude",
        }
    )
    met["filename"] = file
    return met


def merge_aeris_met(aeris, met):
    """
    Merge aeris and met dataframes, interpolating met data to match aeris timestamps

    Parameters
    ----------
    aeris : pd.DataFrame
        Aeris data
    met : pd.DataFrame
        Met data
    """
    data = aeris.copy(deep=True)
    data["latitude"] = np.interp(data.index, met.index, met["latitude"])
    data["longitude"] = np.interp(data.index, met.index, met["longitude"])
    data["altitude"] = np.interp(data.index, met.index, met["Altitude (m)"])
    data["wind_speed"] = np.interp(data.index, met.index, met["GPSCorWindSpeed (kts)"])
    data["vehicle_speed"] = np.interp(data.index, met.index, met["VehicleSpeed"])

    met["u_wind"], met["v_wind"] = wind_components(
        met["GPSCorWindSpeed (kts)"], met["GPSCorWindDirTrue (deg)"]
    )
    interp_u_wind = np.interp(data.index, met.index, met["u_wind"])
    interp_v_wind = np.interp(data.index, met.index, met["v_wind"])
    data["wind_direction"] = wind_direction(interp_u_wind, interp_v_wind)

    # Convert to geodataframe
    data = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326",
    )

    # Convert time to Mountain Time
    data = UTC2MTN(data, driver="pandas", localize=True)
    data.index.names = ["Time_MTN"]

    return data


def calculate_enhancements(data, window="1h"):
    """Add ``CH4_base``/``C2H6_base`` (:func:`lair.background.rolling_baseline`) and the
    enhancements over them, ``CH4_ex``/``C2H6_ex``.

    ``window`` is the baseline window: anything ``pd.Timedelta`` takes, or a number of hours
    (rolling_baseline's old integer-hours convention; a bare number would otherwise be read
    as nanoseconds and make every enhancement zero).
    """
    if isinstance(window, (int, float)):
        window = pd.Timedelta(hours=window)
    data["CH4_base"] = rolling_baseline(data["CH4 (ppb)"], window=window)
    data["C2H6_base"] = rolling_baseline(data["C2H6 (ppb)"], window=window)

    data["CH4_ex"] = data["CH4 (ppb)"] - data["CH4_base"]
    data["C2H6_ex"] = data["C2H6 (ppb)"] - data["C2H6_base"]
    return data


def enhanced_R_and_ratio(data, window="30s"):
    """Add the rolling ``window`` correlation ``R`` of the CH4 and C2H6 enhancements and
    their ratio ``C2/C1`` (after :func:`calculate_enhancements`); the analyzer's own
    columns are kept as ``R_aeris`` and ``C2C1_aeris``. Modifies ``data`` in place."""
    data.rename(columns={"R": "R_aeris", "C2/C1": "C2C1_aeris"}, inplace=True)

    data["R"] = data.CH4_ex.rolling(window).corr(data.C2H6_ex)
    data["C2/C1"] = data["C2H6_ex"] / data["CH4_ex"]

    return data


def plot_windbarbs(data, ws, wd, ax=None, x="longitude", y="latitude"):
    """Draw wind barbs from the speed column ``ws`` and direction column ``wd`` at each
    ``(x, y)``; makes a PlateCarree axes when ``ax`` is not given."""
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": PC})

    data["u"], data["v"] = wind_components(data[ws], data[wd])
    ax.barbs(data[x], data[y], data["u"], data["v"], transform=PC, length=5)

    return ax


def wyomingMap(
    data,
    param,
    ax=None,
    extent=None,
    tiler=None,
    tiler_zoom=12,
    cmap="YlOrRd",
    figsize=(16, 10),
    plot_winds=True,
    windskip=40,
    title=None,
    **kwargs,
):
    """Map a drive coloured by ``param`` (a column of the merged GeoDataFrame from
    :func:`merge_aeris_met`), with every ``windskip``-th wind barb and optional map
    tiles. ``**kwargs`` go to ``GeoDataFrame.plot``."""
    crs = tiler.crs if tiler else PC
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": crs})
    if extent:
        ax.set_extent(extent)
    if tiler:
        ax.add_image(tiler, tiler_zoom)
    if title:
        ax.set_title(title)

    data.plot(column=param, transform=PC, ax=ax, legend=True, cmap=cmap, **kwargs)

    if plot_winds:
        plot_windbarbs(data[::windskip], "wind_speed", "wind_direction", ax=ax)

    if extent:
        add_latlon_ticks(ax, extent=extent, x_rotation=35)
    return ax
