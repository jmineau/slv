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
    met["Time_UTC"] = pd.to_datetime(met_time, errors="coerce")
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


def calculate_enhancements(data, window=1):
    data["CH4_base"] = rolling_baseline(data["CH4 (ppb)"], window=window)
    data["C2H6_base"] = rolling_baseline(data["C2H6 (ppb)"], window=window)

    data["CH4_ex"] = data["CH4 (ppb)"] - data["CH4_base"]
    data["C2H6_ex"] = data["C2H6 (ppb)"] - data["C2H6_base"]
    return data


def enhanced_R_and_ratio(data, window="30s"):
    data.rename(columns={"R": "R_aeris", "C2/C1": "C2C1_aeris"}, inplace=True)

    data["R"] = data.CH4_ex.rolling(window).corr(data.C2H6_ex)
    data["C2/C1"] = data["C2H6_ex"] / data["CH4_ex"]

    return data


def plot_windbarbs(data, ws, wd, ax=None, x="longitude", y="latitude"):
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
