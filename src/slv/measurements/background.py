"""
Background module for the SLV
"""

import datetime as dt
from functools import cached_property

import pandas as pd
import uataq
import xarray as xr
from lair.air import noaa
from lair.air.background import rolling_baseline, thoning

from slv.domain import SLV_LAT, SLV_LON, UT_BBOX


class CarbonTrackerCH4(noaa.CarbonTrackerCH4):
    """
    NOAA GML CarbonTracker CH4 2023 Subclass specifically for the SLV
    """

    def get_Utah_molefractions(self) -> xr.Dataset:
        return self.molefractions.sel(
            longitude=slice(UT_BBOX[0], UT_BBOX[2]),
            latitude=slice(UT_BBOX[1], UT_BBOX[3]),
        )

    def get_SLV_molefractions(self, calc_pressure=False) -> xr.Dataset:
        mf = self.molefractions.sel(longitude=SLV_LON, latitude=SLV_LAT)

        if calc_pressure:
            mf = noaa.CarbonTrackerCH4.calc_molefractions_pressure(mf)

        return mf


class UTAFlask(noaa.Flask):
    """
    NOAA GML Flask Data for UTA site
    """

    def __init__(self, include_preliminary: bool = True, **kwargs):
        super().__init__(specie="ch4", site="uta", **kwargs)

        # Drop bad data
        flags = "..P" if include_preliminary else None
        self.data = noaa.Flask.apply_qaqc(self.data, flags=flags)

        # Reduce to CH4 column
        self.data = self.data.rename(columns={"value": "CH4"})["CH4"]

    @cached_property
    def latitude(self) -> float:
        return self.data.latitude.values[0]

    @cached_property
    def longitude(self) -> float:
        return self.data.longitude.values[0]

    def thoning_curve(
        self, smooth_time: list[dt.datetime] | None = None, **kwargs
    ) -> pd.Series:
        """
        Thoning curve for the UTA flask data

        Parameters
        ----------
        smooth_time : list[dt.datetime] | None
            Time range to smooth over. If None, use the entire time range of the data.
        **kwargs : dict
            Additional arguments to pass to the thonning filter.

        Returns
        -------
        pd.Series
            Thoning curve for the UTA flask data.
        """
        return thoning(self.data, smooth_time=smooth_time, **kwargs)


class UATAQCH4:
    """
    UATAQ Background Data
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, key) -> pd.DataFrame:
        if key not in self._data:
            self._data[key] = self._get_data(key)
        return self._data[key]

    def _get_data(self, key: str) -> pd.Series:
        # Parse key
        if "_" in key:
            site, method = key.split("_")
        else:
            site = key
            method = None

        # Get data
        if site in self._data:
            data = self._data[site]
        else:
            data = uataq.get_obs(site, "CH4")["CH4d_ppm_cal"].dropna()

        data = data.rename("CH4")

        # Resample to hourly
        data = data.resample("1h").mean()

        # Apply method
        if method:
            if method == "base":
                data = rolling_baseline(data)

        return data
