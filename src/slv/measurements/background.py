"""
Background module for the SLV
"""

import datetime as dt
import os
from functools import cached_property
from pathlib import Path

import pandas as pd
import uataq
import xarray as xr
from lair import noaa
from lair.background import rolling_baseline, thoning

from slv import get_data_dir
from slv.domain import SLV_LAT, SLV_LON, UT_BBOX


class CarbonTrackerCH4(noaa.CarbonTrackerCH4):
    """
    NOAA GML CarbonTracker CH4 subclass specifically for the SLV
    """

    def get_Utah_molefractions(self) -> xr.Dataset:
        """CarbonTracker-CH4 mole fractions over Utah (``UT_BBOX``)."""
        return self.molefractions.sel(
            longitude=slice(UT_BBOX[0], UT_BBOX[2]),
            latitude=slice(UT_BBOX[1], UT_BBOX[3]),
        )

    def get_SLV_molefractions(self, calc_pressure=False) -> xr.Dataset:
        """CarbonTracker-CH4 mole fractions in the grid column over the SLV; with
        ``calc_pressure``, adds the pressure of each level."""
        mf = self.molefractions.sel(longitude=SLV_LON, latitude=SLV_LAT)

        if calc_pressure:
            mf = noaa.CarbonTrackerCH4.calc_molefractions_pressure(mf)

        return mf


def user_gml_dir() -> Path:
    """``$SLV_USER_DATA_DIR/gml``: where slv keeps the NOAA GML files it downloads."""
    return get_data_dir("SLV_USER_DATA_DIR") / "gml"


def lair_gml_dir() -> Path | None:
    """lair's NOAA GML directory (the shared group copy): ``$LAIR_GML_DIR``, or the
    built-in ``lair.noaa.GML_DIR`` of lair releases that still have one."""
    d = os.environ.get("LAIR_GML_DIR") or getattr(noaa, "GML_DIR", None)
    return Path(d) if d else None


class GMLDiscrete(noaa.GMLData):
    """
    NOAA GML discrete sample data with QC filtering and Thoning curve support.

    The file is read from :func:`user_gml_dir` if it is there, else from lair's group copy
    (:func:`lair_gml_dir`); if neither has it (or ``refresh``) it is downloaded over FTP
    into :func:`user_gml_dir`, never into the shared group directory. Compute nodes have
    no outbound network, so fetch a new file once on a login node
    (``GMLDiscrete("ch4", "mbo", sample_type="pfp", refresh=True)``). ``gml_dir`` pins
    one directory instead.
    """

    def __init__(
        self,
        specie: str,
        site: str,
        value_col: str = "value",
        include_preliminary: bool = True,
        gml_dir: str | Path | None = None,
        refresh: bool = False,
        **kwargs,
    ):
        if gml_dir is None:
            gml_dir = self._find(specie, site, refresh, **kwargs)
        super().__init__(specie=specie, site=site, gml_dir=gml_dir, **kwargs)

        if refresh or not self.filepath.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            try:
                self.download()
            except Exception as e:  # ftplib / socket errors
                raise RuntimeError(
                    f"Could not download {self.filename} from NOAA GML ({e}). Compute "
                    "nodes have no outbound network: fetch it once on a login node "
                    f"with GMLDiscrete({specie!r}, {site!r}, ..., refresh=True)."
                ) from e

        # Drop bad data
        flags = "..P" if include_preliminary else None
        self._raw = noaa.GMLData.apply_qaqc(self.data, flags=flags)

        # Extract value column
        self.data = self._raw.rename(columns={value_col: specie.upper()})[
            specie.upper()
        ]

    @staticmethod
    def _find(specie, site, refresh, **kwargs) -> Path:
        """The directory to read from: the user cache, else the group copy, else the user
        cache (to download into)."""
        try:
            user = user_gml_dir()
        except OSError:
            user = None
        dirs = [d for d in (user, lair_gml_dir()) if d is not None]
        if not refresh:
            for d in dirs:
                probe = noaa.GMLData(specie=specie, site=site, gml_dir=d, **kwargs)
                if probe.filepath.exists():
                    return d
        if user is None:
            raise OSError(
                f"No NOAA GML file for {specie} at {site} in {lair_gml_dir()}; set "
                "SLV_USER_DATA_DIR so it can be downloaded there."
            )
        return user

    @cached_property
    def latitude(self) -> float:
        """Latitude of the sampling site."""
        return self._raw.latitude.values[0]

    @cached_property
    def longitude(self) -> float:
        """Longitude of the sampling site."""
        return self._raw.longitude.values[0]

    def thoning_curve(
        self, smooth_time: list[dt.datetime] | None = None, **kwargs
    ) -> pd.Series:
        """
        Thoning curve fit to the discrete sample data.

        Parameters
        ----------
        smooth_time : list[dt.datetime] | None
            Times to evaluate the smooth curve at. If None, use the data times.
        **kwargs : dict
            Additional arguments to pass to the Thoning filter.

        Returns
        -------
        pd.Series
            Smoothed curve.
        """
        return thoning(self.data, smooth_time=smooth_time, **kwargs)


class UTAFlask(GMLDiscrete):
    """
    NOAA GML Flask Data for UTA site
    """

    def __init__(self, **kwargs):
        super().__init__(specie="ch4", site="uta", **kwargs)


class UATAQCH4:
    """
    UATAQ Background Data

    Index by site (``"hdp"``) for hourly CH4, or by ``"<site>_base"`` for its
    rolling baseline.
    """

    def __init__(self):
        self._data = {}

    def __getitem__(self, key) -> pd.Series:
        if key not in self._data:
            self._data[key] = self._get_data(key)
        return self._data[key]

    def _get_data(self, key: str) -> pd.Series:
        # Parse key
        site, _, method = key.partition("_")
        if method not in ("", "base"):
            raise ValueError(f"Unknown method {method!r} in key {key!r}; use 'base'.")

        # Get data
        if site in self._data:
            data = self._data[site]
        else:
            data = uataq.get_obs(site, "CH4")["CH4d_ppm_cal"].dropna()

        data = data.rename("CH4")

        # Resample to hourly
        data = data.resample("1h").mean()

        # Apply method
        if method == "base":
            data = rolling_baseline(data)

        return data
