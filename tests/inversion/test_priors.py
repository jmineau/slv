"""Tests for prior construction: aligning inventory periods to the flux times."""

import sys
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from slv.inversion import priors


def inventory(times, values=None):
    """A one-cell inventory labelled at period starts, like lair's inventories."""
    times = pd.DatetimeIndex(times)
    if values is None:
        values = times.year
    return xr.DataArray(
        np.asarray(values, dtype=float).reshape(-1, 1, 1),
        coords={"time": times, "lat": [40.5], "lon": [-112.0]},
        dims=("time", "lat", "lon"),
        name="flux",
        attrs={"units": "umol/m2/s"},
    )


def annual(years):
    return inventory([f"{y}-01-01" for y in years])


class TestAlignToFluxTimes:
    def test_months_take_their_own_year(self):
        # Nearest-neighbour matching gave Aug-Dec the next year's field.
        flux_times = pd.date_range("2016-01-01", "2016-12-01", freq="MS")
        out = priors.align_to_flux_times(annual([2015, 2016, 2017]), flux_times)
        assert (out.values.ravel() == 2016).all()

    def test_time_coordinate_is_flux_times(self):
        flux_times = pd.date_range("2016-01-01", periods=5, freq="MS")
        out = priors.align_to_flux_times(annual([2016]), flux_times)
        pd.testing.assert_index_equal(
            out.indexes["time"], flux_times, check_names=False
        )

    def test_before_inventory_holds_first_period(self):
        flux_times = pd.date_range("2014-06-01", "2015-02-01", freq="MS")
        out = priors.align_to_flux_times(annual([2015, 2016]), flux_times)
        assert (out.values.ravel() == 2015).all()

    def test_after_inventory_holds_last_period(self):
        # e.g. EPA holds 2020 for 2021-2023
        flux_times = pd.date_range("2021-01-01", "2023-12-01", freq="QS")
        out = priors.align_to_flux_times(annual([2019, 2020]), flux_times)
        assert (out.values.ravel() == 2020).all()

    def test_monthly_inventory_daily_fluxes(self):
        monthly = inventory(
            pd.date_range("2020-01-01", periods=3, freq="MS"), values=[1, 2, 3]
        )
        flux_times = pd.DatetimeIndex(["2020-01-01", "2020-01-20", "2020-02-28"])
        out = priors.align_to_flux_times(monthly, flux_times)
        assert out.values.ravel().tolist() == [1.0, 1.0, 2.0]


class _FakeInventory:
    """Stands in for a lair inventory: clip / convert_units are no-ops."""

    def __init__(self, data):
        self.data = data

    def clip(self, **kwargs):
        return self

    def convert_units(self, units):
        return self


@pytest.fixture
def identity_regrid(monkeypatch):
    """xesmf is conda-only: an identity regridder, and sectors that are already summed."""
    xe = types.ModuleType("xesmf")
    xe.Regridder = lambda src, dst, method: lambda da: da.copy()
    monkeypatch.setitem(sys.modules, "xesmf", xe)
    monkeypatch.setattr(priors.inventories, "sum_sectors", lambda data: data)


def monthly_values(prior):
    return prior.groupby(level="time").first().tolist()


class TestLoadersAlignByPeriod:
    flux_times = pd.date_range("2016-06-01", "2017-05-01", freq="MS")
    expected = [2016.0] * 7 + [2017.0] * 5

    def test_epa_express(self, monkeypatch, identity_regrid):
        inv = annual(range(2012, 2021))
        monkeypatch.setattr(
            priors.inventories, "EPAv2", lambda **kwargs: _FakeInventory(inv)
        )
        prior = priors.load_epa_prior(
            out_grid=None, flux_times=self.flux_times, flux_freq="MS", express=True
        )
        assert monthly_values(prior) == self.expected

    def test_edgar(self, monkeypatch, identity_regrid):
        inv = annual(range(2012, 2023))
        monkeypatch.setattr(
            priors.inventories, "EDGARv8", lambda *args, **kwargs: _FakeInventory(inv)
        )
        prior = priors.load_edgar_prior(
            out_grid=None, flux_times=self.flux_times, flux_freq="MS"
        )
        assert monthly_values(prior) == self.expected
