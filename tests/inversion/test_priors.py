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


class TestGetSlvPrior:
    @pytest.fixture
    def calls(self, monkeypatch):
        seen = {}
        for name in ("load_epa_prior", "load_edgar_prior", "build_constant_prior"):
            monkeypatch.setattr(
                priors, name, lambda _n=name, **kw: seen.setdefault(_n, kw) and _n
            )
        return seen

    def test_epa_in_jacobian_units(self, calls):
        assert priors.get_slv_prior("EPA", None, [], express=True) == "load_epa_prior"
        assert calls["load_epa_prior"]["units"] == "umol/m2/s"
        assert calls["load_epa_prior"]["express"] is True

    def test_edgar_drops_the_epa_only_kwarg(self, calls):
        priors.get_slv_prior("edgar", None, [], express=True)
        assert "express" not in calls["load_edgar_prior"]

    def test_constant(self, calls):
        priors.get_slv_prior("constant", None, [], value=0.5)
        assert calls["build_constant_prior"]["value"] == 0.5

    def test_unknown_prior_raises(self):
        with pytest.raises(ValueError, match="Unsupported prior"):
            priors.get_slv_prior("odiac", None, [])


def test_constant_prior_fills_every_cell_and_time():
    grid = xr.DataArray(
        np.zeros((2, 3)), coords={"lat": [40.5, 40.6], "lon": [-112.0, -111.9, -111.8]}
    )
    times = pd.date_range("2020-01-01", periods=2, freq="MS")
    prior = priors.build_constant_prior(grid, times, value=0.25, units="umol/m2/s")
    assert len(prior) == 2 * 2 * 3 and (prior == 0.25).all()
    assert prior.name == "flux" and prior.index.names == ["time", "lon", "lat"]


def _sectors(times, **values):
    ds = xr.Dataset(
        {
            name: (("time", "lat", "lon"), np.asarray(v, float).reshape(-1, 1, 1))
            for name, v in values.items()
        },
        coords={"time": pd.DatetimeIndex(times), "lat": [40.5], "lon": [-112.0]},
    )
    return ds


def test_epa_monthly_uses_the_monthly_inventory_alone(monkeypatch, identity_regrid):
    # lair >= v2026.12.6: EPAv2(scale_by_month=True) keeps every sector, holding the
    # annual-only ones (landfill) at their annual rate in every month
    months = pd.date_range("2016-01-01", periods=12, freq="MS")
    monthly_ds = _sectors(months, landfill=[10.0] * 12, gas=np.arange(12.0))

    def fake_epa(scale_by_month=False, **kwargs):
        assert scale_by_month, "the annual inventory is no longer needed"
        return _FakeInventory(monthly_ds)

    def fake_sum(ds):
        total = ds.to_array("sector").sum("sector")
        total.attrs["units"] = "umol/m2/s"
        return total

    monkeypatch.setattr(priors.inventories, "EPAv2", fake_epa)
    monkeypatch.setattr(priors.inventories, "sum_sectors", fake_sum)
    prior, regridder = priors.load_epa_prior(
        out_grid=None, flux_times=months, express=False, return_regridder=True
    )
    assert monthly_values(prior) == (10.0 + np.arange(12.0)).tolist()
    assert callable(regridder)


@pytest.fixture
def epa_v2_dir(tmp_path, monkeypatch):
    """A tiny EPA v2 archive (as in lair's tests): one annual file per year with
    ``emi_ch4_<code>_<name>`` variables, and monthly scale factors for only the first
    two sectors. Points ``$LAIR_INVENTORY_DIR`` at it."""
    lat, lon = np.array([40.05, 40.15]), np.array([-111.95, -111.85])
    d = tmp_path / "EPA" / "v2"
    (d / "monthly_scale_factors").mkdir(parents=True)
    names = ["1A_Combustion_Stationary", "3B_Manure_Management"]
    names += ["3A_Enteric_Fermentation", "5A1_Landfills_MSW"]
    for year in [2017, 2018]:
        ds = xr.Dataset(
            {
                f"emi_ch4_{n}": (("time", "lat", "lon"), np.full((1, 2, 2), i + 1.0))
                for i, n in enumerate(names)
            },
            coords={"time": [pd.Timestamp(f"{year}-01-01")], "lat": lat, "lon": lon},
        )
        ds["grid_cell_area"] = (("time", "lat", "lon"), np.ones((1, 2, 2)))
        ds.to_netcdf(d / f"Gridded_GHGI_Methane_v2_{year}.nc")
        months = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        sf = xr.Dataset(
            {
                f"monthly_scale_factor_{n}": (
                    ("time", "lat", "lon"),
                    np.ones((12, 2, 2)) * (months.month.values[:, None, None] / 6.5),
                )
                for n in names[:2]
            },
            coords={"time": months, "lat": lat, "lon": lon},
        )
        sf.to_netcdf(
            d
            / "monthly_scale_factors"
            / f"Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_{year}.nc"
        )
    monkeypatch.setenv("LAIR_INVENTORY_DIR", str(tmp_path))
    return tmp_path


def test_epa_monthly_keeps_annual_only_sectors(epa_v2_dir, monkeypatch):
    """On a real (tiny) lair EPAv2 archive, every sector reaches the prior."""
    xe = types.ModuleType("xesmf")
    xe.Regridder = lambda src, dst, method: lambda da: da.copy()
    monkeypatch.setitem(sys.modules, "xesmf", xe)

    months = pd.date_range("2018-01-01", periods=12, freq="MS")
    prior = priors.load_epa_prior(out_grid=None, flux_times=months)
    # sectors 1 + 2 scaled by month/6.5, the annual-only 3 + 4 at their annual rate
    expected = 3.0 * months.month.values / 6.5 + 7.0
    np.testing.assert_allclose(monthly_values(prior), expected)


def test_coarser_flux_freq_averages_the_inventory(monkeypatch, identity_regrid):
    months = pd.date_range("2016-01-01", periods=12, freq="MS")
    inv = inventory(months, values=np.arange(1.0, 13.0))
    monkeypatch.setattr(
        priors.inventories, "EPAv2", lambda **kwargs: _FakeInventory(inv)
    )
    quarters = pd.date_range("2016-01-01", periods=4, freq="QS")
    prior = priors.load_epa_prior(
        out_grid=None, flux_times=quarters, flux_freq="QS", express=True
    )
    assert monthly_values(prior) == [2.0, 5.0, 8.0, 11.0]
