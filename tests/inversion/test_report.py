"""Tests for the reporting mixin: Desroziers diagnostics, domain totals, summary."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fips.problems.flux import FluxInversionPipeline

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import SLVMethaneInversion


def pipeline(**kwargs):
    """A pipeline without fips' __init__ (no data access)."""
    obj = object.__new__(SLVMethaneInversion)
    obj.config = InversionConfig(**kwargs)
    return obj


class _Variances:
    def __init__(self, series):
        self.variances = SimpleNamespace(to_series=lambda: series.copy())


OBS_INDEX = pd.MultiIndex.from_arrays(
    [
        ["wbb", "wbb", "hdp", "hdp"],
        pd.to_datetime(["2020-01-05", "2020-02-05", "2020-01-05", "2020-02-05"]),
    ],
    names=["obs_location", "obs_time"],
)


@pytest.fixture
def desroziers_pipeline():
    p = pipeline()
    p.problem = SimpleNamespace(
        desroziers=_Variances(pd.Series([2.0, 4.0, 1.0, 3.0], index=OBS_INDEX)),
        modeldata_mismatch=_Variances(pd.Series([1.0, 1.0, 2.0, 2.0], index=OBS_INDEX)),
    )
    return p


def test_desroziers_by_site(desroziers_pipeline):
    out = desroziers_pipeline.desroziers_diagnostic()
    assert list(out.index) == ["hdp", "wbb"]
    assert out.loc["wbb"].tolist() == [3.0, 1.0, 3.0]  # diagnosed, specified, ratio
    assert out.loc["hdp"].tolist() == [2.0, 2.0, 1.0]


def test_desroziers_per_obs(desroziers_pipeline):
    out = desroziers_pipeline.desroziers_diagnostic(groupby=None)
    assert out.index.equals(OBS_INDEX)
    np.testing.assert_allclose(out["ratio"], [2.0, 4.0, 0.5, 1.5])


def test_desroziers_by_month_only(desroziers_pipeline):
    out = desroziers_pipeline.desroziers_diagnostic(groupby=None, freq="MS")
    assert list(out.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    np.testing.assert_allclose(out["diagnosed"], [1.5, 3.5])
    np.testing.assert_allclose(out["ratio"], [1.0, 3.5 / 1.5])


def test_desroziers_by_site_and_month(desroziers_pipeline):
    out = desroziers_pipeline.desroziers_diagnostic(freq="MS")
    assert out.index.names == ["obs_location", "obs_time"]
    assert out.loc[("wbb", pd.Timestamp("2020-02-01")), "ratio"] == 4.0


def test_total_flux_is_mass_per_interval():
    pytest.importorskip("xesmf")  # lair's cell areas
    p = pipeline(tstart="2020-01-01", tend="2020-04-01", flux_freq="MS")
    times = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"])
    index = pd.MultiIndex.from_product(
        [times, [40.525, 40.575], [-111.975, -111.925]], names=["time", "lat", "lon"]
    )
    flux = pd.Series(1.0, index=index, name="flux")  # 1 umol/m2/s everywhere

    total = p.calculate_total_flux(flux, units="Gg/m2/s")

    # 0.1 x 0.1 deg on a sphere, 16.04 g/mol, each month's seconds
    r = 6371008.8
    area = (
        r**2 * np.radians(0.1) * (np.sin(np.radians(40.6)) - np.sin(np.radians(40.5)))
    )
    seconds = np.array([31, 29, 31]) * 86400
    expected = area * 1e-6 * 16.04 * seconds / 1e9
    assert list(total.index) == list(times)
    np.testing.assert_allclose(total.to_numpy(), expected, rtol=2e-3)


def _flux(values, times):
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(times), [40.5], [-112.0, -111.9]],
        names=["time", "lat", "lon"],
    )
    return pd.Series(np.repeat(values, 2), index=index, dtype=float)


@pytest.fixture
def summarize_pipeline(monkeypatch):
    """fips' own summary off; totals = sum over cells (no xesmf needed)."""
    monkeypatch.setattr(FluxInversionPipeline, "summarize", lambda self: None)
    times = ["2020-01-01", "2021-01-01", "2022-01-01"]
    p = pipeline(tstart="2020-01-01", tend="2023-01-01", flux_freq="YS")
    p.problem = SimpleNamespace(
        prior_fluxes=_flux([1.0, 1.0, 1.0], times).rename("flux"),
        posterior_fluxes=_flux([1.0, 2.0, 3.0], times).rename("flux"),
    )
    p.calculate_total_flux = lambda fluxes, units=None: fluxes.groupby("time").sum()
    return p


def test_summarize_prints_totals_and_trend(summarize_pipeline, capsys):
    summarize_pipeline.summarize()
    out = capsys.readouterr().out
    assert "DOMAIN TOTAL EMISSIONS [Gg per YS interval]" in out
    assert "Mean Prior:     2.00" in out  # two cells of 1
    assert "Mean Posterior: 4.00" in out
    assert "Mean Change:    +100.0%" in out
    assert "Trend:          +2.00 [Gg per YS interval]/yr" in out


def test_summarize_totals_use_the_reconstructed_posterior(summarize_pipeline, capsys):
    p = summarize_pipeline
    # coverage filter state: cell (40.5, -112.0) removed and held at a prior of 5
    full = _flux([5.0, 5.0, 5.0], ["2020-01-01", "2021-01-01", "2022-01-01"])
    p._full_prior = {"flux": full}
    p._removed_cells = {(40.5, -112.0)}
    kept = p.problem.posterior_fluxes
    p.problem.posterior_fluxes = kept[kept.index.get_level_values("lon") == -111.9]

    p.summarize()

    out = capsys.readouterr().out
    # prior: 5 + 5 per year; posterior: 5 (held) + 1, 2, 3
    assert "Mean Prior:     10.00" in out
    assert "Mean Posterior: 7.00" in out
