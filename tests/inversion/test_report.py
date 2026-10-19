"""Tests for the reports: Desroziers diagnostics, domain totals, the printed summary."""

import fips
import numpy as np
import pandas as pd
import pytest
from fips import Independent, Operator, Problem, Vector

from slv.inversion import report
from slv.inversion.config import InversionConfig

OBS_INDEX = pd.MultiIndex.from_arrays(
    [
        ["wbb", "wbb", "hdp", "hdp"],
        pd.to_datetime(["2020-01-05", "2020-02-05", "2020-01-05", "2020-02-05"]),
    ],
    names=["obs_location", "obs_time"],
)


def flux(values, times, lons=(-112.0, -111.9)):
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(times), [40.5], list(lons)], names=["time", "lat", "lon"]
    )
    return pd.Series(np.repeat(values, len(lons)), index=index, dtype=float)


H = np.random.default_rng(0).random((4, 6))


def solved(prior, obs_std=(1.0, 1.0, 2.0, 2.0), gamma=1.0):
    """A posterior of four obs that see every flux element."""
    x0 = Vector.from_series(prior, block="flux")
    obs = Vector.from_series(
        pd.Series([2.0, 3.0, 1.5, 2.5], index=OBS_INDEX), block="concentration"
    )
    problem = Problem(
        prior=x0,
        prior_cov=Independent().with_std(0.5),
        obs=obs,
        obs_cov=Independent().with_std(np.asarray(obs_std)),
        forward=Operator(H[:, : len(x0)], obs.space, x0.space),
        offset=1.0,
    )
    return fips.solve(problem, gamma=gamma)


@pytest.fixture
def post():
    return solved(flux([0.1, 0.2], ["2020-01-01", "2020-02-01"]))


def test_desroziers_per_obs_is_the_product_of_the_residuals(post):
    out = report.desroziers_diagnostic(post, groupby=None)
    problem = post.problem
    before = problem.obs.values - problem.predict(problem.prior).values
    after = problem.obs.values - post.modeled_obs.values
    np.testing.assert_allclose(out["diagnosed"], after * before)
    np.testing.assert_allclose(out["specified"], [1.0, 1.0, 4.0, 4.0])
    np.testing.assert_allclose(out["ratio"], out["diagnosed"] / out["specified"])


def test_desroziers_by_site(post):
    per_obs = report.desroziers_diagnostic(post, groupby=None)
    out = report.desroziers_diagnostic(post)
    assert list(out.index) == ["hdp", "wbb"]
    wbb = per_obs.xs("wbb", level="obs_location")
    np.testing.assert_allclose(out.loc["wbb", "diagnosed"], wbb["diagnosed"].mean())
    assert out.loc["wbb", "specified"] == 1.0


def test_desroziers_by_month_only(post):
    out = report.desroziers_diagnostic(post, groupby=None, freq="MS")
    assert list(out.index) == list(pd.to_datetime(["2020-01-01", "2020-02-01"]))
    np.testing.assert_allclose(out["specified"], [2.5, 2.5])


def test_desroziers_by_site_and_month(post):
    out = report.desroziers_diagnostic(post, freq="MS")
    assert out.index.names == ["obs_location", "obs_time"]
    assert out.loc[("hdp", pd.Timestamp("2020-02-01")), "specified"] == 4.0


def test_desroziers_specified_is_the_error_the_solve_used():
    # with gamma, the obs error the solve used is the mismatch divided by gamma
    post = solved(flux([0.1, 0.2], ["2020-01-01", "2020-02-01"]), gamma=4.0)
    out = report.desroziers_diagnostic(post, groupby=None)
    np.testing.assert_allclose(out["specified"], [0.25, 0.25, 1.0, 1.0])


def test_total_flux_is_mass_per_interval():
    pytest.importorskip("xesmf")  # lair's cell areas
    config = InversionConfig(tstart="2020-01-01", tend="2020-04-01", flux_freq="MS")
    times = pd.DatetimeIndex(["2020-01-01", "2020-02-01", "2020-03-01"])
    index = pd.MultiIndex.from_product(
        [times, [40.525, 40.575], [-111.975, -111.925]], names=["time", "lat", "lon"]
    )
    fluxes = pd.Series(1.0, index=index, name="flux")  # 1 umol/m2/s everywhere

    total = report.calculate_total_flux(config, fluxes, units="Gg/m2/s")

    # 0.1 x 0.1 deg on a sphere, 16.04 g/mol, each month's seconds
    r = 6371008.8
    area = (
        r**2 * np.radians(0.1) * (np.sin(np.radians(40.6)) - np.sin(np.radians(40.5)))
    )
    seconds = np.array([31, 29, 31]) * 86400
    expected = area * 1e-6 * 16.04 * seconds / 1e9
    assert list(total.index) == list(times)
    np.testing.assert_allclose(total.to_numpy(), expected, rtol=2e-3)


TIMES = ["2020-01-01", "2021-01-01", "2022-01-01"]


@pytest.fixture
def summing(monkeypatch):
    """Totals as the sum over cells (no xesmf needed)."""
    monkeypatch.setattr(
        report,
        "calculate_total_flux",
        lambda config, fluxes, units=None: fluxes.groupby("time").sum(),
    )
    return InversionConfig(tstart="2020-01-01", tend="2023-01-01", flux_freq="YS")


def test_summarize_prints_the_fit_totals_and_trend(summing, capsys):
    post = solved(flux([1.0, 1.0, 1.0], TIMES))
    report.summarize(summing, post)
    out = capsys.readouterr().out
    assert "FLUX INVERSION SUMMARY REPORT" in out
    assert "State Vector Size (n):       6" in out
    assert "Observation Vector Size (m): 4" in out
    assert f"Degrees of Freedom (DOFS):   {post.dofs:.2f}" in out
    assert f"Reduced Chi-Square:          {post.reduced_chi2:.3f}" in out
    assert "DOMAIN TOTAL EMISSIONS [Gg per YS interval]" in out
    assert "Mean Prior:     2.00" in out  # two cells of 1
    estimate = post.estimate.to_series()
    mean_total = estimate.groupby(level="time").sum().mean()
    assert f"Mean Posterior: {mean_total:.2f}" in out
    assert "Trend:" in out


def test_summarize_totals_cover_the_cells_left_out(summing, capsys):
    full = solved(flux([5.0, 5.0, 5.0], TIMES)).problem
    # the coverage filter left out lon -112.0, held at its prior of 5
    keep = full.prior.space.select("flux")
    keep = fips.Space({"flux": keep[keep.get_level_values("lon") == -111.9]})
    post = fips.solve(full.restrict(keep))

    report.summarize(summing, post, full)

    out = capsys.readouterr().out
    kept = post.estimate.to_series().groupby(level="time").sum()
    assert "Mean Prior:     10.00" in out  # 5 + 5 per year
    assert f"Mean Posterior: {(kept + 5.0).mean():.2f}" in out
