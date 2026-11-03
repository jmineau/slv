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
        obs_cov=Independent().with_std(pd.Series(obs_std, index=OBS_INDEX)),
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


# ---------------------------------------------------------------------------
# fit_metrics and the printed fit (#57)
# ---------------------------------------------------------------------------

FIT_OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * 4, pd.date_range("2020-01-05", periods=4)],
    names=["obs_location", "obs_time"],
)
FIT_CELLS = pd.MultiIndex.from_product(
    [pd.to_datetime(["2020-01-01"]), [40.5], [-112.0, -111.9]],
    names=["time", "lat", "lon"],
)
FIT_H = np.array([[1.0, 0.5], [0.2, 1.0], [0.7, 0.7], [0.3, 0.9]])
FIT_X0 = np.array([2.0, 1.0])
FIT_BACKGROUND = np.array([1.0, 3.0, 2.0, 4.0])  # varies from obs to obs


def fit_problem_solved(truth_scale):
    """
    A solved problem with a varying background and a known high or low prior.

    The observations are ``H (truth_scale x0) + background``, so with
    ``truth_scale < 1`` the prior's model is above them everywhere.
    """
    x0 = Vector.from_series(pd.Series(FIT_X0, index=FIT_CELLS), block="flux")
    z = FIT_H @ (truth_scale * FIT_X0) + FIT_BACKGROUND
    z = z + np.array([0.05, -0.1, 0.1, -0.05])  # noise, so no fit is exact
    obs = Vector.from_series(pd.Series(z, index=FIT_OBS_INDEX), block="concentration")
    background = Vector.from_series(
        pd.Series(FIT_BACKGROUND, index=FIT_OBS_INDEX), block="concentration"
    )
    problem = Problem(
        prior=x0,
        prior_cov=Independent().with_std(0.2),
        obs=obs,
        obs_cov=Independent().with_std(0.1),
        forward=Operator(FIT_H, obs.space, x0.space),
        offset=background,
    )
    return fips.solve(problem), z


class TestFitMetrics:
    def test_bias_is_model_minus_obs(self):
        # the prior is twice the truth: its model is above the observations
        post, z = fit_problem_solved(truth_scale=0.5)
        prior_model = FIT_H @ FIT_X0 + FIT_BACKGROUND
        fit = report.fit_metrics(post)
        assert (prior_model > z).all()
        assert fit.loc["prior", "conc_mean_bias"] == pytest.approx(
            np.mean(prior_model - z)
        )
        assert fit.loc["prior", "conc_mean_bias"] > 0
        post_model = post.modeled_obs.values
        assert fit.loc["posterior", "conc_mean_bias"] == pytest.approx(
            np.mean(post_model - z)
        )

    def test_bias_is_negative_when_the_model_is_low(self):
        post, _ = fit_problem_solved(truth_scale=2.0)
        assert report.fit_metrics(post).loc["prior", "conc_mean_bias"] < 0

    def test_rmse_is_the_root_mean_square_of_the_residual(self):
        post, z = fit_problem_solved(truth_scale=0.5)
        prior_model = FIT_H @ FIT_X0 + FIT_BACKGROUND
        fit = report.fit_metrics(post)
        assert fit.loc["prior", "conc_rmse"] == pytest.approx(
            np.sqrt(np.mean((prior_model - z) ** 2))
        )
        assert fit.loc["posterior", "conc_rmse"] == pytest.approx(post.rmse)

    def test_the_two_r2_are_the_hand_formulas(self):
        post, z = fit_problem_solved(truth_scale=0.5)
        enhancement = z - FIT_BACKGROUND
        models = {
            "prior": FIT_H @ FIT_X0,
            "posterior": post.modeled_obs.values - FIT_BACKGROUND,
        }
        fit = report.fit_metrics(post)
        for state, modeled in models.items():
            ss_res = np.sum((enhancement - modeled) ** 2)
            ss_tot = np.sum((enhancement - enhancement.mean()) ** 2)
            assert fit.loc[state, "enh_r2"] == pytest.approx(1 - ss_res / ss_tot)
            corr = np.corrcoef(enhancement, modeled)[0, 1]
            assert fit.loc[state, "enh_corr2"] == pytest.approx(corr**2)
        # the squared correlation is fips's R2, and ignores that the prior is
        # twice too large; the variance R2 does not
        assert fit.loc["posterior", "enh_corr2"] == pytest.approx(post.r2)
        assert fit.loc["prior", "enh_corr2"] > 0.95
        assert fit.loc["prior", "enh_r2"] < 0.5 * fit.loc["prior", "enh_corr2"]

    def test_r2_is_of_the_enhancement_not_the_concentration(self):
        # a background that follows the observations would give a high R2 of the
        # concentration whatever the state does
        post, z = fit_problem_solved(truth_scale=0.5)
        prior_model = FIT_H @ FIT_X0 + FIT_BACKGROUND
        conc_r2 = 1 - np.sum((z - prior_model) ** 2) / np.sum((z - z.mean()) ** 2)
        assert report.fit_metrics(post).loc["prior", "enh_r2"] < conc_r2 - 0.1

    def test_a_constant_enhancement_has_no_r2(self):
        x0 = Vector.from_series(pd.Series(FIT_X0, index=FIT_CELLS), block="flux")
        obs = Vector.from_series(
            pd.Series(2.0, index=FIT_OBS_INDEX), block="concentration"
        )
        problem = Problem(
            prior=x0,
            prior_cov=Independent().with_std(0.2),
            obs=obs,
            obs_cov=Independent().with_std(0.1),
            forward=Operator(FIT_H, obs.space, x0.space),
        )
        fit = report.fit_metrics(fips.solve(problem))
        assert fit["enh_r2"].isna().all()
        assert fit["enh_corr2"].isna().all()
        assert fit["conc_rmse"].notna().all()


def test_summarize_prints_the_bias_as_model_minus_obs_and_both_r2(summing, capsys):
    post, _ = fit_problem_solved(truth_scale=0.5)
    fit = report.fit_metrics(post)

    report.summarize(summing, post)

    out = capsys.readouterr().out
    assert (
        f"Mean Bias, model - obs (Prior):     {fit.loc['prior', 'conc_mean_bias']:.4f}"
        in out
    )
    assert (
        f"Mean Bias, model - obs (Posterior): {fit.loc['posterior', 'conc_mean_bias']:.4f}"
        in out
    )
    assert fit.loc["prior", "conc_mean_bias"] > 0
    assert f"(Prior):     {fit.loc['prior', 'enh_r2']:.3f}" in out
    assert f"(Posterior): {fit.loc['posterior', 'enh_r2']:.3f}" in out
    assert "Enhancement corr^2 (Posterior):" in out
    assert "R^2 (Prior):" not in out and "Bayesian R^2" not in out


def test_the_sweep_and_the_printed_report_give_the_same_bias(
    summing, monkeypatch, capsys
):
    from slv.inversion import sweep

    monkeypatch.setattr(sweep, "calculate_total_flux", report.calculate_total_flux)
    post, _ = fit_problem_solved(truth_scale=0.5)  # the model is above the obs

    row = sweep.collect_metrics(post, summing)
    report.summarize(summing, post)

    printed = capsys.readouterr().out
    for state, label in (("prior", "Prior):    "), ("posterior", "Posterior):")):
        bias = row[f"{state}_conc_mean_bias"]
        assert bias > 0
        assert f"(%s {bias:.4f}" % label in printed
