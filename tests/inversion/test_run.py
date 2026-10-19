"""Tests for run and the plot hooks, on the made-up inversion of conftest.py."""

import fips
import numpy as np
import pytest

from slv.inversion import pipelines, report
from slv.inversion.coverage import covered_state


@pytest.fixture
def steps(monkeypatch):
    """The report and plot steps of run, recorded instead of done."""
    seen = []
    for name in ("plot_inputs", "summarize", "plot_results", "plot_diagnostics"):
        monkeypatch.setattr(report, name, lambda *a, _n=name, **k: seen.append((_n, a)))
    return seen


def test_run_builds_solves_and_summarizes(synthetic, steps):
    config = synthetic()
    post = pipelines.run(config)
    assert isinstance(post, fips.Posterior)
    assert [s[0] for s in steps] == ["summarize"]
    assert post.options.gamma == 1.0
    _, (cfg, summarized, problem) = steps[0]
    assert cfg is config and summarized is post and problem is post.problem


def test_run_solves_the_problem_it_is_given(synthetic, steps):
    config = synthetic()
    problem = pipelines.build_problem(config)
    post = pipelines.run(config, problem)
    assert post.problem is problem


def test_config_gamma_reaches_the_solver(synthetic, steps):
    post = pipelines.run(synthetic(gamma=3.0))
    assert post.options.gamma == 3.0


def test_run_applies_the_filter_and_plots(synthetic, steps):
    config = synthetic(
        jacobian_coverage_percentile=30,
        plot_inputs=True,
        plot_results=True,
        plot_diagnostics=True,
    )
    problem = pipelines.build_problem(config)
    post = pipelines.run(config, problem)
    assert [s[0] for s in steps] == [
        "plot_inputs",
        "summarize",
        "plot_results",
        "plot_diagnostics",
    ]
    keep = covered_state(problem, 30)
    assert len(keep) < len(problem.prior)
    assert post.problem.prior.space == keep
    # the reports get the whole problem
    assert steps[1][1][2] is problem


# --------------------------------------------------------------------------- plot hooks


@pytest.fixture
def viz_calls(monkeypatch):
    """Every viz function report.py calls, recorded instead of drawn."""
    calls = []
    for name in (
        "plot_grid",
        "plot_concentrations",
        "plot_inventory",
        "plot_prior_with_coverage",
        "plot_fluxes",
        "plot_reconstructed_posterior",
        "plot_total_fluxes_over_time",
        "plot_residuals",
        "plot_background_and_bias",
        "plot_fluxes_by_timestep",
        "plot_desroziers",
        "plot_removed_contribution",
    ):
        monkeypatch.setattr(
            report.viz, name, lambda *a, _n=name, **k: calls.append((_n, a, k))
        )
    monkeypatch.setattr(report, "plot_obs", lambda post: calls.append(("plot_obs",)))
    monkeypatch.setattr(
        report,
        "calculate_total_flux",
        lambda config, fluxes, units=None: f"total:{fluxes.name}:{units}",
    )
    monkeypatch.setattr(report.plt, "show", lambda: None)
    return calls


@pytest.fixture(params=[False, True], ids=["whole", "filtered"])
def solved(request, synthetic):
    """The config, the whole problem and its posterior, with or without the filter."""
    config = synthetic()
    problem = pipelines.build_problem(config)
    kept = problem
    if request.param:
        kept = problem.restrict(covered_state(problem, 30))
    return request.param, config, problem, fips.solve(kept)


def test_plot_inputs(viz_calls, solved):
    filtered, config, problem, post = solved
    report.plot_inputs(config, problem, post.problem)
    names = [c[0] for c in viz_calls]
    prior_plot = "plot_prior_with_coverage" if filtered else "plot_inventory"
    assert names == ["plot_grid", "plot_concentrations", prior_plot]
    assert viz_calls[0][2]["sites"] == ["wbb"]


def test_plot_results(viz_calls, solved):
    filtered, config, problem, post = solved
    report.plot_results(config, post, problem)
    names = [c[0] for c in viz_calls]
    expected = ["plot_fluxes", "plot_total_fluxes_over_time", "plot_obs"]
    if filtered:
        expected.insert(1, "plot_reconstructed_posterior")
    assert names == [*expected, "plot_residuals", "plot_background_and_bias"]
    (totals,) = [c for c in viz_calls if c[0] == "plot_total_fluxes_over_time"]
    assert totals[1] == ("total:prior:Gg/m2/s", "total:posterior:Gg/m2/s")
    assert totals[2]["units"] == "Gg per MS interval"


def test_plot_diagnostics(viz_calls, solved):
    filtered, config, problem, post = solved
    report.plot_diagnostics(config, post, problem)
    names = [c[0] for c in viz_calls]
    expected = ["plot_fluxes_by_timestep", "plot_desroziers"]
    if filtered:
        expected.append("plot_removed_contribution")
        (removed,) = [c for c in viz_calls if c[0] == "plot_removed_contribution"]
        contribution, offset = removed[1]
        assert (contribution > 0).any()
        np.testing.assert_allclose(offset - contribution, 1.95)
    assert names == expected
    (desroziers,) = [c for c in viz_calls if c[0] == "plot_desroziers"]
    assert set(desroziers[2]) == {"by_site", "per_obs", "timeseries"}
