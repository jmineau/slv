"""Tests for SLVMethaneInversion.run and the plot hooks, with every step stubbed."""

from types import SimpleNamespace

import pytest

from slv.inversion import report
from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import SLVMethaneInversion


class FakeProblem:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.solve_kwargs = None

    def solve(self, **kwargs):
        self.solve_kwargs = kwargs


def pipeline(steps, **config):
    """A pipeline whose steps record themselves in ``steps``."""
    p = object.__new__(SLVMethaneInversion)
    p.config = InversionConfig(tstart="2020-01-01", tend="2020-03-01", **config)
    p._InverseProblem = FakeProblem
    p.estimator = "bayesian"
    p.get_inputs = lambda: steps.append("inputs") or {"prior": "full"}

    def coverage_filter(inputs):
        steps.append("filter")
        return {**inputs, "prior": "filtered"}

    p._apply_jacobian_coverage_filter = coverage_filter
    p.summarize = lambda: steps.append("summarize")
    for name in ("plot_inputs", "plot_results", "plot_diagnostics"):
        setattr(p, name, lambda problem, _n=name: steps.append(_n))
    return p


def test_run_builds_solves_and_summarizes():
    steps = []
    p = pipeline(steps, plot_inputs=False, plot_results=False)
    problem = p.run(extra="x")
    assert steps == ["inputs", "summarize"]
    assert problem is p.problem
    assert problem.init_kwargs == {"prior": "full", "extra": "x"}
    assert problem.solve_kwargs == {"estimator": "bayesian"}


def test_run_applies_the_filter_gamma_and_plots():
    steps = []
    p = pipeline(
        steps, jacobian_coverage_percentile=10, gamma=3.0, plot_diagnostics=True
    )
    problem = p.run(estimator_kwargs={"gamma": 5.0, "tol": 1e-6})
    assert steps == [
        "inputs",
        "filter",
        "plot_inputs",
        "summarize",
        "plot_results",
        "plot_diagnostics",
    ]
    assert problem.init_kwargs == {"prior": "filtered"}
    # the explicit estimator kwargs override the config's gamma
    assert problem.solve_kwargs == {"estimator": "bayesian", "gamma": 5.0, "tol": 1e-6}


def test_config_gamma_reaches_the_solver():
    p = pipeline([], plot_inputs=False, plot_results=False, gamma=3.0)
    assert p.run().solve_kwargs["gamma"] == 3.0


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
    monkeypatch.setattr(report.plt, "show", lambda: None)
    return calls


class _Series:
    """Enough of a pandas Series for the plot hooks: to_xarray and a name."""

    name = "flux"

    def to_xarray(self):
        return "xr"


def _plot_pipeline(filtered, monkeypatch):
    p = object.__new__(SLVMethaneInversion)
    p.config = InversionConfig(tstart="2020-01-01", tend="2020-03-01")
    if filtered:
        p._full_prior = {"flux": _Series()}
        p._removed_cells = set()
        monkeypatch.setattr(
            SLVMethaneInversion, "_all_cells", property(lambda self: {(1, 1)})
        )
        p.reconstruct_posterior = lambda: _Series()
        p._removed_contribution = "removed"
    p.calculate_total_flux = lambda fluxes, units=None: f"total:{units}"
    p.desroziers_diagnostic = lambda groupby="obs_location", freq=None: (groupby, freq)
    problem = SimpleNamespace(
        concentrations="obs",
        prior_fluxes=_Series(),
        posterior_fluxes=_Series(),
        constant={"concentration": "bg"},
        plot=SimpleNamespace(concentrations=lambda: None),
    )
    return p, problem


@pytest.mark.parametrize("filtered", [False, True])
def test_plot_inputs(viz_calls, monkeypatch, filtered):
    p, problem = _plot_pipeline(filtered, monkeypatch)
    p.plot_inputs(problem)
    names = [c[0] for c in viz_calls]
    prior_plot = "plot_prior_with_coverage" if filtered else "plot_inventory"
    assert names == ["plot_grid", "plot_concentrations", prior_plot]
    assert viz_calls[0][2]["sites"] == ["wbb"]


@pytest.mark.parametrize("filtered", [False, True])
def test_plot_results(viz_calls, monkeypatch, filtered):
    p, problem = _plot_pipeline(filtered, monkeypatch)
    p.plot_results(problem)
    names = [c[0] for c in viz_calls]
    expected = ["plot_fluxes", "plot_total_fluxes_over_time", "plot_residuals"]
    if filtered:
        expected.insert(1, "plot_reconstructed_posterior")
    assert names == [*expected, "plot_background_and_bias"]
    (totals,) = [c for c in viz_calls if c[0] == "plot_total_fluxes_over_time"]
    assert totals[1] == ("total:Gg/m2/s", "total:Gg/m2/s")
    assert totals[2]["units"] == "Gg per MS interval"


@pytest.mark.parametrize("filtered", [False, True])
def test_plot_diagnostics(viz_calls, monkeypatch, filtered):
    p, problem = _plot_pipeline(filtered, monkeypatch)
    p.plot_diagnostics(problem)
    names = [c[0] for c in viz_calls]
    expected = ["plot_fluxes_by_timestep", "plot_desroziers"]
    if filtered:
        expected.append("plot_removed_contribution")
    assert names == expected
    (desroziers,) = [c for c in viz_calls if c[0] == "plot_desroziers"]
    assert desroziers[2] == {
        "by_site": ("obs_location", None),
        "per_obs": (None, None),
        "timeseries": ("obs_location", "MS"),
    }
