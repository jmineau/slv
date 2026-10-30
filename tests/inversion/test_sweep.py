"""Tests for slv.inversion.sweep and the content-addressed fips_cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import fips
import numpy as np
import pandas as pd
import pytest
from fips import Independent, Operator, Problem, Vector

from slv.inversion import pipelines, sweep
from slv.inversion.cache import (
    DEFAULT_COMPONENT_DEPS,
    _component_hash,
    _version_tag,
    fips_cache,
)
from slv.inversion.config import InversionConfig
from slv.inversion.report import fit_metrics
from slv.inversion.sweep import (
    _RESULT_COLUMNS,
    Sweep,
    SweepResults,
    _append_row,
    collect_metrics,
    config_id,
    run_sweep_job,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> InversionConfig:
    """Build an InversionConfig with keyword overrides (no I/O)."""
    return InversionConfig(**overrides)


# ---------------------------------------------------------------------------
# 1. Content-addressed component hash tests
# ---------------------------------------------------------------------------


class TestComponentHash:
    """The cache hash should depend only on the fields listed in COMPONENT_DEPS."""

    def test_obs_hash_stable_for_irrelevant_change(self):
        """Changing prior_base_std must NOT change the obs hash."""
        c1 = _cfg(prior_base_std=0.01)
        c2 = _cfg(prior_base_std=0.05)
        fields = DEFAULT_COMPONENT_DEPS["obs"]
        assert _component_hash(c1, fields) == _component_hash(c2, fields)

    def test_obs_hash_changes_with_sites(self):
        """Changing sites MUST change the obs hash."""
        c1 = _cfg(sites=["wbb"])
        c2 = _cfg(sites=["wbb", "hw"])
        fields = DEFAULT_COMPONENT_DEPS["obs"]
        assert _component_hash(c1, fields) != _component_hash(c2, fields)

    def test_prior_error_hash_changes_with_prior_base_std(self):
        """Changing prior_base_std MUST change the prior_error hash."""
        c1 = _cfg(prior_base_std=0.01)
        c2 = _cfg(prior_base_std=0.03)
        fields = DEFAULT_COMPONENT_DEPS["prior_error"]
        assert _component_hash(c1, fields) != _component_hash(c2, fields)

    def test_prior_error_hash_stable_for_mdm_change(self):
        """Changing mdm_config must NOT change the prior_error hash."""
        c1 = _cfg(mdm_config={})
        c2 = _cfg(mdm_config={"bg": {"std": 0.02}})
        fields = DEFAULT_COMPONENT_DEPS["prior_error"]
        assert _component_hash(c1, fields) == _component_hash(c2, fields)

    def test_mdm_hash_changes_with_mdm_config(self):
        """Changing mdm_config MUST change the modeldata_mismatch hash."""
        c1 = _cfg(mdm_config={})
        c2 = _cfg(mdm_config={"transport": {"fraction": 0.5}})
        fields = DEFAULT_COMPONENT_DEPS["modeldata_mismatch"]
        assert _component_hash(c1, fields) != _component_hash(c2, fields)

    def test_unknown_mdm_component_is_rejected(self):
        """A retired/misspelled MDM key must fail loudly, not be ignored (and share a hash)."""
        import pytest

        from slv.inversion.config import get_mdm_comp_configs

        with pytest.raises(ValueError, match="transport_pbl"):
            get_mdm_comp_configs({"transport_pbl": {"std": 0.20}})

    def test_forward_operator_hash_changes_with_time(self):
        """Changing tend MUST change the forward_operator hash (obs & prior dep)."""
        c1 = _cfg(tend="2023-01-01")
        c2 = _cfg(tend="2024-01-01")
        fields = DEFAULT_COMPONENT_DEPS["forward_operator"]
        assert _component_hash(c1, fields) != _component_hash(c2, fields)

    def test_hash_is_12_chars(self):
        """Hash length must be exactly 12 hex chars."""
        c = _cfg()
        h = _component_hash(c, DEFAULT_COMPONENT_DEPS["obs"])
        assert len(h) == 12
        int(h, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# 2. Content-addressed fips_cache integration
# ---------------------------------------------------------------------------


class TestFipsCache:
    """
    Smoke-tests for the cache decorator.

    Hashed components live under ``<cache>/.fips/<version tag>/<component>/<hash>.pkl``.
    """

    @staticmethod
    def _component_files(cache_dir, component):
        return list((cache_dir / ".fips" / _version_tag() / component).glob("*.pkl"))

    @staticmethod
    def _get_obs(calls=None):
        @fips_cache("obs")
        def get_obs(config):
            if calls is not None:
                calls.append(1)
            return {"val": 42}

        return get_obs

    def test_cache_miss_saves_file(self, tmp_path):
        result = self._get_obs()(_cfg(cache=str(tmp_path)))
        assert result == {"val": 42}
        assert len(self._component_files(tmp_path, "obs")) == 1

    def test_cache_hit_loads_file(self, tmp_path):
        calls = []
        get_obs = self._get_obs(calls)
        get_obs(_cfg(cache=str(tmp_path)))
        assert get_obs(_cfg(cache=str(tmp_path))) == {"val": 42}
        assert len(calls) == 1

    def test_two_configs_sharing_obs_deps_share_cache_file(self, tmp_path):
        """Same sites/time → same obs hash → same cache file."""
        get_obs = self._get_obs()
        get_obs(_cfg(cache=str(tmp_path), prior_base_std=0.01))
        get_obs(_cfg(cache=str(tmp_path), prior_base_std=0.05))
        assert len(self._component_files(tmp_path, "obs")) == 1

    def test_different_sites_produce_different_cache_files(self, tmp_path):
        get_obs = self._get_obs()
        get_obs(_cfg(cache=str(tmp_path), sites=["wbb"]))
        get_obs(_cfg(cache=str(tmp_path), sites=["wbb", "hw"]))
        assert len(self._component_files(tmp_path, "obs")) == 2

    def test_cache_overwrite_all_clears_component(self, tmp_path):
        get_obs = self._get_obs()
        get_obs(_cfg(cache=str(tmp_path)))
        assert len(self._component_files(tmp_path, "obs")) == 1
        get_obs(_cfg(cache=str(tmp_path), cache_overwrite="all"))
        # File count stays 1 (old file deleted, new one written)
        assert len(self._component_files(tmp_path, "obs")) == 1

    def test_cache_disabled_does_not_create_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._get_obs()(_cfg())  # cache=False by default
        assert not (tmp_path / ".fips").exists()


# ---------------------------------------------------------------------------
# 3. Sweep class config generation
# ---------------------------------------------------------------------------


class TestSweepConfigs:
    def test_cartesian_product_count(self):
        sweep = Sweep(
            cache="/tmp/sweep",
            prior_base_std=[0.01, 0.02, 0.03],
            prior_std_frac=[0.3, 0.7],
        )
        assert len(sweep.configs) == 6  # 3 × 2

    def test_all_values_covered(self):
        sweep = Sweep(
            cache="/tmp/sweep",
            prior_base_std=[0.01, 0.02],
        )
        stds = {cfg.prior_base_std for cfg, _ in sweep.configs}
        assert stds == {0.01, 0.02}

    def test_cache_injected(self):
        sweep = Sweep(
            cache="/tmp/my_cache",
            prior_base_std=[0.01],
        )
        assert all(cfg.cache == "/tmp/my_cache" for cfg, _ in sweep.configs)

    def test_pipeline_propagated(self):
        sweep = Sweep(
            cache="/tmp/sweep",
            pipeline=_fake_pipeline,
            prior_base_std=[0.01],
        )
        assert all(p is _fake_pipeline for _, p in sweep.configs)

    def test_default_pipeline_is_run(self):
        sweep = Sweep(cache="/tmp/sweep", prior_base_std=[0.01])
        assert all(p is pipelines.run for _, p in sweep.configs)

    def test_base_config_fields_inherited(self):
        base = _cfg(tstart="2020-01-01", tend="2021-01-01")
        sweep = Sweep(
            cache="/tmp/sweep",
            base_config=base,
            prior_base_std=[0.01, 0.02],
        )
        for cfg, _ in sweep.configs:
            assert str(cfg.tstart) == "2020-01-01"
            assert str(cfg.tend) == "2021-01-01"

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="Unknown InversionConfig field"):
            Sweep(cache="/tmp/sweep", not_a_real_field=[1, 2])

    def test_no_sweep_params_raises(self):
        with pytest.raises(ValueError):
            Sweep(cache="/tmp/sweep")

    def test_single_sweep_param(self):
        sweep = Sweep(cache="/tmp/sweep", prior_time_scale=["16d", "32d"])
        assert len(sweep.configs) == 2


# ---------------------------------------------------------------------------
# 4. config_id
# ---------------------------------------------------------------------------


class TestConfigId:
    def test_same_config_same_id(self):
        c1 = _cfg(prior_base_std=0.019)
        c2 = _cfg(prior_base_std=0.019)
        assert config_id(c1) == config_id(c2)

    def test_different_config_different_id(self):
        c1 = _cfg(prior_base_std=0.01)
        c2 = _cfg(prior_base_std=0.02)
        assert config_id(c1) != config_id(c2)

    def test_cache_path_does_not_affect_id(self):
        """Changing only the cache path must not change the scientific ID."""
        c1 = _cfg(cache="/scratch/a")
        c2 = _cfg(cache="/scratch/b")
        assert config_id(c1) == config_id(c2)

    def test_id_is_16_chars(self):
        cid = config_id(_cfg())
        assert len(cid) == 16
        int(cid, 16)  # valid hex


# ---------------------------------------------------------------------------
# 5. collect_metrics
# ---------------------------------------------------------------------------


def _solved():
    """A small solved inversion: four obs, a flux block over two months and two cells."""
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2023-01-01", "2023-02-01"]), [40.5], [-112.0, -111.9]],
        names=["time", "lat", "lon"],
    )
    prior = Vector.from_series(
        pd.Series([1.0, 2.0, 1.5, 2.5], index=index), block="flux"
    )
    obs_index = pd.MultiIndex.from_arrays(
        [["wbb"] * 4, pd.date_range("2023-01-05", periods=4, freq="10D")],
        names=["obs_location", "obs_time"],
    )
    obs = Vector.from_series(
        pd.Series([2.1, 2.3, 2.2, 2.6], index=obs_index), block="concentration"
    )
    H = np.random.default_rng(0).random((4, 4)) * 0.1
    problem = Problem(
        prior=prior,
        prior_cov=Independent().with_std(0.5),
        obs=obs,
        obs_cov=Independent().with_std(0.05),
        forward=Operator(H, obs.space, prior.space),
        offset=1.9,
    )
    return fips.solve(problem)


@pytest.fixture
def summing(monkeypatch):
    """Totals as the sum over the cells (no xesmf needed)."""
    monkeypatch.setattr(
        sweep,
        "calculate_total_flux",
        lambda config, fluxes, units=None: fluxes.groupby(level="time").sum(),
    )


class TestCollectMetrics:
    def test_basic_metrics_present(self, summing):
        row = collect_metrics(_solved(), _cfg())
        for key in ("reduced_chi2", "chi2_distance", "DOFS", "uncertainty_reduction"):
            assert key in row
        assert row["config_id"] == config_id(_cfg())
        assert list(row)[: len(_RESULT_COLUMNS)] == list(_RESULT_COLUMNS)

    def test_fit_metrics_are_the_posterior_s(self, summing):
        post = _solved()
        row = collect_metrics(post, _cfg())
        assert row["reduced_chi2"] == pytest.approx(post.reduced_chi2)
        assert row["chi2_distance"] == pytest.approx(abs(post.reduced_chi2 - 1))
        assert row["posterior_enh_corr2"] == pytest.approx(post.r2)
        assert row["RMSE"] == pytest.approx(post.rmse)
        assert row["DOFS"] == pytest.approx(post.dofs)
        assert row["n_obs"] == 4 and row["n_state"] == 4

    def test_flux_totals(self, summing):
        post = _solved()
        row = collect_metrics(post, _cfg())
        assert row["prior_flux_mean"] == pytest.approx(3.5)  # (1 + 2, 1.5 + 2.5) / 2
        totals = post.estimate.to_series().groupby(level="time").sum()
        assert row["posterior_flux_mean"] == pytest.approx(totals.mean())

    def test_concentration_residuals(self, summing):
        row = collect_metrics(_solved(), _cfg())
        assert "prior_conc_rmse" in row
        assert "posterior_conc_rmse" in row

    def test_r2_columns_are_the_helper_s(self, summing):
        post = _solved()
        row = collect_metrics(post, _cfg())
        fit = fit_metrics(post)
        for state in ("prior", "posterior"):
            for name in ("conc_rmse", "conc_mean_bias", "enh_r2", "enh_corr2"):
                assert row[f"{state}_{name}"] == pytest.approx(fit.loc[state, name])
        assert "R2" not in row

    def test_swept_params_captured(self, summing):
        config = _cfg(prior_base_std=0.019, prior_std_frac=0.5)
        row = collect_metrics(
            _solved(), config, swept_params=["prior_base_std", "prior_std_frac"]
        )
        assert row["cfg_prior_base_std"] == pytest.approx(0.019)
        assert row["cfg_prior_std_frac"] == pytest.approx(0.5)


def _solved_with_background(background: float, truth_scale: float = 1.0):
    """
    A solved inversion with a constant *background*.

    The observations are ``H @ (truth_scale * x0) + background``, so the prior
    fits them exactly when ``truth_scale`` is 1.
    """
    obs_index = pd.MultiIndex.from_arrays(
        [["wbb"] * 3, pd.date_range("2020-01-05", periods=3, freq="D")],
        names=["obs_location", "obs_time"],
    )
    cells = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), [40.5], [-112.0, -111.9]],
        names=["time", "lat", "lon"],
    )
    H = np.array([[1.0, 0.5], [0.2, 1.0], [0.7, 0.7]])
    x0 = np.array([0.01, 0.02])
    z = H @ (truth_scale * x0) + background
    prior = Vector.from_series(pd.Series(x0, index=cells), block="flux")
    obs = Vector.from_series(pd.Series(z, index=obs_index), block="concentration")
    problem = Problem(
        prior=prior,
        prior_cov=Independent().with_std(0.01),
        obs=obs,
        obs_cov=Independent().with_std(0.01),
        forward=Operator(H, obs.space, prior.space),
        offset=background,
    )
    return fips.solve(problem)


class TestConcentrationResiduals:
    """
    The residuals compare the modeled concentration with the observed one.

    The model adds the background to ``H x``, so a residual taken against the
    observed enhancement (obs minus background) is off by the background (#27).
    """

    def test_prior_that_fits_the_obs_has_no_residual(self, summing):
        # The issue's reproduction: z = H x0 + c with c = 1.9 ppm.
        row = collect_metrics(_solved_with_background(1.9), _cfg())
        for col in ("prior_conc_rmse", "prior_conc_mean_bias"):
            assert row[col] == pytest.approx(0.0, abs=1e-12)
        for col in ("posterior_conc_rmse", "posterior_conc_mean_bias"):
            assert row[col] == pytest.approx(0.0, abs=1e-9)

    def test_residuals_are_model_minus_obs(self, summing):
        # The prior flux is two thirds of the truth: residual = (H x + c) - obs.
        post = _solved_with_background(1.9, truth_scale=1.5)
        problem = post.problem
        obs = problem.obs.to_series("concentration")
        prior_resid = problem.predict(problem.prior).to_series("concentration") - obs
        post_resid = post.modeled_obs.to_series("concentration") - obs

        row = collect_metrics(post, _cfg())

        assert row["prior_conc_rmse"] == pytest.approx(np.sqrt((prior_resid**2).mean()))
        assert row["prior_conc_mean_bias"] == pytest.approx(prior_resid.mean())
        assert row["posterior_conc_rmse"] == pytest.approx(
            np.sqrt((post_resid**2).mean())
        )
        assert row["posterior_conc_mean_bias"] == pytest.approx(post_resid.mean())
        assert (
            row["prior_conc_mean_bias"] < 0
        )  # model - obs: the model is below the obs
        assert row["n_obs"] == 3

    def test_residuals_do_not_depend_on_the_background(self, summing):
        keys = (
            "prior_conc_rmse",
            "posterior_conc_rmse",
            "prior_conc_mean_bias",
            "posterior_conc_mean_bias",
        )
        rows = [
            collect_metrics(_solved_with_background(bg, truth_scale=1.5), _cfg())
            for bg in (0.0, 1.9)
        ]
        for key in keys:
            assert rows[0][key] == pytest.approx(rows[1][key], abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Sweep.run() resume behaviour
# ---------------------------------------------------------------------------


class TestSweepRun:
    def _make_trivial_sweep(self, cache_dir):
        return Sweep(
            cache=str(cache_dir),
            prior_base_std=[0.01, 0.02],
        )

    def test_writes_csv(self, tmp_path):
        sweep = self._make_trivial_sweep(tmp_path / "cache")
        # Patch _run_single to avoid real inversions
        fake_row = {
            "config_id": "abc123",
            "reduced_chi2": 0.95,
            "chi2_distance": 0.05,
        }
        with patch.object(Sweep, "_run_single", return_value=fake_row):
            sweep.run(results_dir=tmp_path, n_jobs=1)
        assert (tmp_path / "sweep_results.csv").exists()

    def test_grid_json_written(self, tmp_path):
        sweep = self._make_trivial_sweep(tmp_path / "cache")
        with patch.object(
            Sweep,
            "_run_single",
            return_value={"config_id": "x", "reduced_chi2": 1.0, "chi2_distance": 0.0},
        ):
            sweep.run(results_dir=tmp_path, n_jobs=1)
        grid_path = tmp_path / "sweep_grid.json"
        assert grid_path.exists()
        grid = json.loads(grid_path.read_text())
        assert len(grid) == 2

    def test_resume_skips_done_configs(self, tmp_path):
        sweep = self._make_trivial_sweep(tmp_path / "cache")
        cid_first = config_id(sweep.configs[0][0])

        # Pre-write first config as done
        pd.DataFrame(
            [{"config_id": cid_first, "reduced_chi2": 1.0, "chi2_distance": 0.0}]
        ).to_csv(tmp_path / "sweep_results.csv", index=False)

        call_count = []

        def fake_run(args):
            call_count.append(1)
            return {"config_id": "new", "reduced_chi2": 1.1, "chi2_distance": 0.1}

        with patch.object(Sweep, "_run_single", side_effect=fake_run):
            sweep.run(results_dir=tmp_path, n_jobs=1)

        # Only the second config should have been run
        assert len(call_count) == 1

    def test_n_jobs_zero_writes_grid_only(self, tmp_path):
        sweep = self._make_trivial_sweep(tmp_path / "cache")
        with patch.object(Sweep, "_run_single") as mock_run:
            sweep.run(results_dir=tmp_path, n_jobs=0)
            mock_run.assert_not_called()
        assert (tmp_path / "sweep_grid.json").exists()
        assert not (tmp_path / "sweep_results.csv").exists()


# ---------------------------------------------------------------------------
# 7. SweepResults analysis
# ---------------------------------------------------------------------------


class TestSweepResults:
    def _make_csv(self, tmp_path) -> Path:
        rows = [
            {
                "config_id": "a",
                "reduced_chi2": 0.95,
                "chi2_distance": 0.05,
                "posterior_flux_mean": 10.0,
                "cfg_prior_base_std": 0.01,
            },
            {
                "config_id": "b",
                "reduced_chi2": 1.50,
                "chi2_distance": 0.50,
                "posterior_flux_mean": 12.0,
                "cfg_prior_base_std": 0.02,
            },
            {
                "config_id": "c",
                "reduced_chi2": 1.08,
                "chi2_distance": 0.08,
                "posterior_flux_mean": 10.5,
                "cfg_prior_base_std": 0.01,
            },
        ]
        path = tmp_path / "sweep_results.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_df_sorted_by_chi2_distance(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        assert r.df.iloc[0]["config_id"] == "a"  # chi2_distance=0.05
        assert r.df.iloc[1]["config_id"] == "c"  # chi2_distance=0.08

    def test_best_filters_by_tolerance(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        best = r.best(tol=0.1)
        assert set(best["config_id"]) == {"a", "c"}
        assert "b" not in best["config_id"].values

    def test_best_orders_by_distance_to_target(self, tmp_path):
        # df is sorted by distance to 1 (a, c, b); for target 1.5, b is the best
        r = SweepResults(self._make_csv(tmp_path))
        best = r.best(target=1.5, tol=0.6)  # a (0.95) is 0.55 away
        assert list(best["config_id"]) == ["b", "c", "a"]

    def test_best_empty_when_none_match(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        best = r.best(tol=0.01)
        assert len(best) == 0

    def test_stability_returns_dataframe(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        stab = r.stability("cfg_prior_base_std")
        assert isinstance(stab, pd.DataFrame)
        assert "posterior_flux_mean_std" in stab.columns

    def test_sensitivity_ranks_params(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        sens = r.sensitivity()
        assert isinstance(sens, pd.DataFrame)

    def test_to_config_round_trip(self, tmp_path):
        r = SweepResults(self._make_csv(tmp_path))
        row = r.best().iloc[0]
        cfg, pipeline = r.to_config(row)
        assert isinstance(cfg, InversionConfig)
        assert pipeline is pipelines.run
        assert abs(cfg.prior_base_std - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# 8. Results CSV: one column set for success and error rows
# ---------------------------------------------------------------------------


def _fake_pipeline(config):
    """Returns a solved inversion instead of running one."""
    return _solved()


def _failing_pipeline(config):
    raise RuntimeError("boom")


class TestResultRows:
    @pytest.fixture
    def sweep(self, tmp_path):
        return Sweep(cache=str(tmp_path / "cache"), prior_base_std=[0.01, 0.02])

    def _row(self, sweep, pipeline, tmp_path, i=0):
        cfg = sweep.configs[i][0]
        return sweep._run_single((cfg, pipeline, str(tmp_path), None, False))

    def test_success_and_error_rows_share_columns(self, sweep, tmp_path, summing):
        ok = self._row(sweep, _fake_pipeline, tmp_path)
        err = self._row(sweep, _failing_pipeline, tmp_path)
        assert list(ok) == list(err)
        assert ok["error"] is None and err["error"] == "boom"
        assert isinstance(ok["runtime_seconds"], float)
        assert isinstance(err["runtime_seconds"], float)

    @pytest.mark.parametrize("first", ["ok", "error"])
    def test_mixed_rows_read_back(self, sweep, tmp_path, first, summing):
        # Rows are appended by position: an error row after a success row used to
        # make read_csv fail, and one before it shifted the metrics into `error`.
        rows = {
            "ok": self._row(sweep, _fake_pipeline, tmp_path, 0),
            "error": self._row(sweep, _failing_pipeline, tmp_path, 1),
        }
        csv_path = tmp_path / "sweep_results.csv"
        for key in (first, "error" if first == "ok" else "ok"):
            _append_row(csv_path, rows[key])

        results = SweepResults(csv_path)
        assert len(results.df) == 2
        assert results.failed()["config_id"].tolist() == [rows["error"]["config_id"]]
        ok = results.df.set_index("config_id").loc[rows["ok"]["config_id"]]
        assert ok["reduced_chi2"] == pytest.approx(_solved().reduced_chi2)


class TestAppendRow:
    def test_aligns_by_column_name(self, tmp_path):
        csv_path = tmp_path / "r.csv"
        _append_row(csv_path, {"a": 1, "b": 2, "c": 3})
        _append_row(csv_path, {"c": 30, "a": 10})
        df = pd.read_csv(csv_path)
        assert df["a"].tolist() == [1, 10]
        assert df["c"].tolist() == [3, 30]
        assert df["b"].isna().tolist() == [False, True]

    def test_new_column_widens_old_file(self, tmp_path):
        csv_path = tmp_path / "r.csv"
        pd.DataFrame([{"config_id": "old", "reduced_chi2": 1.0}]).to_csv(
            csv_path, index=False
        )
        _append_row(
            csv_path, {"config_id": "new", "error": "boom", "reduced_chi2": None}
        )
        df = pd.read_csv(csv_path)
        assert df["config_id"].tolist() == ["old", "new"]
        assert df["error"].isna().tolist() == [True, False]

    def test_none_row_is_skipped(self, tmp_path):
        csv_path = tmp_path / "r.csv"
        _append_row(csv_path, None)
        assert not csv_path.exists()


def test_run_sweep_job_rows_read_back(tmp_path, monkeypatch, summing):
    swp = Sweep(cache=str(tmp_path / "cache"), prior_base_std=[0.01, 0.02])
    swp.run(results_dir=tmp_path, n_jobs=0)
    grid = json.loads((tmp_path / "sweep_grid.json").read_text())
    assert grid[0]["pipeline"] == "slv.inversion.pipelines.run"

    # the grid names the pipeline, which is looked up when the job runs
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(pipelines, "run", _fake_pipeline)
    run_sweep_job(tmp_path)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setattr(pipelines, "run", _failing_pipeline)
    run_sweep_job(tmp_path)

    results = SweepResults(tmp_path / "sweep_results.csv")
    assert len(results.df) == 2
    assert results.failed()["error"].tolist() == ["boom"]
    assert set(_RESULT_COLUMNS) <= set(results.df.columns)
    assert results.df["runtime_seconds"].notna().all()


def test_grid_only_sweep_returns_empty_results(tmp_path):
    sweep = Sweep(cache=str(tmp_path / "cache"), prior_base_std=[0.01, 0.02])
    results = sweep.run(results_dir=tmp_path, n_jobs=0)
    assert isinstance(results, SweepResults)
    assert results.df.empty
    assert results.best().empty


def test_sensitivity_best_value_is_closest_to_one(tmp_path):
    path = tmp_path / "sweep_results.csv"
    pd.DataFrame(
        {
            "config_id": ["a", "b", "c"],
            "reduced_chi2": [0.3, 1.05, 1.9],
            "RMSE": [0.05, 0.02, 0.04],
            "cfg_prior_base_std": [0.01, 0.02, 0.03],
        }
    ).to_csv(path, index=False)
    sens = SweepResults(path).sensitivity()
    # the lowest chi2 (0.3) used to count as best
    assert sens.loc[0, "best_value"] == pytest.approx(0.02)
    # without a target (any metric but reduced_chi2) the lowest is best
    assert SweepResults(path).sensitivity("RMSE").loc[0, "best_value"] == (
        pytest.approx(0.02)
    )
    assert SweepResults(path).sensitivity("RMSE", target=0.05).loc[0, "best_value"] == (
        pytest.approx(0.01)
    )
