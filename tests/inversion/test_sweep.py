"""Tests for slv.inversion.sweep and the content-addressed fips_cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import (
    DEFAULT_COMPONENT_DEPS,
    SLVMethaneInversion,
    _component_hash,
)
from slv.inversion.sweep import (
    Sweep,
    SweepResults,
    collect_metrics,
    config_id,
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
        c2 = _cfg(mdm_config={"transport_pbl": {"std": 0.20}})
        fields = DEFAULT_COMPONENT_DEPS["prior_error"]
        assert _component_hash(c1, fields) == _component_hash(c2, fields)

    def test_mdm_hash_changes_with_mdm_config(self):
        """Changing mdm_config MUST change the modeldata_mismatch hash."""
        c1 = _cfg(mdm_config={})
        c2 = _cfg(mdm_config={"transport_pbl": {"std": 0.20}})
        fields = DEFAULT_COMPONENT_DEPS["modeldata_mismatch"]
        assert _component_hash(c1, fields) != _component_hash(c2, fields)

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
    """Smoke-tests for the cache decorator using fake fips objects."""

    def _make_pipeline(self, config, cache_dir):
        """Build a minimal mock pipeline to exercise the cache decorator."""
        from slv.inversion.pipelines import fips_cache

        class FakeObj:
            def __init__(self, val):
                self.val = val

            def to_file(self, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(self.val))

            @classmethod
            def from_file(cls, path):
                return cls(int(path.read_text()))

        class FakePipeline:
            def __init__(self, cfg):
                self.config = cfg

            COMPONENT_DEPS = DEFAULT_COMPONENT_DEPS

            @fips_cache(FakeObj, "obs")
            def get_obs(self):
                return FakeObj(42)

        config.cache = str(cache_dir)
        return FakePipeline(config)

    def test_cache_miss_saves_file(self, tmp_path):
        config = _cfg()
        p = self._make_pipeline(config, tmp_path)
        result = p.get_obs()
        assert result.val == 42
        # File should exist somewhere under tmp_path/obs/
        obs_files = list((tmp_path / "obs").glob("*.pkl"))
        assert len(obs_files) == 1

    def test_cache_hit_loads_file(self, tmp_path):
        config = _cfg()
        p = self._make_pipeline(config, tmp_path)
        p.get_obs()  # populate cache
        # Patch method to track whether it's called on 2nd invocation
        with patch.object(type(p), "get_obs", wraps=p.get_obs):
            # Create a fresh pipeline pointing to same cache
            p2 = self._make_pipeline(config, tmp_path)
            result = p2.get_obs()
        # Value comes from cache file
        assert result.val == 42

    def test_two_configs_sharing_obs_deps_share_cache_file(self, tmp_path):
        """Same sites/time → same obs hash → same cache file."""
        c1 = _cfg(prior_base_std=0.01)
        c2 = _cfg(prior_base_std=0.05)
        p1 = self._make_pipeline(c1, tmp_path)
        p2 = self._make_pipeline(c2, tmp_path)
        p1.get_obs()
        p2.get_obs()
        # Only one unique hash file should exist
        obs_files = list((tmp_path / "obs").glob("*.pkl"))
        assert len(obs_files) == 1

    def test_different_sites_produce_different_cache_files(self, tmp_path):
        c1 = _cfg(sites=["wbb"])
        c2 = _cfg(sites=["wbb", "hw"])
        p1 = self._make_pipeline(c1, tmp_path)
        p2 = self._make_pipeline(c2, tmp_path)
        p1.get_obs()
        p2.get_obs()
        obs_files = list((tmp_path / "obs").glob("*.pkl"))
        assert len(obs_files) == 2

    def test_cache_overwrite_all_clears_component(self, tmp_path):
        config = _cfg()
        p = self._make_pipeline(config, tmp_path)
        p.get_obs()
        obs_files_before = list((tmp_path / "obs").glob("*.pkl"))
        assert len(obs_files_before) == 1

        config.cache_overwrite = "all"
        p2 = self._make_pipeline(config, tmp_path)
        p2.get_obs()
        # File count stays 1 (old file deleted, new one written)
        obs_files_after = list((tmp_path / "obs").glob("*.pkl"))
        assert len(obs_files_after) == 1

    def test_cache_disabled_does_not_create_files(self, tmp_path):
        config = _cfg()  # cache=False by default
        from slv.inversion.pipelines import fips_cache

        class FakeObj:
            def to_file(self, p):
                pass

            @classmethod
            def from_file(cls, p):
                pass

        class FakePipeline:
            def __init__(self, cfg):
                self.config = cfg

            COMPONENT_DEPS = DEFAULT_COMPONENT_DEPS

            @fips_cache(FakeObj, "obs")
            def get_obs(self):
                return FakeObj()

        p = FakePipeline(config)
        p.get_obs()
        assert not (tmp_path / "obs").exists()


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

    def test_pipeline_cls_propagated(self):
        sweep = Sweep(
            cache="/tmp/sweep",
            pipeline_cls=SLVMethaneInversion,
            prior_base_std=[0.01],
        )
        assert all(cls is SLVMethaneInversion for _, cls in sweep.configs)

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
        sweep = Sweep(cache="/tmp/sweep", bg_baseline_window=["7d", "14d"])
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


class TestCollectMetrics:
    def _mock_problem_and_pipeline(self):
        """Build minimal mocks for FluxProblem and pipeline."""
        estimator = MagicMock()
        estimator.reduced_chi2 = 1.05
        estimator.R2 = 0.82
        estimator.RMSE = 0.03
        estimator.DOFS = 45.0
        estimator.uncertainty_reduction = 0.35

        problem = MagicMock()
        problem.estimator = estimator
        problem.enhancement = pd.Series([1.0, 2.0, 3.0])
        problem.prior_concentrations = pd.Series([1.1, 1.9, 3.1])
        problem.posterior_concentrations = pd.Series([1.0, 2.0, 3.0])
        problem.prior_fluxes = pd.Series(
            [1.0, 2.0], index=pd.to_datetime(["2023-01", "2023-02"])
        )
        problem.posterior_fluxes = pd.Series(
            [1.2, 2.1], index=pd.to_datetime(["2023-01", "2023-02"])
        )

        pipeline = MagicMock()
        pipeline.calculate_total_flux.side_effect = (
            lambda fluxes, units=None: pd.Series([float(fluxes.sum())])
        )

        return problem, pipeline

    def test_basic_metrics_present(self):
        problem, pipeline = self._mock_problem_and_pipeline()
        config = _cfg()
        row = collect_metrics(problem, config, pipeline)
        assert "reduced_chi2" in row
        assert "chi2_distance" in row
        assert "DOFS" in row
        assert "uncertainty_reduction" in row
        assert "config_id" in row

    def test_chi2_value(self):
        problem, pipeline = self._mock_problem_and_pipeline()
        row = collect_metrics(problem, _cfg(), pipeline)
        assert abs(row["reduced_chi2"] - 1.05) < 1e-9
        assert abs(row["chi2_distance"] - 0.05) < 1e-9

    def test_concentration_residuals(self):
        problem, pipeline = self._mock_problem_and_pipeline()
        row = collect_metrics(problem, _cfg(), pipeline)
        assert "prior_conc_rmse" in row
        assert "posterior_conc_rmse" in row
        # Perfect posterior → zero RMSE
        assert abs(row["posterior_conc_rmse"]) < 1e-9

    def test_swept_params_captured(self):
        problem, pipeline = self._mock_problem_and_pipeline()
        config = _cfg(prior_base_std=0.019, prior_std_frac=0.5)
        row = collect_metrics(
            problem, config, pipeline, swept_params=["prior_base_std", "prior_std_frac"]
        )
        assert row["cfg_prior_base_std"] == pytest.approx(0.019)
        assert row["cfg_prior_std_frac"] == pytest.approx(0.5)


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
        cfg, cls = r.to_config(row)
        assert isinstance(cfg, InversionConfig)
        assert cls is SLVMethaneInversion
        assert abs(cfg.prior_base_std - 0.01) < 1e-9
