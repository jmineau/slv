"""Tests for inversion pipeline caching and bias classes."""

import pickle
from unittest.mock import MagicMock

import pandas as pd
import pytest

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import (
    SLVMethaneInversionWithBias,
    SLVMethaneInversionWithSiteGroupBias,
    fips_cache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pipeline(cls, **config_kwargs):
    """Instantiate a pipeline subclass without calling FluxInversionPipeline.__init__."""
    obj = object.__new__(cls)
    obj.config = InversionConfig(**config_kwargs)
    return obj


def make_obs_vector(locations, times):
    """Build a minimal mock Vector whose ["concentration"].index is a MultiIndex."""
    index = pd.MultiIndex.from_arrays(
        [locations, times], names=["obs_location", "obs_time"]
    )
    mock_block = MagicMock()
    mock_block.index = index

    mock_vector = MagicMock()
    mock_vector.__getitem__ = (
        lambda self, key: mock_block if key == "concentration" else None
    )
    return mock_vector


def make_bias_vector(bias_index):
    """Build a minimal mock Vector whose ["bias"].index is bias_index."""
    mock_block = MagicMock()
    mock_block.index = bias_index

    mock_vector = MagicMock()
    mock_vector.__getitem__ = lambda self, key: mock_block if key == "bias" else None
    return mock_vector


class FakeCacheObject:
    """Module-level picklable object used by TestFipsCache."""

    def __init__(self, value):
        self.value = value

    def to_file(self, path):
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def from_file(cls, path):
        return pickle.loads(path.read_bytes())


# ---------------------------------------------------------------------------
# fips_cache decorator
# ---------------------------------------------------------------------------


class TestFipsCache:
    """Test caching behaviour of the fips_cache decorator."""

    def test_caching_disabled_always_calls_method(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(cache=False)

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(42)

        p = FakePipeline()
        p.compute()
        p.compute()
        assert call_count == 2

    def test_first_call_saves_cache(self, tmp_path):
        class FakePipeline:
            config = InversionConfig(cache=str(tmp_path))

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                return FakeCacheObject(99)

        FakePipeline().compute()
        assert (tmp_path / "test_data.pkl").exists()

    def test_second_call_loads_from_cache(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(cache=str(tmp_path))

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(99)

        FakePipeline().compute()
        FakePipeline().compute()
        assert call_count == 1

    def test_cache_overwrite_all_recomputes(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(cache=str(tmp_path), cache_overwrite="all")

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(99)

        FakePipeline().compute()
        FakePipeline().compute()
        assert call_count == 2

    def test_cache_overwrite_specific_stem_recomputes(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(cache=str(tmp_path), cache_overwrite=["test_data"])

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(99)

        FakePipeline().compute()
        FakePipeline().compute()
        assert call_count == 2

    def test_cache_overwrite_other_stem_still_loads_cache(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(
                cache=str(tmp_path), cache_overwrite=["other_data"]
            )

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(99)

        FakePipeline().compute()
        FakePipeline().compute()
        assert call_count == 1

    def test_cache_overwrite_single_string_matches_stem(self, tmp_path):
        call_count = 0

        class FakePipeline:
            config = InversionConfig(cache=str(tmp_path), cache_overwrite="test_data")

            @fips_cache(FakeCacheObject, "test_data")
            def compute(self):
                nonlocal call_count
                call_count += 1
                return FakeCacheObject(99)

        FakePipeline().compute()
        FakePipeline().compute()
        assert call_count == 2


# ---------------------------------------------------------------------------
# SLVMethaneInversionWithBias.get_bias
# ---------------------------------------------------------------------------


class TestGetBias:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversionWithBias,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
        )

    def test_get_bias_returns_series(self, pipeline):
        bias = pipeline.get_bias()
        assert isinstance(bias, pd.Series)

    def test_get_bias_length(self, pipeline):
        bias = pipeline.get_bias()
        assert len(bias) == 3  # Jan, Feb, Mar 2020

    def test_get_bias_all_zeros(self, pipeline):
        bias = pipeline.get_bias()
        assert (bias == 0.0).all()

    def test_get_bias_index_name(self, pipeline):
        bias = pipeline.get_bias()
        assert bias.index.name == "time"

    def test_get_bias_index_values(self, pipeline):
        bias = pipeline.get_bias()
        assert bias.index[0] == pd.Timestamp("2020-01-01")
        assert bias.index[-1] == pd.Timestamp("2020-03-01")


# ---------------------------------------------------------------------------
# SLVMethaneInversionWithBias.get_bias_jacobian
# ---------------------------------------------------------------------------


class TestGetBiasJacobian:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversionWithBias,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
        )

    def test_each_obs_maps_to_exactly_one_time(self, pipeline):
        obs_times = pd.to_datetime(["2020-01-15", "2020-02-10", "2020-03-20"])
        locs = ["loc_a", "loc_b", "loc_c"]
        obs = make_obs_vector(locs, obs_times)
        bias_index = pd.Index(pipeline.config.flux_times, name="time")
        prior = make_bias_vector(bias_index)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert (jac.sum(axis=1) == 1.0).all()

    def test_obs_maps_to_correct_month(self, pipeline):
        obs_times = pd.to_datetime(["2020-02-15"])
        obs = make_obs_vector(["loc_a"], obs_times)
        bias_index = pd.Index(pipeline.config.flux_times, name="time")
        prior = make_bias_vector(bias_index)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert jac[pd.Timestamp("2020-02-01")].iloc[0] == 1.0
        assert jac[pd.Timestamp("2020-01-01")].iloc[0] == 0.0

    def test_jacobian_columns_match_bias_index(self, pipeline):
        obs_times = pd.to_datetime(["2020-01-15"])
        obs = make_obs_vector(["loc_a"], obs_times)
        bias_index = pd.Index(pipeline.config.flux_times, name="time")
        prior = make_bias_vector(bias_index)

        jac = pipeline.get_bias_jacobian(obs, prior)
        pd.testing.assert_index_equal(jac.columns, bias_index)


# ---------------------------------------------------------------------------
# SLVMethaneInversionWithSiteGroupBias.get_site_group
# ---------------------------------------------------------------------------


class TestGetSiteGroup:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversionWithSiteGroupBias,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
        )

    def test_uataq_site(self, pipeline):
        assert pipeline.get_site_group("wbb") == "UATAQ"

    def test_daq_site(self, pipeline):
        assert pipeline.get_site_group("hw") == "DAQ"

    def test_unknown_site(self, pipeline):
        assert pipeline.get_site_group("nonexistent") == "unknown"


# ---------------------------------------------------------------------------
# SLVMethaneInversionWithSiteGroupBias.get_bias
# ---------------------------------------------------------------------------


class TestSiteGroupBiasGetBias:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversionWithSiteGroupBias,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
        )

    def test_returns_series(self, pipeline):
        bias = pipeline.get_bias()
        assert isinstance(bias, pd.Series)

    def test_has_multiindex(self, pipeline):
        bias = pipeline.get_bias()
        assert isinstance(bias.index, pd.MultiIndex)

    def test_index_names(self, pipeline):
        bias = pipeline.get_bias()
        assert bias.index.names == ["site_group", "time"]

    def test_site_groups_present(self, pipeline):
        bias = pipeline.get_bias()
        groups = bias.index.get_level_values("site_group").unique().tolist()
        assert "UATAQ" in groups
        assert "DAQ" in groups

    def test_length(self, pipeline):
        # 2 groups × 3 months
        bias = pipeline.get_bias()
        assert len(bias) == 2 * 3

    def test_all_zeros(self, pipeline):
        bias = pipeline.get_bias()
        assert (bias == 0.0).all()


# ---------------------------------------------------------------------------
# SLVMethaneInversionWithSiteGroupBias.get_bias_jacobian
# ---------------------------------------------------------------------------


class TestSiteGroupBiasJacobian:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversionWithSiteGroupBias,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
            location_site_map={
                "wbb_loc": "wbb",  # UATAQ
                "hw_loc": "hw",  # DAQ
            },
        )

    @pytest.fixture
    def prior(self, pipeline):
        bias = pipeline.get_bias()
        return make_bias_vector(bias.index)

    def test_uataq_obs_maps_to_uataq_column(self, pipeline, prior):
        obs_times = pd.to_datetime(["2020-01-15"])
        obs = make_obs_vector(["wbb_loc"], obs_times)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert jac[("UATAQ", pd.Timestamp("2020-01-01"))].iloc[0] == 1.0
        assert jac[("DAQ", pd.Timestamp("2020-01-01"))].iloc[0] == 0.0

    def test_daq_obs_maps_to_daq_column(self, pipeline, prior):
        obs_times = pd.to_datetime(["2020-02-20"])
        obs = make_obs_vector(["hw_loc"], obs_times)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert jac[("DAQ", pd.Timestamp("2020-02-01"))].iloc[0] == 1.0
        assert jac[("UATAQ", pd.Timestamp("2020-02-01"))].iloc[0] == 0.0

    def test_each_obs_maps_to_exactly_one_column(self, pipeline, prior):
        obs_times = pd.to_datetime(["2020-01-15", "2020-02-10", "2020-03-05"])
        locs = ["wbb_loc", "hw_loc", "wbb_loc"]
        obs = make_obs_vector(locs, obs_times)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert (jac.sum(axis=1) == 1.0).all()

    def test_jacobian_columns_match_bias_index(self, pipeline, prior):
        obs_times = pd.to_datetime(["2020-01-15"])
        obs = make_obs_vector(["wbb_loc"], obs_times)

        jac = pipeline.get_bias_jacobian(obs, prior)
        pd.testing.assert_index_equal(jac.columns, prior["bias"].index)
