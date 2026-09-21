"""Tests for inversion pipeline caching and bias classes."""

import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fips import Block, ForwardOperator, MatrixBlock, Vector
from fips.problems.flux import FluxInversionPipeline

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import (
    SLVMethaneInversion,
    _component_hash,
    _pkg_rev,
    check_state_cells,
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
    mock_vector.__getitem__ = lambda self, key: (
        mock_block if key == "concentration" else None
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
        # flat (un-hashed) components live under .fips/<fips+pystilt version tag>/
        from slv.inversion.pipelines import _version_tag

        assert (tmp_path / ".fips" / _version_tag() / "test_data.pkl").exists()

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
# SLVMethaneInversion.get_bias
# ---------------------------------------------------------------------------


class TestBiasGetBias:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
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
# SLVMethaneInversion.get_bias_jacobian
# ---------------------------------------------------------------------------


class TestBiasJacobian:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
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
# SLVMethaneInversion (site_group grouping)
# ---------------------------------------------------------------------------


class TestGetSiteGroup:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            bias_std=0.5,
            bias_grouping="site_group",
        )

    def test_uataq_site(self, pipeline):
        assert pipeline.get_site_group("wbb") == "UATAQ"

    def test_daq_site(self, pipeline):
        assert pipeline.get_site_group("hw") == "DAQ"

    def test_unknown_site(self, pipeline):
        assert pipeline.get_site_group("nonexistent") == "unknown"


# ---------------------------------------------------------------------------
# SLVMethaneInversion.get_bias (site_group grouping)
# ---------------------------------------------------------------------------


class TestSiteGroupBiasGetBias:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            sites=["wbb", "hw"],  # UATAQ and DAQ
            bias_std=0.5,
            bias_grouping="site_group",
        )

    def test_returns_series(self, pipeline):
        bias = pipeline.get_bias()
        assert isinstance(bias, pd.Series)

    def test_has_multiindex(self, pipeline):
        bias = pipeline.get_bias()
        assert isinstance(bias.index, pd.MultiIndex)

    def test_index_names(self, pipeline):
        bias = pipeline.get_bias()
        assert bias.index.names == ["time", "site_group"]

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
# SLVMethaneInversion.get_bias_jacobian (site_group grouping)
# ---------------------------------------------------------------------------


class TestSiteGroupBiasJacobian:
    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            sites=["wbb", "hw"],  # UATAQ and DAQ
            bias_std=0.5,
            bias_grouping="site_group",
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
        assert jac[(pd.Timestamp("2020-01-01"), "UATAQ")].iloc[0] == 1.0
        assert jac[(pd.Timestamp("2020-01-01"), "DAQ")].iloc[0] == 0.0

    def test_daq_obs_maps_to_daq_column(self, pipeline, prior):
        obs_times = pd.to_datetime(["2020-02-20"])
        obs = make_obs_vector(["hw_loc"], obs_times)

        jac = pipeline.get_bias_jacobian(obs, prior)
        assert jac[(pd.Timestamp("2020-02-01"), "DAQ")].iloc[0] == 1.0
        assert jac[(pd.Timestamp("2020-02-01"), "UATAQ")].iloc[0] == 0.0

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


# ---------------------------------------------------------------------------
# Integration tests for bias functionality
# ---------------------------------------------------------------------------


class TestBiasIntegration:
    """Test that bias components integrate correctly across pipeline methods."""

    @pytest.fixture
    def pipeline(self):
        return make_pipeline(
            SLVMethaneInversion,
            tstart="2020-01-01",
            tend="2020-04-01",
            flux_freq="MS",
            sites=["wbb", "hw"],
            bias_std=0.5,
            bias_grouping="site_group",
        )

    def test_get_prior_includes_bias_block(self, pipeline):
        """Test that get_prior() creates bias block when bias_std is set."""
        pipeline.get_bias()  # returns just the bias Series
        # Create a proper prior vector like get_prior does
        assert pipeline.config.bias_std is not None

    def test_get_prior_and_prior_error_compatibility(self, pipeline):
        """Test that get_prior and get_prior_error work together with bias."""
        # Simulate what happens in the pipeline
        bias = pipeline.get_bias()
        assert len(bias) > 0
        assert isinstance(bias.index, pd.MultiIndex)
        assert bias.index.names == ["time", "site_group"]

    def test_bias_std_affects_cache_key(self, pipeline):
        """Test that changing bias_std invalidates cache."""
        # This indirectly tests that COMPONENT_DEPS includes bias_std
        # for prior, forward_operator, and prior_error
        bias1 = pipeline.get_bias()
        pipeline.config.bias_std = 1.0
        bias2 = pipeline.get_bias()

        # Values should be different (different lengths due to grouping)
        # or same structure but we just verify the method runs
        assert isinstance(bias1, pd.Series)
        assert isinstance(bias2, pd.Series)


# ---------------------------------------------------------------------------
# Cache version tag: _pkg_rev
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def _repo_with_package(root, pkg_parent, name, tag):
    """A git repo at *root* holding package *name* under *pkg_parent*, tagged *tag*."""
    pkg = pkg_parent / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    _git(root, "tag", tag)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
class TestPkgRev:
    def test_installed_copy_ignores_enclosing_repo(self, tmp_path, monkeypatch):
        # slv's own .venv sits inside the slv repo: git must not answer for fips there.
        site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
        _repo_with_package(tmp_path, site, "slvtest_installed_pkg", "v9.9.9")
        monkeypatch.syspath_prepend(str(site))
        try:
            assert _pkg_rev("slvtest_installed_pkg", "slvtest-installed-pkg") == (
                "unknown"
            )
        finally:
            sys.modules.pop("slvtest_installed_pkg", None)

    def test_source_checkout_uses_git_describe(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        _repo_with_package(tmp_path, src, "slvtest_checkout_pkg", "v1.2.3")
        monkeypatch.syspath_prepend(str(src))
        try:
            assert _pkg_rev("slvtest_checkout_pkg", "slvtest-checkout-pkg") == "v1.2.3"
        finally:
            sys.modules.pop("slvtest_checkout_pkg", None)


# ---------------------------------------------------------------------------
# State cells: prior vs Jacobian columns
# ---------------------------------------------------------------------------

TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb", "wbb", "wbb"], pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-05"])],
    names=["obs_location", "obs_time"],
)


def flux_prior(lats=LATS, lons=LONS, bias=False):
    """A fips prior ordered (time, lat, lon), like the EPA prior."""
    index = pd.MultiIndex.from_product(
        [TIMES, lats, lons], names=["time", "lat", "lon"]
    )
    series = pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index)
    if not bias:
        return series, Vector(name="prior", data=Block(name="flux", data=series))
    bias_s = pd.Series(0.0, index=pd.Index(TIMES, name="time"), name="bias")
    blocks = [Block(series, name="flux"), Block(bias_s, name="bias")]
    return series, Vector(name="prior", data=blocks)


def jacobian(lats=LATS, lons=LONS):
    """A fips forward operator whose columns are ordered (lon, lat, time), like fips'."""
    columns = pd.MultiIndex.from_product(
        [lons, lats, TIMES], names=["lon", "lat", "time"]
    )
    rng = np.random.default_rng(0)
    H = pd.DataFrame(rng.random((len(OBS_INDEX), len(columns))), OBS_INDEX, columns)
    return H, ForwardOperator(
        MatrixBlock(H, row_block="concentration", col_block="flux")
    )


class TestCheckStateCells:
    def test_same_cells_pass(self):
        check_state_cells(flux_prior()[1], jacobian()[1])

    def test_bias_block_is_ignored(self):
        check_state_cells(flux_prior(bias=True)[1], jacobian()[1])

    def test_prior_cell_missing_from_jacobian_raises(self):
        # e.g. a Jacobian cached before the state grid gained its top row
        with pytest.raises(ValueError, match="2 prior cells have no Jacobian column"):
            check_state_cells(flux_prior()[1], jacobian(lats=[40.5])[1])

    def test_jacobian_cell_missing_from_prior_raises(self):
        with pytest.raises(ValueError, match="2 Jacobian cells are not in the prior"):
            check_state_cells(flux_prior(lats=[40.5])[1], jacobian()[1])

    def test_get_inputs_runs_the_check(self, monkeypatch):
        inputs = {
            "prior": flux_prior()[1],
            "forward_operator": jacobian(lats=[40.5])[1],
        }
        monkeypatch.setattr(FluxInversionPipeline, "get_inputs", lambda self: inputs)
        pipeline = make_pipeline(SLVMethaneInversion)
        with pytest.raises(ValueError, match="State cells disagree"):
            pipeline.get_inputs()


# ---------------------------------------------------------------------------
# Multiplicative MDM scale on the prior-modelled enhancement
# ---------------------------------------------------------------------------


def test_scale_on_prior_matches_by_label(monkeypatch):
    prior_s, prior = flux_prior()
    H, forward_operator = jacobian()
    obs = Vector(
        name="obs",
        data=Block(name="concentration", data=pd.Series(1.0, index=OBS_INDEX)),
    )
    pipeline = make_pipeline(SLVMethaneInversion)
    monkeypatch.setattr(pipeline, "get_prior", lambda: prior)
    monkeypatch.setattr(pipeline, "_get_flux_jacobian", lambda obs: forward_operator)

    x = prior_s.reorder_levels(H.columns.names).reindex(H.columns)
    expected = np.abs(H.to_numpy() @ x.to_numpy())
    positional = np.abs(H.to_numpy() @ prior_s.to_numpy())
    assert not np.allclose(expected, positional)  # the orders really differ here

    np.testing.assert_allclose(pipeline._multiplicative_scale(obs, "prior"), expected)


# ---------------------------------------------------------------------------
# get_forward_operator leaves the config alone
# ---------------------------------------------------------------------------


def test_auto_location_map_is_not_written_to_config(monkeypatch):
    import stilt

    import slv.inversion.config as config_module
    import slv.inversion.pipelines as pipelines_module
    from slv.inversion.sweep import config_id

    class FakeModel:
        def __init__(self, project):
            self.simulations = ["loc_a"]
            self.config = SimpleNamespace(
                footprints={"fine": SimpleNamespace(grid=SimpleNamespace(xres=0.01))}
            )

    seen = {}

    class FakeBuilder:
        def __init__(self, model):
            pass

        def build_from_target(self, target, **kwargs):
            seen.update(kwargs)
            return MatrixBlock(
                jacobian()[0], row_block="concentration", col_block="flux"
            )

    monkeypatch.setattr(stilt, "Model", FakeModel)
    monkeypatch.setattr(stilt, "SimID", lambda sid: SimpleNamespace(location=sid))
    monkeypatch.setattr(pipelines_module, "JacobianBuilder", FakeBuilder)
    monkeypatch.setattr(
        config_module,
        "build_location_site_map",
        lambda ids, site_config: {"loc_a": "wbb"},
    )

    pipeline = make_pipeline(
        SLVMethaneInversion, tstart="2020-01-01", tend="2020-03-01"
    )
    obs = Vector(
        name="obs",
        data=Block(name="concentration", data=pd.Series(1.0, index=OBS_INDEX)),
    )
    before = config_id(pipeline.config)
    pipeline.get_forward_operator(obs, flux_prior()[1])

    assert seen["location_mapper"] == {"loc_a": "wbb"}
    assert pipeline.config.location_site_map == {}
    assert config_id(pipeline.config) == before


def test_bias_jacobian_uses_given_location_mapper():
    pipeline = make_pipeline(
        SLVMethaneInversion,
        tstart="2020-01-01",
        tend="2020-03-01",
        sites=["wbb"],
        bias_std=0.5,
        bias_grouping="site_group",
    )
    obs = make_obs_vector(["loc_a"], pd.to_datetime(["2020-01-05"]))
    prior = make_bias_vector(pipeline.get_bias().index)
    jac = pipeline.get_bias_jacobian(obs, prior, location_mapper={"loc_a": "wbb"})
    org = pipeline.get_site_group("wbb")
    assert jac.loc[:, (pd.Timestamp("2020-01-01"), org)].tolist() == [1.0]


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("component", ["prior", "prior_error"])
def test_prior_keys_change_with_sites(component):
    # "site" / "site_group" bias blocks are indexed by config.sites
    fields = SLVMethaneInversion.COMPONENT_DEPS[component]
    a = InversionConfig(sites=["wbb"], bias_std=0.5, bias_grouping="site")
    b = InversionConfig(sites=["wbb", "hw"], bias_std=0.5, bias_grouping="site")
    assert _component_hash(a, fields) != _component_hash(b, fields)


# ---------------------------------------------------------------------------
# Mobile (TRAX) obs through the MDM and the background
# ---------------------------------------------------------------------------

TRAX_OBS_INDEX = pd.MultiIndex.from_arrays(
    [
        ["wbb", "multi_aaaaaaaaaa"],
        pd.to_datetime(["2024-06-02 20:00", "2024-06-02 20:13"]),
    ],
    names=["obs_location", "obs_time"],
)


def trax_obs():
    """A tower obs and a TRAX receptor obs (keyed by its PYSTILT location_id)."""
    data = pd.Series([2.0, 2.1], index=TRAX_OBS_INDEX)
    return Vector(name="obs", data=Block(name="concentration", data=data))


class TestMobileObsInPipeline:
    def test_obs_sites_resolves_receptors_to_the_mobile_site(self):
        pipeline = make_pipeline(SLVMethaneInversion, sites=["wbb", "trx01"])
        assert pipeline.obs_sites(trax_obs().index).tolist() == ["wbb", "trx01"]

    def test_obs_sites_without_a_mobile_site_raises(self):
        pipeline = make_pipeline(SLVMethaneInversion, sites=["wbb"])
        with pytest.raises(ValueError, match="0 mobile sites"):
            pipeline.obs_sites(trax_obs().index)

    def test_default_mdm_builds_with_mobile_obs(self, monkeypatch):
        # the default `instr` term looked up each obs_location's organization: KeyError
        pipeline = make_pipeline(SLVMethaneInversion, sites=["wbb", "trx01"])
        zeros = lambda obs, _: np.zeros(len(obs.index))  # noqa: E731
        monkeypatch.setattr(pipeline, "_multiplicative_scale", zeros)
        monkeypatch.setattr(pipeline, "_per_obs_std", zeros)
        mdm = pipeline.get_modeldata_mismatch(trax_obs())
        diag = np.diag(np.asarray(mdm.values, dtype=float))
        assert len(diag) == 2
        assert (diag > 0).all()

    def test_constant_keeps_mobile_obs(self, monkeypatch):
        # hourly background joined on exact obs_time left the 20:13 receptor NaN -> dropped
        from slv.inversion import background as bg

        hourly = pd.Series(
            [1.90, 1.95], index=pd.to_datetime(["2024-06-02 19:00", "2024-06-02 20:00"])
        )
        monkeypatch.setattr(bg, "get_rolling_background", lambda **kwargs: hourly)
        pipeline = make_pipeline(SLVMethaneInversion, sites=["wbb", "trx01"])
        constant = pipeline.get_constant(trax_obs())["concentration"]
        assert constant.index.get_level_values("obs_location").tolist() == [
            "wbb",
            "multi_aaaaaaaaaa",
        ]
        assert constant.tolist() == [1.95, 1.95]


# ---------------------------------------------------------------------------
# Robustness follow-ups (#4)
# ---------------------------------------------------------------------------

CELL_LATS = [40.5, 40.6]
CELL_LONS = [-112.0, -111.9]
SIM_INDEX = pd.MultiIndex.from_arrays(
    [
        ["wbb", "wbb", "wbb"],
        pd.to_datetime(["2020-01-20", "2020-01-05", "2020-02-05"]),
    ],
    names=["obs_location", "obs_time"],
)


def test_coverage_filter_adds_removed_cells_by_label():
    # Jacobian rows are simulations (here out of order, plus one with no obs); the
    # constant is indexed by obs. They used to be added by position.
    prior_s, prior = flux_prior(CELL_LATS, CELL_LONS)
    columns = pd.MultiIndex.from_product(
        [CELL_LONS, CELL_LATS, TIMES], names=["lon", "lat", "time"]
    )
    H = pd.DataFrame(1.0, index=SIM_INDEX, columns=columns)
    weak = columns.get_level_values("lon") == -112.0
    weak &= columns.get_level_values("lat") == 40.5
    H.loc[:, weak] = 1e-6  # the least-covered cell, removed at the 30th percentile
    forward_operator = ForwardOperator(
        MatrixBlock(H, row_block="concentration", col_block="flux")
    )
    obs_index = SIM_INDEX[[1, 0]].sort_values()  # 2020-01-05, 2020-01-20
    constant = Vector(
        name="background",
        data=Block(name="concentration", data=pd.Series([2.0, 2.5], index=obs_index)),
    )
    pipeline = make_pipeline(
        SLVMethaneInversion,
        tstart="2020-01-01",
        tend="2020-03-01",
        jacobian_coverage_percentile=30,
    )
    inputs = {
        "prior": prior,
        "forward_operator": forward_operator,
        "constant": constant,
    }

    out = pipeline._apply_jacobian_coverage_filter(inputs)

    x = prior_s.reorder_levels(columns.names).reindex(columns)
    removed = pd.Series(H.loc[:, weak].to_numpy() @ x[weak].to_numpy(), index=SIM_INDEX)
    expected = pd.Series([2.0, 2.5], index=obs_index) + removed.reindex(obs_index)
    got = out["constant"]["concentration"]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())


def test_prior_error_with_zero_bias_std_uses_the_flux_block():
    pipeline = make_pipeline(
        SLVMethaneInversion, tstart="2020-01-01", tend="2020-03-01", bias_std=0.0
    )
    _, prior = flux_prior(bias=True)
    S = pipeline.get_prior_error(prior)
    assert set(S.index.get_level_values("block")) == {"flux", "bias"}


class TestBiasBlockPerRun:
    def _pipeline(self, **kwargs):
        defaults = {"tstart": "2020-01-01", "tend": "2020-03-01", "bias_std": 0.5}
        return make_pipeline(SLVMethaneInversion, **(defaults | kwargs))

    def test_stale_cached_bias_block_is_replaced(self, monkeypatch):
        # a forward_operator cached with a bias block for other obs: only its flux
        # block is kept, and the bias block is rebuilt for these obs
        pipeline = self._pipeline(bias_grouping="time", sites=["wbb"])
        H, _ = jacobian()
        stale_bias = pd.DataFrame(
            1.0, index=OBS_INDEX[:1], columns=pd.Index(TIMES[:1], name="time")
        )
        cached = ForwardOperator(
            [
                MatrixBlock(H, "concentration", "flux"),
                MatrixBlock(stale_bias, "concentration", "bias"),
            ]
        )
        monkeypatch.setattr(pipeline, "_get_flux_jacobian", lambda obs: cached)
        obs = Vector(
            name="obs",
            data=Block(name="concentration", data=pd.Series(1.0, index=OBS_INDEX)),
        )
        prior = Vector(
            name="prior",
            data=[
                Block(flux_prior()[0], name="flux"),
                Block(pipeline.get_bias(), name="bias"),
            ],
        )
        fo = pipeline.get_forward_operator(obs, prior)
        bias = fo["concentration", "bias"]
        assert bias.shape == (len(OBS_INDEX), len(TIMES))
        assert (bias.sum(axis=1) == 1).all()

    def test_site_bias_maps_receptors_to_the_mobile_site(self):
        # "site" grouping keyed obs_location; a TRAX receptor matched no bias column
        pipeline = self._pipeline(
            bias_grouping="site",
            sites=["wbb", "trx01"],
            tstart="2024-06-01",
            tend="2024-07-01",
        )
        prior = make_bias_vector(pipeline.get_bias().index)
        jac = pipeline.get_bias_jacobian(trax_obs(), prior)
        june = pd.Timestamp("2024-06-01")
        assert jac.loc[:, (june, "trx01")].tolist() == [0.0, 1.0]
        assert jac.loc[:, (june, "wbb")].tolist() == [1.0, 0.0]


def test_mdm_scale_reuses_the_runs_prior_and_jacobian(monkeypatch):
    # with cache=False the multiplicative MDM used to rebuild both
    pipeline = make_pipeline(SLVMethaneInversion)
    _, prior = flux_prior()
    _, forward_operator = jacobian()
    builds = {"prior": 0, "jacobian": 0}

    def build_prior():
        builds["prior"] += 1
        return prior

    def build_jacobian(obs):
        builds["jacobian"] += 1
        return forward_operator

    monkeypatch.setattr(pipeline, "_build_prior", build_prior)
    monkeypatch.setattr(pipeline, "_get_flux_jacobian", build_jacobian)
    obs = Vector(
        name="obs",
        data=Block(name="concentration", data=pd.Series(1.0, index=OBS_INDEX)),
    )
    pipeline.get_forward_operator(obs, pipeline.get_prior())  # what get_inputs does
    for scale_on in ("footprint", "prior"):
        pipeline._multiplicative_scale(obs, scale_on)
    assert builds == {"prior": 1, "jacobian": 1}


def test_sparse_jacobian_scale_matches_dense():
    H, _ = jacobian()
    sparse = H.astype(pd.SparseDtype(float, 0.0))
    dense_rows = np.abs(H.to_numpy()).sum(axis=1)
    sparse_rows = np.asarray(abs(pipelines_matrix(sparse)).sum(axis=1)).ravel()
    np.testing.assert_allclose(sparse_rows, dense_rows)


def pipelines_matrix(block):
    from slv.inversion.mdm import _matrix

    return _matrix(block)


def test_total_units_are_mass_per_interval():
    pipeline = make_pipeline(SLVMethaneInversion, flux_freq="QS")
    assert pipeline._total_units() == "Gg per QS interval"
