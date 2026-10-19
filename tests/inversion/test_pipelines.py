"""Tests for the inversion pipeline: the component cache, the bias block and the problem."""

import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fips import Operator, Problem, Space

from slv.inversion import mdm, pipelines
from slv.inversion.cache import _pkg_rev, _version_tag, fips_cache
from slv.inversion.config import InversionConfig
from slv.inversion.coverage import check_state_cells, matrix
from slv.inversion.pipelines import build_problem, get_bias, obs_sites

from .conftest import (
    LATS,
    LONS,
    NO_BACKGROUND,
    NO_FOOTPRINT,
    OBS_INDEX,
    TIMES,
    jacobian,
    prior_series,
)

# ---------------------------------------------------------------------------
# fips_cache decorator
# ---------------------------------------------------------------------------


class TestFipsCache:
    """Test caching behaviour of the fips_cache decorator."""

    @staticmethod
    def component(calls):
        @fips_cache("test_data")
        def compute(config):
            calls.append(1)
            return {"value": 99}

        return compute

    def test_caching_disabled_always_calls_the_function(self):
        calls = []
        compute = self.component(calls)
        config = InversionConfig(cache=False)
        compute(config)
        compute(config)
        assert len(calls) == 2

    def test_first_call_saves_cache(self, tmp_path):
        self.component([])(InversionConfig(cache=str(tmp_path)))
        # flat (un-hashed) components live under .fips/<fips+pystilt version tag>/
        assert (tmp_path / ".fips" / _version_tag() / "test_data.pkl").exists()

    def test_second_call_loads_from_cache(self, tmp_path):
        calls = []
        compute = self.component(calls)
        assert compute(InversionConfig(cache=str(tmp_path))) == {"value": 99}
        assert compute(InversionConfig(cache=str(tmp_path))) == {"value": 99}
        assert len(calls) == 1

    @pytest.mark.parametrize(
        ("overwrite", "calls"),
        [("all", 2), (["test_data"], 2), ("test_data", 2), (["other_data"], 1)],
    )
    def test_cache_overwrite(self, tmp_path, overwrite, calls):
        seen = []
        compute = self.component(seen)
        config = InversionConfig(cache=str(tmp_path), cache_overwrite=overwrite)
        compute(config)
        compute(config)
        assert len(seen) == calls


def test_package_release_rekeys_obs_but_not_the_jacobian(tmp_path, monkeypatch):
    import slv.inversion.cache as cache

    @fips_cache("obs")
    def get_obs(config):
        return "obs"

    @fips_cache("forward_operator")
    def get_fo(config):
        return "fo"

    def files(component):
        return sorted((tmp_path / ".fips").glob(f"*/{component}/*.pkl"))

    config = InversionConfig(cache=str(tmp_path))
    monkeypatch.setattr(cache, "_release", lambda p: "2026.12.2")
    get_obs(config)
    get_fo(config)
    monkeypatch.setattr(cache, "_release", lambda p: "2026.12.3")  # a new lair release
    get_obs(config)
    get_fo(config)
    assert len(files("obs")) == 2
    assert len(files("forward_operator")) == 1


def test_a_cached_operator_loads_as_it_was(tmp_path):
    @fips_cache("forward_operator")
    def get_fo(config):
        return jacobian()

    config = InversionConfig(cache=str(tmp_path))
    built, loaded = get_fo(config), get_fo(config)
    assert loaded is not built
    assert loaded.rows == built.rows and loaded.cols == built.cols
    np.testing.assert_array_equal(loaded.dense(), built.dense())


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

    def test_installed_release_is_named_as_its_tag(self, tmp_path, monkeypatch):
        # A release names one cache folder, installed (metadata "9.9.9") or checked
        # out (git describe "v9.9.9").
        site = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
        _repo_with_package(tmp_path, site, "slvtest_release_pkg", "v0.0.1")
        monkeypatch.syspath_prepend(str(site))
        monkeypatch.setattr("slv.inversion.cache._dist_version", lambda name: "9.9.9")
        try:
            assert _pkg_rev("slvtest_release_pkg", "slvtest-release-pkg") == "v9.9.9"
        finally:
            sys.modules.pop("slvtest_release_pkg", None)

    def test_source_checkout_uses_git_describe(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        _repo_with_package(tmp_path, src, "slvtest_checkout_pkg", "v1.2.3")
        monkeypatch.syspath_prepend(str(src))
        try:
            assert _pkg_rev("slvtest_checkout_pkg", "slvtest-checkout-pkg") == "v1.2.3"
        finally:
            sys.modules.pop("slvtest_checkout_pkg", None)


# ---------------------------------------------------------------------------
# The bias block
# ---------------------------------------------------------------------------


def obs_space(locations, times):
    index = pd.MultiIndex.from_arrays(
        [locations, pd.to_datetime(times)], names=["obs_location", "obs_time"]
    )
    return Space({"concentration": index})


def bias_frame(config, space):
    """The bias block's forward operator as a DataFrame (obs by bias label)."""
    prior, forward = get_bias(config, space)
    assert (prior.values == 0.0).all()
    frame = forward.to_frame()
    frame.index = frame.index.droplevel("block")
    frame.columns = frame.columns.droplevel("block")
    return frame


def config(**kwargs):
    defaults = dict(
        tstart="2020-01-01", tend="2020-04-01", flux_freq="MS", bias_std=0.5
    )
    return InversionConfig(**(defaults | kwargs))


class TestTimeBias:
    def test_one_bias_per_month_with_obs(self):
        space = obs_space(["a", "b", "c"], ["2020-01-15", "2020-02-10", "2020-02-20"])
        frame = bias_frame(config(), space)
        # March has no obs, so no bias
        assert list(frame.columns) == list(pd.to_datetime(["2020-01-01", "2020-02-01"]))
        assert frame.columns.name == "obs_time"
        assert (frame.sum(axis=1) == 1.0).all()
        assert frame.loc[("b", pd.Timestamp("2020-02-10")), pd.Timestamp("2020-02-01")]


class TestSiteGroupBias:
    @pytest.fixture
    def cfg(self):
        return config(
            sites=["wbb", "hw"],  # UATAQ and DAQ
            bias_grouping="site_group",
            location_site_map={"wbb_loc": "wbb", "hw_loc": "hw"},
        )

    def test_obs_map_to_their_organization(self, cfg):
        space = obs_space(
            ["wbb_loc", "hw_loc", "wbb_loc"], ["2020-01-15", "2020-02-10", "2020-03-05"]
        )
        frame = bias_frame(cfg, space)
        assert frame.columns.names == ["obs_time", "site_group"]
        assert (frame.sum(axis=1) == 1.0).all()
        jan, feb = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")
        assert frame.loc[("wbb_loc", pd.Timestamp("2020-01-15")), (jan, "UATAQ")] == 1
        assert frame.loc[("hw_loc", pd.Timestamp("2020-02-10")), (feb, "DAQ")] == 1


def test_site_bias_maps_receptors_to_the_mobile_site():
    # "site" grouping keyed obs_location; a TRAX receptor matched no bias column
    cfg = config(
        bias_grouping="site",
        sites=["wbb", "trx01"],
        tstart="2024-06-01",
        tend="2024-07-01",
    )
    frame = bias_frame(cfg, TRAX_SPACE)
    june = pd.Timestamp("2024-06-01")
    assert frame[(june, "trx01")].tolist() == [0.0, 1.0]
    assert frame[(june, "wbb")].tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# State cells: prior vs Jacobian columns
# ---------------------------------------------------------------------------


def flux_index(lats=LATS, lons=LONS):
    return pd.MultiIndex.from_product([TIMES, lats, lons], names=["time", "lat", "lon"])


def jacobian_columns(lats=LATS, lons=LONS):
    return pd.MultiIndex.from_product([lons, lats, TIMES], names=["lon", "lat", "time"])


class TestCheckStateCells:
    def test_same_cells_pass(self):
        check_state_cells(flux_index(), jacobian_columns())

    def test_only_the_flux_block_is_compared(self):
        state = Space(
            {"flux": flux_index(), "bias": pd.Index(TIMES, name="obs_time")}
        ).index
        check_state_cells(state, jacobian_columns())

    def test_prior_cell_missing_from_jacobian_raises(self):
        # e.g. a Jacobian cached before the state grid gained its top row
        with pytest.raises(ValueError, match="2 prior cells have no Jacobian column"):
            check_state_cells(flux_index(), jacobian_columns(lats=[40.5]))

    def test_jacobian_cell_missing_from_prior_raises(self):
        with pytest.raises(ValueError, match="2 Jacobian cells are not in the prior"):
            check_state_cells(flux_index(lats=[40.5]), jacobian_columns())

    def test_build_problem_runs_the_check(self, synthetic, monkeypatch):
        J = jacobian()
        flux = J.cols.select("flux")
        keep = flux.get_level_values("lat") == LATS[0]
        part = Operator(
            sp.csr_array(matrix(J)[:, np.flatnonzero(keep)]),
            J.rows,
            Space({"flux": flux[keep]}),
        )
        monkeypatch.setattr(pipelines, "_project_jacobian", lambda c, p, locs: part)
        with pytest.raises(ValueError, match="State cells disagree"):
            build_problem(synthetic())


def test_matrix_keeps_a_sparse_operator_sparse():
    J = jacobian()
    dense = Operator(J.dense(), J.rows, J.cols)
    assert sp.issparse(matrix(J)) and not sp.issparse(matrix(dense))
    np.testing.assert_array_equal(matrix(J).toarray(), matrix(dense))


# ---------------------------------------------------------------------------
# build_problem on the made-up inversion
# ---------------------------------------------------------------------------


def test_build_problem_puts_every_input_on_the_obs_and_the_state(synthetic, capsys):
    problem = build_problem(synthetic())
    assert isinstance(problem, Problem)
    assert problem.prior.space.names == ("flux",)
    assert problem.obs.space.names == ("concentration",)
    assert len(problem.prior) == len(TIMES) * len(LATS) * len(LONS)
    # the obs without a background is dropped, the one without a footprint is kept
    obs = problem.obs.space.select("concentration")
    assert NO_BACKGROUND not in obs and NO_FOOTPRINT in obs
    assert len(obs) == len(OBS_INDEX) - 1
    assert "Dropping 1/6 obs with missing background" in capsys.readouterr().out
    # the background is the offset
    np.testing.assert_allclose(problem.offset.values, 1.95)


def test_the_prior_and_jacobian_cells_meet_after_rounding(synthetic):
    # the prior's latitudes carry float noise (a regrid); the Jacobian's do not
    problem = build_problem(synthetic())
    lats = problem.prior.space.select("flux").get_level_values("lat")
    assert sorted(set(lats)) == LATS
    J = jacobian()
    receptor = OBS_INDEX[0]
    H = problem.forward.to_frame()
    H.index = H.index.droplevel("block")
    H.columns = H.columns.droplevel("block")
    expected = J.to_frame().droplevel("block").loc[receptor]
    expected.index = expected.index.droplevel("block")
    expected = expected.reorder_levels(H.columns.names)
    np.testing.assert_allclose(H.loc[receptor, expected.index], expected.to_numpy())


def test_an_obs_without_a_footprint_is_a_row_of_zeros(synthetic):
    problem = build_problem(synthetic())
    H = problem.forward.to_frame()
    H.index = H.index.droplevel("block")
    assert (H.loc[NO_FOOTPRINT] == 0).all()
    assert (H.drop(NO_FOOTPRINT).sum(axis=1) > 0).all()


def test_the_prior_error_is_kept_as_kronecker_factors(synthetic):
    from fips.linalg import KroneckerOperator

    problem = build_problem(synthetic())
    assert isinstance(problem.prior_cov.data, KroneckerOperator)
    prior = prior_series()
    std = 0.01 + 0.5 * prior.to_numpy()
    np.testing.assert_allclose(problem.prior_cov.variances.values, std**2, rtol=1e-6)


def test_the_bias_block_joins_the_state(synthetic):
    problem = build_problem(synthetic(bias_std=0.0))
    assert problem.prior.space.names == ("flux", "bias")
    assert len(problem.prior["bias"]) == 2  # the two months with obs
    np.testing.assert_allclose(problem.prior_cov["bias", "bias"].dense(), 0.0)
    assert (problem.forward["concentration", "bias"].dense().sum(axis=1) == 1).all()


def test_the_obs_are_averaged_to_days(synthetic):
    hourly = build_problem(synthetic())
    daily = build_problem(synthetic(aggregate_obs="1D"))
    days = daily.obs.space.select("concentration").get_level_values("obs_time")
    assert list(days) == list(
        pd.to_datetime(["2020-01-05", "2020-01-06", "2020-02-05"])
    )
    z = hourly.obs.values
    # 01-06 has one obs left (its 20:00 obs had no background)
    np.testing.assert_allclose(daily.obs.values, [z[:2].mean(), z[2], z[3:].mean()])
    np.testing.assert_allclose(daily.offset.values, 1.95)


def test_filter_state_space_keeps_only_the_time_range():
    cfg = config(tstart="2020-01-01", tend="2020-03-01", min_obs_per_interval=1)
    times = pd.to_datetime(
        ["2019-12-31 20:00", "2020-01-05 20:00", "2020-02-05 20:00", "2020-03-01 20:00"]
    )
    obs = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=obs_space(["wbb"] * 4, times).select("concentration"),
    )
    prior = pd.Series(
        1.0,
        index=pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2019-12-01", "2020-01-01", "2020-02-01"]),
                [40.5],
                [-112.0],
            ],
            names=["time", "lat", "lon"],
        ),
    )
    obs, prior = pipelines.filter_state_space(cfg, obs, prior)
    assert obs.tolist() == [2.0, 3.0]
    assert len(prior) == 2


def test_filter_state_space_drops_obs_of_sparse_intervals():
    cfg = config(tstart="2020-01-01", tend="2020-03-01", min_obs_per_interval=2)
    times = pd.to_datetime(["2020-01-05 20:00", "2020-01-06 20:00", "2020-02-05 20:00"])
    obs = pd.Series(
        [1.0, 2.0, 3.0], index=obs_space(["wbb"] * 3, times).select("concentration")
    )
    prior = prior_series()
    obs, prior = pipelines.filter_state_space(cfg, obs, prior)
    # February has one obs: dropped; its flux stays in the state, held by the prior
    assert obs.tolist() == [1.0, 2.0]
    assert len(prior) == len(prior_series())


def test_get_obs_passes_the_config_to_the_loader(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"CH4": [2.0, 2.5]}, index=OBS_INDEX[:2])

    monkeypatch.setattr(pipelines, "get_slv_observations", fake)
    cfg = InversionConfig(
        tstart="2020-01-01",
        tend="2020-03-01",
        sites=["wbb", "hdp"],
        subset_hours=[13, 14],
        utc_offset=-6,
        filter_pcaps=False,
        mobile_obs="receptor_obs.parquet",
    )
    obs = pipelines.get_obs(cfg)
    assert obs.tolist() == [2.0, 2.5] and obs.name == "obs"
    assert seen["sites"] == ["wbb", "hdp"]
    assert seen["subset_hours"] == [13, 14]
    assert seen["utc_offset"] == -6
    assert seen["filter_pcaps"] is False
    assert seen["mobile_obs"] == "receptor_obs.parquet"
    assert seen["time_range"] == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-03-01"),
    )


# ---------------------------------------------------------------------------
# Multiplicative MDM scales
# ---------------------------------------------------------------------------


def test_scale_on_prior_matches_by_label():
    J = jacobian()
    prior = prior_series()
    prior.index = flux_index()  # the Jacobian's own cells, in another level order
    obs = pd.Series(1.0, index=OBS_INDEX)
    cells = J.cols.select("flux")
    x = prior.reorder_levels(cells.names).reindex(cells).to_numpy()
    expected = np.abs(J.dense() @ x)
    positional = np.abs(J.dense() @ prior.to_numpy())
    assert not np.allclose(expected, positional)  # the orders really differ here

    got = mdm.multiplicative_scale(
        obs, "prior", prior=prior, jacobian=J, constant=pd.Series(dtype=float)
    )
    receptors = J.rows.select("concentration")
    on_obs = pd.Series(expected, index=receptors).reindex(OBS_INDEX).fillna(0.0)
    np.testing.assert_allclose(got, on_obs.to_numpy())


def test_scale_on_the_footprint_is_zero_without_one():
    J = jacobian()
    obs = pd.Series(1.0, index=OBS_INDEX)
    got = mdm.multiplicative_scale(
        obs, "footprint", prior=prior_series(), jacobian=J, constant=pd.Series()
    )
    rows = np.abs(J.dense()).sum(axis=1) * float(np.mean(prior_series().values))
    expected = pd.Series(rows, index=J.rows.select("concentration")).reindex(OBS_INDEX)
    np.testing.assert_allclose(got, expected.fillna(0.0).to_numpy())
    assert got[OBS_INDEX.get_loc(NO_FOOTPRINT)] == 0.0


def test_scale_on_obs_is_the_enhancement():
    obs = pd.Series(2.0, index=OBS_INDEX)
    constant = pd.Series(1.9, index=OBS_INDEX.drop(NO_BACKGROUND))
    got = mdm.multiplicative_scale(
        obs, "obs", prior=prior_series(), jacobian=jacobian(), constant=constant
    )
    expected = np.where(OBS_INDEX == NO_BACKGROUND, 0.0, 0.1)
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# get_flux_jacobian leaves the config alone
# ---------------------------------------------------------------------------


def test_auto_location_map_is_not_written_to_config(monkeypatch):
    import stilt

    import slv.inversion.config as config_module
    from slv.inversion.sweep import config_id

    seen = {}
    cfg = InversionConfig(tstart="2020-01-01", tend="2020-03-01")

    def fake_project(path):
        bins = cfg.flux_time_bins
        when = bins[0].left + pd.Timedelta(hours=cfg.subset_hours_utc[0])
        return SimpleNamespace(
            simulations=pd.DataFrame(
                {
                    "receptor": ["r0"],
                    "variant": "hrrr",
                    "time": [when],
                    "location": "loc_a",
                }
            ),
            variants={"hrrr": SimpleNamespace(footprint=object())},
        )

    def fake_project_jacobian(project, sel, target, bins, **kwargs):
        seen.update(kwargs)
        return jacobian()

    monkeypatch.setattr(stilt, "Project", fake_project)
    monkeypatch.setattr(pipelines, "project_jacobian", fake_project_jacobian)
    monkeypatch.setattr(
        config_module,
        "build_location_site_map",
        lambda ids, site_config: {"loc_a": "wbb"},
    )
    before = config_id(cfg)
    pipelines.get_flux_jacobian(cfg, pd.Series(1.0, index=OBS_INDEX))

    assert seen["location_mapper"] == {"loc_a": "wbb"}
    assert cfg.location_site_map == {}
    assert config_id(cfg) == before


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
TRAX_SPACE = Space({"concentration": TRAX_OBS_INDEX})


def trax_obs():
    """A tower obs and a TRAX receptor obs (keyed by its PYSTILT location_id)."""
    return pd.Series([2.0, 2.1], index=TRAX_OBS_INDEX, name="obs")


class TestMobileObsInPipeline:
    def test_obs_sites_resolves_receptors_to_the_mobile_site(self):
        cfg = InversionConfig(sites=["wbb", "trx01"])
        locations = TRAX_OBS_INDEX.get_level_values("obs_location")
        assert obs_sites(cfg, locations).tolist() == ["wbb", "trx01"]

    def test_obs_sites_without_a_mobile_site_raises(self):
        cfg = InversionConfig(sites=["wbb"])
        with pytest.raises(ValueError, match="0 mobile sites"):
            obs_sites(cfg, TRAX_OBS_INDEX.get_level_values("obs_location"))

    def test_default_mdm_builds_with_mobile_obs(self, monkeypatch):
        # the default `instr` term looked up each obs_location's organization: KeyError
        cfg = InversionConfig(sites=["wbb", "trx01"])
        zeros = lambda obs, *a, **k: np.zeros(len(obs))  # noqa: E731
        monkeypatch.setattr(mdm, "multiplicative_scale", zeros)
        monkeypatch.setattr(mdm, "per_obs_std", lambda config, obs, src: zeros(obs))
        obs = trax_obs()
        S = mdm.get_modeldata_mismatch(
            cfg,
            obs,
            prior=None,
            jacobian=None,
            constant=None,
            obs_sites=obs_sites(cfg, obs.index.get_level_values("obs_location")),
        )
        diag = S.variances.values
        assert len(diag) == 2
        assert (diag > 0).all()
        assert S.name == "modeldata_mismatch"

    def test_constant_keeps_mobile_obs(self, monkeypatch):
        # hourly background joined on exact obs_time left the 20:13 receptor NaN -> dropped
        from slv.inversion import background as bg

        hourly = pd.Series(
            [1.90, 1.95], index=pd.to_datetime(["2024-06-02 19:00", "2024-06-02 20:00"])
        )
        monkeypatch.setattr(bg, "get_rolling_background", lambda **kwargs: hourly)
        cfg = InversionConfig(sites=["wbb", "trx01"])
        constant = pipelines.get_constant(cfg, trax_obs())
        assert constant.index.get_level_values("obs_location").tolist() == [
            "wbb",
            "multi_aaaaaaaaaa",
        ]
        assert constant.tolist() == [1.95, 1.95]
        assert constant.name == "background"


def test_total_units_are_mass_per_interval():
    from slv.inversion.report import total_units

    assert total_units(InversionConfig(flux_freq="QS")) == "Gg per QS interval"


# ---------------------------------------------------------------------------
# Package releases in the component cache keys
# ---------------------------------------------------------------------------


def _checkout(root, name, pyproject):
    """An editable-style checkout: <root>/pyproject.toml and <root>/src/<name>/."""
    (root / "src" / name).mkdir(parents=True)
    (root / "src" / name / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text(pyproject)
    return root / "src"


class TestPkgVersion:
    def test_static_pyproject_version(self, tmp_path, monkeypatch):
        from slv.inversion.cache import _pkg_version

        src = _checkout(
            tmp_path, "slvtest_static_pkg", '[project]\nname = "x"\nversion = "1.2.3"\n'
        )
        monkeypatch.syspath_prepend(str(src))
        try:
            assert _pkg_version("slvtest_static_pkg", "x") == "1.2.3"
        finally:
            sys.modules.pop("slvtest_static_pkg", None)

    @pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
    def test_scm_version_is_the_latest_tag(self, tmp_path, monkeypatch):
        # a setuptools-scm checkout: the release is the latest tag, not the commit
        from slv.inversion.cache import _pkg_version

        src = _checkout(
            tmp_path,
            "slvtest_scm_pkg",
            '[project]\nname = "x"\ndynamic = ["version"]\n',
        )
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "init")
        _git(tmp_path, "tag", "v2026.12.3")
        (tmp_path / "later.txt").write_text("after the release")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-qm", "work after the release")
        monkeypatch.syspath_prepend(str(src))
        try:
            assert _pkg_version("slvtest_scm_pkg", "x") == "2026.12.3"
        finally:
            sys.modules.pop("slvtest_scm_pkg", None)
