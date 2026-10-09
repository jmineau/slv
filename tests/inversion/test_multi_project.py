"""
Tests for building one flux Jacobian from several PYSTILT projects.

The UOU/DAQ footprints live in the production project and the TRAX ones in their own, so a
joint inversion stacks the rows each project contributes.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fips import ForwardOperator, MatrixBlock
from stilt.footprint import Jacobian

import slv.inversion.pipelines as pl
from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import SLVMethaneInversion, stack_jacobians

TIMES = pd.to_datetime(["2024-06-01", "2024-07-01"])
COLS = pd.MultiIndex.from_product(
    [[-111.95, -111.85], [40.65, 40.75], TIMES], names=["lon", "lat", "time"]
)


def _block(locations, times, cols=COLS, value=None, sparse=False):
    idx = pd.MultiIndex.from_arrays(
        [locations, times], names=["obs_location", "obs_time"]
    )
    rng = np.random.default_rng(len(locations))
    data = (
        np.full((len(idx), len(cols)), value)
        if value is not None
        else rng.random((len(idx), len(cols)))
    )
    return MatrixBlock(
        pd.DataFrame(data, idx, cols),
        name="jacobian",
        row_block="concentration",
        col_block="flux",
        sparse=sparse,
    )


def _pipeline(**cfg):
    obj = object.__new__(SLVMethaneInversion)
    obj.config = InversionConfig(cache=False, **cfg)
    return obj


def _obs(locations, times):
    idx = pd.MultiIndex.from_arrays(
        [locations, pd.to_datetime(times)], names=["obs_location", "obs_time"]
    )
    return pd.Series(2.0, index=idx)


# --------------------------------------------------------------------------- config


def test_stilt_projects_normalises_one_or_many():
    assert InversionConfig(stilt_project="/a").stilt_projects == [Path("/a")]
    assert InversionConfig(stilt_project=["/a", Path("/b")]).stilt_projects == [
        Path("/a"),
        Path("/b"),
    ]


def test_stilt_project_left_as_given_so_single_project_cache_key_is_unchanged():
    assert InversionConfig(stilt_project="/a").stilt_project == "/a"


def test_empty_project_list_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        _ = InversionConfig(stilt_project=[]).stilt_projects


def test_adding_a_project_changes_the_forward_operator_cache_key():
    from slv.inversion.cache import DEFAULT_COMPONENT_DEPS, _component_hash

    deps = DEFAULT_COMPONENT_DEPS["forward_operator"]
    one = _component_hash(InversionConfig(stilt_project="/prod"), deps)
    two = _component_hash(InversionConfig(stilt_project=["/prod", "/trax"]), deps)
    assert one != two


# --------------------------------------------------------------------------- stacking


def test_stack_concatenates_disjoint_rows():
    a = _block(["wbb", "wbb"], TIMES)
    b = _block(["multi_aaaaaaaaaa"], TIMES[:1])
    H = stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)], sparse=False)
    assert len(H.data) == 3
    assert set(H.data.index.get_level_values("obs_location")) == {
        "wbb",
        "multi_aaaaaaaaaa",
    }
    pd.testing.assert_frame_equal(H.data.loc[a.data.index], a.data, check_names=False)
    pd.testing.assert_frame_equal(H.data.loc[b.data.index], b.data, check_names=False)


def test_stack_aligns_columns_and_zero_fills():
    a = _block(["wbb"], TIMES[:1], cols=COLS[:4])
    b = _block(["multi_aaaaaaaaaa"], TIMES[:1], cols=COLS[4:])
    H = stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)], sparse=False)
    assert len(H.data.columns) == len(COLS)
    assert (H.data.loc[a.data.index, COLS[4:]] == 0).all().all()
    assert (H.data.loc[b.data.index, COLS[:4]] == 0).all().all()


def test_stack_rejects_an_obs_in_two_projects():
    a = _block(["wbb"], TIMES[:1])
    b = _block(["wbb"], TIMES[:1])
    with pytest.raises(ValueError, match="more than one STILT project") as e:
        stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)])
    assert "/prod" in str(e.value) and "/trax" in str(e.value)


def test_stack_stays_sparse():
    a = _block(["wbb"], TIMES[:1], value=0.0, sparse=True)
    b = _block(["multi_aaaaaaaaaa"], TIMES[:1], value=0.0, sparse=True)
    H = stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)], sparse=True)
    assert H.is_sparse


# --------------------------------------------------------------------------- pipeline


def test_flux_jacobian_stacks_every_contributing_project(monkeypatch):
    built = {
        Path("/prod"): _block(["wbb"], TIMES[:1]),
        Path("/trax"): _block(["multi_aaaaaaaaaa"], TIMES[:1]),
        Path("/empty"): None,  # a project with no sims matching the obs is skipped
    }
    monkeypatch.setattr(
        SLVMethaneInversion, "_project_jacobian", lambda self, p, locs: built[p]
    )
    p = _pipeline(stilt_project=["/prod", "/trax", "/empty"])
    H = p._get_flux_jacobian(_obs(["wbb", "multi_aaaaaaaaaa"], TIMES[:1].repeat(2)))
    assert isinstance(H, ForwardOperator)
    rows = H.blocks["concentration", "flux"].data.index.get_level_values("obs_location")
    assert set(rows) == {"wbb", "multi_aaaaaaaaaa"}


def test_single_project_is_passed_through_unstacked(monkeypatch):
    blk = _block(["wbb"], TIMES[:1])
    monkeypatch.setattr(
        SLVMethaneInversion, "_project_jacobian", lambda self, p, locs: blk
    )
    monkeypatch.setattr(
        pl,
        "stack_jacobians",
        lambda *a, **k: pytest.fail("should not stack one project"),
    )
    H = _pipeline(stilt_project="/prod")._get_flux_jacobian(_obs(["wbb"], TIMES[:1]))
    pd.testing.assert_frame_equal(
        H.blocks["concentration", "flux"].data, blk.data, check_names=False
    )


def test_no_project_matching_the_obs_raises(monkeypatch):
    monkeypatch.setattr(
        SLVMethaneInversion, "_project_jacobian", lambda self, p, locs: None
    )
    with pytest.raises(ValueError, match="None of the STILT projects"):
        _pipeline(stilt_project=["/prod", "/trax"])._get_flux_jacobian(
            _obs(["wbb"], TIMES[:1])
        )


# --------------------------------------------------------------------------- per project


class _FakeProject:
    """A ``stilt.Project`` that records what the per-project build asked for."""

    calls: list = []

    def __init__(self, path, locations, variants, when):
        self.path = Path(path)
        self.simulations = pd.DataFrame(
            {
                "receptor": [f"r{i}" for i in range(len(locations))] * len(variants),
                "variant": [n for n in variants for _ in locations],
                "time": [when] * len(locations) * len(variants),
                "location": list(locations) * len(variants),
            }
        )
        self.variants = {
            n: SimpleNamespace(footprint=object() if has else None)
            for n, has in variants.items()
        }

    def jacobian(self, sel, target, time_bins, workers=None):
        from scipy import sparse

        _FakeProject.calls.append(
            {
                "project": self.path,
                "variant": set(sel["variant"]),
                "locations": set(sel["location"]),
            }
        )
        columns = pd.MultiIndex.from_product(
            [time_bins.left, [-111.95], [40.65]], names=["time", "lon", "lat"]
        )
        data = sparse.csr_matrix(np.ones((len(sel), len(columns))))
        return Jacobian(data, pd.Index(sel["receptor"]), columns, [], [])


def _fake_stilt(monkeypatch, projects):
    """``projects``: path -> (receptor location ids, {variant: has a footprint})."""
    import stilt

    cfg = InversionConfig(cache=False)
    when = cfg.flux_time_bins[0].left + pd.Timedelta(hours=cfg.subset_hours_utc[0])

    def project(path):
        locations, variants = projects[Path(path)]
        return _FakeProject(path, locations, variants, when)

    monkeypatch.setattr(stilt, "Project", project)
    _FakeProject.calls = []


def test_project_jacobian_keeps_only_sims_matching_the_obs(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {
            Path("/trax"): (
                ["multi_aaaaaaaaaa", "multi_bbbbbbbbbb"],
                {"hrrr": True},
            ),
        },
    )
    p = _pipeline(stilt_project="/trax", location_site_map={"x": "y"})
    H = p._project_jacobian(Path("/trax"), {"multi_aaaaaaaaaa"})
    assert _FakeProject.calls[0]["locations"] == {"multi_aaaaaaaaaa"}
    assert H is not None


def test_project_with_no_matching_sims_returns_none(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_cccccccccc"], {"hrrr": True})},
    )
    p = _pipeline(stilt_project="/prod", location_site_map={"x": "y"})
    assert p._project_jacobian(Path("/prod"), {"wbb"}) is None
    assert _FakeProject.calls == []


def test_variant_none_picks_the_one_variant_with_a_footprint(monkeypatch):
    # the production project after migration: the base run and a particles-only error run
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_aaaaaaaaaa"], {"hrrr": True, "hrrr-err": False})},
    )
    p = _pipeline(stilt_project="/prod", location_site_map={"x": "y"})
    p._project_jacobian(Path("/prod"), {"multi_aaaaaaaaaa"})
    assert _FakeProject.calls[0]["variant"] == {"hrrr"}


def test_variant_none_with_several_footprint_variants_raises(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_aaaaaaaaaa"], {"hrrr": True, "hrrr-zi08": True})},
    )
    p = _pipeline(stilt_project="/prod", location_site_map={"x": "y"})
    with pytest.raises(ValueError, match="set InversionConfig.variant"):
        p._project_jacobian(Path("/prod"), {"multi_aaaaaaaaaa"})


def test_named_variant_missing_from_a_project_raises(monkeypatch):
    _fake_stilt(monkeypatch, {Path("/trax"): (["multi_aaaaaaaaaa"], {"hrrr": True})})
    p = _pipeline(
        stilt_project="/trax", variant="hrrr-zi08", location_site_map={"x": "y"}
    )
    with pytest.raises(ValueError, match="not in the STILT project"):
        p._project_jacobian(Path("/trax"), {"multi_aaaaaaaaaa"})


def test_project_jacobian_drops_sims_outside_the_window_or_hours(monkeypatch):
    _fake_stilt(monkeypatch, {Path("/prod"): (["loc_a"], {"hrrr": True})})
    p = _pipeline(stilt_project="/prod", location_site_map={"loc_a": "wbb"})
    inside = _FakeProject(
        "/prod", ["loc_a"], {"hrrr": True}, pd.Timestamp("2000-01-01")
    ).simulations
    good = p.config.flux_time_bins[0].left + pd.Timedelta(
        hours=p.config.subset_hours_utc[0]
    )
    sims = pd.concat(
        [
            inside.assign(receptor="early"),  # before the flux window
            inside.assign(receptor="night", time=good.floor("D")),  # hour not kept
            inside.assign(receptor="kept", time=good),
        ]
    )

    def project(path):
        fake = _FakeProject(path, [], {"hrrr": True}, good)
        fake.simulations = sims
        return fake

    import stilt

    monkeypatch.setattr(stilt, "Project", project)
    H = p._project_jacobian(Path("/prod"), {"wbb"})
    assert list(H.data.index) == [("wbb", good)]


# --------------------------------------------------------------------------- project_jacobian


def _stilt_jacobian(data, receptors):
    from scipy import sparse

    columns = pd.MultiIndex.from_product(
        [TIMES, [-111.95, -111.85], [40.65]], names=["time", "lon", "lat"]
    )
    return Jacobian(
        sparse.csr_matrix(np.asarray(data, dtype=float)),
        pd.Index(receptors, name="receptor"),
        columns,
        [],
        [],
    )


def test_project_jacobian_labels_rows_by_location_and_time():
    sel = pd.DataFrame(
        {
            "receptor": ["r0", "r1", "r2"],
            "variant": "hrrr",
            "time": TIMES[[0, 0, 1]],
            "location": ["loc_a", "multi_aaaaaaaaaa", "loc_a"],
        }
    )
    J = _stilt_jacobian(
        [[1, 0, 0, 2], [0, 1e-20, 0, 0], [0, 3, 0, 0]], ["r0", "r1", "r2"]
    )
    project = SimpleNamespace(jacobian=lambda *a, **k: J)
    H = pl.project_jacobian(
        project, sel, None, None, location_mapper={"loc_a": "wbb"}, sparse=False
    )
    # r1 is below the threshold everywhere, so it has no row
    assert list(H.data.index) == [("wbb", TIMES[0]), ("wbb", TIMES[1])]
    assert H.data.index.names == ["obs_location", "obs_time"]
    assert H.data.columns.names == ["lon", "lat", "time"]
    assert H.data.loc[("wbb", TIMES[0]), (-111.95, 40.65, TIMES[0])] == 1
    assert H.data.loc[("wbb", TIMES[0]), (-111.85, 40.65, TIMES[1])] == 2
    assert H.data.loc[("wbb", TIMES[1]), (-111.85, 40.65, TIMES[0])] == 3


def test_project_jacobian_stays_sparse():
    sel = pd.DataFrame(
        {"receptor": ["r0"], "variant": "hrrr", "time": TIMES[:1], "location": ["a"]}
    )
    J = _stilt_jacobian([[1, 0, 0, 0]], ["r0"])
    H = pl.project_jacobian(
        SimpleNamespace(jacobian=lambda *a, **k: J), sel, None, None
    )
    assert H.is_sparse
    assert H.data.sum().sum() == 1


def test_project_jacobian_with_no_rows_is_none():
    sel = pd.DataFrame(
        {"receptor": ["r0"], "variant": "hrrr", "time": TIMES[:1], "location": ["a"]}
    )
    J = _stilt_jacobian([[0, 0, 0, 0]], ["r0"])
    project = SimpleNamespace(jacobian=lambda *a, **k: J)
    assert pl.project_jacobian(project, sel, None, None) is None
