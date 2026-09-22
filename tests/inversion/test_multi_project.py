"""Tests for building one flux Jacobian from several PYSTILT projects.

The UOU/DAQ footprints live in the production project and the TRAX ones in their own, so a
joint inversion stacks the rows each project contributes.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fips import ForwardOperator, MatrixBlock

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


class _FakeBuilder:
    """Records what the per-project build asked for."""

    calls: list = []

    def __init__(self, model):
        self.model = model

    def build_from_target(self, target, **kw):
        _FakeBuilder.calls.append({"project": self.model.path, **kw})
        return _block(sorted(kw["location_ids"]), [TIMES[0]] * len(kw["location_ids"]))


def _fake_stilt(monkeypatch, projects):
    """``projects``: path -> (sim ids, footprint xres by name)."""
    import stilt

    def model(path):
        sims, feet = projects[Path(path)]
        return SimpleNamespace(
            path=Path(path),
            simulations=sims,
            config=SimpleNamespace(
                footprints={
                    n: SimpleNamespace(grid=SimpleNamespace(xres=x))
                    for n, x in feet.items()
                }
            ),
        )

    monkeypatch.setattr(stilt, "Model", model)
    monkeypatch.setattr(pl, "JacobianBuilder", _FakeBuilder)
    _FakeBuilder.calls = []


def test_project_jacobian_keeps_only_sims_matching_the_obs(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {
            Path("/trax"): (
                [
                    "hrrr_202406012000_multi_aaaaaaaaaa",
                    "hrrr_202406012100_multi_bbbbbbbbbb",
                ],
                {"0.01": 0.01},
            ),
        },
    )
    p = _pipeline(stilt_project="/trax", location_site_map={"x": "y"})
    H = p._project_jacobian(Path("/trax"), {"multi_aaaaaaaaaa"})
    assert _FakeBuilder.calls[0]["location_ids"] == {"multi_aaaaaaaaaa"}
    assert H is not None


def test_project_with_no_matching_sims_returns_none(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["hrrr_202406012000_multi_cccccccccc"], {"0.01": 0.01})},
    )
    p = _pipeline(stilt_project="/prod", location_site_map={"x": "y"})
    assert p._project_jacobian(Path("/prod"), {"wbb"}) is None
    assert _FakeBuilder.calls == []


def test_footprint_none_picks_each_projects_finest(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {
            Path("/prod"): (
                ["hrrr_202406012000_multi_aaaaaaaaaa"],
                {"0.1": 0.1, "0.01": 0.01, "0.05": 0.05},
            ),
        },
    )
    p = _pipeline(stilt_project="/prod", location_site_map={"x": "y"})
    p._project_jacobian(Path("/prod"), {"multi_aaaaaaaaaa"})
    assert _FakeBuilder.calls[0]["footprint"] == "0.01"


def test_named_footprint_missing_from_a_project_raises(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/trax"): (["hrrr_202406012000_multi_aaaaaaaaaa"], {"0.01": 0.01})},
    )
    p = _pipeline(stilt_project="/trax", footprint="0.05", location_site_map={"x": "y"})
    with pytest.raises(ValueError, match="not in the STILT project"):
        p._project_jacobian(Path("/trax"), {"multi_aaaaaaaaaa"})
