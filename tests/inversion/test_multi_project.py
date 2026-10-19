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
import scipy.sparse as sp
from fips import Operator, Space
from fips.linalg import SparseOperator
from stilt.footprint import Jacobian

import slv.inversion.pipelines as pl
from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import stack_jacobians

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
    return Operator(
        sp.csr_array(data) if sparse else data,
        Space({"concentration": idx}),
        Space({"flux": cols}),
        name="jacobian",
    )


def frame(H):
    """An operator of one block on each axis as a DataFrame on the blocks' labels."""
    df = H.to_frame()
    df.index = df.index.droplevel("block")
    df.columns = df.columns.droplevel("block")
    return df


def _config(**cfg):
    return InversionConfig(cache=False, **cfg)


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
    H = frame(stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)]))
    assert len(H) == 3
    assert set(H.index.get_level_values("obs_location")) == {
        "wbb",
        "multi_aaaaaaaaaa",
    }
    for part in (a, b):
        pd.testing.assert_frame_equal(
            H.loc[frame(part).index], frame(part), check_names=False
        )


def test_stack_aligns_columns_and_zero_fills():
    a = _block(["wbb"], TIMES[:1], cols=COLS[:4])
    b = _block(["multi_aaaaaaaaaa"], TIMES[:1], cols=COLS[4:])
    H = frame(stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)]))
    assert len(H.columns) == len(COLS)
    assert (H.loc[frame(a).index, COLS[4:]] == 0).all().all()
    assert (H.loc[frame(b).index, COLS[:4]] == 0).all().all()


def test_stack_rejects_an_obs_in_two_projects():
    a = _block(["wbb"], TIMES[:1])
    b = _block(["wbb"], TIMES[:1])
    with pytest.raises(ValueError, match="more than one STILT project") as e:
        stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)])
    assert "/prod" in str(e.value) and "/trax" in str(e.value)


def test_stack_stays_sparse():
    a = _block(["wbb"], TIMES[:1], value=0.0, sparse=True)
    b = _block(["multi_aaaaaaaaaa"], TIMES[:1], value=0.0, sparse=True)
    H = stack_jacobians([(Path("/prod"), a), (Path("/trax"), b)])
    assert isinstance(H.data, SparseOperator)


# --------------------------------------------------------------------------- pipeline


def test_flux_jacobian_stacks_every_contributing_project(monkeypatch):
    built = {
        Path("/prod"): _block(["wbb"], TIMES[:1]),
        Path("/trax"): _block(["multi_aaaaaaaaaa"], TIMES[:1]),
        Path("/empty"): None,  # a project with no sims matching the obs is skipped
    }
    monkeypatch.setattr(pl, "_project_jacobian", lambda config, p, locs: built[p])
    config = _config(stilt_project=["/prod", "/trax", "/empty"])
    H = pl.get_flux_jacobian(
        config, _obs(["wbb", "multi_aaaaaaaaaa"], TIMES[:1].repeat(2))
    )
    assert isinstance(H, Operator)
    rows = H.rows.select("concentration").get_level_values("obs_location")
    assert set(rows) == {"wbb", "multi_aaaaaaaaaa"}


def test_single_project_is_passed_through_unstacked(monkeypatch):
    blk = _block(["wbb"], TIMES[:1])
    monkeypatch.setattr(pl, "_project_jacobian", lambda config, p, locs: blk)
    monkeypatch.setattr(
        pl,
        "stack_jacobians",
        lambda *a, **k: pytest.fail("should not stack one project"),
    )
    H = pl.get_flux_jacobian(_config(stilt_project="/prod"), _obs(["wbb"], TIMES[:1]))
    assert H is blk


def test_no_project_matching_the_obs_raises(monkeypatch):
    monkeypatch.setattr(pl, "_project_jacobian", lambda config, p, locs: None)
    with pytest.raises(ValueError, match="None of the STILT projects"):
        pl.get_flux_jacobian(
            _config(stilt_project=["/prod", "/trax"]), _obs(["wbb"], TIMES[:1])
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
    p = _config(stilt_project="/trax", location_site_map={"x": "y"})
    H = pl._project_jacobian(p, Path("/trax"), {"multi_aaaaaaaaaa"})
    assert _FakeProject.calls[0]["locations"] == {"multi_aaaaaaaaaa"}
    assert H is not None


def test_project_with_no_matching_sims_returns_none(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_cccccccccc"], {"hrrr": True})},
    )
    p = _config(stilt_project="/prod", location_site_map={"x": "y"})
    assert pl._project_jacobian(p, Path("/prod"), {"wbb"}) is None
    assert _FakeProject.calls == []


def test_variant_none_picks_the_one_variant_with_a_footprint(monkeypatch):
    # the production project after migration: the base run and a particles-only error run
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_aaaaaaaaaa"], {"hrrr": True, "hrrr-err": False})},
    )
    p = _config(stilt_project="/prod", location_site_map={"x": "y"})
    pl._project_jacobian(p, Path("/prod"), {"multi_aaaaaaaaaa"})
    assert _FakeProject.calls[0]["variant"] == {"hrrr"}


def test_variant_none_with_several_footprint_variants_raises(monkeypatch):
    _fake_stilt(
        monkeypatch,
        {Path("/prod"): (["multi_aaaaaaaaaa"], {"hrrr": True, "hrrr-zi08": True})},
    )
    p = _config(stilt_project="/prod", location_site_map={"x": "y"})
    with pytest.raises(ValueError, match="set InversionConfig.variant"):
        pl._project_jacobian(p, Path("/prod"), {"multi_aaaaaaaaaa"})


def test_named_variant_missing_from_a_project_raises(monkeypatch):
    _fake_stilt(monkeypatch, {Path("/trax"): (["multi_aaaaaaaaaa"], {"hrrr": True})})
    p = _config(
        stilt_project="/trax", variant="hrrr-zi08", location_site_map={"x": "y"}
    )
    with pytest.raises(ValueError, match="not in the STILT project"):
        pl._project_jacobian(p, Path("/trax"), {"multi_aaaaaaaaaa"})


def test_project_jacobian_drops_sims_outside_the_window_or_hours(monkeypatch):
    _fake_stilt(monkeypatch, {Path("/prod"): (["loc_a"], {"hrrr": True})})
    p = _config(stilt_project="/prod", location_site_map={"loc_a": "wbb"})
    inside = _FakeProject(
        "/prod", ["loc_a"], {"hrrr": True}, pd.Timestamp("2000-01-01")
    ).simulations
    good = p.flux_time_bins[0].left + pd.Timedelta(hours=p.subset_hours_utc[0])
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
    H = pl._project_jacobian(p, Path("/prod"), {"wbb"})
    assert list(H.rows.select("concentration")) == [("wbb", good)]


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
    df = frame(H)
    assert list(df.index) == [("wbb", TIMES[0]), ("wbb", TIMES[1])]
    assert df.index.names == ["obs_location", "obs_time"]
    assert df.columns.names == ["lon", "lat", "time"]
    assert df.loc[("wbb", TIMES[0]), (-111.95, 40.65, TIMES[0])] == 1
    assert df.loc[("wbb", TIMES[0]), (-111.85, 40.65, TIMES[1])] == 2
    assert df.loc[("wbb", TIMES[1]), (-111.85, 40.65, TIMES[0])] == 3


def test_project_jacobian_stays_sparse():
    sel = pd.DataFrame(
        {"receptor": ["r0"], "variant": "hrrr", "time": TIMES[:1], "location": ["a"]}
    )
    J = _stilt_jacobian([[1, 0, 0, 0]], ["r0"])
    H = pl.project_jacobian(
        SimpleNamespace(jacobian=lambda *a, **k: J), sel, None, None
    )
    assert isinstance(H.data, SparseOperator)
    assert H.dense().sum() == 1


def test_project_jacobian_with_no_rows_is_none():
    sel = pd.DataFrame(
        {"receptor": ["r0"], "variant": "hrrr", "time": TIMES[:1], "location": ["a"]}
    )
    J = _stilt_jacobian([[0, 0, 0, 0]], ["r0"])
    project = SimpleNamespace(jacobian=lambda *a, **k: J)
    assert pl.project_jacobian(project, sel, None, None) is None
