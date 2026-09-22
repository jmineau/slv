"""Tests for the Jacobian coverage filter and the posterior reconstruction."""

import numpy as np
import pandas as pd
from fips import Block, ForwardOperator, MatrixBlock, Vector

from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import SLVMethaneInversion

TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
WEAK = (40.5, -112.0)  # (lat, lon) of the cell the obs barely see
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * 3, pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-05"])],
    names=["obs_location", "obs_time"],
)


def pipeline(**kwargs):
    """A pipeline without fips' __init__ (no data access)."""
    obj = object.__new__(SLVMethaneInversion)
    obj.config = InversionConfig(tstart="2020-01-01", tend="2020-03-01", **kwargs)
    return obj


def inputs(bias=False):
    """Prior 1..8 over (time, lat, lon); unit Jacobian except the weak cell (1e-6)."""
    index = pd.MultiIndex.from_product(
        [TIMES, LATS, LONS], names=["time", "lat", "lon"]
    )
    flux = pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index)
    columns = pd.MultiIndex.from_product(
        [LONS, LATS, TIMES], names=["lon", "lat", "time"]
    )
    H = pd.DataFrame(1.0, index=OBS_INDEX, columns=columns)
    weak = (columns.get_level_values("lat") == WEAK[0]) & (
        columns.get_level_values("lon") == WEAK[1]
    )
    H.loc[:, weak] = 1e-6
    blocks = [MatrixBlock(H, row_block="concentration", col_block="flux")]
    prior_blocks = [Block(flux, name="flux")]
    if bias:
        bias_index = pd.Index(TIMES, name="time")
        Hb = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], OBS_INDEX, bias_index)
        blocks.append(MatrixBlock(Hb, row_block="concentration", col_block="bias"))
        prior_blocks.append(Block(pd.Series(0.0, index=bias_index), name="bias"))
    constant = pd.Series(2.0, index=OBS_INDEX)
    return flux, {
        "prior": Vector(name="prior", data=prior_blocks),
        "forward_operator": ForwardOperator(blocks),
        "constant": Vector(
            name="background", data=Block(name="concentration", data=constant)
        ),
        "prior_error": None,
    }


def cells(index):
    """(lat, lon) pairs of the flux entries of a state index."""
    if "block" in index.names:
        index = index[index.get_level_values("block") == "flux"]
    return set(
        zip(index.get_level_values("lat"), index.get_level_values("lon"), strict=True)
    )


def test_weak_cell_leaves_prior_jacobian_and_prior_error():
    p = pipeline(jacobian_coverage_percentile=30)
    flux, inp = inputs()
    out = p._apply_jacobian_coverage_filter(inp)

    retained = cells(flux.index) - {WEAK}
    assert p._removed_cells == {WEAK}
    assert p._retained_cells == retained
    assert p._all_cells == cells(flux.index)
    # removed at every time step, from all three
    assert cells(out["prior"].index) == retained
    assert cells(out["forward_operator"].columns) == retained
    assert cells(out["prior_error"].index) == retained
    assert len(out["prior"]["flux"]) == len(retained) * len(TIMES)
    # the kept cells keep their prior values
    kept = out["prior"]["flux"]
    pd.testing.assert_series_equal(
        kept, flux.loc[kept.index], check_names=False, check_freq=False
    )


def test_weak_cell_is_held_at_the_prior_in_the_constant():
    p = pipeline(jacobian_coverage_percentile=30)
    _, inp = inputs()
    out = p._apply_jacobian_coverage_filter(inp)
    # weak-cell prior is 1 (Jan) and 5 (Feb), seen with 1e-6 by every obs
    expected = 2.0 + 1e-6 * (1.0 + 5.0)
    np.testing.assert_allclose(out["constant"]["concentration"].to_numpy(), expected)


def test_nothing_below_the_threshold_returns_the_inputs():
    p = pipeline(jacobian_coverage_percentile=0)  # threshold = the weakest cell
    _, inp = inputs()
    out = p._apply_jacobian_coverage_filter(inp)
    assert out is inp
    assert p._all_cells is None and p._retained_cells is None


def test_bias_block_is_kept():
    p = pipeline(jacobian_coverage_percentile=30, bias_std=0.1)
    _, inp = inputs(bias=True)
    out = p._apply_jacobian_coverage_filter(inp)

    pd.testing.assert_series_equal(
        out["prior"]["bias"], inp["prior"]["bias"], check_names=False
    )
    bias_cols = out["forward_operator"].columns.get_level_values("block") == "bias"
    assert bias_cols.sum() == len(TIMES)
    bias_err = out["prior_error"].blocks["bias", "bias"].data
    np.testing.assert_allclose(np.diag(bias_err.to_numpy()), 0.1**2)
    assert cells(out["prior_error"].index) == cells(out["prior"].index)


def test_reconstruct_posterior_puts_the_prior_back_in_removed_cells():
    p = pipeline(jacobian_coverage_percentile=30)
    flux, inp = inputs()
    out = p._apply_jacobian_coverage_filter(inp)
    posterior = (out["prior"]["flux"] * 10).rename("posterior")

    full = p.reconstruct_posterior(posterior)

    assert full.name == "posterior"
    assert set(full.index) == set(flux.index)
    is_weak = (full.index.get_level_values("lat") == WEAK[0]) & (
        full.index.get_level_values("lon") == WEAK[1]
    )
    np.testing.assert_allclose(full[is_weak].to_numpy(), flux[is_weak].to_numpy())
    np.testing.assert_allclose(
        full[~is_weak].to_numpy(), (flux[~is_weak] * 10).to_numpy()
    )


def test_reconstruct_posterior_without_the_filter_passes_through():
    p = pipeline()
    posterior = pd.Series([1.0, 2.0], name="posterior")
    assert p.reconstruct_posterior(posterior) is posterior
