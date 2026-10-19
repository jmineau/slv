"""Tests for the Jacobian coverage filter: which cells it keeps, and the problem it gives."""

import numpy as np
import pandas as pd
from fips import Covariance, Independent, Operator, Problem, Space, Vector

from slv.inversion.coverage import covered_state

TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
WEAK = (40.5, -112.0)  # (lat, lon) of the cell the obs barely see
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * 3, pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-05"])],
    names=["obs_location", "obs_time"],
)


def problem(bias=False):
    """Prior 1..8 over (time, lat, lon); unit Jacobian except the weak cell (1e-6)."""
    index = pd.MultiIndex.from_product(
        [TIMES, LATS, LONS], names=["time", "lat", "lon"]
    )
    flux = pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index)
    weak = (index.get_level_values("lat") == WEAK[0]) & (
        index.get_level_values("lon") == WEAK[1]
    )
    H = np.where(weak, 1e-6, 1.0)[None, :].repeat(len(OBS_INDEX), axis=0)
    blocks = {"flux": flux}
    forward = {("concentration", "flux"): pd.DataFrame(H, OBS_INDEX, index)}
    if bias:
        bias_index = pd.Index(TIMES, name="obs_time")
        blocks["bias"] = pd.Series(0.0, index=bias_index)
        forward[("concentration", "bias")] = pd.DataFrame(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], OBS_INDEX, bias_index
        )
    prior = Vector.from_blocks(blocks, name="prior")
    obs = Vector.from_series(pd.Series(2.5, index=OBS_INDEX), block="concentration")
    prior_cov = Covariance.block_diag(
        **{name: Independent().with_std(0.1) for name in blocks}
    )
    return Problem(
        prior=prior,
        prior_cov=prior_cov,
        obs=obs,
        obs_cov=Independent().with_std(0.01),
        forward=Operator.from_blocks(forward),
        offset=2.0,
    )


def cells(space: Space) -> set:
    """(lat, lon) pairs of the flux labels of a space."""
    flux = space.select("flux")
    return set(
        zip(flux.get_level_values("lat"), flux.get_level_values("lon"), strict=True)
    )


def test_weak_cell_is_left_out_at_every_time(capsys):
    p = problem()
    keep = covered_state(p, 30)
    assert cells(keep) == cells(p.prior.space) - {WEAK}
    assert len(keep.select("flux")) == (len(LATS) * len(LONS) - 1) * len(TIMES)
    assert (
        "Keeping 3/4 cells (removed 1 below 30th percentile" in capsys.readouterr().out
    )


def test_weak_cell_is_held_at_the_prior_in_the_offset():
    p = problem()
    small = p.restrict(covered_state(p, 30))
    # weak-cell prior is 1 (Jan) and 5 (Feb), seen with 1e-6 by every obs
    np.testing.assert_allclose(small.offset.values, 2.0 + 1e-6 * (1.0 + 5.0))
    assert cells(small.prior.space) == cells(small.prior_cov.rows)
    assert cells(small.forward.cols) == cells(small.prior.space)


def test_nothing_below_the_threshold_keeps_every_cell():
    p = problem()
    keep = covered_state(p, 0)  # threshold = the weakest cell
    assert keep == p.prior.space


def test_bias_block_is_kept():
    p = problem(bias=True)
    keep = covered_state(p, 30)
    assert keep.names == ("flux", "bias")
    assert keep.select("bias").equals(p.prior.space.select("bias"))
    small = p.restrict(keep)
    np.testing.assert_allclose(small.prior_cov["bias", "bias"].variances.values, 0.01)


def test_combine_first_puts_the_prior_back_in_removed_cells():
    import fips

    p = problem()
    post = fips.solve(p.restrict(covered_state(p, 30)))
    full = post.estimate.combine_first(p.prior).to_series().droplevel("block")
    prior = p.prior.to_series().droplevel("block")
    is_weak = (full.index.get_level_values("lat") == WEAK[0]) & (
        full.index.get_level_values("lon") == WEAK[1]
    )
    np.testing.assert_array_equal(full[is_weak], prior[is_weak])
    assert not np.allclose(full[~is_weak], prior[~is_weak])
