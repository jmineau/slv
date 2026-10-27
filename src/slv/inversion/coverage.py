"""
State cells: the prior/Jacobian agreement check and the Jacobian coverage filter.

The filter only chooses the cells to keep; :meth:`fips.Problem.restrict` holds the others
at their prior.
"""

import numpy as np
import pandas as pd
from fips import Problem, Space


def _flux_cells(index: pd.Index) -> set[tuple[float, float]]:
    """The (lon, lat) cells of the flux entries of a state index."""
    if "block" in index.names:
        index = index[index.get_level_values("block") == "flux"]
    lon = np.round(index.get_level_values("lon").to_numpy(dtype=float), 6)
    lat = np.round(index.get_level_values("lat").to_numpy(dtype=float), 6)
    return set(zip(lon, lat, strict=True))


def check_state_cells(prior: pd.Index, jacobian: pd.Index) -> None:
    """
    Raise when the prior's flux cells and the Jacobian's flux columns differ.

    The Jacobian is placed on the prior's labels, a cell it lacks getting a column of
    zeros, so a cell only in the prior would silently get zero sensitivity (a cell only
    in the Jacobian makes the placing raise). The usual cause is a cached prior or
    Jacobian built for a different state grid.

    Parameters
    ----------
    prior, jacobian : pd.Index
        The prior's labels and the Jacobian's columns, with ``lat`` and ``lon`` levels.
        With a ``block`` level, only the ``flux`` labels are compared.
    """
    prior_cells = _flux_cells(prior)
    jac_cells = _flux_cells(jacobian)
    if prior_cells and prior_cells != jac_cells:
        raise ValueError(
            f"State cells disagree: {len(prior_cells - jac_cells)} prior cells have no "
            f"Jacobian column and {len(jac_cells - prior_cells)} Jacobian cells are not in "
            "the prior. A cached prior or forward_operator was probably built for a "
            "different state grid; rebuild with "
            'cache_overwrite=["prior", "prior_error", "forward_operator", '
            '"modeldata_mismatch"].'
        )


def covered_state(problem: Problem, percentile: float) -> Space:
    """
    The state without its poorly covered cells: those below ``percentile`` of coverage.

    A cell's coverage is its absolute sensitivity, summed over every obs and time, per
    obs location (divided by their number). Cells below the ``percentile``-th
    percentile of all cells' are left out at every time, which keeps the prior error's
    Kronecker structure; every other block of the state is kept whole. The problem of
    the kept state is ``problem.restrict(keep)``, which holds the left-out cells at
    their prior (their modeled enhancement moves into the offset);
    ``post.estimate.combine_first(problem.prior)`` puts the estimate back on every cell.

    Parameters
    ----------
    problem : Problem
        The problem, with a ``flux`` block in its state and ``concentration`` in its obs.
    percentile : float
        The coverage percentile below which a cell is left out (10 drops the
        least-sensitive 10%).

    Returns
    -------
    Space
        The state to keep.
    """
    flux = problem.prior.space.select("flux")
    H = problem.forward["concentration", "flux"]
    column_sums = np.asarray(abs(H.sparse()).sum(axis=0)).ravel()
    cells = pd.MultiIndex.from_arrays(
        [flux.get_level_values("lat"), flux.get_level_values("lon")]
    )
    obs = problem.obs.space.select("concentration")
    n_obs_locations = obs.get_level_values("obs_location").nunique()
    coverage = pd.Series(column_sums, index=cells).groupby(level=[0, 1]).sum()
    coverage = coverage / n_obs_locations

    threshold = np.percentile(coverage.to_numpy(), percentile)
    retained = coverage.index[coverage >= threshold]
    n_total, n_removed = len(coverage), int((coverage < threshold).sum())
    print(
        f"  Keeping {n_total - n_removed}/{n_total} cells "
        f"(removed {n_removed} below {percentile}th percentile, "
        f"threshold={threshold:.2e})"
    )
    blocks = {
        name: problem.prior.space.select(name) for name in problem.prior.space.names
    }
    blocks["flux"] = flux[cells.isin(retained)]
    return Space(blocks)
