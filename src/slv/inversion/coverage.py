"""State cells: the prior/Jacobian agreement check and the Jacobian coverage filter
for :class:`~slv.inversion.pipelines.SLVMethaneInversion`."""

from typing import Any

import numpy as np
import pandas as pd
from fips import Block, CovarianceMatrix, ForwardOperator, MatrixBlock, Vector
from fips.covariance import DiagonalError
from fips.problems.flux.problem import FluxProblem

from slv.inversion.config import InversionConfig
from slv.inversion.covariances import build_prior_error


def _flux_cells(index: pd.Index) -> set[tuple[float, float]]:
    """The (lon, lat) cells of the flux entries of a fips state index."""
    if "block" in index.names:
        index = index[index.get_level_values("block") == "flux"]
    lon = np.round(index.get_level_values("lon").to_numpy(dtype=float), 6)
    lat = np.round(index.get_level_values("lat").to_numpy(dtype=float), 6)
    return set(zip(lon, lat, strict=True))


def check_state_cells(prior: Vector, forward_operator: ForwardOperator) -> None:
    """Raise when the prior's flux cells and the Jacobian's flux columns differ.

    fips reindexes the Jacobian onto the prior's index with zero fill, so a cell only
    in the prior would silently get zero sensitivity, and a cell only in the Jacobian
    would silently lose its contribution. The usual cause is a cached prior or
    Jacobian built for a different state grid.
    """
    prior_cells = _flux_cells(prior.index)
    jac_cells = _flux_cells(forward_operator.columns)
    if prior_cells and prior_cells != jac_cells:
        raise ValueError(
            f"State cells disagree: {len(prior_cells - jac_cells)} prior cells have no "
            f"Jacobian column and {len(jac_cells - prior_cells)} Jacobian cells are not in "
            "the prior. A cached prior or forward_operator was probably built for a "
            "different state grid; rebuild with "
            'cache_overwrite=["prior", "prior_error", "forward_operator", '
            '"modeldata_mismatch"].'
        )


class CoverageFilterMixin:
    """Drops poorly-constrained cells from the state (``jacobian_coverage_percentile``),
    holding them at the prior, and reinstates them in the reported posterior."""

    config: InversionConfig
    problem: FluxProblem

    @property
    def _all_cells(self) -> set | None:
        """Return the set of all (lat, lon) cells, or None if no filter applied."""
        if not hasattr(self, "_full_prior"):
            return None
        full_flux = self._full_prior["flux"]
        lats = full_flux.index.get_level_values("lat")
        lons = full_flux.index.get_level_values("lon")
        return set(zip(lats, lons, strict=False))

    @property
    def _retained_cells(self) -> set | None:
        """Return the set of retained (lat, lon) cells, or None if no filter applied."""
        if self._all_cells is None:
            return None
        return self._all_cells - self._removed_cells

    def _apply_jacobian_coverage_filter(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Remove cells with insufficient Jacobian coverage from the state vector.

        Cells are removed across ALL time steps to preserve the Kronecker
        structure of the prior error covariance.  Removed cells are stored
        so they can be reinstated at the prior value via ``reconstruct_posterior``.
        """
        percentile = self.config.jacobian_coverage_percentile
        prior = inputs["prior"]
        forward_operator = inputs["forward_operator"]
        has_bias = self.config.bias_std is not None

        # --- Compute mean per-observation-location sensitivity per cell ---
        flux_blk = forward_operator.blocks["concentration", "flux"]
        flux_jac = flux_blk.data  # (obs_time, obs_location) × (lat, lon, time)

        n_obs_locations = flux_jac.index.get_level_values("obs_location").nunique()

        col_abs_sums = flux_jac.abs().sum(axis=0)
        lat_vals = col_abs_sums.index.get_level_values("lat")
        lon_vals = col_abs_sums.index.get_level_values("lon")
        cell_total = col_abs_sums.groupby([lat_vals, lon_vals]).sum()
        cell_coverage = cell_total / n_obs_locations

        threshold = np.percentile(cell_coverage.values, percentile)
        retained_cells = set(cell_coverage[cell_coverage >= threshold].index.tolist())
        removed_cells = set(cell_coverage[cell_coverage < threshold].index.tolist())

        n_total = len(cell_coverage)
        n_removed = len(removed_cells)
        print(
            f"  Keeping {n_total - n_removed}/{n_total} cells "
            f"(removed {n_removed} below {percentile}th percentile, "
            f"threshold={threshold:.2e})"
        )

        if n_removed == 0:
            return inputs

        # Store for posterior reconstruction
        self._full_prior = prior
        self._removed_cells = removed_cells

        # --- Helper: boolean mask for a MultiIndex with lat/lon levels ---
        def _cell_mask(index):
            lats = index.get_level_values("lat")
            lons = index.get_level_values("lon")
            return pd.array(
                [
                    (lat, lon) in retained_cells
                    for lat, lon in zip(lats, lons, strict=False)
                ],
                dtype=bool,
            )

        # --- Filter prior ---
        flux_series = prior["flux"]
        filtered_flux = flux_series[_cell_mask(flux_series.index)]

        if has_bias:
            filtered_prior = Vector(
                name=prior.name,
                data=[Block(filtered_flux, name="flux"), prior.blocks["bias"]],
            )
        else:
            filtered_prior = Vector(
                name=prior.name, data=Block(name="flux", data=filtered_flux)
            )

        # --- Filter forward operator ---
        jac_mask = _cell_mask(flux_jac.columns)
        removed_jac_mask = ~jac_mask
        filtered_flux_jac = flux_jac.loc[:, jac_mask]
        filtered_flux_blk = MatrixBlock(
            filtered_flux_jac,
            row_block="concentration",
            col_block="flux",
            sparse=self.config.sparse_jacobian,
        )

        if has_bias:
            bias_blk = forward_operator.blocks["concentration", "bias"]
            filtered_fo = ForwardOperator([filtered_flux_blk, bias_blk])
        else:
            filtered_fo = ForwardOperator(filtered_flux_blk)

        # --- Add removed cells' contribution to the constant ---
        removed_flux = flux_series[~_cell_mask(flux_series.index)]
        removed_flux_jac = flux_jac.loc[:, removed_jac_mask]
        removed_fo = ForwardOperator(
            MatrixBlock(
                removed_flux_jac,
                row_block="concentration",
                col_block="flux",
                sparse=self.config.sparse_jacobian,
            )
        )
        removed_prior = Vector(
            name="removed_prior",
            data=Block(name="flux", data=removed_flux),
        )
        removed_contribution = removed_fo.convolve(removed_prior)
        self._removed_contribution = removed_contribution

        constant = inputs["constant"]
        constant_data = constant["concentration"]
        # Jacobian rows are simulations, constant rows are obs: match them by label (they
        # only line up by position once aggregate_obs_space has aggregated both).
        removed = removed_contribution
        if "block" in removed.index.names:
            removed = removed.droplevel("block")
        removed = removed.reindex(constant_data.index, fill_value=0.0)
        updated_constant_data = constant_data + removed.to_numpy()
        updated_constant = Vector(
            name=constant.name,
            data=Block(name="concentration", data=updated_constant_data),
        )

        # --- Rebuild prior error for the smaller grid ---
        flux_prior_vec = Vector(filtered_prior.blocks["flux"])
        S_0 = build_prior_error(
            flux_prior_vec,
            base_std=self.config.prior_base_std,
            std_frac=self.config.prior_std_frac,
            time_scale=self.config.prior_time_scale,
            spatial_scale=self.config.prior_spatial_scale,
        )

        if not has_bias:
            prior_error = CovarianceMatrix(name="prior_error", data=S_0)
        else:
            flux_err_blk = CovarianceMatrix(name="prior_error", data=S_0).blocks[
                "flux", "flux"
            ]
            bias_index = filtered_prior["bias"].index
            bias_err = DiagonalError(
                name="bias_error",
                variances=self.config.bias_std**2,  # pyright: ignore[reportOptionalOperand]
            ).build(bias_index)
            bias_err_blk = MatrixBlock(bias_err, "bias", "bias")
            prior_error = CovarianceMatrix(
                name="prior_error", data=[flux_err_blk, bias_err_blk]
            )

        return {
            **inputs,
            "prior": filtered_prior,
            "forward_operator": filtered_fo,
            "prior_error": prior_error,
            "constant": updated_constant,
        }

    def reconstruct_posterior(
        self, posterior_fluxes: pd.Series | None = None
    ) -> pd.Series:
        """Reconstruct the full posterior by inserting prior for unconstrained cells.

        Parameters
        ----------
        posterior_fluxes : pd.Series, optional
            Posterior flux series from the inversion.  If None, uses
            ``self.problem.posterior_fluxes``.

        Returns
        -------
        pd.Series
            Full posterior with constrained cells from the inversion and
            unconstrained cells filled with prior values.
        """
        if posterior_fluxes is None:
            posterior_fluxes = self.problem.posterior_fluxes

        if not hasattr(self, "_full_prior") or not hasattr(self, "_removed_cells"):
            return posterior_fluxes

        # Get the full prior flux series
        full_flux = self._full_prior["flux"]

        # Start with prior, then overwrite constrained cells
        full_posterior = full_flux.astype(float).copy()
        full_posterior.name = posterior_fluxes.name
        full_posterior.loc[posterior_fluxes.index] = posterior_fluxes.values

        return full_posterior
