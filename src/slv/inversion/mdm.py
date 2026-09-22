"""Model-data mismatch (S_z) for :class:`~slv.inversion.pipelines.SLVMethaneInversion`."""

import numpy as np
import pandas as pd
from fips import CovarianceMatrix, Vector
from fips.covariance import CovarianceBuilder

from slv.inversion import viz
from slv.inversion.cache import fips_cache
from slv.inversion.config import InversionConfig
from slv.inversion.covariances import build_mdm_error
from slv.inversion.data import get_slv_subhour_std


def _matrix(block: pd.DataFrame):
    """A Jacobian block as a matrix without densifying it: scipy CSR if it is sparse."""
    if len(block.columns) and all(isinstance(t, pd.SparseDtype) for t in block.dtypes):
        return block.sparse.to_coo().tocsr()
    return block.to_numpy(dtype=float)


class ModelDataMismatchMixin:
    """Builds the model-data mismatch covariance from ``config.mdm_components``.

    Multiplicative terms scale with the run's prior and flux Jacobian; per-obs terms
    with data-derived stds. Expects the host pipeline's ``get_prior``,
    ``_get_flux_jacobian``, ``get_constant`` and ``obs_sites``.
    """

    config: InversionConfig

    def _multiplicative_scale(self, obs: Vector, scale_on: str) -> np.ndarray:
        """Per-obs scale [ppm] for a multiplicative MDM term.

        ``scale_on="obs"``   -> |obs - background| (observed enhancement; the actual signal,
        so extreme shallow-PBL days the prior under-predicts still get a large error).
        ``scale_on="prior"`` -> |H x_prior| (prior-modeled enhancement; under-scales where
        the EPA prior is low -- exactly the over-leveraged days). Obs lacking the needed
        term get 0 (floor-only error).
        ``scale_on="footprint"`` -> |H| * mean(x_prior): footprint strength in enhancement
        units. Down-weights extreme shallow-PBL days by how much air they integrate,
        independent of the obs (no circularity) or the EPA pattern -- so it eases both the
        low-obs winter dips and the high-obs spikes, which share an extreme |H|.
        """
        if scale_on == "footprint":
            prior = self.get_prior()
            flux = self._flux_block(obs)
            xref = float(np.mean(prior["flux"].values))
            row_abs = np.asarray(abs(_matrix(flux)).sum(axis=1)).ravel()
            s = pd.Series(row_abs * xref, index=flux.index)
            key = obs.index.droplevel(
                [n for n in obs.index.names if n not in flux.index.names]
            )
            return s.reindex(key).fillna(0.0).to_numpy()
        if scale_on == "obs":
            bg = self.get_constant(obs)["concentration"]
            obs_s = obs.to_series()
            key = obs_s.index.droplevel(
                [n for n in obs_s.index.names if n not in bg.index.names]
            )
            bg_a = (
                pd.Series(np.asarray(bg.values, dtype=float), index=bg.index)
                .reindex(key)
                .to_numpy()
            )
            return np.abs(np.nan_to_num(obs_s.to_numpy() - bg_a, nan=0.0))
        prior = self.get_prior()
        flux = self._flux_block(obs)
        # Match the prior to the Jacobian columns by label, not position: their level
        # orders differ (the EPA prior is (time, lat, lon), the Jacobian (lon, lat, time)).
        x = (
            prior["flux"]
            .reorder_levels(flux.columns.names)
            .reindex(flux.columns, fill_value=0.0)
        )
        e = pd.Series(_matrix(flux) @ x.to_numpy(dtype=float), index=flux.index)
        key = obs.index.droplevel(
            [n for n in obs.index.names if n not in flux.index.names]
        )
        return np.abs(e.reindex(key).fillna(0.0).to_numpy())

    def _flux_block(self, obs: Vector) -> pd.DataFrame:
        """The flux Jacobian block (obs x state) this run already built or loaded."""
        flux = getattr(self, "_flux_jacobian", None)
        if flux is None:
            flux = self._flux_jacobian = self._get_flux_jacobian(obs)
        return flux["concentration", "flux"]

    def _per_obs_std(self, obs: Vector, src: str) -> np.ndarray:
        """Per-obs std [ppm] for a data-derived MDM term.

        ``src="subhour"`` -> each obs's within-hour CH4 std (temporal representativeness error:
        the sub-hour variability an hour-mean footprint cannot represent). Obs with no
        multi-point hour (std undefined) get 0 -- no representativeness penalty.
        """
        if src != "subhour":
            raise ValueError(f"unknown per_obs MDM source {src!r}")
        s = get_slv_subhour_std(
            sites=self.config.sites,
            site_config=self.config.site_config,
            time_range=self.config.time_range,
            subset_hours=self.config.subset_hours,
            filter_pcaps=self.config.filter_pcaps,
            num_processes=self.config.num_processes,
            utc_offset=self.config.utc_offset,
        )
        key = obs.index.droplevel(
            [n for n in obs.index.names if n not in s.index.names]
        )
        return s.reindex(key).fillna(0.0).to_numpy()

    @fips_cache(CovarianceMatrix, "modeldata_mismatch")
    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        """Model-data mismatch covariance: the sum of ``config.mdm_components``.

        Each component is built by :func:`~slv.inversion.covariances.build_mdm_error`
        over the obs index, with site-keyed terms resolved through :meth:`obs_sites`
        (TRAX receptors belong to their mobile site). A ``multiplicative`` component's
        std is ``fraction`` times the per-obs enhancement scale (``scale_on``); a
        ``per_obs`` component's is ``fraction`` times a per-obs std (e.g. the within-hour
        std for ``subhour``).
        """
        obs_sites = self.obs_sites(obs.index)
        components = []
        for comp in self.config.mdm_components:
            c = dict(comp)
            if c.pop("multiplicative", False):
                scale = self._multiplicative_scale(obs, c.pop("scale_on", "footprint"))
                c["std"] = c.pop("fraction") * scale  # sigma = fraction * enhancement
            elif (src := c.pop("per_obs", None)) is not None:
                # sigma = fraction * per-obs data-derived std (e.g. subhour within-hour std)
                c["std"] = c.pop("fraction", 1.0) * self._per_obs_std(obs, src)
            components.append(
                build_mdm_error(
                    obs_index=obs.index,
                    site_config=self.config.site_config,
                    obs_sites=obs_sites,
                    **c,
                )
            )

        if self.config.plot_diagnostics:
            built_comps = {comp.name: comp.build(obs.index) for comp in components}
            viz.plot_mdm_components(built_comps)
            return CovarianceMatrix(
                name="modeldata_mismatch",
                data=np.add.reduce([comp.to_numpy() for comp in built_comps.values()]),
                index=obs.index,
            )

        return CovarianceMatrix(
            name="modeldata_mismatch",
            data=CovarianceBuilder(components).build(obs.index),
        )
