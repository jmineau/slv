"""The model-data mismatch (S_z) of the SLV inversion: the sum of ``config.mdm_components``."""

import numpy as np
import pandas as pd
from fips import Covariance, Operator, Space

from slv.inversion import viz
from slv.inversion.cache import fips_cache
from slv.inversion.config import InversionConfig
from slv.inversion.covariances import build_mdm_error
from slv.inversion.data import get_slv_subhour_std


def _on_obs(values: pd.Series, obs: pd.Series) -> np.ndarray:
    """``values`` (keyed by some of the obs' levels) for each obs, 0 where it has none."""
    key = obs.index.droplevel(
        [n for n in obs.index.names if n not in values.index.names]
    )
    return values.reindex(key).fillna(0.0).to_numpy()


def multiplicative_scale(
    obs: pd.Series,
    scale_on: str,
    *,
    prior: pd.Series,
    jacobian: Operator,
    constant: pd.Series,
) -> np.ndarray:
    """
    Per-obs scale [ppm] for a multiplicative MDM term.

    ``scale_on="obs"`` gives ``|obs - background|``, the observed enhancement: the actual
    signal, so extreme shallow-PBL days the prior under-predicts still get a large error.
    ``scale_on="prior"`` gives ``|H x_prior|``, the prior-modeled enhancement; it
    under-scales where the EPA prior is low, exactly the over-leveraged days. Obs
    lacking the needed term get 0 (floor-only error). ``scale_on="footprint"`` gives
    ``|H| * mean(x_prior)``, the footprint strength in enhancement units. It
    down-weights extreme shallow-PBL days by how much air they integrate, independent of
    the obs (no circularity) or the EPA pattern, so it eases both the low-obs winter
    dips and the high-obs spikes, which share an extreme ``|H|``.

    Parameters
    ----------
    obs : pd.Series
        The obs, indexed by ``(obs_location, obs_time)``.
    scale_on : {"footprint", "obs", "prior"}
        The scale.
    prior : pd.Series
        The flux prior, as :func:`~slv.inversion.pipelines.get_prior` gives it.
    jacobian : Operator
        The flux Jacobian, as :func:`~slv.inversion.pipelines.get_flux_jacobian` gives
        it (rows: receptors; columns: cells and times).
    constant : pd.Series
        The background of each obs, as :func:`~slv.inversion.pipelines.get_constant`
        gives it.
    """
    receptors = jacobian.rows.select("concentration")
    if scale_on == "footprint":
        xref = float(np.mean(prior.to_numpy()))
        row_abs = np.asarray(abs(jacobian.sparse()).sum(axis=1)).ravel()
        return _on_obs(pd.Series(row_abs * xref, index=receptors), obs)
    if scale_on == "obs":
        key = obs.index.droplevel(
            [n for n in obs.index.names if n not in constant.index.names]
        )
        bg = constant.reindex(key).to_numpy(dtype=float)
        return np.abs(np.nan_to_num(obs.to_numpy() - bg, nan=0.0))
    # Match the prior to the Jacobian columns by label, not position: their level
    # orders differ (the EPA prior is (time, lat, lon), the Jacobian (lon, lat, time)).
    cells = jacobian.cols.select("flux")
    x = prior.reorder_levels(cells.names).reindex(cells, fill_value=0.0)
    e = pd.Series(jacobian.sparse() @ x.to_numpy(dtype=float), index=receptors)
    return np.abs(_on_obs(e, obs))


def per_obs_std(config: InversionConfig, obs: pd.Series, src: str) -> np.ndarray:
    """
    Per-obs std [ppm] for a data-derived MDM term.

    ``src="subhour"`` -> each obs's within-hour CH4 std (temporal representativeness error:
    the sub-hour variability an hour-mean footprint cannot represent). Obs with no
    multi-point hour (std undefined) get 0 -- no representativeness penalty.
    """
    if src != "subhour":
        raise ValueError(f"unknown per_obs MDM source {src!r}")
    s = get_slv_subhour_std(
        sites=config.sites,
        site_config=config.site_config,
        time_range=config.time_range,
        subset_hours=config.subset_hours,
        filter_pcaps=config.filter_pcaps,
        num_processes=config.num_processes,
        utc_offset=config.utc_offset,
    )
    return _on_obs(s, obs)


def _footprint_key(config: InversionConfig, obs: pd.Series, **_) -> list:
    """The footprints the MDM is built from, for its cache key (a multiplicative term scales with the Jacobian's rows)."""
    # imported here: pipelines imports this module
    from slv.inversion.pipelines import _footprint_key as footprints

    return footprints(config, obs)


@fips_cache("modeldata_mismatch", key=_footprint_key)
def get_modeldata_mismatch(
    config: InversionConfig,
    obs: pd.Series,
    *,
    prior: pd.Series,
    jacobian: Operator,
    constant: pd.Series,
    obs_sites: pd.Index,
) -> Covariance:
    """
    Model-data mismatch covariance: the sum of ``config.mdm_components``, on the obs.

    Each component is built by :func:`~slv.inversion.covariances.build_mdm_error`
    over the obs index, with site-keyed terms resolved through ``obs_sites`` (TRAX
    receptors belong to their mobile site). A ``multiplicative`` component's std is
    ``fraction`` times the per-obs enhancement scale (:func:`multiplicative_scale`,
    ``scale_on``); a ``per_obs`` component's is ``fraction`` times a per-obs std (e.g.
    the within-hour std for ``subhour``, :func:`per_obs_std`).

    Parameters
    ----------
    config : InversionConfig
        The inversion.
    obs : pd.Series
        The obs the problem has, indexed by ``(obs_location, obs_time)``.
    prior, jacobian, constant :
        The run's prior, flux Jacobian and background, for the multiplicative terms.
    obs_sites : pd.Index
        The site of each obs (:func:`~slv.inversion.pipelines.obs_sites`).

    Returns
    -------
    Covariance
        S_z on the obs' labels, as the one block ``concentration``.
    """
    space = Space({"concentration": obs.index})
    components = {}
    for comp in config.mdm_components:
        c = dict(comp)
        if c.pop("multiplicative", False):
            scale = multiplicative_scale(
                obs,
                c.pop("scale_on", "footprint"),
                prior=prior,
                jacobian=jacobian,
                constant=constant,
            )
            c["std"] = c.pop("fraction") * scale  # sigma = fraction * enhancement
        elif (src := c.pop("per_obs", None)) is not None:
            # sigma = fraction * per-obs data-derived std (e.g. subhour within-hour std)
            c["std"] = c.pop("fraction", 1.0) * per_obs_std(config, obs, src)
        components[c["name"]] = build_mdm_error(
            obs_index=obs.index,
            site_config=config.site_config,
            obs_sites=obs_sites,
            **c,
        )

    if config.plot_diagnostics:
        built = {name: m.build(space) for name, m in components.items()}
        viz.plot_mdm_components({name: S.to_frame() for name, S in built.items()})

    models = list(components.values())
    total = models[0]
    for model in models[1:]:
        total = total + model
    return total.build(space, name="modeldata_mismatch")
