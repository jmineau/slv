"""Prior error and model-data mismatch covariances for the SLV inversion."""

import numpy as np
import pandas as pd
from fips import Constant, Covariance, Exponential, Independent, Kronecker, Space
from fips.covariance import CovarianceModel
from lair.clock import SEASONS


def normalize_duration(value):
    """
    ``"14d"`` -> ``"14D"``. pandas deprecates the lowercase day unit (Pandas4Warning).
    Config strings keep their spelling -- they are hashed into cache keys and sweep IDs --
    and are converted only where they are parsed.
    """
    if isinstance(value, str) and value.endswith("d"):
        return value[:-1] + "D"
    return value


def build_prior_error(
    prior: pd.Series,
    *,
    time_scale: str,
    spatial_scale: float,
    base_std: float = 0.0,
    std_frac: float = 0.0,
) -> Covariance:
    """
    The prior flux error covariance S_0: a Kronecker product of a time and a space decay.

    Each cell's standard deviation is ``base_std + std_frac * prior``; the correlation
    decays exponentially in time (``time_scale``) and with the great-circle distance
    between cells (``spatial_scale``, km). The covariance is kept as its two factors.

    Parameters
    ----------
    prior : pd.Series
        The flux prior, indexed by (time, lat, lon) in any order of the levels.
    time_scale : str
        The e-folding time of the correlation (``"32d"``).
    spatial_scale : float
        The e-folding distance of the correlation, in km.
    base_std, std_frac : float
        The standard deviation's floor and its fraction of the prior.

    Returns
    -------
    Covariance
        S_0 on the prior's labels, as the one block ``flux``, kept as Kronecker factors.
    """
    # The variances are computed in the prior's own dtype (float32 for the EPA prior)
    # and square-rooted, as slv and fips 0.1 did, so paper 1's S_0 is unchanged.
    variances = (base_std + std_frac * prior) ** 2
    std = np.sqrt(variances.to_numpy())
    model = Kronecker(
        Exponential("time", normalize_duration(time_scale)),
        Exponential(["lat", "lon"], spatial_scale, metric="haversine"),
    )
    return model.with_std(std).build(Space({"flux": prior.index}), name="prior_error")


def build_mdm_error(
    name,
    obs_index,
    std,
    correlated=True,
    scale=None,
    interday=False,
    time_dim="obs_time",
    site_config=None,
    obs_sites=None,
    **kwargs,
) -> CovarianceModel:
    """
    Unified builder for Model-Data Mismatch covariance components.

    Returns a covariance model, to build on the observations' space (whose labels must
    be ``obs_index``, in that order: a per-observation ``std`` is read in that order).

    Parameters
    ----------
    name : str
        Component name (used in error messages)
    obs_index : pd.MultiIndex
        Observation index with 'obs_location' and 'obs_time' levels
    std : float or dict
        Absolute standard deviation in ppm.
        If dict with nested dicts: std[site][season] structure (site/season-specific)
        If dict with scalar values: std[organization] structure (organization-specific)
    correlated : bool
        Whether errors are correlated
    scale : str, optional
        Time scale for correlation decay
    interday : bool
        Whether correlations extend across midnight
    time_dim : str
        Name of the time dimension
    site_config : pd.DataFrame, optional
        Site configuration with 'organization' column, required for organization-based std
    obs_sites : sequence of str, optional
        The site of each observation, aligned with ``obs_index``, for site- or
        organization-keyed ``std``. Defaults to the ``obs_location`` level, which is the
        site for a tower but a PYSTILT location_id for a mobile receptor.
    **kwargs
        Additional unused parameters (for compatibility)
    """
    # Check if std is a dict (site/season-specific or organization-specific) or scalar
    if isinstance(std, dict):
        times = obs_index.get_level_values(time_dim)
        locations = (
            pd.Index(obs_sites)
            if obs_sites is not None
            else obs_index.get_level_values("obs_location")
        )

        # Detect if this is a nested dict (site/season) or flat dict (organization)
        first_value = next(iter(std.values()))
        is_nested = isinstance(first_value, dict)

        if is_nested:
            # Site/season-specific std
            seasons = times.to_series().dt.month.map(SEASONS).values

            # Look up std for each observation by site and season
            std_values = []
            for loc, season in zip(locations, seasons, strict=False):
                try:
                    std_val = std[loc][season]
                    std_values.append(std_val)
                except (KeyError, TypeError):
                    raise ValueError(
                        f"No std value found for site='{loc}', season='{season}' in component '{name}'"
                    ) from None
        else:
            # Organization-specific std (flat dict)
            if site_config is None:
                raise ValueError(
                    f"site_config must be provided for organization-based std in component '{name}'"
                )

            # Look up std for each observation by organization
            std_values = []
            for loc in locations:
                org = site_config.at[loc, "organization"]
                try:
                    std_val = std[org]
                    std_values.append(std_val)
                except KeyError:
                    raise ValueError(
                        f"No std value found for organization='{org}' (site='{loc}') in component '{name}'"
                    ) from None

        std = np.asarray(std_values, dtype=float)
    elif isinstance(std, (np.ndarray, pd.Series, list, tuple)):
        # Per-observation std array, aligned to obs_index (e.g. a multiplicative
        # transport term f * |H x_prior| that scales with the modeled enhancement).
        std = np.asarray(std, dtype=float)
        if len(std) != len(obs_index):
            raise ValueError(
                f"Component '{name}' has {len(std)} stds for {len(obs_index)} obs."
            )

    # Uncorrelated errors (diagonal)
    if not correlated:
        return Independent().with_std(std)

    # Correlated errors: within groups of all non-time levels (e.g. 'obs_location'),
    # and within a day unless ``interday`` (to sever midnight correlations)
    within = [level for level in obs_index.names if level != time_dim]
    if not interday:
        within.append(f"{time_dim}.date")
    if scale is None:
        return Constant(within=within).with_std(std)
    return Exponential(time_dim, normalize_duration(scale), within=within).with_std(std)
