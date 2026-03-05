import pandas as pd
from fips.covariance import (
    BlockDecayError,
    DiagonalError,
    ErrorComponent,
    KroneckerError,
)
from fips.kernels import (
    ConstantCorrelation,
    GridSpatialDecay,
    GridTimeDecay,
    RaggedTimeDecay,
)


def build_prior_error(prior, **kwargs) -> pd.DataFrame:
    """
    Builds the S_0 Kronecker covariance matrix for the SLV flux prior.
    """
    # Get configuration parameters for the prior covariance from kwargs, with defaults
    base_std = kwargs.get("base_std", 0.0)
    std_frac = kwargs.get("std_frac", 0.0)
    time_scale = kwargs.get("time_scale")
    spatial_scale = kwargs.get("spatial_scale")

    # Calculate dynamic variances proportional to the prior flux
    variances = (base_std + std_frac * prior.data) ** 2

    # 2. Define the Kronecker marginals using the strict grid kernels
    S_0 = KroneckerError(
        name="prior_error",
        variances=variances,
        marginal_kernels=[
            ("time", GridTimeDecay(scale=time_scale)),
            (
                ["lat", "lon"],
                GridSpatialDecay(lat_dim="lat", lon_dim="lon", scale=spatial_scale),
            ),
        ],
    )

    # Build and return aligned to the prior's MultiIndex
    return S_0.build(prior.index)


def build_mdm_error(
    name,
    obs_index,
    std,
    correlated=True,
    scale=None,
    interday=False,
    time_dim="obs_time",
    site_config=None,
    **kwargs,
) -> ErrorComponent:
    """
    Unified builder for Model-Data Mismatch covariance components.

    Parameters
    ----------
    name : str
        Component name
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
    **kwargs
        Additional unused parameters (for compatibility)
    """
    # Check if std is a dict (site/season-specific or organization-specific) or scalar
    if isinstance(std, dict):
        times = obs_index.get_level_values(time_dim)
        locations = obs_index.get_level_values("obs_location")

        # Detect if this is a nested dict (site/season) or flat dict (organization)
        first_value = next(iter(std.values()))
        is_nested = isinstance(first_value, dict)

        if is_nested:
            # Site/season-specific std
            seasons = (
                times.to_series()
                .dt.month.map(
                    {
                        12: "DJF",
                        1: "DJF",
                        2: "DJF",
                        3: "MAM",
                        4: "MAM",
                        5: "MAM",
                        6: "JJA",
                        7: "JJA",
                        8: "JJA",
                        9: "SON",
                        10: "SON",
                        11: "SON",
                    }
                )
                .values
            )

            # Look up std for each observation by site and season
            std_values = []
            for loc, season in zip(locations, seasons, strict=False):
                try:
                    std_val = std[loc][season]
                    std_values.append(std_val)
                except (KeyError, TypeError):
                    raise ValueError(
                        f"No std value found for site='{loc}', season='{season}' in component '{name}'"
                    )
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
                    )

        variances = pd.Series(std_values, index=obs_index) ** 2
    else:
        # Scalar std
        variances = std**2

    # Uncorrelated errors (diagonal)
    if not correlated:
        # We don't need groupers or kernels, just drop it on the diagonal
        return DiagonalError(name=name, variances=variances)

    # Correlated errors (block-diagonal with optional time decay)
    # Determine the mathematical kernel
    if scale is None:
        corr_func = ConstantCorrelation()
    else:
        corr_func = RaggedTimeDecay(time_dim=time_dim, scale=scale)

    # Determine the grouping structure
    # Group by all non-time levels (e.g., 'obs_location')
    groupers = [col for col in obs_index.names if col != time_dim]

    if not interday:
        # Append the date to the groupers to strictly sever midnight correlations
        groupers.append(obs_index.get_level_values(time_dim).date)

    return BlockDecayError(
        name=name, variances=variances, groupers=groupers, corr_func=corr_func
    )
