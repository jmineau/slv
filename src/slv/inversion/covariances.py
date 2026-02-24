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


def build_prior_error(prior_block, **kwargs) -> pd.DataFrame:
    """
    Builds the S_0 Kronecker covariance matrix for the SLV flux prior.
    """
    # Get configuration parameters for the prior covariance from kwargs, with defaults
    base_std = kwargs.get("prior_base_std", 0.0)
    std_frac = kwargs.get("prior_std_frac", 0.0)
    time_scale = kwargs.get("prior_time_scale")
    spatial_scale = kwargs.get("prior_spatial_scale")

    # Calculate dynamic variances proportional to the prior flux
    variances = (base_std + std_frac * prior_block.data) ** 2

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
    return S_0.build(prior_block.index)


def build_mdm_component(
    name,
    obs_index,
    std,
    correlated=True,
    scale=None,
    interday=False,
    time_dim="obs_time",
) -> ErrorComponent:
    """
    Unified builder for Model-Data Mismatch covariance components.
    """
    # Covariance components expect variance, so square the standard deviation
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
