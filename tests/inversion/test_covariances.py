"""Tests for site- and organization-keyed MDM terms with mobile (TRAX) obs."""

import numpy as np
import pandas as pd
import pytest

from slv.inversion.covariances import build_mdm_error
from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()
# A tower obs is keyed by its site; a TRAX receptor by its PYSTILT location_id.
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb", "hw", "-111.9_40.7_4"], pd.to_datetime(["2024-01-01 20:00"] * 3)],
    names=["obs_location", "obs_time"],
)
OBS_SITES = ["wbb", "hw", "trx01"]


def variances(component):
    return np.diag(component.build(OBS_INDEX).to_numpy())


def test_organization_std_needs_the_obs_site():
    with pytest.raises(KeyError):
        build_mdm_error(
            "instr", OBS_INDEX, std={"UATAQ": 0.1, "DAQ": 0.3}, site_config=SITE_CONFIG
        )


def test_organization_std_uses_obs_sites():
    comp = build_mdm_error(
        "instr",
        OBS_INDEX,
        std={"UATAQ": 0.1, "DAQ": 0.3},
        correlated=False,
        site_config=SITE_CONFIG,
        obs_sites=OBS_SITES,
    )
    # wbb and trx01 are UATAQ, hw is DAQ
    np.testing.assert_allclose(variances(comp), [0.01, 0.09, 0.01])


def test_site_season_std_uses_obs_sites():
    std = {s: {"DJF": 0.1 * (i + 1)} for i, s in enumerate(OBS_SITES)}
    comp = build_mdm_error(
        "custom", OBS_INDEX, std=std, correlated=False, obs_sites=OBS_SITES
    )
    np.testing.assert_allclose(variances(comp), [0.01, 0.04, 0.09])
