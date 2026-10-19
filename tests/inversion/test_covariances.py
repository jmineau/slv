"""Tests for site- and organization-keyed MDM terms with mobile (TRAX) obs."""

import numpy as np
import pandas as pd
import pytest
from fips import Space

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
    return component.build(Space({"concentration": OBS_INDEX})).variances.values


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [("14d", "14D"), ("32d", "32D"), ("2.8h", "2.8h"), ("7D", "7D"), (None, None)],
)
def test_normalize_duration(value, expected):
    from slv.inversion.covariances import normalize_duration

    assert normalize_duration(value) == expected


def test_correlated_components_stay_within_their_groups():
    times = pd.to_datetime(["2024-01-01 20:00", "2024-01-01 21:00", "2024-01-02 20:00"])
    index = pd.MultiIndex.from_arrays(
        [["wbb"] * 3, times], names=["obs_location", "obs_time"]
    )
    space = Space({"concentration": index})
    within_day = build_mdm_error("aggr", index, std=0.1, scale=None).build(space)
    np.testing.assert_allclose(
        within_day.dense(), 0.01 * np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    )
    across_days = build_mdm_error("bg", index, std=0.1, scale="7d", interday=True)
    C = across_days.build(space).dense()
    assert C[0, 2] == pytest.approx(0.01 * np.exp(-1 / 7))
