"""The UATAQ rows of site_config.csv agree with uataq's own site config."""

import math

import pandas as pd
import pytest
import uataq

from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()
UATAQ_STIDS = list(SITE_CONFIG.index[SITE_CONFIG["organization"] == "UATAQ"])


def test_there_are_uataq_rows():
    """The guard below tests something."""
    assert len(UATAQ_STIDS) > 0


@pytest.mark.parametrize("stid", UATAQ_STIDS)
def test_uataq_row_matches_uataq(stid):
    """
    A UATAQ row repeats what uataq's config holds, so the two must not drift.

    Compared: latitude, longitude (both missing for a mobile site), the active flag
    (uataq's ``is_active``, which a site that is still up leaves out) and the name.
    """
    row = SITE_CONFIG.loc[stid]
    config = uataq.get_site(stid).config

    for column in ("latitude", "longitude"):
        theirs = config.get(column)
        ours = None if pd.isna(row[column]) else float(row[column])
        if theirs is None or ours is None:
            assert theirs == ours, f"{stid} {column}: slv has {ours}, uataq {theirs}"
        else:
            assert math.isclose(ours, theirs, rel_tol=0, abs_tol=1e-9), (
                f"{stid} {column}: slv has {ours}, uataq {theirs}"
            )

    assert bool(row["active"]) == config.get("is_active", True), (
        f"{stid} active: slv has {row['active']}, uataq {config.get('is_active', True)}"
    )

    if config.get("name") is not None and pd.notna(row["name"]):
        assert row["name"] == config["name"], (
            f"{stid} name: slv has {row['name']!r}, uataq {config['name']!r}"
        )
