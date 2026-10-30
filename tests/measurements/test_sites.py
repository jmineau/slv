"""The UATAQ rows of site_config.csv agree with uataq's own site config."""

import math

import pandas as pd
import pytest
import uataq

from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()
UATAQ_STIDS = list(SITE_CONFIG.index[SITE_CONFIG["organization"] == "UATAQ"])

# uataq calls WBB "William Browning"; both places are to say "William Browning
# Building", and the change is in uataq's config. strict: when slv's pinned uataq
# carries the new name this test passes, the run fails, and the mark is removed.
NAME_MARKS = {
    "wbb": pytest.mark.xfail(
        strict=True,
        reason="uataq's WBB name is still 'William Browning'; its config is to be "
        "changed to 'William Browning Building'. Remove this mark when the "
        "uataq pin carries the change.",
    )
}


def test_there_are_uataq_rows():
    """The guard below tests something."""
    assert len(UATAQ_STIDS) > 0


@pytest.mark.parametrize("stid", UATAQ_STIDS)
def test_uataq_row_position_and_status_match_uataq(stid):
    """
    A UATAQ row repeats uataq's latitude, longitude and active flag.

    A mobile site has no latitude or longitude in either place. uataq's
    ``is_active`` is left out for a site that is still up, so a missing key is
    active.
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


@pytest.mark.parametrize(
    "stid",
    [pytest.param(s, marks=NAME_MARKS.get(s, ())) for s in UATAQ_STIDS],
)
def test_uataq_row_name_matches_uataq(stid):
    """A UATAQ row has the name uataq gives its site, where both hold one."""
    ours = SITE_CONFIG.loc[stid, "name"]
    theirs = uataq.get_site(stid).config.get("name")
    if theirs is not None and pd.notna(ours):
        assert ours == theirs, f"{stid} name: slv has {ours!r}, uataq {theirs!r}"
