"""Tests for the inversion background with mobile (TRAX) sites in the config."""

import numpy as np
import pandas as pd
import pytest

from slv.inversion import background as bg
from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()
HOURS = pd.date_range("2024-06-01", periods=24 * 10, freq="h")


@pytest.fixture
def loaded(monkeypatch):
    """Fake hourly tower data; records which sites were loaded."""
    calls = []

    def fake_load(pollutants, sites, **kwargs):
        calls.append(list(sites))
        rng = np.random.default_rng(0)
        return pd.concat(
            pd.DataFrame(
                {"Time_UTC": HOURS, "site": s, "CH4": 2.0 + rng.random(len(HOURS))}
            )
            for s in sites
        )

    monkeypatch.setattr(bg, "load_concentrations", fake_load)
    monkeypatch.setattr(bg, "aggregate_obs", lambda data, **kwargs: data)
    return calls


def rolling(sites, **kwargs):
    return bg.get_rolling_background(
        sites=sites, site_config=SITE_CONFIG, time_range=(HOURS[0], HOURS[-1]), **kwargs
    )


class TestRollingBackgroundSites:
    def test_mobile_sites_are_not_loaded(self, loaded):
        # trx01's per-grid-point rows used to crash the (obs_time, site) unstack
        rolling(["wbb", "trx01"])
        assert loaded == [["wbb"]]

    def test_background_sites_override(self, loaded):
        rolling(["trx01"], background_sites=["wbb", "hw"])
        assert loaded == [["wbb", "hw"]]

    def test_no_stationary_site_raises(self, loaded):
        with pytest.raises(ValueError, match="background_sites"):
            rolling(["trx01"])


class TestBackgroundAtObsTimes:
    def test_obs_take_the_background_of_their_hour(self, monkeypatch):
        hourly = pd.Series(np.arange(len(HOURS), dtype=float), index=HOURS)
        monkeypatch.setattr(bg, "get_rolling_background", lambda **kwargs: hourly)
        obs_times = pd.DatetimeIndex(["2024-06-02 20:00", "2024-06-02 20:13"])

        out = bg.get_slv_background(
            "rolling",
            obs_times=obs_times,
            sites=["wbb", "trx01"],
            site_config=SITE_CONFIG,
            time_range=(HOURS[0], HOURS[-1]),
        )

        # a receptor released at 20:13 used to find no background and be dropped
        expected = hourly[pd.Timestamp("2024-06-02 20:00")]
        assert out.tolist() == [expected, expected]
        pd.testing.assert_index_equal(out.index, obs_times, check_names=False)
        assert out.index.name == "obs_time"
