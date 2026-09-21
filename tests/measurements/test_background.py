"""Tests for UATAQCH4, with the UATAQ reader monkeypatched."""

import pandas as pd
import pytest

from slv.measurements import background
from slv.measurements.background import UATAQCH4


@pytest.fixture
def fake_uataq(monkeypatch):
    def fake_get_obs(site, pollutants):
        index = pd.date_range("2024-01-01", periods=48 * 6, freq="10min")
        return pd.DataFrame({"CH4d_ppm_cal": 2.0}, index=index)

    monkeypatch.setattr(background.uataq, "get_obs", fake_get_obs)


def test_site_returns_hourly_series(fake_uataq):
    data = UATAQCH4()["hdp"]
    assert isinstance(data, pd.Series)
    assert data.name == "CH4"
    assert len(data) == 48


def test_base_method(fake_uataq):
    data = UATAQCH4()["hdp_base"]
    assert isinstance(data, pd.Series)
    assert (data == 2.0).all()


@pytest.mark.parametrize("key", ["hdp_bse", "hdp_base_x"])
def test_unknown_method_raises(fake_uataq, key):
    with pytest.raises(ValueError, match="Unknown method"):
        UATAQCH4()[key]
