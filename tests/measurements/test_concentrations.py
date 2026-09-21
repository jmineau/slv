"""Tests for load_concentrations, with the UATAQ reader monkeypatched."""

import pandas as pd
import pytest

from slv.domain import UTC_OFFSET
from slv.measurements import concentrations
from slv.measurements.concentrations import load_concentrations


def fake_calibrated_lgr(site, instruments, **kwargs):
    """Stand-in for uataq.read_data: calibrated lgr_ugga columns (no H2O)."""
    df = pd.DataFrame(
        {
            "Time_UTC": pd.date_range("2024-01-01", periods=4, freq="1h"),
            "CO2d_ppm_cal": 420.0,
            "ID_CO2": -10.0,
            "CH4d_ppm_cal": [2.0, 2.1, 2.2, 2.3],
            "ID_CH4": -10.0,
            "QAQC_Flag": 0,
        }
    ).set_index("Time_UTC")
    return {instruments: df}


@pytest.fixture
def fake_uataq(monkeypatch):
    monkeypatch.setattr(concentrations.uataq, "read_data", fake_calibrated_lgr)


def test_org_without_instruments_skipped(fake_uataq):
    # uta (NOAA GML) has no instruments in site_config.csv
    with pytest.raises(ValueError, match="No data loaded"):
        load_concentrations("CH4", orgs="NOAA GML")


def test_site_without_instruments_skipped(fake_uataq):
    obs = load_concentrations("CH4", sites=["arc", "wbb"])
    assert set(obs["site"]) == {"wbb"}


def test_missing_pollutant_column_skipped(fake_uataq):
    # Calibrated lgr_ugga files have no H2O; CH4 should still load
    obs = load_concentrations(["CH4", "H2O"], sites="wbb")
    assert "CH4" in obs.columns
    assert "H2O" not in obs.columns
    assert len(obs) == 4


def test_only_missing_pollutant_raises_no_data(fake_uataq):
    with pytest.raises(ValueError, match="No data loaded"):
        load_concentrations("H2O", sites="wbb")


def test_time_mst_uses_domain_offset(fake_uataq):
    obs = load_concentrations("CH4", sites="wbb")
    assert UTC_OFFSET == -7
    expected = obs["Time_UTC"] - pd.Timedelta(hours=7)
    pd.testing.assert_series_equal(obs["Time_MST"], expected, check_names=False)
