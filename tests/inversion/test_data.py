"""Tests for the inversion's obs loading: the spike filter, tower-only runs, tz-aware
mobile obs and the empty sub-hour std."""

import pandas as pd
import pytest

from slv.inversion import data as data_module
from slv.inversion.data import (
    _drop_spike_days,
    get_slv_observations,
    get_slv_subhour_std,
    load_mobile_obs,
)
from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()


def native_record(days=10, spiky_day=3, site="wbb"):
    """Four 15-min samples an hour, 20-21 UTC each day; one day with a plume."""
    rows = []
    for d in range(days):
        for i, minute in enumerate((0, 15, 30, 45)):
            ch4 = 2.0 + 0.001 * i
            if d == spiky_day and i == 2:
                ch4 += 0.5
            t = pd.Timestamp("2024-06-01 20:00") + pd.Timedelta(days=d, minutes=minute)
            rows.append({"Time_UTC": t, "site": site, "CH4": ch4})
    return pd.DataFrame(rows)


def test_spike_filter_drops_the_plume_day():
    obs = native_record()
    out = _drop_spike_days(obs, 0.9)
    days = pd.to_datetime(out.Time_UTC).dt.day.unique().tolist()
    assert 4 not in days and len(days) == 9  # 2024-06-04 is day index 3
    assert len(out) == len(obs) - 4


def test_spike_filter_is_per_site():
    obs = pd.concat(
        [native_record(site="wbb"), native_record(spiky_day=7, site="hdp")],
        ignore_index=True,
    )
    out = _drop_spike_days(obs, 0.9)
    dropped = obs.drop(out.index)
    assert dropped.groupby("site").Time_UTC.first().dt.day.to_dict() == {
        "hdp": 8,
        "wbb": 4,
    }


def test_spike_filter_keeps_days_without_a_multi_point_hour():
    obs = (
        native_record(days=3)
        .groupby(pd.to_datetime(native_record(days=3).Time_UTC).dt.day)
        .head(1)
    )  # one sample per day: no within-hour std
    assert len(_drop_spike_days(obs, 0.5)) == len(obs)
    assert _drop_spike_days(obs.iloc[:0], 0.9).empty


@pytest.fixture
def aggregated(monkeypatch):
    """Native records in, aggregate_obs's arguments recorded (it passes rows through)."""
    seen = []

    def fake_aggregate(obs, **kwargs):
        seen.append(kwargs)
        return obs

    monkeypatch.setattr(
        data_module, "load_concentrations", lambda **kwargs: native_record()
    )
    monkeypatch.setattr(data_module, "aggregate_obs", fake_aggregate)
    return seen


def test_tower_only_observations_need_no_trax_data(aggregated, monkeypatch):
    monkeypatch.delenv("SLV_USER_DATA_DIR", raising=False)
    monkeypatch.delenv("LINGROUP_DATA_DIR", raising=False)
    obs = get_slv_observations(["wbb"], SITE_CONFIG, ("2024-06-01", "2024-06-11"))
    assert aggregated[0]["mobile_points"] is None
    assert obs.index.names == ["obs_location", "obs_time"]
    assert obs.index.get_level_values("obs_location").unique().tolist() == ["wbb"]


def test_trax_points_are_loaded_for_a_mobile_site(aggregated, monkeypatch):
    monkeypatch.setattr(data_module, "load_trax_points", lambda: "points")
    get_slv_observations(["wbb", "trx01"], SITE_CONFIG, ("2024-06-01", "2024-06-11"))
    assert aggregated[0]["mobile_points"] == "points"


def test_spike_filter_is_applied_when_asked(aggregated):
    tr = ("2024-06-01", "2024-06-11")
    kept = get_slv_observations(["wbb"], SITE_CONFIG, tr, filter_spikes=True)
    assert len(kept) == 36  # the plume day's four samples are gone
    assert len(get_slv_observations(["wbb"], SITE_CONFIG, tr)) == 40


def test_mobile_obs_with_tz_aware_times():
    t = pd.DatetimeIndex(
        ["2024-06-01 19:30", "2024-06-01 20:30", "2024-06-02 20:30"], tz="UTC"
    )
    df = pd.DataFrame(
        {"obs_location": "-111.9_40.7_4", "obs_time": t, "CH4": [2.0, 2.1, 2.2]}
    )  # join keys as columns: set as the index
    out = load_mobile_obs(
        df,
        (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-06-02", tz="UTC")),
        subset_hours=[13],  # 20:30 UTC is 13:30 MST
        filter_pcaps=False,
    )
    assert out.CH4.tolist() == [2.1]


def test_subhour_std_without_data_is_empty(monkeypatch):
    monkeypatch.setattr(
        data_module, "load_concentrations", lambda **kwargs: native_record().iloc[:0]
    )
    out = get_slv_subhour_std(["wbb"], SITE_CONFIG, ("2024-06-01", "2024-06-02"))
    assert out.empty and out.index.names == ["obs_location", "obs_time"]


@pytest.mark.parametrize("sites", [["trx01"], []])
def test_subhour_std_skips_mobile_sites(monkeypatch, sites):
    monkeypatch.setattr(data_module, "load_concentrations", _must_not_load)
    assert get_slv_subhour_std(sites, SITE_CONFIG, ("2024-06-01", "2024-06-02")).empty


def _must_not_load(**kwargs):
    raise AssertionError("mobile sites should not be loaded")
