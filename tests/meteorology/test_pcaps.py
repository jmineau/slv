"""Tests for slv.meteorology.pcaps, with soundings replaced by a synthetic VHD."""

import importlib

import pandas as pd
import pytest

from slv.meteorology import pcaps

TIME_RANGE = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-06"))
OTHER_RANGE = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"))
RECORD = TIME_RANGE  # the full synthetic sounding record


@pytest.fixture(autouse=True)
def no_user_data_dir(monkeypatch):
    # Never touch the real cache
    monkeypatch.delenv("SLV_USER_DATA_DIR", raising=False)


@pytest.fixture
def fake_soundings(monkeypatch):
    """Replace soundings + VHD with a 12-hourly VHD series; count the calls."""
    calls = []

    def fake_get_soundings(start=None, end=None, **kwargs):
        calls.append((start, end))
        index = pd.date_range(start or RECORD[0], end or RECORD[1], freq="12h")
        values = [1, 1, 4, 4, 4, 6, 6, 6, 1, 1, 1]
        return pd.Series(values[: len(index)], index=index, dtype=float)

    monkeypatch.setattr(pcaps, "get_soundings", fake_get_soundings)
    monkeypatch.setattr(pcaps.lair.pcaps, "valleyheatdeficit", lambda s: s)
    return calls


@pytest.fixture
def curated_csv(monkeypatch, tmp_path):
    """A synthetic curated pcap_events.csv in a tmp SLV_USER_DATA_DIR."""
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(tmp_path))
    path = tmp_path / "pcap_events.csv"
    path.write_text("start,end\n2020-01-01 00:00:00,2020-01-03 11:59:59\n")
    return path


def test_default_args_read_pcap_events_csv(fake_soundings, curated_csv):
    events = pcaps.get_pcap_events(TIME_RANGE)
    expected = pd.read_csv(curated_csv, parse_dates=["start", "end"])
    pd.testing.assert_frame_equal(events, expected)
    assert fake_soundings == []


def test_non_default_args_use_own_file(fake_soundings, curated_csv, tmp_path):
    before = curated_csv.read_text()

    events = pcaps.get_pcap_events(TIME_RANGE, threshold=5)
    assert fake_soundings == [(None, None)]  # full sounding record
    assert events["start"].iloc[0] == pd.Timestamp("2024-01-03 12:00")
    assert (tmp_path / "pcap_events_t5_m3.csv").exists()

    again = pcaps.get_pcap_events(TIME_RANGE, threshold=5)
    assert len(fake_soundings) == 1
    pd.testing.assert_frame_equal(again, events)

    pcaps.get_pcap_events(TIME_RANGE, min_periods=2)
    assert (tmp_path / "pcap_events_t4.04_m2.csv").exists()
    assert curated_csv.read_text() == before


def test_time_range_does_not_change_cache_file(fake_soundings, curated_csv):
    default = pcaps.get_pcap_events(TIME_RANGE)
    pd.testing.assert_frame_equal(pcaps.get_pcap_events(OTHER_RANGE), default)

    other = pcaps.get_pcap_events(TIME_RANGE, threshold=5)
    pd.testing.assert_frame_equal(
        pcaps.get_pcap_events(OTHER_RANGE, threshold=5), other
    )
    assert len(fake_soundings) == 1


def test_missing_default_cache_built_from_full_record(
    fake_soundings, monkeypatch, tmp_path
):
    # SLV_USER_DATA_DIR set after import must still be used
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(tmp_path))
    events = pcaps.get_pcap_events(OTHER_RANGE)
    assert fake_soundings == [(None, None)]
    assert (tmp_path / "pcap_events.csv").exists()
    assert len(events) == 1


def test_no_cache_without_user_data_dir(fake_soundings, tmp_path):
    events = pcaps.get_pcap_events(TIME_RANGE)
    assert fake_soundings == [TIME_RANGE]
    assert len(events) == 1
    assert not any(tmp_path.iterdir())


def test_sounding_kwargs_not_cached(fake_soundings, monkeypatch, tmp_path):
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(tmp_path))
    pcaps.get_pcap_events(TIME_RANGE, sounding_kwargs={"months": [1]})
    assert not any(tmp_path.iterdir())


def test_import_without_env_vars(monkeypatch):
    monkeypatch.delenv("SLV_SOUNDINGS_DIR", raising=False)
    importlib.reload(pcaps)


def test_get_soundings_reads_station_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv("SLV_SOUNDINGS_DIR", str(tmp_path))
    seen = {}

    def fake_lair_get_soundings(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(pcaps.lair.soundings, "get_soundings", fake_lair_get_soundings)
    pcaps.get_soundings(station="SLC")
    assert seen["sounding_dir"] == tmp_path / "SLC"


def test_filter_pcap_events_multiindex_level(monkeypatch):
    seen = {}

    def fake_get_pcap_events(time_range):
        seen["time_range"] = time_range
        return pd.DataFrame(
            {"start": [pd.Timestamp("2024-01-02")], "end": [pd.Timestamp("2024-01-03")]}
        )

    monkeypatch.setattr(pcaps, "get_pcap_events", fake_get_pcap_events)
    times = pd.date_range("2024-01-01", periods=5, freq="1D")
    index = pd.MultiIndex.from_product(
        [["hdp", "wbb"], times], names=["site", "Time_UTC"]
    )
    data = pd.DataFrame({"CH4": 2.0}, index=index)

    result = pcaps.filter_pcap_events(data, level="Time_UTC")

    assert seen["time_range"] == (times.min(), times.max())
    kept = result.index.get_level_values("Time_UTC")
    assert not kept.isin(times[1:3]).any()
    assert len(result) == 6
