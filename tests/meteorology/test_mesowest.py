"""Tests for slv.meteorology.mesowest, on synthetic archives in a tmp SLV_USER_DATA_DIR."""

import numpy as np
import pandas as pd
import pytest

from slv.meteorology import mesowest

TIMES = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")


def _hourly(u, v, direction):
    speed = np.hypot(u, v)
    return pd.DataFrame(
        {
            "Time": TIMES,
            "wind_speed_set_1": speed,
            "wind_direction_set_1": direction,
            "Uwind": u,
            "Vwind": v,
            "air_temp_set_1": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def archive(monkeypatch, tmp_path):
    """Fixed pull with FIX1 + BOTH; our pull with BOTH only; metadata for all three."""
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(tmp_path))
    root = tmp_path / "mesowest"
    (root / "hourly").mkdir(parents=True)
    (root / "synoptic" / "hourly").mkdir(parents=True)
    # Northerly hour mixing 350 and 10 deg: the vector mean is 0/360, but the fixed
    # pull stored the arithmetic mean of the degrees (180). Last hour is dead calm.
    u = np.array([0.0, 5.0, 0.0, 0.0])
    v = np.array([-5.0, 0.0, 5.0, 0.0])
    _hourly(u, v, [180.0, 270.0, 180.0, 0.0]).to_parquet(
        root / "hourly" / "FIX1_hourly.parquet"
    )
    _hourly(u, v, [180.0, 270.0, 180.0, 0.0]).to_parquet(
        root / "hourly" / "BOTH_hourly.parquet"
    )
    ours = _hourly(u, v, [0.0, 270.0, 180.0, np.nan])
    ours["air_temp_set_1"] = 99.0  # marks which archive was read
    ours.to_parquet(root / "synoptic" / "hourly" / "BOTH_hourly.parquet")
    ours.to_parquet(root / "synoptic" / "hourly" / "NEW_hourly.parquet")
    pd.DataFrame(
        {
            "stid": ["BOTH", "NEW"],
            "name": ["b (synoptic)", "n"],
            "latitude": [40.61, 40.8],
            "longitude": [-112.01, -111.8],
            "elevation_ft": [4300.0, 4500.0],
            "network_id": ["9", "153"],
        }
    ).to_csv(root / "synoptic" / "stations.csv", index=False)
    pd.DataFrame(
        {
            "station_code": ["FIX1", "BOTH", "NOFILE"],
            "station_name": ["a", "b", "c"],
            "LAT": [40.7, 40.6, 40.5],
            "LON": [-111.9, -112.0, -112.1],
            "ELEVATION.ft": [4200, 4300, 4400],
        }
    ).to_csv(root / "stations_metadata.csv", index=False)
    return root


def test_direction_is_the_vector_mean(archive):
    df = mesowest.station_hourly("FIX1")
    # northerly (U=0, V<0 means wind blowing toward the south, i.e. from the north)
    assert df["wind_direction"].iloc[0] % 360 == pytest.approx(0.0)
    assert df["wind_direction"].iloc[1] == pytest.approx(270.0)
    assert df["wind_direction"].iloc[2] == pytest.approx(180.0)


def test_dead_calm_has_no_direction(archive):
    assert np.isnan(mesowest.station_hourly("FIX1")["wind_direction"].iloc[3])


def test_our_pull_is_preferred(archive):
    assert (mesowest.station_hourly("BOTH")["air_temp"] == 99.0).all()
    assert (mesowest.station_hourly("FIX1")["air_temp"] < 99.0).all()


def test_metadata_lists_stations_from_both_archives(archive):
    meta = mesowest.load_station_metadata()
    assert set(meta.index) == {"FIX1", "BOTH", "NEW"}
    assert meta.loc["NEW", "latitude"] == pytest.approx(40.8)  # only in our pull
    assert meta.loc["BOTH", "station_name"] == "b (synoptic)"  # ours wins


def test_missing_station_raises(archive):
    with pytest.raises(FileNotFoundError, match="NOFILE"):
        mesowest.station_hourly("NOFILE")


def test_direction_kept_where_uv_missing(archive):
    root = archive
    df = _hourly(
        np.array([np.nan] * 4), np.array([np.nan] * 4), [10.0, 20.0, 30.0, 40.0]
    )
    df.to_parquet(root / "hourly" / "NOUV_hourly.parquet")
    out = mesowest.station_hourly("NOUV")
    assert out["wind_direction"].tolist() == [10.0, 20.0, 30.0, 40.0]
