"""Tests for aggregate_obs."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from slv.measurements.aggregate import aggregate_obs

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SITE_COORDS = {
    "A": (40.76, -111.89, 10.0),
    "B": (40.72, -111.83, 15.0),
}


def make_stationary(n_per_hour=30, hours=2, sites=("A", "B"), instrument="LGR_UGGA"):
    """Build a minimal stationary observations DataFrame."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    for site in sites:
        lat, lon, height = SITE_COORDS[site]
        for h in range(hours):
            for i in range(n_per_hour):
                rows.append(
                    {
                        "Time_UTC": base
                        + pd.Timedelta(hours=h)
                        + pd.Timedelta(seconds=i * (3600 // n_per_hour)),
                        "site": site,
                        "instrument": instrument,
                        "is_mobile": False,
                        "latitude": lat,
                        "longitude": lon,
                        "height": height,
                        "CH4": 1.9 + 0.01 * i,
                        "CO2": 400.0,
                    }
                )
    return pd.DataFrame(rows)


def make_mobile(n_per_bin=5, n_bins=3, instrument="LGR_UGGA"):
    """Build a minimal mobile observations DataFrame with distinct spatial clusters."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    lats = [40.72, 40.83, 40.94]
    lons = [-111.92, -111.83, -111.74]
    for b in range(n_bins):
        for i in range(n_per_bin):
            rows.append(
                {
                    "Time_UTC": base + pd.Timedelta(seconds=b * 60 + i * 10),
                    "site": "mobile",
                    "instrument": instrument,
                    "is_mobile": True,
                    "latitude": lats[b] + np.random.uniform(-0.001, 0.001),
                    "longitude": lons[b] + np.random.uniform(-0.001, 0.001),
                    "altitude": 1300.0,
                    "height": 2.0,
                    "CH4": 1.9 + 0.01 * i,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stationary: basic aggregation
# ---------------------------------------------------------------------------


class TestStationaryAggregation:
    def test_no_aggregation_returns_original(self):
        obs = make_stationary()
        result = aggregate_obs(obs.copy())
        assert len(result) == len(obs)

    def test_freq_groups_by_site_and_hour(self):
        obs = make_stationary(hours=3, sites=("A", "B"))
        result = aggregate_obs(obs.copy(), freq="h")
        # 2 sites × 3 hours
        assert len(result) == 6
        assert "Time_UTC" in result.columns
        assert "CH4" in result.columns

    def test_freq_mean_is_correct(self):
        obs = make_stationary(n_per_hour=10, hours=1, sites=("A",))
        result = aggregate_obs(obs.copy(), freq="h")
        assert result["CH4"].iloc[0] == pytest.approx(obs["CH4"].mean())

    def test_custom_func_median(self):
        obs = make_stationary(n_per_hour=10, hours=1, sites=("A",))
        result = aggregate_obs(obs.copy(), freq="h", func="median")
        assert result["CH4"].iloc[0] == pytest.approx(obs["CH4"].median())

    def test_numeric_cols_auto_detected(self):
        obs = make_stationary()
        result = aggregate_obs(obs.copy(), freq="h")
        assert "CH4" in result.columns
        assert "CO2" in result.columns

    def test_instrument_joined_after_aggregation(self):
        obs = make_stationary()
        result = aggregate_obs(obs.copy(), freq="h")
        assert "instrument" in result.columns
        assert result["instrument"].iloc[0] == "LGR_UGGA"

    def test_location_columns_preserved_after_aggregation(self):
        obs = make_stationary(n_per_hour=10, hours=1, sites=("A", "B"))
        result = aggregate_obs(obs.copy(), freq="h")
        for site, (lat, lon, height) in SITE_COORDS.items():
            row = result[result["site"] == site].iloc[0]
            assert row["latitude"] == pytest.approx(lat)
            assert row["longitude"] == pytest.approx(lon)
            assert row["height"] == pytest.approx(height)

    def test_no_is_mobile_column_treated_as_stationary(self):
        obs = make_stationary().drop(columns=["is_mobile"])
        result = aggregate_obs(obs.copy(), freq="h")
        assert not result.empty


# ---------------------------------------------------------------------------
# Stationary: min_percent filtering
# ---------------------------------------------------------------------------


class TestStationaryMinPercent:
    def test_full_coverage_passes(self):
        # LGR_UGGA has samples_per_hour=309; give it enough samples
        obs = make_stationary(
            n_per_hour=309, hours=1, sites=("A",), instrument="LGR_UGGA"
        )
        result = aggregate_obs(obs.copy(), freq="h", stationary_min_percent=0.5)
        assert len(result) == 1

    def test_sparse_group_filtered_out(self):
        # 1 sample out of 309 expected → well below any reasonable threshold
        obs = make_stationary(
            n_per_hour=1, hours=1, sites=("A",), instrument="LGR_UGGA"
        )
        with pytest.raises(ValueError, match="No data to aggregate"):
            aggregate_obs(obs.copy(), freq="h", stationary_min_percent=0.5)

    def test_mixed_coverage_filters_only_sparse(self):
        dense = make_stationary(
            n_per_hour=200, hours=1, sites=("A",), instrument="LGR_UGGA"
        )
        sparse = make_stationary(
            n_per_hour=1, hours=1, sites=("B",), instrument="LGR_UGGA"
        )
        obs = pd.concat([dense, sparse], ignore_index=True)
        result = aggregate_obs(obs.copy(), freq="h", stationary_min_percent=0.5)
        assert list(result["site"]) == ["A"]

    def test_fallback_to_sample_rate_for_instrument_without_samples_per_hour(self):
        # Picarro_G2401 has sample_rate='2s' but no samples_per_hour
        # At 2s rate, expected per hour = 1800; give it enough
        obs = make_stationary(
            n_per_hour=1000, hours=1, sites=("A",), instrument="Picarro_G2401"
        )
        result = aggregate_obs(obs.copy(), freq="h", stationary_min_percent=0.5)
        assert len(result) == 1

    def test_no_min_percent_no_filtering(self):
        obs = make_stationary(n_per_hour=1, hours=1, sites=("A",))
        result = aggregate_obs(obs.copy(), freq="h")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Mobile: spatial aggregation
# ---------------------------------------------------------------------------


class TestMobileAggregation:
    def test_grid_aggregation_reduces_rows(self):
        obs = make_mobile(n_per_bin=10, n_bins=3)
        result = aggregate_obs(obs.copy(), mobile_grid_res=0.1)
        assert len(result) < len(obs)
        assert "latitude" in result.columns
        assert "longitude" in result.columns

    def test_nearest_point_aggregation(self):
        obs = make_mobile(n_per_bin=5, n_bins=3)
        points = gpd.GeoDataFrame(
            geometry=[Point(-111.9, 40.7), Point(-111.85, 40.8), Point(-111.8, 40.9)],
            crs="EPSG:4326",
        )
        result = aggregate_obs(obs.copy(), mobile_points=points)
        assert len(result) == 3
        assert "latitude" in result.columns
        assert "longitude" in result.columns

    def test_grid_plus_freq(self):
        obs = make_mobile(n_per_bin=5, n_bins=3)
        result = aggregate_obs(obs.copy(), mobile_grid_res=0.1, freq="h")
        assert "Time_UTC" in result.columns


# ---------------------------------------------------------------------------
# Mobile: min_count filtering
# ---------------------------------------------------------------------------


class TestMobileMinCount:
    def test_bins_below_min_count_removed(self):
        # bin 0: 10 samples, bins 1-2: 2 samples each
        rows = []
        base = pd.Timestamp("2024-01-01")
        for i in range(10):
            rows.append(
                {
                    "Time_UTC": base + pd.Timedelta(seconds=i),
                    "site": "m",
                    "instrument": "LGR_UGGA",
                    "is_mobile": True,
                    "latitude": 40.70,
                    "longitude": -111.90,
                    "altitude": 0.0,
                    "height": 2.0,
                    "CH4": 2.0,
                }
            )
        for b, (lat, lon) in enumerate([(40.80, -111.85), (40.90, -111.80)]):
            for i in range(2):
                rows.append(
                    {
                        "Time_UTC": base + pd.Timedelta(seconds=100 + b * 10 + i),
                        "site": "m",
                        "instrument": "LGR_UGGA",
                        "is_mobile": True,
                        "latitude": lat,
                        "longitude": lon,
                        "altitude": 0.0,
                        "height": 2.0,
                        "CH4": 2.0,
                    }
                )
        obs = pd.DataFrame(rows)
        result = aggregate_obs(obs, mobile_grid_res=0.1, mobile_min_count=5)
        assert len(result) == 1
        assert result["latitude"].iloc[0] == pytest.approx(40.7, abs=0.05)

    def test_no_min_count_keeps_all_bins(self):
        obs = make_mobile(n_per_bin=1, n_bins=3)
        result = aggregate_obs(obs.copy(), mobile_grid_res=0.1)
        assert len(result) == 3

    def test_all_bins_pass_min_count(self):
        obs = make_mobile(n_per_bin=10, n_bins=3)
        result = aggregate_obs(obs.copy(), mobile_grid_res=0.1, mobile_min_count=5)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Mixed stationary + mobile
# ---------------------------------------------------------------------------


class TestMixed:
    def test_stationary_and_mobile_both_returned(self):
        stationary = make_stationary(n_per_hour=10, hours=1, sites=("A",))
        mobile = make_mobile(n_per_bin=5, n_bins=2)
        obs = pd.concat([stationary, mobile], ignore_index=True)
        result = aggregate_obs(obs, freq="h", mobile_grid_res=0.1)
        assert not result.empty
        assert "CH4" in result.columns

    def test_empty_obs_raises(self):
        obs = pd.DataFrame(columns=["Time_UTC", "site", "is_mobile", "CH4"])
        with pytest.raises(ValueError, match="No data to aggregate"):
            aggregate_obs(obs)
