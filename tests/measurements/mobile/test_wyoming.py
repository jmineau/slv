"""Tests for slv.measurements.mobile.wyoming."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from slv.measurements.mobile.wyoming import (
    calculate_enhancements,
    enhanced_R_and_ratio,
    merge_aeris_met,
    plot_windbarbs,
    read_aeris,
    read_met,
    wyomingMap,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_enhancement_baseline_window_is_in_hours():
    idx = pd.date_range(
        "2024-08-05 12:00", periods=4 * 3600, freq="s", tz="America/Denver"
    )
    ch4 = np.full(len(idx), 2000.0)
    ch4[7200:7260] += 500.0  # a one-minute plume
    data = pd.DataFrame({"CH4 (ppb)": ch4, "C2H6 (ppb)": 5.0}, index=idx)
    out = calculate_enhancements(data.copy())
    assert out.CH4_ex.iloc[7230] > 400 and abs(out.CH4_ex.iloc[0]) < 1
    # a bare number is hours, the same as the default "1h"
    same = calculate_enhancements(data.copy(), window=1)
    pd.testing.assert_series_equal(same.CH4_ex, out.CH4_ex)


def test_read_aeris_converts_ch4_to_ppb(tmp_path):
    path = tmp_path / "aeris.csv"
    path.write_text(
        "Time Stamp,CH4 (ppm),C2H6 (ppb),R,C2/C1,H2O (ppm)\n"
        "08/05/2024 18:00:01.500,2.001,5.0,0.9,0.05,9000\n"
        "not a time,2.5,5.0,0.9,0.05,9000\n"
        "08/05/2024 18:00:00.500,2.000,4.0,0.8,0.04,9000\n"
    )
    aeris = read_aeris(path)
    assert list(aeris.columns) == ["CH4 (ppb)", "C2H6 (ppb)", "R", "C2/C1"]
    assert aeris.index.is_monotonic_increasing and len(aeris) == 2  # bad time dropped
    np.testing.assert_allclose(aeris["CH4 (ppb)"], [2000.0, 2001.0])
    assert aeris.index[0] == pd.Timestamp("2024-08-05 18:00:00.500")


MET_COLUMNS = 26


def _met_row(hhmmss, lat, lon, speed_kts, direction):
    row = ["0"] * MET_COLUMNS
    row[1:5] = [hhmmss, "2024", "8", "5"]
    row[5], row[6], row[8] = str(lat), str(lon), "1300"
    row[21], row[22] = str(direction), str(speed_kts)  # GPS-corrected true dir, kts
    row[25] = "12.5"  # vehicle speed
    return ",".join(row)


def test_read_met_parses_time_and_renames(tmp_path):
    path = tmp_path / "met.csv"
    path.write_text(
        "logger line 1\nlogger line 2\nlogger line 3\nheader\n"
        + _met_row("180002", 40.72, -111.88, 10, 90)
        + "\n"
        + _met_row("180000", 40.70, -111.90, 10, 90)
        + "\n"
    )
    met = read_met(path)
    assert (
        met.index.tolist()
        == pd.to_datetime(["2024-08-05 18:00:00", "2024-08-05 18:00:02"]).tolist()
    )
    assert met.latitude.tolist() == [40.70, 40.72]
    assert met["GPSCorWindSpeed (kts)"].tolist() == [10, 10]
    assert (met.filename == path).all()


def test_merge_interpolates_met_to_aeris_times():
    met_t = pd.to_datetime(["2024-08-05 18:00:00", "2024-08-05 18:00:10"])
    met = pd.DataFrame(
        {
            "latitude": [40.70, 40.80],
            "longitude": [-111.90, -111.80],
            "Altitude (m)": [1300.0, 1310.0],
            "GPSCorWindSpeed (kts)": [10.0, 10.0],
            "GPSCorWindDirTrue (deg)": [90.0, 90.0],
            "VehicleSpeed": [0.0, 20.0],
        },
        index=met_t,
    )
    aeris = pd.DataFrame(
        {"CH4 (ppb)": [2000.0], "C2H6 (ppb)": [5.0]},
        index=pd.to_datetime(["2024-08-05 18:00:05"]),
    )
    data = merge_aeris_met(aeris, met)
    row = data.iloc[0]
    assert row.latitude == pytest.approx(40.75) and row.vehicle_speed == 10.0
    assert row.wind_direction == pytest.approx(90.0)
    assert data.index.name == "Time_MTN"
    assert data.index[0].hour == 12  # 18 UTC is 12 MDT
    assert data.crs.to_epsg() == 4326 and data.geometry.iloc[0].x == pytest.approx(
        -111.85
    )


def test_enhanced_ratio_of_a_single_source():
    idx = pd.date_range("2024-08-05 12:00", periods=120, freq="s")
    ch4_ex = np.sin(np.linspace(0, 6, 120)) + 2
    data = pd.DataFrame(
        {"CH4_ex": ch4_ex, "C2H6_ex": 0.05 * ch4_ex, "R": 0.3, "C2/C1": 0.1}, index=idx
    )
    out = enhanced_R_and_ratio(data)
    np.testing.assert_allclose(out["C2/C1"], 0.05)
    np.testing.assert_allclose(out["R"].iloc[30:], 1.0)
    assert (out["R_aeris"] == 0.3).all() and (out["C2C1_aeris"] == 0.1).all()


def _drive():
    import geopandas as gpd

    n = 50
    return gpd.GeoDataFrame(
        {
            "CH4_ex": np.linspace(0, 100, n),
            "wind_speed": 10.0,
            "wind_direction": 0.0,  # from the north
        },
        geometry=gpd.points_from_xy(np.linspace(-111.9, -111.8, n), np.full(n, 40.7)),
        crs="EPSG:4326",
    ).assign(longitude=lambda d: d.geometry.x, latitude=lambda d: d.geometry.y)


def test_plot_windbarbs_components():
    data = _drive()
    ax = plot_windbarbs(data, "wind_speed", "wind_direction")
    np.testing.assert_allclose(data["u"], 0.0, atol=1e-12)
    np.testing.assert_allclose(data["v"], -10.0)  # blowing toward the south
    assert len(ax.collections[0].get_offsets()) == len(data)


def test_wyoming_map_colours_the_drive_and_thins_the_barbs():
    ax = wyomingMap(
        _drive(),
        "CH4_ex",
        extent=(-111.95, -111.75, 40.65, 40.75),
        windskip=10,
        title="drive",
    )
    assert ax.get_title() == "drive"
    barbs = [c for c in ax.collections if type(c).__name__ == "Barbs"]
    assert len(barbs) == 1 and len(barbs[0].get_offsets()) == 5
