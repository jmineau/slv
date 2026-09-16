"""Tests for slv.measurements.mobile: calibration windows, obs location filter, route buffer."""

import pandas as pd

from slv.measurements.mobile.calibration import (
    CAL_SOURCES,
    filter_cal_source,
    load_trax_uncalibrated_windows,
    select_uncalibrated,
)


def test_packaged_windows_parse():
    w = load_trax_uncalibrated_windows(enabled_only=False)
    assert list(w.columns) == ["start", "end", "reason", "enabled"]
    assert (w.end > w.start).all()
    assert w.enabled.dtype == bool
    # the Jan-Feb 2016 tank-out window is the first row and enabled
    assert w.start.iloc[0] == pd.Timestamp("2016-01-01") and w.enabled.iloc[0]


def test_select_uncalibrated_respects_windows_and_exclusions():
    windows = pd.DataFrame(
        {
            "start": [pd.Timestamp("2016-01-01")],
            "end": [pd.Timestamp("2016-03-12")],
            "reason": ["test"],
            "enabled": [True],
        }
    )
    t = pd.to_datetime(
        [
            "2015-12-31 23:00",
            "2016-01-15 12:00",
            "2016-02-01 00:00",
            "2016-03-12 00:00",
            "2016-04-01 00:00",
        ]
    )
    qaqc = pd.DataFrame({"Time_UTC": t, "CH4": [2.0, 2.1, float("nan"), 2.2, 2.3]})
    out = select_uncalibrated(qaqc, windows)
    # inside the half-open window, non-NaN only
    assert list(out.Time_UTC) == [pd.Timestamp("2016-01-15 12:00")]
    assert (out.cal_source == "uncalibrated").all()
    # times that already have a pipeline calibration are dropped
    out2 = select_uncalibrated(
        qaqc, windows, exclude_times=pd.Index([pd.Timestamp("2016-01-15 12:00")])
    )
    assert out2.empty


def test_filter_cal_source_switch():
    df = pd.DataFrame({"CH4_ppm": [2.0, 2.1, 2.2], "cal_source": list(CAL_SOURCES)})
    assert len(filter_cal_source(df, include_uncalibrated=True)) == 3
    kept = filter_cal_source(df, include_uncalibrated=False)
    assert set(kept.cal_source) == {"pipeline", "manual_cal"}


def test_filter_location_sets():
    from slv.measurements.mobile.obs import LOCATION_SETS, filter_location

    df = pd.DataFrame(
        {
            "state": ["route", "line", "stopped", "yard", "depot", "unknown"],
            "CH4": range(6),
        }
    )
    assert filter_location(df).state.tolist() == ["route", "line", "stopped"]
    assert filter_location(df, "outdoor").state.tolist() == [
        "route",
        "line",
        "stopped",
        "yard",
    ]
    assert len(filter_location(df, "all")) == 6 and len(filter_location(df, None)) == 6
    assert filter_location(df, ("depot",)).state.tolist() == ["depot"]
    assert set(LOCATION_SETS) == {"on_track", "outdoor", "all"}
    # frames without a state column pass through untouched
    assert len(filter_location(df.drop(columns="state"))) == 6


def test_label_trax_location_with_given_states():
    from slv.measurements.mobile.obs import label_trax_location

    idx = pd.date_range("2025-03-01", periods=3, freq="1min")
    states = pd.DataFrame(
        {
            "state": pd.Categorical(["yard", "depot", "line"]),
            "indoor": pd.array([False, True, False], dtype="boolean"),
            "yard_name": ["JRRSC", "JRRSC", None],
        },
        index=idx,
    )
    obs = pd.DataFrame(
        {
            "Time_UTC": [
                idx[0] + pd.Timedelta("10s"),
                idx[1] + pd.Timedelta("59s"),
                idx[2],
                idx[2] + pd.Timedelta("5min"),
            ],
            "CH4_ppm": [2.0, 2.5, 2.1, 2.2],
        }
    )
    out = label_trax_location(obs, states=states)
    assert out.state.tolist() == ["yard", "depot", "line", "unknown"]
    assert out.indoor.tolist()[:3] == [False, True, False] and pd.isna(
        out.indoor.iloc[3]
    )
    assert out.yard_name.tolist()[:2] == ["JRRSC", "JRRSC"]


def test_filter_near_routes_does_not_duplicate_on_shared_track():
    import geopandas as gpd
    from shapely.geometry import LineString

    from slv.measurements.mobile.network import filter_near_routes

    # two lines sharing a trunk (x 0..100), then diverging
    routes = gpd.GeoDataFrame(
        {"line": ["R", "B"]},
        geometry=[
            LineString([(0, 0), (100, 0), (200, 50)]),
            LineString([(0, 0), (100, 0), (200, -50)]),
        ],
        crs="EPSG:32612",
    )
    pts = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=gpd.points_from_xy(
            [50, 190, 150], [5, 45, 300]
        ),  # trunk, R branch, far away
        crs="EPSG:32612",
    ).to_crs("EPSG:4326")
    out = filter_near_routes(pts, routes, 20)
    assert out.id.tolist() == [1, 2]  # point 1 once, not twice
