"""Tests for the TRAX uncalibrated-window handling in slv.measurements.mobile."""

import pandas as pd

from slv.measurements.mobile import (
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
