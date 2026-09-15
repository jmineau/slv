"""Tests for slv.measurements.trax_location (synthetic per-minute features)."""

import numpy as np
import pandas as pd

from slv.measurements.trax_location import (
    STATES,
    classify_location,
    load_depot_footprint,
    state_intervals,
)


def _feat(n, **cols):
    idx = pd.date_range("2025-03-01", periods=n, freq="1min")
    base = {
        "n_gps": 12,
        "x": 422300.0,
        "y": 4508300.0,
        "scatter": 0.1,
        "speed": 0.0,
        "speed_max": 0.0,
        "d_line": 200.0,
        "d_yard": 0.0,
        "in_yard": 1.0,
        "in_depot": False,
        "nsat": 10.0,
    }
    base.update(cols)
    return pd.DataFrame(
        {k: (v if np.ndim(v) else [v] * n) for k, v in base.items()}, index=idx
    )


def test_packaged_footprint_is_near_jrrsc():
    fp = load_depot_footprint()
    minx, miny, maxx, maxy = fp.total_bounds
    assert -111.93 < minx < maxx < -111.91 and 40.72 < miny < maxy < 40.73
    m = load_depot_footprint(meters=True).geometry.iloc[0]
    assert (
        100 < (m.bounds[2] - m.bounds[0]) < 200
        and 150 < (m.bounds[3] - m.bounds[1]) < 250
    )


def test_depot_vs_yard_from_scatter_and_nsat():
    n = 120
    scatter = np.r_[np.full(60, 0.1), np.full(60, 6.0)]
    nsat = np.r_[np.full(60, 10.0), np.full(60, 5.0)]
    f = _feat(n, scatter=scatter, nsat=nsat)
    st = classify_location(f)
    assert (st.state.iloc[:50] == "yard").all()
    assert (st.state.iloc[70:] == "depot").all()
    assert st.state.cat.categories.tolist() == list(STATES)


def test_single_minute_flip_is_smoothed_out():
    f = _feat(60)
    f.loc[f.index[30], "scatter"] = 8.0
    st = classify_location(f)
    assert st.degraded.iloc[30] and not st.degraded_smooth.iloc[30]
    assert (st.state == "yard").all()


def test_line_pass_route_stopped_and_unknown():
    f = _feat(
        5,
        speed_max=[0.0, 12.0, 12.0, 0.0, 0.0],
        d_line=[200.0, 10.0, 300.0, 5000.0, 5000.0],
        d_yard=[0.0, 0.0, 0.0, 3000.0, 3000.0],
        n_gps=[12, 12, 12, 12, 0],
    )
    st = classify_location(f, smooth_min=1)
    assert st.state.tolist() == ["yard", "line", "yard", "stopped", "unknown"]


def test_footprint_marks_frozen_fix_as_depot():
    f = _feat(20, in_depot=True)
    assert (classify_location(f).state == "yard").all()  # off by default
    assert (classify_location(f, use_footprint=True).state == "depot").all()


def test_powered_flag():
    f = _feat(3, volt_min=[13.9, 11.0, 13.0])
    assert classify_location(f).powered.tolist() == [True, False, True]


def test_state_intervals_runs_and_min_length():
    f = _feat(30, scatter=np.r_[np.full(10, 0.1), np.full(20, 6.0)])
    st = classify_location(f, smooth_min=1)
    iv = state_intervals(st)
    assert iv.state.tolist() == ["yard", "depot"] and iv.minutes.tolist() == [10, 20]
    assert iv.start.iloc[1] == f.index[10] and iv.end.iloc[1] == f.index[-1]
    assert state_intervals(st, min_minutes=15).state.tolist() == ["depot"]


def test_label_observations_floors_to_minute():
    from slv.measurements.trax_location import label_observations

    f = _feat(3, scatter=[0.1, 6.0, 6.0])
    st = classify_location(f, smooth_min=1)
    obs = pd.DataFrame(
        {
            "Time_UTC": [
                f.index[0] + pd.Timedelta("20s"),
                f.index[2] + pd.Timedelta("59s"),
                f.index[2] + pd.Timedelta("5min"),
            ],
            "CH4": [2.0, 2.1, 2.2],
        }
    )
    lab = label_observations(obs, st)
    assert lab.tolist()[:2] == ["yard", "depot"] and pd.isna(lab.iloc[2])
