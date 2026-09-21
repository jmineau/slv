"""Tests for slv.measurements.mobile.location (synthetic per-minute features)."""

import numpy as np
import pandas as pd

from slv.measurements.mobile.location import (
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
        "d_track": 200.0,
        "d_yard": 0.0,
        "in_yard": 1.0,
        "in_depot": False,
        "nsat": 10.0,
    }
    base.update(cols)
    return pd.DataFrame(
        {k: (v if np.ndim(v) else [v] * n) for k, v in base.items()}, index=idx
    )


def test_packaged_footprints_and_yards():
    from slv.measurements.mobile.location import load_storage_polygons

    fp = load_depot_footprint()
    assert sorted(fp.name) == ["JRRSC", "MRSC"]
    j = fp[fp.name == "JRRSC"].total_bounds
    assert -111.93 < j[0] < j[2] < -111.91 and 40.72 < j[1] < j[3] < 40.73
    m = fp[fp.name == "MRSC"].total_bounds
    assert -111.91 < m[0] < m[2] < -111.90 and 40.62 < m[1] < m[3] < 40.64
    yards = load_storage_polygons(meters=True)
    assert sorted(yards.name) == ["JRRSC", "MRSC"]
    # each shed footprint lies (almost) inside its yard polygon
    fpm = load_depot_footprint(meters=True)
    for site in ("JRRSC", "MRSC"):
        shed = fpm[fpm.name == site].geometry.iloc[0]
        yard = yards[yards.name == site].geometry.iloc[0]
        assert shed.intersection(yard).area / shed.area > 0.8


def test_depot_vs_yard_from_scatter_and_nsat():
    n = 120
    scatter = np.r_[np.full(60, 0.1), np.full(60, 6.0)]
    nsat = np.r_[np.full(60, 10.0), np.full(60, 5.0)]
    f = _feat(n, scatter=scatter, nsat=nsat)
    st = classify_location(f)
    assert (st.state.iloc[:50] == "yard").all()
    assert (st.state.iloc[70:] == "depot").all()
    assert not st.indoor.iloc[:50].any() and st.indoor.iloc[70:].all()
    assert st.state.cat.categories.tolist() == list(STATES)


def test_single_minute_flip_is_smoothed_out():
    f = _feat(60)
    f.loc[f.index[30], "scatter"] = 8.0
    st = classify_location(f)
    assert st.degraded.iloc[30] and not st.degraded_smooth.iloc[30]
    assert (st.state == "yard").all()


def test_line_pass_route_stopped_and_unknown():
    f = _feat(
        6,
        speed_max=[0.0, 12.0, 12.0, 12.0, 0.0, 0.0],
        d_track=[200.0, 10.0, 200.0, 5.0, 5.0, 5.0],
        d_yard=[0.0, 0.0, 0.0, 3000.0, 3000.0, 3000.0],
        n_gps=[12, 12, 12, 12, 12, 0],
    )
    st = classify_location(f, smooth_min=1)
    assert st.state.tolist() == ["yard", "line", "yard", "route", "stopped", "unknown"]
    assert st.indoor.tolist() == [False, False, False, False, False, pd.NA]


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
    from slv.measurements.mobile.location import label_observations

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


def test_speed_est_fallback_when_no_speed_recorded():
    # no recorded speed: minutes 1-2 move 500 m/min (speed_est ~4 m/s), rest still
    n = 6
    x = np.array([0.0, 0.0, 500.0, 1000.0, 1000.0, 1000.0]) + 422300.0
    est = np.hypot(np.r_[np.nan, x[2:] - x[:-2], np.nan], 0) / 120
    f = _feat(n, x=x, speed_max=np.nan, speed_est=est, d_track=300.0)
    st = classify_location(f, smooth_min=1)
    assert st.state.tolist()[1:4] == ["yard", "yard", "yard"]  # moving in yard -> yard
    f["d_track"] = 10.0
    st = classify_location(f, smooth_min=1)
    assert st.state.tolist() == ["yard", "line", "line", "line", "yard", "yard"]
    # a recorded speed wins over the estimate
    f["speed_max"] = 0.0
    assert (classify_location(f, smooth_min=1).state == "yard").all()


def test_location_features_without_speed_column(monkeypatch):
    import geopandas as gpd
    from shapely import LineString

    import slv.measurements.mobile.location as location
    from slv.measurements.mobile.location import location_features

    # A stand-in track through the fixes, instead of UTA_TRAX.geojson from group data
    # (the yard polygons are packaged).
    track = gpd.GeoDataFrame(
        geometry=[LineString([(-111.921, 40.7234), (-111.917, 40.7237)])],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(
        location, "load_trax_lines", lambda meters=False: track.to_crs("EPSG:32612")
    )

    idx = pd.date_range("2025-03-01", periods=180, freq="1s")
    gps = pd.DataFrame(
        {
            "Latitude_deg": np.linspace(40.7235, 40.7236, 180),
            "Longitude_deg": np.linspace(-111.92, -111.918, 180),
        },
        index=idx,
    )
    f = location_features(gps)
    assert f.yard_name.tolist() == ["JRRSC"] * 3
    assert len(f) == 3 and f.speed_max.isna().all() and f.speed_est.notna().sum() == 1
    assert f.speed_est.dropna().iloc[0] > 0.5


def test_untrusted_positions_are_unknown():
    # near the yard but neither on the line nor inside the buffer -> unknown (ejecta);
    # far from the yard and > 100 m off any track -> unknown; on-track stop stays stopped
    f = _feat(
        4,
        d_yard=[120.0, 120.0, 3000.0, 3000.0],
        d_track=[150.0, 10.0, 130.0, 5.0],
        speed_max=[0.0, 12.0, 0.0, 0.0],
    )
    st = classify_location(f, smooth_min=1)
    assert st.state.tolist() == ["unknown", "line", "unknown", "stopped"]
    # inside the yard buffer the track distance is irrelevant (yard tracks are unmapped)
    f = _feat(2, d_yard=0.0, d_track=250.0)
    assert (classify_location(f, smooth_min=1).state == "yard").all()
