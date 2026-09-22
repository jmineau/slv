"""Tests for slv.measurements.mobile: calibration windows, obs location filter, route buffer."""

import numpy as np
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


def test_low_pressure_rule_keeps_band_and_tags():
    from slv.measurements.mobile.obs import LOW_PRESSURE_BAND, apply_low_pressure_rule

    df = pd.DataFrame(
        {
            "Time_UTC": pd.date_range("2026-08-01", periods=5, freq="10s"),
            "CH4": [2.0, 2.1, 2.2, 2.3, 2.4],
            "QAQC_Flag": [0, -63, -63, -63, -63],
            "Cavity_P_torr": [140.0, 132.0, 105.0, 30.0, float("nan")],
        }
    )
    out = apply_low_pressure_rule(df, LOW_PRESSURE_BAND)
    # flag 0 untouched, -63 inside 100-145 kept and tagged, collapse (30 torr) and NaN dropped
    assert out.CH4.notna().tolist() == [True, True, True, False, False]
    assert out.low_pressure.tolist() == [False, True, True, False, False]
    # band=None drops every -63 row
    out2 = apply_low_pressure_rule(df, None)
    assert out2.CH4.notna().tolist() == [True, False, False, False, False]
    assert not out2.low_pressure.any()


def test_build_chunk_edges_cover_range():
    """The year chunking used by build_trax_obs covers the range without gaps or overlaps."""
    t0, t1 = pd.Timestamp("2014-12-09"), pd.Timestamp("2016-03-05")
    edges = pd.date_range(t0, t1, freq="YS")
    edges = pd.DatetimeIndex([t0, *edges[(edges > t0) & (edges < t1)], t1])
    assert edges.tolist() == [
        t0,
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2016-01-01"),
        t1,
    ]


def test_slope_guard_drops_outlier_slope_rows():
    from slv.measurements.mobile.obs import apply_slope_guard

    t = pd.date_range("2019-06-22", periods=300, freq="min")
    m = pd.Series(1.0, index=range(300))
    m.iloc[100:160] = 0.7  # one bad reference period interpolated over an hour
    df = pd.DataFrame({"Time_UTC": t, "CH4": 2.0, "CH4d_m": m})
    out = apply_slope_guard(df, 0.05)
    assert out.CH4.isna().sum() == 60 and out.CH4.iloc[:100].notna().all()
    # no-op cases: tol=None, missing column, too few rows for a daily median
    assert apply_slope_guard(df, None).CH4.notna().all()
    assert apply_slope_guard(df.drop(columns="CH4d_m"), 0.05).CH4.notna().all()
    assert apply_slope_guard(df.iloc[:50], 0.05).CH4.notna().all()


def test_slope_guard_accepts_string_slopes():
    """uataq multiprocess reads can hand back the slope column as strings."""
    from slv.measurements.mobile.obs import apply_slope_guard

    t = pd.date_range("2019-06-22", periods=200, freq="min")
    m = ["1.0"] * 150 + ["0.7"] * 50
    df = pd.DataFrame(
        {"Time_UTC": t, "CH4": 2.0, "CH4d_m": pd.array(m, dtype="string[pyarrow]")}
    )
    out = apply_slope_guard(df, 0.05)
    assert out.CH4.isna().sum() == 50


def test_gps_numeric_coercion_handles_all_na_string_columns():
    from slv.measurements.mobile.gps import _coerce_numeric

    df = pd.DataFrame(
        {
            "Latitude_deg": [40.7, 40.8],
            "Speed_m_s": pd.array([None, None], dtype="string[pyarrow]"),
            "Status": ["A", "A"],
        }
    )
    out = _coerce_numeric(df)
    assert out["Speed_m_s"].dtype == np.float64 and out["Speed_m_s"].isna().all()
    assert out["Latitude_deg"].dtype == np.float64
    assert out["Latitude_deg"].tolist() == [40.7, 40.8]
    assert out["Status"].tolist() == ["A", "A"]  # non-GPS columns untouched
    # a plain float column takes a numeric reduction (the Arrow string one raised)
    assert np.isnan(out["Speed_m_s"].median())


# --------------------------------------------------------------------------- load / build


def test_load_trax_obs_applies_the_location_filter(tmp_path):
    from slv.measurements.mobile.obs import load_trax_obs

    states = ["route", "line", "stopped", "yard", "depot", "unknown"]
    pd.DataFrame(
        {
            "Time_UTC": pd.date_range("2024-01-01", periods=6, freq="min"),
            "CH4_ppm": 2.0,
            "Latitude_deg": 40.7,
            "Longitude_deg": -111.9,
            "cal_source": "pipeline",
            "low_pressure": False,
            "state": states,
        }
    ).to_parquet(tmp_path / "obs.parquet")
    cache = tmp_path / "obs.parquet"
    assert load_trax_obs(cache).state.tolist() == ["route", "line", "stopped"]
    assert load_trax_obs(cache, location="outdoor").state.tolist() == states[:4]
    assert load_trax_obs(cache, location="all").state.tolist() == states


def _fake_read_lgr(tables):
    """Stand-in for ``obs._read_lgr``: ``tables[(instrument, lvl)]`` is a ``Time_UTC``/``CH4``
    frame, cut to ``time_range`` including both ends as uataq does; a missing entry raises
    ReaderError like a level with no files."""
    import uataq

    def read(site, instrument, lvl, value_col, time_range, num_processes, **kwargs):
        if (instrument, lvl) not in tables:
            raise uataq.errors.ReaderError(f"no {instrument} {lvl} files")
        t = tables[(instrument, lvl)]
        t0, t1 = (pd.Timestamp(x) for x in time_range)
        return t[(t.Time_UTC >= t0) & (t.Time_UTC <= t1)].reset_index(drop=True)

    return read


def _fake_merge_with_gps(site, org, obs, **kwargs):
    return obs.assign(Latitude_deg=40.7, Longitude_deg=-111.9)


def _build(monkeypatch, tables, time_range, windows=None, **kwargs):
    from slv.measurements.mobile import obs as obs_mod

    monkeypatch.setattr(obs_mod, "_read_lgr", _fake_read_lgr(tables))
    monkeypatch.setattr(obs_mod, "merge_with_gps", _fake_merge_with_gps)
    if windows is None:
        windows = pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([])})
    return obs_mod.build_trax_obs(
        time_range=time_range, windows=windows, classify=False, **kwargs
    )


def test_build_keeps_pipeline_over_manual_cal_over_uncalibrated(monkeypatch):
    t = pd.date_range("2024-08-21", periods=3000, freq="s")
    tables = {
        ("lgr_ugga", "calibrated"): pd.DataFrame({"Time_UTC": t[:1000], "CH4": 2.0}),
        ("lgr_ugga_manual_cal", "qaqc"): pd.DataFrame(
            {"Time_UTC": t[:2000], "CH4": 2.1}
        ),
        ("lgr_ugga", "qaqc"): pd.DataFrame({"Time_UTC": t, "CH4": 2.2}),
    }
    windows = pd.DataFrame({"start": [t[0]], "end": [t[-1] + pd.Timedelta("1s")]})
    out = _build(monkeypatch, tables, (t[0], t[-1]), windows)
    assert out.cal_source.tolist() == (
        ["pipeline"] * 1000 + ["manual_cal"] * 1000 + ["uncalibrated"] * 1000
    )


def test_build_skips_an_unreadable_uncalibrated_window(monkeypatch):
    t = pd.date_range("2016-01-10", periods=10, freq="10s")
    tables = {("lgr_ugga", "calibrated"): pd.DataFrame({"Time_UTC": t, "CH4": 2.0})}
    windows = pd.DataFrame({"start": [t[0]], "end": [t[-1]]})
    out = _build(monkeypatch, tables, (t[0], t[-1]), windows)
    assert len(out) == 10 and (out.cal_source == "pipeline").all()


def test_build_with_no_data_returns_empty_frame_with_columns(monkeypatch):
    import geopandas as gpd

    out = _build(monkeypatch, {}, ("2016-01-01", "2016-02-01"))
    assert isinstance(out, gpd.GeoDataFrame) and out.empty
    assert {"Time_UTC", "CH4_ppm", "cal_source", "low_pressure"} <= set(out.columns)
    assert str(out.crs) == "EPSG:4326"


def test_build_chunks_do_not_duplicate_a_row_on_the_edge(monkeypatch):
    t = pd.to_datetime(
        ["2015-12-31 23:59:50", "2016-01-01 00:00:00", "2016-01-01 00:00:10"]
    )
    tables = {("lgr_ugga", "calibrated"): pd.DataFrame({"Time_UTC": t, "CH4": 2.0})}
    out = _build(monkeypatch, tables, ("2015-12-01", "2016-02-01"))
    assert out.Time_UTC.tolist() == list(t)


def test_merge_with_gps_reads_the_requested_site(monkeypatch):
    import numpy as np
    import pytest
    import uataq

    from slv.measurements.mobile.gps import merge_with_gps

    t = pd.date_range("2024-06-01", periods=100, freq="s")
    gps = pd.DataFrame(
        {
            "Pi_Time": t,
            "Latitude_deg": 40.7,
            "Longitude_deg": np.linspace(-111.95, -111.90, 100),
            "Altitude_msl": np.linspace(1280.0, 1300.0, 100),
        },
        index=pd.Index(t, name="Time_UTC"),
    )
    seen = []

    def fake_read(SID, **kwargs):
        seen.append(SID)
        return {"gps": gps}

    monkeypatch.setattr(uataq, "read_data", fake_read)
    obs = pd.DataFrame({"Time_UTC": t, "CH4": 2.0})
    out = merge_with_gps("trx02", "UATAQ", obs, routes=False, storage_polygon=False)
    assert seen == ["trx02"] and len(out) and "Latitude_deg" in out
    with pytest.raises(ValueError, match="not supported"):
        merge_with_gps("trx02", "horel", obs, routes=False, storage_polygon=False)
