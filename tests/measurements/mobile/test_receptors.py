"""Tests for slv.measurements.mobile.receptors on a synthetic straight-line network."""

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

from slv.measurements.mobile.network import UTM12
from slv.measurements.mobile.receptors import (
    RECEPTOR_COLUMNS,
    build_trax_receptors,
    find_segment_crossings,
)

X0, Y0 = 420000.0, 4508000.0
TO_LONLAT = Transformer.from_crs(UTM12, "EPSG:4326", always_xy=True)


def _network(n=120, spacing=50.0, seg_points=40):
    """n points east-west at y=Y0, `seg_points` per segment. Straight line, so the
    along-route coordinate is just the distance east of the first point."""
    x = X0 + spacing * np.arange(n)
    return gpd.GeoDataFrame(
        {
            "point": np.arange(n),
            "lines": "R",
            "segment": np.arange(n) // seg_points,
            "s_R": x - X0,
        },
        geometry=gpd.points_from_xy(x, np.full(n, Y0)),
        crs=UTM12,
    )


def _fixes(speed=10.0, t0="2024-06-01 12:00:00", gap_min=20, length=5950.0, step=1.0):
    """Out along x at `speed` m/s (1 fix per `step` s), a gap, then back."""
    xs = np.arange(0.0, length + 1e-6, speed * step)
    out_t = pd.Timestamp(t0) + pd.to_timedelta(np.arange(len(xs)) * step, unit="s")
    back_t = (
        out_t[-1]
        + pd.Timedelta(minutes=gap_min)
        + pd.to_timedelta(np.arange(len(xs)) * step, unit="s")
    )
    x = np.r_[X0 + xs, X0 + xs[::-1]]
    lon, lat = TO_LONLAT.transform(x, np.full(len(x), Y0))
    return pd.DataFrame(
        {"Time_UTC": np.r_[out_t, back_t], "Latitude_deg": lat, "Longitude_deg": lon}
    )


def test_crossings_split_by_segment_and_gap():
    pts = _network()
    cr = find_segment_crossings(_fixes(), pts)
    assert len(cr) == 6  # 3 segments out, 3 back
    assert cr.segment.tolist() == [0, 1, 2, 2, 1, 0]
    assert (cr.n_segment_points == 40).all()
    assert (cr.n_points == 40).all()  # 1-s fixes at 10 m/s hit every 50-m point
    assert (cr.span_m.between(1900, 2000)).all()
    assert (cr.coverage > 0.97).all()
    assert (cr.segment_extent_m == 1950).all()
    assert (cr.n_fix.between(195, 205)).all()
    assert (cr.t_start <= cr.t_median).all() and (cr.t_median <= cr.t_end).all()
    assert cr.crossing.tolist() == list(range(6))


def test_ten_second_sampling_keeps_span():
    pts = _network()
    cr = find_segment_crossings(_fixes(step=10.0), pts)
    assert len(cr) == 6
    assert (cr.n_points < 40).all()  # every other point gets a fix
    assert (cr.span_m >= 1800).all()  # but the covered length is unchanged
    assert (cr.coverage > 0.9).all()


def test_off_track_fixes_are_dropped():
    pts = _network()
    clean = find_segment_crossings(_fixes(), pts)
    f = _fixes()
    f.loc[:9, "Latitude_deg"] += 0.01  # the first 10 fixes ~1 km north of the track
    cr = find_segment_crossings(f, pts)
    # the first crossing loses exactly those 10 fixes (1 s apart) and nothing else changes
    assert cr.n_fix.iloc[0] == clean.n_fix.iloc[0] - 10
    assert cr.t_start.iloc[0] == clean.t_start.iloc[0] + pd.Timedelta(seconds=10)
    pd.testing.assert_series_equal(cr.n_fix.iloc[1:], clean.n_fix.iloc[1:])
    # with the distance cut off, they would be kept (snapped to the nearest point)
    kept = find_segment_crossings(f, pts, max_point_dist=5000.0)
    assert kept.n_fix.iloc[0] == clean.n_fix.iloc[0]


def test_receptors_one_multipoint_per_crossing():
    pts = _network()
    cr = find_segment_crossings(_fixes(), pts)
    rec = build_trax_receptors(cr, pts, altitude=4.0)
    assert list(rec.columns) == RECEPTOR_COLUMNS
    assert rec.groupby("r_idx").size().eq(40).all()
    assert rec.groupby("r_idx").time.nunique().eq(1).all()
    assert rec.r_idx.nunique() == 6
    assert (rec.altitude == 4.0).all() and (rec.altitude_ref == "agl").all()
    assert rec.time.dt.tz is not None and rec.time.dt.second.eq(0).all()
    # every point of the segment is present, whatever was hit
    seg0 = rec[rec.r_idx == 0]
    assert sorted(np.round(seg0.longitude, 6)) == sorted(
        np.round(TO_LONLAT.transform(pts.geometry.x[:40], pts.geometry.y[:40])[0], 6)
    )


def test_min_span_filters_and_duplicates_collapse():
    pts = _network()
    cr = find_segment_crossings(_fixes(), pts)
    assert build_trax_receptors(cr, pts, min_span_m=5000).empty
    assert build_trax_receptors(cr, pts, min_coverage=1.01).empty
    cr2 = pd.concat(
        [cr, cr.assign(crossing=cr.crossing + 100)]
    )  # same segment+minute twice
    rec = build_trax_receptors(cr2, pts)
    assert rec.r_idx.nunique() == 6


def _junction_network():
    """One segment: a Red-only arm (0-39), the shared trunk (40-79), a Blue-only arm (80-119)."""
    pts = _network()
    pts["segment"] = 0
    pts["lines"] = ["R"] * 40 + ["RB"] * 40 + ["B"] * 40
    pts["s_B"] = np.where(
        pts.lines.str.contains("B"), pts.geometry.x - X0 - 40 * 50.0, np.nan
    )
    pts.loc[~pts.lines.str.contains("R"), "s_R"] = np.nan
    return pts


def test_release_points_follow_the_train_line():
    pts = _junction_network()
    f = _fixes(
        length=3950.0
    )  # out and back over the Red arm + trunk only (points 0-79)
    cr = find_segment_crossings(f, pts)
    assert len(cr) == 2 and (cr.lines == "R").all()
    rec = build_trax_receptors(cr, pts)
    assert (
        rec.groupby("r_idx").size().eq(80).all()
    )  # Red arm + trunk, never the Blue arm
    # a crossing that only touched the trunk releases from the trunk alone
    f2 = _fixes(length=5950.0)
    lo, hi = TO_LONLAT.transform([pts.geometry.x[40], pts.geometry.x[79]], [Y0, Y0])[0]
    f2 = f2[(f2.Longitude_deg >= lo - 1e-9) & (f2.Longitude_deg <= hi + 1e-9)]
    cr2 = find_segment_crossings(f2, pts)
    assert (cr2.lines == "RB").all()
    assert build_trax_receptors(cr2, pts).groupby("r_idx").size().eq(40).all()


def _parked(minutes=90, t0="2024-06-01 18:40:00", jitter=8.0, seed=0):
    """1-s fixes at one spot for `minutes`, with GPS jitter, then a departure."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(t0, periods=minutes * 60, freq="1s")
    x = X0 + rng.normal(0, jitter, len(t))
    y = Y0 + rng.normal(0, jitter, len(t))
    # then drive away east for 10 minutes
    t2 = t[-1] + pd.to_timedelta(np.arange(1, 601), unit="s")
    x2, y2 = X0 + 10.0 * np.arange(1, 601), np.full(600, Y0)
    lon, lat = TO_LONLAT.transform(np.r_[x, x2], np.r_[y, y2])
    return pd.DataFrame(
        {"Time_UTC": np.r_[t, t2], "Latitude_deg": lat, "Longitude_deg": lon}
    )


def test_find_dwells_picks_out_the_parked_period():
    from slv.measurements.mobile.receptors import find_dwells

    d = find_dwells(_parked(), radius=150.0, min_duration="20min")
    assert len(d) == 1
    row = d.iloc[0]
    assert row.n_minutes >= 88  # the parked minutes, not the drive-away
    assert row.spread_m < 60
    assert (
        pd.Timestamp("2024-06-01 18:40")
        <= row.t_start
        <= pd.Timestamp("2024-06-01 18:41")
    )
    assert row.t_end <= pd.Timestamp("2024-06-01 20:12")


def test_short_stop_is_not_a_dwell():
    from slv.measurements.mobile.receptors import find_dwells

    assert find_dwells(_parked(minutes=8), min_duration="20min").empty


def test_dwell_receptors_are_hourly_points_at_the_sampled_time():
    from slv.measurements.mobile.receptors import build_dwell_receptors, find_dwells

    fixes = _parked()  # 18:40 -> 20:10 UTC
    d = find_dwells(fixes, min_duration="20min")
    rec = build_dwell_receptors(fixes, d, min_minutes=30)
    assert list(rec.columns) == RECEPTOR_COLUMNS
    # 18:40-19:00 is 20 min (dropped), 19:00-20:00 full, 20:00-20:10 is 10 min (dropped)
    assert len(rec) == 1
    assert rec.r_idx.iloc[0].startswith("dwell_0_")
    t = pd.Timestamp(rec.time.iloc[0])
    assert (
        pd.Timestamp("2024-06-01 19:25", tz="UTC")
        <= t
        <= pd.Timestamp("2024-06-01 19:35", tz="UTC")
    )
    assert rec.groupby("r_idx").size().eq(1).all()  # a point receptor, not multipoint
    assert build_dwell_receptors(fixes, d, min_minutes=10).shape[0] == 3


def test_hours_window_selects_local_afternoon():
    from slv.measurements.mobile.receptors import build_dwell_receptors, find_dwells

    fixes = _parked(minutes=240, t0="2024-06-01 16:00:00")  # 09:00-13:00 MST
    d = find_dwells(fixes, min_duration="20min")
    allh = build_dwell_receptors(fixes, d, min_minutes=30)
    aft = build_dwell_receptors(fixes, d, min_minutes=30, hours=range(12, 17))
    local = pd.DatetimeIndex(allh.time) - pd.Timedelta(hours=7)
    assert set(local.hour) == {9, 10, 11, 12}
    assert set((pd.DatetimeIndex(aft.time) - pd.Timedelta(hours=7)).hour) == {12}
    assert len(aft) < len(allh)


def test_hours_window_also_applies_to_crossings():
    pts = _network()
    cr = find_segment_crossings(_fixes(t0="2024-06-01 18:00:00"), pts)  # 11:00 MST
    assert build_trax_receptors(cr, pts, hours=range(12, 17)).empty
    assert not build_trax_receptors(cr, pts, hours=[11]).empty
    assert len(build_trax_receptors(cr, pts, hours=None)) == len(
        build_trax_receptors(cr, pts)
    )


def test_dwell_r_idx_is_hourly_and_sub_hourly_freq_is_refused():
    import pytest

    from slv.measurements.mobile.receptors import build_dwell_receptors, find_dwells

    fixes = _parked()  # 18:40 -> 20:10 UTC
    d = find_dwells(fixes, min_duration="20min")
    # the r_idx format existing STILT footprints and receptor_obs are keyed on
    assert build_dwell_receptors(fixes, d).r_idx.tolist() == ["dwell_0_2024060119"]
    two = build_dwell_receptors(fixes, d, freq="2h")
    assert two.r_idx.tolist() == ["dwell_0_2024060118"]
    with pytest.raises(ValueError, match="shorter than an hour"):
        build_dwell_receptors(fixes, d, freq="30min", min_minutes=10)


def test_empty_results_pass_through_without_crashing():
    from slv.measurements.mobile.receptors import (
        build_dwell_receptors,
        find_dwells,
        label_dwell_site,
    )

    pts = _network()
    fixes = _parked(minutes=8)  # data, but too short to be a dwell
    d = find_dwells(fixes, min_duration="20min")
    assert d.empty and "longitude" in d.columns and "t_start" in d.columns
    lab = label_dwell_site(d, pts)
    assert lab.empty and list(lab.columns) == ["segment", "yard_name"]
    assert list(build_dwell_receptors(fixes, d).columns) == RECEPTOR_COLUMNS

    far = _fixes()
    far["Latitude_deg"] += 0.05  # ~5 km off the track: no fix snaps to a point
    cr = find_segment_crossings(far, pts)
    assert cr.empty
    assert pd.api.types.is_string_dtype(cr.lines)
    assert cr.n_segment_points.dtype == np.int64
    rec = build_trax_receptors(cr, pts)
    assert rec.empty and list(rec.columns) == RECEPTOR_COLUMNS


def test_dwell_yard_label_uses_the_classifier_buffer():
    from shapely.geometry import Point

    from slv.measurements.mobile.location import YARD_BUFFER
    from slv.measurements.mobile.network import load_storage_polygons
    from slv.measurements.mobile.receptors import label_dwell_site

    jrrsc = load_storage_polygons(meters=True).set_index("name").geometry["JRRSC"]
    east = max(jrrsc.exterior.coords, key=lambda c: c[0])  # nothing lies further east
    spots = [
        jrrsc.representative_point(),
        Point(east[0] + 20, east[1]),  # 20 m outside the drawn edge
        Point(east[0] + YARD_BUFFER + 20, east[1]),  # beyond the buffer
    ]
    assert [round(jrrsc.distance(p)) for p in spots] == [0, 20, 50]
    lon, lat = TO_LONLAT.transform([p.x for p in spots], [p.y for p in spots])
    dwells = pd.DataFrame({"longitude": lon, "latitude": lat})

    lab = label_dwell_site(dwells, _network())
    assert lab.yard_name.iloc[:2].tolist() == ["JRRSC", "JRRSC"]
    assert pd.isna(lab.yard_name.iloc[2])
    strict = label_dwell_site(dwells, _network(), yard_buffer=0)
    assert strict.yard_name.iloc[0] == "JRRSC" and pd.isna(strict.yard_name.iloc[1])
