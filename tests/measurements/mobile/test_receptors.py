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
    """n points east-west at y=Y0, `seg_points` per segment."""
    x = X0 + spacing * np.arange(n)
    return gpd.GeoDataFrame(
        {"point": np.arange(n), "lines": "R", "segment": np.arange(n) // seg_points},
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
    assert (cr.n_fix.between(195, 205)).all()
    assert (cr.t_start <= cr.t_median).all() and (cr.t_median <= cr.t_end).all()
    assert cr.crossing.tolist() == list(range(6))


def test_ten_second_sampling_keeps_span():
    pts = _network()
    cr = find_segment_crossings(_fixes(step=10.0), pts)
    assert len(cr) == 6
    assert (cr.n_points < 40).all()  # every other point gets a fix
    assert (cr.span_m >= 1800).all()  # but the covered length is unchanged


def test_off_track_fixes_are_dropped():
    pts = _network()
    f = _fixes()
    f.loc[:9, "Latitude_deg"] += 0.01  # ~1 km north of the track
    cr = find_segment_crossings(f, pts)
    assert cr.n_fix.iloc[0] == cr.n_fix.iloc[1] - 10 or cr.n_fix.iloc[0] < 200


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
