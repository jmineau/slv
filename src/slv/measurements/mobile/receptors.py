"""STILT receptors for the TRAX record: one multipoint receptor per 2-km segment crossing.

The paper-2 inversion samples TRAX CH4 at the staged 2-km network points
(:func:`~slv.measurements.mobile.network.load_trax_points`). Each staged point owns a
*segment*: the 50-m network points nearest to it. Both point sets are generated the same
way, ``lair.geo.points_along_line`` on the UTA line network (:func:`load_trax_points`), so
shared track is one row of points (:func:`load_trax_network_points`); a crossing on one
line releases only from that line's points of the segment (:func:`release_points`).
Every time the train crosses a segment
(:func:`find_segment_crossings`) it gets one PYSTILT ``MultiPointReceptor`` that releases
particles from all of the segment's 50-m points at the median crossing time
(:func:`build_trax_receptors`). PYSTILT spreads ``numpar`` evenly over the release points
and writes their mean footprint, so the 50-m footprints come out already aggregated to
the 2-km point. The CH4 value that pairs with a receptor is the mean over the same 50-m
points during the crossing (i.e. the transect-matrix row averaged over the segment).

Segment membership is fixed by geometry alone: it does not depend on the inlet lag or on
which periods pass calibration QC, so receptors can be built (and run) before those
decisions are final; QC later selects which crossings enter the inversion. Crossings are
found from the GPS positions of the on-track rows of ``obs.parquet``, so periods with no
LGR data get no receptors.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree

from slv.measurements.mobile.network import (
    USER_DIR,
    UTM12,
    load_trax_lines,
    load_trax_points,
)
from slv.measurements.mobile.obs import LOCATION_SETS

#: Letters of the TRAX lines as :func:`load_trax_points` tags them (bit i of a line mask).
LINE_LETTERS = "RGBS"
#: Column order of a PYSTILT receptor CSV (``r_idx`` groups rows into one receptor).
RECEPTOR_COLUMNS = [
    "r_idx",
    "time",
    "longitude",
    "latitude",
    "altitude",
    "altitude_ref",
]

_TO_UTM = Transformer.from_crs("EPSG:4326", UTM12, always_xy=True)
_TO_LONLAT = Transformer.from_crs(UTM12, "EPSG:4326", always_xy=True)


def load_trax_network_points(
    spacing: int = 50,
    segment_spacing: int = 2000,
    meters: bool = False,
    rebuild: bool = False,
) -> gpd.GeoDataFrame:
    """The 50-m network points with their 2-km segment membership.

    Both point sets come from :func:`load_trax_points` (``lair.geo.points_along_line`` on
    the UTA line network, so shared track is one row of points and every pair is at least
    ``spacing`` apart; each point is tagged with the letters of the lines within
    ``spacing/2`` of it). Columns: ``point`` (stable integer id), ``lines`` (e.g. ``"BGR"``
    on the downtown trunk), ``segment`` (index of the nearest staged 2-km point),
    ``segment_dist_m``, and ``s_R``/``s_G``/``s_B``/``s_S`` — distance along each line the
    point sits on (metres from that line's start, NaN off it), which is what makes
    "how much of the segment did this crossing cover" measurable along the track instead
    of as a straight line (:func:`find_segment_crossings`). Cached as ``$SLV_USER_DATA_DIR/trax/points_<spacing>m_network.geojson``
    (UTM 12 N); returned in lon/lat unless ``meters=True``. Generating the 50-m points the
    first time takes ~1 h (pure-Python graph walk over a 5-m graph of the network; 1,397
    points, 2026-09-17) — do it on a compute node; afterwards ``load_trax_points`` reads its
    own cache in seconds.
    """
    cache = USER_DIR / "trax" / f"points_{spacing}m_network.geojson"
    if cache.exists() and not rebuild:
        pts = gpd.read_file(cache)
    else:
        fine = load_trax_points(spacing, meters=True)
        segs = load_trax_points(segment_spacing, meters=True)
        xy = np.c_[fine.geometry.x, fine.geometry.y]
        d, seg = cKDTree(np.c_[segs.geometry.x, segs.geometry.y]).query(xy)
        pts = gpd.GeoDataFrame(
            {
                "point": np.arange(len(fine)),
                "lines": fine["lines"].fillna("").astype(str).values,
                "segment": seg.astype(int),
                "segment_dist_m": np.round(d, 1),
            },
            geometry=fine.geometry.values,
            crs=fine.crs,
        )
        for letter, geom in _line_geometries().items():
            on = pts["lines"].str.contains(letter).to_numpy()
            along = np.full(len(pts), np.nan)
            if on.any():
                along[on] = shapely.line_locate_point(geom, pts.geometry.values[on])
            pts[f"s_{letter}"] = np.round(along, 1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        pts.to_file(cache, index=False)
    pts = pts.to_crs(UTM12) if pts.crs != UTM12 else pts
    return pts if meters else pts.to_crs("EPSG:4326")


def _line_geometries() -> dict[str, shapely.LineString]:
    """UTA route geometry per line letter, in :data:`UTM12` (metres)."""
    lines = load_trax_lines().to_crs(UTM12)
    return {str(r.line).upper(): r.geometry for r in lines.itertuples()}


def load_trax_fixes(
    cache: str | Path | None = None, location: str | tuple[str, ...] | None = "on_track"
) -> pd.DataFrame:
    """``Time_UTC``, ``Latitude_deg``, ``Longitude_deg`` of the ``location`` rows of the cached
    TRAX record (``obs.parquet``), one row per distinct second, time-sorted.

    Reads only those columns (pyarrow, filtered on ``state``) — the full parquet with CH4 is
    ~6 GB in memory, the fixes alone ~1 GB.
    """
    import pyarrow.parquet as pq

    cache = USER_DIR / "trax" / "obs.parquet" if cache is None else Path(cache)
    states = LOCATION_SETS[location] if isinstance(location, str) else location
    filters = None if states is None else [("state", "in", list(states))]
    t = pq.read_table(
        cache,
        columns=["Time_UTC", "Latitude_deg", "Longitude_deg", "state"],
        filters=filters,
    )
    df = t.to_pandas().drop(columns="state")
    return df.drop_duplicates("Time_UTC").sort_values("Time_UTC").reset_index(drop=True)


def _posix_seconds(times) -> np.ndarray:
    t = pd.DatetimeIndex(times)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.values.astype("datetime64[s]").astype("int64")


def find_segment_crossings(
    fixes: pd.DataFrame,
    points: gpd.GeoDataFrame | None = None,
    max_gap: str | pd.Timedelta = "10min",
    max_point_dist: float = 60.0,
) -> pd.DataFrame:
    """Split the GPS record into crossings of the 2-km segments.

    Each fix is snapped to the nearest 50-m network point (dropped if farther than
    ``max_point_dist`` m) and inherits its segment; a crossing is a run of consecutive
    fixes on one segment with no gap longer than ``max_gap``. One row per crossing:

    ``crossing``, ``segment``, ``lines`` (letters common to every point hit — the line the
    train was on; ``"BGR"`` when only shared trunk points were hit), ``t_start``, ``t_end``,
    ``t_median``, ``n_fix``, ``n_points`` (distinct 50-m points hit), ``n_segment_points``
    (points the segment has on that line), ``span_m``, ``segment_extent_m`` and
    ``coverage``.

    ``span_m`` is measured **along the route** (the ``s_<line>`` coordinate of the points
    hit, from :func:`load_trax_network_points`), not as a straight line from the first fix:
    on a bent segment a full traverse can be 2.6 km of rail but only 1.8 km of chord, so a
    chord would make the same coverage look different on straight and bent segments.
    ``segment_extent_m`` is the along-route length of the release geometry
    (:func:`release_points`) and ``coverage = span_m / segment_extent_m`` is the fraction of
    it the train actually drove.
    """
    if points is None:
        points = load_trax_network_points(meters=True)
    letters = [c for c in LINE_LETTERS if f"s_{c}" in points.columns]
    if not letters:
        raise ValueError(
            "points must carry along-route coordinates (s_R/s_G/...); rebuild them with "
            "load_trax_network_points(rebuild=True)."
        )
    if not fixes.Time_UTC.is_monotonic_increasing:
        fixes = fixes.sort_values("Time_UTC")
    x, y = _TO_UTM.transform(fixes.Longitude_deg.values, fixes.Latitude_deg.values)
    d, ip = cKDTree(np.c_[points.geometry.x, points.geometry.y]).query(np.c_[x, y])
    ok = d <= max_point_dist
    ip = ip[ok]
    ts = _posix_seconds(fixes.Time_UTC.values[ok])
    seg = points["segment"].values[ip]
    pt = points["point"].values[ip]
    point_masks = _line_masks(points["lines"].values)
    mask = point_masks[ip]
    gap_s = pd.Timedelta(max_gap).total_seconds()
    new = np.ones(len(ts), bool)
    new[1:] = (np.diff(ts) > gap_s) | (np.diff(seg) != 0)
    starts = np.flatnonzero(new)
    # lines common to every point hit in a crossing: bitwise AND over each run of fixes
    common = np.bitwise_and.reduceat(mask, starts) if len(ts) else np.zeros(0, np.int64)

    # along-route coordinate of every fix, on the line its crossing was on
    S = points[[f"s_{c}" for c in letters]].to_numpy(float)
    bits = np.array([1 << LINE_LETTERS.index(c) for c in letters])

    def _col(m: int) -> int:
        """Index into ``letters`` of the first line in bit mask ``m`` (-1 if none)."""
        hit = np.flatnonzero(np.bitwise_and(int(m), bits))
        return int(hit[0]) if len(hit) else -1

    first_mask = mask[starts]
    # a crossing whose points share no line (disjoint arms) falls back to its first point
    cross_col = np.array(
        [
            c if (c := _col(m)) >= 0 else _col(f)
            for m, f in zip(common, first_mask, strict=True)
        ],
        dtype=int,
    )
    sizes = np.diff(np.r_[starts, len(ts)])
    s_fix = S[ip, np.repeat(cross_col, sizes)] if len(ts) else np.zeros(0)

    df = pd.DataFrame(
        {
            "crossing": np.cumsum(new) - 1,
            "segment": seg,
            "t": ts,
            "point": pt,
            "s": s_fix,
        }
    )
    g = df.groupby("crossing", sort=True)
    out = pd.DataFrame(
        {
            "segment": g.segment.first().astype(int),
            "lines": [_mask_to_lines(int(m)) for m in common],
            "t_start": pd.to_datetime(g.t.min(), unit="s"),
            "t_end": pd.to_datetime(g.t.max(), unit="s"),
            "t_median": pd.to_datetime(g.t.median().round().astype("int64"), unit="s"),
            "n_fix": g.size(),
            "n_points": g.point.nunique(),
            "span_m": (g.s.max() - g.s.min()).round(1),
        }
    )
    # release-geometry extent, once per (segment, lines) pair
    extent, n_rel = {}, {}
    for segment, lines in (
        out[["segment", "lines"]].drop_duplicates().itertuples(index=False)
    ):
        rel = points.loc[release_points(points, int(segment), str(lines))]
        col = _col(int(_line_masks([lines])[0]))
        vals = (
            rel[f"s_{letters[col]}"].to_numpy(float) if col >= 0 else np.array([np.nan])
        )
        vals = vals[~np.isnan(vals)]
        extent[(segment, lines)] = (
            round(float(vals.max() - vals.min()), 1) if len(vals) else np.nan
        )
        n_rel[(segment, lines)] = len(rel)
    key = list(zip(out.segment, out.lines, strict=True))
    out["segment_extent_m"] = [extent[k] for k in key]
    out["n_segment_points"] = [n_rel[k] for k in key]
    out["coverage"] = (out.span_m / out.segment_extent_m).round(3)
    return out.reset_index()


def _line_masks(lines) -> np.ndarray:
    """Bit mask per point from its ``lines`` letters (R=1, G=2, B=4, S=8)."""
    bits = {c: 1 << i for i, c in enumerate(LINE_LETTERS)}
    return np.array(
        [sum(bits[c] for c in str(s) if c in bits) for s in lines], dtype=np.int64
    )


def _mask_to_lines(mask: int) -> str:
    return "".join(c for i, c in enumerate(LINE_LETTERS) if mask & (1 << i))


def release_points(points: gpd.GeoDataFrame, segment: int, lines: str) -> pd.Index:
    """Index of the segment's points on every line in ``lines`` (all of them if ``lines`` is empty).

    At a junction a staged 2-km point collects the arms of several lines (downtown,
    segment 21 has 92 points on three arms); a train on one line only samples that line's
    arms, so a receptor releases from the segment points whose ``lines`` contain every
    letter the crossing's fixes had in common — the shared trunk plus that line's arm.
    """
    sel = points["segment"].values == segment
    for c in lines:
        sel &= points["lines"].str.contains(c).values
    return points.index[sel]


def build_trax_receptors(
    crossings: pd.DataFrame,
    points: gpd.GeoDataFrame | None = None,
    min_coverage: float = 0.5,
    min_span_m: float = 0.0,
    max_duration: str | pd.Timedelta | None = "15min",
    altitude: float = 4.0,
    altitude_ref: str = "agl",
    time_round: str = "1min",
) -> pd.DataFrame:
    """PYSTILT receptor table (:data:`RECEPTOR_COLUMNS`) from a crossings table.

    One ``r_idx`` (= ``crossing``) per crossing that drove at least ``min_coverage`` of its
    release geometry along the route (and at least ``min_span_m`` of it in metres, if set),
    with one row per 50-m point of the segment on the crossing's line
    (:func:`release_points`; every such point, not only those with a fix, so a
    segment × line receptor geometry is always the same and its PYSTILT location id
    stable). ``time`` is the crossing's median fix time rounded to ``time_round`` (UTC);
    crossings of one segment and line that round to the same minute are kept once.
    ``altitude`` is the roof inlet height in m AGL.

    ``max_duration`` drops passes that are not traverses: at the three line termini the
    train dwells inside one segment between runs, so the pass can read 20 min or more and
    its median fix time is not the time it drove the track. ``None`` keeps them.
    """
    if points is None:
        points = load_trax_network_points(meters=True)
    lon, lat = _TO_LONLAT.transform(points.geometry.x.values, points.geometry.y.values)
    pts = pd.DataFrame(
        {
            "segment": points["segment"].values,
            "point": points["point"].values,
            "longitude": np.round(lon, 6),
            "latitude": np.round(lat, 6),
        },
        index=points.index,
    )
    keep = (crossings.coverage >= min_coverage) & (crossings.span_m >= min_span_m)
    if max_duration is not None:
        dur = (crossings.t_end - crossings.t_start).dt.total_seconds()
        keep &= dur <= pd.Timedelta(max_duration).total_seconds()
    c = crossings.loc[keep, ["crossing", "segment", "lines", "t_median"]].copy()
    t = pd.DatetimeIndex(c.t_median)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    c["time"] = t.round(time_round)
    c = c.drop_duplicates(["segment", "lines", "time"])
    sets = [
        pts.loc[release_points(points, int(segment), str(lines))].assign(lines=lines)
        for segment, lines in c[["segment", "lines"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    rel = pd.concat(sets, ignore_index=True) if sets else pts.iloc[:0].assign(lines="")
    rec = (
        c[["crossing", "segment", "lines", "time"]]
        .merge(rel, on=["segment", "lines"], how="inner")
        .sort_values(["crossing", "point"])
        .rename(columns={"crossing": "r_idx"})
    )
    rec["altitude"] = float(altitude)
    rec["altitude_ref"] = altitude_ref
    return rec[RECEPTOR_COLUMNS].reset_index(drop=True)


__all__ = [
    "LINE_LETTERS",
    "RECEPTOR_COLUMNS",
    "build_trax_receptors",
    "find_segment_crossings",
    "load_trax_fixes",
    "load_trax_network_points",
    "release_points",
]
