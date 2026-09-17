"""STILT receptors for the TRAX record: one multipoint receptor per 2-km segment crossing.

The paper-2 inversion samples TRAX CH4 at the staged 2-km network points
(:func:`~slv.measurements.mobile.network.load_trax_points`). Each staged point owns a
*segment*: the 50-m track points nearest to it. The 50-m points are the per-line track
files behind the transect matrices (``TRAX_50m_<line>_2022-01-25.csv``), merged into one
network point set so that shared track (the downtown trunk) is one set of points
(:func:`load_trax_network_points`). Every time the train crosses a segment
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
from lair.transects import merge_route_points
from pyproj import Transformer
from scipy.spatial import cKDTree

from slv.measurements.mobile.network import USER_DIR, UTM12, load_trax_points
from slv.measurements.mobile.obs import LOCATION_SETS

#: Date stamp of the per-line 50-m track files (``lat,lon`` rows, no header) under
#: ``$SLV_USER_DATA_DIR/trax/tracks/`` — Logan's Red/Green/Blue track points as re-derived
#: 2022-01-25 (copied from ``~/wkspace/mobile/trax/transect/tracks``).
TRACKS_DATE = "2022-01-25"
TRACK_LINES = ("r", "g", "b")
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


def load_trax_tracks(tracks_dir: str | Path | None = None) -> dict[str, np.ndarray]:
    """Ordered ``(lon, lat)`` arrays of the per-line 50-m track points, keyed ``r``/``g``/``b``."""
    d = USER_DIR / "trax" / "tracks" if tracks_dir is None else Path(tracks_dir)
    out = {}
    for line in TRACK_LINES:
        t = pd.read_csv(
            d / f"TRAX_50m_{line}_{TRACKS_DATE}.csv", header=None, names=["lat", "lon"]
        )
        out[line] = np.c_[t.lon.values, t.lat.values]
    return out


def load_trax_network_points(
    spacing: int = 50,
    snap_tol: float = 25.0,
    segment_spacing: int = 2000,
    meters: bool = False,
    rebuild: bool = False,
    tracks_dir: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """The merged 50-m network points with their 2-km segment membership.

    Columns: ``point`` (stable integer id), ``lines`` (letters of the lines whose track
    runs through the point, e.g. ``"RG"``), ``segment`` (index of the nearest staged
    2-km point in :func:`load_trax_points`), ``segment_dist_m``. Cached as
    ``$SLV_USER_DATA_DIR/trax/points_<spacing>m_network.geojson`` (UTM 12 N); returned in
    lon/lat unless ``meters=True``.
    """
    cache = USER_DIR / "trax" / f"points_{spacing}m_network.geojson"
    if cache.exists() and not rebuild:
        pts = gpd.read_file(cache)
    else:
        tracks = load_trax_tracks(tracks_dir)
        routes = [np.c_[_TO_UTM.transform(t[:, 0], t[:, 1])] for t in tracks.values()]
        net, index = merge_route_points(routes, snap_tol)
        member = {line: np.zeros(len(net), bool) for line in tracks}
        for line, idx in zip(tracks, index, strict=True):
            member[line][idx] = True
        lines = [
            "".join(line.upper() for line in tracks if member[line][i])
            for i in range(len(net))
        ]
        segs = load_trax_points(segment_spacing, meters=True)
        d, seg = cKDTree(np.c_[segs.geometry.x, segs.geometry.y]).query(net)
        pts = gpd.GeoDataFrame(
            {
                "point": np.arange(len(net)),
                "lines": lines,
                "segment": seg.astype(int),
                "segment_dist_m": np.round(d, 1),
            },
            geometry=gpd.points_from_xy(net[:, 0], net[:, 1]),
            crs=UTM12,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        pts.to_file(cache, index=False)
    pts = pts.to_crs(UTM12) if pts.crs != UTM12 else pts
    return pts if meters else pts.to_crs("EPSG:4326")


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
    ``crossing``, ``segment``, ``t_start``, ``t_end``, ``t_median``, ``n_fix``, ``n_points``
    (distinct 50-m points hit), ``n_segment_points`` (points the segment has), ``span_m``
    (farthest fix from the crossing's first fix — the length of track covered, robust to
    the 10-s sampling era where only every other 50-m point gets a fix).
    """
    if points is None:
        points = load_trax_network_points(meters=True)
    if not fixes.Time_UTC.is_monotonic_increasing:
        fixes = fixes.sort_values("Time_UTC")
    x, y = _TO_UTM.transform(fixes.Longitude_deg.values, fixes.Latitude_deg.values)
    d, ip = cKDTree(np.c_[points.geometry.x, points.geometry.y]).query(np.c_[x, y])
    ok = d <= max_point_dist
    ts = _posix_seconds(fixes.Time_UTC.values[ok])
    seg = points["segment"].values[ip[ok]]
    pt = points["point"].values[ip[ok]]
    x, y = x[ok], y[ok]
    gap_s = pd.Timedelta(max_gap).total_seconds()
    new = np.ones(len(ts), bool)
    new[1:] = (np.diff(ts) > gap_s) | (np.diff(seg) != 0)
    df = pd.DataFrame(
        {
            "crossing": np.cumsum(new) - 1,
            "segment": seg,
            "t": ts,
            "point": pt,
            "x": x,
            "y": y,
        }
    )
    g = df.groupby("crossing", sort=True)
    df["r"] = np.hypot(df.x - g.x.transform("first"), df.y - g.y.transform("first"))
    g = df.groupby("crossing", sort=True)
    out = pd.DataFrame(
        {
            "segment": g.segment.first().astype(int),
            "t_start": pd.to_datetime(g.t.min(), unit="s"),
            "t_end": pd.to_datetime(g.t.max(), unit="s"),
            "t_median": pd.to_datetime(g.t.median().round().astype("int64"), unit="s"),
            "n_fix": g.size(),
            "n_points": g.point.nunique(),
            "span_m": g.r.max().round(1),
        }
    )
    seg_size = points.groupby("segment").size()
    out["n_segment_points"] = out.segment.map(seg_size).astype(int)
    return out.reset_index()


def build_trax_receptors(
    crossings: pd.DataFrame,
    points: gpd.GeoDataFrame | None = None,
    min_span_m: float = 1000.0,
    altitude: float = 4.0,
    altitude_ref: str = "agl",
    time_round: str = "1min",
) -> pd.DataFrame:
    """PYSTILT receptor table (:data:`RECEPTOR_COLUMNS`) from a crossings table.

    One ``r_idx`` (= ``crossing``) per crossing whose ``span_m`` is at least ``min_span_m``,
    with one row per 50-m point of the segment (every point, not only those with a fix, so
    a segment's receptor geometry is always the same and its PYSTILT location id stable).
    ``time`` is the crossing's median fix time rounded to ``time_round`` (UTC); crossings
    of one segment that round to the same minute are kept once. ``altitude`` is the roof
    inlet height in m AGL.
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
        }
    )
    c = crossings.loc[
        crossings.span_m >= min_span_m, ["crossing", "segment", "t_median"]
    ].copy()
    t = pd.DatetimeIndex(c.t_median)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    c["time"] = t.round(time_round)
    c = c.drop_duplicates(["segment", "time"])
    rec = (
        c[["crossing", "segment", "time"]]
        .merge(pts, on="segment", how="inner")
        .sort_values(["crossing", "point"])
        .rename(columns={"crossing": "r_idx"})
    )
    rec["altitude"] = float(altitude)
    rec["altitude_ref"] = altitude_ref
    return rec[RECEPTOR_COLUMNS].reset_index(drop=True)


__all__ = [
    "RECEPTOR_COLUMNS",
    "TRACKS_DATE",
    "TRACK_LINES",
    "build_trax_receptors",
    "find_segment_crossings",
    "load_trax_fixes",
    "load_trax_network_points",
    "load_trax_tracks",
]
