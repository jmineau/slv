"""
TRAX transect matrices: ``obs[transect, point]`` over the 50-m network points.

Two generations of matrices:

* **Network matrices** (this builder, 2026-09-24) — one matrix over the 50-m *network*
  points of :func:`~slv.measurements.mobile.receptors.load_trax_network_points` (1,397
  points; shared track is one row of points), built from ``obs.parquet`` by
  :func:`build_network_transects`. Every transit of every line is a row, tagged with its
  line, direction and time, so pooling the lines is a concatenation and a source on the
  downtown trunk is scored once from all passes. Written per year by
  :func:`build_trax_transects` to ``$SLV_USER_DATA_DIR/trax/transects/network/`` and read
  back by :func:`load_network_transects`.
* **Archived per-line matrices** (Logan Mitchell's algorithm as rewritten in 2022; Dec 2014 –
  Apr 2023, single 9-s inlet lag, built from the duplicated-row parquet) — read by
  :func:`load_transects`. Provenance and cross-checks only.

The builder is Mitchell et al. (2018)'s algorithm on the generic pieces in
:mod:`lair.transects`, with the TRAX specifics here:

1. the on-track rows of ``obs.parquet`` (:func:`~slv.measurements.mobile.obs.load_trax_obs`,
   so calibration provenance, the epoch offsets and the location classifier all apply);
2. each sample is moved to where its air was taken in, ``lag_s`` earlier along the track
   (:func:`lair.transects.lag_positions`), with the lag of its instrument epoch
   (:func:`~slv.measurements.mobile.receptor_obs.inlet_lag_seconds`);
3. samples snap to the nearest network point (60 m, like the STILT crossings);
4. the line the train is on is resolved per sample by a vote among the letters of its own
   point: the line with the most single-line points hit within ±30 min (:func:`assign_lines`);
5. that line's along-route coordinate is cut into one-way transits at reversals of travel
   and at gaps (:func:`lair.transects.split_transits`), and the samples are averaged onto
   ``[transit, point]`` with terminus dwells trimmed (:func:`lair.transects.transect_matrix`).

**Coverage and short transits.** Unlike the archived per-line matrices -- built by chasing
terminus polygons, so every row was (close to) a full end-to-end run -- this builder keeps
any one-way run of at least ``min_span_m``, including short turns that never reach the
line's ends (a car shuttling Salt Lake Central <-> a few km out, real UTA service, ~15% of
Red and ~29% of Green transits by span in the 2014-2026 record) and partial runs cut short
by a mid-route reversal. Each transit's ``coverage`` coordinate is its span over that line's
full known extent; a persistence metric that baselines a transit against its own low
percentile (:func:`lair.transects.enhancement`) should filter on ``coverage`` first; a short
loop that sits in a plume the whole way reads as background against itself. See
``sources/persistence/`` for the filter in use.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from lair import transects as tr
from pyproj import Transformer

from slv.measurements.mobile.network import user_dir

TRANSECT_LINES = {"r": "Red", "g": "Green", "b": "Blue"}

#: Sub-directory of ``$SLV_USER_DATA_DIR/trax/transects/`` holding the network matrices.
NETWORK_DIR = "network"


def load_transects(line: str, months=None, transects_dir: str | Path | None = None):
    """
    Concatenate the archived transect matrices for one TRAX line into an xarray Dataset.

    Files: ``$SLV_USER_DATA_DIR/trax/transects/trx01_CH4_<line>_YYYY-MM.nc`` (dims
    ``transect`` x ``point``; variables ``obs`` [ppm], ``time`` [POSIX s], ``n``; point
    coords ``lat``/``lon``). ``months`` is an optional iterable of ``"YYYY-MM"`` strings.
    Returns the Dataset with a ``month`` coordinate on the transect dimension.
    """
    import xarray as xr

    d = (
        user_dir() / "trax" / "transects"
        if transects_dir is None
        else Path(transects_dir)
    )
    files = sorted(d.glob(f"trx01_CH4_{line}_*.nc"))
    if months is not None:
        want = set(months)
        files = [f for f in files if f.stem.split("_")[-1] in want]
    if not files:
        raise FileNotFoundError(f"no transect files for line {line!r} in {d}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.assign_coords(
            month=("transect", [f.stem.split("_")[-1]] * ds.sizes["transect"])
        )
        parts.append(ds)
    out = xr.concat(parts, dim="transect", combine_attrs="drop_conflicts")
    return out.assign_coords(transect=np.arange(out.sizes["transect"]))


# ---------------------------------------------------------------------------
# Network matrices
# ---------------------------------------------------------------------------


def assign_lines(
    point_lines: np.ndarray,
    time_s: np.ndarray,
    hit: np.ndarray,
    window_s: float = 1800.0,
    letters: str = "RGBS",
) -> np.ndarray:
    """
    The line letter each sample is on, resolved from the points around it in time.

    ``point_lines`` is the ``lines`` string of each sample's network point (``"R"``,
    ``"BR"``, ``"BGR"`` on the trunk, ...), ``time_s`` the sorted sample times (POSIX s) and
    ``hit`` whether the sample snapped to a point at all. Each sample is given the line,
    **among the letters of its own point**, with the most *exclusive* hits (samples on
    single-line points) within ``window_s`` before and after it. A sample on Red-only track
    is Red; a sample on the Red/Blue shared stretch is Red when the train was recently on
    Red-only track and Blue when it was recently on Blue-only track (e.g. a car in Blue
    service that just left the Blue-only stub at Salt Lake Central). The candidate
    restriction matters: a train that arrives from Green-only track and turns onto a
    Red/Blue-shared stretch must not carry ``G`` there, which a plain forward-fill did.
    Samples with no eligible vote (a point with no line tag, or half an hour on shared
    track with no exclusive point either side) come back ``""`` and are left out of every
    line's transits.
    """
    n = len(point_lines)
    tags = np.asarray(point_lines, dtype=str)
    t = np.asarray(time_s, dtype=float)
    lo = np.searchsorted(t, t - window_s, side="left")
    hi = np.searchsorted(t, t + window_s, side="right")
    best = np.full(n, "", dtype=object)
    best_votes = np.zeros(n)
    for c in letters:
        excl = hit & (tags == c)
        cum = np.r_[0, np.cumsum(excl)]
        votes = (cum[hi] - cum[lo]).astype(float)
        cand = hit & (np.char.find(tags, c) >= 0)
        votes = np.where(cand, votes, 0.0)
        better = votes > best_votes
        best[better] = c
        best_votes[better] = votes[better]
    return best


def _headings(points: gpd.GeoDataFrame, letters: list[str]) -> dict[str, str]:
    """Compass heading of travel with increasing ``s_<line>`` for each line letter."""
    lonlat = points.to_crs("EPSG:4326") if points.crs != "EPSG:4326" else points
    out = {}
    for c in letters:
        s = lonlat[f"s_{c}"].to_numpy(float)
        on = np.isfinite(s)
        i0, i1 = np.flatnonzero(on)[s[on].argmin()], np.flatnonzero(on)[s[on].argmax()]
        dlat = lonlat.geometry.y.iloc[i1] - lonlat.geometry.y.iloc[i0]
        dlon = lonlat.geometry.x.iloc[i1] - lonlat.geometry.x.iloc[i0]
        if abs(dlat) >= abs(dlon):
            out[c] = "N" if dlat > 0 else "S"
        else:
            out[c] = "E" if dlon > 0 else "W"
    return out


_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def build_network_transects(
    obs: pd.DataFrame,
    points: gpd.GeoDataFrame,
    lag: float | pd.DataFrame | None = None,
    max_point_dist: float = 60.0,
    max_gap: str | pd.Timedelta = "10min",
    reversal_m: float = 500.0,
    min_span_m: float = 1000.0,
    max_dwell_s: float | None = 180.0,
    species: str = "CH4_ppm",
    return_samples: bool = False,
):
    """
    Build the ``[transect, point]`` matrix of one obs frame on the network points.

    Parameters
    ----------
    obs
        Georeferenced samples with ``Time_UTC``, ``Longitude_deg``, ``Latitude_deg`` and
        the ``species`` column (``obs.parquet`` rows; ``cal_source`` and ``low_pressure``
        are summarised per transit when present). Any order; sorted here.
    points
        :func:`~slv.measurements.mobile.receptors.load_trax_network_points` (any CRS;
        needs ``point``, ``lines``, ``segment`` and the ``s_<line>`` columns).
    lag
        Inlet lag: ``None``, seconds, or the per-epoch table with ``start``/``end``/``lag_s``
        (:func:`~slv.measurements.mobile.receptor_obs.inlet_lag_seconds`).
    max_point_dist
        Snap radius to a network point [m]; farther samples are off-network.
    max_gap
        A transit (and a run for the line assignment and lag) ends at a longer gap.
    reversal_m, min_span_m
        :func:`lair.transects.split_transits`: turning-point prominence that ends a transit,
        and the shortest along-route span kept.
    max_dwell_s
        :func:`lair.transects.transect_matrix`: samples at a point later than this after
        the transit first reached it are dropped (terminus dwells).
    return_samples
        Also return the per-sample assignment (``Time_UTC``, ``x``, ``y`` lagged UTM
        position, ``point`` (-1 off-network), ``line`` (``""`` unresolved), ``transit``
        (-1 not in a kept transit)), for diagnostics.

    Returns
    -------
    xarray.Dataset
        dims ``transect`` x ``point``; variables ``obs`` (ppm), ``time`` (mean POSIX s of
        the samples in the cell) and ``n`` (samples); point coords ``point`` (network id),
        ``lon``, ``lat``, ``lines``, ``segment``, ``s_<line>``; transect coords ``line``,
        ``direction`` (+1 with increasing ``s``), ``heading`` (compass letter of travel),
        ``t_start``, ``t_end``, ``s_min``, ``s_max``, ``n_fix``, ``n_points`` (points with
        data), ``coverage`` (``(s_max - s_min)`` over that line's full known extent -- a
        short turn or a partial run reads well below 1; see the module docstring),
        ``lag_s``, ``lag_source``, and ``frac_uncalibrated`` / ``frac_low_pressure``
        when the tags are present. Attributes record every parameter.
    """
    import xarray as xr

    from slv.measurements.mobile.network import UTM12
    from slv.measurements.mobile.receptor_obs import inlet_lag_seconds

    pts = points.to_crs(UTM12) if points.crs != UTM12 else points
    pts = pts.reset_index(drop=True)
    letters = [c[2:] for c in pts.columns if c.startswith("s_")]
    n_points = len(pts)

    obs = obs.sort_values("Time_UTC").reset_index(drop=True)
    t = pd.DatetimeIndex(obs["Time_UTC"])
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    t_s = t.values.astype("datetime64[ns]").astype("int64") / 1e9
    gap_s = pd.Timedelta(max_gap).total_seconds()

    # 1. where the air was taken in
    lag_s, lag_src = inlet_lag_seconds(t, lag)
    x, y = Transformer.from_crs("EPSG:4326", UTM12, always_xy=True).transform(
        obs["Longitude_deg"].to_numpy(float), obs["Latitude_deg"].to_numpy(float)
    )
    xy = tr.lag_positions(t_s, np.c_[x, y], lag_s, max_gap_s=gap_s)

    # 2. snap to the network and name the line
    pidx, _ = tr.snap_to_route(
        xy, np.c_[pts.geometry.x, pts.geometry.y], max_point_dist
    )
    hit = pidx >= 0
    line = assign_lines(pts["lines"].to_numpy(dtype=str)[pidx], t_s, hit)

    # 3. transits per line, on that line's along-route coordinate
    S = {c: pts[f"s_{c}"].to_numpy(float) for c in letters}
    # full known extent of each line, from every network point on it (not just those hit) --
    # what a transit's own span is measured against to catch short turns and partial runs
    line_extent_m = {
        c: float(np.nanmax(S[c]) - np.nanmin(S[c]))
        if np.isfinite(S[c]).any()
        else np.nan
        for c in letters
    }
    transit = np.full(len(obs), -1, dtype=int)
    tables = []
    for c in letters:
        on = line == c
        if not on.any():
            continue
        s = np.full(len(obs), np.nan)
        s[on] = S[c][pidx[on]]
        tr_c, tab = tr.split_transits(
            t_s, s, max_gap_s=gap_s, reversal_m=reversal_m, min_span_m=min_span_m
        )
        if tab.empty:
            continue
        offset = sum(len(x) for x in tables)
        transit[tr_c >= 0] = tr_c[tr_c >= 0] + offset
        tab = tab.reset_index(drop=True)
        tab["line"] = c
        tables.append(tab)
    if not tables:
        table = pd.DataFrame(columns=[*tr.TRANSIT_COLUMNS, "line"])
    else:
        table = pd.concat(tables, ignore_index=True)
    n_transits = len(table)

    # 4. the matrix
    values = pd.to_numeric(obs[species], errors="coerce").to_numpy(float)
    m_obs, m_t, m_n = tr.transect_matrix(
        transit, pidx, values, t_s, n_transits, n_points, max_dwell_s=max_dwell_s
    )

    # per-transit tags
    headings = _headings(pts, letters)
    heading = [
        headings[ln] if d > 0 else _OPPOSITE[headings[ln]]
        for ln, d in zip(table["line"], table["direction"], strict=True)
    ]
    in_transit = transit >= 0
    by = pd.Series(transit[in_transit])
    per = pd.DataFrame(index=np.arange(n_transits))
    per["lag_s"] = pd.Series(lag_s[in_transit]).groupby(by.values).median()
    per["lag_source"] = (
        pd.Series(lag_src[in_transit])
        .groupby(by.values)
        .agg(lambda s: s.mode().iloc[0])
    )
    for col, name in (
        ("cal_source", "frac_uncalibrated"),
        ("low_pressure", "frac_low_pressure"),
    ):
        if col in obs.columns:
            flag = (
                (obs[col].to_numpy() == "uncalibrated")
                if col == "cal_source"
                else obs[col].to_numpy(dtype=bool)
            )
            per[name] = (
                pd.Series(flag[in_transit].astype(float)).groupby(by.values).mean()
            )

    lonlat = pts.to_crs("EPSG:4326")
    coords = {
        "point": pts["point"].to_numpy(),
        "lon": ("point", lonlat.geometry.x.round(5).to_numpy()),
        "lat": ("point", lonlat.geometry.y.round(5).to_numpy()),
        "lines": ("point", pts["lines"].astype(str).to_numpy()),
        "segment": ("point", pts["segment"].to_numpy()),
        **{f"s_{c}": ("point", S[c]) for c in letters},
        "transect": np.arange(n_transits),
        "line": ("transect", table["line"].astype(str).to_numpy()),
        "direction": ("transect", table["direction"].to_numpy(dtype=np.int8)),
        "heading": ("transect", np.array(heading, dtype=str)),
        "t_start": ("transect", pd.to_datetime(table["t_start"], unit="s").to_numpy()),
        "t_end": ("transect", pd.to_datetime(table["t_end"], unit="s").to_numpy()),
        "s_min": ("transect", table["s_min"].to_numpy(float)),
        "s_max": ("transect", table["s_max"].to_numpy(float)),
        "coverage": (
            "transect",
            np.array(
                [
                    (smax - smin) / line_extent_m[ln]
                    for smin, smax, ln in zip(
                        table["s_min"], table["s_max"], table["line"], strict=True
                    )
                ],
                dtype=float,
            ),
        ),
        "n_fix": ("transect", table["n"].to_numpy(dtype=np.int64)),
        "n_points": ("transect", np.isfinite(m_obs).sum(axis=1)),
        "lag_s": ("transect", per["lag_s"].to_numpy(float)),
        "lag_source": ("transect", per["lag_source"].astype(str).to_numpy()),
    }
    for name in ("frac_uncalibrated", "frac_low_pressure"):
        if name in per.columns:
            coords[name] = ("transect", per[name].to_numpy(float))
    ds = xr.Dataset(
        {
            "obs": (("transect", "point"), m_obs),
            "time": (("transect", "point"), m_t),
            "n": (("transect", "point"), m_n),
        },
        coords=coords,
        attrs={
            "description": "TRAX transect matrix on the 50-m network points",
            "species": species,
            "obs_units": "ppm",
            "time_units": "POSIX seconds (mean of the samples in the cell)",
            "max_point_dist_m": max_point_dist,
            "max_gap": str(pd.Timedelta(max_gap)),
            "reversal_m": reversal_m,
            "min_span_m": min_span_m,
            "max_dwell_s": -1 if max_dwell_s is None else max_dwell_s,
            "lag": (
                "none"
                if lag is None
                else f"{lag} s"
                if isinstance(lag, (int, float))
                else "per-epoch table"
            ),
            "n_samples": int(len(obs)),
            "n_off_network": int((~hit).sum()),
            "n_unresolved_line": int((hit & (line == "")).sum()),
            "n_in_transits": int(in_transit.sum()),
        },
    )
    if return_samples:
        samples = pd.DataFrame(
            {
                "Time_UTC": t,
                "x": xy[:, 0],
                "y": xy[:, 1],
                "point": pidx,
                "line": line,
                "transit": transit,
            }
        )
        return ds, samples
    return ds


def network_transects_dir(transects_dir: str | Path | None = None) -> Path:
    """
    The network transects folder: ``NETWORK_DIR`` under *transects_dir*, by default
    ``trax/transects`` in the user data directory.
    """
    base = (
        user_dir() / "trax" / "transects"
        if transects_dir is None
        else Path(transects_dir)
    )
    return base / NETWORK_DIR


def build_trax_transects(
    years,
    lag: float | pd.DataFrame | None = None,
    transects_dir: str | Path | None = None,
    points: gpd.GeoDataFrame | None = None,
    site: str = "trx01",
    load_kwargs: dict | None = None,
    **build_kwargs,
) -> list[Path]:
    """
    Build and write the network matrices, one NetCDF per calendar year.

    Each year is read from ``obs.parquet`` on its own (``load_trax_obs(time_range=...)``,
    ``load_kwargs`` forwarded, default location ``on_track``), built with
    :func:`build_network_transects` (``lag`` and ``build_kwargs`` forwarded) and written to
    ``<transects_dir>/network/<site>_CH4_<year>.nc``. Returns the files written; a year with
    no transits is skipped.
    """
    from slv.measurements.mobile.obs import load_trax_obs
    from slv.measurements.mobile.receptors import load_trax_network_points

    pts = load_trax_network_points(meters=True) if points is None else points
    out_dir = network_transects_dir(transects_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for year in years:
        year = int(year)
        obs = load_trax_obs(
            time_range=(f"{year}-01-01", f"{year + 1}-01-01"), **(load_kwargs or {})
        )
        obs = pd.DataFrame(obs.drop(columns="geometry"))
        if obs.empty:
            print(f"{year}: no observations")
            continue
        ds = build_network_transects(obs, pts, lag=lag, **build_kwargs)
        if ds.sizes["transect"] == 0:
            print(f"{year}: no transits")
            continue
        ds.attrs.update(
            site=site,
            year=year,
            created=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        )
        path = out_dir / f"{site}_CH4_{year}.nc"
        enc = {v: {"zlib": True, "complevel": 4} for v in ("obs", "time", "n")}
        ds.to_netcdf(path, encoding=enc)
        print(
            f"{year}: {ds.sizes['transect']} transits from {ds.attrs['n_samples']:,} samples "
            f"({ds.attrs['n_off_network']:,} off-network, "
            f"{ds.attrs['n_unresolved_line']:,} unresolved line) -> {path}"
        )
        written.append(path)
    return written


def load_network_transects(
    years=None, transects_dir: str | Path | None = None, site: str = "trx01"
):
    """
    The network transect matrices of ``years`` (all on disk by default), concatenated
    along ``transect`` with a fresh 0..N-1 index. See :func:`build_network_transects` for
    the layout.
    """
    import xarray as xr

    d = network_transects_dir(transects_dir)
    files = sorted(d.glob(f"{site}_CH4_*.nc"))
    if years is not None:
        want = {str(int(y)) for y in years}
        files = [f for f in files if f.stem.split("_")[-1] in want]
    if not files:
        raise FileNotFoundError(f"no network transect files for {site} in {d}")
    parts = [xr.open_dataset(f) for f in files]
    out = xr.concat(parts, dim="transect", combine_attrs="drop_conflicts")
    return out.assign_coords(transect=np.arange(out.sizes["transect"]))
