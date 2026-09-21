"""TRAX CH4 observations paired one-to-one with the STILT receptors.

The inversion joins a Jacobian row to an observation on ``(obs_location, obs_time)``: for a
mobile receptor ``obs_location`` is the receptor's PYSTILT ``location_id`` (it has no site
name to map to) and ``obs_time`` is its release time. So every observation here is built
from the receptor it belongs to, and keyed with the *same* identifiers the simulations were
given -- the receptors are read back with :func:`stilt.read_receptors`, the function the
batch workers use, so the keys cannot drift from the footprints.

Two receptor types (see :mod:`.receptors`):

* **crossing** -- one pass of a 2-km segment. The observation is the mean CH4 over the pass,
  i.e. over the samples logged between the pass's first and last GPS fix.
* **dwell** -- one hour of the train parked at one outdoor spot. The observation is the mean
  CH4 over the samples in that hour of the dwell.

**Inlet lag.** A sample logged at time *t* is air that entered the roof inlet at *t - lag*,
so the air drawn in while the train was on the segment during ``[t_start, t_end]`` is logged
during ``[t_start + lag, t_end + lag]``. The window is shifted by the lag for its epoch
(:func:`inlet_lag_seconds`); the lag is still being refined, so it is an input, and the
receptors themselves do not depend on it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

#: Location states whose CH4 counts for each receptor type (``obs.parquet`` ``state``).
#: A crossing is a traverse, so on-track only; a dwell may be parked on a yard track, but
#: indoor shed air is never ambient and is excluded for both.
CROSSING_STATES = ("route", "line", "stopped")
DWELL_STATES = ("route", "line", "stopped", "yard")

_DWELL_RE = re.compile(r"^dwell_(\d+)_(\d{10})$")


def inlet_lag_seconds(
    times, lag: float | pd.DataFrame | None
) -> tuple[np.ndarray, np.ndarray]:
    """Inlet lag in seconds at each of ``times``, and where it came from.

    ``lag`` is ``None`` (no shift), a number (one lag throughout) or a table with ``start``,
    ``end`` and ``lag_s`` columns -- the curated per-epoch table from the inlet-lag workflow.
    A time inside an epoch gets that epoch's lag (source ``"epoch"``); a time in a gap
    between epochs gets the nearest epoch's lag (source ``"nearest"``), so data are kept but
    the assumption stays visible and can be filtered on.
    """
    t = pd.DatetimeIndex(pd.to_datetime(times))
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    n = len(t)
    if lag is None:
        return np.zeros(n), np.full(n, "none", dtype=object)
    if isinstance(lag, (int, float)):
        return np.full(n, float(lag)), np.full(n, "constant", dtype=object)

    start = np.asarray(pd.to_datetime(lag["start"]), dtype="datetime64[ns]")
    # an epoch's end date is inclusive: it covers that whole day
    end = np.asarray(
        pd.to_datetime(lag["end"]) + pd.Timedelta(days=1), dtype="datetime64[ns]"
    )
    lag_s = lag["lag_s"].to_numpy(float)
    out = np.full(n, np.nan)
    src = np.full(n, "nearest", dtype=object)
    tv = np.asarray(t, dtype="datetime64[ns]")
    for s, e, v in zip(start, end, lag_s, strict=True):
        inside = (tv >= s) & (tv < e)
        out[inside] = v
        src[inside] = "epoch"
    gap = np.isnan(out)
    if gap.any():
        # distance from each gap time to each epoch (0 if inside, else to the nearer edge)
        d = np.minimum(
            np.abs(tv[gap][:, None] - start[None, :]),
            np.abs(tv[gap][:, None] - end[None, :]),
        )
        out[gap] = lag_s[d.argmin(axis=1)]
    return out, src


def _receptor_keys(receptors_csv: str | Path) -> pd.DataFrame:
    """``r_idx``, ``obs_location`` and ``obs_time`` for every receptor in a PYSTILT CSV,
    computed by :func:`stilt.read_receptors` exactly as the batch workers compute them."""
    import stilt

    recs = stilt.read_receptors(receptors_csv)
    order = pd.read_csv(
        receptors_csv, usecols=["r_idx"], dtype={"r_idx": str}
    ).r_idx.unique()
    if len(order) != len(recs):
        raise ValueError(
            f"{receptors_csv}: {len(order)} r_idx values but {len(recs)} receptors -- "
            "the receptor CSV and read_receptors disagree on grouping."
        )
    return pd.DataFrame(
        {
            "r_idx": order,
            "obs_location": [str(r.location_id) for r in recs],
            "obs_time": pd.DatetimeIndex([r.time for r in recs]),
            "kind": ["dwell" if _DWELL_RE.match(k) else "crossing" for k in order],
        }
    )


def _windows(
    keys: pd.DataFrame, crossings: pd.DataFrame, dwells: pd.DataFrame
) -> pd.DataFrame:
    """Unlagged sample window ``[t0, t1)`` of each receptor, from the tables that built it."""
    t0 = pd.Series(pd.NaT, index=keys.index, dtype="datetime64[ns]")
    t1 = t0.copy()

    cross = keys.kind == "crossing"
    if cross.any():
        c = crossings.set_index(crossings["crossing"].astype(str))
        ids = keys.loc[cross, "r_idx"]
        found = ids.isin(c.index)
        hit = ids[found]
        t0.loc[hit.index] = pd.to_datetime(c.loc[hit, "t_start"]).to_numpy()
        # t_end is the last fix; include it
        t1.loc[hit.index] = (
            pd.to_datetime(c.loc[hit, "t_end"]) + pd.Timedelta(seconds=1)
        ).to_numpy()

    dwell = keys.kind == "dwell"
    if dwell.any():
        dw = dwells.set_index(dwells["dwell"].astype(int))
        for i, r in keys.loc[dwell, "r_idx"].items():
            m = _DWELL_RE.match(r)
            if m is None:  # kind came from this same pattern, so this cannot happen
                continue
            did = int(m.group(1))
            hour = pd.Timestamp(f"{m.group(2)[:8]} {m.group(2)[8:]}:00")
            if did not in dw.index:
                continue
            row = dw.loc[did]
            t0.loc[i] = max(hour, pd.Timestamp(row.t_start))
            t1.loc[i] = min(hour + pd.Timedelta(hours=1), pd.Timestamp(row.t_end))
    return pd.DataFrame({"t0": t0, "t1": t1}, index=keys.index)


def trax_receptor_observations(
    receptors_csv: str | Path,
    crossings: pd.DataFrame,
    dwells: pd.DataFrame,
    ch4: pd.DataFrame,
    lag: float | pd.DataFrame | None = None,
    min_samples: int = 5,
    include_uncalibrated: bool = True,
    include_low_pressure: bool = True,
    crossing_states: Sequence[str] = CROSSING_STATES,
    dwell_states: Sequence[str] = DWELL_STATES,
) -> pd.DataFrame:
    """One CH4 observation per STILT receptor, keyed for the inversion.

    Parameters
    ----------
    receptors_csv
        The PYSTILT receptor CSV the simulations were run from (``receptors_YYYY.csv``).
    crossings, dwells
        The tables that built those receptors (``crossings_YYYY.parquet``,
        ``dwells_YYYY.parquet``), which carry each receptor's sample window.
    ch4
        The georeferenced CH4 record (``obs.parquet`` / :func:`load_trax_obs` with
        ``location="all"``) with ``Time_UTC``, ``CH4_ppm``, ``state``, ``cal_source`` and
        ``low_pressure``.
    lag
        Inlet lag: ``None``, seconds, or the per-epoch table (:func:`inlet_lag_seconds`).
    min_samples
        Receptors with fewer usable samples in their window get no observation.

    Returns
    -------
    DataFrame indexed ``(obs_location, obs_time)`` -- the inversion's observation key -- with
    ``CH4`` [ppm, window mean], ``ch4_std``, ``n``, ``kind``, ``r_idx``, ``lag_s``,
    ``lag_source`` and ``frac_uncalibrated``.
    """
    keys = _receptor_keys(receptors_csv)
    win = _windows(keys, crossings, dwells)
    mid = win.t0 + (win.t1 - win.t0) / 2
    lag_s, lag_src = inlet_lag_seconds(mid.fillna(pd.Timestamp("2000-01-01")), lag)
    shift = pd.to_timedelta(lag_s, unit="s")
    lo, hi = (win.t0 + shift).to_numpy(), (win.t1 + shift).to_numpy()

    rec = ch4.copy()
    t = pd.DatetimeIndex(rec["Time_UTC"])
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    rec["Time_UTC"] = t
    rec = rec.loc[rec["CH4_ppm"].notna()]
    if not include_uncalibrated and "cal_source" in rec:
        rec = rec.loc[rec["cal_source"] != "uncalibrated"]
    if not include_low_pressure and "low_pressure" in rec:
        rec = rec.loc[~rec["low_pressure"].fillna(False).astype(bool)]
    rec = rec.sort_values("Time_UTC")
    tv = rec["Time_UTC"].to_numpy()
    ch = rec["CH4_ppm"].to_numpy(float)
    state = (
        rec["state"].to_numpy(object)
        if "state" in rec
        else np.full(len(rec), "route", object)
    )
    uncal = (
        (rec["cal_source"] == "uncalibrated").to_numpy()
        if "cal_source" in rec
        else np.zeros(len(rec), bool)
    )
    ok_cross = np.isin(state, list(crossing_states))
    ok_dwell = np.isin(state, list(dwell_states))

    a = np.searchsorted(tv, lo, side="left")
    b = np.searchsorted(tv, hi, side="left")
    rows = []
    for i, (ia, ib) in enumerate(zip(a, b, strict=True)):
        if pd.isna(win.t0.iat[i]) or ib <= ia:
            continue
        ok = (ok_dwell if keys.kind.iat[i] == "dwell" else ok_cross)[ia:ib]
        vals = ch[ia:ib][ok]
        if len(vals) < min_samples:
            continue
        rows.append(
            (
                keys.obs_location.iat[i],
                keys.obs_time.iat[i],
                float(vals.mean()),
                float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                len(vals),
                keys.kind.iat[i],
                keys.r_idx.iat[i],
                float(lag_s[i]),
                lag_src[i],
                float(uncal[ia:ib][ok].mean()),
            )
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "obs_location",
            "obs_time",
            "CH4",
            "ch4_std",
            "n",
            "kind",
            "r_idx",
            "lag_s",
            "lag_source",
            "frac_uncalibrated",
        ],
    )
    return out.set_index(["obs_location", "obs_time"]).sort_index()


__all__ = [
    "CROSSING_STATES",
    "DWELL_STATES",
    "inlet_lag_seconds",
    "trax_receptor_observations",
]
