"""Calibration provenance of the TRAX CH4 record: ``cal_source`` tags and the windows
where the analyzer ran without a reference tank but its raw data are usable.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd

CAL_SOURCES = ("pipeline", "manual_cal", "uncalibrated")
"""Provenance tag carried by every TRAX observation in ``cal_source``:

- ``pipeline``: pipeline-calibrated value (``CH4d_ppm_cal`` from the ``calibrated`` level).
- ``manual_cal``: the ``lgr_ugga_manual_cal`` instrument (installed 2023-11-18, when the
  on-board tank was removed; a tank ran again in 2024 and the last pipeline calibration is
  2024-08-20); the pipeline applies no calibration, so this is the analyzer's raw ``CH4d_ppm``.
- ``uncalibrated``: raw ``CH4d_ppm`` from the ``lgr_ugga`` qaqc level inside a window listed in
  ``trax_uncalibrated_windows.csv`` (tank empty, no valid reference). Same treatment as
  ``manual_cal``; the LGR's gain was within 0.5% of unity on either side of every window.
"""


def load_trax_uncalibrated_windows(
    path: str | Path | None = None, enabled_only: bool = True
) -> pd.DataFrame:
    """Windows where the LGR ran without a valid reference tank but the raw data are good.

    Packaged in ``trax_uncalibrated_windows.csv`` (columns: start, end, reason, enabled; start/end
    are ISO dates or date-times, e.g. ``2015-10-11T16:00``).
    Set ``enabled`` to false to drop a window without deleting the row.
    """
    if path is None:
        with (
            files(__package__).joinpath("trax_uncalibrated_windows.csv").open("r") as f
        ):
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"], format="ISO8601")  # dates or date-times
    df["end"] = pd.to_datetime(df["end"], format="ISO8601")
    df["enabled"] = df["enabled"].astype(str).str.lower().isin(("true", "1", "yes"))
    if enabled_only:
        df = df.loc[df["enabled"].to_numpy()]
    return pd.DataFrame(df).reset_index(drop=True)


def load_trax_epoch_offsets(
    path: str | Path | None = None, enabled_only: bool = True
) -> pd.DataFrame:
    """Analyzer-epoch offsets of the uncalibrated (manual-cal) era, ppm.

    Packaged in ``trax_epoch_offsets.csv`` (start, end, analyzer, offset_ppm, n_hours,
    enabled, note). Since the tank came off the train (2023-11-17) the pipeline applies no
    calibration at all, so each analyzer carries its own gain. ``offset_ppm`` is the epoch's
    median TRAX-minus-UOU afternoon difference measured where the train runs within 2 km of
    the UOU tower, against a tank-calibrated reference level of 0.000 ppm (the tank epochs
    sit at -0.005 to +0.009 ppm, which is the method's own uncertainty). Values are
    -0.002 to -0.024 ppm; :func:`apply_epoch_offset` subtracts them.

    Derived in ``measurements/trax/audit/`` (``checks/answers.py``); rebuild it there after
    an analyzer swap or a rebuild of ``obs.parquet``.
    """
    if path is None:
        with files(__package__).joinpath("trax_epoch_offsets.csv").open("r") as f:
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"], format="ISO8601")
    df["end"] = pd.to_datetime(df["end"], format="ISO8601")
    df["enabled"] = df["enabled"].astype(str).str.lower().isin(("true", "1", "yes"))
    if enabled_only:
        df = df.loc[df["enabled"].to_numpy()]
    return pd.DataFrame(df).reset_index(drop=True)


def epoch_offset_column(times, offsets: pd.DataFrame | None = None) -> pd.Series:
    """The epoch offset (ppm) in force at each time, 0.0 outside every epoch.

    Stored in ``obs.parquet`` as ``epoch_offset_ppm`` so the correction can be applied or
    removed at load time without rebuilding (:func:`apply_epoch_offset`).
    """
    if offsets is None:
        offsets = load_trax_epoch_offsets()
    t = pd.DatetimeIndex(pd.to_datetime(times))
    out = pd.Series(0.0, index=range(len(t)))
    for start, end, off in zip(
        offsets["start"], offsets["end"], offsets["offset_ppm"], strict=True
    ):
        out[(t >= start) & (t < end)] = float(off)
    return out


def apply_epoch_offset(df: pd.DataFrame, apply: bool = True) -> pd.DataFrame:
    """Subtract the analyzer-epoch offset from ``CH4_ppm`` (no-op when ``apply`` is False).

    Needs the ``epoch_offset_ppm`` column written by :func:`build_trax_obs`; frames built
    before it existed are returned unchanged. The uncorrected value is recoverable as
    ``CH4_ppm + epoch_offset_ppm``, so ``load_trax_obs(epoch_offset=False)`` and
    ``True`` differ only by this step — test the effect without rebuilding.
    """
    if not apply or "epoch_offset_ppm" not in df.columns:
        return df
    out = df.copy()
    out["CH4_ppm"] = out["CH4_ppm"] - pd.to_numeric(
        out["epoch_offset_ppm"], errors="coerce"
    ).fillna(0.0)
    return out


def select_uncalibrated(
    qaqc: pd.DataFrame, windows: pd.DataFrame, exclude_times: pd.Index | None = None
) -> pd.DataFrame:
    """Rows of a qaqc-level LGR frame that fall inside the uncalibrated windows.

    ``qaqc`` must have a ``Time_UTC`` column (or a datetime index) and a ``CH4`` column that
    has already passed :func:`slv.measurements.pollutants.normalize_pollutant`. Rows whose
    time is in ``exclude_times`` (e.g. times that do have a pipeline calibration) are dropped.
    Returns ``Time_UTC``, ``CH4`` and ``cal_source == "uncalibrated"``.
    """
    df = qaqc if "Time_UTC" in qaqc.columns else qaqc.reset_index()
    t = pd.to_datetime(df["Time_UTC"])
    mask = pd.Series(False, index=df.index)
    for start, end in zip(windows["start"], windows["end"], strict=True):
        mask |= (t >= start) & (t < end)
    out = df.loc[mask & df["CH4"].notna(), ["Time_UTC", "CH4"]].copy()
    if exclude_times is not None and len(exclude_times):
        out = out[~out["Time_UTC"].isin(exclude_times)]
    out["cal_source"] = "uncalibrated"
    return pd.DataFrame(out).reset_index(drop=True)


def filter_cal_source(
    df: pd.DataFrame, include_uncalibrated: bool = True
) -> pd.DataFrame:
    """Drop the ``uncalibrated`` rows when ``include_uncalibrated`` is False.

    ``manual_cal`` rows are always kept: after the last pipeline calibration (2024-08-20)
    they are the only data.
    """
    if include_uncalibrated or "cal_source" not in df.columns:
        return df
    return pd.DataFrame(df.loc[(df["cal_source"] != "uncalibrated").to_numpy()])
