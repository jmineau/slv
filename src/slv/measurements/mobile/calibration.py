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
- ``manual_cal``: the ``lgr_ugga_manual_cal`` instrument (no on-board tank since Nov 2023);
  the pipeline applies no calibration, so this is the analyzer's raw ``CH4d_ppm``.
- ``uncalibrated``: raw ``CH4d_ppm`` from the ``lgr_ugga`` qaqc level inside a window listed in
  ``trax_uncalibrated_windows.csv`` (tank empty, no valid reference). Same treatment as
  ``manual_cal``; the LGR's gain was within 0.5% of unity on either side of every window.
"""


def load_trax_uncalibrated_windows(
    path: str | Path | None = None, enabled_only: bool = True
) -> pd.DataFrame:
    """Windows where the LGR ran without a valid reference tank but the raw data are good.

    Packaged in ``trax_uncalibrated_windows.csv`` (columns: start, end, reason, enabled).
    Set ``enabled`` to false to drop a window without deleting the row.
    """
    if path is None:
        with (
            files(__package__).joinpath("trax_uncalibrated_windows.csv").open("r") as f
        ):
            df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["enabled"] = df["enabled"].astype(str).str.lower().isin(("true", "1", "yes"))
    if enabled_only:
        df = df.loc[df["enabled"].to_numpy()]
    return pd.DataFrame(df).reset_index(drop=True)


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

    ``manual_cal`` rows are always kept: they are the only post-Nov-2023 data.
    """
    if include_uncalibrated or "cal_source" not in df.columns:
        return df
    return pd.DataFrame(df.loc[(df["cal_source"] != "uncalibrated").to_numpy()])
