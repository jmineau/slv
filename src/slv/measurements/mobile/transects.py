"""The archived TRAX transect matrices (transect x point, per line and month)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slv.measurements.mobile.network import user_dir

TRANSECT_LINES = {"r": "Red", "g": "Green", "b": "Blue"}


def load_transects(line: str, months=None, transects_dir: str | Path | None = None):
    """Concatenate the archived transect matrices for one TRAX line into an xarray Dataset.

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
