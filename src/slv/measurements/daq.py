from pathlib import Path

import pandas as pd

from slv import get_data_dir


def load_picarro_g2307(
    site: str, lvl: str, time_range=None, data_dir: Path | None = None
):
    """
    Load final Picarro G2307 data for a specific site.
    """
    if data_dir is None:
        data_dir = get_data_dir("SLV_DAQ_DIR")
    data_dir = Path(data_dir)
    site = site.lower()
    pattern = f"{site}/picarro_g2307/{lvl}/*.dat"
    data = pd.concat(
        pd.read_csv(f, parse_dates=["Time_UTC"]) for f in data_dir.rglob(pattern)
    )

    # Filter to time range of interest
    if time_range is not None:
        data = data.set_index("Time_UTC").sort_index()
        data = data.loc[time_range]
        data = data.reset_index()

    return data.dropna(subset="CH4d_ppm_cal")
