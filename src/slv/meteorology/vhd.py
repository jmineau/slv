"""Valley heat deficit (VHD) for the Salt Lake Valley.

VHD is the energy per unit area needed to mix the valley atmosphere to a dry
adiabat, computed from the SLC sounding. It is the standard continuous measure
of how strongly the valley is stratified: near zero when the valley is well
mixed, several MJ m-2 during a persistent cold-air pool.

The series is staged under ``$SLV_USER_DATA_DIR/heatdeficit`` and is twice
daily, at the 00 and 12 UTC sounding times.
"""

from __future__ import annotations

import pandas as pd

from slv import get_data_dir

#: The VHD series to prefer, longest first. Both are integrated to 2200 m ASL.
_FILES = ("vhd22_1998_2024.csv", "vhd22_2014_2024.csv")


def load_vhd(time_range: tuple | None = None) -> pd.Series:
    """Valley heat deficit in MJ m-2, indexed by ``Time_UTC`` at the sounding times.

    The stored column is in J m-2; it is converted here so the units match how
    the quantity is reported (Whiteman et al. 2014 uses MJ m-2).
    """
    directory = get_data_dir("SLV_USER_DATA_DIR") / "heatdeficit"
    for name in _FILES:
        path = directory / name
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No valley heat deficit series in {directory}")

    df = pd.read_csv(path, parse_dates=["Time_UTC"])
    series = df.set_index("Time_UTC")["VHD22"].sort_index() / 1e6
    series.name = "VHD_MJ_m2"
    if time_range is not None:
        series = series.loc[slice(*time_range)]
    return series
