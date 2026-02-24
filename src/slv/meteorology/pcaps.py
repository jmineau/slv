import os
from pathlib import Path

import lair.pcaps
import lair.soundings
import pandas as pd

from slv import get_data_dir

PCAP_EVENTS_CSV = (
    Path(pcap_dir) / "pcap_events.csv"
    if (pcap_dir := os.environ.get("SLV_PCAP_DIR"))
    else None
)


def get_soundings(
    station="SLC",
    start=None,
    end=None,
    sounding_dir=None,
    months=None,
    driver="pandas",
    **kwargs,
):
    if sounding_dir is None:
        sounding_dir = get_data_dir("SLV_SOUNDINGS_DIR")
    return lair.soundings.get_soundings(
        station=station,
        start=start,
        end=end,
        sounding_dir=sounding_dir,
        months=months,
        driver=driver,
        **kwargs,
    )


def get_pcap_events(time_range, threshold=4.04, min_periods=3, sounding_kwargs=None):
    """Determines PCAP events based on valley heat deficit from soundings.

    Parameters
    ----------
    time_range : tuple
        (start, end) timestamps to define the period for which to determine PCAP events.
    threshold : float
        Valley heat deficit threshold to identify PCAP events. Default is 4.04 K from Whiteman (2014).
    min_periods : int
        Minimum number of sounding periods that must exceed the threshold to define a PCAP event. Default is 3.
    sounding_kwargs : dict
        Additional keyword arguments to pass to the get_soundings function.

    Notes
    -----
    If the environment variable ``SLV_PCAP_DIR`` is set, events are cached as
    ``$SLV_PCAP_DIR/pcap_events.csv`` and reloaded on subsequent calls.
    """
    if PCAP_EVENTS_CSV is not None and PCAP_EVENTS_CSV.exists():
        print(f"Loading cached PCAP events from {PCAP_EVENTS_CSV}")
        return pd.read_csv(PCAP_EVENTS_CSV, parse_dates=["start", "end"])

    driver = "xarray"  # Use xarray for aligned (interpolated values) soundings
    soundings = get_soundings(
        start=time_range[0], end=time_range[1], driver=driver, **(sounding_kwargs or {})
    )
    vhd = lair.pcaps.valleyheatdeficit(soundings)
    events = lair.pcaps.determine_pcap_events(
        vhd, threshold=threshold, min_periods=min_periods
    )

    if PCAP_EVENTS_CSV is not None:
        PCAP_EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving PCAP events to {PCAP_EVENTS_CSV}")
        events.to_csv(PCAP_EVENTS_CSV, index=False)

    return events


def filter_pcap_events(data: pd.Series | pd.DataFrame, level=None):
    time_range = (data.index.min(), data.index.max())
    events = get_pcap_events(time_range)
    return lair.pcaps.filter_pcap_events(data, events=events, level=level)
