import pandas as pd

import lair.soundings
import lair.pcaps

from slv import get_data_dir


def get_soundings(station='SLC', start=None, end=None, sounding_dir=None, months=None,
                  driver='pandas', **kwargs):
    if sounding_dir is None:
        sounding_dir = get_data_dir("SLV_SOUNDINGS_DIR")
    return lair.soundings.get_soundings(station=station, start=start, end=end, sounding_dir=sounding_dir,
                         months=months, driver=driver, **kwargs)


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
    """
    driver = 'xarray'  # Use xarray for aligned (interpolated values) soundings
    soundings = get_soundings(start=time_range[0], end=time_range[1], driver=driver, **(sounding_kwargs or {}))
    vhd = lair.pcaps.valleyheatdeficit(soundings)
    return lair.pcaps.determine_pcap_events(vhd, threshold=threshold, min_periods=min_periods)


def filter_pcap_events(data: pd.Series | pd.DataFrame, level=None):
    time_range = (data.index.min(), data.index.max())
    events = get_pcap_events(time_range)
    return lair.pcaps.filter_events(data, events, level=level)