from pathlib import Path

import lair.pcaps
import lair.soundings
import pandas as pd

from slv import get_data_dir


def get_soundings(
    station="SLC",
    start=None,
    end=None,
    sounding_dir=None,
    months=None,
    driver="pandas",
    **kwargs,
):
    """Load upper-air soundings for ``station`` with :func:`lair.soundings.get_soundings`.

    If ``sounding_dir`` is not given, soundings are read from
    ``$SLV_SOUNDINGS_DIR/<station>``.
    """
    if sounding_dir is None:
        sounding_dir = get_data_dir("SLV_SOUNDINGS_DIR") / station
    return lair.soundings.get_soundings(
        station=station,
        start=start,
        end=end,
        sounding_dir=sounding_dir,
        months=months,
        driver=driver,
        **kwargs,
    )


DEFAULT_THRESHOLD = 4.04  # K, Whiteman et al. (2014)
DEFAULT_MIN_PERIODS = 3


def _pcap_events_cache(threshold, min_periods) -> Path | None:
    """Cache file for PCAP events found with ``threshold`` and ``min_periods``, or
    None if ``SLV_USER_DATA_DIR`` is not set.

    The defaults use ``pcap_events.csv``; other parameters get their own file.
    """
    try:
        data_dir = get_data_dir("SLV_USER_DATA_DIR")
    except OSError:
        return None
    if threshold == DEFAULT_THRESHOLD and min_periods == DEFAULT_MIN_PERIODS:
        return data_dir / "pcap_events.csv"
    return data_dir / f"pcap_events_t{threshold:g}_m{min_periods}.csv"


def _determine_pcap_events(time_range, threshold, min_periods, sounding_kwargs=None):
    """PCAP events from the soundings in ``time_range`` ((None, None) for all)."""
    driver = "xarray"  # Use xarray for aligned (interpolated values) soundings
    soundings = get_soundings(
        start=time_range[0], end=time_range[1], driver=driver, **(sounding_kwargs or {})
    )
    vhd = lair.pcaps.valleyheatdeficit(soundings)
    return lair.pcaps.determine_pcap_events(
        vhd, threshold=threshold, min_periods=min_periods
    )


def get_pcap_events(
    time_range,
    threshold=DEFAULT_THRESHOLD,
    min_periods=DEFAULT_MIN_PERIODS,
    sounding_kwargs=None,
):
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
    If the environment variable ``SLV_USER_DATA_DIR`` is set, events are cached per
    (threshold, min_periods): ``$SLV_USER_DATA_DIR/pcap_events.csv`` for the defaults,
    ``pcap_events_t<threshold>_m<min_periods>.csv`` otherwise. A missing cache file is
    built from the full sounding record, and every cached event is returned whatever
    the ``time_range``. Without the cache, or with ``sounding_kwargs``, events are
    determined from the soundings in ``time_range`` only.
    """
    cache = None
    if not sounding_kwargs:
        cache = _pcap_events_cache(threshold, min_periods)

    if cache is None:
        return _determine_pcap_events(
            time_range, threshold, min_periods, sounding_kwargs
        )

    if not cache.exists():
        events = _determine_pcap_events((None, None), threshold, min_periods)
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving PCAP events to {cache}")
        events.to_csv(cache, index=False)

    print(f"Loading cached PCAP events from {cache}")
    return pd.read_csv(cache, parse_dates=["start", "end"])


def filter_pcap_events(data: pd.Series | pd.DataFrame, level=None):
    """Drop the rows of ``data`` that fall within a PCAP event.

    ``data`` must have a datetime index; for a MultiIndex, ``level`` names the
    time level.
    """
    times = data.index.get_level_values(level) if level is not None else data.index
    time_range = (times.min(), times.max())
    events = get_pcap_events(time_range)
    return lair.pcaps.filter_pcap_events(data, events=events, level=level)
