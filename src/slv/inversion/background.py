import pandas as pd
from lair.background import rolling_baseline

from slv.measurements import aggregate_obs, load_concentrations


def get_slv_background(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    num_processes: int = 1,
    filter_pcaps: bool = False,
    baseline_window: int = 24,
    min_periods: int = 1,
) -> pd.Series:
    """Fetches background by applying a rolling baseline to the raw data.

    Returns Series with obs_time index and regional background concentration.
    """
    data = load_concentrations(
        pollutants=["CH4"],
        sites=sites,
        site_config=site_config,
        time_range=time_range,
        num_processes=num_processes,
        filter_pcaps=filter_pcaps,
    )
    data = aggregate_obs(
        data,
        freq="1h",  # hourly aggregation
        mobile_grid_res=0.02,  # ~2km grid for mobile sites
        stationary_min_percent=0.75,  # require 75% of expected samples for stationary sites
        mobile_min_count=10,  # require at least 10 samples per mobile grid cell
    )
    data = data.rename(columns={"Time_UTC": "obs_time"})
    # Pivot to get site columns
    df = data.set_index(["obs_time", "site"])["CH4"].unstack(fill_value=None)

    bg_dict = {}
    for site in df.columns:
        bg_dict[site] = rolling_baseline(
            df[site],
            window=baseline_window,
            min_periods=min_periods,
        )

    bg_df = pd.DataFrame(bg_dict)

    # Average across sites to get a single regional background
    background = bg_df.mean(axis=1)
    background.name = "concentration"
    background.index.name = "obs_time"
    return background
