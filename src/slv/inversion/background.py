import pandas as pd
from lair.background import rolling_baseline

from slv.inversion.data import load_concentrations


def get_slv_background(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    num_processes: int = 1,
    filter_pcaps: bool = True,
    baseline_window: int = 24,
    min_periods: int = 1,
) -> pd.Series:
    """Fetches background by applying a rolling baseline to the raw data."""
    data = load_concentrations(
        sites=sites,
        site_config=site_config,
        time_range=time_range,
        num_processes=num_processes,
        filter_pcaps=filter_pcaps,
        subset_hours=None,
    )
    df = data.unstack(level="obs_location")

    bg_dict = {}
    for site in df.columns:
        bg_dict[site] = rolling_baseline(
            df[site].resample("1h").mean(),
            window=baseline_window,
            min_periods=min_periods,
        )

    bg_df = pd.DataFrame(bg_dict)

    # Average across sites to get a single regional background
    background = bg_df.mean(axis=1)
    background.name = "concentration"
    background.index.name = "obs_time"
    return background
