import pandas as pd

from slv.measurements import aggregate_obs, load_concentrations


def get_slv_observations(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    num_processes: int = 1,
) -> pd.DataFrame:
    """Fetches observations for the pipeline.

    Returns DataFrame with obs_time, site, CH4, and optional location columns.
    """
    obs = load_concentrations(
        pollutants=["CH4"],
        sites=sites,
        time_range=time_range,
        site_config=site_config,
        subset_hours=subset_hours,
        filter_pcaps=filter_pcaps,
        num_processes=num_processes,
    )

    obs = aggregate_obs(
        obs,
        freq="1h",  # hourly aggregation
        mobile_grid_res=0.02,  # ~2km grid for mobile sites
        stationary_min_percent=0.75,  # require 75% of expected samples for stationary sites
        mobile_min_count=10,  # require at least 10 samples per mobile grid cell
    )

    obs = obs.rename(
        columns={"Time_UTC": "obs_time", "site": "obs_location"}
    )  # TODO what about mobile sites with multiple obs_locations?
    return obs.set_index(["obs_location", "obs_time"])["CH4"].to_frame()
