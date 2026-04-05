import pandas as pd

from slv.measurements import aggregate_obs, load_concentrations
from slv.measurements.mobile import load_trax_points


def get_slv_observations(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    num_processes: int = 1,
) -> pd.DataFrame:
    """Fetches observations for the pipeline.

    Returns DataFrame indexed by (obs_location, obs_time) with a CH4 column.
    For stationary sites, obs_location is the site name (e.g. "wbb").
    For mobile sites, obs_location is the STILT location_id string
    ("{lon}_{lat}_{zagl}"), which directly matches the simulation location_id
    so no separate location mapper is needed for mobile sites.
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
        freq="1h",
        mobile_points=load_trax_points(),  # snap to fixed TRAX route points
        stationary_min_percent=0.75,
        mobile_min_count=10,
    )

    # Build obs_location: site name for stationary, location_id for mobile.
    # Mobile location_ids use the snapped TRAX route point coordinates, which
    # already have 5dp precision from load_trax_points() and integer zagl.
    if "is_mobile" in obs.columns:
        mobile = obs["is_mobile"].astype(bool)
        obs["obs_location"] = obs["site"]
        obs.loc[mobile, "obs_location"] = (
            obs.loc[mobile, "longitude"].round(5).astype(str)
            + "_"
            + obs.loc[mobile, "latitude"].round(5).astype(str)
            + "_"
            + obs.loc[mobile, "height"].round(0).astype(int).astype(str)
        )
    else:
        obs["obs_location"] = obs["site"]

    obs = obs.rename(columns={"Time_UTC": "obs_time"})
    return obs.set_index(["obs_location", "obs_time"])["CH4"].to_frame()
