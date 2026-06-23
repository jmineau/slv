import pandas as pd

from slv.measurements import aggregate_obs, load_concentrations
from slv.measurements.mobile import load_trax_points


def _drop_spike_days(obs: pd.DataFrame, percentile: float) -> pd.DataFrame:
    """Drop (site, day) obs whose within-hour CH4 variance is anomalously high.

    For each site, computes the within-hour std of the native CH4 record,
    averages it over each day, and flags days above ``percentile`` of that
    site's distribution. High within-hour variance signals a passing plume or
    non-steady conditions that a footprint (which convolves the period mean)
    cannot faithfully represent, so those obs are removed before aggregation.
    Days with no multi-point hour (std undefined) are left in.
    """
    if obs.empty:
        return obs
    t = pd.to_datetime(obs["Time_UTC"])
    g = pd.DataFrame(
        {
            "site": obs["site"].to_numpy(),
            "ch4": obs["CH4"].to_numpy(),
            "hour": t.dt.floor("h").to_numpy(),
            "day": t.dt.floor("D").to_numpy(),
        }
    )
    hour_std = g.groupby(["site", "day", "hour"])["ch4"].std()
    daily_spike = hour_std.groupby(level=["site", "day"]).mean()
    thr = daily_spike.groupby(level="site").transform(lambda s: s.quantile(percentile))
    flagged = daily_spike.index[daily_spike > thr]
    if len(flagged) == 0:
        return obs
    keep = ~pd.MultiIndex.from_arrays([g["site"], g["day"]]).isin(flagged)
    return obs[keep]


def get_slv_observations(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    filter_spikes: bool = False,
    spike_percentile: float = 0.90,
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

    if filter_spikes:
        obs = _drop_spike_days(obs, spike_percentile)

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
