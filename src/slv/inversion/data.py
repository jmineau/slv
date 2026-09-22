"""Observations for the inversion: tower and TRAX CH4, and the per-obs within-hour std."""

from pathlib import Path

import numpy as np
import pandas as pd

from slv.domain import UTC_OFFSET
from slv.measurements import aggregate_obs, load_concentrations
from slv.measurements.mobile import load_trax_points


def split_sites(
    sites: list[str], site_config: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Split ``sites`` into (stationary, mobile) by their ``site_config`` type."""
    mobile = [
        s
        for s in sites
        if s in site_config.index and site_config.at[s, "type"] == "mobile"
    ]
    return [s for s in sites if s not in mobile], mobile


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


def load_mobile_obs(
    mobile_obs: str | Path | pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    utc_offset: int = UTC_OFFSET,
) -> pd.DataFrame:
    """Receptor-paired mobile observations, filtered like the stationary ones.

    ``mobile_obs`` is the output of
    :func:`slv.measurements.mobile.trax_receptor_observations` (or a parquet of it): indexed
    ``(obs_location, obs_time)`` with ``obs_location`` the receptor's PYSTILT location_id and
    ``obs_time`` its naive-UTC release time. Applies the same time range, local-hour window
    (``subset_hours`` in standard time, ``utc_offset`` from UTC) and PCAP-day exclusion the
    stationary path applies, and returns just the ``CH4`` column.
    """
    df = (
        pd.read_parquet(mobile_obs)
        if not isinstance(mobile_obs, pd.DataFrame)
        else mobile_obs
    )
    if list(df.index.names) != ["obs_location", "obs_time"]:
        df = df.set_index(["obs_location", "obs_time"])
    t = pd.DatetimeIndex(df.index.get_level_values("obs_time"))
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    t0, t1 = (pd.Timestamp(x) for x in time_range)
    t0 = t0.tz_convert("UTC").tz_localize(None) if t0.tz is not None else t0
    t1 = t1.tz_convert("UTC").tz_localize(None) if t1.tz is not None else t1
    keep = np.asarray((t >= t0) & (t < t1))
    if subset_hours is not None:
        keep &= np.asarray((t + pd.Timedelta(hours=utc_offset)).hour.isin(subset_hours))
    df = df.iloc[np.flatnonzero(keep)]
    if filter_pcaps and not df.empty:
        from slv.meteorology.pcaps import filter_pcap_events

        times = pd.DatetimeIndex(df.index.get_level_values("obs_time"))
        pos = pd.Series(np.arange(len(df)), index=times)
        df = df.iloc[np.sort(filter_pcap_events(pos).to_numpy())]
    return df[["CH4"]]


def get_slv_observations(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    filter_spikes: bool = False,
    spike_percentile: float = 0.90,
    num_processes: int = 1,
    mobile_obs: str | Path | pd.DataFrame | None = None,
    utc_offset: int = UTC_OFFSET,
) -> pd.DataFrame:
    """Fetches observations for the pipeline.

    Returns DataFrame indexed by (obs_location, obs_time) with a CH4 column.
    For stationary sites, obs_location is the site name (e.g. "wbb").
    For mobile sites, obs_location is the STILT location_id string, which directly
    matches the simulation location_id so no separate location mapper is needed.

    ``mobile_obs`` (receptor-paired observations, :func:`load_mobile_obs`) supplies the
    mobile sites' obs when given: each is keyed exactly like the receptor it pairs with
    (a crossing's multipoint location_id, a dwell's point location_id) at its actual release
    time. Without it, mobile sites fall back to the old hourly aggregation onto the staged
    2-km points, whose keys only match point receptors at those points.
    """
    stationary, mobile_sites = split_sites(sites, site_config)
    if mobile_obs is not None and mobile_sites:
        parts = []
        if stationary:
            parts.append(
                get_slv_observations(
                    stationary,
                    site_config,
                    time_range,
                    subset_hours=subset_hours,
                    filter_pcaps=filter_pcaps,
                    filter_spikes=filter_spikes,
                    spike_percentile=spike_percentile,
                    num_processes=num_processes,
                    utc_offset=utc_offset,
                )
            )
        parts.append(
            load_mobile_obs(
                mobile_obs,
                time_range,
                subset_hours=subset_hours,
                filter_pcaps=filter_pcaps,
                utc_offset=utc_offset,
            )
        )
        return pd.concat(parts).sort_index()

    obs = load_concentrations(
        pollutants=["CH4"],
        sites=sites,
        time_range=time_range,
        site_config=site_config,
        subset_hours=subset_hours,
        filter_pcaps=filter_pcaps,
        num_processes=num_processes,
        utc_offset=utc_offset,
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


def _empty_subhour_std() -> pd.Series:
    index = pd.MultiIndex.from_arrays([[], []], names=["obs_location", "obs_time"])
    return pd.Series(dtype=float, index=index, name="subhour_std")


def get_slv_subhour_std(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    num_processes: int = 1,
    utc_offset: int = UTC_OFFSET,
) -> pd.Series:
    """Per-obs within-hour CH4 std -- the temporal representativeness error for the ``subhour``
    MDM component.

    Loads the native (sub-hourly) record and returns, for each hourly obs, the std of the
    sub-hour CH4 measurements in that hour: precisely the sub-hour signal an hour-mean footprint
    cannot represent (the same quantity the spike filter thresholds, kept per-obs instead of used
    to drop the top decile). Indexed (obs_location, obs_time) to match ``get_slv_observations``;
    hours with a single native point (std undefined) are NaN, so the caller can fill 0 (no
    representativeness penalty). Stationary sites only (obs_location == site); mobile obs return
    no rows and get 0 downstream, so mobile sites are not loaded at all (their record would go
    through the full GPS merge only to be discarded).
    """
    sites, _ = split_sites(sites, site_config)
    if not sites:
        return _empty_subhour_std()
    obs = load_concentrations(
        pollutants=["CH4"],
        sites=sites,
        time_range=time_range,
        site_config=site_config,
        subset_hours=subset_hours,
        filter_pcaps=filter_pcaps,
        num_processes=num_processes,
        utc_offset=utc_offset,
    )
    if obs.empty:
        return _empty_subhour_std()
    t = pd.to_datetime(obs["Time_UTC"])
    tmp = pd.DataFrame(
        {
            "obs_location": obs["site"].to_numpy(),  # stationary: obs_location == site
            "obs_time": t.dt.floor("h").to_numpy(),
            "ch4": obs["CH4"].to_numpy(),
        }
    )
    return tmp.groupby(["obs_location", "obs_time"])["ch4"].std().rename("subhour_std")
