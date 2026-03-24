from collections.abc import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from slv.measurements import instruments


def aggregate_obs(
    obs: pd.DataFrame,
    by: str | list[str] | Callable | None = None,
    freq: str | None = None,
    func: str = "mean",
    mobile_points: gpd.GeoDataFrame | None = None,
    mobile_grid_res: float | None = None,
    stationary_min_percent: float | None = None,
    mobile_min_count: int | None = None,
) -> pd.DataFrame:
    """
    Flexible aggregation for both stationary and mobile sites.
    - Stationary: group by site, time, and location columns (constant per site)
    - Mobile: group by spatial point/grid and time (or user grouping)
    Returns grouped DataFrame with original column names.
    """
    obs["Time_UTC"] = pd.to_datetime(obs["Time_UTC"])

    if "is_mobile" in obs.columns:
        stationary = obs[~obs["is_mobile"]]
        mobile = obs[obs["is_mobile"]]
    else:
        stationary = obs
        mobile = pd.DataFrame()

    agg_frames: list[pd.DataFrame] = []

    # --- Stationary ---
    if not stationary.empty:
        if by is not None or freq is not None:
            if by is not None:
                group_keys = by if isinstance(by, (list, tuple)) else [by]
            else:
                stationary = stationary.copy()
                stationary["agg_time"] = (
                    stationary["Time_UTC"].dt.to_period(freq).dt.to_timestamp()
                )
                group_keys = ["site", "agg_time"] + [
                    c
                    for c in [
                        "latitude",
                        "longitude",
                        "height",
                        "altitude",
                        "is_mobile",
                    ]
                    if c in stationary.columns and stationary[c].notna().any()
                ]

                if stationary_min_percent is not None:
                    freq_hours = pd.tseries.frequencies.to_offset(freq).nanos / 3.6e12
                    inst_expected = {}
                    for inst_name in stationary["instrument"].unique():
                        inst_cls = instruments.REGISTRY[inst_name]
                        if hasattr(inst_cls, "samples_per_hour"):
                            inst_expected[inst_name] = (
                                inst_cls.samples_per_hour * freq_hours
                            )
                        else:
                            inst_expected[inst_name] = (
                                freq_hours
                                * 3600
                                / pd.to_timedelta(inst_cls.sample_rate).total_seconds()
                            )
                    stationary["_expected"] = stationary["instrument"].map(
                        inst_expected
                    )
                    stationary = stationary.groupby(group_keys).filter(
                        lambda g: len(g) / g["_expected"].iloc[0]
                        >= stationary_min_percent
                    )
                    stationary = stationary.drop(columns="_expected")

            if not stationary.empty:
                numeric_cols = stationary.select_dtypes(include="number").columns
                agg_dict = {col: func for col in numeric_cols if col not in group_keys}
                agg_dict["instrument"] = lambda x: " ".join(sorted(x.unique()))
                grouped = stationary.groupby(group_keys).agg(agg_dict).reset_index()
                agg_frames.append(grouped.rename(columns={"agg_time": "Time_UTC"}))
        else:
            agg_frames.append(stationary)

    # --- Mobile ---
    if not mobile.empty:
        group_keys = ["is_mobile", "height"]

        if mobile_points is not None:
            obs_pts = np.array(
                list(zip(mobile["longitude"], mobile["latitude"], strict=False))
            )
            agg_pts = np.array(
                list(
                    zip(
                        mobile_points.geometry.x, mobile_points.geometry.y, strict=False
                    )
                )
            )
            _, idx = cKDTree(agg_pts).query(obs_pts, k=1)
            mobile["agg_lat"] = mobile_points.geometry.y.values[idx]
            mobile["agg_lon"] = mobile_points.geometry.x.values[idx]
            group_keys.extend(["site", "agg_lat", "agg_lon"])
        elif mobile_grid_res is not None:
            mobile["grid_lat"] = (
                mobile["latitude"] / mobile_grid_res
            ).round() * mobile_grid_res
            mobile["grid_lon"] = (
                mobile["longitude"] / mobile_grid_res
            ).round() * mobile_grid_res
            group_keys.extend(["site", "grid_lat", "grid_lon"])
        else:
            # No spatial binning — group on exact lat/lon
            group_keys.extend(["site", "latitude", "longitude", "altitude"])

        if by is not None or freq is not None:
            if by is not None:
                group_keys = by if isinstance(by, (list, tuple)) else [by]
            else:
                mobile["agg_time"] = (
                    mobile["Time_UTC"].dt.to_period(freq).dt.to_timestamp()
                )
                group_keys.append("agg_time")

        if mobile_min_count is not None:
            mobile = mobile.groupby(group_keys).filter(
                lambda g: len(g) >= mobile_min_count
            )

        numeric_cols = mobile.select_dtypes(include="number").columns
        # When spatial binning is active, lat/lon come from renamed bin columns
        spatial_active = mobile_points is not None or mobile_grid_res is not None
        exclude = group_keys + (["latitude", "longitude"] if spatial_active else [])
        agg_dict = {col: func for col in numeric_cols if col not in exclude}
        agg_dict["instrument"] = lambda x: " ".join(sorted(x.unique()))

        grouped = mobile.groupby(group_keys).agg(agg_dict).reset_index()
        agg_frames.append(
            grouped.rename(
                columns={
                    "agg_time": "Time_UTC",
                    "agg_lat": "latitude",
                    "agg_lon": "longitude",
                    "grid_lat": "latitude",
                    "grid_lon": "longitude",
                }
            )
        )

    if not agg_frames:
        raise ValueError("No data to aggregate — all observations were filtered out.")
    return pd.concat(agg_frames, ignore_index=True)
