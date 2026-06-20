import pandas as pd
from lair.background import rolling_baseline

from slv.measurements import aggregate_obs, load_concentrations
from slv.measurements.background import GMLDiscrete


def get_slv_background(
    background: str,
    obs_times,
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    num_processes: int = 1,
    filter_pcaps: bool = False,
    **kwargs,
) -> pd.Series:
    """Dispatch background calculation by type.

    Returns Series with obs_time index and background concentration.
    """
    if background == "rolling":
        return get_rolling_background(
            sites=sites,
            site_config=site_config,
            time_range=time_range,
            num_processes=num_processes,
            filter_pcaps=filter_pcaps,
            **kwargs,
        )
    elif background == "gml":
        return get_gml_background(obs_times=obs_times, **kwargs)
    elif background == "ct_stilt":
        return get_ct_stilt_background(obs_times=obs_times, **kwargs)
    else:
        raise ValueError(f"Unsupported background: {background}")


def get_rolling_background(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    num_processes: int = 1,
    filter_pcaps: bool = True,
    baseline_window: str = "14d",
    min_periods: int = int(24 * 3.5),
) -> pd.Series:
    """Rolling 1st-percentile baseline applied to in-situ observations."""
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
        freq="1h",
        mobile_grid_res=0.02,
        stationary_min_percent=0.75,
        mobile_min_count=10,
    )
    data = data.rename(columns={"Time_UTC": "obs_time"})
    df = data.set_index(["obs_time", "site"])["CH4"].unstack(fill_value=None)

    bg_dict = {}
    for site in df.columns:
        bg_dict[site] = rolling_baseline(
            df[site],
            window=baseline_window,
            min_periods=min_periods,
        )

    bg_df = pd.DataFrame(bg_dict)
    background = bg_df.mean(axis=1)
    background.name = "concentration"
    background.index.name = "obs_time"
    return background


def get_gml_background(
    obs_times,
    specie: str = "ch4",
    site: str = "mbo",
    **kwargs,
) -> pd.Series:
    """Thoning curve fit to GML discrete sample data."""
    if "sample_type" not in kwargs:
        if site.lower() == "mbo":
            sample_type = "pfp"
        elif site.lower() == "uta":
            sample_type = "flask"
        else:
            raise ValueError(f"Unsupported site for GML background: {site}")
    else:
        sample_type = kwargs.pop("sample_type")
    gml = GMLDiscrete(specie=specie, site=site, sample_type=sample_type)

    background = gml.thoning_curve(smooth_time=obs_times, **kwargs)
    background.name = "concentration"
    background.index.name = "obs_time"
    return background / 1000  # convert from ppb to ppm


def get_ct_stilt_background(
    obs_times,
    csv_path,
    value_col: str = "ct_ch4_ppm",
    **kwargs,
) -> pd.Series:
    """CarbonTracker-STILT endpoint background [ppm], joined to obs by UTC date.

    Reads the daily CT-STILT background product (built by sampling the
    CarbonTracker-CH4 field at STILT trajectory endpoints -- see
    ``lair.noaa.CarbonTracker.background`` + ``stilt.Trajectories.endpoints``) and
    aligns it to ``obs_times`` by date. ``csv_path`` is passed explicitly (e.g. via
    ``background_kwargs``) so the package stays decoupled from any workspace layout.
    Obs whose date is absent from the product get NaN.
    """
    ct = pd.read_csv(csv_path)
    day = pd.to_datetime(ct["obs_time"], utc=True).dt.tz_localize(None).dt.normalize()
    daily = pd.Series(ct[value_col].to_numpy(), index=day)
    obs_times = pd.DatetimeIndex(obs_times)
    background = pd.Series(
        daily.reindex(obs_times.normalize()).to_numpy(),
        index=obs_times,
        name="concentration",
    )
    background.index.name = "obs_time"
    return background  # CSV is already ppm
