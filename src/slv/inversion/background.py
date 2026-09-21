import pandas as pd
from lair.background import rolling_baseline

from slv.inversion.covariances import normalize_duration
from slv.inversion.data import split_sites
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

    Returns Series with obs_time index and background concentration. The hourly rolling
    baseline is looked up for the hour each obs time falls in, so a mobile receptor released
    at 20:13 gets the 20:00 background (an exact-time join would leave it NaN and drop it).
    """
    if background == "rolling":
        hourly = get_rolling_background(
            sites=sites,
            site_config=site_config,
            time_range=time_range,
            num_processes=num_processes,
            filter_pcaps=filter_pcaps,
            **kwargs,
        )
        obs_times = pd.DatetimeIndex(obs_times)
        at_obs = pd.Series(
            hourly.reindex(obs_times.floor("h")).to_numpy(),
            index=obs_times,
            name="concentration",
        )
        at_obs.index.name = "obs_time"
        return at_obs
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
    background_sites: list[str] | None = None,
) -> pd.Series:
    """Hourly rolling 1st-percentile baseline, averaged over the stationary sites.

    The baseline comes from the towers: mobile sites in ``sites`` are left out (a train
    sampling the urban core is not a background site, and its per-grid-point rows do not
    form one hourly series). ``background_sites`` picks the sites explicitly instead, e.g.
    towers for a TRAX-only inversion (``background_kwargs={"background_sites": [...]}``).
    """
    if background_sites is None:
        background_sites, _ = split_sites(sites, site_config)
        if not background_sites:
            raise ValueError(
                f"No stationary sites in {sites} for the rolling background; set "
                'background_kwargs={"background_sites": [...]} to choose baseline sites.'
            )
    data = load_concentrations(
        pollutants=["CH4"],
        sites=list(background_sites),
        site_config=site_config,
        time_range=time_range,
        num_processes=num_processes,
        filter_pcaps=filter_pcaps,
    )
    data = aggregate_obs(data, freq="1h", stationary_min_percent=0.75)
    data = data.rename(columns={"Time_UTC": "obs_time"})
    df = data.set_index(["obs_time", "site"])["CH4"].unstack(fill_value=None)

    bg_dict = {}
    for site in df.columns:
        bg_dict[site] = rolling_baseline(
            df[site],
            window=normalize_duration(baseline_window),
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
