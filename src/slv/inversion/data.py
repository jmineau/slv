import time

import pandas as pd
import uataq

from slv.measurements.daq import load_picarro_g2307
from slv.meteorology.pcaps import filter_pcap_events


def load_concentrations(
    sites: list[str],
    time_range: tuple,
    site_config: pd.DataFrame,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    num_processes: int = 1,
) -> pd.Series:
    """Loads and concatenates concentration data for the specified sites and time range."""
    dfs = {}
    for site in sites:
        try:
            config = site_config.loc[site]
        except KeyError:
            print(f"Site {site} not found in site_config. Skipping.")
            continue

        lvl = "calibrated"

        if config["organization"] == "UATAQ":
            print(f"Loading UATAQ data for {site}...")
            instrument = "lgr_ugga"
            df = uataq.read_data(
                site,
                instruments=instrument,
                lvl=lvl,
                time_range=time_range,
                num_processes=num_processes,
            )[instrument]
            df["org"] = "UATAQ"
            dfs[site] = df.reset_index()
        elif config["organization"] == "DAQ":
            print(f"Loading DAQ data for {site}...")
            df = load_picarro_g2307(
                site=site,
                lvl=lvl,
                time_range=time_range,
            )
            df = df.rename(columns={"ID": "ID_CH4"})
            df["org"] = "DAQ"
            dfs[site] = df
        else:
            raise ValueError(
                f"Unknown organization for site {site}: {config['organization']}."
            )

    # Stack into a single DataFrame with a multi-index (site, time)
    data = pd.concat(
        (df.assign(site=site) for site, df in dfs.items()), ignore_index=True
    )
    data = data.set_index(["site", "Time_UTC"]).sort_index()

    # Reduce data to columns of interest
    pollutant_col = "CH4d_ppm_cal"
    id_col = "ID_CH4"
    columns_of_interest = [
        "org",
        pollutant_col,
        id_col,
        "QAQC_Flag",
    ]
    data = data[columns_of_interest]

    # Filter to valid observations
    valid_flags = [
        2,  # filled from backup
        1,  # manually passed
        0,  # auto passed
        -64,  # cavity temperature out of range (5, 45)
        -140,  # formaldehyde out of range
    ]

    conditions = (
        data[pollutant_col].notnull()
        & data.QAQC_Flag.isin(valid_flags)
        & (data[id_col] == -10)
    )
    data = data[conditions]
    data = data.rename(columns={pollutant_col: "concentration"})

    # Hourly Observations
    groupers = ["org", "site", data.index.get_level_values("Time_UTC").floor("h")]
    hourly = (
        data.reset_index()
        .groupby(groupers)["concentration"]
        .agg(["mean", "size"])
        .reset_index(level="org")
    )
    hourly = hourly.rename(columns={"mean": "concentration", "size": "count"})

    # Filter to hours with sufficient data
    group_min_count = {
        "DAQ": 38,
        "UATAQ": 232,
    }
    hourly["min_count"] = hourly["org"].map(group_min_count).values
    hourly = hourly[hourly["count"] >= hourly["min_count"]]
    hourly = hourly.drop(columns=["org", "count", "min_count"])

    # Set multi-index
    obs = hourly.sort_index()
    obs.index.names = ["obs_location", "obs_time"]
    # Subset hours if specified
    if subset_hours is not None:
        obs = obs[obs.index.get_level_values("obs_time").hour.isin(subset_hours)]

    if filter_pcaps:
        print("Filtering PCAP events from observations...")
        step_start = time.perf_counter()
        obs = filter_pcap_events(obs, level="obs_time")
        print(f"PCAP events filtered in {time.perf_counter() - step_start:.2f}s")

    return obs["concentration"]


def get_slv_observations(
    sites: list[str],
    site_config: pd.DataFrame,
    time_range: tuple,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = True,
    num_processes: int = 1,
) -> pd.Series:
    """Fetches observations for the pipeline."""
    obs = load_concentrations(
        sites=sites,
        time_range=time_range,
        site_config=site_config,
        subset_hours=subset_hours,
        filter_pcaps=filter_pcaps,
        num_processes=num_processes,
    )
    return obs
