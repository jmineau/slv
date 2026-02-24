import pandas as pd
import uataq

from slv.measurements.daq import load_picarro_g2307
from slv.measurements.daq import sites as daq_sites
from slv.meteorology.pcaps import filter_pcap_events


def load_site_data(
    sites, time_range, num_processes=1, filter_pcaps=True
) -> pd.DataFrame:
    """Core function to fetch and filter raw site data (Shared by Obs and Background)."""
    pollutant_col = "CH4d_ppm_cal"
    valid_flags = [2, 1, 0, -64, -140]

    data_dict = {}

    if "wbb" in sites:
        wbb = uataq.get_obs(
            "wbb", "ch4", time_range=time_range, num_processes=num_processes
        )[pollutant_col]
        data_dict["wbb"] = wbb

    for site in sites:
        if site in daq_sites:
            daq_df = load_picarro_g2307(site, lvl="calibrated")
            conditions = (
                daq_df[pollutant_col].notnull()
                & daq_df.QAQC_Flag.isin(valid_flags)
                & (daq_df["ID"] == -10)
            )
            daq_df = (
                daq_df[conditions].set_index("Time_UTC").sort_index().loc[time_range]
            )
            data_dict[site] = daq_df[pollutant_col]

    data = pd.DataFrame(data_dict)

    if filter_pcaps:
        data = filter_pcap_events(data)

    return data


def get_slv_observations(
    sites: list[str],
    time_range: tuple,
    num_processes: int = 1,
    filter_pcaps: bool = True,
) -> pd.Series:
    """Fetches observations for the pipeline."""
    df = load_site_data(
        sites=sites,
        time_range=time_range,
        num_processes=num_processes,
        filter_pcaps=filter_pcaps,
    )
    # Stack into a single multi-index series (site, time)
    obs_series = df.unstack().dropna()
    obs_series.index.names = ["obs_location", "obs_time"]
    return obs_series
