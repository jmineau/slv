from pathlib import Path

import pandas as pd
import uataq
from uataq.timerange import TimeRange

from slv import get_data_dir
from slv.measurements import instruments
from slv.measurements.mobile import merge_with_gps
from slv.measurements.pollutants import normalize_pollutant
from slv.measurements.sites import load_site_config
from slv.meteorology.pcaps import filter_pcap_events


def load_concentrations(
    pollutants: list[str] | str,
    orgs: list[str] | str | None = None,
    sites: list[str] | str | None = None,
    time_range: TimeRange | tuple | None = None,
    site_config: pd.DataFrame | None = None,
    include_location: bool = False,
    valid_range: dict[str, tuple[float, float]] | None = None,
    valid_flags: dict[str, set] | None = None,
    subset_hours: list[int] | None = None,
    filter_pcaps: bool = False,
    num_processes: int = 1,
    mobile_kwargs: dict | None = None,
) -> pd.DataFrame:
    if site_config is None:
        site_config = load_site_config()

    if isinstance(pollutants, str):
        pollutants = [pollutants]

    if isinstance(orgs, str):
        orgs = [orgs]

    if sites is None:
        if orgs is None:
            raise ValueError("Must provide at least one of orgs or sites.")
        else:
            sites = site_config[site_config["organization"].isin(orgs)].index.tolist()
    else:
        if isinstance(sites, str):
            sites = [sites]
        sites = [s.lower() for s in sites]

    if isinstance(pollutants, str):
        pollutants = [pollutants]

    time_range = TimeRange(time_range)

    # Initial loop through sites to determine if any are mobile
    has_mobile = False
    for s in sites:
        if s in site_config.index and site_config.at[s, "type"] == "mobile":
            has_mobile = True

            # force include_location to True if any site is mobile
            include_location = True

    site_dfs: list[pd.DataFrame] = []
    for site in sites:
        try:
            config = site_config.loc[site]
            if isinstance(config, pd.DataFrame):
                config = config.iloc[0]
        except KeyError:
            print(f"Site {site} not found in site_config. Skipping.")
            continue

        org = config["organization"]
        site_type = config["type"]
        instrument_list = config["instruments"].split()

        inst_dfs: list[pd.DataFrame] = []
        for instr_name in instrument_list:
            # Find instrument class
            instr_class = instruments.REGISTRY.get(instr_name)
            if instr_class is None:
                print(f"Instrument {instr_name} not recognized for site {site}.")
                continue

            # Check if instrument supports any of the requested pollutants
            supported_pollutants = [
                pol for pol in pollutants if pol in instr_class.pollutants
            ]
            if not supported_pollutants:
                print(
                    f"Instrument {instr_name} does not support any requested pollutants for site {site}."
                )
                continue

            # Load data based on organization and instrument
            lvl = "calibrated" if getattr(instr_class, "calibrated", True) else "qaqc"

            # --- UATAQ ---
            if org == "UATAQ":
                try:
                    df = uataq.read_data(
                        site,
                        instruments=instr_name,
                        lvl=lvl,
                        time_range=time_range,
                        num_processes=num_processes,
                    )[instr_name].reset_index()
                except Exception as e:
                    print(f"Failed to load {instr_name} for {site}: {e}")
                    continue

            # -- DAQ with Picarro G2307 ---
            elif org == "DAQ" and instr_name == "picarro_g2307":
                data_dir = Path(get_data_dir("SLV_DAQ_DIR"))
                pattern = f"{site}/picarro_g2307/{lvl}/*.dat"
                files = list(data_dir.rglob(pattern))
                if not files:
                    print(
                        f"No DAQ files found for {site}. Skipping instrument {instr_name}."
                    )
                    continue
                df = pd.concat(
                    [pd.read_csv(f, parse_dates=["Time_UTC"]) for f in files]
                )
                if time_range is not None:
                    df = df.set_index("Time_UTC").sort_index()
                    df = df.loc[time_range.start : time_range.stop].reset_index()

                # ID column is built from CH4
                df = df.rename(
                    columns={
                        "ID": "ID_CH4",
                    }
                )
                df["ID_H2CO"] = df["ID_CH4"]

            else:
                print(
                    f"Unknown org/instrument combo for site {site}: {org}/{instr_name}"
                )
                continue

            if df.empty:
                print(f"No data for {site} instrument {instr_name}. Skipping.")
                continue

            df["Time_UTC"] = pd.to_datetime(df["Time_UTC"])
            df["instrument"] = instr_name

            # Normalize columns for each supported pollutant
            for pol in supported_pollutants:
                pol_range = valid_range.get(pol) if valid_range else None
                pol_flags = valid_flags.get(pol) if valid_flags else None
                conc_col = instr_class.pollutants[pol]
                if conc_col in df.columns:
                    df = df.rename(columns={conc_col: pol})
                df[pol] = normalize_pollutant(df, pol, pol_range, pol_flags)

            # Drop rows where all pollutants are NaN
            df = df.dropna(subset=supported_pollutants, how="all")

            inst_dfs.append(df)

        if inst_dfs:
            # Combine instruments for this site
            site_obs = pd.concat(inst_dfs, ignore_index=True)

            if site_type == "mobile":
                # For mobile sites, attempt to merge with GPS data
                site_obs = merge_with_gps(
                    site=site,
                    org=org,
                    obs=site_obs,
                    time_range=time_range,
                    num_processes=num_processes,
                    **mobile_kwargs if mobile_kwargs is not None else {},
                ).rename(
                    columns={
                        "Latitude_deg": "latitude",
                        "Longitude_deg": "longitude",
                        "Altitude_msl": "altitude",
                    }
                )

            elif include_location:
                # For stationary sites, add location from config
                site_obs["latitude"] = config["latitude"]
                site_obs["longitude"] = config["longitude"]

            if site_type == "mobile" or include_location:
                # Add height above ground column
                site_obs["height"] = config["height_agl"]

            if has_mobile:
                # Add is_mobile column
                site_obs["is_mobile"] = site_type == "mobile"

            site_obs["site"] = site
            site_obs["org"] = org

            site_dfs.append(site_obs)

    if not site_dfs:
        raise ValueError("No data loaded for any requested sites/instruments.")

    # Combine all sites into single DataFrame
    obs = pd.concat(site_dfs, ignore_index=True)
    obs = obs.sort_values(["Time_UTC", "site"]).reset_index(drop=True)

    # Calculate mountain standard time
    # TODO this is hardcoded, any easy way to get Local Standard Time offset for each site?
    obs["Time_MST"] = obs.Time_UTC - pd.to_timedelta("7h")

    if subset_hours is not None:
        # Filter to specified hours of day (MOUNTAIN STANDARD TIME)
        obs = obs[obs["Time_MST"].dt.hour.isin(subset_hours)]

    if filter_pcaps and not obs.empty:
        # Filter out PCAP events
        obs = filter_pcap_events(obs.set_index("Time_UTC")).reset_index()

    # Keep only requested columns
    column_order = (
        ["Time_UTC", "Time_MST", "site", "org", "instrument"]
        + pollutants
        + ["is_mobile", "latitude", "longitude", "height", "altitude"]
    )
    columns_to_keep = [c for c in column_order if c in obs.columns]
    obs = obs[columns_to_keep]

    # Force cols to numeric if possible
    for col in obs.columns:
        if col not in [
            "Time_UTC",
            "Time_MST",
            "site",
            "org",
            "instrument",
            "is_mobile",
        ]:
            try:
                obs[col] = pd.to_numeric(obs[col], errors="raise")
            except Exception as e:
                print(f"Error converting column {col} to numeric: {e}")

    return obs
