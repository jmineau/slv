from importlib.resources import files

import pandas as pd


def load_site_config() -> pd.DataFrame:
    """Loads the internal site_config.csv into a Pandas DataFrame."""
    
    # Locate the file dynamically within the installed package
    csv_path = files(__package__).joinpath("site_config.csv")
    
    # Read the file using standard Pandas
    with csv_path.open("r") as f:
        site_df = pd.read_csv(f, index_col="stid")
        
    return site_df


def get_site_coordinates(site_id: str) -> tuple[float, float]:
    """Convenience function to pull specific coordinates."""
    df = load_site_config()
    
    if site_id not in df.index:
        raise ValueError(f"Site {site_id} not found in site_config.csv")
        
    row = df.loc[site_id]
    return row["latitude"], row["longitude"]