import numpy as np
import pandas as pd

defaults = {
    "CH4": {
        "valid_range": (1.70, 300),
        "valid_flags": {-64, -140},
    },
    "CO2": {
        "valid_range": (350, 5000),
        "valid_flags": {-64, -140},
    },
}


def normalize_pollutant(
    df: pd.DataFrame,
    pollutant: str,
    valid_range: tuple[float, float] | None = None,
    valid_flags: set | None = None,
) -> pd.Series:
    """Extract and filter a pollutant column by QA/QC and range criteria.
    Looks for column named {pollutant}d_ppm_cal or {pollutant}_ppm.
    Applies ID check, QAQC_Flag check, and valid range filtering.
    Returns Series with NaN for invalid values.
    """
    if valid_flags is None:
        base_flags = {0, 1, 2}
        pol_flags = defaults.get(pollutant, {}).get("valid_flags", set())
        valid_flags = base_flags.union(pol_flags)

    if valid_range is None:
        valid_range = defaults.get(pollutant, {}).get("valid_range", None)

    if pollutant not in df.columns:
        raise ValueError(f"Pollutant column {pollutant} not found in DataFrame.")

    values = df[pollutant].copy()
    valid = pd.Series(True, index=df.index)

    id_col = f"ID_{pollutant}"
    if id_col in df.columns:
        valid &= df[id_col] == -10

    if "QAQC_Flag" in df.columns:
        valid &= df["QAQC_Flag"].isin(valid_flags)

    if valid_range is not None:
        vmin, vmax = valid_range
        valid &= (values >= vmin) & (values <= vmax)

    values[~valid] = np.nan

    return values
