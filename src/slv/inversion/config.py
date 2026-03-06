from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from cartopy.io.img_tiles import GoogleTiles
from lair.geo import generate_regular_grid, write_rio_crs

from slv.domain import UTC_OFFSET, XMAX, XMIN, YMAX, YMIN
from slv.measurements.sites import load_site_config

# Default MDM component parameters
# Notes:
# - `std`: absolute standard deviation in ppm (can be float or dict[site][season])
DEFAULT_MDM_CONFIG = {
    "part": {"std": 0.00047, "correlated": False},
    "instr": {
        "std": {"UATAQ": 0.0033, "DAQ": 0.0033 * 3},
        "correlated": False,
    },  # per org error
    "aggr": {"std": 0.06, "scale": None, "interday": False},
    "bg": {"std": 0.011, "scale": "7d", "interday": True},
    "transport_wind": {  # per site per season error
        "std": {
            "bv": {"DJF": 0.0027, "JJA": 0.001, "MAM": 0.0011, "SON": 0.0021},
            "ed": {"DJF": 0.001, "JJA": 0.0002, "MAM": 0.0002, "SON": 0.0003},
            "hw": {"DJF": 0.0018, "JJA": 0.0006, "MAM": 0.0007, "SON": 0.0011},
            "rb": {"DJF": 0.0022, "JJA": 0.0004, "MAM": 0.0005, "SON": 0.0008},
            "ut": {"DJF": 0.0025, "JJA": 0.001, "MAM": 0.0011, "SON": 0.0015},
            "wbb": {"DJF": 0.0016, "JJA": 0.0004, "MAM": 0.0006, "SON": 0.0009},
            "zz": {"DJF": 0.0085, "JJA": 0.0022, "MAM": np.nan, "SON": 0.0042},
        },
        "scale": "2.8h",
        "interday": False,
    },
    "transport_pbl": {
        "std": 0.15 * 0.0514,
        "scale": "2.8h",
        "interday": False,
    },  # 15% of typical PBL enhancement (0.0514 ppm)
}


def get_mdm_comp_configs(config: dict) -> list[dict]:
    """Build MDM components list from config dict, merging with defaults."""
    merged_components = []
    for name, default_params in DEFAULT_MDM_CONFIG.items():
        params = {**default_params, **(config.get(name, {}))}
        merged_components.append({"name": name, **params})
    return merged_components


def build_location_site_map(
    simulation_ids: list[str],
    site_config: pd.DataFrame,
) -> dict[str, str]:
    """Build location mapper from STILT simulation IDs to site names.

    Parses simulation IDs (format: "lon_lat_height") and matches them to sites
    in the site_config based on coordinate proximity.

    Parameters
    ----------
    simulation_ids : list[str]
        List of STILT simulation IDs (e.g., "-111.847672_40.766189_35.0").
    site_config : pd.DataFrame
        Site configuration with latitude/longitude columns indexed by site name.

    Returns
    -------
    dict[str, str]
        Mapping from location ID to site name.
        Unmapped IDs are omitted from the result.

    Examples
    --------
    >>> sim_ids = ["-111.847672_40.766189_35.0", "-111.884505_40.902945_4.0"]
    >>> site_config = load_site_config()
    >>> mapper = build_location_site_map(sim_ids, site_config)
    >>> mapper["-111.847672_40.766189_35.0"]
    'wbb'
    """

    mapper = {}

    for sim_id in simulation_ids:
        # Parse simulation ID to extract coordinates
        try:
            parts = sim_id.split("_")
            if len(parts) != 4:
                continue

            location_id = "_".join(parts[1:])
            if location_id in mapper:
                continue

            lon_sim, lat_sim, z_sim = float(parts[1]), float(parts[2]), float(parts[3])
        except (ValueError, IndexError):
            continue

        matches = site_config[
            np.isclose(site_config["latitude"].astype(float), lat_sim)
            & np.isclose(site_config["longitude"].astype(float), lon_sim)
            & np.isclose(site_config["height_agl"].astype(float), z_sim)
        ]

        if len(matches) == 1:
            site_name = matches.index[0]
            mapper[location_id] = site_name
        else:
            # No match or multiple matches found; skip this simulation ID
            continue

    return mapper


@dataclass
class InversionConfig:
    # --- Space & Time ---
    tstart: pd.Timestamp | str = "2015-06-01"
    tend: pd.Timestamp | str = "2025-02-01"
    flux_freq: str = "MS"
    utc_offset: int = UTC_OFFSET

    # Local afternoon hours (12 PM to 4 PM)
    afternoon_hours_local: list[int] = field(
        default_factory=lambda: [12, 13, 14, 15, 16]
    )

    # Grid boundaries and resolution
    xmin: float = XMIN
    xmax: float = XMAX
    ymin: float = YMIN
    ymax: float = YMAX
    dx: float = 0.05
    dy: float = 0.05

    # --- Obs & Background ---
    sites: list[str] = field(default_factory=lambda: ["wbb"])
    filter_pcaps: bool = True

    bg_baseline_window: str = "14d"
    bg_min_periods: int = int(24 * 3.5)  # 3.5 days worth of hourly obs

    aggregate_obs: bool | str = (
        False  # Whether to aggregate obs space ('1d' for daily, '12h' for 12-hourly, etc.)
    )

    # Maps STILT simulation IDs (format: "lon_lat_height") to site names.
    # If empty (default), auto-generated from site_config and simulation paths.
    location_site_map: dict[str, str] = field(default_factory=dict)

    # --- Prior ---
    prior: str = "epa"
    prior_kwargs: dict = field(default_factory=dict)

    # --- Jacobian ---
    stilt_path: str = (
        "/uufs/chpc.utah.edu/common/home/u6036966/wkspace/methane/SLV/stilt/wbb"
    )
    sparse_jacobian: bool = True

    # --- Prior Error Covariance (S_0) ---
    prior_base_std: float = 0.019
    prior_std_frac: float = 0.5
    prior_time_scale: str = "32d"
    prior_spatial_scale: float = 5.0

    # --- Model-Data Mismatch (S_z) ---
    mdm_config: dict = field(default_factory=dict)

    @cached_property
    def mdm_components(self) -> list[dict]:
        """Build MDM components from config (merges with defaults)."""
        return get_mdm_comp_configs(self.mdm_config)

    # --- Bias ---
    # Set bias_std to enable the bias block.
    bias_std: float | None = None  # Prior std-dev for each bias state (None = disabled)
    # Bias grouping strategy:
    #   None or "time": one bias per time interval
    #   "site": one bias per (time, obs_location)
    #   "site_group": one bias per (time, organization)
    bias_grouping: str | None = None

    # --- Inversion Solver Settings ---
    min_obs_per_interval: int = 60
    min_sims_per_interval: int = 70

    # Regularization parameter: scales observation error by 1/gamma.
    # gamma > 1: reduces obs error weight, forces solution toward data
    # gamma < 1: increases obs error weight, stays closer to prior
    # gamma = 1: no scaling (default)
    gamma: float | None = None

    # --- Cache ---
    # False/None = no caching, True = cache in cwd, str/Path = cache in that directory
    cache: bool | str | Path = False
    # Recompute cached outputs by stem (e.g. ["obs", "prior_error"]).
    # Set to "all" to recompute every cached output.
    cache_overwrite: str | list[str] = field(default_factory=list)

    # --- Compute ---
    num_processes: int = 8
    timeout: int = 100  # seconds (to avoid hanging processes; avg process is ~2s)

    # --- Plotting ---
    plot_inputs: bool = True
    plot_results: bool = True
    plot_diagnostics: bool = False

    output_units: str = "Gg/m2/s"

    tiler: GoogleTiles = field(
        default_factory=lambda: GoogleTiles(style="satellite", cache=True)
    )
    tiler_zoom: int = 10

    @property
    def bbox(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def extent(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def map_extent(self):
        # Add a buffer around the bbox for better visualization
        buffer = 0.05
        return (
            self.xmin - buffer,
            self.xmax + buffer,
            self.ymin - buffer,
            self.ymax + buffer,
        )

    @property
    def resolution(self) -> str:
        return f"{self.dx}x{self.dy}"

    @cached_property
    def grid(self):
        # Create a regular grid with specified bounds and resolution
        grid = generate_regular_grid(
            xmin=self.xmin,
            xmax=self.xmax,
            dx=self.dx,
            ymin=self.ymin,
            ymax=self.ymax,
            dy=self.dy,
            x_label="lon",
            y_label="lat",
        )
        grid = grid.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        grid = write_rio_crs(grid, crs="EPSG:4326")
        return grid

    @cached_property
    def grid_coords(self):
        return pd.MultiIndex.from_product(
            [self.grid["lon"].values, self.grid["lat"].values]
        ).to_list()

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (pd.Timestamp(self.tstart), pd.Timestamp(self.tend))

    @property
    def flux_time_bins(self):
        """Generates time bins for flux estimation based on the time range and flux frequency."""
        t0, t1 = self.time_range
        return pd.interval_range(start=t0, end=t1, freq=self.flux_freq, closed="left")

    @property
    def flux_times(self) -> pd.DatetimeIndex:
        """Returns the left edges of the flux time bins, which represent the time points for flux estimation."""
        return self.flux_time_bins.left

    @property
    def subset_hours_utc(self) -> list[float]:
        """Dynamically converts local afternoon hours to UTC for data subsetting."""
        return [(hour - self.utc_offset) % 24 for hour in self.afternoon_hours_local]

    @cached_property
    def site_config(self):
        return load_site_config()
