from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd
from cartopy.io.img_tiles import GoogleTiles
from lair.geo import generate_regular_grid, write_rio_crs

from slv.domain import UTC_OFFSET, XMAX, XMIN, YMAX, YMIN
from slv.measurements.sites import load_site_config


@dataclass
class MDMComponent:
    name: str
    std: float
    correlated: bool = True
    scale: str | float | None = None
    interday: bool = False


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

    location_site_map: dict[str, str] = field(
        default_factory=lambda: {"-111.847672_40.766189_35.0": "wbb"}
    )

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
    mdm_components: list[MDMComponent] = field(
        default_factory=lambda: [
            MDMComponent(name="part", std=0.00047, correlated=False),
            MDMComponent(name="instr_wbb", std=0.0033, correlated=False),
            MDMComponent(name="aggr", std=0.06, scale=None, interday=False),
            MDMComponent(name="bg", std=0.011, scale="7d", interday=True),
            MDMComponent(name="transport_wind", std=0.01, scale="2.8h", interday=False),
            MDMComponent(name="transport_pbl", std=0.15, scale="2.8h", interday=False),
        ]
    )

    # --- Bias ---
    # Set bias_std to enable the bias block.  The default get_bias() builds one
    # bias state per flux_time interval (all initialised to 0.0).
    # Override SLVMethaneInversionWithBias.get_bias() for a custom index.
    bias_std: float | None = None  # Prior std-dev for each bias state (None = disabled)
    bias_jacobian: pd.DataFrame | float = (
        1.0  # Jacobian mapping bias states → concentrations
    )

    # --- Inversion Solver Settings ---
    min_obs_per_interval: int = 60
    min_sims_per_interval: int = 70

    # --- Cache ---
    # False/None = no caching, True = cache in cwd, str/Path = cache in that directory
    cache: bool | str | Path = False

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
