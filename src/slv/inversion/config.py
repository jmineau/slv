"""Configuration for an SLV methane inversion (:class:`InversionConfig`)."""

import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io.img_tiles import GoogleTiles
from lair.geo import write_rio_crs

from slv.domain import UTC_OFFSET, XMAX, XMIN, YMAX, YMIN
from slv.measurements.sites import load_site_config

# CHPC location of the production PYSTILT project (fallback when SLV_STILT_DIR is unset).
DEFAULT_STILT_PROJECT = (
    "/uufs/chpc.utah.edu/common/home/lin-group27/jkm/stilt/simulations/stilt"
)


#: Supported ``flux_freq`` values and the lair inventory time step each maps to (the
#: domain totals in ``summarize`` need one).
FLUX_FREQ_TIME_STEPS = {
    "YS": "annual",
    "QS": "quarterly",
    "MS": "monthly",
    "2W": "biweekly",
    "D": "daily",
}
BIAS_GROUPINGS = (None, "time", "site", "site_group")
BACKGROUNDS = ("rolling", "gml", "ct_stilt")
PRIORS = ("epa", "edgar", "constant")


def stilt_project_dir() -> Path:
    """The production PYSTILT project: ``$SLV_STILT_DIR`` if set, else
    :data:`DEFAULT_STILT_PROJECT`. Use this in scripts instead of spelling the path out;
    ``stilt_project_dir() / "simulations" / "by-id"`` is the per-receptor tree."""
    return Path(os.environ.get("SLV_STILT_DIR", DEFAULT_STILT_PROJECT))


# Default MDM component parameters
# Notes:
# - `std`: absolute standard deviation in ppm (can be float or dict[site][season])
# - Footprint-dependent terms recomputed 2026-06 on the full-hour PYSTILT sims
#   (see inversions/mdm/): `part` = finite-particle SEM; `transport_wind` =
#   median var(error)-var(regular). `aggr` right-sized from the inflated 60 ppb
#   placeholder to the measured TRAX per-pass spatial std (mean, Mallia 2020).
#   `instr`/`bg`/`transport_pbl` unchanged (no footprints).
DEFAULT_MDM_CONFIG = {
    "part": {
        "std": 0.000765,
        "correlated": False,
    },  # finite-particle SEM, full footprints
    "instr": {
        "std": {"UATAQ": 0.0033, "DAQ": 0.0033 * 3},
        "correlated": False,
    },  # per org error
    "aggr": {
        "std": 0.0257,
        "scale": None,
        "interday": False,
    },  # spatial representativeness: TRAX per-pass spatial std (mean)
    # Temporal representativeness: sigma = fraction * (per-obs within-hour CH4 std). The hour-mean
    # footprint cannot represent sub-hour structure, so each obs is down-weighted by how unsteady
    # it actually was -- a passing-plume day gets a large error instead of being dropped. Replaces
    # the binary spike filter (top-10% within-hour-variance drop), which removed those days' real
    # emission signal and biased the total low (see diagnostics/obs_qc/spike_filter_cost.py); this
    # keeps them, weighted. The temporal counterpart to aggr (spatial). fraction=2.0 calibrated
    # to reduced chi^2 ~ 1 with filter_spikes=False (headline 36 dropping them -> 37.6 keeping
    # them weighted; see diagnostics/obs_qc/subhour_calibration.py). The ~2x factor absorbs the
    # daily aggregation: the term is built per-hour then averaged to daily, which shrinks it.
    "subhour": {
        "per_obs": "subhour",
        "fraction": 2.0,
        "correlated": False,
    },
    "bg": {"std": 0.011, "scale": "7d", "interday": True},
    # Multiplicative transport + representation error: sigma = fraction * |H| * mean(x_prior)
    # (footprint strength in enhancement units). Extreme shallow-PBL footprint days -- which
    # STILT models least reliably and which over-leverage the winter months under a flat error
    # -- get a proportionally large error, regardless of what the obs did (no circularity) or
    # the EPA spatial pattern. Scaling on the observed enhancement instead trusts the low-obs
    # winter dips (the strong footprint sees little) and biases the total low; scaling on the
    # prior under-weights exactly the low-EPA-cell footprints that over-leverage. Supersedes
    # the additive transport_wind + transport_pbl terms (recoverable from git history).
    # fraction calibrated to reduced chi^2 ~ 1 on the WBB inversion (f=0.80; see
    # diagnostics/transport_error/footprint_scale_sweep.py).
    "transport": {
        "fraction": 0.8,
        "multiplicative": True,
        "scale_on": "footprint",  # |H|*mean(x_prior); also "obs" (|obs-bg|) or "prior" (|H x_prior|)
        "scale": "2.8h",
        "interday": False,
    },
}


def get_mdm_comp_configs(config: dict) -> list[dict]:
    """Build MDM components list from config dict, merging with defaults.

    Raises ``ValueError`` for a component name that is not in
    :data:`DEFAULT_MDM_CONFIG`: a misspelled or retired key (e.g. the old
    ``transport_pbl`` / ``transport_wind``) would otherwise be ignored silently, and
    the MDM cache key would not change either.
    """
    unknown = set(config) - set(DEFAULT_MDM_CONFIG)
    if unknown:
        raise ValueError(
            f"Unknown MDM component(s) {sorted(unknown)}; "
            f"valid names: {sorted(DEFAULT_MDM_CONFIG)}"
        )
    merged_components = []
    for name, default_params in DEFAULT_MDM_CONFIG.items():
        params = {**default_params, **(config.get(name, {}))}
        merged_components.append({"name": name, **params})
    return merged_components


def build_location_site_map(
    location_ids: list[str],
    site_config: pd.DataFrame,
) -> dict[str, str]:
    """Build location mapper from STILT location IDs to site names.

    Parses location IDs (format: "lon_lat_height") and matches them to sites
    in the site_config: within 1e-5 deg (~1 m) in lat/lon and 0.5 m in height.
    (``np.isclose``'s default relative tolerance allowed ~0.0011 deg of longitude,
    ~95 m.)

    Parameters
    ----------
    location_ids : list[str]
        List of STILT location IDs (e.g., "-111.847672_40.766189_35").
    site_config : pd.DataFrame
        Site configuration with latitude/longitude columns indexed by site name.

    Returns
    -------
    dict[str, str]
        Mapping from location ID to site name.
        Unmapped IDs are omitted from the result.

    Examples
    --------
    >>> loc_ids = ["-111.847672_40.766189_35", "-111.884505_40.902945_4"]
    >>> site_config = load_site_config()
    >>> mapper = build_location_site_map(loc_ids, site_config)
    >>> mapper["-111.847672_40.766189_35"]
    'wbb'
    """
    mapper = {}

    for location_id in set(location_ids):
        if location_id in mapper:
            continue

        parts = location_id.split("_")
        if len(parts) != 3:
            continue

        try:
            lon, lat, z = float(parts[0]), float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            continue

        matches = site_config[
            np.isclose(site_config["latitude"].astype(float), lat, rtol=0, atol=1e-5)
            & np.isclose(site_config["longitude"].astype(float), lon, rtol=0, atol=1e-5)
            & np.isclose(site_config["height_agl"].astype(float), z, rtol=0, atol=0.5)
        ]

        if len(matches) == 1:
            mapper[location_id] = matches.index[0]

    return mapper


@dataclass
class InversionConfig:
    """Settings for one :class:`~slv.inversion.pipelines.SLVMethaneInversion` run.

    Invalid choices (``flux_freq``, ``background``, ``prior``, ``bias_grouping``, MDM
    component names, an empty domain) raise ``ValueError`` at construction. Cached
    components are keyed on the fields each one depends on (see
    :mod:`slv.inversion.cache`), so changing a field's default spelling forces a rebuild.

    Parameters
    ----------
    tstart, tend : str or pd.Timestamp
        Inversion period; flux intervals start at ``tstart`` and the last one ends at
        ``tend``.
    flux_freq : str
        Flux interval: ``"YS"``, ``"QS"``, ``"MS"``, ``"2W"`` or ``"D"``.
    utc_offset : int
        Hours from UTC to local standard time, for ``subset_hours``.
    subset_hours : list of int
        Local hours of obs to keep (default 12-16, a well-mixed afternoon boundary layer).
    xmin, xmax, ymin, ymax : float
        Domain in degrees (default :mod:`slv.domain`). Snapped outward to whole cells,
        see :attr:`state_grid`.
    dx, dy : float
        Cell size in degrees.
    sites : list of str
        Sites from the site config. A mobile site (e.g. ``"trx01"``) takes its obs per
        TRAX receptor.
    mobile_obs : str or Path, optional
        Receptor-paired obs for the mobile sites (parquet indexed
        ``(obs_location, obs_time)``, from
        :func:`~slv.measurements.mobile.trax_receptor_observations`).
    filter_pcaps : bool
        Drop obs during persistent cold-air pool events.
    filter_spikes : bool
        Drop days whose within-hour CH4 std is above ``spike_percentile``. Superseded by
        the ``subhour`` MDM term, which keeps those days and down-weights them.
    spike_percentile : float
        Percentile threshold for ``filter_spikes``.
    background : str
        ``"rolling"`` (baseline from the stationary sites), ``"gml"`` or ``"ct_stilt"``.
    background_kwargs : dict
        Passed to the background loader (e.g. ``csv_path`` for ``ct_stilt``).
    aggregate_obs : bool or str
        ``False``, or a frequency (e.g. ``"1D"``) to average obs and Jacobian rows to.
    location_site_map : dict
        STILT location id -> site. Empty: matched from the site config coordinates.
    prior : str
        ``"epa"``, ``"edgar"`` or ``"constant"``, see
        :func:`~slv.inversion.priors.get_slv_prior`.
    prior_kwargs : dict
        Passed to the prior loader (e.g. ``{"express": True}``).
    stilt_project : str or Path
        PYSTILT project holding the footprints (``$SLV_STILT_DIR``, else
        :data:`DEFAULT_STILT_PROJECT`).
    footprint : str, optional
        Footprint config name or hash; ``None`` takes the finest in the project. The cache
        key sees this value, not what ``None`` resolved to, so set it explicitly.
    sparse_jacobian : bool
        Keep the Jacobian sparse.
    prior_base_std, prior_std_frac : float
        Prior error std per cell, ``prior_base_std + prior_std_frac * prior`` (umol/m2/s).
    prior_time_scale : str
        e-folding time of the prior error correlation (e.g. ``"32d"``).
    prior_spatial_scale : float
        e-folding distance of the prior error correlation, km.
    mdm_config : dict
        Overrides of :data:`DEFAULT_MDM_CONFIG`, by component name.
    bias_std : float, optional
        Prior std of the bias states (ppm); ``None`` leaves the bias block out.
    bias_grouping : str, optional
        ``None`` / ``"time"`` (one bias per interval), ``"site"`` or ``"site_group"``
        (per interval and site or organization).
    jacobian_coverage_percentile : float, optional
        Hold the least-constrained cells (below this percentile of mean Jacobian
        sensitivity) at the prior. ``None`` keeps every cell.
    min_obs_per_interval : int
        Drop flux intervals with fewer obs.
    min_sims_per_interval : int
        Read by fips but not applied.
    gamma : float, optional
        Divides the obs error (``> 1`` fits the data more closely).
    cache : bool, str or Path
        ``False``: no cache; ``True``: the working directory; else that directory.
    cache_overwrite : str or list of str
        Components to rebuild (e.g. ``["obs"]``), or ``"all"``.
    num_processes : int
        Workers for obs loading and the Jacobian build.
    timeout : int
        Per-task timeout, s, of the parallel Jacobian build.
    plot_inputs, plot_results, plot_diagnostics : bool
        Which plots :meth:`~slv.inversion.pipelines.SLVMethaneInversion.run` draws.
    output_units : str
        Flux units for domain totals (e.g. ``"Gg/m2/s"`` gives Gg per interval).
    tiler, tiler_zoom
        Map tiles for the plots.
    """

    # --- Space & Time ---
    tstart: pd.Timestamp | str = "2015-06-01"
    tend: pd.Timestamp | str = "2025-02-01"
    flux_freq: str = "MS"
    utc_offset: int = UTC_OFFSET

    # Hours of day (local time) to subset obs for inversion (e.g., afternoon hours when boundary layer is typically more developed).
    subset_hours: list[int] = field(
        default_factory=lambda: [
            12,
            13,
            14,
            15,
            16,
        ]  # Local afternoon hours (12 PM to 4 PM)
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
    # Prebuilt receptor-paired observations for the mobile sites (a parquet indexed
    # (obs_location, obs_time) with a CH4 column, from
    # slv.measurements.mobile.trax_receptor_observations). When set, mobile sites take
    # their obs from here instead of the hourly point aggregation, so each obs is keyed
    # exactly like the TRAX receptor it pairs with. None keeps the old mobile path.
    mobile_obs: str | Path | None = None
    filter_pcaps: bool = True

    # Drop days with anomalously high within-hour CH4 variance: passing plumes /
    # non-steady conditions that a footprint (which convolves the period mean)
    # cannot faithfully represent. Off by default; enabled for the WBB inversion
    # (see inversions/wbb/_shared.py). Takes reduced chi^2 from ~2 to ~1.2 there.
    filter_spikes: bool = False
    spike_percentile: float = 0.90  # within-hour-std percentile flagged as spiky

    background: str = "rolling"  # "rolling", "gml"
    background_kwargs: dict = field(default_factory=dict)

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
    # PYSTILT project(s) holding the footprints. Default: the production project; override
    # per-machine with the SLV_STILT_DIR env var (set in ~/.env alongside the other
    # SLV_*_DIR vars) rather than editing this. A list combines projects -- e.g. the
    # production project (UOU, DAQ) and the TRAX project -- into one Jacobian: each project
    # contributes the rows for the obs it has footprints for (see ``stilt_projects``).
    stilt_project: str | Path | list[str | Path] = field(
        default_factory=lambda: os.environ.get("SLV_STILT_DIR", DEFAULT_STILT_PROJECT)
    )
    # Named footprint config or hash; None = the finest in the project. The cache key sees
    # only this value, not what None resolved to: set it explicitly if the project may gain
    # a finer footprint.
    footprint: str | None = None
    sparse_jacobian: bool = True

    # --- Prior Error Covariance (S_0) ---
    prior_base_std: float = 0.019
    prior_std_frac: float = 0.5
    prior_time_scale: str = "32d"
    prior_spatial_scale: float = 5.0

    # --- Model-Data Mismatch (S_z) ---
    mdm_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Reject settings that would otherwise only fail after the Jacobian is built."""
        if self.flux_freq not in FLUX_FREQ_TIME_STEPS:
            raise ValueError(
                f"flux_freq={self.flux_freq!r}; supported: {list(FLUX_FREQ_TIME_STEPS)}"
            )
        if self.bias_grouping not in BIAS_GROUPINGS:
            raise ValueError(
                f"bias_grouping={self.bias_grouping!r}; expected one of {BIAS_GROUPINGS}"
            )
        if self.background not in BACKGROUNDS:
            raise ValueError(
                f"background={self.background!r}; expected one of {BACKGROUNDS}"
            )
        if str(self.prior).lower() not in PRIORS:
            raise ValueError(f"prior={self.prior!r}; expected one of {PRIORS}")
        get_mdm_comp_configs(self.mdm_config)  # raises on unknown component names
        if not (self.dx > 0 and self.dy > 0):
            raise ValueError(f"dx and dy must be positive, got {self.dx}, {self.dy}")
        if not (self.xmin < self.xmax and self.ymin < self.ymax):
            raise ValueError(f"empty domain: {self.bbox}")

    #: Derived values cached on first access; setting any field drops them, so a config
    #: changed after use (a sweep, a notebook) never serves a stale grid or MDM list.
    _DERIVED = ("grid", "state_grid", "grid_coords", "mdm_components", "site_config")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self.__dataclass_fields__:
            for attr in self._DERIVED:
                self.__dict__.pop(attr, None)

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

    # --- Jacobian Coverage Filter ---
    # Percentile threshold for Jacobian coverage filtering.  Per-cell coverage
    # is the mean absolute Jacobian sensitivity per observation location.
    # Cells below this percentile are removed from the state vector across all
    # time steps and held fixed at the prior.
    # E.g., 10 removes the least-sensitive 10% of cells.
    # None = no filtering (default).
    jacobian_coverage_percentile: float | None = None

    # --- Inversion Solver Settings ---
    min_obs_per_interval: int = 60
    # Read by fips' filter_state_space but not applied: it filters on the obs count only
    # (counting simulations needs the Jacobian, built after that filter).
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
        """``(xmin, ymin, xmax, ymax)``."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def extent(self):
        """``(xmin, xmax, ymin, ymax)``, cartopy's order."""
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def map_extent(self):
        """:attr:`extent` padded by 0.05 degrees, for maps."""
        buffer = 0.05
        return (
            self.xmin - buffer,
            self.xmax + buffer,
            self.ymin - buffer,
            self.ymax + buffer,
        )

    @property
    def resolution(self) -> str:
        """Cell size as ``"{dx}x{dy}"``."""
        return f"{self.dx}x{self.dy}"

    @cached_property
    def grid(self):
        """The flux cells as a lon/lat ``DataArray`` of zeros: the prior's regrid target.

        Built from :attr:`state_grid`'s axes, so the prior and the Jacobian columns
        enumerate the same cell centres by construction.
        """
        x, y = self.state_grid.axes
        grid = xr.DataArray(
            np.zeros((len(y), len(x))), coords={"lat": y, "lon": x}, dims=("lat", "lon")
        )
        grid = grid.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        grid = write_rio_crs(grid, crs="EPSG:4326")
        return grid

    @cached_property
    def state_grid(self):
        """
        The flux cells as a PYSTILT ``Grid``: the domain snapped outward to whole cells.

        This is the one definition of the flux cells -- the Jacobian is built on it and
        :attr:`grid` (the prior) takes its axes. When the extent is not a whole number of
        cells the last row/column is still a whole cell, reaching past ``xmax``/``ymax``
        (``ymax=40.93`` with ``dy=0.05`` gives a top row centred at 40.925, covering
        40.90-40.95).
        """
        from stilt import Grid

        eps = 1e-9
        nx = int(np.ceil((self.xmax - self.xmin) / self.dx - eps))
        ny = int(np.ceil((self.ymax - self.ymin) / self.dy - eps))
        # stilt counts floor((max - min) / res) cells, and the float error of that
        # subtraction (40.93 - 40.45 = 0.47999999999999687) can drop the last whole cell,
        # so the upper bounds carry a negligible pad.
        grid = Grid(
            xmin=self.xmin,
            xmax=round(self.xmin + nx * self.dx, 10) + eps,
            ymin=self.ymin,
            ymax=round(self.ymin + ny * self.dy, 10) + eps,
            xres=self.dx,
            yres=self.dy,
        )
        x, y = grid.axes
        if (len(x), len(y)) != (nx, ny):
            raise ValueError(
                f"state grid has {len(x)}x{len(y)} cells, expected {nx}x{ny} "
                f"(dx={self.dx}, dy={self.dy})"
            )
        return grid

    @cached_property
    def grid_coords(self):
        """Every cell centre as a ``(lon, lat)`` tuple."""
        return pd.MultiIndex.from_product(
            [self.grid["lon"].values, self.grid["lat"].values]
        ).to_list()

    @property
    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """``(tstart, tend)`` as timestamps."""
        return (pd.Timestamp(self.tstart), pd.Timestamp(self.tend))

    @property
    def stilt_projects(self) -> list[Path]:
        """``stilt_project`` as a list of paths, whether one project or several was given.

        ``stilt_project`` itself is left as given, so a single-project config keeps the same
        cache key it always had; only code that opens the projects normalises it.
        """
        p = self.stilt_project
        items = p if isinstance(p, (list, tuple)) else [p]
        if not items:
            raise ValueError(
                "stilt_project is an empty list; give at least one project."
            )
        return [Path(x) for x in items]

    @property
    def mobile_obs_key(self) -> str | None:
        """Cache fingerprint of ``mobile_obs``: its path plus size and mtime.

        Hashing the path alone would miss a file rebuilt in place -- e.g. regenerated with a
        revised inlet lag -- and silently reuse obs, Jacobian and MDM built from the old
        one. Size+mtime change whenever the file is rewritten.
        """
        if self.mobile_obs is None:
            return None
        path = Path(self.mobile_obs)
        if not path.exists():
            return f"{path}|missing"
        st = path.stat()
        return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"

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
        return [(hour - self.utc_offset) % 24 for hour in self.subset_hours]

    @cached_property
    def site_config(self):
        """The packaged site config (:func:`~slv.measurements.sites.load_site_config`)."""
        return load_site_config()
