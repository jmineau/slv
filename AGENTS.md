> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

# AGENTS.md — Developer and Agent Guide for slv

`slv` is the user's **Salt Lake Valley** research codebase — project-specific
plumbing that ties together `lair`, `uataq`, `fips`, and `pystilt` to support
methane (and other trace gas) emission estimation in the SLV. Active research
code; calendar-versioned (e.g. `2026.2.0`).

PyPI/import name: `slv`. Source in `src/slv/`.

## What slv is

- An **integration layer**, not a general library. Its job is to make the
  user's everyday SLV workflows reproducible: defining the domain, loading
  measurements and inventories, running flux inversions (via `fips`),
  building figures.
- Depends directly on the user's own packages — `lair[science]` and
  `uataq` are git-installed from main; `fips[flux]` (which pulls in
  `pystilt`) lives in the `inversion` extra.

## Module layout

```
src/slv/
  __init__.py        version metadata + get_data_dir(env_var) helper
  domain.py          SLV inversion domain constants (XMIN/XMAX/YMIN/YMAX,
                     BBOX, EXTENT)
  basemap.py         SaltLake — cartopy map of the valley; chainable add_* layers
                     (population, trax, sites, mesowest, points, interstates, borders,
                     inventory, legend, inset, north_arrow). Data via load_* functions
                     ($SLV_SPATIAL_DIR group data; $SLV_USER_DATA_DIR census ACS 2022 +
                     mesowest/); tiles="terrain" = Stadia stamen_terrain ($STADIA_API_KEY,
                     network when drawn; attribution added). Tests: tests/test_basemap.py,
                     synthetic data, no network.
                     (tiler, inventory, borders, TRAX, sites, ...)
  measurements/
    __init__.py      re-exports aggregate_obs, load_concentrations,
                     get_site_coordinates, load_site_config
    aggregate.py     observation aggregation helpers
    background.py    background concentration logic
    concentrations.py  load measurement concentrations
    instruments.py   site/instrument metadata
    mobile/          TRAX / mobile platform subpackage (one concern per module; the
                     package __init__ re-exports every public function and table, so
                     `from slv.measurements.mobile import X` works for them; classifier
                     thresholds and GPS constants stay in their modules)
      network.py     TRAX lines/points, storage yards (JRRSC and MRSC both
                     packaged), shed footprints, filter_near_routes (dissolved
                     buffers - no duplicates on shared track), get_geodf, UTM12;
                     group_dir()/user_dir() resolve the data roots on use (GROUP_DIR /
                     USER_DIR via module __getattr__) -- keep slv importable without
                     the CHPC env vars, CI relies on it
      gps.py         thin uataq wrappers: read_lin_gps (lin gps qaqc; uataq names
                     the sat count N_Satellites -> renamed N_Sat), read_horel_cr1000
                     (horel gps+cr1000 at lvl="raw" = the h5 files; the horel
                     qaqc/final CSVs lack NSAT/RSTS), read_trax_gps (by era: lin
                     before 2018-11-19, horel after, fill_gaps=True fills horel
                     outages from the lin GPS), merge_with_gps (horel_fallback:
                     rows the lin GPS never covered are georeferenced from the horel
                     logger where the Pi clock checks out -- lin_clock_offset_by_day /
                     clock_checked_days / fill_gps_gaps_with_horel; gps_source tags each row)
      location.py    per-minute indoor/outdoor classifier: depot (degraded GPS:
                     scatter > 1 m or <= 6 sats, 15-min vote) / yard / line
                     (pass-by within 300 m of a yard) / route / stopped / unknown
                     (no fix, > 100 m off track, shed ejecta); speed_est fallback
                     for GPGGA-only eras; state_intervals, label_observations
      calibration.py cal_source tags + trax_uncalibrated_windows.csv handling;
                     trax_epoch_offsets.csv (manual-cal era analyzer offsets vs UOU,
                     -0.002..-0.024 ppm): epoch_offset_column stores them per row,
                     apply_epoch_offset subtracts them (load_trax_obs(epoch_offset=False)
                     to test without, no rebuild needed)
      obs.py         build_trax_obs (per-year chunks; slope guard on calibrated rows; every GPS fix + cal_source + low_pressure + state/indoor/
                     yard_name) / load_trax_obs(location="on_track"|"outdoor"|"all")
                     cached at $SLV_USER_DATA_DIR/trax/obs.parquet; LOCATION_SETS.
                     2026-09-22 audit rules, all from _read_support (qaqc CH4/flag/
                     cavity P read WITHOUT normalize_pollutant): apply_pressure_rule
                     (flag -64 overwrites -63 in lgr_ugga_qaqc.r, so pressure is
                     re-checked here; manual_cal rows exempt -- apply_low_pressure_rule
                     owns them), apply_dropout_rule + dropout_times (+-8 min around any
                     atmosphere row < 1.70 ppm: a laser dropout reads low for minutes to
                     hours and its shoulder passes QC), recover_hot_rows (flag -64 rows
                     have a fitted CH4d_m but a blanked CH4d_ppm_cal: qaqc value / slope)
      transects.py   load_transects (archived transect x point matrices)
      receptors.py   STILT receptors for paper 2: load_trax_network_points (the 50-m
                     points from load_trax_points(50) — same points_along_line generator
                     as the 2-km points, so shared track is one row of points — each
                     tagged with its nearest 2-km segment; cached points_50m_network.geojson),
                     load_trax_fixes (GPS columns of obs.parquet only), find_segment_crossings
                     (runs of fixes on one segment, gap <= 10 min; `lines` = letters common
                     to the points hit), release_points (segment points on the crossing's
                     line — a junction segment holds several lines' arms), build_trax_receptors
                     (one PYSTILT MultiPointReceptor per crossing at the median time;
                     coverage measured along the route via s_<line>, duration filter),
                     find_dwells / build_dwell_receptors (parked-train point receptors,
                     r_idx dwell_<d>_<YYYYMMDDHH>, freq >= 1h) / label_dwell_site (yard),
                     filter_receptor_hours (fixed UTC_OFFSET, standard-time hours).
                     Driver: SLV/stilt/ops/build_trax_receptors.py
      receptor_obs.py the CH4 observation for each TRAX receptor: trax_receptor_observations
                     (mean CH4 over a crossing's pass / a dwell's hour, window shifted by the
                     per-epoch inlet lag, keyed (obs_location, obs_time) via
                     stilt.read_receptors so it matches the footprints exactly),
                     inlet_lag_seconds. Driver: SLV/stilt/ops/build_trax_receptor_obs.py.
                     Consumed by InversionConfig.mobile_obs -> get_slv_observations
                     (load_mobile_obs); cache keys on mobile_obs_key (path+size+mtime).
      wyoming.py     Wyoming mobile lab (Aeris + met readers, ratios, wind-barb map)
      jrrsc.geojson, mrsc.geojson (yards), trax_depots.geojson (sheds),
      trax_uncalibrated_windows.csv  packaged data; the yard polygons ship with
                     the package so nothing but UTA_TRAX.geojson comes from group data
    pollutants.py    pollutant metadata
    sites.py         site_config loader; site_coordinates resolver
    site_config.csv  packaged data
  emissions/
    __init__.py      re-exports plot_point_sources
    point_sources.py  CH4 point source plotting
    ch4_point_sources.csv  packaged data
  meteorology/
    __init__.py      (currently empty)
    pcaps.py         SLV-specific PCAP wrappers
  inversion/
    __init__.py      re-exports InversionConfig, SLVMethaneInversion, Sweep,
                     run_sweep_job, SweepResults
    config.py        InversionConfig — pydantic-ish config for an inversion;
                     DEFAULT_STILT_PROJECT + stilt_project_dir() ($SLV_STILT_DIR);
                     state_grid (stilt Grid, domain snapped outward to whole cells) is
                     the one definition of the flux cells, `grid` takes its axes
    pipelines.py     SLVMethaneInversion — the pipeline proper: get_obs / get_prior /
                     get_forward_operator (+ cached _get_flux_jacobian) / get_prior_error /
                     get_constant, state filtering, obs aggregation, run, obs_sites.
                     Composed from the mixins below (MRO: Reporting, CoverageFilter, Bias,
                     ModelDataMismatch, fips FluxInversionPipeline); every method keeps its
                     name, so subclasses override as before. Re-exports the cache helpers.
    cache.py         component cache: COMPONENT_DEPS sets, _component_hash, _pkg_rev /
                     _version_tag, fips_cache
    mdm.py           ModelDataMismatchMixin: get_modeldata_mismatch, multiplicative scale,
                     per-obs (subhour) std
    bias.py          BiasMixin: get_bias, get_bias_jacobian, get_site_group
    coverage.py      check_state_cells; CoverageFilterMixin (Jacobian coverage filter,
                     reconstruct_posterior)
    report.py        ReportingMixin: domain totals, summarize, desroziers, plot_*
    priors.py        prior flux construction
    covariances.py   covariance construction (kernels, scaling)
    data.py          inversion-time data loading
    background.py    inversion background model
    viz.py           inversion-specific plotting
    sweep.py         Sweep / run_sweep_job / SweepResults — hyperparam sweeps
  py.typed           ships type hints
tests/               pytest
docs/                Sphinx
ci/environment.yml   conda env (for the xesmf install path)
.claude/             local Claude Code settings (`settings.local.json`)
```

## Data files and env vars

`slv.get_data_dir(env_var)` is the canonical way to resolve external data
paths. Pattern:

```python
from slv import get_data_dir
DAQ_DIR = get_data_dir("SLV_DAQ_DIR")
```

One exception: `InversionConfig.stilt_project` defaults to `SLV_STILT_DIR` if set,
else `DEFAULT_STILT_PROJECT` (the CHPC production PYSTILT project), so the config
stays usable without the env var.

Raises `OSError` with a clear message if `env_var` is unset — don't paper
over with `os.environ.get(...)`. Documented env vars belong in the README
or here.

Packaged CSV data (`site_config.csv`, `trax_uncalibrated_windows.csv`,
`ch4_point_sources.csv`) ships via
`[tool.setuptools.package-data]` in `pyproject.toml`. Don't move these
without updating that.

## Public API (effective)

`slv.__init__` is intentionally small (just `get_data_dir` and version
metadata). Subpackages are the API surface:

```python
from slv.domain import BBOX, EXTENT, XMIN, XMAX, YMIN, YMAX
from slv.basemap import SaltLake
from slv.measurements import aggregate_obs, load_concentrations, get_site_coordinates
from slv.emissions import plot_point_sources
from slv.inversion import (
    InversionConfig, SLVMethaneInversion,
    Sweep, run_sweep_job, SweepResults,
)
```

## Install paths

Two paths depending on `xesmf` (regridding) needs:

```bash
# Standard (uv) — what the user runs day-to-day
uv sync                        # base
uv sync --extra inversion      # adds fips[flux] + joblib + seaborn

# With xesmf (needs conda-built ESMF)
conda env create -f ci/environment.yml
conda activate slv
pip install --no-deps -e .
```

The `inversion` extra is what unlocks `slv.inversion` (it depends on
`fips[flux]`, which in turn depends on `pystilt`). The base install is
enough for `domain`, `basemap`, `measurements`, `emissions`.

## Dev commands

Driven by `just` + `uv`. There is **no `just install`** recipe — use
`uv sync` directly.

| Command | What it does |
|---|---|
| `just test` | `uv run pytest -v` |
| `just quality-check` | ruff (`src/slv`) + pyright (`src/slv`) + tests |
| `just ruff` | `uv run ruff check --fix` + `uv run ruff format` on `src/slv` |
| `just build-docs` | clean + Sphinx HTML build |
| `just pre-commit` | `uv run pre-commit run --all-files` |
| `just clean` | wipe build artifacts, caches, coverage, docs |

CI: `.github/workflows/` has `tests.yml`, `quality.yml`, `docs.yml`, all on the
conda env in `ci/environment.yml` (no `uv` there -- workflows call tools directly,
not `just`). No CHPC data roots are set in CI: tests must not read group data. pyright
reports the pandas-noise rules as warnings (`[tool.pyright]`) and fails on the rest;
docstr-coverage fails below `--fail-under` in `quality.yml` (90; docstrings at 95 %, so
new public functions need one). The CI env carries the inversion extra's conda deps
(seaborn, joblib) because slv is installed `--no-deps`; add new extras there too. The docs
build is warning-free: keep it so (`sphinx-build -q -b html docs <out>`), and numpy
docstring sections must be ones napoleon knows (an unknown header such as "Cache layout"
is read as more parameters).

## Invariants to respect

- **Domain constants live in one place** (`slv.domain`). Don't hardcode
  `XMIN`/`XMAX`/`YMIN`/`YMAX` elsewhere; import from `slv.domain`.
- **Data paths via `get_data_dir`**, not raw `os.environ.get`. The
  error message tells users which env var to set.
- **Upstream pins**: `lair` and `uataq` are pulled from git main
  (no version pin). Bumping behavior in those repos can silently
  change `slv` results — coordinate changes.
- **Inversion pipeline structure** mirrors `fips.InversionPipeline`'s
  template-method pattern. New inversion variants should subclass
  `SLVMethaneInversion` (or `fips.InversionPipeline`), not rewrite the
  loop.
- **Packaged CSVs are part of the API contract.** Schema changes break
  downstream notebooks.
- **Inversion cache keying** (`inversion/cache.py`): `@fips_cache` writes to
  `{cache_dir}/.fips/{version_tag}/{component}/{hash}.pkl`, where `version_tag` is
  the `git describe` rev of the editable fips/pystilt checkouts (`_version_tag` /
  `_pkg_rev`) — so committing or editing those packages busts the cache
  automatically, no reinstall needed. A copy under `site-packages` (the uv `.venv`,
  which sits inside the slv repo) uses its metadata version instead: `git describe`
  there would answer with slv's rev. The conda `slv` env imports fips/PYSTILT from
  their checkouts. `get_inputs` raises when the prior's cells and the Jacobian's
  columns differ (stale caches after a grid change). Components other than
  the Jacobian also hash the *release* of lair / uataq (`COMPONENT_PACKAGES`,
  `_release`: an editable checkout's static pyproject version or latest git tag, else
  the installed version) -- a release, not an edit, rebuilds obs/prior/MDM/constant. Caveats: (1) slv's *own* rev is deliberately
  not in the key, so when you change slv component-building logic
  (obs/prior/mdm/constant) force a rebuild via `config.cache_overwrite`;
  (2) reinstall editable packages (`uv sync` / `pip install -e`) when you cut a
  version so `importlib.metadata` matches the source — this does not affect the
  cache key but keeps `pip show` and other version consumers honest.

## Conventions and tooling

- **Python**: 3.11+ (`ruff.target-version = "py311"`; pandas 3 and current xarray need 3.11).
- **Linting**: ruff selects `E, F, UP, B, SIM, I` and ignores `E501`.
  (Note: no `D` pydocstyle rules here, unlike `fips`.)
- **Types**: pyright using `.venv`. `py.typed` shipped.
- **Coverage**: `tests/` only; standard `coverage.exclude_also` patterns.
- **Dev group**: brings in `slv[inversion]`, ipykernel, ruff, pyright,
  pytest, sphinx, pre-commit.

## Common workflows

### Standard SLV figure
```python
from slv.basemap import SaltLake
m = (SaltLake(tiles="terrain")            # Stadia stamen_terrain; None = no tiles
     .add_population()                    # block-group density + colorbar panel
     .add_trax(lines="RG")
     .add_sites(["wbb", "ldf", "hdp"], labels={"wbb": "UOU"},
                label_offset={"ldf": (-45, 12)})
     .add_mesowest()
     .add_legend().add_inset().add_north_arrow())
m.fig.savefig("slv.png", dpi=300)
```

### Methane inversion
```python
from slv.inversion import SLVMethaneInversion, InversionConfig
cfg = InversionConfig(...)
inv = SLVMethaneInversion(cfg)
result = inv.run()    # internally drives fips.InverseProblem.solve()
```

### Hyperparameter sweep
```python
from slv.inversion import Sweep, run_sweep_job
sweep = Sweep(...)
results = run_sweep_job(sweep)
```

## Gotchas

- `slv.meteorology.__init__` is currently empty — submodules
  (`pcaps`) must be imported explicitly (`from slv.meteorology import pcaps`).
- The `inversion` extra has `# "xesmf"` commented out — installing
  `xesmf` is intentionally out of `pip`'s scope here.
- `get_data_dir` raises `OSError` rather than `KeyError` (intentional;
  matches the user's expectation for "missing config file" semantics).
- `basemap.SaltLake` mutates a matplotlib figure with cartopy axes;
  it is not thread-safe.
- Mobile (TRAX) obs are keyed by PYSTILT location_id, not site. Anything keyed by
  site must resolve through `SLVMethaneInversion.obs_sites` (receptor -> the single
  mobile site in `config.sites`) -- the MDM and the bias Jacobian do. The rolling
  background uses stationary sites only (`split_sites`; override with
  `background_kwargs={"background_sites": [...]}`) and each obs takes its hour's
  value; `get_slv_subhour_std` skips mobile sites.
- `get_forward_operator` is a thin wrapper: `_get_flux_jacobian` is the cached
  (`forward_operator`) flux Jacobian; the bias block is rebuilt every run. `get_prior`
  memoizes `_build_prior` per pipeline; `_multiplicative_scale` reuses both.
- `InversionConfig.__post_init__` validates flux_freq / bias_grouping / background /
  prior / mdm_config / domain; setting any field drops the cached derived values.
- Never change a config field's default spelling (e.g. `"32d"`) or a
  `COMPONENT_DEPS` set casually: both are hashed into cache keys (a forward_operator
  key change means a full Jacobian rebuild) and sweep config_ids. Normalize at the
  parse site instead (`covariances.normalize_duration`).
- Tests: `tests/conftest.py` points `SLV_STILT_DIR` at an empty tmp dir, so a test that
  reaches the Jacobian build without stubbing `_get_flux_jacobian` fails fast instead
  of scanning the production project. Pipeline methods are tested on
  `object.__new__(SLVMethaneInversion)` with only `config` set (no fips `__init__`, no
  data); viz tests stub `GeoAxes.add_image` so no tiles are fetched; the TRAX loaders run
  on a synthetic two-line network (`tests/measurements/mobile/test_network.py`). Tests
  needing xesmf (lair cell areas) `importorskip` it: they run in the conda env / CI only.
  Measure coverage with `--cov=slv`: `--cov=slv.inversion.viz` imports numpy twice and
  fails collection.
- fips `Block` squeezes its data, so a one-element Series becomes a scalar and fails
  validation ("All levels in the row index of Block must be named"). Build test
  vectors with at least two entries; a real one-interval bias block would hit it too
  (fips bug, not fixed there yet).
- `get_slv_observations` loads the TRAX route points only when a mobile site is in
  `sites`; tower-only runs need no TRAX data.
- `merge_with_gps` drops fixes outside `gps.ALTITUDE_RANGE_MSL` (1000-3500 m, room for
  canyon and mountain roads); it is on the `build_trax_obs` path, so changing it changes
  `obs.parquet` on the next rebuild.
- `GMLDiscrete` never downloads into lair's shared GML dir (`$LAIR_GML_DIR`;
  lair main dropped the built-in `lair.noaa.GML_DIR` and all its CHPC paths,
  see `LAIR_*` env vars): it reads
  `$SLV_USER_DATA_DIR/gml`, then the group copy, then downloads into the user cache.
  Tests stub `noaa.GMLData.download`.
- `InversionConfig.mdm_config` keys must be names in `DEFAULT_MDM_CONFIG`
  (part, instr, aggr, subhour, bg, transport); `get_mdm_comp_configs` raises on
  anything else. Retired keys (`transport_pbl`, `transport_wind`) used to be
  ignored silently, leaving the MDM cache hash unchanged.
- fips-managed cache files live under `<cache>/.fips/<fips+pystilt version
  tag>/<component>/<hash>.pkl` (`_version_tag()`); tests that look for cache
  files must build the path the same way.
- uataq names the GPS satellite count `N_Satellites`; the horel `qaqc`/`final`
  CSVs lack it (and RMC status) but `lvl="raw"` (the h5 files) has both. The
  pilot-era horel logger (to 2018-11-19) is 1-min and unusable for the
  location classifier -- `read_trax_gps` switches sources by era.
