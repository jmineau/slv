# Changelog

All notable changes to `slv` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are calendar-based (YYYY.M.PATCH).

## [Unreleased]

### Added

- TRAX dwell receptors: `find_dwells` picks out the periods a train sits parked at one
  outdoor spot (a yard track, a terminus platform) and `build_dwell_receptors` turns
  each hour of one into a PYSTILT point receptor at the dwell's median position.
  `label_dwell_site` names the service yard a dwell is in (`yard_name`). Both receptor
  builders take a local-time `hours` window (`filter_receptor_hours`)
- Receptor-paired TRAX observations: `trax_receptor_observations` gives one CH4
  observation per receptor (mean over a crossing's pass or a dwell's hour, shifted by
  the inlet lag, `inlet_lag_seconds`), keyed like the footprints;
  `InversionConfig.mobile_obs` points the inversion at that table
- `build_trax_obs` slope guard (`apply_slope_guard`): pipeline-calibrated rows whose
  slope is more than 5 % off the day's median are dropped, which removes the hours
  after a bad reference period
- Crossing coverage along the route: network points carry `s_<line>`, their distance
  along each line, and `build_trax_receptors` keeps crossings that drove at least
  `min_coverage` (default 0.5) of their release geometry and lasted at most
  `max_duration` (default 15 min; terminus dwells are not traverses). It was a
  straight-line span from the first fix, which under-scored bent segments
- Docstrings for the public API (docstring coverage 95 %, CI floor 90 %); the usage
  guide covers the map, measurements, an inversion, sweeps and TRAX receptors (#6)
- Tests (#6): line coverage 69 -> 89 % (326 -> 422 tests). New: the coverage filter and
  `reconstruct_posterior`, Desroziers, domain totals and the summary, `run` and the plot
  hooks, obs-space filtering and aggregation, every viz plot, the inversion obs /
  background / prior paths, the TRAX network and GPS readers (a synthetic two-line
  network through the real `line_locate_point` path), transects and the Wyoming lab.
  Two weak tests fixed (the off-track drop test passed without the drop; an always-true
  assertion)

### Changed

- Decisions from #7:
  - `merge_with_gps` drops fixes by a fixed plausible altitude range for the Salt Lake
    area, `ALTITUDE_RANGE_MSL` = 1000-3500 m (canyon roads and the Uinta passes fit;
    `altitude_range=None` keeps all), instead of the 1st-99th percentile of each chunk.
    That cut dropped 2 % of every chunk, the high end of a line included, and every fix
    when the altitude was constant or missing; fixes without an altitude are now kept.
    `obs.parquet` gains those fixes when it is next rebuilt
  - `label_dwell_site` counts a dwell as in a yard within the location classifier's
    `YARD_BUFFER` (30 m) of its polygon (`yard_buffer=`); it had to be strictly inside,
    so a train parked 20 m past JRRSC's drawn edge was not labelled
  - `load_trax_points(resolution_factor=...)` caches to its own file
    (`points_<spacing>m_rf<factor>.geojson`); it returned the default points. The
    default file name is unchanged
  - `GMLDiscrete` reads `$SLV_USER_DATA_DIR/gml`, then lair's group copy, and downloads
    into `$SLV_USER_DATA_DIR/gml` only when neither has the file (`refresh=True` to
    re-fetch), never into the shared group directory; a failed download says to fetch
    on a login node (compute nodes have no outbound network)
  - point sources: "Northrop Grumman" spelled right; LPG stations have their own marker
    (tri-down, beside CNG's tri-up)
  - `load_concentrations` says the Wyoming mobile lab (`wyo`) is read by
    `slv.measurements.mobile.wyoming` instead of "Unknown org/instrument combo"
  - `build_dwell_receptors` accepts only a 1-hour `freq`: `receptor_obs` averages each
    dwell obs over the hour its `r_idx` names, so a 2-hour receptor (released at the
    bin's median) was paired with the first hour's obs only
- The 50-m TRAX network points come from `load_trax_points(50)`, like the 2-km points,
  so shared track has one set of points; a crossing's receptor releases only from the
  segment points on the line it drove, not every arm of a junction segment
- Docs: `release` is read from the installed package; the install pages name the
  real install paths (`uv sync` installs the `dev` group; there is no `dev` or `docs`
  extra) and the data environment variables; the docs build has no warnings

- `slv.basemap` rewritten (#5). `SaltLake(bbox, ax=, tiles="terrain")` makes the map
  and chainable `add_*` methods layer it: `add_population` (ACS 2022 block-group
  density with a colorbar panel), `add_trax`, `add_sites` (open circles, optional
  labels), `add_mesowest`, `add_points`, `add_interstates`, `add_borders`,
  `add_inventory`, `add_legend` (stacked-line TRAX entry), `add_inset`,
  `add_north_arrow`. Data come from `load_*` functions over the SLV env vars (the old
  paths had all moved); tiles are Stadia's `stamen_terrain`. The old constructor flags
  (`TRAX=True`, `UUCON=True`, ...) and the unimplemented stubs are gone

- `slv.inversion.pipelines` is split by concern: `cache` (component cache),
  `mdm` (model-data mismatch), `bias`, `coverage` (state-cell check and coverage
  filter), `report` (totals, summary, plots). `SLVMethaneInversion` combines them as
  mixins, keeps every method name, and is imported as before; the cache helpers are
  still importable from `pipelines`
- Component cache keys include the release versions of lair and uataq: obs, prior,
  prior error, model-data mismatch and constant rebuild after a new lair / uataq
  release (a checkout's latest tag or static pyproject version, else the installed
  version), the Jacobian does not. These components rebuild once on upgrading
- Python >= 3.11 (was 3.10): pandas 3 and current xarray need 3.11, and CI only
  ever tested 3.11 / 3.12
- `slv.measurements.mobile.network` resolves `$LINGROUP_DATA_DIR` /
  `$SLV_USER_DATA_DIR` when used (`group_dir()`, `user_dir()`) instead of at import,
  so slv imports without the CHPC data roots set. `GROUP_DIR` / `USER_DIR` are still
  importable and resolve on access
- CI (#3): the location-features test no longer reads the TRAX lines from group data;
  the docs workflow calls `sphinx-build` directly (its conda env has no `uv`); pyright
  reports the pandas-annotation rules as warnings and fails on the rest (0 errors);
  docstring coverage fails below 63 %; the workflows no longer need placeholder data
  roots. Ruff is clean, `basemap.py` included
- Relocked `lair` (f01c1c7) and `uataq` (a4dbfa6). A fresh install now gets lair's
  quarterly / biweekly `absolute_emissions`, so `flux_freq="QS"` / `"2W"` totals work
  outside the conda env, plus lair's soundings / pcaps / noaa / pandas-3 fixes

### Fixed

- `SweepResults.best(target=...)` returned the matches ordered by distance to 1, not to
  `target`, so `best().iloc[0]` was not the closest match for `target != 1`
- `merge_with_gps` joined the GPS fixes (lon/lat) with `storage_polygon` in the
  polygon's own CRS, so a yard given in UTM matched nothing and the parked fixes were
  kept (the packaged yards are lon/lat, so the default was right)
- `get_slv_observations` loaded the TRAX route points for every run, so a tower-only
  inversion needed `$SLV_USER_DATA_DIR` and the TRAX point cache; they are loaded only
  when a mobile site is in `sites` (the aggregation only uses them for mobile rows)
- `load_trax_points` failed writing its cache when `$SLV_USER_DATA_DIR/trax` did not
  exist yet
- GPS readers cast the numeric GPS columns to float64, and the slope guard its slope
  column: per-year reads of the GPGGA-only years (Dec 2015 - Jan 2018) and uataq's
  multiprocess reads returned Arrow strings, which broke the per-minute medians
- Inversion robustness (#4):
  - the Jacobian coverage filter added the removed cells' contribution to the
    constant by position (Jacobian rows are simulations, constant rows obs); now by
    label
  - the bias Jacobian is rebuilt every run instead of cached with the flux Jacobian,
    whose key leaves out the obs filters the bias block is indexed by (a stale bias
    block survived a filter change); the "site" / "site_group" groupings now resolve
    TRAX receptors to their mobile site instead of giving them no bias column
  - `InversionConfig` rejects an unsupported `flux_freq`, `bias_grouping`,
    `background`, `prior` or MDM component, and an empty domain, at construction
    (they used to fail after the Jacobian build); setting a field drops the cached
    `grid` / `state_grid` / `mdm_components`
  - `build_location_site_map` matches within ~1 m (was `np.isclose`'s relative
    tolerance, ~95 m in longitude); the production project maps identically
  - the obs hour window follows `config.utc_offset` (`load_concentrations(utc_offset=)`),
    as the Jacobian's already did
  - `get_prior_error` treated `bias_std=0.0` as no bias
  - domain totals are labelled as mass per flux interval (e.g. "Gg per MS interval");
    they were printed as "Gg/m2/s" and plotted as "g/s"
  - with `cache=False` the multiplicative MDM rebuilt the prior and the whole Jacobian,
    and it made a sparse Jacobian dense; it now reuses the run's and stays sparse
  - `SweepResults.sensitivity()` took the lowest chi^2 as best (now closest to 1, or
    `target=`); `Sweep.run(n_jobs=0)` returned an unusable results object
  - duration strings such as `"14d"` are converted to `"14D"` where pandas parses
    them (lowercase `d` is deprecated); config spellings, and so cache keys and sweep
    IDs, are unchanged
- TRAX (mobile) obs through the inversion (#2). With a mobile site in `sites`:
  the default MDM `instr` term looked up each `obs_location`'s organization, but a
  receptor obs is keyed by its PYSTILT location_id (`KeyError`); the rolling
  background loaded the train and crashed unstacking its per-grid-point rows; the
  hourly background was joined on exact `obs_time`, so a receptor released at 20:13
  found none and was silently dropped; and the `subhour` MDM term loaded (and
  GPS-merged) the whole TRAX record only to discard it. MDM terms now look up each
  obs's site (`SLVMethaneInversion.obs_sites`: receptors belong to the mobile site),
  the rolling baseline comes from the stationary sites (`background_sites` in
  `background_kwargs` to choose them), each obs takes the background of its hour,
  and `subhour` loads stationary sites only. `get_slv_subhour_std` also crashed
  building its empty result
- Prior time alignment: `load_epa_prior` / `load_edgar_prior` matched flux times to
  inventory times by nearest neighbour, so with an annual inventory Aug-Dec took the
  *next* year's field (and daily fluxes past mid-month the next month's). Each flux
  time now takes the inventory period it falls in (`priors.align_to_flux_times`)
- `InversionConfig.grid` (the prior) and `state_grid` (the Jacobian) enumerated
  different cells except at 0.05 and 0.1 deg: lair kept partial cells only when
  their centre fell inside the domain and rounded 0.025-deg centres to 3 decimals,
  and stilt dropped a whole cell to float error in `(40.93 - 40.45) / 0.01`. fips
  zero-fills the missing Jacobian columns, silently. `grid` is now built from
  `state_grid.axes` (unchanged at 0.05 / 0.1 deg), and `get_inputs` raises when the
  prior's cells and the Jacobian's columns differ (e.g. stale caches)
- Cache version tag in a venv inside the slv repo (`uv sync`'s `.venv`): `git
  describe` from fips/pystilt in `site-packages` answered with slv's revision, so
  every slv commit orphaned the whole cache. Installed copies now use their
  metadata version; editable checkouts still use `git describe`
- `prior` / `prior_error` cache keys now include `sites` (the site / site-group
  bias blocks are indexed by it)
- The auto-built STILT location -> site map is no longer written into the config
  on a Jacobian cache miss, which changed the run's sweep `config_id`
- Multiplicative MDM with `scale_on="prior"` multiplied the Jacobian and the prior
  by position although their index orders differ; now matched by label
- Sweep results CSV: success and error rows had different columns and were
  appended by position, so a mixed CSV failed to read or shifted metrics into
  `error`. Every row now has the same columns (appends align to the header);
  `runtime_seconds` is filled
- `aggregate_obs`: multi-unit `freq` ("15min", "2h") did not bin (only truncated
  to the base unit); rows without a position crashed the `mobile_points` match; a
  tuple `by` failed; the caller's frame was modified
- `load_concentrations`: sites with no instruments (`arc`, `uta`) and
  `orgs="NOAA GML"` crashed; a pollutant missing from an instrument's data now
  skips that pollutant instead of failing the call
- `get_pcap_events` returned the cached events whatever `threshold` /
  `min_periods`; the cache is now per parameter pair (the defaults keep
  `pcap_events.csv`). `filter_pcap_events(level=...)` took the time range of the
  whole MultiIndex; `get_soundings` pointed lair at `$SLV_SOUNDINGS_DIR` instead of
  its station directory
- `UATAQCH4` silently ignored unknown key suffixes; it now raises
- `load_trax_obs(location=...)` never applied the location filter, so the default
  `"on_track"` returned shed and `unknown` rows too
- `merge_with_gps` always read trx01's GPS, whatever the site
- `build_trax_obs`: which source won a duplicated timestamp was not deterministic
  (unstable sort); rows exactly on a chunk boundary were read twice; an unreadable
  uncalibrated window aborted its chunk; an empty time range crashed
- `mobile.receptors`: empty dwell / crossing results crashed the builders;
  `build_dwell_receptors` refuses `freq` under an hour (its `r_idx` names the hour)
- `mobile.wyoming.calculate_enhancements`: the default `window=1` was read as 1 ns,
  so every enhancement was zero; a number is now hours (default `"1h"`)

Caches: the `prior` and `prior_error` keys change once (they now include `sites`).
Jacobians cached at a resolution other than 0.05 / 0.1 deg were built on the old
state grid; the new cell check raises on them, and they need
`cache_overwrite=["prior", "prior_error", "forward_operator", "modeldata_mismatch"]`.

## [2026.9.1] - 2026-09-17

### Added

- `InversionConfig.state_grid`: the flux state geometry as a PYSTILT `Grid`
  (same cells as `grid`). The flux Jacobian is now built with
  `JacobianBuilder.build_from_target(config.state_grid, ...)`; the Jacobian
  columns are unchanged.

- `measurements.mobile.receptors`: TRAX STILT receptors, one PYSTILT
  `MultiPointReceptor` per 2-km segment crossing, releasing from every 50-m
  network point of the segment at the median crossing time, so the footprint
  PYSTILT writes is already the 2-km aggregate. Crossings come from the
  on-track GPS fixes of `obs.parquet` (same segment, gap <= 10 min) and must
  cover at least 1 km of track.

- `measurements.mobile.location`: per-minute TRAX location classifier (depot / yard /
  green-line pass-by / route / stopped) from GPS scatter and satellite count, with a
  `powered` flag from the CR1000 battery voltage; `state_intervals`,
  `label_observations`; `mobile.gps.read_horel_cr1000`; packaged shed footprints
  `trax_depots.geojson`.
  Works without a recorded speed (GPGGA-only eras) via a position-derived
  `speed_est`; `read_lin_gps` / `read_trax_gps` pick the GPS source by era.
  Untrusted positions (> 100 m off any track, or near a yard but neither on
  a track nor in the yard buffer) are `unknown`. Output carries `indoor`
  (depot → True, other located states → False, unknown → NA) and `yard_name`.
  Second storage yard: the Midvale Rail Service Center (`MRSC`, trx03's home;
  packaged `mrsc.geojson`, added to `mobile.storage_locations`), with shed
  footprints for both yards in `trax_depots.geojson`
- `build_trax_obs` merges every GPS fix and tags each observation with `state`,
  `indoor`, `yard_name` (`label_trax_location`); `load_trax_obs(location=...)`
  filters at load time (`LOCATION_SETS`: `on_track` default, `outdoor`, `all`).
  Caches without a `state` column must be rebuilt

### Fixed

- `get_mdm_comp_configs` rejects unknown MDM component names instead of
  ignoring them silently (a retired key such as `transport_pbl` left the
  defaults, and the MDM cache hash, unchanged)
- Inversion cache tests updated for the `.fips/<version tag>/` layout; the
  MDM-hash test used the retired `transport_pbl` key
- `merge_with_gps` duplicated observations where TRAX lines share track (the
  downtown trunk): the per-line route buffers are now dissolved before the
  spatial join (`filter_near_routes`). The legacy `data/trax/data.parquet` was
  built with the same join and carries those duplicates

- `slv.inversion.config.stilt_project_dir()`: the production PYSTILT project as a
  `Path` (`$SLV_STILT_DIR`, else `DEFAULT_STILT_PROJECT`), for scripts

### Changed

- `slv.measurements.mobile` is now a subpackage (`network`, `gps`, `location`,
  `calibration`, `obs`, `transects`); its `__init__` re-exports the public API so
  existing `from slv.measurements.mobile import ...` lines keep working.
  `slv.measurements.trax_location` was folded into `mobile.location` (classifier)
  and `mobile.gps` (readers); import from `slv.measurements.mobile` instead.
  `slv.measurements.wyoming` (the Wyoming mobile lab) moved to
  `slv.measurements.mobile.wyoming`. Both yard polygons are now packaged
  (`jrrsc.geojson` copied from the group spatial dir, `mrsc.geojson`); the depot
  footprint column is `name` (was `site`)
- `mobile.gps` readers are thin uataq wrappers (`raw` level for the horel logger,
  `qaqc` for the lin GPS) instead of reading the h5/CSV files directly; output
  is unchanged (verified bit-identical on Mar 2025 and Jun 2016)
- `load_trax_obs` no longer drops yard-parked data at build time; the old
  route-buffer + storage-polygon behaviour is the `on_track` default (now also
  keeps pass-bys at the yards and drops shed multipath ejecta)

## [2026.9.0] - 2026-09-02

### Added

- EDGAR prior option for the inversion pipeline (`load_edgar_prior`)
- Sub-hour temporal-representativeness MDM component (supersedes the spike filter)
- Footprint-strength scaling for the multiplicative transport MDM component
- Multiplicative (enhancement-scaled) transport MDM component
- `CITATION.cff` and `.zenodo.json` citation metadata; releases are archived
  (and DOI-minted) on Zenodo

## [2026.2.0] - 2026-02

Baseline of the production WBB CH4 inversion configuration (pre-changelog).
