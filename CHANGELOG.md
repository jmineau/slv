# Changelog

All notable changes to `slv` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are calendar-based (YYYY.M.PATCH).

## [Unreleased]

### Changed

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

- `measurements.trax_location`: per-minute TRAX location classifier (depot / yard /
  green-line pass-by / route / stopped) from GPS scatter and satellite count, with a
  `powered` flag from the CR1000 battery voltage; `read_horel_cr1000`,
  `state_intervals`, `label_observations`; packaged `jrrsc_depot.geojson`.
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
