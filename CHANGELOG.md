# Changelog

All notable changes to `slv` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are calendar-based (YYYY.M.PATCH).

## [Unreleased]

### Added

- `InversionConfig.state_grid`: the flux state geometry as a PYSTILT `Grid`
  (same cells as `grid`). The flux Jacobian is now built with
  `JacobianBuilder.build_from_target(config.state_grid, ...)`; the Jacobian
  columns are unchanged.

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
