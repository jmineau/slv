# Changelog

All notable changes to `slv` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions are calendar-based (YYYY.M.PATCH).

## [Unreleased]

### Added

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

### Changed

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
