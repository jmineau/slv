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
  Untrusted positions (> 100 m off any track, or near the yard but neither on
  the line nor in the yard buffer) are `unknown`

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
