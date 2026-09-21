"""Mobile-platform measurements: the TRAX light-rail trains and the Wyoming mobile lab.

Submodules, one concern each:

* :mod:`.network` — TRAX lines, staged points, storage yards (JRRSC, MRSC), shed
  footprints, route-buffer filter.
* :mod:`.gps` — GPS readers (lin-group air-trend, horel-group CR1000 logger,
  era-based :func:`read_trax_gps`) and :func:`merge_with_gps`.
* :mod:`.location` — per-minute location classifier: indoor (shed) vs outdoor
  (yard / pass-by / route / stopped), ``state`` / ``indoor`` / ``yard_name``.
* :mod:`.calibration` — ``cal_source`` provenance and the uncalibrated windows.
* :mod:`.obs` — :func:`build_trax_obs` / :func:`load_trax_obs`, the cached CH4 record
  with both tag sets and a load-time ``location`` filter.
* :mod:`.transects` — the archived transect matrices.
* :mod:`.receptors` — STILT receptors: 50-m network points, 2-km segment crossings,
  one PYSTILT multipoint receptor per crossing, and dwell (parked-train) receptors.
* :mod:`.receptor_obs` — the CH4 observation for each of those receptors, keyed
  ``(obs_location, obs_time)`` exactly as the inversion joins them to the footprints.
* :mod:`.wyoming` — the Wyoming mobile lab (Aeris + met readers, enhancement
  ratios, wind-barb map); not re-exported, import from the module.

Every public function and table is re-exported here, so
``from slv.measurements.mobile import X`` works for all of them; the classifier thresholds
(:mod:`.location`) and GPS column/flag constants (:mod:`.gps`) stay in their modules.
"""

from .calibration import (
    CAL_SOURCES,
    filter_cal_source,
    load_trax_uncalibrated_windows,
    select_uncalibrated,
)
from .gps import (
    HOREL_POST_PILOT,
    merge_with_gps,
    read_horel_cr1000,
    read_lin_gps,
    read_trax_gps,
)
from .location import (
    STATES,
    classify_location,
    label_observations,
    location_features,
    state_intervals,
)
from .network import (
    UTM12,
    filter_near_routes,
    get_geodf,
    load_depot_footprint,
    load_storage_polygons,
    load_trax_lines,
    load_trax_points,
    storage_locations,
)
from .obs import (
    LOCATION_SETS,
    LOW_PRESSURE_BAND,
    SLOPE_TOL,
    apply_low_pressure_rule,
    apply_slope_guard,
    build_trax_obs,
    filter_location,
    label_trax_location,
    load_trax_obs,
)
from .receptor_obs import (
    CROSSING_STATES,
    DWELL_STATES,
    inlet_lag_seconds,
    trax_receptor_observations,
)
from .receptors import (
    LINE_LETTERS,
    RECEPTOR_COLUMNS,
    build_dwell_receptors,
    build_trax_receptors,
    filter_receptor_hours,
    find_dwells,
    find_segment_crossings,
    label_dwell_site,
    load_trax_fixes,
    load_trax_network_points,
    release_points,
)
from .transects import TRANSECT_LINES, load_transects

__all__ = [
    "CAL_SOURCES",
    "CROSSING_STATES",
    "DWELL_STATES",
    "HOREL_POST_PILOT",
    "LINE_LETTERS",
    "LOCATION_SETS",
    "LOW_PRESSURE_BAND",
    "RECEPTOR_COLUMNS",
    "SLOPE_TOL",
    "STATES",
    "TRANSECT_LINES",
    "UTM12",
    "apply_low_pressure_rule",
    "apply_slope_guard",
    "build_dwell_receptors",
    "build_trax_obs",
    "build_trax_receptors",
    "classify_location",
    "filter_cal_source",
    "filter_location",
    "filter_near_routes",
    "filter_receptor_hours",
    "find_dwells",
    "find_segment_crossings",
    "get_geodf",
    "inlet_lag_seconds",
    "label_dwell_site",
    "label_observations",
    "label_trax_location",
    "load_depot_footprint",
    "load_storage_polygons",
    "load_transects",
    "load_trax_fixes",
    "load_trax_lines",
    "load_trax_network_points",
    "load_trax_obs",
    "load_trax_points",
    "load_trax_uncalibrated_windows",
    "location_features",
    "merge_with_gps",
    "read_horel_cr1000",
    "read_lin_gps",
    "read_trax_gps",
    "release_points",
    "select_uncalibrated",
    "state_intervals",
    "storage_locations",
    "trax_receptor_observations",
]
