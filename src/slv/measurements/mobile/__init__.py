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
* :mod:`.wyoming` — the Wyoming mobile lab (Aeris + met readers, enhancement
  ratios, wind-barb map); not re-exported, import from the module.

Everything public is re-exported here, so ``from slv.measurements.mobile import X``
works for all of it.
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
    build_trax_obs,
    filter_location,
    label_trax_location,
    load_trax_obs,
)
from .transects import TRANSECT_LINES, load_transects

__all__ = [
    "CAL_SOURCES",
    "HOREL_POST_PILOT",
    "LOCATION_SETS",
    "STATES",
    "TRANSECT_LINES",
    "UTM12",
    "build_trax_obs",
    "classify_location",
    "filter_cal_source",
    "filter_location",
    "filter_near_routes",
    "get_geodf",
    "label_observations",
    "label_trax_location",
    "load_depot_footprint",
    "load_storage_polygons",
    "load_transects",
    "load_trax_lines",
    "load_trax_obs",
    "load_trax_points",
    "load_trax_uncalibrated_windows",
    "location_features",
    "merge_with_gps",
    "read_horel_cr1000",
    "read_lin_gps",
    "read_trax_gps",
    "select_uncalibrated",
    "state_intervals",
    "storage_locations",
]
