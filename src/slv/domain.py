"""Domain definitions for the Salt Lake Valley."""

# --- Spatial domain (SLV inversion grid) ---
XMIN: float = -112.2
XMAX: float = -111.75
YMIN: float = 40.45
YMAX: float = 40.93

BBOX: tuple[float, float, float, float] = (XMIN, YMIN, XMAX, YMAX)  # (W, S, E, N)
EXTENT: tuple[float, float, float, float] = (
    XMIN,
    XMAX,
    YMIN,
    YMAX,
)  # cartopy convention

# Representative CarbonTracker grid point for the SLV
SLV_LON: float = -112.5
SLV_LAT: float = 41.0

# Wider display bounds used for mapping / basemap figures
MAP_BBOX: tuple[float, float, float, float] = (
    -112.25,
    40.4,
    -111.62,
    40.95,
)  # (W, S, E, N)

# --- Utah state bounds (for regional data subsetting) ---
UT_BBOX: tuple[float, float, float, float] = (
    -114.05190414404143,  # W
    36.99790491333913,  # S
    -109.04124972989729,  # E
    42.0019276110661,  # N
)

# --- Time ---
UTC_OFFSET: int = -7  # Mountain Standard Time (UTC-7); note: does not account for DST
