# SLV CH4 Point Sources
from functools import lru_cache
from importlib.resources import files

import pandas as pd

from lair.geo import PC


@lru_cache(maxsize=1)
def _load_points() -> pd.DataFrame:
    """Load the bundled CH4 point-sources CSV (cached after first read)."""
    csv_path = files(__package__).joinpath("ch4_point_sources.csv")
    with csv_path.open("r") as f:
        return pd.read_csv(f)

markers = {
    'cng': '2',
    'industrial': 'X',
    'landfill': 's',
    'powerplant': 'D',
    'unknown': 'p',
    'refinery': '^',
    'wastewater': '+'
}


def plot_point_sources(kind, ax, color='black', **kwargs):
    """Plot point sources of a given kind on an axis."""
    points = _load_points()
    for index, row in points[points['category'] == kind].iterrows():
        ax.scatter(row['longitude'], row['latitude'], transform=PC,
                   label=row['name'], c=color,
                   marker=markers.get(kind), **kwargs)

    return ax
