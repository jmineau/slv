"""A made-up inversion small enough to build and solve in a test, with no data access."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fips import Operator, Space

from slv.inversion import mdm, pipelines, priors
from slv.inversion.config import InversionConfig

#: Two months, two by two cells, hourly obs at one tower over three days.
TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
OBS_TIMES = pd.to_datetime(
    [
        "2020-01-05 20:00",
        "2020-01-05 21:00",
        "2020-01-06 20:00",
        "2020-01-06 21:00",
        "2020-02-05 20:00",
        "2020-02-05 21:00",
    ]
)
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * len(OBS_TIMES), OBS_TIMES], names=["obs_location", "obs_time"]
)
#: The obs without a background (2020-01-06 20:00) and the one without a footprint
#: (2020-02-05 21:00).
NO_BACKGROUND = OBS_INDEX[2]
NO_FOOTPRINT = OBS_INDEX[5]


def prior_series() -> pd.Series:
    """The prior as get_slv_prior gives it: (time, lat, lon), float32, float noise."""
    index = pd.MultiIndex.from_product(
        [TIMES, [lat + 1e-9 for lat in LATS], LONS], names=["time", "lat", "lon"]
    )
    values = (np.arange(len(index)) + 1.0) * 0.01
    return pd.Series(values.astype(np.float32), index=index, name="flux")


def jacobian() -> Operator:
    """
    The Jacobian as get_flux_jacobian gives it: receptor rows, (lon, lat, time) columns.

    One obs has no footprint, and one receptor is not an obs.
    """
    extra = pd.MultiIndex.from_arrays(
        [["wbb"], pd.to_datetime(["2020-01-07 20:00"])],
        names=["obs_location", "obs_time"],
    )
    rows = OBS_INDEX.drop(NO_FOOTPRINT).append(extra)
    cols = pd.MultiIndex.from_product([LONS, LATS, TIMES], names=["lon", "lat", "time"])
    rng = np.random.default_rng(0)
    values = rng.random((len(rows), len(cols))) * 0.1
    # each receptor sees only the flux of its own month
    month = rows.get_level_values("obs_time").to_period("M").to_timestamp()
    values[np.asarray(month)[:, None] != cols.get_level_values("time").values] = 0.0
    return Operator(
        sp.csr_array(values),
        Space({"concentration": rows}),
        Space({"flux": cols}),
        name="jacobian",
    )


def observations() -> pd.DataFrame:
    """The obs as get_slv_observations gives them: a CH4 column."""
    return pd.DataFrame(
        {"CH4": 2.0 + 0.01 * np.arange(len(OBS_INDEX))}, index=OBS_INDEX
    )


def background() -> pd.Series:
    """The background as get_slv_background gives it: by obs_time, one time missing."""
    times = OBS_TIMES.drop(NO_BACKGROUND[1])
    return pd.Series(1.95, index=pd.Index(times, name="obs_time"), name="concentration")


@pytest.fixture
def synthetic(monkeypatch):
    """
    Stub the data loaders, and return a function that makes a config for them.

    ``build_problem`` then runs on the made-up inversion above, through every slv step
    and the real model-data mismatch (the within-hour std is zero).
    """
    monkeypatch.setattr(pipelines, "get_slv_observations", lambda **k: observations())
    monkeypatch.setattr(priors, "get_slv_prior", lambda **k: prior_series())
    monkeypatch.setattr(pipelines, "get_slv_background", lambda **k: background())
    monkeypatch.setattr(
        pipelines, "_project_jacobian", lambda config, project, locs: jacobian()
    )
    monkeypatch.setattr(
        mdm, "get_slv_subhour_std", lambda **k: pd.Series(0.0, index=OBS_INDEX)
    )

    def make(**overrides) -> InversionConfig:
        settings = dict(
            sites=["wbb"],
            tstart="2020-01-01",
            tend="2020-03-01",
            flux_freq="MS",
            xmin=-112.05,
            xmax=-111.85,
            ymin=40.45,
            ymax=40.65,
            dx=0.1,
            dy=0.1,
            min_obs_per_interval=1,
            aggregate_obs=False,
            prior_base_std=0.01,
            prior_std_frac=0.5,
            plot_inputs=False,
            plot_results=False,
            plot_diagnostics=False,
        )
        return InversionConfig(**(settings | overrides))

    return make
