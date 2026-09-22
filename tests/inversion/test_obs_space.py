"""Tests for the pipeline's obs-space steps: get_obs, filter_state_space and
aggregate_obs_space."""

import numpy as np
import pandas as pd
from fips import Block, CovarianceMatrix, ForwardOperator, MatrixBlock, Vector

from slv.inversion import pipelines
from slv.inversion.config import InversionConfig
from slv.inversion.pipelines import SLVMethaneInversion


def pipeline(**kwargs):
    """A pipeline without fips' __init__ (no data access)."""
    obj = object.__new__(SLVMethaneInversion)
    obj.config = InversionConfig(tstart="2020-01-01", tend="2020-03-01", **kwargs)
    return obj


def obs_vector(times, values):
    index = pd.MultiIndex.from_arrays(
        [["wbb"] * len(times), pd.to_datetime(times)],
        names=["obs_location", "obs_time"],
    )
    return Vector(
        name="obs",
        data=Block(name="concentration", data=pd.Series(values, index=index)),
    )


def _values(vector, level):
    series = vector.to_series()
    return dict(zip(series.index.get_level_values(level), series, strict=True))


def test_get_obs_passes_the_config_to_the_loader(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        # two obs: fips' Block squeezes a one-element Series to a scalar
        index = pd.MultiIndex.from_arrays(
            [["wbb"] * 2, pd.to_datetime(["2020-01-05 20:00", "2020-01-06 20:00"])],
            names=["obs_location", "obs_time"],
        )
        return pd.DataFrame({"CH4": [2.0, 2.5]}, index=index)  # as get_slv_observations

    monkeypatch.setattr(pipelines, "get_slv_observations", fake)
    p = pipeline(
        sites=["wbb", "hdp"],
        subset_hours=[13, 14],
        utc_offset=-6,
        filter_pcaps=False,
        mobile_obs="receptor_obs.parquet",
    )
    obs = p.get_obs()

    assert obs.to_series().tolist() == [2.0, 2.5]
    assert seen["sites"] == ["wbb", "hdp"]
    assert seen["subset_hours"] == [13, 14]
    assert seen["utc_offset"] == -6
    assert seen["filter_pcaps"] is False
    assert seen["mobile_obs"] == "receptor_obs.parquet"
    assert seen["time_range"] == (
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-03-01"),
    )


def _state():
    obs = obs_vector(
        [
            "2019-12-31 20:00",  # before tstart
            "2020-01-05 20:00",
            "2020-01-06 20:00",
            "2020-02-05 20:00",  # the only obs in February
            "2020-03-01 20:00",  # at tend (exclusive)
        ],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
    index = pd.MultiIndex.from_product(
        [
            pd.DatetimeIndex(["2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01"]),
            [40.5],
            [-112.0],
        ],
        names=["time", "lat", "lon"],
    )
    prior = Vector(
        name="prior",
        data=Block(name="flux", data=pd.Series([9.0, 1.0, 2.0, 3.0], index=index)),
    )
    return obs, prior


def test_filter_state_space_trims_to_the_time_range():
    p = pipeline(min_obs_per_interval=1)
    obs, prior = p.filter_state_space(*_state())
    assert sorted(_values(obs, "obs_time").values()) == [2.0, 3.0, 4.0]
    assert _values(prior, "time") == {
        pd.Timestamp("2020-01-01"): 1.0,
        pd.Timestamp("2020-02-01"): 2.0,
    }


def test_filter_state_space_drops_obs_of_sparse_intervals():
    p = pipeline(min_obs_per_interval=2)
    obs, prior = p.filter_state_space(*_state())
    # February has one obs: dropped; its flux stays in the state, held by the prior
    assert sorted(_values(obs, "obs_time").values()) == [2.0, 3.0]
    assert len(prior.to_series()) == 2


def _obs_space_inputs():
    obs = obs_vector(
        [
            "2020-01-05 19:00",
            "2020-01-05 21:00",
            "2020-01-06 19:00",
            "2020-01-06 21:00",
        ],
        [2.0, 2.2, 3.0, 3.4],
    )
    index = obs["concentration"].index
    columns = pd.MultiIndex.from_product(
        [[-112.0], [40.5], pd.DatetimeIndex(["2020-01-01"])],
        names=["lon", "lat", "time"],
    )
    H = pd.DataFrame([[1.0], [3.0], [5.0], [7.0]], index=index, columns=columns)
    forward_operator = ForwardOperator(
        MatrixBlock(H, row_block="concentration", col_block="flux")
    )
    mdm = CovarianceMatrix(
        name="mdm",
        data=MatrixBlock(
            pd.DataFrame(np.eye(4), index, index),
            row_block="concentration",
            col_block="concentration",
        ),
    )
    # no background for the 2020-01-06 19:00 obs
    constant = Vector(
        name="background",
        data=Block(
            name="concentration",
            data=pd.Series([1.9, 2.0, 2.1], index=index[[0, 1, 3]]),
        ),
    )
    return obs, forward_operator, mdm, constant


def test_aggregate_obs_space_drops_obs_without_background():
    p = pipeline()  # aggregate_obs=False
    obs, H, S, c = p.aggregate_obs_space(*_obs_space_inputs())
    assert sorted(obs.to_series().tolist()) == [2.0, 2.2, 3.4]
    assert H.data.shape[0] == 4  # the aggregator is off: the rest is untouched


def test_aggregate_obs_space_averages_to_days():
    p = pipeline(aggregate_obs="1D")
    obs, H, S, c = p.aggregate_obs_space(*_obs_space_inputs())

    days = [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06")]
    assert list(obs.to_series().index.get_level_values("obs_time")) == days
    np.testing.assert_allclose(obs.to_series(), [2.1, 3.4])  # the gap obs left out
    np.testing.assert_allclose(H.data.to_numpy().ravel(), [2.0, 7.0])
    np.testing.assert_allclose(np.diag(S.data.to_numpy()), [0.5, 1.0])  # var / n
    np.testing.assert_allclose(c.to_series(), [1.95, 2.1])
