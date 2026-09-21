"""Tests for slv.measurements.mobile.receptor_obs: observations paired to STILT receptors."""

import numpy as np
import pandas as pd
import pytest

from slv.measurements.mobile.receptor_obs import (
    inlet_lag_seconds,
    trax_receptor_observations,
)
from slv.measurements.mobile.receptors import (
    build_dwell_receptors,
    build_trax_receptors,
    find_dwells,
    find_segment_crossings,
)
from tests.measurements.mobile.test_receptors import _fixes, _network, _parked

LAG_TABLE = pd.DataFrame(
    {
        "start": ["2024-01-01", "2024-07-01"],
        "end": ["2024-05-31", "2024-12-31"],
        "lag_s": [10.0, 20.0],
    }
)


# --------------------------------------------------------------------------- inlet lag


def test_lag_none_and_constant():
    t = pd.date_range("2024-02-01", periods=3, freq="h")
    s, src = inlet_lag_seconds(t, None)
    assert (s == 0).all() and (src == "none").all()
    s, src = inlet_lag_seconds(t, 7)
    assert (s == 7).all() and (src == "constant").all()


def test_lag_table_epoch_nearest_and_inclusive_end():
    t = pd.to_datetime(
        [
            "2024-03-01 00:00",
            "2024-05-31 23:00",
            "2024-06-10 00:00",
            "2024-06-28 00:00",
            "2024-09-01 00:00",
        ]
    )
    s, src = inlet_lag_seconds(t, LAG_TABLE)
    assert s[0] == 10 and src[0] == "epoch"
    assert s[1] == 10 and src[1] == "epoch"  # end date is inclusive
    assert s[2] == 10 and src[2] == "nearest"  # gap, closer to the first epoch
    assert s[3] == 20 and src[3] == "nearest"  # gap, closer to the second
    assert s[4] == 20 and src[4] == "epoch"


# --------------------------------------------------------------------------- pairing


def _write_crossing_receptors(tmp_path, fixes, pts):
    cr = find_segment_crossings(fixes, pts)
    rec = build_trax_receptors(cr, pts)
    f = tmp_path / "receptors.csv"
    rec.to_csv(f, index=False)
    return f, cr


def _ch4_from_fixes(fixes, value, state="route"):
    """A CH4 record on the fix timestamps; ``value`` is a scalar or a function of time."""
    t = pd.DatetimeIndex(fixes.Time_UTC)
    ch4 = value(t) if callable(value) else np.full(len(t), float(value))
    return pd.DataFrame(
        {
            "Time_UTC": t,
            "CH4_ppm": ch4,
            "state": state,
            "cal_source": "pipeline",
            "low_pressure": False,
        }
    )


def _ch4_grid(fixes, value, pad_s=120, state="route"):
    """A CH4 record on a continuous 1-s grid, as the analyzer logs it -- through gaps in the
    GPS and past the last fix, so a lag-shifted window always has samples to average."""
    t = pd.date_range(
        pd.Timestamp(fixes.Time_UTC.min()),
        pd.Timestamp(fixes.Time_UTC.max()) + pd.Timedelta(seconds=pad_s),
        freq="1s",
    )
    ch4 = value(t) if callable(value) else np.full(len(t), float(value))
    return pd.DataFrame(
        {
            "Time_UTC": t,
            "CH4_ppm": ch4,
            "state": state,
            "cal_source": "pipeline",
            "low_pressure": False,
        }
    )


def test_one_observation_per_receptor_keyed_like_the_simulations(tmp_path):
    import stilt

    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    obs = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), _ch4_from_fixes(fixes, 2.0)
    )

    recs = stilt.read_receptors(f)
    keys = {(str(r.location_id), pd.Timestamp(r.time)) for r in recs}
    assert set(obs.index) == keys  # the inversion join key, exactly
    assert len(obs) == len(recs)
    assert obs.index.names == ["obs_location", "obs_time"]
    assert (obs.kind == "crossing").all()
    assert np.allclose(obs.CH4, 2.0)
    assert (
        obs.index.get_level_values("obs_time").tz is None
    )  # naive UTC, like receptor.time


def _ramp(t0):
    """CH4 rising 1e-4 ppm per second from t0, so a window's mean is its midpoint value."""

    def ramp(t):
        return 2.0 + (t - t0).total_seconds().to_numpy() * 1e-4

    return ramp


def test_crossing_mean_is_over_its_own_window(tmp_path):
    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    # CH4 rises linearly with time, so each crossing's mean is its window's midpoint value
    t0 = pd.Timestamp(fixes.Time_UTC.iloc[0])
    ramp = _ramp(t0)
    obs = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), _ch4_from_fixes(fixes, ramp)
    )
    c = cr.set_index(cr.crossing.astype(str))
    for r_idx, row in obs.set_index("r_idx").iterrows():
        mid = (
            c.loc[r_idx, "t_start"]
            + (c.loc[r_idx, "t_end"] - c.loc[r_idx, "t_start"]) / 2
        )
        assert (
            pytest.approx(2.0 + (mid - t0).total_seconds() * 1e-4, abs=2e-4) == row.CH4
        )


def test_lag_shifts_the_window_later(tmp_path):
    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    t0 = pd.Timestamp(fixes.Time_UTC.iloc[0])
    ramp = _ramp(t0)
    ch4 = _ch4_grid(fixes, ramp)
    base = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), ch4, lag=None
    )
    lagged = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), ch4, lag=30
    )
    j = base.join(lagged, rsuffix="_lag")
    # on a rising ramp, reading 30 s later gives +30 s * 1e-4 ppm
    assert np.allclose(j.CH4_lag - j.CH4, 30 * 1e-4, atol=2e-4)
    assert (lagged.lag_s == 30).all() and (lagged.lag_source == "constant").all()


def test_indoor_and_off_track_samples_are_excluded(tmp_path):
    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    ch4 = _ch4_from_fixes(fixes, 2.0)
    # corrupt every other sample, but mark it as depot air: it must not reach the mean
    ch4.loc[ch4.index[::2], ["CH4_ppm", "state"]] = [99.0, "depot"]
    obs = trax_receptor_observations(f, cr, pd.DataFrame(columns=["dwell"]), ch4)
    assert np.allclose(obs.CH4, 2.0)


def test_uncalibrated_switch_and_fraction(tmp_path):
    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    ch4 = _ch4_from_fixes(fixes, 2.0)
    ch4.loc[ch4.index[::4], "cal_source"] = "uncalibrated"
    keep = trax_receptor_observations(f, cr, pd.DataFrame(columns=["dwell"]), ch4)
    assert keep.frac_uncalibrated.between(0.2, 0.3).all()
    drop = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), ch4, include_uncalibrated=False
    )
    assert (drop.frac_uncalibrated == 0).all()
    assert (drop.n < keep.n).all()


def test_min_samples_drops_thin_windows(tmp_path):
    pts = _network()
    fixes = _fixes()
    f, cr = _write_crossing_receptors(tmp_path, fixes, pts)
    ch4 = _ch4_from_fixes(fixes, 2.0)
    few = trax_receptor_observations(
        f, cr, pd.DataFrame(columns=["dwell"]), ch4, min_samples=10_000
    )
    assert few.empty


def test_dwell_observation_covers_its_hour(tmp_path):
    fixes = _parked(minutes=240, t0="2024-06-01 16:00:00")  # parked 16:00-20:00 UTC
    d = find_dwells(fixes, min_duration="30min")
    rec = build_dwell_receptors(fixes, d, min_minutes=30)
    f = tmp_path / "receptors.csv"
    rec.to_csv(f, index=False)
    # CH4 = hour of day, so each dwell-hour's mean identifies which hour it averaged
    ch4 = _ch4_from_fixes(
        fixes, lambda t: t.hour.to_numpy().astype(float), state="yard"
    )
    obs = trax_receptor_observations(f, pd.DataFrame(columns=["crossing"]), d, ch4)
    assert (obs.kind == "dwell").all()
    assert len(obs) == len(rec.r_idx.unique())
    # every dwell observation averages one clock hour, and it is that receptor's hour
    for r_idx, row in obs.set_index("r_idx").iterrows():
        assert float(r_idx[-2:]) == row.CH4
