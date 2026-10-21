"""Tests for the inversion background: mobile (TRAX) sites, and the CT-STILT CSV in the cache key."""

import os

import numpy as np
import pandas as pd
import pytest

from slv.inversion import background as bg
from slv.measurements.sites import load_site_config

SITE_CONFIG = load_site_config()
HOURS = pd.date_range("2024-06-01", periods=24 * 10, freq="h")


@pytest.fixture
def loaded(monkeypatch):
    """Fake hourly tower data; records which sites were loaded."""
    calls = []

    def fake_load(pollutants, sites, **kwargs):
        calls.append(list(sites))
        rng = np.random.default_rng(0)
        return pd.concat(
            pd.DataFrame(
                {"Time_UTC": HOURS, "site": s, "CH4": 2.0 + rng.random(len(HOURS))}
            )
            for s in sites
        )

    monkeypatch.setattr(bg, "load_concentrations", fake_load)
    monkeypatch.setattr(bg, "aggregate_obs", lambda data, **kwargs: data)
    return calls


def rolling(sites, **kwargs):
    return bg.get_rolling_background(
        sites=sites, site_config=SITE_CONFIG, time_range=(HOURS[0], HOURS[-1]), **kwargs
    )


class TestRollingBackgroundSites:
    def test_mobile_sites_are_not_loaded(self, loaded):
        # trx01's per-grid-point rows used to crash the (obs_time, site) unstack
        rolling(["wbb", "trx01"])
        assert loaded == [["wbb"]]

    def test_background_sites_override(self, loaded):
        rolling(["trx01"], background_sites=["wbb", "hw"])
        assert loaded == [["wbb", "hw"]]

    def test_no_stationary_site_raises(self, loaded):
        with pytest.raises(ValueError, match="background_sites"):
            rolling(["trx01"])


class TestBackgroundAtObsTimes:
    def test_obs_take_the_background_of_their_hour(self, monkeypatch):
        hourly = pd.Series(np.arange(len(HOURS), dtype=float), index=HOURS)
        monkeypatch.setattr(bg, "get_rolling_background", lambda **kwargs: hourly)
        obs_times = pd.DatetimeIndex(["2024-06-02 20:00", "2024-06-02 20:13"])

        out = bg.get_slv_background(
            "rolling",
            obs_times=obs_times,
            sites=["wbb", "trx01"],
            site_config=SITE_CONFIG,
            time_range=(HOURS[0], HOURS[-1]),
        )

        # a receptor released at 20:13 used to find no background and be dropped
        expected = hourly[pd.Timestamp("2024-06-02 20:00")]
        assert out.tolist() == [expected, expected]
        pd.testing.assert_index_equal(out.index, obs_times, check_names=False)
        assert out.index.name == "obs_time"


def test_ct_stilt_background_is_joined_by_utc_date(tmp_path):
    csv = tmp_path / "ct.csv"
    pd.DataFrame(
        {
            "obs_time": ["2024-06-01 00:00:00+00:00", "2024-06-02 00:00:00+00:00"],
            "ct_ch4_ppm": [1.95, 1.97],
        }
    ).to_csv(csv, index=False)
    obs_times = pd.DatetimeIndex(
        ["2024-06-01 19:00", "2024-06-01 21:30", "2024-06-02 20:00", "2024-06-03 20:00"]
    )
    out = bg.get_slv_background(
        "ct_stilt",
        obs_times=obs_times,
        sites=["wbb"],
        site_config=SITE_CONFIG,
        time_range=(obs_times[0], obs_times[-1]),
        csv_path=csv,
    )
    assert out.index.name == "obs_time" and out.name == "concentration"
    np.testing.assert_allclose(out.iloc[:3], [1.95, 1.95, 1.97])
    assert np.isnan(out.iloc[3])  # a date outside the product


class _FakeGML:
    made = []

    def __init__(self, specie, site, sample_type):
        _FakeGML.made.append((site, sample_type))

    def thoning_curve(self, smooth_time, **kwargs):
        return pd.Series(1900.0, index=pd.DatetimeIndex(smooth_time))  # ppb


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ("mbo", "pfp")),
        ({"site": "uta"}, ("uta", "flask")),
        ({"site": "uta", "sample_type": "pfp"}, ("uta", "pfp")),
    ],
)
def test_gml_background_in_ppm(monkeypatch, kwargs, expected):
    _FakeGML.made.clear()
    monkeypatch.setattr(bg, "GMLDiscrete", _FakeGML)
    obs_times = pd.DatetimeIndex(["2024-06-01 20:00"])
    out = bg.get_gml_background(obs_times=obs_times, **kwargs)
    assert _FakeGML.made == [expected]
    assert out.tolist() == [1.9] and out.index.name == "obs_time"


def test_unsupported_backgrounds_raise(monkeypatch):
    monkeypatch.setattr(bg, "GMLDiscrete", _FakeGML)
    with pytest.raises(ValueError, match="GML background"):
        bg.get_gml_background(obs_times=[], site="brw")
    with pytest.raises(ValueError, match="Unsupported background"):
        bg.get_slv_background(
            "odiac", [], sites=["wbb"], site_config=SITE_CONFIG, time_range=None
        )


# --------------------------------------------------------------------------- CT-STILT CSV


def _write_ct(csv, values, mtime_ns):
    days = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
    pd.DataFrame(
        {"obs_time": days.strftime("%Y-%m-%d %H:%M:%S%z"), "ct_ch4_ppm": values}
    ).to_csv(csv, index=False)
    os.utime(
        csv, ns=(mtime_ns, mtime_ns)
    )  # explicit: some filesystems keep whole seconds


def _ct_config(csv, cache):
    from slv.inversion import InversionConfig

    return InversionConfig(
        tstart="2020-01-01",
        tend="2020-02-01",
        background="ct_stilt",
        background_kwargs={"csv_path": str(csv)},
        cache=str(cache),
        plot_inputs=False,
        plot_results=False,
    )


def test_background_kwargs_key_fingerprints_the_csv(tmp_path):
    csv = tmp_path / "ct.csv"
    _write_ct(csv, [1.90, 1.91], 1_000_000_000_000_000_000)
    cfg = _ct_config(csv, tmp_path)
    key = cfg.background_kwargs_key
    assert key["csv_path"].startswith(str(csv.resolve()))
    _write_ct(csv, [1.80, 1.81], 1_000_000_001_000_000_000)  # same size, rewritten
    assert cfg.background_kwargs_key != key
    assert cfg.background_kwargs == {"csv_path": str(csv)}  # the config is left alone


def test_background_kwargs_key_keeps_other_kwargs():
    from slv.inversion import InversionConfig

    kwargs = {"baseline_window": "14d", "background_sites": ["wbb"]}
    assert InversionConfig(background_kwargs=kwargs).background_kwargs_key == kwargs


def test_constant_and_mdm_key_on_the_csv_fingerprint_not_the_path():
    from slv.inversion.cache import DEFAULT_COMPONENT_DEPS

    for comp in ("constant", "modeldata_mismatch"):
        assert "background_kwargs_key" in DEFAULT_COMPONENT_DEPS[comp], comp
        assert "background_kwargs" not in DEFAULT_COMPONENT_DEPS[comp], comp


def test_ct_stilt_background_rebuilt_when_the_csv_is_rewritten(tmp_path):
    # issue #30: the cached background was read after its CSV was rewritten in place
    from fips import Block, Vector

    from slv.inversion import SLVMethaneInversion

    csv = tmp_path / "ct.csv"
    idx = pd.MultiIndex.from_arrays(
        [["wbb"] * 3, pd.date_range("2020-01-01 20:00", periods=3, freq="D")],
        names=["obs_location", "obs_time"],
    )
    obs = Vector(
        name="obs",
        data=Block(pd.Series([2.0, 2.1, 2.2], index=idx), name="concentration"),
    )

    def background():
        inv = object.__new__(SLVMethaneInversion)
        inv.config = _ct_config(csv, tmp_path / "cache")
        return inv.get_constant(obs).to_series().round(3).to_list()

    _write_ct(csv, [1.90, 1.91, 1.92], 1_000_000_000_000_000_000)
    assert background() == [1.9, 1.91, 1.92]
    assert background() == [1.9, 1.91, 1.92]  # unchanged file: read from the cache
    _write_ct(csv, [1.80, 1.81, 1.82], 1_000_000_001_000_000_000)
    assert background() == [1.8, 1.81, 1.82]
    assert len(list((tmp_path / "cache").glob(".fips/*/constant/*.pkl"))) == 2
