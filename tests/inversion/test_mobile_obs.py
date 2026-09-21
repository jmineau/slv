"""Tests for receptor-paired mobile observations in the inversion (slv.inversion.data)."""

import os
import time

import numpy as np
import pandas as pd

from slv.inversion.config import InversionConfig
from slv.inversion.data import get_slv_observations, load_mobile_obs


def _mobile_obs():
    """Receptor-paired obs: two crossings and a dwell over one day (naive-UTC times)."""
    idx = pd.MultiIndex.from_tuples(
        [
            ("multi_aaaaaaaaaa", pd.Timestamp("2024-06-01 18:30")),  # 11:30 MST
            ("multi_aaaaaaaaaa", pd.Timestamp("2024-06-01 20:10")),  # 13:10 MST
            ("multi_bbbbbbbbbb", pd.Timestamp("2024-06-01 22:45")),  # 15:45 MST
            ("-111.92_40.72_4", pd.Timestamp("2024-06-02 00:20")),  # 17:20 MST
        ],
        names=["obs_location", "obs_time"],
    )
    return pd.DataFrame(
        {
            "CH4": [2.01, 2.02, 2.03, 2.04],
            "n": [18, 20, 19, 300],
            "kind": ["crossing"] * 3 + ["dwell"],
        },
        index=idx,
    )


def test_load_mobile_obs_returns_ch4_on_the_join_key():
    out = load_mobile_obs(
        _mobile_obs(), ("2024-06-01", "2024-06-03"), filter_pcaps=False
    )
    assert list(out.columns) == ["CH4"]
    assert out.index.names == ["obs_location", "obs_time"]
    assert len(out) == 4


def test_load_mobile_obs_time_range_is_half_open():
    out = load_mobile_obs(
        _mobile_obs(), ("2024-06-01 20:10", "2024-06-01 22:45"), filter_pcaps=False
    )
    assert out.index.get_level_values("obs_time").tolist() == [
        pd.Timestamp("2024-06-01 20:10")
    ]


def test_load_mobile_obs_local_hour_window():
    # afternoon 12-16 MST keeps 13:10 and 15:45, drops 11:30 and 17:20
    out = load_mobile_obs(
        _mobile_obs(),
        ("2024-06-01", "2024-06-03"),
        subset_hours=[12, 13, 14, 15, 16],
        filter_pcaps=False,
        utc_offset=-7,
    )
    local = (
        pd.DatetimeIndex(out.index.get_level_values("obs_time")) - pd.Timedelta(hours=7)
    ).hour
    assert sorted(local) == [13, 15]


def test_load_mobile_obs_accepts_tz_aware_range():
    rng = (pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-06-03", tz="UTC"))
    assert len(load_mobile_obs(_mobile_obs(), rng, filter_pcaps=False)) == 4


def test_load_mobile_obs_applies_pcap_filter(monkeypatch):
    import slv.meteorology.pcaps as pcaps

    # pretend 2024-06-01 22:00-23:00 UTC is a PCAP event
    def fake(data, level=None):
        t = data.index
        return data[~((t >= "2024-06-01 22:00") & (t < "2024-06-01 23:00"))]

    monkeypatch.setattr(pcaps, "filter_pcap_events", fake)
    out = load_mobile_obs(
        _mobile_obs(), ("2024-06-01", "2024-06-03"), filter_pcaps=True
    )
    assert pd.Timestamp("2024-06-01 22:45") not in out.index.get_level_values(
        "obs_time"
    )
    assert len(out) == 3


def test_load_mobile_obs_reads_parquet(tmp_path):
    f = tmp_path / "trax_obs.parquet"
    _mobile_obs().to_parquet(f)
    assert (
        len(load_mobile_obs(f, ("2024-06-01", "2024-06-03"), filter_pcaps=False)) == 4
    )


def test_get_slv_observations_uses_mobile_obs_for_trax_only(monkeypatch):
    import slv.inversion.data as data
    from slv.measurements import load_site_config

    def boom(*a, **k):  # the slow GPS-merge path must not run when mobile_obs is given
        raise AssertionError(
            "load_concentrations should not be called for a TRAX-only run"
        )

    monkeypatch.setattr(data, "load_concentrations", boom)
    out = get_slv_observations(
        ["trx01"],
        load_site_config(),
        ("2024-06-01", "2024-06-03"),
        filter_pcaps=False,
        mobile_obs=_mobile_obs(),
    )
    assert set(out.index.get_level_values("obs_location")) == {
        "multi_aaaaaaaaaa",
        "multi_bbbbbbbbbb",
        "-111.92_40.72_4",
    }
    assert np.isclose(out.CH4.sum(), 2.01 + 2.02 + 2.03 + 2.04)


def test_mobile_obs_key_changes_when_the_file_is_rewritten(tmp_path):
    f = tmp_path / "trax_obs.parquet"
    _mobile_obs().to_parquet(f)
    cfg = InversionConfig(mobile_obs=f)
    k1 = cfg.mobile_obs_key
    assert k1 and str(f.resolve()) in k1
    time.sleep(0.01)
    _mobile_obs().assign(CH4=9.9).to_parquet(
        f
    )  # rebuilt in place, e.g. a new inlet lag
    os.utime(f)
    assert cfg.mobile_obs_key != k1
    assert InversionConfig().mobile_obs_key is None


def test_mobile_obs_key_is_a_forward_operator_dependency():
    from slv.inversion.pipelines import DEFAULT_COMPONENT_DEPS

    for comp in ("obs", "forward_operator", "modeldata_mismatch", "constant"):
        assert "mobile_obs_key" in DEFAULT_COMPONENT_DEPS[comp], comp


def test_split_sites():
    from slv.inversion.data import split_sites
    from slv.measurements.sites import load_site_config

    stationary, mobile = split_sites(["wbb", "trx01", "hw"], load_site_config())
    assert stationary == ["wbb", "hw"]
    assert mobile == ["trx01"]


def test_subhour_std_does_not_load_mobile_sites(monkeypatch):
    # mobile obs get 0 anyway; loading trx01 ran the full TRAX GPS merge for nothing
    import slv.inversion.data as data_module
    from slv.measurements.sites import load_site_config

    loaded = []

    def fake_load(pollutants, sites, **kwargs):
        loaded.append(list(sites))
        return pd.DataFrame(
            {
                "Time_UTC": pd.to_datetime(["2024-06-01 20:00", "2024-06-01 20:30"]),
                "site": "wbb",
                "CH4": [2.0, 2.2],
            }
        )

    monkeypatch.setattr(data_module, "load_concentrations", fake_load)
    site_config = load_site_config()
    tr = ("2024-06-01", "2024-06-02")

    out = data_module.get_slv_subhour_std(["wbb", "trx01"], site_config, tr)
    assert loaded == [["wbb"]]
    assert out.loc[("wbb", pd.Timestamp("2024-06-01 20:00"))] > 0

    loaded.clear()
    out = data_module.get_slv_subhour_std(["trx01"], site_config, tr)
    assert loaded == []
    assert out.empty


def test_obs_hour_window_uses_the_configured_utc_offset(monkeypatch):
    # the obs filter used the fixed slv.domain offset whatever config.utc_offset said
    import slv.inversion.data as data_module
    from slv.measurements.sites import load_site_config

    seen = []

    def fake_load(pollutants, sites, **kwargs):
        seen.append(kwargs["utc_offset"])
        return pd.DataFrame(
            {
                "Time_UTC": pd.to_datetime(["2024-06-01 20:00"]),
                "site": ["wbb"],
                "CH4": [2.0],
            }
        )

    monkeypatch.setattr(data_module, "load_concentrations", fake_load)
    monkeypatch.setattr(data_module, "load_trax_points", lambda: None)
    monkeypatch.setattr(data_module, "aggregate_obs", lambda obs, **kwargs: obs)
    site_config = load_site_config()
    tr = ("2024-06-01", "2024-06-02")
    data_module.get_slv_observations(["wbb"], site_config, tr, utc_offset=-6)
    data_module.get_slv_subhour_std(["wbb"], site_config, tr, utc_offset=-6)
    assert seen == [-6, -6]
