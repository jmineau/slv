"""Tests for UATAQCH4, with the UATAQ reader monkeypatched."""

import pandas as pd
import pytest

from slv.measurements import background
from slv.measurements.background import UATAQCH4


@pytest.fixture
def fake_uataq(monkeypatch):
    def fake_get_obs(site, pollutants):
        index = pd.date_range("2024-01-01", periods=48 * 6, freq="10min")
        return pd.DataFrame({"CH4d_ppm_cal": 2.0}, index=index)

    monkeypatch.setattr(background.uataq, "get_obs", fake_get_obs)


def test_site_returns_hourly_series(fake_uataq):
    data = UATAQCH4()["hdp"]
    assert isinstance(data, pd.Series)
    assert data.name == "CH4"
    assert len(data) == 48


def test_base_method(fake_uataq):
    data = UATAQCH4()["hdp_base"]
    assert isinstance(data, pd.Series)
    assert (data == 2.0).all()


@pytest.mark.parametrize("key", ["hdp_bse", "hdp_base_x"])
def test_unknown_method_raises(fake_uataq, key):
    with pytest.raises(ValueError, match="Unknown method"):
        UATAQCH4()[key]


# --------------------------------------------------------------------------- GMLDiscrete

GML_FILE = "ch4_mbo_surface-pfp_1_ccgg_event.txt"


def write_gml(gml_dir, value):
    d = gml_dir / "ch4" / "pfp"
    d.mkdir(parents=True, exist_ok=True)
    (d / GML_FILE).write_text(
        "# synthetic NOAA GML event file\n"
        "site_code datetime value qcflag latitude longitude\n"
        f"MBO 2024-06-01T12:00:00Z {value} ... 43.98 -121.69\n"
        f"MBO 2024-06-02T12:00:00Z {value} .X. 43.98 -121.69\n"  # rejected flag
    )


@pytest.fixture
def gml_dirs(tmp_path, monkeypatch):
    """A group copy and a user cache, and a download that writes into its target."""
    group, user = tmp_path / "group_gml", tmp_path / "user"
    monkeypatch.setenv("LAIR_GML_DIR", str(group))  # lair main has no GML_DIR
    monkeypatch.delattr(background.noaa, "GML_DIR", raising=False)
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(user))
    downloads = []

    def fake_download(self):
        downloads.append(self.gml_dir)
        write_gml(self.gml_dir, 3000.0)

    monkeypatch.setattr(background.noaa.GMLData, "download", fake_download)
    return group, user / "gml", downloads


def gml(**kwargs):
    return background.GMLDiscrete("ch4", "mbo", sample_type="pfp", **kwargs)


def test_gml_reads_the_group_copy_without_downloading(gml_dirs):
    group, _, downloads = gml_dirs
    write_gml(group, 1950.0)
    g = gml()
    assert downloads == [] and g.gml_dir == group
    assert g.data.tolist() == [1950.0]  # the flagged sample is dropped


def test_gml_user_cache_wins_over_the_group_copy(gml_dirs):
    group, user, downloads = gml_dirs
    write_gml(group, 1950.0)
    write_gml(user, 1960.0)
    assert gml().data.tolist() == [1960.0] and downloads == []


def test_gml_downloads_into_the_user_cache_never_the_group(gml_dirs):
    group, user, downloads = gml_dirs
    assert gml().data.tolist() == [3000.0]
    assert downloads == [user] and not group.exists()
    write_gml(group, 1950.0)
    gml(refresh=True)  # refresh re-downloads, again into the user cache
    assert downloads == [user, user]


def test_gml_download_failure_says_to_fetch_on_a_login_node(gml_dirs, monkeypatch):
    def offline(self):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(background.noaa.GMLData, "download", offline)
    with pytest.raises(RuntimeError, match="login node"):
        gml()


def test_lair_gml_dir_falls_back_to_the_old_built_in(monkeypatch, tmp_path):
    monkeypatch.delenv("LAIR_GML_DIR", raising=False)
    monkeypatch.setattr(background.noaa, "GML_DIR", tmp_path, raising=False)
    assert background.lair_gml_dir() == tmp_path
    monkeypatch.delattr(background.noaa, "GML_DIR")
    assert background.lair_gml_dir() is None


def test_gml_without_a_user_dir_and_no_group_copy(gml_dirs, monkeypatch):
    monkeypatch.delenv("SLV_USER_DATA_DIR")
    with pytest.raises(OSError, match="SLV_USER_DATA_DIR"):
        gml()
