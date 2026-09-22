"""Tests for load_transects on synthetic monthly transect files."""

import numpy as np
import pytest
import xarray as xr

from slv.measurements.mobile.transects import load_transects


def write_month(d, line, month, n_transects, fill):
    ds = xr.Dataset(
        {
            "obs": (("transect", "point"), np.full((n_transects, 3), fill)),
            "n": (("transect", "point"), np.ones((n_transects, 3), dtype=int)),
        },
        coords={
            "lat": ("point", [40.70, 40.71, 40.72]),
            "lon": ("point", [-111.9] * 3),
        },
    )
    ds.to_netcdf(d / f"trx01_CH4_{line}_{month}.nc")


@pytest.fixture
def transects_dir(tmp_path):
    write_month(tmp_path, "r", "2019-01", 2, 2.0)
    write_month(tmp_path, "r", "2019-02", 3, 2.1)
    write_month(tmp_path, "g", "2019-01", 4, 2.2)
    return tmp_path


def test_months_are_concatenated_in_order(transects_dir):
    ds = load_transects("r", transects_dir=transects_dir)
    assert ds.sizes == {"transect": 5, "point": 3}
    assert ds.transect.values.tolist() == list(range(5))
    assert ds.month.values.tolist() == ["2019-01"] * 2 + ["2019-02"] * 3
    np.testing.assert_allclose(ds.obs.isel(point=0), [2.0, 2.0, 2.1, 2.1, 2.1])


def test_months_filter(transects_dir):
    ds = load_transects("r", months=["2019-02"], transects_dir=transects_dir)
    assert set(ds.month.values) == {"2019-02"} and ds.sizes["transect"] == 3


def test_default_directory_is_under_the_user_data_dir(transects_dir, monkeypatch):
    user = transects_dir.parent / "user"
    (user / "trax").mkdir(parents=True)
    (user / "trax" / "transects").symlink_to(transects_dir)
    monkeypatch.setenv("SLV_USER_DATA_DIR", str(user))
    assert load_transects("g").sizes["transect"] == 4


def test_missing_files_raise(transects_dir):
    with pytest.raises(FileNotFoundError, match="line 'b'"):
        load_transects("b", transects_dir=transects_dir)
    with pytest.raises(FileNotFoundError):
        load_transects("r", months=["2030-01"], transects_dir=transects_dir)
