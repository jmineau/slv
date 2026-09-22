"""Tests for the TRAX GPS readers and the GPS merge, with uataq reads stubbed."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import uataq
from shapely.geometry import LineString, box

from slv.measurements.mobile import gps
from slv.measurements.mobile.gps import HOREL_POST_PILOT
from slv.measurements.mobile.network import UTM12


def _raise_reader_error(*args, **kwargs):
    raise uataq.errors.ReaderError("no files")


def _frame(times, **columns):
    return pd.DataFrame(columns, index=pd.DatetimeIndex(times, name="Time_UTC"))


# --------------------------------------------------------------------------- lin


def test_read_lin_gps_renames_drops_flags_and_sorts(monkeypatch):
    raw = _frame(
        ["2016-01-01 00:00:02", "2016-01-01 00:00:00", "2016-01-01 00:00:01"],
        Latitude_deg=[40.2, 40.0, 40.1],
        N_Satellites=pd.array(["9", "7", "8"], dtype="string[pyarrow]"),
        Speed_m_s=pd.array([None] * 3, dtype="string[pyarrow]"),  # GPGGA era
        QAQC_Flag=[0, 20, -21],  # -21 bad fix quality: dropped; 20 storage box: kept
    )
    seen = {}

    def fake_read(site, **kwargs):
        seen.update(site=site, **kwargs)
        return {"gps": raw}

    monkeypatch.setattr(uataq, "read_data", fake_read)
    out = gps.read_lin_gps(("2016-01-01", "2016-01-02"), site="trx02")

    assert seen["site"] == "trx02" and seen["lvl"] == "qaqc"
    assert out.index.name == "Time_UTC" and out.index.is_monotonic_increasing
    assert out.Latitude_deg.tolist() == [40.0, 40.2]
    assert "N_Satellites" not in out and out.N_Sat.tolist() == [7.0, 9.0]
    assert out.Speed_m_s.dtype == np.float64 and out.Speed_m_s.isna().all()


def test_read_lin_gps_without_files_is_empty(monkeypatch):
    monkeypatch.setattr(uataq, "read_data", _raise_reader_error)
    assert gps.read_lin_gps(("2016-01-01", "2016-01-02")).empty


# --------------------------------------------------------------------------- horel


def test_read_horel_cr1000_joins_gps_and_logger(monkeypatch):
    t = ["2020-01-01 00:00:05", "2020-01-01 00:00:00", "2020-01-01 00:00:05"]
    gps_raw = _frame(
        t, Latitude_deg=[40.1, 40.0, 40.1], N_Satellites=[9, 8, 9], Instrument_Time=t
    )
    cr1000 = _frame(t[1:2], Battery_Voltage_V=["12.6"])
    seen = {}

    def fake_read(site, **kwargs):
        seen.update(kwargs)
        return {"gps": gps_raw, "cr1000": cr1000}

    monkeypatch.setattr(uataq, "read_data", fake_read)
    out = gps.read_horel_cr1000(("2020-01-01", "2020-01-02"))

    assert seen["group"] == "horel" and seen["lvl"] == "raw"
    assert len(out) == 2 and out.index.is_monotonic_increasing  # duplicate dropped
    assert "Instrument_Time" not in out and out.N_Sat.tolist() == [8.0, 9.0]
    assert out.Battery_Voltage_V.iloc[0] == 12.6 and np.isnan(
        out.Battery_Voltage_V.iloc[1]
    )
    assert out.Speed_m_s.isna().all()  # the pilot logger has no speed: added as NaN


def test_read_horel_cr1000_logger_only_and_empty(monkeypatch):
    cr1000 = _frame(["2020-01-01"], Battery_Voltage_V=[12.0])
    monkeypatch.setattr(uataq, "read_data", lambda site, **kw: {"cr1000": cr1000})
    assert gps.read_horel_cr1000(
        ("2020-01-01", "2020-01-02")
    ).Battery_Voltage_V.tolist() == [12.0]
    monkeypatch.setattr(uataq, "read_data", lambda site, **kw: {})
    assert gps.read_horel_cr1000(("2020-01-01", "2020-01-02")).empty
    monkeypatch.setattr(uataq, "read_data", _raise_reader_error)
    assert gps.read_horel_cr1000(("2020-01-01", "2020-01-02")).empty


# --------------------------------------------------------------------------- era switch


@pytest.fixture
def era_calls(monkeypatch):
    calls = []

    def reader(name):
        def read(time_range, site="trx01"):
            calls.append((name, *time_range))
            return _frame([time_range[0]], source=[name])

        return read

    monkeypatch.setattr(gps, "read_lin_gps", reader("lin"))
    monkeypatch.setattr(gps, "read_horel_cr1000", reader("horel"))
    return calls


def test_trax_gps_spanning_the_switch_reads_both(era_calls):
    start, end = pd.Timestamp("2018-11-01"), pd.Timestamp("2018-12-01")
    out = gps.read_trax_gps((start, end))
    assert era_calls == [
        ("lin", start, HOREL_POST_PILOT),
        ("horel", HOREL_POST_PILOT, end),
    ]
    assert out.source.tolist() == ["lin", "horel"]


@pytest.mark.parametrize(
    ("time_range", "source"),
    [(("2017-01-01", "2017-02-01"), "lin"), (("2020-01-01", "2020-02-01"), "horel")],
)
def test_trax_gps_one_era(era_calls, time_range, source):
    gps.read_trax_gps(time_range)
    assert [c[0] for c in era_calls] == [source]


# --------------------------------------------------------------------------- merge


X0, Y0 = 420000.0, 4508000.0


def test_merge_with_gps_filters_before_merging(monkeypatch):
    # 10 fixes eastward along a track; #0 has an outlier altitude, #9 is 200 m off the
    # track, #4-#5 are inside the storage yard
    x = X0 + 100.0 * np.arange(10)
    y = np.full(10, Y0)
    y[9] += 200
    lon, lat = (
        gpd.GeoSeries(gpd.points_from_xy(x, y), crs=UTM12)
        .to_crs(4326)
        .pipe(lambda s: (s.x.to_numpy(), s.y.to_numpy()))
    )
    t = pd.date_range("2024-06-01 12:00", periods=10, freq="s")
    alt = np.full(10, 1300.0)
    alt[0] = 5000.0
    alt[1] = -100.0
    fixes = _frame(t, Pi_Time=t, Latitude_deg=lat, Longitude_deg=lon, Altitude_msl=alt)
    monkeypatch.setattr(uataq, "read_data", lambda site, **kw: {"gps": fixes})
    merged = {}

    def fake_merge(obs, gps_df, on):
        merged.update(obs=obs, gps=gps_df, on=on)
        return obs

    monkeypatch.setattr(uataq.sites.MobileSite, "merge_gps", staticmethod(fake_merge))
    routes = gpd.GeoDataFrame(
        geometry=[LineString([(X0, Y0), (X0 + 900, Y0)])], crs=UTM12
    )
    yard = gpd.GeoDataFrame(
        geometry=[box(X0 + 350, Y0 - 20, X0 + 550, Y0 + 20)], crs=UTM12
    )
    obs = pd.DataFrame({"Time_UTC": t, "CH4": 2.0})

    gps.merge_with_gps(
        "trx01", "UATAQ", obs, routes=routes, route_buffer=50, storage_polygon=yard
    )

    kept = merged["gps"].index.tolist()
    # #0 / #1 outside the plausible altitudes; #9 off track; #4, #5 in the yard
    assert kept == [t[i] for i in (2, 3, 6, 7, 8)]
    assert "geometry" not in merged["gps"]
    assert merged["on"] == "Pi_Time" and merged["obs"].index.name == "Pi_Time"


def test_altitude_range_keeps_mountains_and_missing_altitudes(monkeypatch):
    t = pd.date_range("2024-06-01 12:00", periods=6, freq="s")
    # valley floor, a canyon road, Guardsman Pass, no altitude, then two junk fixes
    alt = [1300.0, 2600.0, 2980.0, np.nan, 0.0, 12000.0]
    fixes = _frame(
        t, Pi_Time=t, Latitude_deg=40.6, Longitude_deg=-111.6, Altitude_msl=alt
    )
    monkeypatch.setattr(uataq, "read_data", lambda site, **kw: {"gps": fixes})
    seen = {}
    monkeypatch.setattr(
        uataq.sites.MobileSite,
        "merge_gps",
        staticmethod(lambda obs, g, on: seen.setdefault("gps", g)),
    )
    obs = pd.DataFrame({"Time_UTC": t, "CH4": 2.0})
    no_filters = {"routes": False, "storage_polygon": False}

    gps.merge_with_gps("trx01", "UATAQ", obs, **no_filters)
    assert seen.pop("gps").index.tolist() == list(t[:4])

    gps.merge_with_gps("trx01", "UATAQ", obs, altitude_range=None, **no_filters)
    assert len(seen.pop("gps")) == 6


def test_constant_altitude_keeps_every_fix(monkeypatch):
    # the old 1st-99th percentile cut dropped every fix when the altitude never changed
    t = pd.date_range("2024-06-01 12:00", periods=5, freq="s")
    fixes = _frame(
        t, Pi_Time=t, Latitude_deg=40.6, Longitude_deg=-111.9, Altitude_msl=1300.0
    )
    monkeypatch.setattr(uataq, "read_data", lambda site, **kw: {"gps": fixes})
    seen = {}
    monkeypatch.setattr(
        uataq.sites.MobileSite,
        "merge_gps",
        staticmethod(lambda obs, g, on: seen.setdefault("gps", g)),
    )
    obs = pd.DataFrame({"Time_UTC": t, "CH4": 2.0})
    gps.merge_with_gps("trx01", "UATAQ", obs, routes=False, storage_polygon=False)
    assert len(seen["gps"]) == 5
