"""Tests for slv.inversion.provenance and where run and the sweep put it."""

from __future__ import annotations

import json
from importlib.metadata import version

import pandas as pd

from slv.inversion import provenance
from slv.inversion.config import InversionConfig
from slv.inversion.sweep import config_id

VERSIONS = ("slv", "fips", "pystilt", "lair", "uataq")


class TestPackageVersions:
    def test_the_five_packages_in_order(self):
        assert tuple(provenance.package_versions()) == VERSIONS

    def test_versions_are_the_installed_ones(self):
        assert provenance.package_versions() == {p: version(p) for p in VERSIONS}

    def test_a_missing_package_reads_unknown(self, monkeypatch):
        def fake(name):
            if name == "uataq":
                raise provenance.PackageNotFoundError(name)
            return "1.0"

        monkeypatch.setattr(provenance, "version", fake)
        found = provenance.package_versions()
        assert found["uataq"] == "unknown" and found["slv"] == "1.0"

    def test_columns_are_named_for_the_package(self):
        cols = provenance.version_columns()
        assert tuple(cols) == provenance.VERSION_COLUMNS
        assert cols["pystilt_version"] == version("pystilt")


class TestRecord:
    def test_holds_the_versions_and_the_config_as_strings(self):
        rec = provenance.record(InversionConfig())
        assert set(rec) == {*provenance.VERSION_COLUMNS, "config"}
        assert all(isinstance(v, str) for v in rec.values())

    def test_config_has_every_field_but_the_tiler(self):
        config = InversionConfig(prior_base_std=0.02, sites=["wbb", "trx01"])
        fields = json.loads(provenance.record(config)["config"])
        assert fields["prior_base_std"] == 0.02
        assert fields["sites"] == ["wbb", "trx01"]
        assert "tiler" not in fields
        assert "cache" in fields and "stilt_project" in fields

    def test_config_rebuilds_the_config(self):
        days = pd.date_range("2016-01-01", "2016-01-31", freq="D")
        config = InversionConfig(
            tstart="2016-01-01",
            tend="2016-02-01",
            flux_freq="MS",
            bias_freq="D",
            bias_std=pd.Series(0.01, index=days),
            mdm_config={"transport": {"fraction": 0.6}},
            background_kwargs={"a": 1},
        )
        text = provenance.record(config)["config"]
        rebuilt = InversionConfig(**json.loads(text))
        assert provenance.config_json(rebuilt) == text
        assert config_id(rebuilt) == config_id(config)
