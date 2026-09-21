"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def no_production_stilt_project(tmp_path, monkeypatch):
    """Point InversionConfig.stilt_project at an empty directory.

    Its default is the production PYSTILT project on CHPC; a test that reaches the
    Jacobian build without stubbing it would otherwise scan ~117 k real simulations.
    """
    monkeypatch.setenv("SLV_STILT_DIR", str(tmp_path / "no-stilt-project"))
