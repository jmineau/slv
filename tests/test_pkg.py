"""Test basic functionality of slv."""

import pathlib

import pytest

import slv


def test_version():
    """Test that version is defined."""
    assert hasattr(slv, "__version__")
    assert isinstance(slv.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(slv, "__author__")
    assert isinstance(slv.__author__, str)


def test_email():
    """Test that email is defined."""
    assert hasattr(slv, "__email__")
    assert isinstance(slv.__email__, str)


def test_get_data_dir_returns_the_variable_as_a_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SLV_DAQ_DIR", str(tmp_path))
    assert slv.get_data_dir("SLV_DAQ_DIR") == tmp_path
    assert isinstance(slv.get_data_dir("SLV_DAQ_DIR"), pathlib.Path)


@pytest.mark.parametrize("value", [None, ""])
def test_get_data_dir_unset_or_empty_raises_value_error(monkeypatch, value):
    # An empty variable is not the current directory: it counts as unset.
    if value is None:
        monkeypatch.delenv("SLV_DAQ_DIR", raising=False)
    else:
        monkeypatch.setenv("SLV_DAQ_DIR", value)
    with pytest.raises(ValueError, match="SLV_DAQ_DIR"):
        slv.get_data_dir("SLV_DAQ_DIR")
