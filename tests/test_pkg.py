"""Test basic functionality of slv."""

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
