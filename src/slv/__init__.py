"""
Salt Lake Valley py

Salt Lake Valley python modules
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path

try:
    __version__ = _version("slv")  # set by setuptools-scm from git tags
except PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0+unknown"
__author__ = "James Mineau"
__email__ = "James.Mineau@utah.edu"


def get_data_dir(env_var: str) -> Path:
    """
    Return the directory stored in *env_var*.

    This calls ``lair.config.get_data_dir`` (imported on the call, so
    ``import slv`` stays light). Set the variable in your shell profile, e.g.::

        export SLV_DAQ_DIR=/path/to/SLV/data/DAQ/processed

    Parameters
    ----------
    env_var : str
        Name of the environment variable, e.g. ``"SLV_DAQ_DIR"``.

    Returns
    -------
    Path
        The directory.

    Raises
    ------
    ValueError
        If the variable is unset or empty.
    """
    from lair.config import get_data_dir as _get_data_dir

    return _get_data_dir(env_var)
