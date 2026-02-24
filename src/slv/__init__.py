"""Salt Lake Valley py

Salt Lake Valley python modules
"""

import os
from pathlib import Path

__version__ = "2026.2.0"
__author__ = "James Mineau"
__email__ = "James.Mineau@utah.edu"


def get_data_dir(env_var: str) -> Path:
    """Return the directory stored in *env_var*, raising a clear error if unset.

    Set environment variables in your shell profile, e.g.::

        export SLV_DAQ_DIR=/path/to/SLV/data/DAQ/processed
    """
    value = os.environ.get(env_var)
    if value is None:
        raise EnvironmentError(
            f"Environment variable '{env_var}' is not set. "
            f"Please set it to the appropriate data directory before using this feature."
        )
    return Path(value)
