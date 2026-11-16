"""
Which code and settings made an inversion result.

:func:`~slv.inversion.pipelines.run` stores :func:`record` as ``post.attrs`` and the
sweep writes :func:`version_columns` into each row of its results CSV, so a result
says which versions of slv and of the packages it computes with produced it. The
versions are read from the installed packages' metadata, not from the source tree:
for an editable install they are the version at the last install, so reinstall after
a new tag. They stay out of the cache keys (``docs/contributing.rst``, "The component
cache").
"""

from __future__ import annotations

import dataclasses
import json
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from slv.inversion.config import InversionConfig

#: The distributions whose versions are recorded: slv and the packages it computes with.
PACKAGES = ("slv", "fips", "pystilt", "lair", "uataq")

#: ``<package>_version``: the column of each package's version in a sweep's results.
VERSION_COLUMNS = tuple(f"{name}_version" for name in PACKAGES)

#: The config fields left out of the record: the tiler is a cartopy object, not a setting.
_UNRECORDED_FIELDS = frozenset({"tiler"})


def _to_json(v: Any) -> Any:
    """
    Recursively convert a value to a JSON-serializable form.

    A Series (a per-interval ``bias_std``) becomes a dict of its values by label, so
    the id sees every value and a grid or results CSV gives it back.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, pd.Series):
        return {str(k): float(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json(i) for i in v]
    if isinstance(v, dict):
        return {k: _to_json(vv) for k, vv in sorted(v.items())}
    return str(v)


def package_versions() -> dict[str, str]:
    """
    Return the installed version of slv, fips, pystilt, lair and uataq.

    Returns
    -------
    dict
        Distribution name to version string, in the order of :data:`PACKAGES`. A
        package that is not installed reads ``"unknown"``.

    Examples
    --------
    >>> from slv.inversion.provenance import package_versions
    >>> list(package_versions())
    ['slv', 'fips', 'pystilt', 'lair', 'uataq']
    """
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def version_columns() -> dict[str, str]:
    """
    Return :func:`package_versions` keyed ``<package>_version``, as the sweep's columns.

    Returns
    -------
    dict
        ``{"slv_version": ..., "fips_version": ..., ...}``.

    Examples
    --------
    >>> from slv.inversion.provenance import version_columns
    >>> list(version_columns())[:2]
    ['slv_version', 'fips_version']
    """
    return dict(zip(VERSION_COLUMNS, package_versions().values(), strict=True))


def config_json(config: InversionConfig) -> str:
    """
    Return every field of a config as a JSON string, except the tiler.

    The same fields the sweep's grid file holds: a dict or list as itself, a per-interval
    ``bias_std`` as a dict by interval start, anything else (a path, a timestamp) as
    its string. ``InversionConfig(**json.loads(...))`` rebuilds the config, as
    :func:`~slv.inversion.sweep.run_sweep_job` does from the grid.

    Parameters
    ----------
    config : InversionConfig
        The config to write.

    Returns
    -------
    str
        The fields as JSON, with sorted keys.
    """
    fields = {
        f.name: getattr(config, f.name)
        for f in dataclasses.fields(config)
        if f.name not in _UNRECORDED_FIELDS
    }
    return json.dumps(_to_json(fields), sort_keys=True)


def record(config: InversionConfig) -> dict[str, str]:
    """
    Return the record of a run: the package versions and the config, all strings.

    Every value is a string, so the dict can be set as the attributes of a netCDF
    file or written as JSON as it is.

    Parameters
    ----------
    config : InversionConfig
        The config of the run.

    Returns
    -------
    dict
        The :func:`version_columns` keys and ``"config"``, the :func:`config_json` text.

    Examples
    --------
    Read the config back from a result's record:

    >>> import json
    >>> from slv.inversion import InversionConfig
    >>> from slv.inversion.provenance import record
    >>> attrs = record(InversionConfig(prior_base_std=0.02))
    >>> sorted(attrs)
    ['config', 'fips_version', 'lair_version', 'pystilt_version', 'slv_version', 'uataq_version']
    >>> json.loads(attrs["config"])["prior_base_std"]
    0.02
    """
    return {**version_columns(), "config": config_json(config)}
