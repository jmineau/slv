"""Content-addressed cache for the inversion pipeline's components.

Each component (obs, prior, forward operator, ...) is pickled under
``{cache}/.fips/{version tag}/{component}/{hash}.pkl``: the tag pins the fips + pystilt
source revision and the hash covers only the config fields that component depends on.
"""

import functools
import hashlib
import importlib
import json
import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

# ---------------------------------------------------------------------------
# Component dependency sets: maps each cache component to the InversionConfig fields
# that actually affect that component's output.  Used to compute content-addressed
# cache paths so that only truly-stale components are recomputed when parameters
# change.
# ---------------------------------------------------------------------------
_OBS_DEPS: frozenset[str] = frozenset(
    {
        "tstart",
        "tend",
        "sites",
        "filter_pcaps",
        "subset_hours",
        "utc_offset",
        # fingerprint, not the raw path: a mobile_obs file rebuilt in place must re-key
        "mobile_obs_key",
    }
)
#: Obs-filter deps: change WHICH obs survive (and so the obs-indexed MDM/constant),
#: but NOT the footprint Jacobian, which is built per simulation receptor (filtered
#: only by obs_location) and reindexed to the surviving obs. Keeping these out of the
#: forward_operator key avoids a needless (expensive) Jacobian rebuild when toggled.
_OBS_FILTER_DEPS: frozenset[str] = frozenset({"filter_spikes", "spike_percentile"})
#: State-filter deps: filter_state_space drops obs in flux intervals (months) with too
#: few obs (min_obs_per_interval). fips reads min_sims_per_interval but does not apply
#: it -- counting simulations needs the Jacobian, built after this filter -- so it only
#: sits in the key. The obs-indexed MDM and constant are built AFTER that filter, so
#: they must re-key on it -- otherwise a run at a lower threshold reuses the
#: higher-threshold MDM and back-fills the recovered months
#: with ZERO variance, giving a singular S_z. Not on forward_operator (receptor-based,
#: reindexed) or obs (get_obs returns the full record; the filter is applied downstream).
_STATE_FILTER_DEPS: frozenset[str] = frozenset(
    {"min_obs_per_interval", "min_sims_per_interval"}
)
_PRIOR_DEPS: frozenset[str] = frozenset(
    {
        "prior",
        "prior_kwargs",
        "dx",
        "dy",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "flux_freq",
        "tstart",
        "tend",
    }
)

#: Default mapping of pipeline component → config fields that affect it.
DEFAULT_COMPONENT_DEPS: dict[str, frozenset[str]] = {
    "obs": _OBS_DEPS | _OBS_FILTER_DEPS,
    "prior": _PRIOR_DEPS,
    "forward_operator": _OBS_DEPS
    | _PRIOR_DEPS
    | {
        "stilt_project",
        "sparse_jacobian",
        "footprint",
        # num_processes and timeout intentionally excluded: they are
        # computational knobs that do not change the Jacobian result.
    },
    "prior_error": _PRIOR_DEPS
    | {
        "prior_base_std",
        "prior_std_frac",
        "prior_time_scale",
        "prior_spatial_scale",
    },
    # A multiplicative MDM term scales with the enhancement (observed obs-background, or the
    # prior-modeled H x_prior), so the MDM also depends on the background and forward-operator
    # + prior deps -- not just the obs deps.
    "modeldata_mismatch": _OBS_DEPS
    | _OBS_FILTER_DEPS
    | _STATE_FILTER_DEPS
    | _PRIOR_DEPS
    | {
        "mdm_components",
        "stilt_project",
        "sparse_jacobian",
        "footprint",
        "background",
        "background_kwargs",
    },
    "constant": _OBS_DEPS
    | _OBS_FILTER_DEPS
    | _STATE_FILTER_DEPS
    | {"background", "background_kwargs"},
}


#: Packages whose *release* each component also keys on, beyond fips + pystilt (which are in
#: the version tag). lair builds the prior and the backgrounds and uataq reads the obs; the
#: prior error and the MDM are built from those. Not the forward_operator: the Jacobian is
#: fips + pystilt only, and re-keying it would mean a full rebuild.
DEFAULT_COMPONENT_PACKAGES: dict[str, tuple[str, ...]] = {
    "obs": ("lair", "uataq"),
    "prior": ("lair",),
    "prior_error": ("lair",),
    "modeldata_mismatch": ("lair", "uataq"),
    "constant": ("lair", "uataq"),
}


def _json_default(v):
    """Fallback JSON serializer: converts non-primitive types to strings."""
    if isinstance(v, (list, tuple)):
        return [_json_default(i) for i in v]
    if isinstance(v, dict):
        return {k: _json_default(vv) for k, vv in sorted(v.items())}
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return str(v)


def _component_hash(
    config, fields: frozenset[str], packages: tuple[str, ...] = ()
) -> str:
    """Return the first 12 hex chars of sha256 over the given config fields, plus the
    release versions of ``packages`` (see :func:`_release`)."""
    data = {f: _json_default(getattr(config, f)) for f in sorted(fields)}
    if packages:
        data["__packages__"] = {p: _release(p) for p in sorted(packages)}
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def _source_dir(import_name: str) -> Path | None:
    """The directory a package is imported from, or None if it can't be imported."""
    try:
        mod = importlib.import_module(import_name)
    except ImportError:
        return None
    return Path(mod.__file__).resolve().parent if mod.__file__ else None


def _is_installed(src: Path) -> bool:
    """A regular install (under site-packages), as opposed to an editable checkout."""
    return bool({"site-packages", "dist-packages"} & set(src.parts))


def _pkg_version(import_name: str, dist_name: str) -> str:
    """Release version of a package, for cache keying -- coarser than :func:`_pkg_rev`.

    An installed copy reports its metadata version. An editable checkout's metadata is
    frozen at install time, so the source is read instead: the static
    ``[project].version`` of its pyproject.toml, or for a setuptools-scm package, its
    latest git tag. Either changes on a release, not on every commit or edit.
    """
    src = _source_dir(import_name)
    if src is not None and not _is_installed(src):
        pyproject = next(
            (
                d / "pyproject.toml"
                for d in [src, *src.parents][:4]
                if (d / "pyproject.toml").exists()
            ),
            None,
        )
        if pyproject is not None:
            project = tomllib.loads(pyproject.read_text()).get("project", {})
            if "version" in project:
                return str(project["version"])
            if "version" in project.get("dynamic", []):
                try:
                    out = subprocess.run(
                        [
                            "git",
                            "-C",
                            str(pyproject.parent),
                            "describe",
                            "--tags",
                            "--abbrev=0",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        return out.stdout.strip().removeprefix("v")
                except (OSError, subprocess.SubprocessError):
                    pass
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return "unknown"


@functools.cache
def _release(package: str) -> str:
    """:func:`_pkg_version` of ``package`` (import and distribution names match), once
    per process."""
    return _pkg_version(package, package)


def _pkg_rev(import_name: str, dist_name: str) -> str:
    """Source revision of a package, for cache keying.

    For an editable git checkout (the dev setup), ``git describe`` yields a tag +
    commits-since + short SHA (+ ``-dirty``), so any commit *or* uncommitted edit
    to the package busts the cache automatically -- no reinstall or version bump
    needed.  Falls back to the installed metadata version for a regular install.

    A regular install lives under ``site-packages``, and git must not be asked about
    it: a venv inside another repo (slv's own ``.venv``) would answer with *that*
    repo's revision, so every slv commit would orphan the whole cache.
    """
    src = _source_dir(import_name)
    if src is not None and not _is_installed(src):
        try:
            out = subprocess.run(
                [
                    "git",
                    "-C",
                    str(src),
                    "describe",
                    "--tags",
                    "--always",
                    "--dirty",
                    "--abbrev=8",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        return _dist_version(dist_name)
    except PackageNotFoundError:
        return "unknown"


@functools.lru_cache(maxsize=1)
def _version_tag() -> str:
    """``.fips/`` cache namespace pinning the running fips + pystilt source revision.

    Computed once per process from ``_pkg_rev`` so that committing or editing
    fips/pystilt code lands the cache in a fresh tree instead of silently reusing a
    stale component -- e.g. a Jacobian built by the old footprint aggregation.

    slv's own revision is intentionally *not* in the key: slv changes often, so
    when you change slv component-building logic (obs/prior/mdm/constant), force a
    rebuild via ``config.cache_overwrite`` rather than relying on the tag.
    """
    return f"fips-{_pkg_rev('fips', 'fips')}_pystilt-{_pkg_rev('stilt', 'pystilt')}"


def fips_cache(cls, filename):
    """Content-addressed cache decorator for pipeline methods.

    Parameters
    ----------
    cls :
        The fips class to use for ``cls.from_file`` / ``result.to_file``.
    filename :
        Stage name / cache stem (e.g. ``"obs"``, ``"prior_error"``).

    Cache layout
    ------------
    Files are stored under
    ``{cache_dir}/.fips/{version_tag}/{component}/{hash}.pkl`` where
    ``version_tag`` pins the running fips/pystilt source revision (``git describe``
    of the editable checkout, else installed metadata; see ``_version_tag``) and
    ``hash`` is derived only from the config fields that actually affect that
    component (see ``COMPONENT_DEPS``), plus the release versions of the packages it
    is built with (lair / uataq, see ``COMPONENT_PACKAGES``; a new lair release rebuilds
    the obs and prior, not the Jacobian).  Committing/bumping a package therefore
    lands in a fresh tree rather than silently reusing a stale component.  This
    means:

    * Changing ``prior_base_std`` reuses ``obs.pkl`` and ``forward_operator.pkl``
      untouched, and only regenerates ``prior_error.pkl``.
    * Configs that share the same upstream parameters share the same files —
      no manual cache wiping is needed.

    ``config.cache`` controls caching:

    * ``False`` / ``None`` — no caching (default)
    * ``True``             — cache in the current working directory
    * ``str`` / ``Path``  — cache in that directory

    ``config.cache_overwrite`` controls forced recomputation:

    * ``[]``        — never overwrite (default)
    * ``"all"``     — overwrite every component
    * ``[component, …]`` — overwrite specific components (all hashes for that component
      are deleted and the component is recomputed fresh)
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = getattr(self.config, "cache", False)
            if not cache:
                return method(self, *args, **kwargs)

            overwrite = getattr(self.config, "cache_overwrite", [])
            if overwrite == "all":
                should_overwrite = True
            elif isinstance(overwrite, str):
                should_overwrite = filename == overwrite
            else:
                should_overwrite = filename in set(overwrite)

            cache_dir = Path.cwd() if cache is True else Path(cache)
            # All fips-managed cache files live under ``.fips/<version tag>/`` so
            # the workflow tree stays clean and a fips/pystilt change lands in a
            # fresh tree instead of silently reusing stale components.
            fips_dir = cache_dir / ".fips" / _version_tag()

            # --- Content-addressed path ---
            component_deps = getattr(self, "COMPONENT_DEPS", DEFAULT_COMPONENT_DEPS)
            component_packages = getattr(
                self, "COMPONENT_PACKAGES", DEFAULT_COMPONENT_PACKAGES
            )
            fields = component_deps.get(filename)
            if fields:
                h = _component_hash(
                    self.config, fields, component_packages.get(filename, ())
                )
                component_dir = fips_dir / filename
                path = component_dir / f"{h}.pkl"
            else:
                # Fallback: flat file for components not listed in COMPONENT_DEPS
                h = "flat"
                component_dir = fips_dir
                path = fips_dir / f"{filename}.pkl"

            if path.exists() and not should_overwrite:
                print(f"Loading cached {filename} [{h}] from {path}")
                return cls.from_file(path)

            if should_overwrite and fields and component_dir.exists():
                # Remove all stale hashes for this component before recomputing
                stale = list(component_dir.glob("*.pkl"))
                for s in stale:
                    s.unlink()
                if stale:
                    print(
                        f"Cleared {len(stale)} stale cache file(s) for component '{filename}'"
                    )

            result = method(self, *args, **kwargs)

            component_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving {filename} [{h}] to {path}")
            result.to_file(path)

            return result

        return wrapper

    return decorator
