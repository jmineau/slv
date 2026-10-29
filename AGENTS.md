> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

# AGENTS.md — Developer and Agent Guide for slv

`slv` is the user's **Salt Lake Valley** research codebase — project-specific
plumbing that ties together `lair`, `uataq`, `fips`, and `pystilt` to support
methane (and other trace gas) emission estimation in the SLV. Active research
code; calendar-versioned (e.g. `2026.2.0`).

PyPI/import name: `slv`. Source in `src/slv/`. Contribution mechanics are in
[CONTRIBUTING.md](CONTRIBUTING.md); the longer developer notes (module layout, the
inversion package and its cache, writing the docs, a note on the tests) are on the
Contributing page, [docs/contributing.rst](docs/contributing.rst).

## What slv is

- An **integration layer**, not a general library. Its job is to make the
  user's everyday SLV workflows reproducible: defining the domain, loading
  measurements and inventories, running flux inversions (via `fips`),
  building figures.
- Depends directly on the user's own packages — `lair[science,geo]` and `uataq` (each pinned
  to a release tag) are git-installed; `fips[viz]` and `pystilt`
  live in the `inversion` extra.

## Module layout

Subpackages: `domain`, `basemap`, `measurements` (`mobile/` holds the TRAX code),
`emissions`, `meteorology`, `inversion`. `slv.__init__` is intentionally small (just
`get_data_dir` and version metadata); the subpackages are the API surface. The
module-by-module tour is in `docs/contributing.rst` ("Module layout", "The inversion
package"). Packaged data files (CSVs, GeoJSON) ship via
`[tool.setuptools.package-data]` in `pyproject.toml`: don't move them without updating
that.

## Dev commands

Driven by `just` + `uv`; CI runs the same recipes (`just` lists them all).

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: slv, `slv[inversion]` and the dev tools into `.venv` |
| `just test` | the tests and the `>>>` examples under `src/`, in parallel (extra args go to pytest; `-n 0` for serial) |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly |
| `just docstr` | docstring coverage of the public API (95%) |
| `just quality-check` | lint, type check, docstrings, tests |
| `just build-docs` | clean Sphinx HTML build, running every example; warnings are errors (`just docs-serve` previews) |
| `just pre-commit` | every pre-commit hook on every file |
| `just changelog` / `just release X` | at release: draft CHANGELOG entries / tag and push a release |
| `just clean` | wipe build artifacts, caches, coverage, docs |

What each CI workflow does, the pixi job and the monthly `pixi update` are in
[CONTRIBUTING.md](CONTRIBUTING.md) ("Continuous integration"). What keeps CI green:

- **After changing a dependency, run `pixi lock` as well as `uv lock`**: the pixi job
  fails on a `pixi.lock` that is out of date. xesmf is not in `uv.lock` (conda-forge
  only), so there the tests that need it skip; the `inversion` extra keeps
  `# "xesmf"` commented out on purpose.
- **No CHPC data roots are set in CI**: tests must not read group data.
- **Warnings are errors in the tests**; a test that expects one asserts it with
  `pytest.warns`, and a third-party one is ignored by message in `[tool.pytest]`.
- **pyrefly fails on any error not in `pyrefly-baseline.json`**: fix them over time and
  `--update-baseline`.
- **`uv.lock` is upgraded by the Upgrade lock workflow alone**
  (`lock-upgrade.yml`, monthly `uv lock --upgrade`); Dependabot keeps the Actions and
  pre-commit hooks. `pixi-update.yml` does `pixi.lock`, separately.
- **A pull request leaves `CHANGELOG.md` alone**: at release, `just changelog` drafts
  the entries from the commit messages, so write each one as the line a user would read.
- **A docstring example that needs data CI lacks gets `# doctest: +SKIP`** on its `>>>`
  line (`--doctest-modules` runs the rest as tests).
- **The tooling comes from**
  [jmineau/python-template](https://github.com/jmineau/python-template)
  (`.copier-answers.yml`); `copier update` pulls in its changes.
- **Numpy docstring sections must be ones napoleon knows**: an unknown header such as
  "Cache layout" is read as more parameters.
- **Give each public class attribute and property a docstring that says what it is**
  (and See Also or Examples where they help, as in pandas): it has its own API page,
  and one with no docstring shows an empty row (`docs/contributing.rst`, "Writing the
  docs", says how the pages are generated).
- **Never edit the `gh-pages` branch by hand**: `.github/scripts/docs_versions.py`
  maintains it.

## Invariants to respect

- **Domain constants live in one place** (`slv.domain`). Don't hardcode
  `XMIN`/`XMAX`/`YMIN`/`YMAX` elsewhere; import from `slv.domain`.
- **Data paths via `get_data_dir`** (`get_data_dir("SLV_DAQ_DIR")`), not raw
  `os.environ.get`. It is lair's `config.get_data_dir`: a `ValueError` (not `KeyError`,
  on purpose) naming the env var to set, for an unset or empty variable, so don't
  paper over it. The one exception is
  `InversionConfig.stilt_project`, which defaults to `SLV_STILT_DIR` if set, else
  `DEFAULT_STILT_PROJECT`, so the config stays usable without the env var. Env vars
  are documented on the docs' installation page ("Data locations").
- **Upstream pins**: `lair` and `uataq` are pinned to release tags (`@vYYYY.MM.PATCH`
  and `@vYYYY.M.PATCH`, in `pyproject.toml`; bump them and relock with `uv lock` and
  `pixi lock`). Bumping behavior in those repos can
  silently change `slv` results — coordinate changes.
- **The inversion pipeline is functions of the config** (fips has no pipeline):
  `build_problem` composes the `get_*` components into a `fips.Problem`, and `run`
  solves it with `fips.solve`. A new variant composes its own `build_problem` from the
  same functions; the cached components keep their names and keys.
- **Labels are matched exactly.** fips 0.2 joins labels exactly and never rounds them;
  `build_problem` rounds the float labels of the prior and the Jacobian's columns to
  `LABEL_DECIMALS` (6) so the regridded prior's cells meet the Jacobian's.
- **A receptor with an empty footprint is a row of zeros; one with no footprint drops
  its obs** (`project_jacobian`, `build_problem`): an empty footprint saw no emissions,
  a missing one was never run.
- **Packaged CSVs are part of the API contract.** Schema changes break
  downstream notebooks.
- **Inversion cache keying** (`inversion/cache.py`; how the folder, version tag,
  release hashes and input-file keys work: `docs/contributing.rst`, "The component
  cache"). Input files are keyed by what is in them, never by path alone. Caveats:
  (1) slv's *own* rev is deliberately
  not in the key, so when you change slv component-building logic
  (obs/prior/mdm/constant) force a rebuild via `config.cache_overwrite`; a change to
  what a component holds for the same config (a reader of old files would misread
  them) bumps its number in `DEFAULT_COMPONENT_REVISIONS` instead, which re-keys it;
  (2) reinstall editable packages (`uv sync` / `pip install -e`) when you cut a
  version so `importlib.metadata` matches the source — this does not affect the
  cache key but keeps `pip show` and other version consumers honest.

## Examples in the docs

Examples run when the docs build, so they show real output, and the build fails
when one breaks. A figure in a docstring is a `.. plot::` directive in its Examples
section; code in an `.rst` page is a `.. ipython:: python` block (a worked example of
each, the `@savefig`, `:okexcept:` and `:okwarning:` options, and the notebook switch:
`docs/contributing.rst`, "Writing the docs"). All `.rst` pages share one session, so
each page imports and defines what it uses. Split a long block so that no multi-line
statement is its 10th (or 100th) statement: the directive runs only that statement's
first line. `just build-docs` runs everything, as CI does.

- **Data:** the docs build on GitHub Actions, so an example uses synthetic data,
  a small file in the repository, or a public download, never a machine-specific
  path. Seed random numbers so the figures don't change from build to build.
- **slv's real data is unpublished**, and the docs are public. An example uses
  synthetic data and never real SLV results, measurements, or paths into the
  workspace. xesmf needs conda's ESMF (the pixi environment), which the docs build
  does not have, so an example never regrids.

## Conventions and tooling

- **Python**: 3.11+ (`requires-python`, which ruff also reads; pandas 3 and current xarray need 3.11).
- **Linting and docstrings**: ruff, with the NumPy convention and the summary on the
  line after the opening quotes (D213); the rules, and why D401 and D205 are off,
  are in `pyproject.toml`.
- **Types**: pyrefly (default preset) on `src/`, with a baseline of the errors it
  found on adoption; a new error fails. Suppress one only with
  `# pyrefly: ignore[<code>]` on the line above, after a comment saying why.
  `py.typed` shipped.

## Gotchas

- `basemap.SaltLake` mutates a matplotlib figure with cartopy axes;
  it is not thread-safe.
- Mobile (TRAX) obs are keyed by PYSTILT location_id, not site. Anything keyed by
  site must resolve through `pipelines.obs_sites` (receptor -> the single
  mobile site in `config.sites`) -- the MDM and the bias Jacobian do (the rolling
  background and the subhour std use stationary sites only; see `docs/contributing.rst`,
  "Notes on the pipeline").
- Build a cached cartopy tiler with `basemap.cached_tiler(GoogleTiles, style=...)`,
  never with `cache=True`: two processes building one at the same moment (test workers,
  sweep jobs on one node) can fail with `FileExistsError` (the docstring says why).
  `stadia_tiles` and `InversionConfig`'s default `tiler` both use it. It stays in
  `slv.basemap`, which the base install has and which imports nothing from `slv.inversion`.
- Never change a config field's default spelling (e.g. `"32d"`) or a
  `COMPONENT_DEPS` set casually: both are hashed into cache keys (a forward_operator
  key change means a full Jacobian rebuild) and sweep config_ids. Normalize at the
  parse site instead (`covariances.normalize_duration`).
- Tests: `tests/conftest.py` points `SLV_STILT_DIR` at an empty tmp dir, so a test that
  reaches the Jacobian build without stubbing `_project_jacobian` fails fast instead
  of scanning the paper-1 project. The pipeline functions are tested on a made-up
  inversion: `tests/inversion/conftest.py`'s `synthetic` fixture stubs the loaders and
  makes a config; viz tests stub `GeoAxes.add_image` so no tiles are fetched; the TRAX loaders run
  on a synthetic two-line network (`tests/measurements/mobile/test_network.py`); tests of
  `GMLDiscrete` stub `noaa.GMLData.download`. Tests needing xesmf (lair cell areas)
  `importorskip` it: they run in the pixi env / CI only. A test that looks for fips cache
  files builds the path the way the cache does (`_version_tag()`). Measure coverage with
  `--cov=slv`: `--cov=slv.inversion.viz` imports numpy twice and fails collection. Why the
  pixi env pins the no-MPI ESMF build: `docs/contributing.rst`, "Notes on the tests".
- `merge_with_gps` is on the `build_trax_obs` path, so changing what it drops (the
  altitude range, a repeated reading) changes `obs.parquet` on the next rebuild.
