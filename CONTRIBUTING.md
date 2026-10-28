# Contributing to Salt Lake Valley py

Thank you for considering contributing to Salt Lake Valley py! We welcome contributions from the community.

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science. If you
contribute with an agent, [AGENTS.md](AGENTS.md) at the repository root is
the orientation file it should read.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/slv.git
   cd slv
   ```
3. Install the package with the development dependencies (Python 3.11 or higher):
   ```bash
   # Using uv (installs the `dev` dependency group, which includes the inversion extra):
   uv sync

   # OR using pixi (needed for xesmf, i.e. the EPA / EDGAR priors):
   pixi install
   ```

   `uv sync --no-dev` installs slv alone (add `--extra inversion` for `fips[flux]`
   and `pystilt`); `pixi run pytest` runs the tests in the pixi environment
   (`.pixi/envs/default`). The base install is enough for `domain`, `basemap`,
   `measurements` and `emissions`; the `inversion` extra is what unlocks
   `slv.inversion`.

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow our coding standards:
   - Code is formatted with ruff
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run quality checks (lint, type check, docstrings, and the tests):
   ```bash
   just quality-check
   ```

4. Run test suite (it also runs the `>>>` examples in the docstrings under `src/`):
   ```bash
   just test
   ```

5. Run pre-commit checks:
   ```bash
   just pre-commit
   ```

6. Commit your changes with a [Conventional Commits](https://www.conventionalcommits.org/)
   message (`fix:`, `feat:`, `docs:`, with `!` for a breaking change):
   ```bash
   git add .
   git commit -m "fix(inversion): <what the change does>"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Write clear, descriptive commit messages: they become the changelog entries
- Leave `CHANGELOG.md` alone: the release notes are written at release (see Releasing)
- Ensure all tests pass
- Maintain or improve test coverage
- Update documentation as needed

## Continuous integration

CI (`.github/workflows/`) runs the same `just` recipes:

- `tests.yml` runs `just cov` from `uv.lock` on Linux and Python 3.11 to 3.14.
  xesmf is not in `uv.lock` (conda-forge only), so the tests that need it skip there.
  Its `pixi` job runs the whole suite in the pixi environment (`[tool.pixi]` in
  `pyproject.toml`, locked in `pixi.lock`): conda-forge supplies ESMF and xesmf, and
  everything else comes from the dependencies in `pyproject.toml`, so nothing is
  listed twice. The job fails on a `pixi.lock` that is out of date.
- `quality.yml` runs lint, type check and docstring coverage on Python 3.11.
- `docs.yml` builds the docs on every pull request (warnings fail the build). Pushes
  to `main` publish them as `dev/`, and release tags as one folder per release (and
  `stable/`), on the `gh-pages` branch that GitHub Pages serves;
  `.github/scripts/docs_versions.py` maintains that branch.
- `publish.yml` makes the GitHub Release from a tag (Zenodo archives it).
- `lock-upgrade.yml` and `pixi-update.yml` are described under Dependency updates.

No CHPC data roots are set in CI, so tests must not read group data. A docstring
example that needs data CI lacks gets `# doctest: +SKIP` on its `>>>` line. Warnings are
errors in the tests: a test that expects one asserts it with `pytest.warns`, and a
third-party one is ignored by message in `[tool.pytest]`. pyrefly fails on any error
not in `pyrefly-baseline.json`; fix the listed ones over time and run
`uv run pyrefly check --update-baseline`.

## Releasing

The version comes from git tags, through setuptools-scm, so there is no
version string to bump. Versions are calendar-based, `YYYY.M.PATCH`.

1. Write the changelog: `just changelog` drafts the entries from the commit
   messages since the last release. Edit them into `CHANGELOG.md` under
   `## [Unreleased]`, then rename that heading to `## [X.Y.Z] - YYYY-MM-DD` and
   start a new empty `## [Unreleased]` above it. Commit (`chore(release): X.Y.Z`)
   and push to `main`.
2. Run `just release X.Y.Z`. It checks that the tree is clean, that `main` is in
   sync with GitHub, and that the version is newer than every existing tag, then
   pushes the tag `vX.Y.Z`.
3. The Publish workflow creates a GitHub Release from the CHANGELOG section.
   Zenodo archives the release and mints a DOI. The Documentation workflow
   publishes the docs as `X.Y.Z/` (and as `stable/`) in the version dropdown.

slv is not published to PyPI; it is installed from GitHub.

## Dependency updates

Dependabot opens one pull request a month for each of GitHub Actions and
pre-commit hooks. Merge it when CI passes.

`uv.lock` (the dev tools and every dependency) is updated by the Upgrade lock
workflow alone, not by Dependabot, which would leave a package at its old
version when the newest release needs a newer Python than the oldest one this
project supports. On the 1st of each month `lock-upgrade.yml` runs
`uv lock --upgrade` (`uv.lock` then holds the newest release for each Python; it
never raises the minimum versions in `pyproject.toml`) and, when the lock changes,
opens the pull request `chore/lock-upgrade` (or updates it). GitHub does not start
workflows on a pull request that its own token opened, so the workflow starts
Tests, Code Quality and Documentation on the branch itself; their results show on
the pull request. Merge it when they pass. You can also run it by hand: Actions >
Upgrade lock > Run workflow. It does `uv.lock` only.

`pixi.lock` (the environment with xesmf) is separate: after changing a dependency
in `pyproject.toml`, run `pixi lock` as well as `uv lock`. `pixi-update.yml` runs
`pixi update` monthly (and on demand from the Actions tab) and opens one pull
request, branch `chore/pixi-update`, when `pixi.lock` changes. A pull request
opened with a workflow's own token starts no CI, so the workflow dispatches Tests
on that branch; it needs the repository setting "Allow GitHub Actions to create
and approve pull requests", as the Upgrade lock workflow does.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version; `copier update` pulls in
later template changes. Improvements that would help every package are best
made in the template.

## Reporting Bugs

When reporting bugs, please include:
- Your operating system and Python version
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any error messages or logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain why it would be useful
- Consider submitting a pull request to implement it

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
