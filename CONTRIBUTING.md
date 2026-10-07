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

   # OR using conda (needed for xesmf, i.e. the EPA / EDGAR priors):
   conda env create -f ci/environment.yml
   conda activate slv
   pip install --no-deps -e .
   ```

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

4. Run test suite:
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
- Write clear, descriptive commit messages
- Add user-visible changes to `CHANGELOG.md` under `## [Unreleased]`
- Ensure all tests pass
- Maintain or improve test coverage
- Update documentation as needed

## Releasing

The version comes from git tags, through setuptools-scm, so there is no
version string to bump. Versions are calendar-based, `YYYY.M.PATCH`.

1. Run `just changelog` to draft the entries from your commit messages, edit
   them into `CHANGELOG.md` under `## [Unreleased]`, then rename that heading
   to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty `## [Unreleased]` above
   it. Commit (`chore(release): X.Y.Z`) and push to `main`.
2. Run `just release X.Y.Z`. It checks that the tree is clean, that `main` is in
   sync with GitHub, and that the version is newer than every existing tag, then
   pushes the tag `vX.Y.Z`.
3. The Publish workflow creates a GitHub Release from the CHANGELOG section.
   Zenodo archives the release and mints a DOI. The Documentation workflow
   publishes the docs as `X.Y.Z/` (and as `stable/`) in the version dropdown.

slv is not published to PyPI; it is installed from GitHub.

## Dependency updates

Dependabot opens one pull request a month per kind of pin: GitHub Actions,
pre-commit hooks, and `uv.lock` (it never raises the minimum versions in
`pyproject.toml`). Merge it when CI passes. Keep `ci/environment.yml` in step
with `pyproject.toml` by hand.

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
