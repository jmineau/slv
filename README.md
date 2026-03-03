# Salt Lakey Valley py

[![Tests](https://github.com/jmineau/slv/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/slv/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/slv/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/slv/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/slv)
[![PyPI version](https://badge.fury.io/py/slv.svg)](https://badge.fury.io/py/slv)
[![Python Version](https://img.shields.io/pypi/pyversions/slv.svg)](https://pypi.org/project/slv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

Salt Lake Valley python modules

## Installation

### Standard (uv or pip)

```bash
git clone https://github.com/jmineau/slv.git
cd slv
uv sync  # or: pip install -e .
```

To include the inversion module:

```bash
uv sync --extra inversion  # or: pip install -e ".[inversion]"
```

### With `xesmf` (inversion + regridding)

`xesmf` requires a pre-built ESMF library and must be installed via conda-forge
alongside the rest of the environment:

```bash
git clone https://github.com/jmineau/slv.git
cd slv
conda env create -f ci/environment.yml
conda activate slv
pip install --no-deps -e .
```

## Usage

```python
import slv

# Add usage example here
```

## Documentation

Full documentation is available at [https://jmineau.github.io/slv/](https://jmineau.github.io/slv/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
