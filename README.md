# Salt Lake Valley py

[![Tests](https://github.com/jmineau/slv/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/slv/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/slv/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/slv/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/slv/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/slv)
[![PyPI version](https://badge.fury.io/py/slv.svg)](https://badge.fury.io/py/slv)
[![Python Version](https://img.shields.io/pypi/pyversions/slv.svg)](https://pypi.org/project/slv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258471.svg)](https://doi.org/10.5281/zenodo.22258471)
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

`uv sync` also installs the `dev` group (tests, linting, docs), which includes the
inversion extra.

### With `xesmf` (EPA / EDGAR priors)

The EPA and EDGAR priors are regridded with `xesmf`, which needs a compiled ESMF
library from conda-forge, so install into the conda environment instead:

```bash
git clone https://github.com/jmineau/slv.git
cd slv
conda env create -f ci/environment.yml
conda activate slv
pip install --no-deps -e .
```

## Usage

A map of the valley:

```python
from slv.basemap import SaltLake

m = (SaltLake(tiles="terrain")          # Stadia stamen_terrain; needs $STADIA_API_KEY
     .add_population()
     .add_trax(lines="RG")
     .add_sites(["wbb", "ldf", "hdp"], labels={"wbb": "UOU"})
     .add_mesowest()
     .add_legend().add_inset().add_north_arrow())
m.fig.savefig("slv.png", dpi=300)
```

A methane inversion (builds the Jacobian from the PYSTILT footprints; run it through SLURM):

```python
from slv.inversion import InversionConfig, SLVMethaneInversion

config = InversionConfig(tstart="2016-01-01", tend="2024-01-01", flux_freq="MS",
                         sites=["wbb"], footprint="0.01", cache="./cache")
problem = SLVMethaneInversion(config).run()
```

The [usage guide](https://jmineau.github.io/slv/usage.html) also covers loading
measurements, hyperparameter sweeps and building TRAX receptors.

## Documentation

Full documentation is available at [https://jmineau.github.io/slv/](https://jmineau.github.io/slv/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
