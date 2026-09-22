Installation
============

``slv`` needs Python 3.11 or higher. It installs ``lair`` and ``uataq`` from their
GitHub ``main`` branches.

Standard (uv or pip)
--------------------

.. code-block:: bash

   git clone https://github.com/jmineau/slv.git
   cd slv
   pip install -e .                  # domain, basemap, measurements, emissions
   pip install -e ".[inversion]"     # + slv.inversion (fips, PYSTILT)

With `uv <https://docs.astral.sh/uv/>`_, ``uv sync`` installs the package with the
``dev`` dependency group, which includes the ``inversion`` extra and the test, lint and
docs tools.

With ``xesmf`` (EPA / EDGAR priors)
-----------------------------------

The EPA and EDGAR priors are regridded with ``xesmf``, which needs a compiled ESMF
library from conda-forge. Use the conda environment instead:

.. code-block:: bash

   git clone https://github.com/jmineau/slv.git
   cd slv
   conda env create -f ci/environment.yml
   conda activate slv
   pip install --no-deps -e .

Data locations
--------------

Data outside the package are found through environment variables, read when they are
used; a missing one raises ``OSError`` naming it.

.. list-table::
   :header-rows: 1

   * - Variable
     - Used for
   * - ``SLV_SPATIAL_DIR``
     - census block groups, roads and borders (:mod:`slv.basemap`)
   * - ``SLV_USER_DATA_DIR``
     - ACS population, MesoWest stations, TRAX obs and point caches, PCAP events
   * - ``LINGROUP_DATA_DIR``
     - the TRAX line geometry
   * - ``SLV_DAQ_DIR``
     - Utah DAQ Picarro files
   * - ``SLV_SOUNDINGS_DIR``
     - SLC soundings, for PCAP events (:mod:`slv.meteorology.pcaps`)
   * - ``SLV_STILT_DIR``
     - the PYSTILT project with the footprints (default: the CHPC production project)
   * - ``STADIA_API_KEY``
     - Stadia map tiles (``SaltLake(tiles="terrain")``)
