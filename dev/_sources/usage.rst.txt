Usage
=====

A map of the valley
-------------------

:class:`~slv.basemap.SaltLake` makes a cartopy map of the valley; each ``add_*``
method adds a layer and returns the map, so calls chain.

.. code-block:: python

   from slv.basemap import SaltLake

   m = (
       SaltLake(tiles="terrain")  # Stadia stamen_terrain; needs $STADIA_API_KEY
       .add_population()  # block-group density with a colorbar panel
       .add_trax(lines="RG")  # red and green lines
       .add_sites(["wbb", "ldf", "hdp"], labels={"wbb": "UOU"})
       .add_mesowest()
       .add_legend()
       .add_inset()
       .add_north_arrow()
   )
   m.fig.savefig("slv.png", dpi=300)

``tiles=None`` draws without tiles (no network). Pass ``ax=`` to draw on an existing
cartopy axes, and your own data to any ``add_*`` method in place of the defaults.

Measurements
------------

.. code-block:: python

   from slv.measurements import aggregate_obs, load_concentrations

   obs = load_concentrations(
       "CH4", sites=["wbb", "hdp"], time_range=("2024-01-01", "2024-02-01")
   )
   hourly = aggregate_obs(obs, freq="1h")

A methane inversion
-------------------

An :class:`~slv.inversion.config.InversionConfig` describes the run;
:func:`~slv.inversion.pipelines.build_problem` builds the fips problem from it (obs,
background, prior, Jacobian from the PYSTILT footprints, covariances), and
:func:`~slv.inversion.pipelines.run` builds it, solves it and prints a summary.
Components are cached under ``cache`` keyed on the settings they depend on, so a
second run that changes only, say, ``prior_base_std`` reuses the obs and the Jacobian.
The keys also see the input files: the CT-STILT CSV (``background_kwargs``) and
``mobile_obs`` by their size and modification time, and the STILT projects by how many
of the selected receptors have a footprint and which settings folder they are read
from. A CSV rewritten in place, or simulations run since the last run, rebuild what
depends on them; a cache keyed by path alone could not see either.
An obs the model cannot simulate is dropped before the solve, and the run prints how
many: one with no background, and one whose receptor has no footprint (its STILT
simulation was not run, or failed). A receptor whose footprint is empty, because its
particles left the domain, keeps its obs: it saw no emissions.

.. code-block:: python

   from slv.inversion import InversionConfig, run

   config = InversionConfig(
       tstart="2016-01-01",
       tend="2024-01-01",
       flux_freq="MS",
       sites=["wbb"],
       aggregate_obs="1D",
       background="rolling",
       prior="epa",
       variant="hrrr",
       cache="./cache",
       num_processes=30,
   )
   post = run(config)  # a fips Posterior

   post.estimate["flux"].to_series()  # the posterior flux by time and cell
   post.problem.prior["flux"]  # the prior, a fips Vector
   post.cov(W)  # the posterior covariance of aggregates W (an Operator), such as totals

The flux intervals are whole ``flux_freq`` periods from ``tstart`` to ``tend``, so both
must be on its boundaries: January 1 for ``"YS"``, the first of a quarter for ``"QS"``,
the first of a month for ``"MS"``, Sundays two weeks apart for ``"2W"``. A config that
is off them raises ``ValueError`` rather than leave the obs of the days outside whole
periods with no flux to fit.

Building the Jacobian reads every footprint in the period: run it through SLURM, not
on a login node.

Sweeping hyperparameters
------------------------

:class:`~slv.inversion.sweep.Sweep` runs one inversion per combination of the listed
values, sharing the cache, and collects the fit metrics.

.. code-block:: python

   from slv.inversion import Sweep

   sweep = Sweep(
       cache="./cache",
       base_config=config,
       prior_base_std=[0.01, 0.02],
       gamma=[1.0, 3.0],
   )
   results = sweep.run(results_dir="./sweep", n_jobs=4)
   results.best()  # reduced chi^2 within 0.1 of 1

For a SLURM job array, write the grid with ``sweep.run(results_dir, n_jobs=0)`` and
call :func:`~slv.inversion.sweep.run_sweep_job` in each array task; it runs the config
at ``$SLURM_ARRAY_TASK_ID``.

TRAX receptors
--------------

The TRAX light-rail record becomes STILT receptors: one multi-point receptor per pass
over a 2-km track segment, and one point receptor per hour a train sits parked.

.. code-block:: python

   from slv.measurements.mobile import (
       build_dwell_receptors,
       build_trax_receptors,
       find_dwells,
       find_segment_crossings,
       load_trax_fixes,
       load_trax_network_points,
   )

   points = load_trax_network_points(meters=True)  # 50-m points, 2-km segments
   fixes = load_trax_fixes(location="outdoor")

   dwells = find_dwells(fixes)
   crossings = find_segment_crossings(fixes, points)  # drop the dwell fixes first
   receptors = build_trax_receptors(crossings, points, hours=range(12, 17))
   dwell_receptors = build_dwell_receptors(fixes, dwells, hours=range(12, 17))

The observation for each receptor comes from
:func:`~slv.measurements.mobile.trax_receptor_observations`, and the inversion reads
them through ``InversionConfig(mobile_obs=...)``.
