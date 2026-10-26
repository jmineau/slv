Contributing
============

Contributions are welcome. `CONTRIBUTING.md <https://github.com/jmineau/slv/blob/main/CONTRIBUTING.md>`_
explains how to set up a development environment, run the checks, what each CI workflow
does, and how a pull request and a release go.
`AGENTS.md <https://github.com/jmineau/slv/blob/main/AGENTS.md>`_ is the short guide for
contributors and coding agents: the rules the code keeps, the commands, and the gotchas.
This page holds the longer notes it points to: the module layout, the inversion package
and its component cache, how the docs are written and generated, and a note on the tests.

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science. If you
contribute with an agent, ``AGENTS.md`` at the repository root is the
orientation file it should read.

Module layout
-------------

::

    src/slv/
      __init__.py        version metadata + get_data_dir(env_var) helper
      domain.py          SLV inversion domain constants (XMIN/XMAX/YMIN/YMAX,
                         BBOX, EXTENT)
      basemap.py         SaltLake — cartopy map of the valley; chainable add_* layers
                         (population, trax, sites, mesowest, points, interstates, borders,
                         inventory, legend, inset, north_arrow). Data via load_* functions
                         ($SLV_SPATIAL_DIR group data; $SLV_USER_DATA_DIR census ACS 2022 +
                         mesowest/); tiles="terrain" = Stadia stamen_terrain ($STADIA_API_KEY,
                         network when drawn; attribution added). cached_tiler builds any
                         cartopy tiler with a disk cache (see its docstring). Tests:
                         tests/test_basemap.py, synthetic data, no network.
                         (tiler, inventory, borders, TRAX, sites, ...)
      measurements/
        __init__.py      re-exports aggregate_obs, load_concentrations,
                         get_site_coordinates, load_site_config
        aggregate.py     observation aggregation helpers
        background.py    background concentration logic
        concentrations.py  load measurement concentrations
        instruments.py   site/instrument metadata
        mobile/          TRAX / mobile platform subpackage (one concern per module; the
                         package __init__ re-exports every public function and table, so
                         `from slv.measurements.mobile import X` works for them; classifier
                         thresholds and GPS constants stay in their modules)
          network.py     TRAX lines/points, storage yards (JRRSC and MRSC both
                         packaged), shed footprints, filter_near_routes (dissolved
                         buffers - no duplicates on shared track), get_geodf, UTM12;
                         group_dir()/user_dir() resolve the data roots on use (GROUP_DIR /
                         USER_DIR via module __getattr__) -- keep slv importable without
                         the CHPC env vars, CI relies on it
          gps.py         thin uataq wrappers: read_lin_gps (lin gps qaqc; uataq names
                         the sat count N_Satellites -> renamed N_Sat), read_horel_cr1000
                         (horel gps+cr1000 at lvl="raw" = the h5 files; the horel
                         qaqc/final CSVs lack NSAT/RSTS), read_trax_gps (by era: lin
                         before 2018-11-19, horel after, fill_gaps=True fills horel
                         outages from the lin GPS), merge_with_gps (horel_fallback:
                         rows the lin GPS never covered are georeferenced from the horel
                         logger where the Pi clock checks out -- lin_clock_offset_by_day /
                         clock_checked_days / fill_gps_gaps_with_horel; gps_source tags each row);
                         read_lin_gps, read_horel_cr1000 and read_trax_gps take num_processes
                         (the files are read on that many processes; build_trax_obs passes its
                         own to the location step's GPS read)
          location.py    per-minute indoor/outdoor classifier: depot (degraded GPS:
                         scatter > 1 m or <= 6 sats, 15-min vote) / yard / line
                         (pass-by within 300 m of a yard) / route / stopped / unknown
                         (no fix, > 100 m off track, shed ejecta); speed_est fallback
                         for GPGGA-only eras; state_intervals, label_observations;
                         location_features(num_processes=) measures every fix's distance to the
                         track and the yards on that many threads (shapely releases the GIL;
                         build_trax_obs passes its own num_processes)
          calibration.py cal_source tags + trax_uncalibrated_windows.csv handling;
                         trax_epoch_offsets.csv (manual-cal era analyzer offsets vs UOU,
                         -0.002..-0.024 ppm): epoch_offset_column stores them per row,
                         apply_epoch_offset subtracts them (load_trax_obs(epoch_offset=False)
                         to test without, no rebuild needed)
          obs.py         build_trax_obs (per-year chunks; slope guard on calibrated rows; every GPS fix + cal_source + low_pressure + state/indoor/
                         yard_name) / load_trax_obs(location="on_track"|"outdoor"|"all")
                         cached at $SLV_USER_DATA_DIR/trax/obs.parquet; LOCATION_SETS.
                         load_trax_obs filters on the rows first (cal_source, low_pressure,
                         location), then applies the epoch offset and builds the points for the
                         rows kept only: the whole record is 105 M rows, about half on track.
                         2026-09-22 audit rules, all from _read_support (qaqc CH4/flag/
                         cavity P read WITHOUT normalize_pollutant): apply_pressure_rule
                         (flag -64 overwrites -63 in lgr_ugga_qaqc.r, so pressure is
                         re-checked here; manual_cal rows exempt -- apply_low_pressure_rule
                         owns them and sets the low_pressure tag from the measured pressure,
                         never from flag -63 alone, for the same reason), apply_dropout_rule + dropout_times (+-8 min around any
                         atmosphere row < 1.70 ppm: a laser dropout reads low for minutes to
                         hours and its shoulder passes QC), recover_hot_rows (flag -64 rows
                         have a fitted CH4d_m but a blanked CH4d_ppm_cal: qaqc value / slope)
          transects.py   transect matrices obs[transect, point]: build_network_transects
                         (one matrix over the 50-m network points for all lines: per-epoch
                         inlet lag -> snap 60 m -> line by windowed vote among the point's
                         own letters (assign_lines) -> transits per line via
                         lair.transects.split_transits -> cell means with terminus dwells
                         trimmed), build_trax_transects (one NetCDF per year under
                         $SLV_USER_DATA_DIR/trax/transects/network/), load_network_transects;
                         load_transects reads the archived per-line matrices (to 2023-04,
                         provenance only). load_trax_obs(time_range=) reads one year of
                         obs.parquet via a row-group filter; apply_epoch_offset computes the
                         offset column when a parquet predates it
          receptors.py   STILT receptors for paper 2: load_trax_network_points (the 50-m
                         points from load_trax_points(50) — same points_along_line generator
                         as the 2-km points, so shared track is one row of points — each
                         tagged with its nearest 2-km segment; cached points_50m_network.geojson),
                         load_trax_fixes (GPS columns of obs.parquet only), find_segment_crossings
                         (runs of fixes on one segment, gap <= 10 min; `lines` = letters common
                         to the points hit), release_points (segment points on the crossing's
                         line — a junction segment holds several lines' arms), build_trax_receptors
                         (one PYSTILT MultiPointReceptor per crossing at the median time;
                         coverage measured along the route via s_<line>, duration filter),
                         find_dwells / build_dwell_receptors (parked-train point receptors,
                         r_idx dwell_<d>_<YYYYMMDDHH>, freq >= 1h) / label_dwell_site (yard),
                         filter_receptor_hours (fixed UTC_OFFSET, standard-time hours).
                         Driver: SLV/stilt/ops/build_trax_receptors.py
          receptor_obs.py the CH4 observation for each TRAX receptor: trax_receptor_observations
                         (mean CH4 over a crossing's pass / a dwell's hour, window shifted by the
                         per-epoch inlet lag, keyed (obs_location, obs_time) via
                         stilt.read_receptors so it matches the footprints exactly),
                         inlet_lag_seconds. Driver: SLV/stilt/ops/build_trax_receptor_obs.py.
                         Consumed by InversionConfig.mobile_obs -> get_slv_observations
                         (load_mobile_obs); cache keys on mobile_obs_key (path+size+mtime).
          wyoming.py     Wyoming mobile lab (Aeris + met readers, ratios, wind-barb map)
          jrrsc.geojson, mrsc.geojson (yards), trax_depots.geojson (sheds),
          trax_uncalibrated_windows.csv  packaged data; the yard polygons ship with
                         the package so nothing but UTA_TRAX.geojson comes from group data
        pollutants.py    pollutant metadata
        sites.py         site_config loader (load_site_config); get_site_coordinates resolver
        site_config.csv  packaged data
      emissions/
        __init__.py      re-exports plot_point_sources
        point_sources.py  CH4 point source plotting
        ch4_point_sources.csv  packaged data
      meteorology/
        __init__.py      no imports: import the submodules explicitly
                         (`from slv.meteorology import pcaps`)
        mesowest.py      MesoWest surface observations for the SLV sites (hourly station
                         archives under $SLV_USER_DATA_DIR/mesowest)
        pcaps.py         SLV-specific PCAP wrappers
        vhd.py           valley heat deficit (VHD) from the SLC soundings
      inversion/         the inversion package, below
      py.typed           ships type hints
    tests/               pytest
    docs/                Sphinx
    pixi.lock            the environment with xesmf, locked ([tool.pixi] in pyproject.toml)

``slv.__init__`` is intentionally small (just ``get_data_dir`` and the version
metadata). The subpackages are the API surface:

.. code-block:: python

   from slv.domain import BBOX, EXTENT, XMIN, XMAX, YMIN, YMAX
   from slv.basemap import SaltLake
   from slv.measurements import aggregate_obs, load_concentrations, get_site_coordinates
   from slv.emissions import plot_point_sources

The base install is enough for ``domain``, ``basemap``, ``measurements`` and
``emissions``; the ``inversion`` extra unlocks ``slv.inversion`` (it depends on
``fips[flux]``, which in turn depends on ``pystilt``).

The inversion package
---------------------

This section holds everything about the inversion package's layout, exports, cache
and pipeline internals, in one place.

::

    src/slv/inversion/
      __init__.py      re-exports InversionConfig, SLVMethaneInversion, Sweep,
                       run_sweep_job, SweepResults
      config.py        InversionConfig — pydantic-ish config for an inversion;
                       DEFAULT_STILT_PROJECT + stilt_project_dir() ($SLV_STILT_DIR);
                       state_grid (stilt Grid, domain snapped outward to whole cells) is
                       the one definition of the flux cells, `grid` takes its axes
      pipelines.py     SLVMethaneInversion — the pipeline proper: get_obs / get_prior /
                       get_forward_operator (+ cached _get_flux_jacobian) / get_prior_error /
                       get_constant, state filtering, obs aggregation, run, obs_sites.
                       Composed from the mixins below (MRO: Reporting, CoverageFilter, Bias,
                       ModelDataMismatch, fips FluxInversionPipeline); every method keeps its
                       name, so subclasses override as before. Re-exports the cache helpers.
      cache.py         component cache: COMPONENT_DEPS sets, _component_hash, _pkg_rev /
                       _version_tag, fips_cache
      mdm.py           ModelDataMismatchMixin: get_modeldata_mismatch, multiplicative scale,
                       per-obs (subhour) std
      bias.py          BiasMixin: get_bias, get_bias_jacobian, get_site_group
      coverage.py      check_state_cells; CoverageFilterMixin (Jacobian coverage filter,
                       reconstruct_posterior)
      report.py        ReportingMixin: domain totals, summarize, desroziers, plot_*
      priors.py        prior flux construction
      covariances.py   covariance construction (kernels, scaling)
      data.py          inversion-time data loading
      background.py    inversion background model
      viz.py           inversion-specific plotting
      sweep.py         Sweep / run_sweep_job / SweepResults — hyperparam sweeps

What ``slv.inversion`` exports:

.. code-block:: python

   from slv.inversion import (
       InversionConfig, SLVMethaneInversion,
       Sweep, run_sweep_job, SweepResults,
   )

The component cache
~~~~~~~~~~~~~~~~~~~

``@fips_cache`` (``inversion/cache.py``) writes each component to
``{cache_dir}/.fips/{version_tag}/{component}/{hash}.pkl``. ``version_tag`` is the
``git describe`` rev of the editable fips and pystilt checkouts (``_version_tag`` and
``_pkg_rev``), so committing or editing those packages busts the cache automatically,
with no reinstall. A copy under ``site-packages`` (the uv ``.venv``, which sits inside
the slv repo, or the pixi env) uses its metadata version instead, written as the
release's tag (``v0.1.0b9``): ``git describe`` there would answer with slv's rev, and a
release names the same folder installed or checked out.

``get_inputs`` raises when the prior's cells and the Jacobian's columns differ (stale
caches after a grid change).

Components other than the Jacobian also hash the *release* of lair and uataq
(``COMPONENT_PACKAGES``, ``_release``: an editable checkout's static pyproject version
or latest git tag, else the installed version). A release, not an edit, rebuilds the
obs, prior, MDM and constant.

Input files are keyed by what is in them, never by path alone: ``mobile_obs_key`` and
``background_kwargs_key`` (the CT-STILT ``csv_path``) are path + size + mtime
(``config._file_key``), and the Jacobian and MDM pass ``fips_cache(..., key=...)`` the
footprints they use (``_footprint_key``: per project, the footprint settings folder and
how many selected receptors have a footprint), so it lists the project on every cached
run.

The cache key does not hold slv's own rev (so a change to slv's component-building
logic is forced with ``config.cache_overwrite``, or re-keyed by bumping its number in
``DEFAULT_COMPONENT_REVISIONS``, as AGENTS.md says). When you cut a version, reinstall
the editable packages (``uv sync`` or ``pip install -e``) so that ``importlib.metadata``
matches the source: this does not affect the cache key but keeps ``pip show`` and
other version consumers honest.

Notes on the pipeline
~~~~~~~~~~~~~~~~~~~~~

- Mobile (TRAX) obs are keyed by PYSTILT location_id, not site. Anything keyed by
  site must resolve through ``SLVMethaneInversion.obs_sites`` (receptor -> the single
  mobile site in ``config.sites``); the MDM and the bias Jacobian do. The rolling
  background uses stationary sites only (``split_sites``; override with
  ``background_kwargs={"background_sites": [...]}``) and each obs takes its hour's
  value; ``get_slv_subhour_std`` skips mobile sites.
- ``get_forward_operator`` is a thin wrapper: ``_get_flux_jacobian`` is the cached
  (``forward_operator``) flux Jacobian; the bias block is rebuilt every run.
  ``get_prior`` memoizes ``_build_prior`` per pipeline; ``_multiplicative_scale``
  reuses both.
- ``InversionConfig.__post_init__`` validates flux_freq, bias_grouping, background,
  prior, mdm_config and domain; setting any field drops the cached derived values.
- ``InversionConfig.mdm_config`` keys must be names in ``DEFAULT_MDM_CONFIG`` (part,
  instr, aggr, subhour, bg, transport); ``get_mdm_comp_configs`` raises on anything
  else. Retired keys (``transport_pbl``, ``transport_wind``) used to be ignored
  silently, leaving the MDM cache hash unchanged.
- ``get_slv_observations`` loads the TRAX route points only when a mobile site is in
  ``sites``; tower-only runs need no TRAX data.
- ``InversionConfig.stilt_project`` defaults to ``SLV_STILT_DIR`` if set, else
  ``DEFAULT_STILT_PROJECT`` (the CHPC paper-1 PYSTILT project), so the config stays
  usable without the environment variable.
- ``load_epa_prior`` reads EPA's express product either way
  (``EPAv2(express=True, scale_by_month=not express)``), so the monthly default and
  paper 1's annual ``express=True`` hold the same emissions (27 sectors including
  post-meter, 2012-2020); it relies on lair >= v2026.12.6 holding sectors without
  monthly scale factors at their annual rate.

Writing the docs
----------------

Examples run when the docs build, so they show real output, and the build fails when
one breaks.

**A figure in a docstring:** put a ``.. plot::`` directive in the Examples section.
Its code runs (doctest ``>>>`` lines work), and the figure appears under it.

.. code-block:: rst

   Examples
   --------
   .. plot::

      >>> import matplotlib.pyplot as plt
      >>> _ = plt.plot([0, 1, 2], [0, 1, 4])

**Code in an** ``.rst`` **page:** a ``.. ipython:: python`` block runs, and shows each
line with its output, as in an IPython session. Changing a ``.. code-block:: python`` to
``.. ipython:: python`` is enough to make it run. For a figure, put ``@savefig name.png``
on the line above the plotting call.

.. code-block:: rst

   .. ipython:: python

      import matplotlib.pyplot as plt
      import slv

      slv.__version__

      @savefig squares.png width=5in
      plt.plot([0, 1, 2], [0, 1, 4]);

A block that raises or warns fails the build; ``:okexcept:`` or ``:okwarning:`` under
the directive allows one that is meant to. All ``.rst`` pages share one session, so
each page imports and defines what it uses.

**When they run:** ``just build-docs`` runs everything, as CI does; ``just docs-serve``
reruns a page's blocks whenever that page is rebuilt.

**Notebooks, and code cells in Markdown pages,** are off in this project.
``copier update --data docs_notebooks=true`` turns them on; it adds myst-nb, and
Jupyter with it, to the dev tools.

**The API reference** has a page for each class, with tables of its attributes and
methods, and a page for each member; a subclass lists what it defines and links to what
it inherits (``docs/_templates/autosummary/`` and ``docs/_ext/api_pages.py``, from
python-template). The extension also drops from a class docstring the ``Methods``
section and the ``Attributes`` entries that the tables list. A property or attribute's
page is where its See Also and Examples go, as in pandas: give each public one a
docstring that says what it is, and add See Also or Examples where they help. One with
no docstring shows an empty row.

Notes on the tests
------------------

- The pixi env pins the no-MPI ESMF build (``build = "nompi*"``): with the MPICH build,
  the MPI start-up on ``import xesmf`` sometimes killed the whole pytest process (exit
  143, no traceback), about half the runs until 2026-10-05.
