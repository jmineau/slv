"""
The SLV methane inversion: functions that build the fips :class:`~fips.Problem` and solve it.

:func:`build_problem` loads (or builds) each component -- the obs, the prior, the
background, the flux Jacobian, the prior error and the model-data mismatch -- and puts
them together as one problem; :func:`run` solves it and reports. Each ``get_*`` function
takes the :class:`~slv.inversion.config.InversionConfig` first and is cached by
:func:`~slv.inversion.cache.fips_cache`. The model-data mismatch, the coverage filter
and the reports live in :mod:`.mdm`, :mod:`.coverage` and :mod:`.report`.
"""

import time
from pathlib import Path
from typing import Any

import fips
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fips import Covariance, Independent, Operator, Posterior, Problem, Space, Vector

from slv.inversion import report
from slv.inversion.background import get_slv_background
from slv.inversion.cache import fips_cache
from slv.inversion.config import InversionConfig
from slv.inversion.covariances import build_prior_error
from slv.inversion.coverage import check_state_cells, covered_state
from slv.inversion.data import get_slv_observations, split_sites
from slv.inversion.mdm import get_modeldata_mismatch

#: Decimal places the float labels (the cells' lat and lon) are rounded to before the
#: problem is built. The prior's cells come from a regrid and the Jacobian's from the
#: PYSTILT grid, so the same cell can differ in its last digits; fips 0.1 rounded every
#: float label to 6 decimals, and fips 0.2 rounds only on request.
LABEL_DECIMALS = 6


def _rounded(index: pd.Index) -> pd.Index:
    """``index`` with its float levels rounded to :data:`LABEL_DECIMALS` places."""
    return Space({"labels": index}).round_labels(LABEL_DECIMALS).select("labels")


def stack_jacobians(parts: list[tuple[Path, Operator]]) -> Operator:
    """
    Row-stack the per-project flux Jacobians into one.

    ``parts`` pairs each STILT project with the Jacobian built from it. Every project is
    aggregated onto the same state grid and flux time bins, so the columns agree; they are
    still aligned on their union (missing cells are zero sensitivity) rather than assumed.

    A row -- an ``(obs_location, obs_time)`` -- appearing in two projects is an error: it
    means the same receptor was simulated in both, and there is no principled way to pick
    one footprint. Remove it from one project's receptors, or drop that project.
    """
    rows = [H.rows.select("concentration") for _, H in parts]
    stacked = rows[0].append(rows[1:])
    dup = stacked.duplicated(keep=False)
    if dup.any():
        owners = [str(p) for (p, H) in parts for _ in range(H.shape[0])]
        clash = pd.Series(owners, index=stacked)[dup]
        example = clash.index[0]
        raise ValueError(
            f"{int(dup.sum() // 2)} obs have footprints in more than one STILT project, e.g. "
            f"{example} in {sorted(set(clash.loc[[example]]))}. Each obs must come from "
            "exactly one project."
        )
    cols = parts[0][1].cols.select("flux")
    for _, H in parts[1:]:
        cols = cols.union(H.cols.select("flux"), sort=False)
    blocks = []
    for _, H in parts:
        where = cols.get_indexer(H.cols.select("flux"))
        m = H.sparse().tocoo()
        blocks.append(
            sp.csr_array((m.data, (m.row, where[m.col])), shape=(H.shape[0], len(cols)))
        )
    return Operator(
        sp.csr_array(sp.vstack(blocks)),
        Space({"concentration": stacked}),
        Space({"flux": cols}),
        name="jacobian",
    )


def project_jacobian(
    project: Any,
    sel: pd.DataFrame,
    target: Any,
    flux_bins: pd.IntervalIndex,
    *,
    location_mapper: dict[str, str] | None = None,
    workers: int | None = None,
    threshold: float | None = 1e-15,
    sparse: bool = True,
) -> Operator | None:
    """
    One STILT project's flux Jacobian, from ``stilt.Project.jacobian``.

    ``sel`` is the rows of ``project.simulations`` to use, all of one variant. Each
    footprint is summed onto ``target`` per flux bin. Rows are ``(obs_location,
    obs_time)``: the receptor's location id, renamed by ``location_mapper`` when it names a
    site, and its time; they are the one block ``concentration``. Columns are the target's
    cells and the bins' left edges (``(lon, lat, time)`` for a longitude/latitude grid), the
    one block ``flux``. Entries below ``threshold`` are zero.

    Every receptor with a footprint gets a row. One whose footprint is empty (all its
    particles left the domain) or misses the target gets a row of zeros: it saw no
    emissions, which is a real observation. A receptor with no footprint (not run yet, or
    failed) gets no row, and :func:`build_problem` drops its obs. Returns None when no
    receptor has a footprint.
    """
    J = project.jacobian(sel, target, flux_bins, workers=workers)
    if J.missing:
        print(
            f"  {len(J.missing)} receptors have no footprint yet; their obs are dropped"
        )
    data = J.data.tocsr(copy=True)
    if threshold is not None:
        data.data[np.abs(data.data) < threshold] = 0.0
        data.eliminate_zeros()
    receptors = np.asarray(J.receptors)
    if J.empty:
        print(f"  {len(J.empty)} receptors have an empty footprint; kept as zero rows")
        zeros = sp.csr_array((len(J.empty), data.shape[1]), dtype=data.dtype)
        data = sp.vstack([data, zeros], format="csr")
        receptors = np.concatenate(
            [receptors, np.asarray(J.empty, dtype=receptors.dtype)]
        )
    if not len(receptors):
        return None

    info = sel.drop_duplicates("receptor").set_index("receptor").loc[receptors]
    mapper = location_mapper or {}
    locations = [mapper.get(lid, lid) for lid in info["location"].astype(str)]
    rows = pd.MultiIndex.from_arrays(
        [locations, pd.DatetimeIndex(info["time"])], names=["obs_location", "obs_time"]
    )
    cells = [n for n in J.columns.names if n != "time"]
    cols = J.columns.reorder_levels([*cells, "time"])
    kept = sp.csr_array(data)
    return Operator(
        kept if sparse else kept.toarray(),
        Space({"concentration": rows}),
        Space({"flux": cols}),
        name="jacobian",
    )


@fips_cache("obs")
def get_obs(config: InversionConfig) -> pd.Series:
    """
    The observations (CH4, ppm), indexed by ``(obs_location, obs_time)``, named ``"obs"``.

    Every obs the config's sites and filters give, over the config's whole time range;
    :func:`filter_state_space` trims them afterwards.
    """
    data = get_slv_observations(
        sites=config.sites,
        site_config=config.site_config,
        time_range=config.time_range,
        subset_hours=config.subset_hours,
        filter_pcaps=config.filter_pcaps,
        filter_spikes=config.filter_spikes,
        spike_percentile=config.spike_percentile,
        num_processes=config.num_processes,
        mobile_obs=config.mobile_obs,
        utc_offset=config.utc_offset,
    )
    return data["CH4"].rename("obs")


@fips_cache("prior")
def get_prior(config: InversionConfig) -> pd.Series:
    """The prior flux (umol/m2/s) on the config's grid and flux times, named ``"prior"``."""
    from slv.inversion.priors import get_slv_prior

    # get_slv_prior asks its loaders for no regridder, so it returns the Series; the
    # loaders' return type is not typed per return_regridder
    prior: pd.Series = get_slv_prior(  # pyrefly: ignore[bad-assignment]
        prior=config.prior,
        out_grid=config.grid,
        flux_times=config.flux_times,
        flux_freq=config.flux_freq,
        bbox=config.bbox,
        **config.prior_kwargs,
    )
    return prior.rename("prior")


@fips_cache("forward_operator", key=lambda config, obs: _footprint_key(config, obs))
def get_flux_jacobian(config: InversionConfig, obs: pd.Series) -> Operator:
    """
    The flux Jacobian from the PYSTILT footprints (cached as ``forward_operator``).

    Its rows are the receptors (``(obs_location, obs_time)``) with a footprint at the
    locations of ``obs``, the block ``concentration``; its columns the state grid's cells
    and flux times, the block ``flux``. ``config.stilt_project`` may name several
    projects (``config.stilt_projects``): the UOU and DAQ footprints live in the paper-1
    project and the TRAX ones in their own. Each project contributes the rows for the
    obs it has footprints for, and the rows are stacked into one Jacobian. All projects
    are aggregated onto the same state grid and flux time bins, so their columns line up.
    The cache key also sees the footprints (:func:`_footprint_key`), so simulations run
    since a Jacobian was cached rebuild it.
    """
    obs_locations = set(obs.index.get_level_values("obs_location"))
    parts: list[tuple[Path, Operator]] = []
    for project in config.stilt_projects:
        H = _project_jacobian(config, project, obs_locations)
        if H is not None:
            parts.append((project, H))
    if not parts:
        raise ValueError(
            f"None of the STILT projects {[str(p) for p in config.stilt_projects]} "
            "has simulations matching the obs."
        )
    if len(parts) == 1:
        return parts[0][1]
    return stack_jacobians(parts)


def _footprint_key(
    config: InversionConfig, obs: pd.Series
) -> list[tuple[str | None, int]]:
    """
    What the Jacobian's cache key sees of the footprints: per project, the folder its
    footprints are read from and how many of the selected receptors have one.

    The count changes when simulations missing from a cached Jacobian are run; the
    folder (``settings=<variant>-<hash>``, named by the transport, met and footprint
    settings) changes when they are rerun with other settings, such as a new met crop.
    A project with no matching simulation is left out.
    """
    obs_locations = set(obs.index.get_level_values("obs_location"))
    key = []
    for project in config.stilt_projects:
        selected = _select_simulations(config, project, obs_locations)
        if selected is None:
            continue
        proj, sel, _ = selected
        folder = proj.output.folder("footprints", proj.variants[sel["variant"].iloc[0]])
        count = int(proj.status(sel)["footprint"].sum())
        print(f"  {project.name}: {count} of {len(sel)} receptors have a footprint")
        key.append((None if folder is None else folder.name, count))
    return key


def _project_jacobian(
    config: InversionConfig, project: Path, obs_locations: set
) -> Operator | None:
    """
    One project's Jacobian rows for the obs in ``obs_locations``, or None if it has
    no simulation matching any of them.
    """
    selected = _select_simulations(config, project, obs_locations)
    if selected is None:
        return None
    proj, sel, location_mapper = selected
    return project_jacobian(
        proj,
        sel,
        config.state_grid,
        config.flux_time_bins,
        location_mapper=location_mapper,
        workers=config.num_processes,
        sparse=config.sparse_jacobian,
    )


def _select_simulations(
    config: InversionConfig, project: Path, obs_locations: set
) -> tuple[Any, pd.DataFrame, dict[str, str]] | None:
    """
    The project, its simulations for the obs in ``obs_locations`` (the variant, the
    flux period, the ``subset_hours``) and the location mapper, or None if no
    simulation matches.
    """
    import stilt

    from slv.inversion.config import build_location_site_map

    proj = stilt.Project(project)
    sims = proj.simulations
    sim_locations = set(sims["location"].astype(str))

    # Build location mapper from all simulations in the project.
    # Must happen before filtering so stationary sites can be resolved.
    # Mobile location ids won't appear in the mapper (no site_config entry),
    # so mapper.get(lid, lid) returns the location id itself for mobile sims.
    # The auto-built mapper is not written back to the config: it is only built on a
    # cache miss, and a config changed mid-run would change its sweep config_id.
    location_mapper = config.location_site_map
    if not location_mapper:
        location_mapper = build_location_site_map(
            list(sim_locations), config.site_config
        )
        print(
            f"  {project.name}: auto-generated location mapper for "
            f"{len(location_mapper)} sites"
        )

    # Filter to simulations relevant for this obs set.
    # For stationary sims: mapper resolves location id → site name → in obs.
    # For mobile sims: location id not in mapper → falls back to the location id
    #   itself, which IS the obs_location for mobile sites.
    relevant_location_ids = {
        lid for lid in sim_locations if location_mapper.get(lid, lid) in obs_locations
    }
    if not relevant_location_ids:
        print(f"  {project.name}: no simulations match the obs; skipped")
        return None

    variant = _resolve_variant(config, proj, project)
    bins = config.flux_time_bins
    times = pd.DatetimeIndex(sims["time"])
    keep = (
        (sims["variant"] == variant).to_numpy()
        & sims["location"].astype(str).isin(relevant_location_ids).to_numpy()
        & (times >= bins[0].left)
        & (times <= bins[-1].right)
        & times.hour.isin(config.subset_hours_utc)
    )
    sel = sims[keep]
    if sel.empty:
        print(f"  {project.name}: no simulations in the time range; skipped")
        return None
    return proj, sel, location_mapper


def _resolve_variant(config: InversionConfig, proj, project: Path) -> str:
    """
    ``config.variant`` if set (this project must define it), else the project's one
    variant with a footprint.
    """
    variants = proj.variants
    variant = config.variant
    if variant is not None:
        names = set(variants)
        if variant not in names:
            raise ValueError(
                f"Variant {variant!r} is not in the STILT project {project} "
                f"(it has {sorted(names)}). With several projects, config.variant must "
                "name a variant every project has, or be None."
            )
        return variant
    with_footprint = [n for n, v in variants.items() if v.footprint is not None]
    if len(with_footprint) != 1:
        raise ValueError(
            f"The STILT project {project} has {len(with_footprint)} variants with a "
            f"footprint ({with_footprint}); set InversionConfig.variant to pick one."
        )
    print(f"  {project.name}: using variant {with_footprint[0]!r}")
    return with_footprint[0]


@fips_cache("prior_error")
def get_prior_error(config: InversionConfig, prior: pd.Series) -> Covariance:
    """
    The prior flux error covariance on the labels of ``prior``, the block ``flux``.

    See :func:`~slv.inversion.covariances.build_prior_error`. A bias block is added by
    :func:`build_problem`, per run.
    """
    return build_prior_error(
        prior,
        base_std=config.prior_base_std,
        std_frac=config.prior_std_frac,
        time_scale=config.prior_time_scale,
        spatial_scale=config.prior_spatial_scale,
    )


def obs_sites(config: InversionConfig, locations: pd.Index) -> pd.Index:
    """
    The site of each obs location, for site- and organization-keyed MDM and bias terms.

    A tower obs is keyed by its site; a mobile obs by its receptor's PYSTILT location_id,
    which belongs to the (single) mobile site in ``config.sites``.
    """
    locations = pd.Index(locations)
    known = locations.isin(config.site_config.index)
    if known.all():
        return locations
    _, mobile = split_sites(config.sites, config.site_config)
    if len(mobile) != 1:
        raise ValueError(
            f"{int((~known).sum())} obs are keyed by a location that is not a site, and "
            f"config.sites has {len(mobile)} mobile sites ({mobile}) to assign them to; "
            "exactly one is supported."
        )
    return pd.Index(np.where(known, locations, mobile[0]))


@fips_cache("constant")
def get_constant(config: InversionConfig, obs: pd.Series) -> pd.Series:
    """
    Background concentration for each obs (``config.background``), named ``"background"``.

    Obs whose background is missing (outside the ct_stilt product, rolling-baseline
    gaps) are left out rather than filled; :func:`build_problem` drops those obs.
    """
    obs_times = obs.index.get_level_values("obs_time").unique()
    data = get_slv_background(
        background=config.background,
        obs_times=obs_times,
        sites=config.sites,
        site_config=config.site_config,
        time_range=config.time_range,
        filter_pcaps=config.filter_pcaps,
        num_processes=config.num_processes,
        **config.background_kwargs,
    )

    # Align background to obs index
    data = (
        obs.to_frame("obs")
        .reset_index()
        .join(data, on="obs_time")
        .set_index(["obs_location", "obs_time"])["concentration"]
    )
    return data.dropna().rename("background")


def filter_state_space(
    config: InversionConfig, obs: pd.Series, prior: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Trim obs and prior to ``config.time_range``, then drop the obs of sparse intervals.

    Objects loaded from a wide-range cache are sliced to the run's time window before
    anything is built from them. With ``config.min_obs_per_interval`` above 1, the obs
    of a flux interval with fewer obs than that are dropped (the interval's fluxes stay
    in the state, constrained by the prior alone).
    """
    tstart, tend = config.time_range
    obs_times = obs.index.get_level_values("obs_time")
    obs = obs[(obs_times >= tstart) & (obs_times < tend)]
    prior_times = prior.index.get_level_values("time")
    prior = prior[(prior_times >= tstart) & (prior_times < tend)]

    if config.min_obs_per_interval > 1:
        # The intervals run from each flux time to the next, closed on the right, as
        # fips 0.1's FluxInversionPipeline made them
        times = prior.index.get_level_values("time").unique().sort_values()
        offset = pd.tseries.frequencies.to_offset(config.flux_freq)
        bins = pd.interval_range(
            start=times.min(), end=times.max() + offset, freq=config.flux_freq
        )
        interval = pd.Index(pd.cut(obs.index.get_level_values("obs_time"), bins=bins))
        counts = (
            pd.Series(1, index=obs.index)
            .groupby(interval, observed=True)
            .transform("count")
        )
        obs = obs[(counts >= config.min_obs_per_interval).to_numpy()]
    return obs, prior


def get_bias(config: InversionConfig, obs_space: Space) -> tuple[Vector, Operator]:
    """
    The bias block: one unknown offset per group of obs, by ``config.bias_grouping``.

    ``None`` or ``"time"``: one per flux interval; ``"site"``: per interval and site;
    ``"site_group"``: per interval and site organization. A mobile receptor belongs to
    its mobile site (:func:`obs_sites`, after ``config.location_site_map``). Only groups
    with obs get an offset; each is labeled by its interval's start (``obs_time``) and
    its site (``obs_location``) or organization (``site_group``). Built every run from the
    obs, which change with the obs filters the cache key leaves out.

    Returns
    -------
    prior : Vector
        Zeros, the one block ``bias``.
    forward : Operator
        The obs-by-bias block of the forward operator (1 where an obs has the offset).
    """
    grouping = config.bias_grouping
    by: list[Any] = [config.flux_time_bins.rename("obs_time")]
    if grouping in ("site", "site_group"):
        locations = obs_space.select("concentration").get_level_values("obs_location")
        unique = locations.unique()
        mapper = config.location_site_map or {}
        sites = obs_sites(config, unique.map(lambda loc: mapper.get(loc, loc)))
        if grouping == "site":
            by.append(pd.Series(sites, index=unique, name="obs_location"))
        else:
            organization = config.site_config.organization.to_dict()
            groups = [organization.get(site, "unknown") for site in sites]
            by.append(pd.Series(groups, index=unique, name="site_group"))
    elif grouping not in (None, "time"):
        raise ValueError(
            f"Unknown bias_grouping: {grouping}. "
            f"Expected None, 'time', 'site', or 'site_group'"
        )
    return fips.offset_block(obs_space, by=by, block="bias")


def build_problem(config: InversionConfig) -> Problem:
    """
    The fips problem for ``config``: every component loaded or built, then put together.

    Steps: the obs, prior and background (:func:`get_obs`, :func:`get_prior`,
    :func:`get_constant`); :func:`filter_state_space`; the flux Jacobian
    (:func:`get_flux_jacobian`), checked to cover the prior's cells
    (:func:`~slv.inversion.coverage.check_state_cells`); the prior error; the obs without
    a background are dropped; the model-data mismatch on the obs that are left; the
    bias block when ``config.bias_std`` is set; and the means of the obs over
    ``config.aggregate_obs`` (a day) when it is set. The obs whose receptor has no
    footprint are dropped too (a receptor with an empty footprint has a row of zeros and
    keeps its obs; :func:`project_jacobian`). The Jacobian's rows for the obs are placed
    on them, and the float labels are rounded to :data:`LABEL_DECIMALS`.

    Returns
    -------
    Problem
        The state has the block ``flux`` (and ``bias``), the obs the block
        ``concentration``; the background is the offset.
    """
    step_start = time.perf_counter()
    print("Loading observations...")
    obs = get_obs(config)
    print(f"Observations loaded in {time.perf_counter() - step_start:.2f}s")

    step_start = time.perf_counter()
    print("Loading prior...")
    full_prior = get_prior(config)
    print(f"Prior loaded in {time.perf_counter() - step_start:.2f}s")

    step_start = time.perf_counter()
    print("Loading constant term (if any)...")
    constant = get_constant(config, obs)
    print(f"Constant term loaded in {time.perf_counter() - step_start:.2f}s")

    step_start = time.perf_counter()
    print("Optionally, filtering state space...")
    obs, prior = filter_state_space(config, obs, full_prior)
    print(
        f"Optionally, state space filtered in {time.perf_counter() - step_start:.2f}s"
    )

    step_start = time.perf_counter()
    print("Building forward operator...")
    jacobian = get_flux_jacobian(config, obs)
    check_state_cells(prior.index, jacobian.cols.index)
    print(f"Forward operator built in {time.perf_counter() - step_start:.2f}s")

    prior = prior.set_axis(_rounded(prior.index))
    step_start = time.perf_counter()
    print("Building prior error covariance...")
    prior_error = get_prior_error(config, prior)
    print(f"Prior error covariance built in {time.perf_counter() - step_start:.2f}s")

    # The obs whose background is missing are dropped rather than given a zero one
    have_bg = obs.index.isin(constant.index)
    if not have_bg.all():
        print(
            f"  Dropping {int((~have_bg).sum())}/{len(have_bg)} obs "
            "with missing background"
        )
        obs = obs[have_bg]
    # The obs whose receptor has no footprint (not run, or failed) are dropped: a row
    # of zeros would say they saw no emissions
    receptors = jacobian.rows.select("concentration")
    have_footprint = obs.index.isin(receptors)
    if not have_footprint.all():
        print(
            f"  Dropping {int((~have_footprint).sum())}/{len(have_footprint)} obs "
            "whose receptor has no footprint"
        )
        obs = obs[have_footprint]

    step_start = time.perf_counter()
    print("Building model-data mismatch covariance...")
    mdm = get_modeldata_mismatch(
        config,
        obs,
        prior=full_prior,
        jacobian=jacobian,
        constant=constant,
        obs_sites=obs_sites(config, obs.index.get_level_values("obs_location")),
    )
    print(
        "Model-data mismatch covariance built in "
        f"{time.perf_counter() - step_start:.2f}s"
    )

    obs_vector = Vector.from_series(obs, block="concentration")
    state = Vector.from_series(prior, block="flux")
    offset = Vector.from_series(constant.reindex(obs.index), block="concentration")
    in_obs = receptors.isin(obs.index)
    blocks: dict[tuple[str, str], Any] = {
        ("concentration", "flux"): (
            jacobian.sparse()[in_obs],
            receptors[in_obs],
            _rounded(jacobian.cols.select("flux")),
        )
    }
    prior_cov: Any = prior_error
    if config.bias_std is not None:
        bias_prior, bias_forward = get_bias(config, obs_vector.space)
        state = Vector.from_blocks({"flux": state, "bias": bias_prior}, name="prior")
        blocks[("concentration", "bias")] = bias_forward
        prior_cov = Covariance.block_diag(
            flux=prior_error, bias=Independent().with_std(config.bias_std)
        )
    problem = Problem(
        prior=state,
        prior_cov=prior_cov,
        obs=obs_vector,
        obs_cov=mdm,
        forward=Operator.from_blocks(
            blocks, rows=obs_vector.space, cols=state.space, name="forward_operator"
        ),
        offset=offset,
    )

    step_start = time.perf_counter()
    print("Optionally, aggregating observation space (if needed)...")
    if config.aggregate_obs:
        freq = config.aggregate_obs
        if not isinstance(freq, str):
            raise TypeError(
                f"aggregate_obs is a pandas frequency such as '1D', got {freq!r}."
            )
        levels = [n for n in obs.index.names if n != "obs_time"]
        W = fips.aggregate.mean(
            problem.obs.space,
            by=[*levels, pd.Grouper(level="obs_time", freq=freq)],
            blocks="concentration",
        )
        problem = problem.aggregate(W)
    print(
        "Optionally, observation space aggregated in "
        f"{time.perf_counter() - step_start:.2f}s"
    )
    return problem


def run(config: InversionConfig, problem: Problem | None = None) -> Posterior:
    """
    Build the problem, solve it, summarize and plot.

    Steps: :func:`build_problem` (cached components), the Jacobian coverage filter when
    ``config.jacobian_coverage_percentile`` is set (the poorly covered cells are held at
    their prior: :func:`~slv.inversion.coverage.covered_state`), the solve
    (``config.gamma`` scales the obs error), :func:`~slv.inversion.report.summarize`,
    then the plots the config turns on.

    Parameters
    ----------
    config : InversionConfig
        The inversion.
    problem : Problem, optional
        The problem to solve; :func:`build_problem` of ``config`` by default. Pass the
        one you built to keep the whole state: with the coverage filter on, the
        posterior's problem holds only the kept cells, and
        ``post.estimate.combine_first(problem.prior)`` is the estimate on every cell.

    Returns
    -------
    Posterior
        The solution; ``post.problem`` is the problem that was solved.
    """
    total_start = time.perf_counter()
    print("Getting problem inputs...")
    if problem is None:
        problem = build_problem(config)
    print(f"Inputs prepared in {time.perf_counter() - total_start:.2f}s")

    solved = problem
    if config.jacobian_coverage_percentile is not None:
        step_start = time.perf_counter()
        print("Filtering cells by Jacobian coverage...")
        keep = covered_state(problem, config.jacobian_coverage_percentile)
        if len(keep) < len(problem.prior):
            solved = problem.restrict(keep)
        print(f"Cells filtered in {time.perf_counter() - step_start:.2f}s")

    if config.plot_inputs:
        step_start = time.perf_counter()
        report.plot_inputs(config, problem, solved)
        print(f"Inputs plotted in {time.perf_counter() - step_start:.2f}s")

    print("Solving...")
    step_start = time.perf_counter()
    if config.gamma is None:
        post = fips.solve(solved)
    else:
        post = fips.solve(solved, gamma=config.gamma)
    print(f"Solve completed in {time.perf_counter() - step_start:.2f}s")

    print("Calculating summary...")
    step_start = time.perf_counter()
    report.summarize(config, post, problem)
    print(f"Summary calculated in {time.perf_counter() - step_start:.2f}s")

    if config.plot_results:
        step_start = time.perf_counter()
        report.plot_results(config, post, problem)
        print(f"Results plotted in {time.perf_counter() - step_start:.2f}s")

    if config.plot_diagnostics:
        step_start = time.perf_counter()
        report.plot_diagnostics(config, post, problem)
        print(f"Diagnostics plotted in {time.perf_counter() - step_start:.2f}s")

    print(f"Total pipeline time: {time.perf_counter() - total_start:.2f}s")
    return post
