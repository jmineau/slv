"""The SLV methane inversion pipeline: builds the fips inputs and runs the solve.

The model-data mismatch, bias block, coverage filter, reporting and component cache live
in their own modules (:mod:`.mdm`, :mod:`.bias`, :mod:`.coverage`, :mod:`.report`,
:mod:`.cache`); :class:`SLVMethaneInversion` combines them.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from fips import Block, CovarianceMatrix, ForwardOperator, MatrixBlock, Vector
from fips.aggregators import ObsAggregator
from fips.covariance import DiagonalError
from fips.problems.flux import FluxInversionPipeline, JacobianBuilder
from fips.problems.flux.problem import FluxProblem

from slv.inversion.background import get_slv_background
from slv.inversion.bias import BiasMixin
from slv.inversion.cache import (  # noqa: F401 -- re-exported for existing imports
    DEFAULT_COMPONENT_DEPS,
    _component_hash,
    _pkg_rev,
    _version_tag,
    fips_cache,
)
from slv.inversion.covariances import build_prior_error
from slv.inversion.coverage import CoverageFilterMixin, check_state_cells
from slv.inversion.data import get_slv_observations, split_sites
from slv.inversion.mdm import ModelDataMismatchMixin
from slv.inversion.priors import get_slv_prior
from slv.inversion.report import ReportingMixin


class SLVMethaneInversion(
    ReportingMixin,
    CoverageFilterMixin,
    BiasMixin,
    ModelDataMismatchMixin,
    FluxInversionPipeline,
):
    """SLV-specific implementation of the flux inversion pipeline.

    Supports optional bias correction via config.bias_std and config.bias_grouping.
    When bias_std is set, augments the state vector with bias terms that can be
    grouped by time (default), site, or site organization.
    """

    #: Maps each cache component to the InversionConfig fields it depends on. The bias
    #: block is indexed by ``sites`` under the "site"/"site_group" groupings, so the
    #: prior and prior error key on it too.
    COMPONENT_DEPS: dict[str, frozenset[str]] = {
        **DEFAULT_COMPONENT_DEPS,
        "prior": DEFAULT_COMPONENT_DEPS["prior"]
        | {"bias_std", "bias_grouping", "sites"},
        "forward_operator": DEFAULT_COMPONENT_DEPS["forward_operator"]
        | {"bias_std", "bias_grouping"},
        "prior_error": DEFAULT_COMPONENT_DEPS["prior_error"]
        | {"bias_std", "bias_grouping", "sites"},
    }

    def get_inputs(self) -> dict[str, Any]:
        """fips' inputs, checked that the prior and the Jacobian share their cells."""
        inputs = super().get_inputs()
        check_state_cells(inputs["prior"], inputs["forward_operator"])
        return inputs

    @fips_cache(Vector, "obs")
    def get_obs(self) -> Vector:
        """Passes just the obs attributes to the pure obs function."""
        return Vector(
            name="obs",
            data=Block(
                name="concentration",
                data=get_slv_observations(
                    sites=self.config.sites,
                    site_config=self.config.site_config,
                    time_range=self.config.time_range,
                    subset_hours=self.config.subset_hours,
                    filter_pcaps=self.config.filter_pcaps,
                    filter_spikes=self.config.filter_spikes,
                    spike_percentile=self.config.spike_percentile,
                    num_processes=self.config.num_processes,
                    mobile_obs=self.config.mobile_obs,
                    utc_offset=self.config.utc_offset,
                ),
            ),
        )

    def get_prior(self) -> Vector:
        """Get the prior vector, optionally including bias terms.

        Returns a single-block flux prior if bias_std is None, otherwise
        returns a multi-block [flux, bias] prior. Built (or loaded) once per pipeline:
        the multiplicative MDM asks for it again, which with ``cache=False`` used to
        rebuild it.
        """
        if getattr(self, "_prior", None) is None:
            self._prior = self._build_prior()
        return self._prior

    @fips_cache(Vector, "prior")
    def _build_prior(self) -> Vector:
        prior = get_slv_prior(
            prior=self.config.prior,
            out_grid=self.config.grid,
            flux_times=self.config.flux_times,
            flux_freq=self.config.flux_freq,
            bbox=self.config.bbox,
            **self.config.prior_kwargs,
        )
        flux_prior = Vector(name="prior", data=Block(name="flux", data=prior))

        # Add bias block if enabled
        if self.config.bias_std is None:
            return flux_prior
        bias_blk = Block(self.get_bias(), name="bias")
        return Vector(name="prior", data=[flux_prior.blocks["flux"], bias_blk])

    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        """Get the forward operator, optionally including bias Jacobian.

        Returns a single-block flux Jacobian if bias_std is None, otherwise
        returns a multi-block [flux_jac | bias_jac] operator.

        Only the flux Jacobian is cached. The bias block is a cheap one-hot map indexed
        by the obs, which change with the obs filters the cache key leaves out, so it is
        rebuilt every run (a cached bias block went stale when those filters changed).
        """
        flux = self._get_flux_jacobian(obs)
        self._flux_jacobian = flux
        if self.config.bias_std is None:
            return flux
        # a Jacobian cached before the bias block was split out still carries one
        flux_blk = flux.blocks["concentration", "flux"]
        bias_blk = MatrixBlock(
            self.get_bias_jacobian(obs, prior), "concentration", "bias"
        )
        return ForwardOperator([flux_blk, bias_blk])

    @fips_cache(ForwardOperator, "forward_operator")
    def _get_flux_jacobian(self, obs: Vector) -> ForwardOperator:
        """The flux Jacobian from the PYSTILT footprints (cached as ``forward_operator``)."""
        from stilt import Model, SimID

        from slv.inversion.config import build_location_site_map

        model = Model(self.config.stilt_project)

        # Build location mapper from all simulations in the project.
        # Must happen before filtering so stationary sites can be resolved.
        # Mobile location_ids won't appear in the mapper (no site_config entry),
        # so mapper.get(lid, lid) returns the location_id itself for mobile sims.
        # The auto-built mapper is not written back to the config: it is only built on a
        # cache miss, and a config changed mid-run would change its sweep config_id.
        location_mapper = self.config.location_site_map
        if not location_mapper:
            all_location_ids = list({SimID(sid).location for sid in model.simulations})
            location_mapper = build_location_site_map(
                all_location_ids, self.config.site_config
            )
            print(f"Auto-generated location mapper for {len(location_mapper)} sites")

        # Filter to simulations relevant for this obs set.
        # For stationary sims: mapper resolves location_id → site name → in obs.
        # For mobile sims: location_id not in mapper → falls back to location_id
        #   itself, which IS the obs_location for mobile sites.
        obs_locations = set(obs.index.get_level_values("obs_location"))
        relevant_location_ids = {
            lid
            for lid in {SimID(sid).location for sid in model.simulations}
            if location_mapper.get(lid, lid) in obs_locations
        }

        # Resolve footprint name: None → finest (smallest xres) in project config
        footprint = self.config.footprint
        if footprint is None:
            foot_configs = model.config.footprints
            if not foot_configs:
                raise ValueError(
                    "No footprints configured in the STILT project. "
                    "Set InversionConfig.footprint explicitly."
                )
            footprint = min(foot_configs, key=lambda n: foot_configs[n].grid.xres)
            print(f"  Auto-selected finest footprint: '{footprint}'")

        # Build flux Jacobian
        jacobian_builder = JacobianBuilder(model)
        jacobian = jacobian_builder.build_from_target(
            self.config.state_grid,
            flux_times=self.config.flux_time_bins,
            footprint=footprint,
            location_ids=relevant_location_ids,
            subset_hours=self.config.subset_hours_utc,
            location_mapper=location_mapper,
            num_processes=self.config.num_processes,
            timeout=self.config.timeout,
            sparse=self.config.sparse_jacobian,
        )

        return ForwardOperator(jacobian)

    @fips_cache(CovarianceMatrix, "prior_error")
    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        """Get prior error covariance, optionally including bias error.

        Returns a single-block flux error if bias_std is None, otherwise
        returns a multi-block [flux_err, bias_err] covariance.
        """
        # Build flux error
        flux_prior = Vector(
            prior.blocks["flux"] if self.config.bias_std is not None else prior.data
        )
        S_0 = build_prior_error(
            flux_prior,
            base_std=self.config.prior_base_std,
            std_frac=self.config.prior_std_frac,
            time_scale=self.config.prior_time_scale,
            spatial_scale=self.config.prior_spatial_scale,
        )

        # Return flux-only error if no bias
        if self.config.bias_std is None:
            return CovarianceMatrix(name="prior_error", data=S_0)

        # Add bias error
        flux_err_blk = CovarianceMatrix(name="prior_error", data=S_0).blocks[
            "flux", "flux"
        ]
        bias_index = prior["bias"].index
        bias_err = DiagonalError(
            name="bias_error", variances=self.config.bias_std**2
        ).build(bias_index)
        bias_err_blk = MatrixBlock(bias_err, "bias", "bias")

        return CovarianceMatrix(name="prior_error", data=[flux_err_blk, bias_err_blk])

    def obs_sites(self, obs_index: pd.Index) -> pd.Index:
        """The site of each obs, for site- and organization-keyed MDM and bias terms.

        A tower obs is keyed by its site; a mobile obs by its receptor's PYSTILT location_id,
        which belongs to the (single) mobile site in ``config.sites``.
        """
        return self._resolve_sites(obs_index.get_level_values("obs_location"))

    def _resolve_sites(self, locations) -> pd.Index:
        """Sites for obs locations: a site stays itself, anything else is a receptor of
        the single mobile site in ``config.sites``."""
        locations = pd.Index(locations)
        known = locations.isin(self.config.site_config.index)
        if known.all():
            return pd.Index(locations)
        _, mobile = split_sites(self.config.sites, self.config.site_config)
        if len(mobile) != 1:
            raise ValueError(
                f"{int((~known).sum())} obs are keyed by a location that is not a site, and "
                f"config.sites has {len(mobile)} mobile sites ({mobile}) to assign them to; "
                "exactly one is supported."
            )
        return pd.Index(np.where(known, locations, mobile[0]))

    @fips_cache(Vector, "constant")
    def get_constant(self, obs: Vector) -> Vector:
        """Background concentration for each obs (``config.background``).

        Obs whose background is missing (outside the ct_stilt product, rolling-baseline
        gaps) are dropped here rather than filled: fips vectors reject NaN, and
        ``aggregate_obs_space`` reduces the obs to match.
        """
        obs_times = obs.data.index.get_level_values("obs_time").unique()
        data = get_slv_background(
            background=self.config.background,
            obs_times=obs_times,
            sites=self.config.sites,
            site_config=self.config.site_config,
            time_range=self.config.time_range,
            filter_pcaps=self.config.filter_pcaps,
            num_processes=self.config.num_processes,
            **self.config.background_kwargs,
        )

        # Align background to obs index
        data = (
            obs.data.reset_index()
            .join(data, on="obs_time", lsuffix="_obs")
            .set_index(["obs_location", "obs_time"])["concentration"]
        )

        # Drop obs whose background is missing (e.g. days outside the ct_stilt
        # product, or rolling-baseline edge gaps). fips Vectors reject NaN, and
        # aggregate_obs_space reduces obs to match so these obs are dropped rather
        # than back-filled to zero by the obs.index reindex downstream.
        data = data.dropna()

        return Vector(name="background", data=Block(name="concentration", data=data))

    def filter_state_space(self, obs: Vector, prior: Vector) -> tuple[Vector, Vector]:
        """Trim obs and prior to ``config.time_range``, then run interval filter.

        This ensures that objects loaded from a wide-range cache are sliced
        down to the current run's time window before any downstream builders
        (forward operator, covariances) see them.
        """
        tstart, tend = self.config.time_range

        # --- Filter obs by obs_time ---
        obs_series = obs.to_series()
        obs_times = obs_series.index.get_level_values("obs_time")
        obs_series = obs_series[(obs_times >= tstart) & (obs_times < tend)]
        obs = Vector(obs_series, name=obs.name)

        # --- Filter prior by time ---
        prior_series = prior.to_series()
        prior_times = prior_series.index.get_level_values("time")
        prior_series = prior_series[(prior_times >= tstart) & (prior_times < tend)]
        prior = Vector(prior_series, name=prior.name)

        # Delegate min-obs / min-sims interval filtering to FluxInversionPipeline
        return super().filter_state_space(obs, prior)

    def aggregate_obs_space(
        self,
        obs: Vector,
        forward_operator: ForwardOperator,
        modeldata_mismatch: CovarianceMatrix,
        constant: Vector | None,
    ) -> tuple[Vector, ForwardOperator, CovarianceMatrix, Vector | None]:
        """Aggregates the observation space if specified in the config."""
        # Drop obs whose background is missing. get_constant returns a NaN-free
        # (reduced) constant; reduce obs to match here -- before the aggregator and
        # the InverseProblem reindex everything to obs.index -- so gap-day obs are
        # dropped rather than back-filled to zero. Generalizes the old
        # CTStiltBackgroundInversion obs-dropping to any background.
        if constant is not None:
            obs_series = obs.to_series()
            bg_index = constant["concentration"].index
            # obs carries a leading "block" index level that the background lacks;
            # match on the levels the background is actually indexed by.
            obs_key = obs_series.index
            extra = [n for n in obs_key.names if n not in bg_index.names]
            if extra:
                obs_key = obs_key.droplevel(extra)
            have_bg = obs_key.isin(bg_index)
            if not have_bg.all():
                print(
                    f"  Dropping {int((~have_bg).sum())}/{len(have_bg)} obs "
                    "with missing background"
                )
                obs = Vector(obs_series[have_bg], name=obs.name)

        if self.config.aggregate_obs:
            aggregator = ObsAggregator(
                level="obs_time", freq=self.config.aggregate_obs, blocks="concentration"
            )
            obs, forward_operator, modeldata_mismatch, constant = aggregator.apply(  # pyright: ignore[reportAssignmentType]
                obs, forward_operator, modeldata_mismatch, constant
            )
        return obs, forward_operator, modeldata_mismatch, constant

    def run(self, estimator_kwargs: dict | None = None, **kwargs) -> FluxProblem:
        """Build the inputs, solve, summarize and plot.

        Steps: :meth:`get_inputs` (cached components), the Jacobian coverage filter when
        ``config.jacobian_coverage_percentile`` is set, the solve (``config.gamma`` scales
        the obs error), :meth:`summarize`, then the ``plot_*`` methods the config turns on.

        Parameters
        ----------
        estimator_kwargs : dict, optional
            Passed to ``FluxProblem.solve``; overrides ``gamma`` from the config.
        **kwargs
            Passed to the ``FluxProblem`` constructor.

        Returns
        -------
        FluxProblem
            The solved problem (also kept as ``self.problem``).
        """
        total_start = time.perf_counter()
        print("Getting problem inputs...")
        inputs = self.get_inputs()
        print(f"Inputs prepared in {time.perf_counter() - total_start:.2f}s")

        # Apply Jacobian-based cell filtering
        if self.config.jacobian_coverage_percentile is not None:
            step_start = time.perf_counter()
            print("Filtering cells by Jacobian coverage...")
            inputs = self._apply_jacobian_coverage_filter(inputs)
            print(f"Cells filtered in {time.perf_counter() - step_start:.2f}s")

        print("Initializing solver...")
        step_start = time.perf_counter()
        self.problem = self._InverseProblem(
            **inputs,
            **kwargs,
        )
        print(f"Solver initialized in {time.perf_counter() - step_start:.2f}s")

        if self.config.plot_inputs:
            step_start = time.perf_counter()
            self.plot_inputs(self.problem)
            print(f"Inputs plotted in {time.perf_counter() - step_start:.2f}s")

        print("Solving...")
        step_start = time.perf_counter()
        # Build estimator kwargs: config gamma + explicit overrides
        solve_kwargs = {}
        if self.config.gamma is not None:
            solve_kwargs["gamma"] = self.config.gamma
        if estimator_kwargs:
            solve_kwargs.update(estimator_kwargs)
        self.problem.solve(estimator=self.estimator, **solve_kwargs)
        print(f"Solve completed in {time.perf_counter() - step_start:.2f}s")

        # Print summary report
        print("Calculating summary...")
        step_start = time.perf_counter()
        self.summarize()
        print(f"Summary calculated in {time.perf_counter() - step_start:.2f}s")

        if self.config.plot_results:
            step_start = time.perf_counter()
            self.plot_results(self.problem)
            print(f"Results plotted in {time.perf_counter() - step_start:.2f}s")

        if self.config.plot_diagnostics:
            step_start = time.perf_counter()
            self.plot_diagnostics(self.problem)
            print(f"Diagnostics plotted in {time.perf_counter() - step_start:.2f}s")

        print(f"Total pipeline time: {time.perf_counter() - total_start:.2f}s")

        return self.problem
