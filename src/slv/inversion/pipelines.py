import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fips import Block, CovarianceMatrix, ForwardOperator, MatrixBlock, Vector
from fips.aggregators import ObsAggregator
from fips.covariance import CovarianceBuilder, DiagonalError
from fips.problems.flux import FluxInversionPipeline, JacobianBuilder
from fips.problems.flux.problem import FluxProblem
from lair import inventories

from slv.inversion import viz
from slv.inversion.background import get_slv_background
from slv.inversion.covariances import build_mdm_error, build_prior_error
from slv.inversion.data import get_slv_observations
from slv.inversion.priors import get_slv_prior

# ---------------------------------------------------------------------------
# Component dependency sets: maps each cache component to the InversionConfig fields
# that actually affect that component's output.  Used to compute content-addressed
# cache paths so that only truly-stale components are recomputed when parameters
# change.
# ---------------------------------------------------------------------------
_OBS_DEPS: frozenset[str] = frozenset(
    {
        "tstart",
        "tend",
        "sites",
        "filter_pcaps",
        "subset_hours",
        "utc_offset",
    }
)
_PRIOR_DEPS: frozenset[str] = frozenset(
    {
        "prior",
        "prior_kwargs",
        "dx",
        "dy",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "flux_freq",
        "tstart",
        "tend",
    }
)

#: Default mapping of pipeline component → config fields that affect it.
DEFAULT_COMPONENT_DEPS: dict[str, frozenset[str]] = {
    "obs": _OBS_DEPS,
    "prior": _PRIOR_DEPS,
    "forward_operator": _OBS_DEPS
    | _PRIOR_DEPS
    | {
        "stilt_path",
        "sparse_jacobian",
        # num_processes and timeout intentionally excluded: they are
        # computational knobs that do not change the Jacobian result.
    },
    "prior_error": _PRIOR_DEPS
    | {
        "prior_base_std",
        "prior_std_frac",
        "prior_time_scale",
        "prior_spatial_scale",
    },
    "modeldata_mismatch": _OBS_DEPS | {"mdm_config"},
    "constant": _OBS_DEPS | {"bg_baseline_window", "bg_min_periods"},
}


def _json_default(v):
    """Fallback JSON serializer: converts non-primitive types to strings."""
    if isinstance(v, (list, tuple)):
        return [_json_default(i) for i in v]
    if isinstance(v, dict):
        return {k: _json_default(vv) for k, vv in sorted(v.items())}
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    return str(v)


def _component_hash(config, fields: frozenset[str]) -> str:
    """Return the first 12 hex chars of sha256 over the given config fields."""
    data = {f: _json_default(getattr(config, f)) for f in sorted(fields)}
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def fips_cache(cls, filename):
    """Content-addressed cache decorator for pipeline methods.

    Parameters
    ----------
    cls :
        The fips class to use for ``cls.from_file`` / ``result.to_file``.
    filename :
        Stage name / cache stem (e.g. ``"obs"``, ``"prior_error"``).

    Cache layout
    ------------
    Files are stored under ``{cache_dir}/{component}/{hash}.pkl`` where ``hash``
    is derived only from the config fields that actually affect that component
    (see ``COMPONENT_DEPS``).  This means:

    * Changing ``prior_base_std`` reuses ``obs.pkl`` and ``forward_operator.pkl``
      untouched, and only regenerates ``prior_error.pkl``.
    * Configs that share the same upstream parameters share the same files —
      no manual cache wiping is needed.

    ``config.cache`` controls caching:

    * ``False`` / ``None`` — no caching (default)
    * ``True``             — cache in the current working directory
    * ``str`` / ``Path``  — cache in that directory

    ``config.cache_overwrite`` controls forced recomputation:

    * ``[]``        — never overwrite (default)
    * ``"all"``     — overwrite every component
    * ``[component, …]`` — overwrite specific components (all hashes for that component
      are deleted and the component is recomputed fresh)
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = getattr(self.config, "cache", False)
            if not cache:
                return method(self, *args, **kwargs)

            overwrite = getattr(self.config, "cache_overwrite", [])
            if overwrite == "all":
                should_overwrite = True
            elif isinstance(overwrite, str):
                should_overwrite = filename == overwrite
            else:
                should_overwrite = filename in set(overwrite)

            cache_dir = Path.cwd() if cache is True else Path(cache)

            # --- Content-addressed path ---
            component_deps = getattr(self, "COMPONENT_DEPS", DEFAULT_COMPONENT_DEPS)
            fields = component_deps.get(filename)
            if fields:
                h = _component_hash(self.config, fields)
                component_dir = cache_dir / filename
                path = component_dir / f"{h}.pkl"
            else:
                # Fallback: flat file for components not listed in COMPONENT_DEPS
                component_dir = cache_dir
                path = cache_dir / f"{filename}.pkl"

            if path.exists() and not should_overwrite:
                print(
                    f"Loading cached {filename} [{h if fields else 'flat'}] from {path}"
                )
                return cls.from_file(path)

            if should_overwrite and fields and component_dir.exists():
                # Remove all stale hashes for this component before recomputing
                stale = list(component_dir.glob("*.pkl"))
                for s in stale:
                    s.unlink()
                if stale:
                    print(
                        f"Cleared {len(stale)} stale cache file(s) for component '{filename}'"
                    )

            result = method(self, *args, **kwargs)

            component_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving {filename} [{h if fields else 'flat'}] to {path}")
            result.to_file(path)

            return result

        return wrapper

    return decorator


class SLVMethaneInversion(FluxInversionPipeline):
    """SLV-specific implementation of the flux inversion pipeline.

    Supports optional bias correction via config.bias_std and config.bias_grouping.
    When bias_std is set, augments the state vector with bias terms that can be
    grouped by time (default), site, or site organization.
    """

    #: Maps each cache component to the InversionConfig fields it depends on.
    COMPONENT_DEPS: dict[str, frozenset[str]] = {
        **DEFAULT_COMPONENT_DEPS,
        "prior": DEFAULT_COMPONENT_DEPS["prior"] | {"bias_std", "bias_grouping"},
        "forward_operator": DEFAULT_COMPONENT_DEPS["forward_operator"]
        | {"bias_std", "bias_grouping"},
        "prior_error": DEFAULT_COMPONENT_DEPS["prior_error"]
        | {"bias_std", "bias_grouping"},
    }

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
                    num_processes=self.config.num_processes,
                ),
            ),
        )

    @fips_cache(Vector, "prior")
    def get_prior(self) -> Vector:
        """Get the prior vector, optionally including bias terms.

        Returns a single-block flux prior if bias_std is None, otherwise
        returns a multi-block [flux, bias] prior.
        """
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

    @fips_cache(ForwardOperator, "forward_operator")
    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        """Get the forward operator, optionally including bias Jacobian.

        Returns a single-block flux Jacobian if bias_std is None, otherwise
        returns a multi-block [flux_jac | bias_jac] operator.
        """
        from slv.inversion.config import build_location_site_map

        simulations = sorted(list(Path(self.config.stilt_path).glob("out/by-id/*")))
        print(f"Found {len(simulations)} simulations")

        # Auto-generate location mapper if not provided
        location_mapper = self.config.location_site_map
        if not location_mapper:
            sim_ids = [sim.name for sim in simulations]
            location_mapper = build_location_site_map(sim_ids, self.config.site_config)
            print(f"Auto-generated location mapper for {len(location_mapper)} sites")
            self.config.location_site_map = location_mapper

        # Build flux Jacobian
        jacobian_builder = JacobianBuilder(simulations)
        jacobian = jacobian_builder.build_from_coords(
            self.config.grid_coords,
            flux_times=self.config.flux_time_bins,
            resolution=self.config.resolution,
            subset_hours=self.config.subset_hours_utc,
            location_mapper=location_mapper,
            num_processes=self.config.num_processes,
            timeout=self.config.timeout,
            sparse=self.config.sparse_jacobian,
        )

        # Return flux-only operator if no bias
        if self.config.bias_std is None:
            return ForwardOperator(jacobian)

        # Add bias Jacobian
        flux_jac_blk = ForwardOperator(jacobian).blocks["concentration", "flux"]
        bias_jac_blk = MatrixBlock(
            self.get_bias_jacobian(obs, prior), "concentration", "bias"
        )
        return ForwardOperator([flux_jac_blk, bias_jac_blk])

    @fips_cache(CovarianceMatrix, "prior_error")
    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        """Get prior error covariance, optionally including bias error.

        Returns a single-block flux error if bias_std is None, otherwise
        returns a multi-block [flux_err, bias_err] covariance.
        """
        # Build flux error
        flux_prior = Vector(
            prior.blocks["flux"] if self.config.bias_std else prior.data
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

    @fips_cache(CovarianceMatrix, "modeldata_mismatch")
    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        components = [
            build_mdm_error(
                obs_index=obs.index, site_config=self.config.site_config, **comp
            )
            for comp in self.config.mdm_components
        ]

        if self.config.plot_diagnostics:
            built_comps = {comp.name: comp.build(obs.index) for comp in components}
            viz.plot_mdm_components(built_comps)
            return CovarianceMatrix(
                name="modeldata_mismatch",
                data=np.add.reduce([comp.to_numpy() for comp in built_comps.values()]),
                index=obs.index,
            )

        return CovarianceMatrix(
            name="modeldata_mismatch",
            data=CovarianceBuilder(components).build(obs.index),
        )

    @fips_cache(Vector, "constant")
    def get_constant(self, obs: Vector) -> Vector:
        data = get_slv_background(
            sites=self.config.sites,
            site_config=self.config.site_config,
            time_range=self.config.time_range,
            baseline_window=self.config.bg_baseline_window,
            filter_pcaps=self.config.filter_pcaps,
            num_processes=self.config.num_processes,
        )

        # Duplicate obs for each location
        data = (
            obs.data.reset_index()
            .join(data, on="obs_time", lsuffix="_obs")
            .set_index(["obs_location", "obs_time"])["concentration"]
        )

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
        if self.config.aggregate_obs:
            aggregator = ObsAggregator(
                level="obs_time", freq=self.config.aggregate_obs, blocks="concentration"
            )
            obs, forward_operator, modeldata_mismatch, constant = aggregator.apply(
                obs, forward_operator, modeldata_mismatch, constant
            )
        return obs, forward_operator, modeldata_mismatch, constant

    def fluxes_as_inventory(self, fluxes: pd.Series) -> inventories.Inventory:
        """Converts a flux vector to an inventory format for easier analysis."""
        ds = fluxes.to_xarray().to_dataset()

        time_step = {
            "YS": "annual",
            "MS": "monthly",
            "D": "daily",
        }[self.config.flux_freq]

        return inventories.Inventory(
            ds, pollutant="CH4", src_units="umol/m2/s", time_step=time_step
        )

    def calculate_total_flux(self, fluxes: pd.Series, units=None) -> pd.Series:
        inventory = self.fluxes_as_inventory(fluxes)
        if units:
            inventory = inventory.convert_units(units)
        return (
            inventory.absolute_emissions[fluxes.name]
            .sum(dim=("lat", "lon"))
            .to_series()
        )

    def plot_inputs(self, problem: FluxProblem):
        config = self.config

        # --- Plot Grid ---
        viz.plot_grid(
            config.grid,
            extent=config.map_extent,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
            sites=config.sites,
            site_config=config.site_config,
        )

        # --- Plot Concentrations ---
        viz.plot_concentrations(problem.concentrations)

        # --- Plot Fluxes ---
        viz.plot_inventory(
            problem.prior_fluxes.to_xarray(),
            extent=config.map_extent,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
        )

        plt.show()

    def plot_results(self, problem: FluxProblem):
        config = self.config
        # --- Plot Fluxes ---
        viz.plot_fluxes(
            problem,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
            add_sites=True,
            sites=config.sites,
            site_config=config.site_config,
        )

        total_prior = self.calculate_total_flux(
            problem.prior_fluxes, units=config.output_units
        )
        total_posterior = self.calculate_total_flux(
            problem.posterior_fluxes, units=config.output_units
        )
        viz.plot_total_fluxes_over_time(total_prior, total_posterior)

        # --- Plot Concentrations ---
        problem.plot.concentrations()

        # --- Plot Residuals ---
        viz.plot_residuals(problem)

        # --- Plot Background and Bias ---
        viz.plot_background_and_bias(problem)

        plt.show()

    def plot_diagnostics(self, problem: FluxProblem):
        config = self.config
        # --- Plot Fluxes by Timestep ---
        viz.plot_fluxes_by_timestep(
            problem,
            extent=config.map_extent,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
            add_sites=True,
            sites=config.sites,
            site_config=config.site_config,
        )

        plt.show()

    def _apply_jacobian_coverage_filter(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Remove cells with insufficient Jacobian coverage from the state vector.

        Cells are removed across ALL time steps to preserve the Kronecker
        structure of the prior error covariance.  Removed cells are stored
        so they can be reinstated at the prior value via ``reconstruct_posterior``.
        """
        threshold = self.config.min_jacobian_coverage
        prior = inputs["prior"]
        forward_operator = inputs["forward_operator"]
        has_bias = self.config.bias_std is not None

        # --- Compute per-cell coverage ---
        flux_blk = forward_operator.blocks["concentration", "flux"]
        flux_jac = (
            flux_blk.data
        )  # DataFrame: (obs_location, obs_time) × (lon, lat, time)

        col_abs_sums = flux_jac.abs().sum(axis=0)
        lat_vals = col_abs_sums.index.get_level_values("lat")
        lon_vals = col_abs_sums.index.get_level_values("lon")
        cell_coverage = col_abs_sums.groupby([lat_vals, lon_vals]).sum()

        retained_cells = set(cell_coverage[cell_coverage >= threshold].index.tolist())
        removed_cells = set(cell_coverage[cell_coverage < threshold].index.tolist())

        n_total = len(cell_coverage)
        n_removed = len(removed_cells)
        print(
            f"  Keeping {n_total - n_removed}/{n_total} cells "
            f"(removed {n_removed} below threshold {threshold:.2e})"
        )

        if n_removed == 0:
            return inputs

        # Store for posterior reconstruction
        self._full_prior = prior
        self._removed_cells = removed_cells

        # --- Helper: boolean mask for a MultiIndex with lat/lon levels ---
        def _cell_mask(index):
            lats = index.get_level_values("lat")
            lons = index.get_level_values("lon")
            return pd.array(
                [
                    (lat, lon) in retained_cells
                    for lat, lon in zip(lats, lons, strict=False)
                ],
                dtype=bool,
            )

        # --- Filter prior ---
        flux_series = prior["flux"]
        filtered_flux = flux_series[_cell_mask(flux_series.index)]

        if has_bias:
            filtered_prior = Vector(
                name=prior.name,
                data=[Block(filtered_flux, name="flux"), prior.blocks["bias"]],
            )
        else:
            filtered_prior = Vector(
                name=prior.name, data=Block(name="flux", data=filtered_flux)
            )

        # --- Filter forward operator ---
        jac_mask = _cell_mask(flux_jac.columns)
        filtered_flux_jac = flux_jac.loc[:, jac_mask]
        filtered_flux_blk = MatrixBlock(
            filtered_flux_jac,
            row_block="concentration",
            col_block="flux",
            sparse=self.config.sparse_jacobian,
        )

        if has_bias:
            bias_blk = forward_operator.blocks["concentration", "bias"]
            filtered_fo = ForwardOperator([filtered_flux_blk, bias_blk])
        else:
            filtered_fo = ForwardOperator(filtered_flux_blk)

        # --- Rebuild prior error for the smaller grid ---
        flux_prior_vec = Vector(filtered_prior.blocks["flux"])
        S_0 = build_prior_error(
            flux_prior_vec,
            base_std=self.config.prior_base_std,
            std_frac=self.config.prior_std_frac,
            time_scale=self.config.prior_time_scale,
            spatial_scale=self.config.prior_spatial_scale,
        )

        if not has_bias:
            prior_error = CovarianceMatrix(name="prior_error", data=S_0)
        else:
            flux_err_blk = CovarianceMatrix(name="prior_error", data=S_0).blocks[
                "flux", "flux"
            ]
            bias_index = filtered_prior["bias"].index
            bias_err = DiagonalError(
                name="bias_error", variances=self.config.bias_std**2
            ).build(bias_index)
            bias_err_blk = MatrixBlock(bias_err, "bias", "bias")
            prior_error = CovarianceMatrix(
                name="prior_error", data=[flux_err_blk, bias_err_blk]
            )

        return {
            **inputs,
            "prior": filtered_prior,
            "forward_operator": filtered_fo,
            "prior_error": prior_error,
        }

    def reconstruct_posterior(
        self, posterior_fluxes: pd.Series | None = None
    ) -> pd.Series:
        """Reconstruct the full posterior by inserting prior for unconstrained cells.

        Parameters
        ----------
        posterior_fluxes : pd.Series, optional
            Posterior flux series from the inversion.  If None, uses
            ``self.problem.posterior_fluxes``.

        Returns
        -------
        pd.Series
            Full posterior with constrained cells from the inversion and
            unconstrained cells filled with prior values.
        """
        if posterior_fluxes is None:
            posterior_fluxes = self.problem.posterior_fluxes

        if not hasattr(self, "_full_prior") or not hasattr(self, "_removed_cells"):
            return posterior_fluxes

        # Get the full prior flux series
        full_flux = self._full_prior["flux"]

        # Start with prior, then overwrite constrained cells
        full_posterior = full_flux.copy()
        full_posterior.name = posterior_fluxes.name
        full_posterior.loc[posterior_fluxes.index] = posterior_fluxes.values

        return full_posterior

    def run(self, estimator_kwargs: dict | None = None, **kwargs) -> FluxProblem:
        total_start = time.perf_counter()
        print("Getting problem inputs...")
        inputs = self.get_inputs()
        print(f"Inputs prepared in {time.perf_counter() - total_start:.2f}s")

        # Apply Jacobian-based cell filtering
        if self.config.min_jacobian_coverage is not None:
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

    def get_site_group(self, site: str) -> str:
        """Map site to organization group."""
        return self.config.site_config.organization.to_dict().get(site, "unknown")

    def get_bias(self) -> pd.Series:
        """Build the bias prior based on config.bias_grouping.

        Returns a zero-valued Series with index determined by bias_grouping:
          - None or "time": one bias per time interval
          - "site": one bias per (time, obs_location)
          - "site_group": one bias per (time, organization)

        Override this method for non-zero initial values or custom groupings.
        """
        grouping = self.config.bias_grouping

        if grouping in (None, "time"):
            # Time-only bias (default)
            index = pd.Index(self.config.flux_times, name="time")

        elif grouping == "site":
            # Per-site bias
            index = pd.MultiIndex.from_product(
                [self.config.flux_times, self.config.sites],
                names=["time", "obs_location"],
            )

        elif grouping == "site_group":
            # Per-site-group (organization) bias - only for configured sites
            site_groups = (
                self.config.site_config.loc[self.config.sites, "organization"]
                .unique()
                .tolist()
            )
            index = pd.MultiIndex.from_product(
                [self.config.flux_times, site_groups], names=["time", "site_group"]
            )

        else:
            raise ValueError(
                f"Unknown bias_grouping: {grouping}. "
                f"Expected None, 'time', 'site', or 'site_group'"
            )

        return pd.Series(0.0, index=index, name="bias")

    def get_bias_jacobian(self, obs: Vector, prior: Vector) -> pd.DataFrame:
        """Build the obs × bias Jacobian based on config.bias_grouping.

        Maps each observation to its corresponding bias term:
          - time: match by time interval only
          - site: match by (time, obs_location)
          - site_group: match by (time, organization)
        """
        obs_index = obs["concentration"].index
        bias_index = prior["bias"].index
        obs_times = obs_index.get_level_values("obs_time")
        grouping = self.config.bias_grouping

        # Bin obs times into flux intervals
        cut = pd.cut(obs_times, bins=self.config.flux_time_bins)
        flux_times = cut.map(lambda iv: iv.left if pd.notna(iv) else None)

        if grouping in (None, "time"):
            # Time-only: simple one-hot encoding
            jac = pd.get_dummies(cut, dtype=float)
            jac.columns = jac.columns.map(lambda iv: iv.left)
            jac.index = obs_index

        elif grouping == "site":
            # Per-site: match (time, obs_location)
            obs_sites = obs_index.get_level_values("obs_location")
            bias_keys = pd.Series(
                list(zip(flux_times, obs_sites, strict=True)),
                index=obs_index,
                dtype=object,
            )
            jac = pd.get_dummies(bias_keys, dtype=float)
            jac.columns = pd.MultiIndex.from_tuples(
                jac.columns, names=["time", "obs_location"]
            )

        elif grouping == "site_group":
            # Per-site-group: match (time, organization)
            obs_locs = obs_index.get_level_values("obs_location")
            # Map obs_location to site first, then to organization
            location_to_site = self.config.location_site_map or {}
            obs_sites = obs_locs.map(lambda loc: location_to_site.get(loc, loc))
            obs_site_groups = obs_sites.map(self.get_site_group)
            bias_keys = pd.Series(
                list(zip(flux_times, obs_site_groups, strict=True)),
                index=obs_index,
                dtype=object,
            )
            jac = pd.get_dummies(bias_keys, dtype=float)
            jac.columns = pd.MultiIndex.from_tuples(
                jac.columns, names=["time", "site_group"]
            )

        else:
            raise ValueError(f"Unknown bias_grouping: {grouping}")

        # Align to bias index (handles any time-range trimming)
        return jac.reindex(columns=bias_index, fill_value=0.0)
