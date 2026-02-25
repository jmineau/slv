import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from fips import Block, CovarianceMatrix, ForwardOperator, MatrixBlock, Vector
from fips.aggregators import ObsAggregator
from fips.covariance import CovarianceBuilder, DiagonalError
from fips.problems.flux import FluxInversionPipeline, JacobianBuilder
from fips.problems.flux.problem import FluxProblem
from lair import inventories

from slv.inversion import viz
from slv.inversion.background import get_slv_background
from slv.inversion.covariances import build_mdm_component, build_prior_error
from slv.inversion.data import get_slv_observations
from slv.inversion.priors import get_slv_prior


def fips_cache(cls, filename):
    """Cache decorator for pipeline methods that return fips objects.

    Parameters
    ----------
    cls :
        The fips class to use for ``cls.from_file`` / ``result.to_file``.
    filename :
        Stem of the cache file (e.g. ``"obs"`` → ``obs.pkl``).

    Reads ``self.config.cache`` to determine caching behaviour:

    * ``False`` / ``None`` — no caching (default)
    * ``True``             — cache in the current working directory
    * ``str`` / ``Path``  — cache in the given directory
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            cache = getattr(self.config, "cache", False)
            if not cache:
                return method(self, *args, **kwargs)

            cache_dir = Path.cwd() if cache is True else Path(cache)
            path = cache_dir / f"{filename}.pkl"

            if path.exists():
                print(f"Loading cached {filename} from {path}")
                return cls.from_file(path)

            result = method(self, *args, **kwargs)

            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving {filename} to {path}")
            result.to_file(path)

            return result

        return wrapper

    return decorator


class SLVMethaneInversion(FluxInversionPipeline):
    """SLV-specific implementation of the flux inversion pipeline."""

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
                    subset_hours=self.config.subset_hours_utc,
                    filter_pcaps=self.config.filter_pcaps,
                    num_processes=self.config.num_processes,
                ),
            ),
        )

    @fips_cache(Vector, "prior")
    def get_prior(self) -> Vector:
        prior = get_slv_prior(
            prior=self.config.prior,
            out_grid=self.config.grid,
            flux_times=self.config.flux_times,
            bbox=self.config.bbox,
            **self.config.prior_kwargs,
        )
        return Vector(name="prior", data=Block(name="flux", data=prior))

    @fips_cache(ForwardOperator, "forward_operator")
    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        simulations = sorted(list(Path(self.config.stilt_path).glob("out/by-id/*")))
        print(f"Found {len(simulations)} simulations")

        jacobian_builder = JacobianBuilder(simulations)
        jacobian = jacobian_builder.build_from_coords(
            self.config.grid_coords,
            flux_times=self.config.flux_time_bins,
            resolution=self.config.resolution,
            subset_hours=self.config.subset_hours_utc,
            location_mapper=self.config.location_site_map,
            num_processes=self.config.num_processes,
            timeout=self.config.timeout,
            sparse=self.config.sparse_jacobian,
        )
        return ForwardOperator(jacobian)

    @fips_cache(CovarianceMatrix, "prior_error")
    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        S_0 = build_prior_error(
            prior,
            base_std=self.config.prior_base_std,
            std_frac=self.config.prior_std_frac,
            time_scale=self.config.prior_time_scale,
            spatial_scale=self.config.prior_spatial_scale,
        )
        return CovarianceMatrix(name="prior_error", data=S_0)

    @fips_cache(CovarianceMatrix, "modeldata_mismatch")
    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        components = []
        for comp in self.config.mdm_components:
            components.append(
                build_mdm_component(
                    name=comp.name,
                    obs_index=obs.index,
                    std=comp.std,
                    correlated=comp.correlated,
                    scale=comp.scale,
                    interday=comp.interday,
                )
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

    def aggregate_obs_space(
        self,
        obs: Vector,
        forward_operator: ForwardOperator,
        modeldata_mistmatch: CovarianceMatrix,
        constant: Vector | None,
    ) -> tuple[Vector, ForwardOperator, CovarianceMatrix, Vector | None]:
        """Aggregates the observation space if specified in the config."""
        if self.config.aggregate_obs:
            aggregator = ObsAggregator(
                level="obs_time", freq=self.config.aggregate_obs, blocks="concentration"
            )
            obs, forward_operator, modeldata_mistmatch, constant = aggregator.apply(
                obs, forward_operator, modeldata_mistmatch, constant
            )
        return obs, forward_operator, modeldata_mistmatch, constant

    def fluxes_as_inventory(self, fluxes: pd.Series) -> inventories.Inventory:
        """Converts a flux vector to an inventory format for easier analysis."""
        ds = fluxes.to_xarray().to_dataset()

        time_step = {
            "a": "annual",
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

    def run(self, **kwargs) -> FluxProblem:
        total_start = time.perf_counter()
        print("Getting problem inputs...")
        inputs = self.get_inputs()
        print(f"Inputs prepared in {time.perf_counter() - total_start:.2f}s")

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
        self.problem.solve(estimator=self.estimator)
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


class SLVMethaneInversionWithBias(SLVMethaneInversion):
    """SLV methane inversion pipeline with an additional background bias block.

    Extends SLVMethaneInversion by augmenting the state vector with a bias
    correction term, allowing systematic offsets (e.g., per-site or per-period
    background biases) to be jointly estimated alongside fluxes.

    Requires ``config.bias_std`` to be set.
    Override ``get_bias()`` for a custom bias index or initial values.
    ``config.bias_jacobian`` defaults to ``1.0`` (scalar identity mapping).
    """

    def get_bias(self) -> pd.Series:
        """Build the bias prior as a zero-valued Series indexed by ``config.flux_times``.

        Override this method in a subclass to use a custom index (e.g.,
        per-site x per-interval) or non-zero initial values.  The returned
        Series must have a named index so it can form a well-labelled
        ``Block``.
        """
        return pd.Series(
            0.0,
            index=pd.Index(self.config.flux_times, name="time"),
            name="bias",
        )

    def get_prior(self) -> Vector:
        """Returns a multi-block prior Vector: [flux, bias]."""
        flux_prior = super().get_prior()
        bias_blk = Block(self.get_bias(), name="bias")
        return Vector(name="prior", data=[flux_prior.blocks["flux"], bias_blk])

    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        """Returns a multi-block ForwardOperator: [flux_jac | bias_jac].
        """
        
        # Build flux jacobian; parent doesn't actually use prior values, only grid coords
        flux_only_prior = Vector(prior.blocks["flux"])
        fo = super().get_forward_operator(obs, flux_only_prior)

        bias_prior_index = prior["bias"].index
        bias_jacobian = self.config.bias_jacobian

        if isinstance(bias_jacobian, (float, int)):
            bias_jac_blk = MatrixBlock(
                bias_jacobian,
                "concentration",
                "bias",
                name="bias_jacobian",
                index=obs["concentration"].index,
                columns=bias_prior_index,
            )
        else:
            bias_jac_blk = MatrixBlock(bias_jacobian, "concentration", "bias")

        flux_jac_blk = fo.blocks["concentration", "flux"]
        return ForwardOperator([flux_jac_blk, bias_jac_blk])

    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        """Returns a multi-block prior error CovarianceMatrix: [flux_err, bias_err]."""
        flux_err = super().get_prior_error(Vector(prior.blocks["flux"]))
        flux_err_blk = flux_err.blocks["flux", "flux"]

        bias_index = prior["bias"].index
        bias_err = DiagonalError(name="bias_error", variances=self.config.bias_std ** 2).build(bias_index)
        bias_err_blk = MatrixBlock(bias_err, "bias", "bias")

        return CovarianceMatrix(name="prior_error", data=[flux_err_blk, bias_err_blk])
