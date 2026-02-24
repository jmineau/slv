import matplotlib.pyplot as plt
import pandas as pd

from lair import inventories

from fips import Block, CovarianceMatrix, Vector, ForwardOperator
from fips.aggregators import ObsAggregator
from fips.covariance import CovarianceBuilder
from fips.problems.flux import FluxInversionPipeline, JacobianBuilder
from fips.problems.flux.problem import FluxProblem

from slv.inversion import viz
from slv.inversion.covariances import build_mdm_component, build_prior_error
from slv.inversion.data import get_slv_observations
from slv.inversion.background import get_slv_background
from slv.inversion.priors import get_slv_prior


class SLVMethaneInversion(FluxInversionPipeline):
    """SLV-specific implementation of the flux inversion pipeline."""

    def get_obs(self) -> Vector:
        """Passes just the obs attributes to the pure obs function."""
        return Vector(name='obs',
                      data=Block(name='concentration',
                                 data=get_slv_observations(
            sites=self.config.sites,
            time_range=self.config.time_range,
            filter_pcaps=self.config.filter_pcaps,
            num_processes=self.config.num_processes
        )))

    def get_prior(self):
        prior = get_slv_prior(
            prior=self.config.prior,
            out_grid=self.config.grid,
            bbox=self.config.bbox,
            extent=self.config.extent,
            units=self.config.prior_units,
            **self.config.prior_kwargs
        )
        return Vector(name='prior',
                      data=Block(name='flux',
                                 data=prior))

    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        simulations = sorted(list(self.config.stilt_path.glob('out/by-id/*')))
        print(f'Found {len(simulations)} simulations')

        jacobian_builder = JacobianBuilder(simulations)
        jacobian = jacobian_builder.build_from_coords(self.config.grid_coords,
                                        flux_times=self.config.flux_time_bins,
                                        resolution=self.config.resolution,
                                        subset_hours=self.config.subset_hours_utc,
                                        location_mapper=self.config.location_site_map,
                                        num_processes=self.config.num_processes,
                                        timeout=self.config.timeout,
                                        sparse=self.config.sparse_jacobian
                                        )
        return ForwardOperator(jacobian)

    def get_prior_error(self, prior: Vector):
        S_0 = build_prior_error(prior.data, base_std=self.config.prior_base_std,
                                 std_frac=self.config.prior_std_frac,
                                 time_scale=self.config.prior_time_scale,
                                 spatial_scale=self.config.prior_spatial_scale)
        return CovarianceMatrix(name='prior_error', data=S_0)

    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        components = []
        for comp in self.config.mdm_components:
            components.append(build_mdm_component(
                name=comp.name,
                obs_index=obs.index,
                std=comp.std,
                correlated=comp.correlated,
                scale=comp.scale,
                interday=comp.interday
            ))
        
        return CovarianceMatrix(name='modeldata_mismatch',
                                data=CovarianceBuilder(components).build(obs.index))

    def get_constant(self):
        return Vector(name='background',
                      data=Block(name='concentration',
                                 data=get_slv_background(
            sites=self.config.sites,
            time_range=self.config.time_range,
            baseline_window=self.config.bg_baseline_window,
            filter_pcaps=self.config.filter_pcaps,
            num_processes=self.config.num_processes
        )))

    def aggregate_obs_space(self, obs: Vector, forward_operator: ForwardOperator, modeldata_mistmatch: CovarianceMatrix, constant: Vector | None
                            ) -> tuple[Vector, ForwardOperator, CovarianceMatrix, Vector | None]:
        """Aggregates the observation space if specified in the config."""
        if self.config.aggregate_obs:
            aggregator = ObsAggregator(
                level="obs_time",
                freq=self.config.aggregate_obs,
                blocks="concentration"
            )
            obs, forward_operator, modeldata_mistmatch, constant = aggregator.apply(obs, forward_operator, modeldata_mistmatch, constant)
        return obs, forward_operator, modeldata_mistmatch, constant

    def fluxes_as_inventory(self, fluxes: pd.Series) -> inventories.Inventory:
        """Converts a flux vector to an inventory format for easier analysis."""
        ds = fluxes.to_xarray().to_dataset()
        ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

        time_step = {
            'a': 'annual',
            'MS': 'monthly',
            'D': 'daily',
        }[self.config.flux_freq]

        return inventories.Inventory(ds, pollutant='CH4', src_units='umol/m2/s', time_step=time_step)

    def calculate_total_flux(self, fluxes: pd.Series, units=None) -> pd.Series:
        inventory = self.fluxes_as_inventory(fluxes)
        if units:
            inventory = inventory.convert_units(units)
        return inventory.absolute_emissions['flux'].sum(dim=('lat', 'lon')).to_series()

    def plot_inputs(self, problem: FluxProblem):
        config = self.config

        # --- Plot Grid ---
        viz.plot_grid(config.grid, extent=config.map_extent, tiler=config.tiler, zoom=config.tiler_zoom,
                      sites=config.sites, site_config=config.site_config)

        # --- Plot Concentrations ---
        viz.plot_concentrations(problem.obs)

        # --- Plot Fluxes ---
        viz.plot_inventory(problem.prior_fluxes.to_xarray(), extent=config.map_extent, tiler=config.tiler, zoom=config.tiler_zoom)

        plt.show()

    def plot_results(self, problem: FluxProblem):
        config = self.config
        # --- Plot Fluxes ---
        viz.plot_fluxes(problem, tiler=config.tiler, zoom=config.tiler_zoom,
                    add_sites=True, sites=config.sites, site_config=config.site_config)
        
        total_prior = self.calculate_total_flux(problem.prior_fluxes, units=config.output_units)
        total_posterior = self.calculate_total_flux(problem.posterior_fluxes, units=config.output_units)
        viz.plot_total_fluxes_over_time(total_prior, total_posterior)

        # --- Plot Concentrations ---
        problem.plot.concentrations()

        plt.show()

    def run(self, **kwargs) -> FluxProblem:
        print("Getting problem inputs...")
        inputs = self.get_inputs()

        print("Initializing solver...")
        self.problem = self._InverseProblem(
            **inputs,
            **kwargs,
        )

        if self.config.plot_inputs:
            self.plot_inputs(self.problem)

        print("Solving...")
        self.problem.solve(estimator=self.estimator)

        # Print summary report
        self.summarize()

        if self.config.plot_results:
            self.plot_results(self.problem)

        return self.problem