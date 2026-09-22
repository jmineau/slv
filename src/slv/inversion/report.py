"""Domain totals, the summary report and plots for
:class:`~slv.inversion.pipelines.SLVMethaneInversion`."""

import matplotlib.pyplot as plt
import pandas as pd
from fips.problems.flux.problem import FluxProblem
from lair import inventories

from slv.inversion import viz
from slv.inversion.config import InversionConfig


class ReportingMixin:
    """Domain-total emissions, the printed summary, Desroziers diagnostics and plots."""

    config: InversionConfig
    problem: FluxProblem

    def fluxes_as_inventory(self, fluxes: pd.Series) -> inventories.Inventory:
        """Converts a flux vector to an inventory format for easier analysis."""
        ds = fluxes.to_xarray().to_dataset()

        from slv.inversion.config import FLUX_FREQ_TIME_STEPS

        time_step = FLUX_FREQ_TIME_STEPS[self.config.flux_freq]

        return inventories.Inventory(
            ds, pollutant="CH4", src_units="umol/m2/s", time_step=time_step
        )

    def _total_units(self) -> str:
        """Unit of :meth:`calculate_total_flux`: it integrates each flux interval over its
        cells and its duration, so the mass part of ``output_units`` per interval."""
        units = self.config.output_units or "umol/m2/s"
        return f"{units.split('/')[0]} per {self.config.flux_freq} interval"

    def calculate_total_flux(self, fluxes: pd.Series, units=None) -> pd.Series:
        """Domain-total emission per flux interval.

        The flux field is integrated over its cells' areas and each interval's duration
        (lair ``absolute_emissions``), so the result is a mass per interval, e.g. Gg per
        ``MS`` interval for ``units="Gg/m2/s"``.

        Parameters
        ----------
        fluxes : pd.Series
            Flux in umol/m2/s indexed by (time, lat, lon); its ``name`` must be set.
        units : str, optional
            Convert to these flux units before integrating (e.g. ``"Gg/m2/s"``).

        Returns
        -------
        pd.Series
            Total per flux time.
        """
        inventory = self.fluxes_as_inventory(fluxes)
        if units:
            inventory = inventory.convert_units(units)
        return (
            inventory.absolute_emissions[fluxes.name]
            .sum(dim=("lat", "lon"))
            .to_series()
        )

    def desroziers_diagnostic(
        self,
        groupby: str | list[str] | None = "obs_location",
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Compare Desroziers-diagnosed vs specified observation error variances.

        Parameters
        ----------
        groupby : str, list of str, or None
            Index level(s) to aggregate by. Default groups by site.
            Use None for per-observation values.
        freq : str, optional
            Temporal resampling frequency (e.g. 'MS', 'QS', 'YS').
            Groups obs_time into bins at this frequency in addition to
            any levels in ``groupby``.

        Returns
        -------
        pd.DataFrame
            Columns: diagnosed, specified, ratio (diagnosed / specified).
        """
        diagnosed = self.problem.desroziers.variances.to_series()
        diagnosed.name = "diagnosed"

        specified = self.problem.modeldata_mismatch.variances.to_series()
        specified.name = "specified"

        result = pd.concat([diagnosed, specified], axis=1)

        if groupby is not None or freq is not None:
            groupers = []
            if groupby is not None:
                if isinstance(groupby, str):
                    groupby = [groupby]
                groupers.extend(groupby)
            if freq is not None:
                groupers.append(pd.Grouper(level="obs_time", freq=freq))
            result = result.groupby(groupers).mean()

        result["ratio"] = result["diagnosed"] / result["specified"]

        return result

    def summarize(self) -> None:
        """Print fips' summary, then prior and posterior domain totals.

        Totals are per flux interval (:meth:`calculate_total_flux`) in
        ``config.output_units``. With the coverage filter on, the removed cells are put
        back at their prior (:meth:`reconstruct_posterior`) so both totals cover the full
        domain. With more than two intervals a linear trend of the posterior is printed.
        """
        from scipy import stats

        super().summarize()
        config = self.config
        problem = self.problem

        prior_flux = problem.prior_fluxes
        post_flux = problem.posterior_fluxes

        # --- Domain total emissions (requires inventory/regridding) ---
        if self._retained_cells is not None:
            full_prior = self._full_prior["flux"].astype(float)
            full_prior.name = prior_flux.name
            reconstructed = self.reconstruct_posterior()
            total_prior = self.calculate_total_flux(
                full_prior, units=config.output_units
            )
            total_post = self.calculate_total_flux(
                reconstructed, units=config.output_units
            )
        else:
            total_prior = self.calculate_total_flux(
                prior_flux, units=config.output_units
            )
            total_post = self.calculate_total_flux(post_flux, units=config.output_units)

        units = self._total_units()

        print("--------------------------------------------------")
        print(f"DOMAIN TOTAL EMISSIONS [{units}]:")
        summary = pd.DataFrame(
            {
                "prior": total_prior,
                "posterior": total_post,
                "change_%": ((total_post / total_prior) - 1) * 100,
            }
        )
        print(summary.to_string(float_format="%.2f"))
        print(f"  Mean Prior:     {total_prior.mean():.2f}")
        print(f"  Mean Posterior: {total_post.mean():.2f}")
        print(
            f"  Mean Change:    {((total_post.mean() / total_prior.mean()) - 1) * 100:+.1f}%"
        )

        if len(total_post) > 2:
            t = total_post.index
            t_years = t.year + (t.day_of_year - 1) / 365.25
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                t_years, total_post.values
            )
            print(
                f"  Trend:          {slope:+.2f} [{units}]/yr (R^2={r_value**2:.3f}, p={p_value:.3g})"
            )
        print("==================================================")

    def plot_inputs(self, problem: FluxProblem):
        """Plot the grid and sites, the obs time series and the prior fluxes."""
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

        # --- Plot Prior Fluxes ---
        if self._retained_cells is not None:
            full_prior_xr = self._full_prior["flux"].to_xarray()
            viz.plot_prior_with_coverage(
                full_prior_xr,
                self._retained_cells,
                self._all_cells,
                dx=config.dx,
                dy=config.dy,
                extent=config.map_extent,
                tiler=config.tiler,
                zoom=config.tiler_zoom,
            )
        else:
            viz.plot_inventory(
                problem.prior_fluxes.to_xarray(),
                extent=config.map_extent,
                tiler=config.tiler,
                zoom=config.tiler_zoom,
            )

        plt.show()

    def plot_results(self, problem: FluxProblem):
        """Plot the posterior fluxes, the reconstructed full-domain posterior (with the
        coverage filter on), domain totals over time, modelled vs observed
        concentrations, residuals, and background and bias."""
        config = self.config

        # --- Plot Fluxes (inversion domain only) ---
        viz.plot_fluxes(
            problem,
            tiler=config.tiler,
            zoom=config.tiler_zoom,
            add_sites=True,
            sites=config.sites,
            site_config=config.site_config,
        )

        # --- Plot Reconstructed Posterior (full domain) ---
        reconstructed = None
        if self._retained_cells is not None:
            reconstructed = self.reconstruct_posterior()
            reconstructed_xr = reconstructed.to_xarray()
            viz.plot_reconstructed_posterior(
                reconstructed_xr,
                self._retained_cells,
                self._all_cells,
                dx=config.dx,
                dy=config.dy,
                extent=config.map_extent,
                tiler=config.tiler,
                zoom=config.tiler_zoom,
                sites=config.sites,
                site_config=config.site_config,
            )

        # --- Total Emissions (use full reconstructed domain) ---
        if reconstructed is not None:
            full_prior = self._full_prior["flux"]
            full_prior.name = problem.prior_fluxes.name
            total_prior = self.calculate_total_flux(
                full_prior, units=config.output_units
            )
            total_posterior = self.calculate_total_flux(
                reconstructed, units=config.output_units
            )
        else:
            total_prior = self.calculate_total_flux(
                problem.prior_fluxes, units=config.output_units
            )
            total_posterior = self.calculate_total_flux(
                problem.posterior_fluxes, units=config.output_units
            )
        viz.plot_total_fluxes_over_time(
            total_prior, total_posterior, units=self._total_units()
        )

        # --- Plot Concentrations ---
        problem.plot.concentrations()

        # --- Plot Residuals ---
        viz.plot_residuals(problem)

        # --- Plot Background and Bias ---
        viz.plot_background_and_bias(problem)

        plt.show()

    def plot_diagnostics(self, problem: FluxProblem):
        """Plot fluxes per time step, the Desroziers diagnostics and, with the coverage
        filter on, the removed cells' contribution to the obs."""
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

        # --- Plot Desroziers Diagnostic ---
        viz.plot_desroziers(
            by_site=self.desroziers_diagnostic(),
            per_obs=self.desroziers_diagnostic(groupby=None),
            timeseries=self.desroziers_diagnostic(freq=config.flux_freq),
        )

        # --- Plot Unconstrained Cells Contribution ---
        if hasattr(self, "_removed_contribution") and problem.constant is not None:
            viz.plot_removed_contribution(
                self._removed_contribution,
                problem.constant["concentration"],
            )

        plt.show()
