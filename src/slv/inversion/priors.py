import xarray as xr
import xesmf as xe
from lair import inventories


def get_slv_prior(prior: str, out_grid, flux_times, bbox=None, extent=None, **kwargs):
    units = "umol/m2/s"  # Must match jacobian (STILT)
    if prior.lower() == "epa":
        return load_epa_prior(
            out_grid=out_grid,
            flux_times=flux_times,
            bbox=bbox,
            extent=extent,
            units=units,
            return_regridder=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported prior: {prior}")


def load_epa_prior(
    out_grid,
    flux_times,
    bbox=None,
    extent=None,
    units=None,
    express=False,
    return_regridder=False,
):
    if not express:
        # Load inventories
        annual = inventories.EPAv2()
        monthly = inventories.EPAv2(scale_by_month=True)

        # Clip to the bounding box or extent
        if any([bbox, extent]):
            annual = annual.clip(bbox=bbox, extent=extent)
            monthly = monthly.clip(bbox=bbox, extent=extent)

        # Convert units
        if units:
            annual = annual.convert_units(units)
            monthly = monthly.convert_units(units)

        # Get annual only variables
        annual_vars = set(annual.data.data_vars)
        monthly_vars = set(monthly.data.data_vars)
        annual_only_vars = annual_vars - monthly_vars

        # Repeat annual data to monthly freq
        repeated_annual = annual.data[annual_only_vars].reindex(
            time=monthly.data.time, method="ffill"
        )

        # Merge annual and monthly data
        merged = xr.merge([repeated_annual, monthly.data])

        # Sum sectors
        total = inventories.sum_sectors(merged)
    else:
        express = inventories.EPAv2(express=True)  # dont scale by month

        # Clip to the bounding box or extent
        if any([bbox, extent]):
            express = express.clip(bbox=bbox, extent=extent)

        # Convert units
        if units:
            express = express.convert_units(units)

        # Sum sectors
        total = inventories.sum_sectors(express.data)

    # Regrid
    regridder = xe.Regridder(total, out_grid, method="conservative")
    inventory: xr.Dataset = regridder(total)

    inventory.name = "flux"  # Rename emissions
    inventory.attrs["units"] = total.attrs["units"]

    if express:
        # Forward fill months with annual value
        prior = (
            inventory.sel(time="2020")
            .reindex(time=flux_times, method="nearest")
            .to_series()
        )
    else:
        prior = inventory.reindex(time=flux_times, method="nearest").to_series()

    if return_regridder:
        return prior, regridder
    return prior
