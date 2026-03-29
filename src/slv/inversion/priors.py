import pandas as pd
import xarray as xr
from lair import inventories


def get_slv_prior(
    prior: str, out_grid, flux_times, flux_freq=None, bbox=None, extent=None, **kwargs
):
    units = "umol/m2/s"  # Must match jacobian (STILT)
    if prior.lower() == "epa":
        return load_epa_prior(
            out_grid=out_grid,
            flux_times=flux_times,
            flux_freq=flux_freq,
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
    flux_freq=None,
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
    import xesmf as xe  # conda-forge only; lazy to avoid import-time failure

    regridder = xe.Regridder(total, out_grid, method="conservative")
    inventory: xr.Dataset = regridder(total)

    inventory.name = "flux"  # Rename emissions
    inventory.attrs["units"] = total.attrs["units"]

    # Resample inventory to target flux frequency if needed
    if flux_freq is not None:
        inv_freq = pd.infer_freq(inventory.time.values)
        if inv_freq is not None:
            ref = pd.Timestamp("2020-01-01")
            target_step = ref + pd.tseries.frequencies.to_offset(flux_freq)
            inv_step = ref + pd.tseries.frequencies.to_offset(inv_freq)
            if target_step > inv_step:
                inventory = inventory.resample(time=flux_freq).mean()

    # Align to exact flux_times (nearest-neighbor fills finer-than-inventory requests)
    prior = inventory.reindex(time=flux_times, method="nearest").to_series()

    if return_regridder:
        return prior, regridder
    return prior
