"""Prior flux fields for the SLV inversion (EPA, EDGAR or constant) on the state grid."""

import pandas as pd
import xarray as xr
from lair import inventories


def get_slv_prior(
    prior: str, out_grid, flux_times, flux_freq=None, bbox=None, extent=None, **kwargs
):
    """Build the prior flux for ``InversionConfig.prior``.

    Parameters
    ----------
    prior : str
        ``"epa"`` (:func:`load_epa_prior`), ``"edgar"`` (:func:`load_edgar_prior`) or
        ``"constant"`` (:func:`build_constant_prior`); case-insensitive.
    out_grid : xr.DataArray
        Target lon/lat grid (``InversionConfig.grid``).
    flux_times : pd.DatetimeIndex
        Start of each flux interval.
    flux_freq : str, optional
        Flux interval (e.g. ``"MS"``); a coarser interval than the inventory's is
        averaged to.
    bbox, extent : tuple, optional
        Clip the inventory before regridding.
    **kwargs
        Passed to the loader (e.g. ``express=True`` for EPA, ``value=`` for constant).

    Returns
    -------
    pd.Series
        Flux in umol/m2/s (the Jacobian's units), indexed by (time, lat, lon) or
        (time, lon, lat).
    """
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
    elif prior.lower() == "edgar":
        kwargs.pop(
            "express", None
        )  # EPA-only kwarg; base_config injects it, N/A to EDGAR
        return load_edgar_prior(
            out_grid=out_grid,
            flux_times=flux_times,
            flux_freq=flux_freq,
            bbox=bbox,
            extent=extent,
            units=units,
            return_regridder=False,
            **kwargs,
        )
    elif prior.lower() == "constant":
        return build_constant_prior(
            out_grid=out_grid,
            flux_times=flux_times,
            units=units,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported prior: {prior}")


def align_to_flux_times(inventory: xr.DataArray | xr.Dataset, flux_times):
    """Give each flux time the inventory period it falls in.

    Inventory times label period starts (lair puts an annual inventory at Jan 1 and a
    monthly one at the 1st), as do ``flux_times``, so each flux time takes the latest
    inventory time at or before it -- a forward fill. Nearest-neighbour matching would
    hand Aug-Dec of an annual inventory the *next* year's field. Flux times before the
    inventory starts hold its first period; times after it ends hold the last (e.g. EPA
    2020 for 2021-2023).
    """
    times = inventory.indexes["time"]
    pos = times.get_indexer(pd.DatetimeIndex(flux_times), method="ffill")
    pos[pos == -1] = 0
    return inventory.isel(time=pos).assign_coords(time=pd.DatetimeIndex(flux_times))


def build_constant_prior(out_grid, flux_times, value=0.0, units=None):
    """Build a spatially and temporally uniform prior.

    Parameters
    ----------
    out_grid : xr.DataArray
        Target grid with lon/lat coordinates.
    flux_times : pd.DatetimeIndex
        Time points for flux estimation.
    value : float
        Constant flux value to fill the prior with (default 0.0).
    units : str, optional
        Units string to attach as metadata.
    """
    grid_da = xr.DataArray(
        data=value,
        coords={"time": flux_times, "lon": out_grid["lon"], "lat": out_grid["lat"]},
        dims=["time", "lon", "lat"],
        name="flux",
        attrs={"units": units} if units else {},
    )
    return grid_da.to_series()


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
    """EPA gridded CH4 inventory (v2) regridded to ``out_grid`` and aligned to ``flux_times``.

    Sectors are summed and regridded conservatively (needs ``xesmf``). With
    ``express=False`` the monthly-scaled sectors are used where lair has them and the
    annual-only sectors are repeated each month; ``express=True`` loads lair's
    pre-summed annual product (faster, no monthly scaling). A flux time after the
    inventory ends takes its last year (EPA 2020 for 2021-2023), see
    :func:`align_to_flux_times`.

    Parameters
    ----------
    out_grid : xr.DataArray
        Target lon/lat grid.
    flux_times : pd.DatetimeIndex
        Start of each flux interval.
    flux_freq : str, optional
        Flux interval; the inventory is averaged to it when coarser than the inventory.
    bbox, extent : tuple, optional
        Clip the inventory before regridding.
    units : str, optional
        Convert to these units (e.g. ``"umol/m2/s"``).
    express : bool
        Use the annual express product.
    return_regridder : bool
        Also return the ``xesmf.Regridder``.

    Returns
    -------
    pd.Series or (pd.Series, xesmf.Regridder)
        The prior flux named ``"flux"``.
    """
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
    import xesmf as xe  # pyright: ignore[reportMissingImports]  # conda-forge only; lazy

    regridder = xe.Regridder(total, out_grid, method="conservative")
    inventory = regridder(total)

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

    # Align to exact flux_times (finer-than-inventory requests repeat their period)
    prior = align_to_flux_times(inventory, flux_times).to_series()

    if return_regridder:
        return prior, regridder
    return prior


def load_edgar_prior(
    out_grid,
    flux_times,
    flux_freq=None,
    bbox=None,
    extent=None,
    units=None,
    return_regridder=False,
):
    """EDGAR v8 annual CH4 prior -- the sensitivity alternative to the EPA prior.

    Mirrors ``load_epa_prior``'s express branch (load -> clip -> convert -> sum sectors ->
    conservative regrid -> resample/align to flux_times). EDGAR v8 annual covers 1970-2022,
    so 2023 holds 2022 (cf. EPA holding 2020 for 2021-2023).
    """
    edgar = inventories.EDGARv8("CH4", time_step="annual")

    if any([bbox, extent]):
        edgar = edgar.clip(bbox=bbox, extent=extent)
    if units:
        edgar = edgar.convert_units(units)

    total = inventories.sum_sectors(edgar.data)

    # Regrid
    import xesmf as xe  # pyright: ignore[reportMissingImports]  # conda-forge only; lazy

    regridder = xe.Regridder(total, out_grid, method="conservative")
    inventory = regridder(total)

    inventory.name = "flux"
    inventory.attrs["units"] = total.attrs["units"]

    if flux_freq is not None:
        inv_freq = pd.infer_freq(inventory.time.values)
        if inv_freq is not None:
            ref = pd.Timestamp("2020-01-01")
            target_step = ref + pd.tseries.frequencies.to_offset(flux_freq)
            inv_step = ref + pd.tseries.frequencies.to_offset(inv_freq)
            if target_step > inv_step:
                inventory = inventory.resample(time=flux_freq).mean()

    prior = align_to_flux_times(inventory, flux_times).to_series()

    if return_regridder:
        return prior, regridder
    return prior
