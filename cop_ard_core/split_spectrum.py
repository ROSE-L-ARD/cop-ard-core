import numpy as np
import xarray as xr
from s1_ard_core.computing.logger import logger, timeit

import cop_ard_core.utils


@timeit
def get_interf_single_burst(
    cslc: xr.Dataset, reference: str, multilook: dict, x_off: int = 0, y_off: int = 0
) -> xr.Dataset:
    """
    Compute the interferometric phase and coherence with respect to a reference date
    applying a boxcar filter
    """
    # multilook of complex interferogram
    interf = (cslc * cslc.sel(acquisition_date=reference).conj()).coarsen(multilook, boundary="trim").mean()

    # compute the power image
    power = np.sqrt((np.abs(cslc) ** 2).coarsen(multilook, boundary="trim").mean())

    # normalize the complex interferohgram
    interf = interf / (power * power.sel(acquisition_date=reference) + 1e-9)

    interf_ds = xr.Dataset()

    # extract phase
    interf_ds["phase"] = xr.apply_ufunc(np.angle, interf, dask="parallelized")

    # extract coherence
    interf_ds["cohe"] = xr.apply_ufunc(np.abs, interf, dask="parallelized")

    # assign pixel coordinates
    interf_ds["azimuth_time"] = (
        multilook["azimuth_time"] / 2 - 0.5 + y_off + np.arange(interf_ds.azimuth_time.size) * multilook["azimuth_time"]
    )
    interf_ds["slant_range_time"] = (
        multilook["slant_range_time"] / 2
        - 0.5
        + x_off
        + np.arange(interf_ds.slant_range_time.size) * multilook["slant_range_time"]
    )

    return interf_ds


@timeit
def extract_coreg_aoi(coreg: xr.Dataset, bbox: tuple, dt_az: float, dt_rg: float, multilook: dict):
    a_min, a_max, r_min, r_max = bbox

    dt_az_ml = dt_az * multilook["azimuth_time"]
    dt_rg_ml = dt_rg * multilook["slant_range_time"]

    # extract sub-band
    cslc = coreg.cstack

    # extract chunksizes
    chunk_burst = {k: v[0] for k, v in cslc.chunksizes.items() if k in ["azimuth_time", "slant_range_time"]}

    # select area according to the output multilook grid
    logger.debug("Select area ...")
    x_min = np.ceil((cslc.slant_range_time.min().values - r_min) / dt_rg_ml) * dt_rg_ml + r_min
    x_min = np.round((x_min - cslc.slant_range_time.min().values) / dt_rg).astype(int)
    x_max = np.floor((cslc.slant_range_time.max().values - r_min) / dt_rg_ml) * dt_rg_ml + r_min
    x_max = np.round((x_max - cslc.slant_range_time.min().values) / dt_rg).astype(int)

    y_min = (np.ceil((cslc.azimuth_time.min().values - a_min).astype(float) / dt_az_ml) * dt_az_ml).astype(
        "timedelta64[ns]"
    ) + a_min
    y_min = np.round((y_min - cslc.azimuth_time.min().values).astype(float) / dt_az).astype(int)
    y_max = (np.floor((cslc.azimuth_time.max().values - a_min).astype(float) / dt_az_ml) * dt_az_ml).astype(
        "timedelta64[ns]"
    ) + a_min
    y_max = np.round((y_max - cslc.azimuth_time.min().values).astype(float) / dt_az).astype(int)

    # select the cslc according to the reference grid
    coreg_sel = coreg.isel(slant_range_time=slice(x_min, x_max), azimuth_time=slice(y_min, y_max)).chunk(chunk_burst)

    return (coreg_sel, y_min, y_max, x_min, x_max)


@timeit
def extract_interf(
    burst_id: str,
    orig: xr.DataTree,
    coreg: xr.Dataset,
    bbox: tuple,
    dt_az: float,
    dt_rg: float,
    multilook: dict,
    reference: str,
    y_poly: xr.Dataset | None = None,
    f0: float | None = None,
    bw: float | None = None,
):
    logger.debug(f"processing burst {burst_id}")

    a_min, a_max, r_min, r_max = bbox
    chunk_burst = {k: v[0] for k, v in coreg.chunksizes.items() if k in ["azimuth_time", "slant_range_time"]}

    # extract aoi
    coreg_sel, y_min, y_max, x_min, x_max = extract_coreg_aoi(coreg, bbox, dt_az, dt_rg, multilook)

    # extract subbands
    if (f0 is not None) and (bw is not None):
        logger.debug("Computing full range spectrum...")

        cslc = cop_ard_core.utils.extract_subband(coreg_sel.cstack, bw, f0).chunk(chunk_burst)
    else:
        cslc = coreg_sel.cstack

    if y_poly is not None:
        # correct the cslc phase according to the polynomial offset
        ph_ramp_diff = cop_ard_core.utils.get_phramp_diff(coreg_sel, orig, y_poly).chunk(chunk_burst)

        # force coordinates of ph_ramp_diff to be the same of cslc
        ph_ramp_diff["slant_range_time"] = cslc.slant_range_time.values
        ph_ramp_diff["azimuth_time"] = cslc.azimuth_time.values

        # detrend the cslc
        cslc = cslc * np.exp(-1j * ph_ramp_diff)

    # set offset with respect to the reference grid
    x_off = np.round((cslc.slant_range_time[0].values - r_min) / dt_rg).astype(int)
    y_off = np.round((cslc.azimuth_time[0].values - a_min).astype(float) / dt_az).astype(int)

    # reset coordinates as pixel wrt the reference grid
    cslc["slant_range_time"] = np.arange(x_min + x_off, x_max + x_off)
    cslc["azimuth_time"] = np.arange(y_min + y_off, y_max + y_off) * dt_az

    logger.debug("extract multilooked interferogram and coherence")
    # extract multilooked interferogram and coherence
    interf = get_interf_single_burst(cslc, reference, multilook=multilook, x_off=x_off, y_off=y_off)

    # force coordinates
    interf["slant_range_time"] = interf.slant_range_time.values * dt_rg + r_min
    interf["azimuth_time"] = (interf.azimuth_time.values * dt_az).astype("timedelta64[ns]") + a_min

    if y_poly is not None:
        logger.debug("save phase_ramp")
        # compute multilooked phase correction
        ph_ramp_diff_ml = ph_ramp_diff.coarsen(multilook, boundary="trim").mean()
        ph_ramp_diff_ml["slant_range_time"] = interf["slant_range_time"]
        ph_ramp_diff_ml["azimuth_time"] = interf["azimuth_time"]
        interf["phase_ramp"] = ph_ramp_diff_ml

    return (burst_id, interf.compute())
