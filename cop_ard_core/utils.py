"""
Shared utilities for SLC processing: rolling multilook, interferogram
computation and normalization, phase wrapping, and coregistration helpers.
"""

import copy

import numpy as np
import xarray as xr
from scipy.constants import speed_of_light


def rolling_ml(a: xr.DataArray, x: xr.DataArray, y: xr.DataArray, multilook: dict) -> xr.DataArray:
    """
    Multilook (rolling mean) the input at the given target coordinates using
    the given window sizes.
    """
    win_dims = {k: f"win_{k}" for k in multilook}

    # 1. Apply a rolling mean (moving window)
    # This computes the mean for EVERY pixel. By default (center=False),
    # xarray places the result at the end of the window.
    ml_view = a.rolling(multilook).construct(win_dims)

    # 2. Build the rolling view, decimate, and select the nearest points based on the provided coordinates
    ml_view_sel = ml_view.sel(azimuth_time=y, slant_range_time=x, method="nearest")

    # 3. Compute the mean along the constructed window dimensions
    res = ml_view_sel.mean(dim=list(win_dims.values()))

    return res


def rolling_kernel(
    a: xr.DataArray, x: xr.DataArray, y: xr.DataArray, multilook: dict, kernel: np.ndarray
) -> xr.DataArray:
    """
    Apply a 2D kernel (convolution) via rolling windows sampled at the target
    coordinates, summing the kernel-weighted window.
    """
    win_dims = {k: f"win_{k}" for k in multilook}

    # 1. Apply a rolling mean (moving window)
    # This computes the mean for EVERY pixel. By default (center=False),
    # xarray places the result at the end of the window.
    ml_view = a.rolling(multilook).construct(win_dims)

    # 2. Build the rolling view, decimate, and select the nearest points based on the provided coordinates
    ml_view_sel = ml_view.sel(azimuth_time=y, slant_range_time=x, method="nearest")

    # 3. setup kernel as DataArray with the same dimensions as the rolling view
    kernel_da = xr.DataArray(kernel, dims=list(win_dims.values()))

    # 4. Compute the mean along the constructed window dimensions
    res = (ml_view_sel * kernel_da).sum(dim=list(win_dims.values()))

    return res


def get_interf(a: xr.DataArray, b: xr.DataArray, posting: dict[str, int]):
    """
    Compute the complex interferogram between two SLCs and multilook it with
    the given posting factors.
    """
    multilook = {k: v * 2 for k, v in posting.items()}
    win_dims = {k: f"win_{k}" for k in multilook}
    slices = {k: slice(v - 1, None, posting[k]) for k, v in multilook.items()}

    # 1. Compute the complex interferogram
    interf = a * b.conj()
    # 2. Apply a rolling mean (moving window)
    # This computes the mean for EVERY pixel. By default (center=False),
    # xarray places the result at the end of the window.

    # 3. Build the rolling view, decimate, and calculate the mean efficiently
    interf_view = interf.rolling(multilook).construct(win_dims)
    interf_view_sel = interf_view.isel(slices)
    interf_rolling = interf_view_sel.mean(dim=list(win_dims.values()))

    return interf_rolling


def get_normalized_interf(
    a: xr.DataArray,
    b: xr.DataArray,
    posting: dict[str, int],
) -> xr.DataArray:
    """
    Compute the normalized interferogram (unit-magnitude coherence form)
    between two SLCs, with invalid values filled with 0.
    """
    return (
        get_interf(a, b, posting) / np.sqrt(get_interf(a, a, posting).real * get_interf(b, b, posting).real)
    ).fillna(0j)


def compute_interf(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    """
    Compute the complex interferogram (reference conjugate times secondary)
    and coarsen it with the given multilook factors.
    """
    # complex multiplication and average #
    # consider only secondary image coordianates, as the reference image is assumed to be the same for all secondary images
    return (reference.conj() * secondary).coarsen(multilook, boundary="trim").mean()  # type: ignore


def get_power(slc: xr.DataArray, multilook: dict):
    """
    Compute the multilooked power (intensity) of an SLC.
    """
    # compute the image power and average
    return (slc * slc.conj()).real.coarsen(multilook, boundary="trim").mean()  # type: ignore


def compute_interf_norm(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    """
    Compute the normalized interferogram between a reference and a secondary
    SLC, filling invalid values with 0.
    """
    # compute interferogram #
    num = compute_interf(reference, secondary, multilook)

    # compute power of the images
    pow_ref = get_power(reference, multilook)
    pow_sec = get_power(secondary, multilook)

    # compute the normalization factor
    den = np.sqrt(pow_ref * pow_sec)

    # fill invalid values with 0 (null-coherence)
    return (num / den).fillna(0j)


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    """
    Wrap values into the interval (-w/2, w/2].
    """
    w_ = w / 2.0
    return ((v + w_) % w) - w_


def get_coordinates(y, x, alpha, beta):
    """
    Calculates coordinates for polynomial evaluation based on offsets and scaling.
    """

    xx = (x - alpha[1]) / beta[1]
    yy = (y - alpha[0]) / beta[0]
    # yy = (y - np.datetime64(int(alpha[0]), "ns")).astype(float) / beta[0]
    return (yy, xx)


def get_daz_corr(coreg_azimuth_time, coreg_slant_range_time, y_poly):
    """
    Evaluate the azimuth coregistration polynomial on the given coordinates,
    returning the per-pixel azimuth time correction.
    """
    az_time_ = (coreg_azimuth_time - np.datetime64(y_poly.attrs["off_y"], "ns")).astype(float)
    c1, c2 = get_coordinates(az_time_, coreg_slant_range_time, y_poly.offset, y_poly.scale_fact)
    return np.polynomial.polynomial.polygrid2d(c1.values, c2.values, y_poly.values)


def get_coeff_demod(dkcoa_polynomial_coeffs: np.ndarray):
    """
    Integrate the Doppler centroid offset polynomial and scale by 2*pi to get
    the demodulation phase ramp coefficients.
    """
    return (2 * np.pi) * np.polynomial.polynomial.polyint(dkcoa_polynomial_coeffs, axis=1)


# @timeit
def get_phramp_diff(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    drg: xr.DataArray,
    daz: xr.DataArray,
    timeca_polynomial_coeffs: np.ndarray,
    azimuth_grid_spacing: float,
    dkcoa_polynomial_coeffs: np.ndarray,
    scp_pixel_azimuth_time: np.datetime64,
    scp_pixel_slant_range_time: np.float64,
    y_poly: xr.DataArray,
):
    """
    Compute the phase ramp difference between the corrected and uncorrected
    azimuth positions, as a 2D polynomial phase ramp DataArray.
    """

    # @timeit
    def poly_combine(rg, az_off, az_ori, coeff_demod):
        """
        Difference of the demodulation ramp evaluated at the corrected and
        original azimuth offsets.
        """
        return np.polynomial.polynomial.polyval2d(rg, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
            rg, az_ori, coeff_demod
        )

    # extract demodulation coefficients
    coeff_demod = get_coeff_demod(dkcoa_polynomial_coeffs)

    # compute spacing and start times
    dtaz = timeca_polynomial_coeffs[1] * azimuth_grid_spacing * 1e9
    scale_daz = 1 / (timeca_polynomial_coeffs[1] * 1e9)

    # compute azimuth correction
    daz_corr = get_daz_corr(azimuth_time, slant_range_time, y_poly)

    # Corrected slant range and azimuth positions
    daz_corr = daz + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

    # compute the position in the secondary image corresponding to the reference one with and without corrections in meters wrt the scene centre
    az_off = ((azimuth_time + daz_corr - scp_pixel_azimuth_time).astype(float) * scale_daz).values
    az_ori = ((azimuth_time + daz - scp_pixel_azimuth_time).astype(float) * scale_daz).values

    # range in meters wrt the burst center
    rg = ((slant_range_time + drg - scp_pixel_slant_range_time).T * (speed_of_light / 2)).values

    # recontruct polynomial function
    ph_ramp_diff_da = xr.DataArray(
        poly_combine(rg, az_off, az_ori, coeff_demod),
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "slant_range_time": slant_range_time,
            "azimuth_time": azimuth_time,
        },
    )

    return ph_ramp_diff_da


def time_to_pixel(
    stack: xr.Dataset | xr.DataArray, y0: float | None = None, x0: float | None = None
) -> xr.Dataset | xr.DataArray:
    """

    Convert from times to pixel dimensions

    """

    if y0 is None:
        y0 = stack.azimuth_time[0].values
    if x0 is None:
        x0 = stack.slant_range_time[0].values

    # spacing computation #
    y_spacing, x_spacing = [
        (stack[e][-1] - stack[e][0]).values / (stack[e].size - 1) for e in ["azimuth_time", "slant_range_time"]
    ]

    # times to pixel coordinates #
    x = np.round((stack.slant_range_time - x0) / x_spacing).values.astype("int32")
    y = np.round((stack.azimuth_time - y0) / y_spacing).values.astype("int32")

    stack = stack.assign_coords({"y": (["azimuth_time"], y), "x": (["slant_range_time"], x)})
    return stack.swap_dims(azimuth_time="y", slant_range_time="x")


def get_best_pol(polarization_values: list) -> str:
    """
    Select the preferred polarization based on a priority list.

    Parameters
    ----------
    polarization_values : list
        List of available polarizations.

    Returns
    -------
    str
        The selected polarization.
    """
    # select preferred polarization
    available_pols = {str(i) for i in polarization_values}
    priorities = ["VV", "HH", "VH", "HV"]
    pol = next((p for p in priorities if p in available_pols), None)
    if pol is None:
        raise Exception(f"No polarization has been found {available_pols}")

    return pol


def sync_sicd_metadata(sicd_dict):
    """
    Synchronizes SICD metadata across polarizations to ensure that
    polarization-independent fields are identical to the best polarization.

    Parameters
    ----------
    sicd_dict : dict
        A dictionary containing SICD metadata dictionaries, keyed by
        polarization strings (e.g., 'VV', 'VH').

    Returns
    -------
    dict
        The synchronized SICD metadata dictionary with uniform shared blocks.
    """
    # Early exit if there is only one (or zero) polarization
    if len(sicd_dict) <= 1:
        return sicd_dict

    best_pol = get_best_pol(list(sicd_dict))
    best_meta = sicd_dict[best_pol]

    shared_blocks = [
        "GeoData",
        "Grid",
        "Timeline",
        "Position",
        "SCPCOA",
        "RMA",
        "ImageCreation",
        "ImageData",
        "RadarCollection",
    ]

    blocks_to_copy = [block for block in shared_blocks if block in best_meta]
    # Parameters to preserve during the copy
    keys_to_save = ["PREDICTED_RNIIRS", "INFORMATION_DENSITY"]

    for pol, meta in sicd_dict.items():
        if pol == best_pol:
            continue

        # 1. COPY SHARED BLOCKS
        for block in blocks_to_copy:
            meta[block] = copy.deepcopy(best_meta[block])

        # 2. MANAGE COLLECTION INFO (Mixed)
        params = meta.get("CollectionInfo", {}).get("Parameters", {})
        pol_specific_params = {k: params[k] for k in keys_to_save if k in params}

        # Copy the entire CollectionInfo block from the best metadata
        if "CollectionInfo" in best_meta:
            meta["CollectionInfo"] = copy.deepcopy(best_meta["CollectionInfo"])
            meta["CollectionInfo"].setdefault("Parameters", {}).update(pol_specific_params)

    return sicd_dict


def get_azimuth_time_raw(
    burst: xr.DataArray,
    timeca_polynomial_coeffs: np.ndarray,
    collection_start_time: np.datetime64,
    multilook: dict | None = None,
) -> xr.DataArray:
    """
    Compute the raw (uncorrected) azimuth time at each pixel from the
    time-COA polynomial, as absolute datetime64 values.
    """
    if multilook is not None:
        az_slice = slice(multilook["azimuth_time"] // 2, None, multilook["azimuth_time"])
        rg_slice = slice(multilook["slant_range_time"] // 2, None, multilook["slant_range_time"])
    else:
        az_slice = slice(None, None, None)
        rg_slice = slice(None, None, None)

    burst_ = burst.isel(azimuth_time=az_slice, slant_range_time=rg_slice)

    # retrieve axis and coefficients
    slant_range_m = burst_.slant_range_m.values
    azimuth_m = burst_.azimuth_m.values

    # compute the relative azimuth time
    timeCOA = np.polynomial.polynomial.polygrid2d(slant_range_m, azimuth_m, timeca_polynomial_coeffs).T

    # convert to timedelta in nanoseconds
    timeCOA = (timeCOA * 1e9).astype("timedelta64[ns]")

    # compute absolute time
    timeCOA = timeCOA + collection_start_time

    # return as dataarray
    return xr.DataArray(
        timeCOA,
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "azimuth_time": burst_.azimuth_time.values,
            "slant_range_time": burst_.slant_range_time.values,
        },
    )
