# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import reduce
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union, cast  # noqa: F401

import dask.array as da
import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.signal
import scipy.signal.windows
import xarray as xr
from numba import njit
from scipy.constants import speed_of_light

from cop_ard_core.computing.etad import apply_etad_corrections
from cop_ard_core.computing.griddata import init_interpolate, interp_dataset, interpolate
from cop_ard_core.computing.logger import logger
from cop_ard_core.computing.simul import (
    extract_simul_on_radar_chunk,
    get_geo_from_radar_bbox,
    get_position,
    get_radar_bbox,
)
from cop_ard_core.computing.slc import get_coefficients

# Define a default grid size for processing
GRID = 11, 11
SIGMA_ORB_2 = {
    "AUX_ANNOTATED": 0.10**2,
    "AUX_PREORB": 1.00**2,
    "AUX_RESORB": 0.10**2,
    "AUX_MOEORB": 0.05**2,
    "AUX_POEORB": 0.05**2,
}
DIMENSIONS_ = ["acquisition_date", "azimuth_time", "slant_range_time"]
OV_FACT = 100_000


def sinc(size: int, ov_fact: int = OV_FACT, kaiser_beta: float = 10) -> np.ndarray:
    """
    Generate a sinc function weighted by a Kaiser window.

    This function creates a one-dimensional sinc kernel, which is often used
    for high-quality signal resampling and interpolation.
    To make it practical, it is truncated and
    weighted by a window function (in this case, a Kaiser window) to taper
    its edges and reduce ringing artifacts (Gibbs phenomenon).

    Parameters
    ----------
    size : int
        The final desired size of the sinc function kernel.
    ov_fact : int, optional
        An oversampling factor to increase the resolution of the
        underlying sinc function before sampling. (Default is OV_FACT)
    kaiser_beta : float, optional
        The beta parameter for the Kaiser window, which controls
        the trade-off between main-lobe width and side-lobe level. (Default is 10)

    Returns
    -------
    np.ndarray
        The computed sinc function values weighted by the Kaiser window.
    """
    # Compute the length of the oversampled sinc to ensure high resolution.
    n = int(size * ov_fact)
    # Ensure n is an odd number so the kernel is symmetric around its center.
    n += 1 - (n % 2)
    # Create the x-coordinates for the sinc function, centered at 0.
    x = (np.arange(n) - (n - 1) / 2.0) / ov_fact
    # Calculate the sinc values and apply the Kaiser window element-wise.
    return np.sinc(x) * scipy.signal.windows.kaiser(n, kaiser_beta, sym=True)


@njit(cache=True)
def _dot_numba(a, b, c):
    """
    Perform a dot product of two vectors with a matrix, optimized with Numba.
    This computes (a @ c.T) @ b efficiently.

    Parameters
    ----------
    a : np.ndarray
        A 1D array (vector) of size M.
    b : np.ndarray
        A 1D array (vector) of size N.
    c : np.ndarray
        A 2D array (matrix) of shape (M, N).

    Returns
    -------
    scalar
        The result of the dot product computation.
    """
    row = c.dtype.type(0.0)
    # This loop computes the dot product of vector 'b' with each row of matrix 'c'.
    for j in range(a.size):
        col = c.dtype.type(0.0)
        for i in range(b.size):
            col += b[i] * c[j, i]
        # This part multiplies the result with the corresponding element of vector 'a' and accumulates.
        row += col * a[j]
    return row


@njit(cache=True)
def _nvalid_numba(image, iv):
    """
    Count the number of valid (non-invalid) pixels in an image, optimized with Numba.
    """
    # If the invalid value is a finite number, compare directly.
    if np.isfinite(iv):
        v = image != iv
    # If the invalid value is NaN or Inf, check for finite values.
    else:
        v = np.isfinite(image)
    return np.count_nonzero(v)


@njit(cache=True)
def _sinc_warp_inner_numba(
    image,
    x_coords,
    y_coords,
    kernel_h,
    kernel_w,
    sinc_x,
    sinc_y,
    valid_threshold,
    output,
    iv,
    ov_fact: int = OV_FACT,
):
    """
    Core inner loop for sinc warping, accelerated with Numba.

    This function iterates over each coordinate, extracts the corresponding
    image patch, and performs sinc interpolation.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    x_coords : np.ndarray
        1D array of x-coordinates (in pixels) at which to interpolate.
    y_coords : np.ndarray
        1D array of y-coordinates (in pixels) at which to interpolate.
    kernel_h : int
        The height of the sinc interpolation kernel.
    kernel_w : int
        The width of the sinc interpolation kernel.
    sinc_x : np.ndarray
        1D array of precomputed sinc kernel values for the x-direction.
    sinc_y : np.ndarray
        1D array of precomputed sinc kernel values for the y-direction.
    valid_threshold : float
        A threshold used to determine if an interpolated pixel is valid.
    output : np.ndarray
        A pre-allocated array where the interpolated values are written.
    iv : np.ndarray
        A pre-allocated array (likely boolean) where the validity mask is written.
    ov_fact : int, optional
        The oversampling factor used for the sinc kernels. (Default is OV_FACT)

    Returns
    -------
    None
        This function modifies `output` and `iv` in place.
    """
    # Calculate the half-size of the kernel for easier indexing.
    x_s = kernel_w // 2
    y_s = kernel_h // 2
    kernel_size = kernel_w * kernel_h
    # Iterate over each point to be interpolated.
    for idx in range(x_coords.shape[0]):
        x_f = x_coords[idx]
        y_f = y_coords[idx]
        # Get the nearest integer pixel coordinates.
        x_i = int(np.round(x_f))
        y_i = int(np.round(y_f))
        # Skip interpolation if the kernel would go outside the image boundaries.
        if (x_i < x_s) or (x_i >= image.shape[1] - x_s) or (y_i < y_s) or (y_i >= image.shape[0] - y_s):
            continue
        # Calculate the fractional (sub-pixel) offset.
        x_d = x_f - x_i
        y_d = y_f - y_i
        # Determine the correct slice from the oversampled sinc LUT based on the fractional offset.
        xstart = int(np.round((-x_s - x_d) * ov_fact + sinc_x.size // 2))
        xend = int(np.round((x_s + 1 - x_d) * ov_fact + sinc_x.size // 2))
        wx = sinc_x[xstart:xend:ov_fact]
        ystart = int(np.round((-y_s - y_d) * ov_fact + sinc_y.size // 2))
        yend = int(np.round((y_s + 1 - y_d) * ov_fact + sinc_y.size // 2))
        wy = sinc_y[ystart:yend:ov_fact]
        # Extract the local image patch (sub-image) around the interpolation point.
        im_sub = image[y_i - y_s : y_i + y_s + 1, x_i - x_s : x_i + x_s + 1]
        # If the patch contains too many invalid pixels, mark the output as invalid.
        if _nvalid_numba(im_sub, iv) < (valid_threshold * kernel_size):
            output[idx] = iv
        # Otherwise, perform the 2D interpolation using the dot product.
        else:
            output[idx] = _dot_numba(wy, wx, im_sub)


def sinc_warp(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    kernel: dict,
    valid_percentage_threshold: float = 0.75,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """
    Apply cardinal interpolation on a 2D image using a sinc kernel.

    This function resamples a 2D image at a new set of coordinates (x, y)
    using a high-quality sinc interpolation method. It's designed to handle
    sub-pixel shifts accurately. The function checks for a minimum number
    of valid pixels within the interpolation kernel's footprint to avoid
    propagating errors from invalid data regions.

    Parameters
    ----------
    image : np.ndarray
        The 2D input image to be interpolated.
    x : np.ndarray
        X-coordinates (in pixels) of the points to interpolate.
    y : np.ndarray
        Y-coordinates (in pixels) of the points to interpolate.
    kernel : dict
        A dictionary containing pre-computed interpolator parameters:
        - `kernel_shape`: Shape of the interpolation kernel (height, width).
        - `ov_fact`: Sinc oversampling factor.
        - `sinc_x`: LUT with the pre-computed sinc function for the x-direction.
        - `sinc_y`: LUT with the pre-computed sinc function for the y-direction.
    valid_percentage_threshold : float, optional
        The minimum percentage of valid pixels
        within the kernel footprint required to produce a valid output.
        Defaults to 0.75.
    invalid_value : float, optional
        The value representing invalid pixels in the input
        and to be used for invalid output. Defaults to 0.0.

    Returns
    -------
    np.ndarray
        An array of interpolated values at the specified (x, y) coordinates.

    Raises
    ------
    Exception
        If the input image is not 2D or if x and y coordinate
        arrays have mismatched shapes.
    """
    # --- Input Validation ---
    if not (image.ndim == 2):
        raise Exception(f"Image must have 2 dimensions, got {image.ndim}")
    if not (x.shape == y.shape):
        raise Exception(f"x and y shape must be the same, got x {x.shape}, y {y.shape}")
    # --- Parameter Extraction ---
    kernel_h, kernel_w = kernel["kernel_shape"]
    sinc_x, sinc_y = kernel["sinc_x"], kernel["sinc_y"]

    # --- Initialization ---
    # Define the invalid value with the same data type as the image.
    iv = image.dtype.type(invalid_value)
    # Create an output array filled with zeros.
    output = np.zeros_like(x, dtype=image.dtype)
    # --- Core Processing ---
    # Call the Numba-optimized inner function to perform the interpolation.
    _sinc_warp_inner_numba(
        image,
        x,
        y,
        kernel_h,
        kernel_w,
        sinc_x,
        sinc_y,
        valid_percentage_threshold,
        output,
        iv,
        kernel["ov_fact"],
    )
    return output


def warp_dazdrg(
    cstack: xr.DataArray,
    daz: xr.DataArray | None,
    drg: xr.DataArray | None,
    simulation_reference: xr.Dataset,
    simulation_secondary: xr.Dataset | None,
    reference_ml: dict,
    margin: dict,
    coregistration_reference_date: str,
    azimuth_grid_spacing: float,
    range_grid_spacing: float,
    compute_daz_drg: bool,
) -> xr.Dataset:
    """
    Compute the necessary shifts in azimuth (daz) and range (drg) to
    coregister a stack of SAR images onto the reference grid.

    Parameters
    ----------
    coreg : xr.Dataset
        The input coregistered dataset to be processed.
    daz : xr.DataArray or None
        An optional DataArray to store azimuth adjustments. If None, a new one will be created.
    drg : xr.DataArray or None
        An optional DataArray to store range adjustments. If None, a new one will be created.
    simulation_reference : xr.Dataset
        The reference simulation dataset.
    simulation_secondary : xr.Dataset or None, optional
        An optional secondary simulation dataset.
    reference_ml : dict
        Reference multi-looking parameters.
    margin : dict
        Margin values for the radar bounding box. `azimuth_time` and `slant_range_time` keys are required.
    coregistration_reference_date : str
        The reference date for coregistration, used to select the appropriate simulation data.
    azimuth_grid_spacing : float
        The grid spacing in the azimuth direction.
    range_grid_spacing : float
        The grid spacing in the range direction.
    compute_daz_drg : bool
        If True, compute the daz and drg values; if False, skip computation and return existing values.

    Returns
    -------
    xr.Dataset
        The updated LUT_coords dataset with daz and drg values.
    """
    # JIT-compile the Numba interpolation function for first use.
    init_interpolate()
    # Define the bounding box of the radar image.
    bbox_radar = np.array([0, 0, cstack.slant_range_time.size, cstack.azimuth_time.size])
    # --- Initialize daz_drg Dataset ---
    # Create an empty xarray Dataset to hold the azimuth and range shifts.
    shape = cstack.shape[1:]
    dtype = {"daz": "timedelta64[ns]", "drg": "float64"}
    if daz is None or drg is None or daz.recompute:
        daz = xr.DataArray(
            da.empty(shape, dtype=dtype["daz"]), dims=DIMENSIONS_, coords={d: cstack[d] for d in DIMENSIONS_}
        )
        drg = xr.DataArray(
            da.empty(shape, dtype=dtype["drg"]), dims=DIMENSIONS_, coords={d: cstack[d] for d in DIMENSIONS_}
        )
    # --- Prepare Data for Interpolation ---
    azimuth_time_ref = daz.azimuth_time[0].values
    slant_range_time_ref = drg.slant_range_time[0].values
    azimuth_time = daz.azimuth_time
    slant_range_time = drg.slant_range_time
    bbox_radar_buffered = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(reference_ml, bbox_radar_buffered)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    if compute_daz_drg:
        # Select the relevant simulation data based on the current radar chunk.
        selection = extract_simul_on_radar_chunk(
            simulation=simulation_reference,
            simulation_secondary=simulation_secondary,
            reference_date=coregistration_reference_date,
            bbox_radar=bbox_radar_buffered,
            bbox_geo=bbox_geo,
        )

        # --- Coregistration and Warping ---
        # Interpolate the simulation data to compute range and azimuth offsets.
        daz, drg = grid_data_block(
            daz,
            drg,
            selection.azimuth_time,
            selection.slant_range_time,
            coregistration_reference_date,
            bbox_radar,
            azimuth_grid_spacing,
            range_grid_spacing,
            azimuth_time_ref,
            slant_range_time_ref,
        )

    daz_drg = xr.Dataset({"daz": daz, "drg": drg})
    return daz_drg


def compute_secondary_image_coordinates(
    coreg_azimuth_time: xr.DataArray,
    coreg_slant_range_time: xr.DataArray,
    slc: xr.DataArray,
    etad_correction_grid: xr.Dataset,
    daz: xr.DataArray,
    drg: xr.DataArray,
    azimuth_grid_spacing: float,
    range_grid_spacing: float,
    y_poly: xr.DataArray | None,
    x_poly: xr.DataArray | None,
    chunk: dict,
):
    """
    Computes final target azimuth and slant-range coordinates by applying
    ETAD and optional polynomial corrections.

    """

    def get_coordinates(alpha, beta, coord_type: str = "pixels"):
        """
        Calculates coordinates for polynomial evaluation based on offsets and scaling.
        """
        if coord_type == "pixels":
            y = coreg_azimuth_time.size + chunk["azimuth_time"].start
            x = coreg_slant_range_time.size + chunk["slant_range_time"].start
        elif coord_type == "times":
            y, x = coreg_azimuth_time, coreg_slant_range_time
        else:
            raise RuntimeError(f"Invalid coordinate type: {coord_type}")
        return (((e - alpha[i]).astype("float64") / beta[i].astype("float64")).values for i, e in enumerate([y, x]))

    # Compute grid spacing (dtaz, dtrg) and reference start times (rg0, az0) from metadata.
    rg0 = slc.isel(slant_range_time=0).slant_range_time.values
    az0 = slc.isel(azimuth_time=0).azimuth_time.values

    # --- Coordinate Calculation and Correction ---
    # Calculate the target azimuth and slant-range times in the original image's grid.
    az = (coreg_azimuth_time + daz).transpose("azimuth_time", "slant_range_time")
    rg = (coreg_slant_range_time + drg).transpose("azimuth_time", "slant_range_time")

    # Apply ETAD (Extended Timing Annotation Dataset) corrections for precise timing.
    daz, drg = apply_etad_corrections(
        az,
        rg,
        etad_correction_grid,
        azimuth_grid_spacing,
        range_grid_spacing,
        rg0,
        az0,
    )
    # Apply optional polynomial corrections to azimuth coordinates.
    if y_poly is not None:
        c1, c2 = get_coordinates(y_poly.offset, y_poly.scale_fact, y_poly.type)
        corr = np.polynomial.polynomial.polygrid2d(c1, c2, y_poly.values)
        daz = daz + np.round(corr * azimuth_grid_spacing).astype("timedelta64[ns]")

    # Apply optional polynomial corrections to range coordinates.
    if x_poly is not None:
        c1, c2 = get_coordinates(x_poly.offset, x_poly.scale_fact, x_poly.type)
        corr = np.polynomial.polynomial.polygrid2d(c1, c2, x_poly.values)
        drg = drg + corr * range_grid_spacing

    # Compute the final target coordinates in the secondary image.
    az = (az + daz).transpose("azimuth_time", "slant_range_time").values
    rg = (rg + drg).transpose("azimuth_time", "slant_range_time").values
    return az, rg, az0, rg0


def warp_cslc(
    coreg_cstack: xr.DataArray,
    coreg_phase_model: xr.DataArray,
    daz: xr.DataArray,
    drg: xr.DataArray,
    coreg_acquisition_date: xr.DataArray,
    coregistration_reference_date: np.datetime64,
    orig_slcs: xr.DataTree,
    orig_etad: xr.DataTree,
    kernel: dict,
    do_reramp: bool = True,
    do_flattening: bool = True,
    x_poly: xr.DataArray | None = None,
    y_poly: xr.DataArray | None = None,
    chunk: dict = {},
):
    """
    Perform sinc interpolation and processing of a stack of SAR images.

    This function iterates through each acquisition in a coregistered SAR
    dataset, applies geometric and polynomial corrections, and warps the
    original SLC (Single Look Complex) image to the reference grid using
    sinc interpolation. It also handles optional phase adjustments like
    flattening and reramping.

    Parameters
    ----------
    coreg : xr.Dataset
        Coregistered SAR dataset containing the target grid.
    daz_drg : xr.Dataset
        A dataset with azimuth and slant-range adjustment values.
    orig : list
        The original SAR dataset used as a reference.
    kernel : dict
        A dictionary containing parameters for the sinc interpolation kernel.
    do_reramp : bool, optional
        If True, apply phase reramping to the warped image. Defaults to True.
    do_flattening : bool, optional
        If True, apply phase flattening. Defaults to False.
    x_poly : xr.DataArray or None, optional
        Optional polynomial corrections for the range (x dimension).
    y_poly : xr.DataArray or None, optional
        Optional polynomial corrections for the azimuth (y dimension).
    store_path : str or None, optional
        Path to save the processed SAR stack.
    chunk : object, optional
        The chunk configuration for saving data.

    Returns
    -------
    xr.Dataset
        The updated coregistered dataset with the warped cstack.
    """
    # Define a margin around the processing area based on the kernel size to avoid edge effects.
    margin = kernel["kernel_shape"] // 2 + 1

    # --- Main Loop: Process each acquisition date ---
    for i, dd in enumerate(coreg_acquisition_date.values):
        # Extract the original SLC image and its metadata for the current date.
        slc = orig_slcs.orig[dd].slc
        single_pol_annotation = slc.attrs[list(slc.attrs.keys())[0]]
        min_tx_frequency = single_pol_annotation["RadarCollection"]["TxFrequency"]["Min"]
        max_tx_frequency = single_pol_annotation["RadarCollection"]["TxFrequency"]["Max"]
        dkcoa_polynomial_coeffs = get_coefficients("Grid.Col.DeltaKCOAPoly.Coefs", single_pol_annotation)
        timeca_polynomial_coeffs = get_coefficients("RMA.INCA.TimeCAPoly.Coefs", single_pol_annotation)
        if dkcoa_polynomial_coeffs is None or timeca_polynomial_coeffs is None:
            raise ValueError(f"Failed to extract polynomial coefficients for acquisition date {dd}")

        dtaz = timeca_polynomial_coeffs[1] * single_pol_annotation["Grid"]["Col"]["SS"] * 1e9
        dtrg = 2 * single_pol_annotation["Grid"]["Row"]["SS"] / speed_of_light
        scp_pixel_row = single_pol_annotation["ImageData"]["SCPPixel"]["Row"]
        scp_pixel_col = single_pol_annotation["ImageData"]["SCPPixel"]["Col"]
        etad_correction_grid = orig_etad[dd].ds

        az, rg, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_cstack.azimuth_time,
            coreg_slant_range_time=coreg_cstack.slant_range_time,
            slc=slc,
            etad_correction_grid=etad_correction_grid,
            daz=daz.sel(acquisition_date=dd),
            drg=drg.sel(acquisition_date=dd),
            azimuth_grid_spacing=dtaz,
            range_grid_spacing=dtrg,
            y_poly=y_poly,
            x_poly=x_poly,
            chunk=chunk,
        )
        # Convert the time-based coordinates to pixel-based coordinates for interpolation.
        x, y = ((rg - rg0) / dtrg, (az - az0).astype(float) / dtaz)
        # --- Warping Execution ---
        # Optimization: If it's the reference image with no corrections, just copy the data without warping.
        if (
            (dd == coregistration_reference_date)
            & (y_poly is None)
            & (x_poly is None)
            & (etad_correction_grid is None)
        ):
            # No warping is required; directly slice the original image.
            y, x = np.round(y).astype(int), np.round(x).astype(int)
            coreg_cstack.loc[:, dd] = (
                slc.isel(azimuth_time=slice(y.min(), y.max() + 1), slant_range_time=slice(x.min(), x.max() + 1))
                .sel(polarization=coreg_cstack.polarization)
                .squeeze("acquisition_date", drop=True)
            )
        else:
            # --- Perform Sinc Warping ---
            # Compute the bounding box of the area to read from the original SLC.
            xmin, xmax = np.floor(x.min() - margin[1]).astype(int), np.ceil(x.max() + margin[1]).astype(int)
            ymin, ymax = np.floor(y.min() - margin[0]).astype(int), np.ceil(y.max() + margin[0]).astype(int)

            # Ensure bounding box coordinates are not negative.
            xmin, ymin = (xmin if xmin > 0 else 0, ymin if ymin > 0 else 0)
            # Translate the target coordinates relative to the selected bounding box origin.
            x, y = (x - xmin, y - ymin)

            # Select the input data from the original SLC based on the bounding box.
            im_in = slc.isel(azimuth_time=slice(ymin, ymax), slant_range_time=slice(xmin, xmax))
            im_in = im_in.squeeze("acquisition_date", drop=True)

            # Identify the output indices that are safely inside the input image boundaries.
            iy, ix = np.where(
                (x >= margin[1])
                & (y >= margin[0])
                & (x < (im_in.slant_range_time.size - margin[1]))
                & (y < (im_in.azimuth_time.size - margin[0]))
            )

            # If no valid output indices are found, skip to the next acquisition.
            if ix.size == 0:
                continue

            # --- Loop over polarizations and warp ---
            for polid, pol in enumerate(coreg_cstack.polarization.values):
                im_out = np.zeros_like(x, dtype=slc.dtype)
                # Perform the sinc warping on the valid indices.
                im_out[iy, ix] = sinc_warp(im_in.sel(polarization=pol).values, x[iy, ix], y[iy, ix], kernel)
                # Assign the warped image to the output coregistered stack.
                coreg_cstack[polid, i, :, :] = im_out
        # --- Optional Phase Corrections ---
        # Optional: Apply phase flattening based on the range displacement.
        if do_flattening:
            f0 = (max_tx_frequency + min_tx_frequency) / 2.0
            flattening_rg = drg.sel(acquisition_date=dd).transpose("azimuth_time", "slant_range_time")
            coreg_phase_model[i, :, :] = (2 * np.pi * f0) * flattening_rg
            coreg_cstack[:, i, :, :] *= np.exp(1j * coreg_phase_model[i, :, :])

        # Optional: Apply a phase ramp (reramping) to the image.
        if do_reramp:
            try:
                # Get the reference point from the scene center pixel (SCP).
                rg0 = slc.isel(slant_range_time=scp_pixel_row).slant_range_time.values
                az0 = slc.isel(azimuth_time=scp_pixel_col).azimuth_time.values
            except Exception as e:
                # Fallback for test cases or missing metadata.
                logger.warning(f"type is: {e.__class__.__name__}, fix for tests")
                rg0 = slc.slant_range_time[0].values
                az0 = slc.azimuth_time[0].values
            # Convert time-based coordinates to a unitless system for polynomial evaluation.
            az = (az - az0).astype(float) / (1e9 * timeca_polynomial_coeffs[1])
            rg = (rg - rg0) * speed_of_light / 2

            # Integrate the Doppler Centroid polynomial to get the phase ramp polynomial.
            coeff = (2 * np.pi) * np.polynomial.polynomial.polyint(
                dkcoa_polynomial_coeffs, axis=1
            )
            # Evaluate the 2D polynomial to get the phase ramp values.
            phramp = np.polynomial.polynomial.polyval2d(rg, az, coeff)
            # Apply the phase ramp to the complex SLC data.
            coreg_cstack[:, i, :, :] *= np.exp(1j * phramp)
    coreg = xr.Dataset({"cstack": coreg_cstack, "phase_model": coreg_phase_model})
    return coreg


def extract_abs_grid(slc: xr.DataArray, dec: int = 0) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Extract a coarse absolute grid from an SLC DataArray.

    This function extracts the azimuth and slant range time coordinates from a
    Single Look Complex (SLC) DataArray. It allows for decimation to create a
    coarser grid, which is useful for reducing computational load in subsequent
    processing steps.

    Parameters
    ----------
    slc : xr.DataArray
        Input SLC DataArray containing azimuth_time and
        slant_range_time coordinates.
    dec : int, optional
        Decimation factor. A value of 10 would mean taking every 10th
        pixel. A value of 0 means no decimation. Defaults to 0.

    Returns
    -------
    tuple
        A tuple containing two `xr.DataArray` objects:
        - The slant_range_time coordinates.
        - The azimuth_time coordinates.
    """
    # Calculate the step size for decimation in azimuth and range.
    stepaz = np.ceil(slc.azimuth_time.size / dec) if dec >= 1 else 1
    steprg = np.ceil(slc.slant_range_time.size / dec) if dec >= 1 else 1

    # Extract the absolute grid by slicing the coordinates with the calculated step.
    az = slc.azimuth_time.isel(azimuth_time=slice(None, None, int(stepaz)))
    rg = slc.slant_range_time.isel(slant_range_time=slice(None, None, int(steprg)))

    return rg, az


def extract_grid(
    slc: xr.DataArray,
    scp_idx: tuple[int, int],
    reference: Dict[str, Any] | None = None,
    scale: Dict[str, Any] | None = None,
    dec: int = 0,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Extract a relative grid of range and azimuth positions from an input SAR image.

    This function computes a grid of relative positions in range (meters) and
    azimuth (seconds) with respect to a defined reference point. This is often
    used to prepare coordinates for polynomial fitting or interpolation, where
    a normalized coordinate system is beneficial.

    Parameters
    ----------
    slc : xr.DataArray
        The SAR image containing the slant_range_time and
        azimuth_time dimensions.
    scp_idx : tuple[int, int]
        A tuple containing the column and row indices of the scene center pixel (SCP) in the SLC image.
        This is used as the default reference point if `reference` is None.
    reference : Dict[str, Any] or None, optional
        A dictionary containing the reference point coordinates
        (`slant_range_time` and `azimuth_time`). If None, the scene
        center from the image metadata is used.
    scale : Dict[str, Any] or None, optional
        A dictionary specifying scaling factors for range (meters)
        and azimuth (seconds). If None, defaults are calculated from
        the image metadata.
    dec : int, optional
        Decimation factor to reduce the grid resolution. Defaults to 0
        (no decimation).

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        A tuple of two xarray.DataArrays representing the relative range (meters)
        and azimuth (seconds) positions.
    """
    # Extract annotations from the SAR image metadata.
    annot = slc.attrs[list(slc.attrs.keys())[0]]

    # --- Determine the Reference Point (Origin) ---
    if reference is None:
        # Default to the scene center pixel (SCP) from image metadata.
        rg0 = slc.isel(slant_range_time=scp_idx[1]).slant_range_time.values
        az0 = slc.isel(azimuth_time=scp_idx[0]).azimuth_time.values
    else:
        # Use the user-provided reference point.
        rg0 = float(reference["slant_range_time"])
        az0 = np.datetime64(reference["azimuth_time"])

    # --- Determine the Scaling Factors ---
    if scale is None:
        # Compute default scaling factors from image metadata.
        # Range scaling factor (converts time to meters).
        drg = speed_of_light / 2
        # Azimuth scaling factor (converts time difference to seconds).
        daz = 1.0 / (1e9 * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1])
    else:
        # Use the user-provided scaling factors.
        drg = float(scale["slant_range_time"])
        daz = float(scale["azimuth_time"])

    # --- Compute Relative Grid ---
    # Get the absolute grid positions (potentially decimated).
    rg, az = extract_abs_grid(slc, dec=dec)
    # Convert absolute positions to relative positions by subtracting the reference and applying scaling.
    return (rg - rg0) * drg, (az - az0).astype(float) * daz


def poly_fit_2d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, x_order: int = 2, y_order: int = 2, rcond: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Perform a fit of data (z) to a two-dimensional polynomial in (x, y).

    This function models a surface `z = f(x, y)` by finding the coefficients of
    a 2D polynomial that best fit the input data points `(x, y, z)` in a
    least-squares sense. It is useful for modeling smooth, slowly varying
    surfaces from scattered data points.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates of the data points.
    y : np.ndarray
        The y-coordinates of the data points.
    z : np.ndarray
        The z-values (surface height) of the data points.
    x_order : int, optional
        The polynomial order for the x-dimension. Defaults to 2.
    y_order : int, optional
        The polynomial order for the y-dimension. Defaults to 2.
    rcond : Optional[float], optional
        A cutoff for small singular values, passed to `scipy.linalg.lstsq`.
        Defaults to None.

    Returns
    -------
    solution : np.ndarray
        The 2D array of polynomial coefficients.
    residuals : np.ndarray
        The sum of squared residuals of the fit.
    rank : int
        The effective rank of the design matrix.
    sing_values : np.ndarray
        The singular values of the design matrix.

    Raises
    ------
    Exception
        If inputs are not numpy arrays or have mismatched sizes.
    """
    # --- Input Validation ---
    if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray)) is False:
        raise Exception("x, y, z must be numpy arrays")
    if ((x.size == z.size) and (y.size == z.size)) is False:
        raise Exception("x, y, z must have the same cardinality size.")

    # --- Data Normalization ---
    # Scale coordinates to the range [0, 1] to improve numerical stability of the fit.
    x_scale = 1.0 / x.max()
    y_scale = 1.0 / y.max()

    x = x.flatten() * x_scale
    y = y.flatten() * y_scale
    z = z.flatten()
    # --- Construct the Design Matrix (A) ---
    # This matrix relates the polynomial coefficients to the z-values.
    # Each column corresponds to a term in the polynomial (e.g., 1, x, y, xy, x^2, ...).
    A = np.empty((x.size, (x_order + 1) * (y_order + 1)), dtype=np.float64)
    for i, index in enumerate(np.ndindex((x_order + 1, y_order + 1))):
        A[:, i] = np.power(x, index[0]) * np.power(y, index[1])
    # --- Perform Least Squares Fit ---
    # Solve the linear system A*t = z for the coefficient vector t.
    sol, residuals, rank, sing_values = scipy.linalg.lstsq(A, z, cond=rcond)
    # --- Post-processing ---
    # Normalize the residuals by the number of data points.
    if isinstance(residuals, (np.ndarray, np.number)):
        residuals /= float(x.size)
    # Rescale the solution coefficients to match the original, unnormalized coordinates.
    sol = (
        np.power(x_scale, np.arange(x_order + 1))[:, np.newaxis]
        * np.reshape(sol, (x_order + 1, y_order + 1))
        * np.power(y_scale, np.arange(y_order + 1))
    )
    return sol, residuals, rank, sing_values


def getLUT(
    coreg: xr.Dataset, daz_drg: xr.Dataset, slc: xr.Dataset, coeffs: np.ndarray, dec: Sequence[int] = GRID
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Generate a Lookup Table (LUT) from polynomial coefficients.

    This function reconstructs a 2D surface on a
    coarse grid using a set of polynomial coefficients. It calculates the
    appropriate coordinates based on the coregistered geometry and then
    evaluates the 2D polynomial to populate the LUT.

    Parameters
    ----------
    coreg : xr.Dataset
        The coregistered dataset defining the output grid.
    daz_drg : xr.Dataset
        The dataset containing displacement fields (`daz` and `drg`).
    slc : xr.Dataset
        The original SAR image dataset, used for metadata.
    coeffs : np.ndarray
        A 2D numpy array of polynomial coefficients.
    dec : Sequence[int], optional
        Decimation factors for the grid (not directly used, assumed in `daz_drg` grid).
        Defaults to GRID.

    Returns
    -------
    lut : xr.DataArray
        The generated lookup table.
    az_ref : xr.DataArray
        The reference azimuth time coordinates of the LUT.
    rg_ref : xr.DataArray
        The reference slant-range time coordinates of the LUT.
    """
    # Select the subset of the coreg grid that matches the coarse daz_drg grid.
    crg = coreg.sel(azimuth_time=daz_drg.azimuth_time.values, slant_range_time=daz_drg.slant_range_time.values)

    # Extract the reference coordinate grids and the displacement values.
    az_ref, rg_ref = crg.azimuth_time, crg.slant_range_time
    daz, drg = daz_drg.daz.squeeze(), daz_drg.drg.squeeze()

    # Mask out any invalid (NaN) displacement values.
    mask = np.isfinite(daz) & np.isfinite(drg)
    daz, drg = daz.where(mask, 0), drg.where(mask, 0)

    # Compute the grid coordinates of the secondary image by applying the displacements.
    az_sec = (az_ref + daz).transpose("azimuth_time", "slant_range_time")
    rg_sec = (rg_ref + drg).transpose("azimuth_time", "slant_range_time")

    # --- Prepare for Coordinate Transformation ---
    # Extract metadata from the secondary (original) SLC image.
    annot = slc.slc.attrs[list(slc.slc.attrs.keys())[0]]
    # Get the reference point and scaling factors to convert from time to meters/seconds.
    try:
        rg0 = slc.isel(slant_range_time=annot["ImageData"]["SCPPixel"]["Row"]).slant_range_time.values
        az0 = slc.isel(azimuth_time=annot["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values
    except Exception as e:
        logger.warning(f"type is: {e.__class__.__name__}, fix for tests")
        rg0 = slc.slant_range_time[0].values
        az0 = slc.azimuth_time[0].values
    # Range scaling factor (converts time to meters).
    dt_rg = speed_of_light / 2
    # Azimuth scaling factor (converts time difference to seconds).
    dt_az = 1.0 / (1e9 * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1])

    # --- Compute LUT ---
    # Convert the secondary grid coordinates to a relative system (in meters and seconds).
    rg_m = (rg_sec - rg0) * dt_rg
    az_m = (az_sec - az0).astype(float) * dt_az

    # Reconstruct the LUT values by evaluating the 2D polynomial at the relative coordinates.
    lut = xr.DataArray(
        np.polynomial.polynomial.polyval2d(rg_m.values, az_m.values, coeffs), dims=["slant_range_time", "azimuth_time"]
    )

    return lut, az_ref, rg_ref


def updateLut(slcs: xr.DataTree, daz_drg: xr.Dataset, path_to_coeff: str, out_name: str, **kwargs) -> xr.Dataset:
    """
    Update or create a Lookup Table (LUT) for a specific parameter.

    This function orchestrates the creation of a LUT (e.g., for Doppler
    frequency, sigma0, etc.) for each acquisition in a stack. It retrieves
    the necessary polynomial coefficients from the metadata, generates the
    LUT for the coregistered grid using `getLUT`, and stores it in a new
    xarray.Dataset with appropriate coordinates and attributes.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing original and coregistered data.
    daz_drg : xr.Dataset
        The dataset with azimuth and range displacement grids.
    path_to_coeff : str
        A string path (e.g., "a.b.c") to locate the
        polynomial coefficients within the metadata dictionary.
    out_name : str
        The name for the output data variable in the new Dataset (e.g., "doppler").
    **kwargs
        Additional parameters, including `GRID` for decimation.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the newly computed LUT.
    """
    grid = kwargs.get("GRID", GRID)
    # Extract metadata from the coregistered stack.
    annot = slcs.coreg.measurements.cstack.attrs
    pols = slcs.coreg.measurements.polarization.values
    # --- Initialize Output LUT DataArray ---
    # Handle cases with and without a polarization dimension.
    if out_name in ["sigma0", "gamma0", "noise"]:
        xLUT = xr.DataArray(
            np.zeros((pols.size, slcs.coreg.measurements.acquisition_date.size, *grid)),
            dims=["polarization", "acquisition_date", "slant_range_time", "azimuth_time"],
            coords={"acquisition_date": slcs.coreg.measurements.acquisition_date.values, "polarization": pols},
        )  # type: Union[xr.Dataset,xr.DataArray]
    else:
        xLUT = xr.DataArray(
            np.zeros((slcs.coreg.measurements.acquisition_date.size, *grid)),
            dims=["acquisition_date", "slant_range_time", "azimuth_time"],
            coords={"acquisition_date": slcs.coreg.measurements.acquisition_date.values},
        )
    # --- Main Loop: Compute LUT for each acquisition ---
    for i, dd in enumerate(slcs.coreg.measurements.acquisition_date.values):
        # Handle per-polarization coefficients if necessary.
        if "polarization" in xLUT.dims:
            for p, pol in enumerate(xLUT.polarization.values):
                annot = slcs.orig[dd].measurements.slc.attrs[pol]
                coeffs = np.array(reduce(dict.get, path_to_coeff.split("."), annot))
                # compute Doppler on the new reference system
                xLUT[p, i, :, :], az_ref, rg_ref = getLUT(
                    slcs.coreg.measurements.sel(acquisition_date=dd),
                    daz_drg.sel(acquisition_date=dd),
                    slcs.orig[dd].measurements.to_dataset(),
                    coeffs,
                    grid,
                )
        else:
            annot = slcs.orig[dd].measurements.slc.attrs
            annot = annot[list(annot.keys())[0]]
            coeffs = np.array(reduce(dict.get, path_to_coeff.split("."), annot))
            # compute Doppler on the new reference system
            xLUT[i, :, :], az_ref, rg_ref = getLUT(
                slcs.coreg.measurements.sel(acquisition_date=dd),
                daz_drg.sel(acquisition_date=dd),
                slcs.orig[dd].measurements.to_dataset(),
                coeffs,
                grid,
            )
    # --- Set Coordinates and Attributes ---
    # Extract reference point and scaling factors for the coregistered grid.
    try:
        rg0 = slcs.coreg.measurements.isel(
            slant_range_time=annot["ImageData"]["SCPPixel"]["Row"]
        ).slant_range_time.values
        az0 = slcs.coreg.measurements.isel(azimuth_time=annot["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values
    except Exception as e:
        logger.warning(f"type is: {e.__class__.__name__}, fix for tests")
        rg0 = slcs.coreg.measurements.slant_range_time[0].values
        az0 = slcs.coreg.measurements.azimuth_time[0].values
    # Range scaling (meters).
    drg = speed_of_light / 2
    # Azimuth scaling (seconds).
    daz = 1.0 / (1e9 * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1])

    # Assign the correct coordinates to the LUT DataArray.
    xLUT = xLUT.assign_coords({"slant_range_time": rg_ref.values, "azimuth_time": az_ref.values})

    # Store the reference point, scaling factors, and polynomial degree as attributes.
    xLUT = xLUT.assign_attrs(
        {
            "reference": {"azimuth_time": az0.astype(str), "slant_range_time": float(rg0)},
            "scale": {"azimuth_time": daz, "slant_range_time": drg},
            "deg": {"azimuth_time": coeffs.shape[1], "slant_range_time": coeffs.shape[0]},
        }
    )

    # Convert the final DataArray to a Dataset.
    xLUT = xLUT.to_dataset(name=out_name)
    return xLUT


def getCoeffFromLut(xLut: xr.DataArray) -> np.ndarray:
    """
    Fit and extract polynomial coefficients from a Lookup Table (LUT).

    This function performs the inverse operation of `getLUT`. It takes a LUT
    defined on a coarse grid and fits a 2D polynomial to its values to
    recover the underlying polynomial model coefficients.

    Parameters
    ----------
    xLut : xr.DataArray
        The input lookup table as an xarray.DataArray, which must
        contain `reference`, `scale`, and `deg` attributes.

    Returns
    -------
    np.ndarray
        A 2D numpy array of the fitted polynomial coefficients.
    """
    # Recreate the coordinate grids (azimuth and range) for the LUT.
    az, rg = xr.broadcast(xLut.azimuth_time, xLut.slant_range_time)
    # Convert the absolute time coordinates to the relative, scaled system used for fitting.
    rg = (rg - xLut.attrs["reference"]["slant_range_time"]) * xLut.attrs["scale"]["slant_range_time"]
    az = (az - np.datetime64(xLut.attrs["reference"]["azimuth_time"])).astype(float) * xLut.attrs["scale"][
        "azimuth_time"
    ]

    # Perform a 2D polynomial fit to the LUT data to find the coefficients.
    coeff, _, _, _ = poly_fit_2d(
        rg.values, az.values, xLut.values, x_order=xLut.deg["slant_range_time"], y_order=xLut.deg["azimuth_time"]
    )

    return coeff


def do_deramp(slc: xr.DataArray, phase_demod: xr.DataArray, scp_idx: tuple[int, int]) -> xr.DataArray:
    """
    Perform deramping of SLC data using a phase demodulation LUT.

    Deramping removes a known phase ramp from the complex SLC data.
    This function first recovers the polynomial model of the phase from
    the `phase_demod` LUT, reconstructs the full-resolution phase screen,
    and then applies it to the SLC image.

    Parameters
    ----------
    slc : xr.DataArray
        The input complex SLC data as an xarray.DataArray.
    phase_demod : xr.DataArray
        The lookup table for phase demodulation, containing
        the necessary attributes for coefficient fitting.
    scp_idx : tuple[int, int]
        The column and row indices of the scene center pixel (SCP) in the SLC image, used as the default reference point for coordinate transformation.

    Returns
    -------
    xr.DataArray
        The deramped (phase-corrected) SLC data.
    """
    # Recover the polynomial coefficients that describe the Doppler frequency.
    coeff = getCoeffFromLut(phase_demod)

    # Integrate the Doppler polynomial to get the phase polynomial.
    coeff_ = (2 * np.pi) * np.polynomial.polynomial.polyint(coeff, axis=1)

    # Reconstruct the original, full-resolution demodulation phase screen.
    rg_out, az_out = extract_grid(slc, scp_idx, phase_demod.reference, phase_demod.scale)
    phramp = np.polynomial.polynomial.polygrid2d(rg_out.values.ravel(), az_out.values.ravel(), coeff_).T

    # Apply the phase ramp to the SLC data by complex multiplication.
    return slc * np.exp(-1j * phramp)


def extract_rg_az(
    slc: xr.DataArray, simulation: xr.Dataset | xr.DataArray, dtaz: float, dtrg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract range and azimuth coordinates from a simulation dataset in the SLC's pixel reference system.

    Parameters
    ----------
    slc : xr.DataArray
        The input SLC DataArray, used for its metadata and
        as the reference for the pixel coordinate system origin
        (minimum azimuth and range times).
    simulation : xr.Dataset
        The simulation dataset containing `slant_range_time` and
        `azimuth_time` coordinates to be converted.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The range coordinates in relative pixels (`rg`).
        - The azimuth coordinates in relative pixels (`az`).
    """
    # Transform the simulation's absolute time coordinates into the SLC's relative pixel coordinates.
    rg = ((simulation.slant_range_time - slc.slant_range_time.min()) / dtrg).values
    az = ((simulation.azimuth_time - slc.azimuth_time.min()).astype(float) / dtaz).values
    # --- Type Checking ---
    if not isinstance(rg, np.ndarray):
        raise Exception(f"Expected np.ndarray, got {type(rg)}")
    if not isinstance(az, np.ndarray):
        raise Exception(f"Expected np.ndarray, got {type(az)}")
    return rg, az


def extract_rg0_az0(slc: xr.DataArray, phase_demod: xr.DataArray) -> tuple[float, float, float, float]:
    """
    Extract range and azimuth coordinates from a simulation dataset in the SLC's pixel reference system.

    Parameters
    ----------
    slc : xr.DataArray
        The input SLC DataArray, used for its metadata and
        as the reference for the pixel coordinate system origin
        (minimum azimuth and range times).
    simulation : xr.Dataset
        The simulation dataset containing `slant_range_time` and
        `azimuth_time` coordinates to be converted.
    dtaz : float
        The azimuth time spacing (in seconds) used for scaling the azimuth coordinates.
    dtrg : float
        The slant range time spacing (in seconds) used for scaling the range coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays:
        - The range coordinates in relative pixels (`rg`).
        - The azimuth coordinates in relative pixels (`az`).
    """
    # Extract metadata from the SLC attributes.
    annot = slc.attrs
    annot = annot[list(annot)[0]]
    # Calculate the time interval per pixel in azimuth and range.
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light

    rg0 = ((float(phase_demod.reference["slant_range_time"]) - slc.slant_range_time.min()) / dtrg).values
    az0 = ((np.datetime64(phase_demod.reference["azimuth_time"]) - slc.azimuth_time.min()).astype(float) / dtaz).values
    drg, daz = float(phase_demod.scale["slant_range_time"]), float(phase_demod.scale["azimuth_time"])
    return rg0, az0, drg, daz


def warp_gslc(
    image: xr.Dataset,
    simulation: xr.Dataset,
    kernel: dict,
    cstack: xr.DataArray,
    phase_model: xr.DataArray,
    phase_demod: xr.DataArray,
    anf: xr.DataArray,
    layover: xr.DataArray,
    shadow: xr.DataArray,
    do_reramp: bool,
    gamma_flattening: bool,
    dtaz: float,
    dtrg: float,
    scp_idx: tuple[int, int],
    eps: float = 1e-3,
    margin: int = 5,
    chunk=None,
    **kwargs,
) -> xr.Dataset:
    """
    Geocodes a chunk of coregistered SLC (CSLC) radar data to a geographic grid.

    This function maps radar coordinates (slant range, azimuth) to a uniform
    ground grid using sinc interpolation. It handles phase deramping, optional
    gamma flattening, and phase reramping for InSAR processing.

    Parameters
    ----------
    image : xarray.Dataset
        Target geographic grid and output buffers.
    simulation : xarray.Dataset
        Radar geometry simulation data mapping the ground grid.
    cslc : xarray.Dataset
        Coregistered SLC measurements and phase conditions.
    kernel : dict
        Sinc interpolation kernel parameters (e.g., {"kernel_shape": (L, L)}).
    margin : int or float
        Margin applied to the bounding box to account for kernel size.
    data_path_static : str
        Path to static radar grid datasets (e.g., layover/shadow masks).
    data_path_tmp : str
        Temporary storage path for writing the output Zarr chunks.
    dtaz : float
        The azimuth time spacing (in seconds) used for coordinate transformation.
    dtrg : float
        The slant range time spacing (in seconds) used for coordinate transformation.
    scp_idx : tuple[int, int]
        The column and row indices of the scene center pixel (SCP) in the original SLC image
    eps : float, optional
        Epsilon value to prevent division by zero in gamma flattening. Default is 1e-6.
    gamma_flattening : bool, optional
        Whether to apply radiometric terrain correction. Default is False.
    do_reramp : bool, optional
        Whether to re-apply the phase modulation post-interpolation. Default is True.
    chunk : tuple or str
        Identifier or region for the current data chunk being processed.
    **kwargs : dict
        Additional keyword arguments. Can include `method` (str) to specify
        the interpolation method (default "linear") and storage options.

    Returns
    -------
    xr.Dataset
        A dataset containing the geocoded SLC data and the corresponding phase model.
    """
    # --- Initialization and Data Loading ---
    method = kwargs.get("method", "linear")
    compute_all = chunk["acquisition_date"].start == 0
    acquisition_dates = image.acquisition_date.values
    # Use the first acquisition's simulation as the reference grid.
    simulation = simulation.isel(acquisition_date=0).squeeze()
    # Define the output region for the current chunk.
    shape, margin = (simulation.y.size - 1, simulation.x.size - 1), kernel["kernel_shape"] // 2 + 1

    # --- Coordinate Transformation ---
    spacing_in = np.abs(simulation.rio.resolution(recalc=True))
    spacing_out = image.rio.resolution(recalc=True)

    xmin = image.x.min().values - margin * spacing_in[0]
    xmax = image.x.max().values + margin * spacing_in[0]
    ymin = image.y.min().values - margin * spacing_in[1]
    ymax = image.y.max().values + margin * spacing_in[1]

    xslice = slice(xmin, xmax) if spacing_out[0] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing_out[1] > 0 else slice(ymax, ymin)

    simulation = simulation.sel(x=xslice, y=yslice)
    # Convert the ground-projected simulation coordinates into the radar's pixel coordinate system.
    rg, az = extract_rg_az(cstack, simulation, dtaz, dtrg)

    # Average coordinates to find the center of each pixel.
    simulation["azimuth_time"].data = az
    simulation["slant_range_time"].data = rg
    x_grid, y_grid = np.meshgrid(image.x, image.y)

    # Call the generic dataset interpolation function.
    simul_out = interp_dataset(
        data=simulation, xi=(y_grid, x_grid), common_dims=("y", "x"), method=method, fill_value=0.0
    )

    rg_interp = simul_out.slant_range_time.values.ravel()
    az_interp = simul_out.azimuth_time.values.ravel()

    # Find the indices of points that can be safely interpolated (i.e., not too close to the edge).
    kernel_margin = kernel["kernel_shape"] // 2 + 1
    (ind,) = np.where(
        (rg_interp >= kernel_margin[1])
        & (az_interp >= kernel_margin[0])
        & (rg_interp < (cstack.slant_range_time.size - kernel_margin[1]))
        & (az_interp < (cstack.azimuth_time.size - kernel_margin[0]))
    )

    # --- Main Geocoding Loop ---
    gstack_buffer = np.zeros_like(image.gstack)
    phase_model_buffer = np.zeros_like(image.phase_model)

    if ind.size > 0:
        # Define the subset of radar coordinates needed for interpolation.
        x, y = rg_interp[ind], az_interp[ind]

        # Calculate the bounding box to read from the coregistered data.
        xmin, xmax = np.round(x.min() - kernel_margin[1]).astype(int), np.round(x.max() + kernel_margin[1]).astype(int)
        ymin, ymax = np.round(y.min() - kernel_margin[0]).astype(int), np.round(y.max() + kernel_margin[0]).astype(int)

        # Translate coordinates to be relative to the bounding box origin.
        x, y = x - xmin, y - ymin

        # --- Prepare Data for Interpolation ---
        # Get reference points and scaling factors for phase calculations.
        rg0, az0, drg0, daz0 = extract_rg0_az0(cstack, phase_demod)

        # Select the required portion of the input data.
        cstack = cstack.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        phase_model = phase_model.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        anf = anf.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        layover = layover.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        shadow = shadow.isel(slant_range_time=slice(xmin, xmax), azimuth_time=slice(ymin, ymax))
        # Load static radar grid data (e.g., layover/shadow masks, area normalizer).
        # Prepare for gamma flattening if enabled.
        if gamma_flattening:
            mask_A = anf > eps

        # --- Loop over each acquisition date ---
        for date_idx, acq_date in enumerate(acquisition_dates):
            # Select the phase demodulation LUT for the current date.
            ph_demod = phase_demod.sel(acquisition_date=acq_date)

            # Warp the phase model onto the geocoded grid.
            warped_phase = sinc_warp(phase_model.sel(acquisition_date=acq_date).values, x, y, kernel)
            phase_model_buffer[date_idx].ravel()[ind] = warped_phase

            # --- Pre-calculate Reramp Array ---
            # Calculated once per date on valid points only, keeping variables isolated.
            if do_reramp:
                az_reramp = (simul_out.azimuth_time.values.ravel()[ind] - az0) * (daz0 / dtaz)
                rg_reramp = (simul_out.slant_range_time.values.ravel()[ind] - rg0) * (drg0 / dtrg)

                coeff_phase_demod = 2 * np.pi * np.polynomial.polynomial.polyint(getCoeffFromLut(ph_demod), axis=1)
                phramp = np.polynomial.polynomial.polyval2d(rg_reramp, az_reramp, coeff_phase_demod)
                reramp_multiplier = np.exp(1j * phramp)

            # --- Loop over each polarization ---
            for pol_idx, pol in enumerate(cstack.polarization.values):
                # Apply deramping to the SLC image to correct its phase prior to interpolation.
                slc = do_deramp(cstack.sel(polarization=pol, acquisition_date=acq_date), ph_demod, scp_idx)

                # Apply gamma flattening if enabled.
                if gamma_flattening:
                    sqrt_A = np.sqrt(A)
                    np.divide(slc.values, sqrt_A.values, out=slc.values, where=mask_A.values)
                    # Zero out the invalid pixels
                    slc.values[..., ~mask_A.values] = 0.0

                # Perform the final sinc interpolation from radar to ground grid.
                warped_gslc = sinc_warp(slc.values, x, y, kernel)

                # Apply phase reramping to the 1D valid array if enabled.
                if do_reramp:
                    warped_gslc *= reramp_multiplier

                # Map the processed 1D array back into the 2D spatial buffer.
                gstack_buffer[pol_idx, date_idx].ravel()[ind] = warped_gslc

    # --- Write Output ---
    # Write the resulting GSLC chunk to the temporary Zarr store.
    ds = xr.Dataset(
        data_vars={
            "gstack": (image.gstack.dims, gstack_buffer),
            "phase_model": (image.phase_model.dims, phase_model_buffer),
        }
    )
    return ds


def grid_data_block(
    daz: xr.DataArray,
    drg: xr.DataArray,
    simulation_azimuth_time: xr.DataArray,
    simulation_slant_range_time: xr.DataArray,
    reference_date: str,
    bbox: np.ndarray,
    azimuth_grid_spacing: float,
    range_grid_spacing: float,
    azimuth_time_ref: np.datetime64,
    slant_range_time_ref: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Interpolate simulation data onto the radar grid for a specific block.

    This function takes simulation data defined on a
    sparse grid and interpolates it onto the dense grid of a radar data block.
    This is a key step in computing the `daz` (azimuth shift) and `drg` (range
    shift) values needed for warping.

    Parameters
    ----------
    interpolated_data : xr.Dataset
        The output dataset defining the target radar grid for
        the current block.
    data : xr.Dataset
        The input simulation dataset containing sparse shift values.
    dt : dict
        A dictionary with time steps and reference times for coordinate conversion.
    reference_date : str
        The reference acquisition date for coregistration.
    bbox : np.ndarray
        The bounding box of the radar data block.

    Returns
    -------
    xr.Dataset
        The `interpolated_data` dataset, now filled with the interpolated `daz`
        and `drg` values.
    """
    # If there is more than one acquisition, we compute the shifts.
    if simulation_azimuth_time.acquisition_date.size > 1:
        # Keep only the essential variables for shift calculation.
        # Select the simulation data corresponding to the reference date.
        reference_azimuth_time = simulation_azimuth_time.sel(acquisition_date=reference_date)
        reference_slant_range_time = simulation_slant_range_time.sel(acquisition_date=reference_date)
        # Calculate the shifts by subtracting the reference from all other acquisitions.
        selection_azimuth_time = (
            simulation_azimuth_time.sel(acquisition_date=daz.acquisition_date) - reference_azimuth_time
        )
        selection_slant_range_time = (
            simulation_slant_range_time.sel(acquisition_date=daz.acquisition_date) - reference_slant_range_time
        )
        # --- Prepare for Interpolation ---
        # Convert the sparse reference grid coordinates to a pixel-based system.
        points = np.array(
            [
                (((reference_azimuth_time - azimuth_time_ref).astype(float)) / (azimuth_grid_spacing * 1e9)).values,
                ((reference_slant_range_time - slant_range_time_ref).astype(float) / range_grid_spacing).values,
            ]
        ).T
        # Stack the range and azimuth shift values to be interpolated.
        z = np.vstack([selection_slant_range_time.values, selection_azimuth_time.astype(float).values])
        z = np.transpose(z, (0, 2, 1))
        # Perform the interpolation from the sparse `points` to the dense `bbox` grid.
        m = interpolate(points, z, bbox)

        # Assign the interpolated results to the output dataset.
        drg[:, :, :] = m[: z.shape[0] // 2, :, :]
        daz[:, :, :] = m[z.shape[0] // 2 :, :, :]
    else:
        # If there's only one acquisition, there are no shifts to compute.
        drg[:, :, :] = 0
        daz[:, :, :] = 0

    return daz, drg


def coreg_qlt(slcs: xr.DataTree, daz_drg: xr.Dataset, **kwargs):
    """
    Compute the quality assessment of the coregistration.

    This function estimates the accuracy of the coregistration by modeling
    the expected geometric error. The error is based on the perpendicular
    baseline between acquisitions, the topographic height uncertainty, and
    the precision of the orbit data used.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing original and coregistered data.
    daz_drg : xr.Dataset
        The dataset with the computed azimuth and range shifts.
    **kwargs
        Additional parameters like `GRID` and orbit error variances.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing the estimated coregistration accuracy.
    """
    # --- Initialization ---
    grid = kwargs.get("GRID", GRID)
    sigma_orb_2 = kwargs.get("SIGMA_ORB_2", SIGMA_ORB_2)
    # Assumed topographic height uncertainty squared (in meters^2).
    sigma_h_2 = 1.0

    # Get the reference acquisition date.
    crg = slcs.coreg.measurements
    reference_date = crg.coregistration_reference_date

    # Select the coarse grid from the coregistered data.
    crg = crg.sel(azimuth_time=daz_drg.azimuth_time.values, slant_range_time=daz_drg.slant_range_time.values)

    # --- Prepare Reference Geometry ---
    # Get metadata and incidence angle for the reference scene.
    annot = crg.cstack.attrs
    annot = annot[list(annot.keys())[0]]
    teta = annot["SCPCOA"]["IncidenceAng"] * np.pi / 180.0
    # Get the azimuth and range coordinates for the reference grid.
    az0_v, rg0_v = [crg.cstack[e] for e in ["azimuth_time", "slant_range_time"]]
    # Stack the 2D grid coordinates into 1D arrays for position calculation.
    az0 = (az0_v + (rg0_v * 0).astype("timedelta64[ns]")).stack(point=["azimuth_time", "slant_range_time"])
    rg0 = (az0_v.astype("float64") * 0 + rg0_v).stack(point=["azimuth_time", "slant_range_time"])
    # Compute the 3D position vectors for each point on the reference grid.
    t0 = np.datetime64(annot["Timeline"]["CollectStart"], "ns")
    arp_coeffs = {e: get_coefficients(f"Position.ARPPoly.{e}.Coefs", annot) for e in ["X", "Y", "Z"]}
    if any(coeff is None for coeff in arp_coeffs.values()):
        raise ValueError("Missing ARP polynomial coefficients in metadata.")
    position0 = get_position(az0, t0, cast(dict[str, np.ndarray], arp_coeffs)).drop_vars("azimuth_time")

    # Initialize the output DataArray for coregistration quality.
    crg_qlt = xr.DataArray(
        np.zeros((crg.acquisition_date.size, *grid)), dims=["acquisition_date", "azimuth_time", "slant_range_time"]
    )
    # --- Main Loop: Compute Quality for each Acquisition ---
    for i, d in enumerate(crg.acquisition_date):
        # Skip the reference date itself, as its error is zero by definition.
        if d != reference_date:
            # Determine the type of orbit data used for this acquisition.
            if ("conditions" in slcs.orig[str(d.values)]) and ("orbit" in slcs.orig[str(d.values)].conditions):
                orb_type = slcs.orig[str(d.values)].conditions.orbit.attrs["orbit_type"]
            else:
                orb_type = "AUX_ANNOTATED"

            # Get metadata for the secondary acquisition.
            annot = slcs.orig[str(d.values)].measurements.slc.attrs
            annot = annot[list(annot.keys())[0]]
            # Calculate the coordinates of the secondary image points corresponding to the reference grid.
            az = az0 + daz_drg.daz.sel(acquisition_date=d).stack(point=["azimuth_time", "slant_range_time"])
            rg = rg0 + daz_drg.drg.sel(acquisition_date=d).stack(point=["azimuth_time", "slant_range_time"])
            # --- Compute Geometric Error (Sigma) ---
            # Calculate the squared baseline (distance between satellite positions).
            t0 = np.datetime64(annot["Timeline"]["CollectStart"], "ns")
            arp_coeffs = {e: get_coefficients(f"Position.ARPPoly.{e}.Coefs", annot) for e in ["X", "Y", "Z"]}
            if any(coeff is None for coeff in arp_coeffs.values()):
                raise ValueError("Missing ARP polynomial coefficients in metadata.", "ns")
            sigma = (
                (get_position(az, t0, cast(dict[str, np.ndarray], arp_coeffs)).drop_vars("azimuth_time") - position0)
                ** 2
            ).sum(dim="axis")
            # Combine the baseline, height uncertainty, and orbit error to estimate the total coregistration error.
            sigma = np.sqrt(
                sigma
                / (rg.drop_vars(rg.coords).rename(point="azimuth_time") * speed_of_light / 2.0 * np.sin(teta)) ** 2
                * sigma_h_2
                + sigma_orb_2[str(orb_type)]
            )
            # Unstack the 1D result back into a 2D grid and assign it to the output.
            crg_qlt[i, :, :] = (
                sigma.rename(azimuth_time="point")
                .assign_coords(az0.drop_vars("acquisition_date", errors="ignore").coords)
                .unstack()
            )

    # Convert the final DataArray to a Dataset.
    crg_qlt = crg_qlt.to_dataset(name="crg_accuracy")

    return crg_qlt
