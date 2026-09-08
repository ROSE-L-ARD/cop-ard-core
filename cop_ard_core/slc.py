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
from __future__ import annotations

from typing import cast
from xml.etree.ElementTree import Element

import numpy as np
import shapely
import shapely.wkt
import xarray as xr

from cop_ard_core.logger import logger
from numba import njit

def merge_dict(res: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.
    """
    merged_dict = {}
    # Iterate through the list of dictionaries.
    for d in res:
        # Update the merged dictionary with the items from the current dictionary.
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


def apply_invalid_mask(
    slc: xr.DataArray, valid_y: np.ndarray, valid_x: np.ndarray, first_idx_y: int = 0, first_idx_x: int = 0
) -> xr.DataArray:
    """
    Apply a mask to the SLC data to zero out invalid regions.
    """
    # Adjust coordinates to be relative to the origin.
    valid_y = valid_y - first_idx_y
    valid_x = valid_x - first_idx_x

    # Fill the top and bottom invalid slices with zeros.
    slc[:, :, np.min((valid_y[1], valid_y[2])) :, :] = slc[:, :, : np.max((0, valid_y[0], valid_y[3])), :] = 0j

    # Fill the left and right invalid slices with zeros.
    slc[:, :, :, np.min((valid_x[2], valid_x[3])) :] = slc[:, :, :, : np.max((0, valid_x[0], valid_x[1]))] = 0j

    return slc


def amplitude_calibration(slc: xr.DataArray, calibration_polynomial_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Apply radiometric calibration to raw DN.

    This function uses a polynomial to convert the
    complex pixel values into a calibrated radiometric quantity, such as radar
    backscatter coefficient (Sigma Nought).

    Args:
        slc: The input SAR image as an xarray DataArray.
        calibration_polynomial_coeffs: The polynomial coefficients for the calibration.

    Returns:
        The calibrated SAR image.
    """
    # Check if radiometric information is available in the metadata.
    if calibration_polynomial_coeffs is None:
        logger.warning("No radiometric calibration information provided: image will be imported as is")
        return slc
    # Evaluate the 2D polynomial on the image grid to get the calibration factor.
    calib_factor = np.sqrt(
        np.polynomial.polynomial.polygrid2d(
            slc.slant_range_m.values, slc.azimuth_m.values, calibration_polynomial_coeffs
        )
    ).T

    # Apply the calibration factor to the complex data.
    slc *= xr.DataArray(
        calib_factor,
        dims=["azimuth_time", "slant_range_time"],
    )

    return slc


def noise_removal(slc: xr.DataArray, noise_polynomial_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Remove thermal noise from the SAR image.

    This function subtracts the estimated thermal noise power from the image
    intensity. The noise level is modeled as a 2D polynomial, with coefficients
    provided in the image metadata.

    Args:
        slc: The input SAR image as an xarray DataArray.
        noise_polynomial_coeffs: The polynomial coefficients for the noise level.

    Returns:
        The noise-corrected SAR image.
    """
    # Check if noise information is available.
    if noise_polynomial_coeffs is None:
        logger.warning("No radiometric noise level information provided: image will be imported as is")
        return slc
    # Extract the polynomial coefficients for the noise level.

    # Evaluate the polynomial to get the noise power across the image.
    noise_power = xr.DataArray(
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, noise_polynomial_coeffs).T,
        dims=["azimuth_time", "slant_range_time"],
    )

    # The calibration polynomial is typically in dB, so convert to linear power.
    noise_power = 10 ** (noise_power / 10)

    # Subtract noise power from the image intensity.
    intensity = cast(xr.DataArray, np.abs(slc) ** 2)
    intensity -= noise_power
    # Set any negative intensity values to a small positive number to avoid errors.
    intensity = intensity.where(intensity >= 0, 1e-9)
    # Reconstruct the complex image from the noise-corrected intensity and original phase.
    slc = np.sqrt(intensity) * np.exp(1j * np.angle(slc))
    return slc


@njit(cache=True)
def deramp_numba(slc, slant_range, azimuth, coeff):

    npol, ndates, n_azimuth, n_slant_range = slc.shape
    deg_slant_range, deg_azimuth = coeff.shape
    out = np.empty_like(slc)

    for i in range(n_slant_range):

        ri = slant_range[i]

        for j in range(n_azimuth):

            aj = azimuth[j]

            phase = 0.0

            for k in range(deg_slant_range - 1, -1, -1):

                row = coeff[k, deg_azimuth - 1]

                for l in range(deg_azimuth - 2, -1, -1):
                    row = row * aj + coeff[k, l]

                phase = phase * ri + row

            c = np.cos(phase)
            s = np.sin(phase)

            rot = c - 1j * s

            for id in range(ndates):
                for pol in range(npol):
                    out[pol, id, j, i] = slc[pol, id, j, i] * rot

    return out


def image_deramp(slc: xr.DataArray, polynomial_dkcoa_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Apply a phase deramping correction to the SAR image.

    This function removes a phase ramp caused by the Doppler frequency variation
    across the scene. The ramp is modeled as the integral of the Doppler Centroid
    frequency, which is itself a 2D polynomial provided in the metadata.

    Args:
        slc: The input SAR image as an xarray DataArray.
        polynomial_dkcoa_coeffs: The polynomial coefficients for the Doppler Centroid frequency.

    Returns:
        The deramped SAR image.
    """
    if polynomial_dkcoa_coeffs is None:
        logger.warning("No Doppler Centroid information provided: skipping deramping step")
        return slc

    # Extract the Doppler Centroid polynomial coefficients and integrate them to get the phase polynomial.
    coeff = ((2 * np.pi) * np.polynomial.polynomial.polyint(polynomial_dkcoa_coeffs, axis=1)).astype(np.float32)

    # Evaluate the phase polynomial on the image grid to get the phase ramp.
    slc.data = deramp_numba(
        slc.values,
        slc.slant_range_m.values.astype(np.float32),
        slc.azimuth_m.values.astype(np.float32),
        coeff,
    )
    
    return slc


def intersect(polygon1: str | list[list[float]], polygon2: str | list[list[float]]):
    """
    Return True if the two input polygons intersect one another.
    Polygons can be passed either as a list of coordinates or as a WKT format string.

    Parameters
    ----------
    polygon1 : str or list
        The first polygon, either as a WKT string or a list of coordinates.
    polygon2 : str or list
        The second polygon, either as a WKT string or a list of coordinates.

    Returns
    -------
    bool
        True if the polygons intersect, False otherwise.
    """
    # Load the first polygon from WKT or a list of coordinates.
    try:
        poly1 = shapely.wkt.loads(polygon1)
    except TypeError:
        poly1 = shapely.geometry.Polygon(polygon1)

    # Load the second polygon.
    try:
        poly2 = shapely.wkt.loads(polygon2)
    except TypeError:
        poly2 = shapely.geometry.Polygon(polygon2)

    # Return the result of the intersection test.
    return poly2.intersects(poly1)


def get_relative_orbit_number(sentinel1_manifest: Element, ns: Element[str] | dict):
    """
    Extracts the relative orbit number from the Sentinel-1 manifest file.

    Parameters:
        in_path (str): The path to the directory containing the 'manifest.safe' file.

    Returns:
        int: The relative orbit number, or None if extraction fails.
    """
    relative_orbit_number = int(sentinel1_manifest.findall(".//safe:relativeOrbitNumber", ns)[0].text)  # type: ignore
    return relative_orbit_number
