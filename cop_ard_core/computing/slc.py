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
from xml.etree.ElementTree import Element
import json
import os
from collections import defaultdict
from typing import Any, Dict, Sequence, Union, cast

import dateutil.parser
import numpy as np
import sarpy.io
import xarray as xr
from scipy.constants import speed_of_light

import cop_ard_core.readers  # register sarpy readers
from cop_ard_core.computing.io import (
    Path,
    path_to_dict,
)
from cop_ard_core.computing.logger import logger, timeit

cop_ard_core.readers.__file__


def get_coefficients(path: str, annot: dict) -> np.ndarray | None:
    """
    Utility function to extract polynomial coefficients from the annotation dictionary.

    Args:
        annot: The annotation dictionary containing the metadata.
        path: The dot-separated path to the desired coefficients (e.g., "Radiometric.SigmaZeroSFPoly.Coefs").
    Returns:
        A numpy array of coefficients if found, otherwise None.
    """
    try:
        coeffs = path_to_dict(annot, path)
        if coeffs is not None:
            return np.array(coeffs)
    except KeyError:
        return None
    return None


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


class SentinelSarpyObj(object):
    """
    A wrapper class to handle Sentinel-1 metadata extraction using Sarpy,
    especially for data stored on S3 where partial file reads are needed.
    """

    def __init__(self, in_path, storage_options={}):
        # Check if the input is already a DataTree.
        self._is_dt = isinstance(in_path, xr.DataTree)
        if not self._is_dt:
            # For file paths, set up original path and a temporary working directory.
            self._input = Path(in_path, storage_options=storage_options)
        else:
            # For DataTree input, use it directly.
            self._input = in_path
        # Import the metadata upon initialization.
        self._import_metadata()

    def _import_metadata(self):
        """
        Import metadata from the Sentinel-1 product.
        """
        import sarpy.io.complex.sentinel

        if self._is_dt:
            # If input is a DataTree, Sarpy can read its metadata directly.
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(self._input)  # ty:ignore[invalid-argument-type]
        else:
            # For file paths (especially S3), copy essential metadata files locally.
            for j in ["manifest.safe", "annotation"]:
                self._input.append(j).ensure_local()
            # Open the local manifest file with Sarpy.
            manifest_path = self._input.append("manifest.safe").replace_dir(self._input._temp_dir.name)._path  # ty:ignore[unresolved-attribute]
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(manifest_path)

        # Extract the collection of SICD (Sensor Independent Complex Data) metadata objects.
        try:
            self.sicd_meta = [j for k, v in self._sentinel_details.get_sicd_collection() for j in v]  # ty:ignore[not-iterable]
        except Exception as e:
            logger.info(f"{e.__class__.__name__}: No bursts have been found.")
            self.sicd_meta = [v for k, v in self._sentinel_details.get_sicd_collection()]


def apply_invalid_mask(slc: xr.DataArray, valid_y: np.ndarray, valid_x: np.ndarray, first_idx_y: int = 0, first_idx_x: int = 0) -> xr.DataArray:
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
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, calibration_polynomial_coeffs)
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
    coeff = (2 * np.pi) * np.polynomial.polynomial.polyint(polynomial_dkcoa_coeffs, axis=1)

    # Evaluate the phase polynomial on the image grid to get the phase ramp.
    phderamp = xr.DataArray(
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, coeff).T,
        dims=["azimuth_time", "slant_range_time"],
    )

    # Apply the deramping by multiplying with the complex conjugate of the phase ramp.
    return slc * np.exp(-1j * phderamp)


def map_index_to_group(dt, swath, burstid, pol):
    """
    Find the corresponding group in a DataTree based on swath, burst ID, and polarization.
    """
    # Iterate through the DataTree nodes.
    for k, v in dt.items():
        # Check if the node's metadata matches the search criteria.
        if swath is not None and burstid is not None:
            cond = (
                (
                    v.other_metadata["general_annotation"]["downlink_information_list"][0]["swath"].upper()
                    == swath.upper()
                )
                and (v.other_metadata["burst_id"] == str(burstid))
                and (v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol)
            )
        else:
            cond = v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol

        if cond:
            return k
    # Raise an error if no match is found.
    raise Exception(f"No correspondence found for swath={swath} burst_id={burstid} pol={pol}")


def get_az_rg_times(annot):
    """
    Compute azimuth and slant range time coordinate arrays from SICD metadata.
    """
    import pandas as pd

    try:
        # Access metadata fields directly from the Sarpy object.
        num_cols = annot.ImageData.FullImage.NumCols
        num_rows = annot.ImageData.FullImage.NumRows
        scp_col = annot.ImageData.SCPPixel.Col
        scp_row = annot.ImageData.SCPPixel.Row
        time_ca_poly = annot.RMA.INCA.TimeCAPoly.Coefs
        collect_start = annot.Timeline.CollectStart
        col_ss = annot.Grid.Col.SS
        row_ss = annot.Grid.Row.SS
        r_ca_scp = annot.RMA.INCA.R_CA_SCP
    except AttributeError:
        # Fallback to dictionary-based access if the input is a dict.
        num_cols = path_to_dict(annot, "ImageData.FullImage.NumCols")
        num_rows = path_to_dict(annot, "ImageData.FullImage.NumRows")
        scp_col = path_to_dict(annot, "ImageData.SCPPixel.Col")
        scp_row = path_to_dict(annot, "ImageData.SCPPixel.Row")
        time_ca_poly = path_to_dict(annot, "RMA.INCA.TimeCAPoly.Coefs")
        raw_time = path_to_dict(annot, "Timeline.CollectStart")
        collect_start = pd.to_datetime(raw_time, utc=True).tz_localize(None).to_numpy()  # ty:ignore[no-matching-overload]
        col_ss = path_to_dict(annot, "Grid.Col.SS")
        row_ss = path_to_dict(annot, "Grid.Row.SS")
        r_ca_scp = path_to_dict(annot, "RMA.INCA.R_CA_SCP")
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_time, slant_range_time = None, None
        raise
    finally:
        # Calculate azimuth time based on the TimeCAPoly polynomial.
        if time_ca_poly is None:
            raise ValueError
        dtaz = time_ca_poly[1] * col_ss
        azimuth_time_float = (np.arange(cast(int, num_cols), dtype="float64") - scp_col) * dtaz + time_ca_poly[0]
        azimuth_time = (np.round(azimuth_time_float * 1e9)).astype("timedelta64[ns]") + collect_start
        # Calculate slant range time based on the distance from the scene center.
        slant_range_time = 2 * ((np.arange(cast(int, num_rows)) - cast(int, scp_row)) * row_ss + r_ca_scp) / speed_of_light
    return azimuth_time, slant_range_time


def get_az_rg_meters(annot):
    """
    Compute azimuth and slant range distance arrays (in meters) from SICD metadata.
    """

    try:
        # Access metadata fields directly from the Sarpy object.
        daz = annot.Grid.Col.SS
        azimuth_m = (np.arange(annot.ImageData.FullImage.NumCols) - annot.ImageData.SCPPixel.Col) * daz
        slant_range_m = (
            np.arange(annot.ImageData.FullImage.NumRows) - annot.ImageData.SCPPixel.Row
        ) * annot.Grid.Row.SS
    except AttributeError:
        # Fallback to dictionary-based access.
        daz = path_to_dict(annot, "Grid.Col.SS")
        drg = path_to_dict(annot, "Grid.Row.SS")
        num_cols = cast(int, path_to_dict(annot, "ImageData.FullImage.NumCols"))
        scp_col = cast(int, path_to_dict(annot, "ImageData.SCPPixel.Col"))
        num_rows = cast(int, path_to_dict(annot, "ImageData.FullImage.NumRows"))
        scp_row = cast(int, path_to_dict(annot, "ImageData.SCPPixel.Row"))
        azimuth_m = (np.arange(num_cols) - scp_col) * daz
        slant_range_m = (np.arange(num_rows) - scp_row) * drg
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_m, slant_range_m = None, None
        raise
    return azimuth_m, slant_range_m


def extract_orbits(
    slcs: xr.DataTree,
    acquisition_date: list[str] | None = None,
    multiburst: bool = False,
    degrees: int = 11,
) -> xr.DataArray:
    """
    Extract and consolidate satellite orbit polynomial coefficients from metadata.
    """
    orbits_datatree = xr.DataTree()
    orbits, reference_times = {}, {}

    # --- Collect Orbit Data ---
    # Iterate through all scenes (bursts or single images) to collect orbit polynomials.
    for burst_id, i in (slcs.bursts if multiburst else {"data": slcs}).items():
        for data_id, j in i.orig.items():
            annot = j.measurements.slc.attrs[list(j.measurements.slc.attrs.keys())[0]]
            orbits.update({data_id: annot["Position"]["ARPPoly"]})
            reference_times.update(
                {data_id: annot["Timeline"]["CollectStart"]},
            )
            # Prioritize pre-computed orbit data if available.
            if "orbit" in j.conditions:
                orbits_datatree[data_id] = j.conditions.orbit.coeffs.sel(acquisition_date=data_id)

    # Filter by specific acquisition dates if provided.
    if acquisition_date is not None:
        orbits = {k: v for k, v in orbits.items() if k in acquisition_date}
        reference_times = {k: v for k, v in reference_times.items() if k in acquisition_date}

    # --- Consolidate Coefficients ---
    # Initialize an array to hold the coefficients.
    coeffs = np.zeros((len(orbits), degrees, 3))
    for counter, (k, v) in enumerate(orbits.items()):
        # Use pre-computed orbit data if available.
        if k in orbits_datatree:
            data = orbits_datatree[k].values
        # Otherwise, extract from the raw metadata.
        else:
            data = np.array([v[e]["Coefs"] for e in ["X", "Y", "Z"]]).T
        if not isinstance(data, np.ndarray):
            raise Exception(f"Expected np.ndarray, got {type(data)}")
        shape = data.shape
        # Place the coefficients into the consolidated array.
        coeffs[counter][: shape[0], : shape[1]] += data

    # --- Create Final DataArray ---
    orbits = xr.DataArray(
        data=coeffs,
        dims=["acquisition_date", "degree", "axis"],
        coords={
            "acquisition_date": list(orbits),
            "degree": np.arange(0, degrees, 1, dtype=int),
            "axis": np.arange(0, 3, 1, dtype=int),
        },
    ).assign_attrs({"reference_times": reference_times})

    return orbits


def extract_acquisition_dates(
    slcs: xr.DataTree,
    multiburst: bool = False,
) -> list[str]:
    """
    Extract a unique, sorted list of acquisition dates from the DataTree.
    """
    acquisition_dates = set()

    # Iterate through all original data nodes.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for j in i.orig.values():
            # Get the acquisition date from the coordinates.
            k = j.measurements.acquisition_date.values[0]
            # Use a set to automatically handle uniqueness.
            acquisition_dates.add(k)

    return list(acquisition_dates)


def intersect(polygon1, polygon2):
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
    import shapely.geometry
    import shapely.wkt

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


def get_footprint(ds, index: int):
    """
    Return the footprint coordinates of an image.

    The footprint of the indexed element (e.g., a single burst) will be returned.

    Parameters
    ----------
    ds : sarpy.io.general.base.SICDTypeReader
        The input data opened by sarpy.io.
    index : int
        Index of the image element whose footprint has to be returned.

    Returns
    -------
    list
        The footprint coordinates.
    """

    # Extract the corner coordinates for all images from the metadata.
    corners = {
        i: [
            (ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lon, ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lat)
            for j in range(5)
        ]
        for i in range(len(ds.sicd_meta))
    }

    #  return the footprint of that specific image.
    return corners[index]


def get_bbox(slcs, reference_date: str | None = None):
    """
    Get the bounding box of the SAR data.

    Parameters
    ----------
    slcs : xr.DataTree
        DataTree of SAR data.
    reference_date : str, optional
        Reference date for selecting specific data.

    Returns
    -------
    list
        Bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    lat, lon = [], []

    # Iterate through all scenes.
    for slc in (slcs.bursts if "bursts" in slcs else {"data": slcs}).values():
        # Optionally filter by a reference date.
        for date in slc.orig if reference_date is None else [reference_date]:
            # Extract corner coordinates from the metadata.
            annot = slc.orig[date].measurements.slc.attrs
            annot = annot[list(annot.keys())[0]]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])

    # Calculate the min/max latitude and longitude, rounding to one decimal place.
    return [
        np.floor(np.min(lon) * 10) / 10,
        np.floor(np.min(lat) * 10) / 10,
        np.ceil(np.max(lon) * 10) / 10,
        np.ceil(np.max(lat) * 10) / 10,
    ]


def get_burst_id(annot) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        annot: An annotation object containing metadata about the collection.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """
    # Get the burst number and swath identifier from the metadata parameters.
    burst_id = int(annot.CollectionInfo.Parameters.get("BURST"))
    swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
    return (swath, burst_id)


def get_burst_id_safe(file_sets: Sequence[dict], annot, relative_orbit_number: int) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        file_sets (Sequence[dict]): A sequence of dictionaries representing file sets with metadata.
        annot: An annotation object containing metadata about the collection.
        relative_orbit_number (int): The relative orbit number for the current observation.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """

    # Identify the corresponding product file
    for file_set in file_sets:
        burst = int(annot.CollectionInfo.Parameters.get("BURST")) - 1
        swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
        polarization = annot.get_processed_polarization().lower().replace(":", "")

        product_file_name = file_set["product"]
        if f"-{swath}-" in os.path.basename(product_file_name) and f"-{polarization}-" in os.path.basename(
            product_file_name,
        ):
            break
    else:
        # If no matching product file is found
        product_file_name = None
    mode = swath[:-1]
    # Extract ANX (Ascending Node Crossing) time from the XML file
    root_node = sarpy.io.complex.sentinel._parse_xml(cast(str, product_file_name), without_ns=True)
    anx_time = float(root_node.findall("./swathTiming/burstList/burst")[burst].find("./azimuthAnxTime").text) # type: ignore
    # Constants for timing calculations
    TBEAM = {"iw": 2.758273, "ew": 3.0388819516}  # Time per beam in seconds
    TORB = 12 * 86400 / 175  # Orbital period in seconds
    DELTA = {
        "iw1": -0.5469335,
        "iw2": 0.2852695,
        "iw3": 1.3633065,
        "ew1": -0.20158940051821037,
        "ew2": 0.4289565994970726,
        "ew3": 0.9952805994812225,
        "ew4": 1.5820385994932618,
        "ew5": 2.192150599480655,
    }  # Swath-specific time offsets
    t = (relative_orbit_number - 1) * TORB + anx_time - DELTA[swath]
    burst_id = int(np.round(t / TBEAM[mode]))
    if mode == "ew":
        burst_id += 11

    return (swath, burst_id)


def get_relative_orbit_number(sentinel1_manifest: Element, ns: dict):
    """
    Extracts the relative orbit number from the Sentinel-1 manifest file.

    Parameters:
        in_path (str): The path to the directory containing the 'manifest.safe' file.

    Returns:
        int: The relative orbit number, or None if extraction fails.
    """
    relative_orbit_number = int(sentinel1_manifest.findall(".//safe:relativeOrbitNumber", ns)[0].text)  # type: ignore
    return relative_orbit_number


def get_margin(annot, border: int = 10):
    step = {
        "azimuth_time": np.timedelta64(
            round((annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"]) * 1e9),
            "ns",
        ),
        "slant_range_time": 2 * annot["Grid"]["Row"]["SS"] / speed_of_light,
    }
    # Define a margin for data selection to avoid edge effects.
    margin = {e: step[e] * border for e in ["azimuth_time", "slant_range_time"]}
    return margin
