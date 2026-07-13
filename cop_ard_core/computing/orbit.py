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
from collections import OrderedDict

from xml.etree.ElementTree import ElementTree
import numpy as np
import sarpy.io.complex.utils
import xarray as xr

from cop_ard_core.computing.io import Path, get_storage_options, normpath, write
from cop_ard_core.computing.logger import logger


def build_orbit_dataset(orbits: list[ElementTree]) -> xr.Dataset:
    """
    Parse one or more XML-based orbit files (.EOF) into an xarray Dataset.

    This function reads the state vectors (position and velocity) from Sentinel-1
    orbit files, which are in an XML format. It extracts the time stamps and
    corresponding position/velocity vectors for each state vector entry.

    Parameters
    ----------
    orbit_paths : str or list[str]
        A single path or a list of paths to the
        orbit file(s) to be parsed.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the parsed orbit data, with
        'position' and 'velocity' as data variables and 'azimuth_time' as the
        primary coordinate. It also includes the orbit type as an attribute.
    """

    # Helper function to read a floating-point value from an XML tag.
    def read_v(osv, tag):
        return float(osv.find(tag).text)

    # Helper function to read a time string from an XML tag.
    def read_t(osv, tag):
        return osv.find(tag).text.split("=")[1]

    state_vectors = OrderedDict()
    otypes = set()
    for orbit in orbits:
        # Helper function to read a floating-point value from an XML tag.
        osvs = orbit.find(".//List_of_OSVs")
        otype = orbit.find(".//File_Type")
        if osvs is None or otype is None:
            raise ValueError("error reading orbit file")
        otypes.add(otype.text)
        # Iterate over each OSV element.
        for osv in osvs.findall("OSV") or []:
            # Extract the different time standards (UTC, TAI, UT1).
            utc, tai, ut1 = [read_t(osv, tag) for tag in ["UTC", "TAI", "UT1"]]
            # Extract the position (X, Y, Z) and velocity (VX, VY, VZ) components.
            x, y, z, vx, vy, vz = [read_v(osv, tag) for tag in ["X", "Y", "Z", "VX", "VY", "VZ"]]
            # Store the parsed data in the dictionary with UTC time as the key.
            state_vectors[utc] = [tai, ut1, x, y, z, vx, vy, vz]

    # --- Convert the collected data into structured numpy arrays ---
    utc = np.array(tuple(state_vectors.keys())).astype("datetime64[ns]")
    tai, ut1, x, y, z, vx, vy, vz = zip(*state_vectors.values())
    tai = np.array(tuple(tai)).astype("datetime64[ns]")
    ut1 = np.array(tuple(ut1)).astype("datetime64[ns]")

    # Stack the individual components into position and velocity vectors.
    position = np.stack([x, y, z], axis=-1)
    velocity = np.stack([vx, vy, vz], axis=-1)

    # --- Create the final xarray Dataset ---
    coords = {
        "azimuth_time": utc,
        "ut1": (["azimuth_time"], ut1),
        "tai": (["azimuth_time"], tai),
        "axis": ["x", "y", "z"],
    }
    position = xr.DataArray(position, dims=["azimuth_time", "axis"], coords=coords)
    velocity = xr.DataArray(velocity, dims=["azimuth_time", "axis"], coords=coords)
    # Extract the orbit file type (e.g., RESORB, POEORB) from the XML header.
    # Ensure all provided orbit files are of the same type.
    if len(otypes) > 1:
        raise RuntimeError("For the same acquisition date must be used the same orbit type (POE or RES)")

    # Create and return the final dataset with the orbit type as an attribute.
    return xr.Dataset({"position": position, "velocity": velocity}).assign_attrs({"type": otypes[0]})


def refit_orbit_coefficients(
    burst_id: str,
    acquisition_date: str,
    orbit: xr.Dataset | None,
    t0: np.datetime64,
    reference_time: np.ndarray,
    poly_degree: int,
) -> xr.Dataset | None:
    """
    Refit orbit polynomials using precise orbit data and update SLC metadata.

    This worker function takes precise orbit state vectors, fits new polynomials
    of a specified degree to them, and then overwrites the original orbit
    polynomial coefficients in the SLC metadata.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    acquisition_date : str
        The acquisition date of the SLC.
    slc : xr.DataTree
        The DataTree node for the specific burst to be updated.
    orbit : xr.Dataset | None
        The precise orbit data as an xarray Dataset. If None, no update is performed.
    poly_degree : int
        The degree of the polynomial to fit to the orbit data.

    Returns
    -------
    tuple[str, str, xr.Dataset | None]
        A tuple containing the burst_id, acquisition_date, and a new Dataset
        with the refitted orbit coefficients, or None if no update was performed.
    """
    # Proceed only if precise orbit data is available.
    if orbit is None:
        # If no orbit data was provided, return None.
        return None

    # Calculate the time vector relative to the reference time for polynomial fitting.
    times = (orbit.azimuth_time - t0).astype("float64") / 1e9
    # Use Sarpy's utility to fit new polynomials to the position and velocity data.
    X, Y, Z = sarpy.io.complex.utils.fit_position_xvalidation(
        times.values, orbit.position.values, orbit.velocity.values, max_degree=poly_degree
    )

    # --- Create and Save New Orbit Dataset ---
    attrs = {
        "reference_time": reference_time,
        "orbit_files": orbit.orbit_file.values.astype("str"),
        "orbit_type": orbit.type,
    }
    orbit_ = xr.Dataset(
        data_vars={
            "coeffs": (
                ["acquisition_date", "degree", "axis"],
                # Structure the new coefficients into the correct shape.
                np.array([v for v in [X, Y, Z]]).T[None, ...],
            )
        },
        coords={
            "acquisition_date": np.array([acquisition_date], dtype=object),
            "degree": range(poly_degree + 1),
            "axis": range(3),
        },
        attrs=attrs,
    )
    return orbit_
