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
import re
from collections import OrderedDict
from datetime import datetime
from typing import Any, Union, cast

import dateutil.parser
import defusedxml.ElementTree as ET
import numpy as np
import sarpy.io.complex.utils
import xarray as xr

from cop_ard_core.computing.computation import map
from cop_ard_core.computing.io import Path, get_storage_options, normpath, write
from cop_ard_core.computing.logger import logger, timeit


def read_orbit_file(orbit_paths, **kwargs):
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

    def read_v(osv, tag):
        return float(osv.find(tag).text)

    # Helper function to read a time string from an XML tag.
    def read_t(osv, tag):
        return osv.find(tag).text.split("=")[1]

    state_vectors = OrderedDict()
    otype = []
    orbit_paths = [orbit_paths] if isinstance(orbit_paths, Path) else orbit_paths
    for orbit_path in orbit_paths:
        local_orbit_path = orbit_path.ensure_local()
        # Helper function to read a floating-point value from an XML tag.
        osvs = ET.parse(local_orbit_path.path).find(".//List_of_OSVs")
        otype.append(ET.parse(local_orbit_path.path).find(".//File_Type").text)

        # Iterate over each OSV element.
        for osv in osvs.findall("OSV"):
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
    if np.unique(otype).size > 1:
        raise RuntimeError("For the same acquisition date must be used the same orbit type (POE or RES)")

    # Create and return the final dataset with the orbit type as an attribute.
    return xr.Dataset({"position": position, "velocity": velocity}).assign_attrs({"type": otype[0]})


def import_orbit(
    orbit_paths: list[Path] | None = None,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
    **kwargs,
) -> Union[xr.Dataset | None]:
    """
    Import and filter orbit data from specified files for a given time interval.

    This function reads orbit data from one or more files and then filters the
    resulting dataset to a specific time range defined by `start_time` and
    `end_time`. It is decorated with a retry mechanism to handle transient
    network or file access issues.

    Parameters
    ----------
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to the orbit files to import.
    start_time : np.datetime64 | None, optional
        The start of the desired time interval for the orbit data.
    end_time : np.datetime64 | None, optional
        The end of the desired time interval for the orbit data.

    Returns
    -------
    xr.Dataset | None
        An xarray.Dataset containing the filtered orbit data, or None if no
        `orbit_paths` are provided.
    """
    # Proceed only if orbit paths are provided.
    if (orbit_paths is not None) and all(e is not None for e in orbit_paths):
        logger.info(f"For the time interval {start_time} - {end_time} these orbit files will be used: {orbit_paths}")
        # Read the raw orbit data from the specified file(s).
        orbit = read_orbit_file(orbit_paths, **kwargs)
    else:
        # If no paths are given, log the information and return None.
        logger.info(f"For the time interval {start_time} - {end_time} no external orbits have been found")
        return None

    # If start_time is not provided, default to the first timestamp in the orbit data.
    if start_time is None:
        start_time = orbit.azimuth_time[0]
    # If end_time is not provided, default to the last timestamp.
    if end_time is None:
        end_time = orbit.azimuth_time[-1]

    # Filter the orbit dataset to the specified time interval and add the source filenames as an attribute.
    return orbit.where((orbit.azimuth_time >= start_time) & (orbit.azimuth_time <= end_time), drop=True).assign(
        orbit_file=(["acquisition_date"], np.array([",".join([e.name for e in orbit_paths])], dtype=object))
    )


def overwrite_coeffs(
    burst_id: str,
    acquisition_date: str,
    slc: xr.DataArray,
    orbit: xr.Dataset | None,
    wdir_path: str,
    poly_degree: int,
    multiburst: bool,
    **kwargs,
) -> tuple[str, str, xr.Dataset | None]:
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
    path = f"bursts/{burst_id}" if multiburst else ""
    data_path = normpath(f"{wdir_path}/{path}/orig/{acquisition_date}/conditions/orbit")

    # Proceed only if precise orbit data is available.
    if orbit is not None:
        logger.info(f"Updating orbit of {burst_id} image: {acquisition_date}")
        attrs = slc.attrs
        # --- Polynomial Fitting ---
        annot = attrs[list(attrs)[0]]
        # Get the reference time from the original annotation.
        t0 = annot["Timeline"]["CollectStart"][:-1]
        # Calculate the time vector relative to the reference time for polynomial fitting.
        times = (orbit.azimuth_time - np.datetime64(t0, "ns")).astype("float64") / 1e9
        # Use Sarpy's utility to fit new polynomials to the position and velocity data.
        X, Y, Z = sarpy.io.complex.utils.fit_position_xvalidation(
            times.values, orbit.position.values, orbit.velocity.values, max_degree=poly_degree
        )

        # --- Create and Save New Orbit Dataset ---
        attrs = {
            "reference_time": annot["Timeline"]["CollectStart"],
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
        # Write the new orbit dataset to the 'conditions/orbit' group in the Zarr store.
        orbit_ = write(
            orbit_,
            data_path,
            mode="w",
            storage_options=get_storage_options(kwargs),
        )
        if not isinstance(orbit_, xr.Dataset):
            raise ValueError(f"Expected xr.Dataset, got {type(orbit_)}")
        result: tuple[str, str, xr.Dataset | None] = (
            burst_id,
            acquisition_date,
            orbit_,
        )
    else:
        # If no orbit data was provided, log and return None.
        logger.info(f"No orbit update for the {burst_id} image: {acquisition_date}")
        result = (burst_id, acquisition_date, None)

    return result


def update_orbit(
    slcs: xr.DataTree,
    wdir_path: str,
    orbit_paths: list[Path] | None = None,
    multiburst: bool = True,
    poly_degree: int = 10,
    border: np.timedelta64 = np.timedelta64(100, "s"),
    enable_dask: bool = True,
    **kwargs,
):
    """
    Orchestrate the update of orbit polynomials for an entire SLC stack.

    This function identifies the required time intervals for each acquisition,
    imports the corresponding precise orbit data, and then maps the
    `overwrite_coeffs` worker function in parallel to refit the polynomials
    and update the metadata for every SLC in the stack.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data to be updated.
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to precise orbit files. If None, it will try to
        find paths from ETAD metadata.
    multiburst : bool, optional
        A boolean flag for multiburst mode. Defaults to True.
    poly_degree : int, optional
        The degree for the new polynomial fit. Defaults to 10.
    border : np.timedelta64, optional
        A time border to add to the acquisition interval when fetching
        orbit data, to ensure sufficient data for fitting.
        Defaults to np.timedelta64(100, "s").

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree with new orbit information.
    """
    slcs_info: dict[str, Any] = {}
    # --- Step 1: Collect SLC Time Intervals and Metadata ---
    # Iterate through all original SLCs in the DataTree.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for k, v in i.orig.items():
            # Retrieve the orbit source file path if available (e.g., from ETAD metadata).
            if orbit_paths is None:
                try:
                    orb_file = v.conditions.ETAD.attrs.get("orbit_file", None)
                except Exception as e:
                    logger.warning(f"type is: {e.__class__.__name__}")
                    logger.warning("Orbit file not found")
                    orb_file = None
            else:
                orb_file = None
            # Extract metadata and the start/end azimuth times.
            annot = v.measurements.slc.attrs[list(v.measurements.slc.attrs.keys())[0]]
            info = [
                [
                    annot["CollectionInfo"]["CollectorName"],
                    orb_file,
                    v.measurements.slc.azimuth_time[0].values,
                    v.measurements.slc.azimuth_time[-1].values,
                ]
            ]
            # Group the information by acquisition date.
            if k in slcs_info:
                slcs_info[k] += info
            else:
                slcs_info[k] = info

    # Consolidate the information for each date to find the overall time range.
    slcs_info = {k: list(zip(*v)) for k, v in slcs_info.items()}
    slcs_info = {k: [v[0][:2], v[1], np.min(v[2]) - border, np.max(v[3]) + border] for k, v in slcs_info.items()}

    # --- Step 2: Match Orbit Files to Time Intervals ---
    if orbit_paths is not None:
        # Parse the validity time from the orbit filenames.
        orbit_intervals = np.array(
            [
                [dateutil.parser.parse(j).isoformat() for j in i.name.split("_V")[1][:-4].split("_")]
                for i in orbit_paths
            ],
            dtype="datetime64[ns]",
        )
        # For each acquisition, find the orbit files that cover its time interval.
        for k, v in slcs_info.items():
            (i_valid,) = np.where((v[2] <= orbit_intervals[:, 1]) & (v[3] >= orbit_intervals[:, 0]))
            if i_valid.size:
                slcs_info[k][1] = np.array(orbit_paths)[i_valid].tolist()

    # --- Step 3: Import Orbit Data in Parallel ---
    dates = list(slcs_info)
    args = [[slcs_info[e][i] for e in dates] for i in range(1, 4)]
    orbits = map(import_orbit, args=args, kwargs=kwargs, combine=False, enable_dask=False)
    if not isinstance(orbits, list):
        raise Exception(f"Expected list, got {type(orbits)}")
    orbits = {d: o for d, o in zip(dates, orbits)}

    # --- Step 4: Refit Polynomials and Update Metadata in Parallel ---
    # Prepare arguments for the `overwrite_coeffs` worker function.
    args = [
        (k, d, v.measurements.slc, orbits[d])
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
        for d, v in s.orig.items()
    ]
    args = list(zip(*args))
    results = cast(
        list,
        map(
            overwrite_coeffs,
            args,
            kwargs={"wdir_path": wdir_path, "poly_degree": poly_degree, "multiburst": multiburst, **kwargs},
            combine=False,
            enable_dask=enable_dask,
        ),
    )
    # --- Step 5: Integrate Results ---
    # Update the main DataTree with the new orbit condition datasets.
    slcs = update_slcs(slcs, multiburst, results)
    return slcs


def update_slcs(slcs: xr.DataTree, multiburst: bool, conditions: list):
    """
    Update the main DataTree with newly computed orbit conditions.

    This function takes the results from the `overwrite_coeffs` parallel map
    operation and places each new orbit dataset into the correct location
    within the `slcs` DataTree structure.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree to be updated.
    multiburst : bool
        A boolean flag for multiburst mode, which determines the
        path structure.
    conditions : list
        The list of results from `overwrite_coeffs`, where each item is
        a tuple (burst_id, acquisition_date, orbit_dataset).

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree.
    """
    # Iterate through the list of results.
    for burst_id, acquisition_date, orbit in conditions:
        if orbit is None:
            continue

        path_prefix = f"bursts/{burst_id}" if multiburst else ""
        base_node = f"{path_prefix}/orig/{acquisition_date}"

        # 2. Assign the new orbit dataset to its place in the DataTree
        slcs[f"{base_node}/conditions/orbit"] = orbit

        # 3. Pre-extract coefficients
        xyz_coeffs = orbit.coeffs.sel(acquisition_date=acquisition_date)

        # 4. Update the metadata attributes
        slc_path = f"{base_node}/measurements/slc"
        attrs = slcs[slc_path].attrs

        for pol in attrs:
            for axis_idx, tag in enumerate(["X", "Y", "Z"]):
                # .tolist() safely converts numpy arrays to standard Python lists
                attrs[pol]["Position"]["ARPPoly"][tag]["Coefs"] = xyz_coeffs.isel(axis=axis_idx).values.tolist()

        # Reassign the updated attributes back to the node
        slcs[slc_path].attrs = attrs

    return slcs


@timeit
def import_ORBIT(
    slcs: xr.DataTree,
    acquisition_dates: list[np.datetime64],
    orbit_files: list[Path],
    wdir_path: str,
    multiburst: bool = False,
    enable_dask: bool = True,
    **kwargs,
):
    """
    High-level wrapper to find and import precise orbits for a stack.

    This function finds the relevant precise orbit files from a given folder that
    correspond to a list of acquisition dates, and then triggers the main
    `update_orbit` process to refit and update the orbit information in the `slcs`
    DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data.
    acquisition_dates : list[np.datetime64]
        A list of acquisition dates to find orbits for.
    orbit_folder : str
        The path to a folder containing orbit files (.EOF).
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
        Defaults to False.

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree containing the new precise
        orbit information.
    """
    orbit_paths = []
    # --- Find Relevant Orbit Files ---
    # For each acquisition date, determine the total time span covered by all its bursts.
    for acquisition_date in acquisition_dates:
        for counter, (k, slc) in enumerate((slcs.bursts if multiburst else {"data": slcs}).items()):
            if acquisition_date in list(slc.orig.keys()):
                az = slc.orig[acquisition_date].measurements.azimuth_time
                s, e = az.min().values, az.max().values
                if counter == 0:
                    start_time = s
                    end_time = e
                else:
                    start_time = min(start_time, s)
                    end_time = max(end_time, e)
        # Find the orbit files whose validity period covers this time span.
        for o in orbit_files:
            # Parse the start and end validity times from the orbit filename.
            pattern = r"V(\d{8}T\d{6})_(\d{8}T\d{6})"
            match = re.search(pattern, o.name)
            if not match:
                raise ValueError(f"No validity dates found in filename: {o.name}")

            # Convert extracted strings directly to numpy datetime objects
            valid_start = np.datetime64(datetime.strptime(match.group(1), "%Y%m%dT%H%M%S"))
            valid_stop = np.datetime64(datetime.strptime(match.group(2), "%Y%m%dT%H%M%S"))
            # Add the orbit file if it completely covers the acquisition time window
            if valid_start <= start_time and end_time <= valid_stop:
                orbit_paths.append(o)

    # --- Trigger Orbit Update ---
    if orbit_paths:
        # Call the main update function with the filtered list of orbit paths.
        update_orbit(
            slcs, multiburst=multiburst, orbit_paths=orbit_paths, wdir_path=wdir_path, enable_dask=enable_dask, **kwargs
        )
    orbit_paths = [o.path for o in orbit_paths]
    return slcs, orbit_paths
