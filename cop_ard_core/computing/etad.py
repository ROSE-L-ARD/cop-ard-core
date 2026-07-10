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
from typing import Any, Union, cast

import numpy as np
import scipy.interpolate
import xarray as xr

from cop_ard_core.computing.io import Path, get_storage_options, normpath, read, write
from cop_ard_core.computing.logger import logger, timeit
from cop_ard_core.computing.orbit import update_orbit


def discover_etaddate_slcsdate(slcs: xr.DataTree, etad: xr.DataTree, multiburst: bool) -> set:
    """
    Discover SLC acquisition dates that fall within the time range of an ETAD product.

    This function checks which Single Look Complex (SLC) acquisitions have an azimuth
    time range that is fully contained within the valid time range of a given
    Extended Timing Annotation Data (ETAD) product.

    Parameters
    ----------
    slcs : xarray.DataTree
        A DataTree object containing the SLCs data, organized by bursts
        if `multiburst` is True.
    etad : xarray.DataTree
        A DataTree object for a single ETAD product, containing its
        global time range in the attributes.
    multiburst : bool
        A boolean flag indicating if the `slcs` DataTree is in
        multiburst format.

    Returns
    -------
    set
        A set of unique acquisition date strings (e.g., "2022-01-01") for all
        SLCs that correspond to the provided ETAD data.
    """
    # Extract the start and end azimuth times of the ETAD correction's validity from its attributes.
    tstart = np.datetime64(etad.attrs["azimuthTimeMin"], "ns")
    tend = np.datetime64(etad.attrs["azimuthTimeMax"], "ns")

    # Initialize a set to store the unique matching acquisition date IDs.
    dateid = set()

    # Iterate over all SLCs (either in bursts or as a single scene).
    for _, i in (slcs.bursts if multiburst else {"data": slcs}).items():
        for name, j in i.orig.items():
            # Check if the SLC's entire azimuth time span is within the ETAD's time span.
            if (j.measurements[0] >= tstart) & (j.measurements[-1] <= tend):
                # If it matches, add the acquisition date to the set.
                dateid.add(name)

    # Return the set of unique SLC acquisition dates covered by this ETAD.
    return dateid


def save_etad(
    etad_burst: xr.DataTree,
    min_acquisition_azimuth_time: np.datetime64,
    min_acquisition_range_time: np.float64,
    etadf: str,
    out_path: str,
    **kwargs,
):
    """
    Save the ETAD correction grids for a single burst to a Zarr store.

    This function extracts the azimuth and range correction grids from an ETAD
    burst, reconstructs their time-based coordinates, and saves them as an
    xarray.Dataset in a specified Zarr store.

    Parameters
    ----------
    etad : xr.DataTree
        The full ETAD DataTree, used for global attributes like min/max times.
    etad_burst : xr.DataTree
        A DataTree node representing a specific burst within the ETAD data.
    etadf : str
        The path to the original source ETAD file, stored as an attribute.
    out_path : str
        The output path for the new Zarr store.

    Returns
    -------
    None
        None. The function writes the data directly to disk.
    """
    # --- Reconstruct Coordinate Grids ---
    # Calculate the absolute start time of the ETAD grid for this burst in azimuth.
    t0az = min_acquisition_azimuth_time + np.timedelta64(
        np.round(etad_burst.attrs["gridStartAzimuthTime"] * 1e9).astype(int), "ns"
    )
    # Calculate the absolute start time of the ETAD grid in range.
    t0rg = min_acquisition_range_time + etad_burst.attrs["gridStartRangeTime"]

    # Extract the grid spacing (sampling interval) in azimuth and range.
    dtaz = etad_burst.attrs["gridSamplingAzimuth"]
    dtrg = etad_burst.attrs["gridSamplingRange"]

    # Create the full coordinate arrays in time units.
    coords = {
        "azimuth_time": np.round(etad_burst.azimuthExtent.values * dtaz * 1e9).astype(int).astype("timedelta64[ns]")
        + t0az,
        "slant_range_time": etad_burst.rangeExtent.values * dtrg + t0rg,
    }

    # --- Create and Save Dataset ---
    # Create the xarray Dataset containing the azimuth and range correction grids.
    etad_ds = xr.Dataset(
        data_vars={
            "azcorr": (["azimuth_time", "slant_range_time"], etad_burst.sumOfCorrectionsAz.values),
            "rgcorr": (["azimuth_time", "slant_range_time"], etad_burst.sumOfCorrectionsRg.values),
        },
        coords=coords,
        attrs={"etad_file": etadf},
    ).chunk(1024)  # Apply chunking for efficient access.

    # Save the dataset to a Zarr store.
    write(etad_ds, out_path, storage_options=get_storage_options(kwargs), skip_read=True)
    return


def update_etad(
    etad_burst: xr.DataTree,
    sw_name: str,
    acquisition_date: str,
    etad_nc_path: str,
    min_acquisition_azimuth_time: np.datetime64,
    min_acquisition_range_time: np.float64,
    wdir_path: str,
    **kwargs,
):
    """
    A worker function to save ETAD data for a burst and return the result.

    This function is designed to be used in a parallel map operation. It constructs
    the appropriate output path, calls `save_etad` to write the data to disk,
    and then reads the data back to return it to the main process.

    Parameters
    ----------
    acquisition_date : str
        The acquisition date string (e.g., "2022-01-01").
    sw_name : str
        The name of the swath (e.g., "IW1").
    etad_nc : str
        The path to the source ETAD NetCDF file.
    etad : xr.DataTree
        The full ETAD DataTree.
    etad_burst : xr.DataTree
        The DataTree node for the specific ETAD burst to process.
    wdir_path : str
        The root working directory for the output.

    Returns
    -------
    tuple
        A tuple containing the acquisition date, the burst ID, and the newly
        created ETAD xarray.Dataset read from disk.
    """
    # Construct the unique burst ID from the swath name and burst number.
    bid = f"{sw_name.lower()}_{etad_burst.attrs['burstId']}"
    # Define the final output path for this burst's ETAD data within the SLC DataTree structure.
    data_path = normpath(f"{wdir_path}/{bid}/orig/{acquisition_date}/conditions/ETAD")
    # Save the ETAD correction grids to the specified path.
    save_etad(etad_burst, min_acquisition_azimuth_time, min_acquisition_range_time, etad_nc_path, data_path, **kwargs)
    # Read the newly created dataset back from disk and return it.
    return (
        acquisition_date,
        bid,
        read(data_path, storage_options=get_storage_options(kwargs)),
    )


def apply_etad_corrections(
    az: xr.DataArray,
    rg: xr.DataArray,
    etad_azimuth_correction_grid: xr.DataArray | None,
    etad_range_correction_grid: xr.DataArray | None,
    dtaz,
    dtrg,
    rg0,
    az0,
):
    """
    Apply ETAD corrections by interpolating the correction grids.

    This function takes azimuth and range time coordinates and applies the
    ETAD corrections by performing a 2D interpolation on the azimuth and range
    correction grids (`azcorr`, `rgcorr`).

    Parameters
    ----------
    az : xr.DataArray
        The input azimuth time coordinates to be corrected.
    rg : xr.DataArray
        The input slant range time coordinates to be corrected.
    orig : xr.DataTree
        The DataTree node for a single acquisition, containing the
        `conditions/ETAD` group with the correction grids.
    dtaz
        The azimuth time step (in ns) of the target grid.
    dtrg
        The slant range time step of the target grid.
    rg0
        The slant range time origin of the target grid.
    az0
        The azimuth time origin of the target grid.

    Returns
    -------
    tuple
        A tuple containing the interpolated azimuth (`daz`) and range (`drg`) corrections.
    """
    # initialize output
    daz = np.timedelta64(0, "ns")  # type: Union[np.timedelta64, np.ndarray]
    drg = np.zeros(1)  # type: np.ndarray

    # --- Prepare Coordinates for Interpolation ---
    # `xi`: The target coordinates where we want to know the correction values, in pixel units.
    xi = np.vstack(((rg - rg0).values.ravel() / dtrg, (az - az0).astype(float).values.ravel() / dtaz)).T.reshape(-1, 2)
    # `point`: The source coordinates of the ETAD correction grid, also in pixel units.
    etad = etad_azimuth_correction_grid if etad_azimuth_correction_grid is not None else etad_range_correction_grid
    if etad is not None:
        point = (
            (etad.slant_range_time - rg0).values / dtrg,
            (etad.azimuth_time - az0).astype(float).values / dtaz,
        )

    # --- Perform Interpolation ---
    # Interpolate the azimuth correction grid.
    # TODO: do a single interpolation
    if etad_azimuth_correction_grid is not None:
        # Use cubic interpolation for smooth results.
        daz = np.round(
            scipy.interpolate.interpn(
                point, etad_azimuth_correction_grid.values.T, xi, method="cubic", fill_value=0.0, bounds_error=False
            )
            * 1e9
        ).astype("timedelta64[ns]")
        daz = daz.reshape(az.shape)

    # Interpolate the range correction grid.
    if etad_range_correction_grid is not None:
        drg = scipy.interpolate.interpn(
            point, etad_range_correction_grid.values.T, xi, method="cubic", fill_value=0.0, bounds_error=False
        )
        drg = drg.reshape(rg.shape)

    # Return the computed correction values.
    return daz, drg
