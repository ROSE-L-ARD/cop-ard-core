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
from typing import Any, Container, Dict, Optional, Tuple, cast

import dask.array as da
import numpy as np
import rioxarray
import scipy.ndimage
import xarray as xr
from numpy.polynomial.polynomial import polyder, polyval
from pyproj import CRS
from scipy.constants import speed_of_light

from cop_ard_core.computing.dem import convert_to_dem_ecef
from cop_ard_core.computing.kernels import main_loop
from cop_ard_core.computing.logger import logger

rioxarray.__version__

DIMENSIONS_ = {
    "azimuth_time": ["acquisition_date", "y", "x"],
    "slant_range_time": ["acquisition_date", "y", "x"],
    "dem_distance": ["acquisition_date", "y", "x", "axis"],
    "h": ["y", "x"],
}

VARS_ = ["azimuth_time", "slant_range_time", "dem_distance", "h"]

DTYPE_ = {"azimuth_time": "datetime64[ns]", "slant_range_time": "float64", "dem_distance": "float64", "h": "float32"}

SPATIAL_DIMS_ = ["y", "x"]


def get_position(azimuth_time: xr.DataArray, t0: np.datetime64, arp_coeffs: dict[str, np.ndarray]) -> xr.DataArray:
    """
    Compute the satellite's ECEF position for given azimuth times.

    This function calculates the satellite's position in the Earth-Centered,
    Earth-Fixed (ECEF) coordinate system. It uses a polynomial model, with
    coefficients and a reference time provided in the SAR image's metadata (annotation),
    to evaluate the position at each specified azimuth time.

    Parameters
    ----------
    azimuth_time : xr.DataArray
        An xarray.DataArray of azimuth times, expressed in
        datetime64[ns].
    t0 : np.datetime64
        The reference time (t0) for the polynomial, typically the start of the
        acquisition, extracted from the annotation metadata.
    arp_coeffs : dict[str, np.ndarray]
        A dictionary containing the polynomial coefficients for the X, Y, and Z
        ECEF coordinates, respectively. These coefficients are typically found in
        the annotation metadata under 'Position' -> 'ARPPoly'.

    Returns
    -------
    xr.DataArray
        An xarray.DataArray containing the computed ECEF positions (X, Y, Z)
        as a function of azimuth time. The dimensions are ["axis", "azimuth_time"].
    """
    # Get the reference time (t0) from the annotation for the polynomial.

    # Calculate the relative azimuth time in seconds with respect to the reference time.
    az_time = (azimuth_time.values - t0).astype(float) / 1e9

    # Reconstruct the position by evaluating the polynomial for each axis (X, Y, Z).
    position = xr.DataArray(
        [polyval(az_time, arp_coeffs[e]) for e in ["X", "Y", "Z"]],
        dims=["axis", "azimuth_time"],
    )

    # Assign the original azimuth time coordinates back to the output DataArray.
    return position.assign_coords({"azimuth_time": azimuth_time.values})


def get_velocity(azimuth_time: xr.DataArray, t0: np.datetime64, arp_coeffs: dict[str, np.ndarray]) -> xr.DataArray:
    """
    Compute the satellite's ECEF velocity for given azimuth times.

    This function calculates the satellite's velocity vector in the ECEF
    coordinate system. It does this by first taking the derivative of the
    position polynomial (from the metadata) and then evaluating this new
    velocity polynomial at each specified azimuth time.

    Parameters
    ----------
    azimuth_time : xr.DataArray
        An xarray.DataArray of azimuth times, expressed in
        datetime64[ns].
    t0 : np.datetime64
        The reference time (t0) for the polynomial, typically the start of the
        acquisition, extracted from the annotation metadata.
    arp_coeffs : dict[str, np.ndarray]
        A dictionary containing the polynomial coefficients for the X, Y, and Z
        ECEF coordinates, respectively. These coefficients are typically found in
        the annotation metadata under 'Position' -> 'ARPPoly'.

    Returns
    -------
    xr.DataArray
        An xarray.DataArray containing the computed ECEF velocities (Vx, Vy, Vz)
        as a function of azimuth time. The dimensions are ["axis", "azimuth_time"].
    """
    # Calculate the relative azimuth time in seconds with respect to the reference time.
    az_time = (azimuth_time.values - t0).astype(float) / 1e9

    # Reconstruct the velocity by evaluating the derivative of the position polynomial for each axis.
    velocity = xr.DataArray(
        [polyval(az_time, polyder(arp_coeffs[e])) for e in ["X", "Y", "Z"]],
        dims=["axis", "azimuth_time"],
    )

    # Assign the original azimuth time coordinates back to the output DataArray.
    return velocity.assign_coords({"azimuth_time": azimuth_time.values})


def init_simulation(
    acquisition_dates: np.ndarray,
    crs: CRS,
    x: np.ndarray,
    y: np.ndarray,
) -> xr.Dataset:
    """
    Initialize an empty Zarr store for the DEM-to-radar simulation results.

    This function creates the structure for the simulation output on disk. It sets up
    an xarray.Dataset with the correct dimensions, coordinates, data types, and
    chunking for all the variables that will be computed in the simulation
    (e.g., azimuth_time, slant_range_time, etc.).

    Parameters
    ----------
    acquisition_dates : np.ndarray
        A numpy array of acquisition dates for the stack.
    crs : CRS
        The pyproj CRS object of the output grid.
    x : np.ndarray
        A numpy array of the x-coordinates for the output grid.
    y : np.ndarray
        A numpy array of the y-coordinates for the output grid.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset representing the initialized (but mostly empty) simulation data
    """
    n_dates = acquisition_dates.size

    # Define the shape for each variable in the simulation dataset.
    shape = {
        "azimuth_time": [n_dates, y.size, x.size],
        "slant_range_time": [n_dates, y.size, x.size],
        "dem_distance": [n_dates, y.size, x.size, 3],
        "h": [y.size, x.size],
    }

    # --- Create the Dataset Structure ---
    simulation = xr.Dataset(
        coords={
            "acquisition_date": acquisition_dates,
            "y": y,
            "x": x,
            "axis": [0, 1, 2],
        },
        # Create empty Dask arrays for each data variable.
        data_vars={
            e: xr.DataArray(
                da.empty(
                    shape=shape[e],
                    dtype=DTYPE_[e],
                ),
                dims=DIMENSIONS_[e],
            )
            for e in VARS_
            if e not in ["slant_range_time"]  # slant_range_time will be filled with NaNs later
        },
    )

    # Set the Coordinate Reference System (CRS) for the spatial dimensions.
    simulation.rio.write_crs(crs, inplace=True)

    # --- Initialize slant_range_time with Invalid Values ---
    # This is done separately to mark all pixels as unprocessed initially.
    simulation = simulation.assign(
        {
            "slant_range_time": xr.full_like(simulation.azimuth_time, fill_value=np.nan, dtype="float64"),
        }
    )

    return simulation


def compute_simulation(
    simulation: xr.Dataset,
    dem: xr.DataArray,
    orbits: xr.DataArray,
) -> xr.Dataset:
    """
    Compute radar geometry for a single block of the DEM.

    This is the core worker function that runs on each Dask block. It takes a chunk
    of the DEM, converts its coordinates to ECEF, and then uses the `sarsen` library
    to solve the range-Doppler equations, finding the corresponding azimuth time and
    slant range time for each DEM pixel.

    Parameters
    ----------
    simulation : xr.Dataset
        The block of the simulation dataset to be filled.
    dem : xr.DataArray
        The corresponding block of the DEM.
    orbits : xr.DataArray
        The full satellite orbit data.

    Returns
    -------
    xr.Dataset
        The updated simulation dataset.
    """
    # Extract the orbit data and reference time for the current acquisition date.
    reference_times = orbits.reference_times
    position = orbits.sel(acquisition_date=simulation.acquisition_date.values[0])
    position.attrs = {
        "referenceTime": np.datetime64(reference_times[simulation.acquisition_date.values[0]][:-1], "ns"),
        "scaleTime": 1e-9,
    }

    # Proceed only if there are unprocessed pixels in this block.
    s = simulation.drop_vars(["h"])
    if simulation.slant_range_time.isnull().any().values:
        # Replace any NaNs in the DEM with 0 for the geometric calculations.
        mask = np.isfinite(dem)
        dem = dem.where(mask, 0, drop=False)

        # --- Coordinate Conversion ---
        # If the DEM is not already in ECEF (EPSG:4978), convert it.
        if (dem.rio.crs is not None) and (dem.rio.crs.to_epsg() == 4978):
            dem_ecef = dem
        else:
            dem_ecef = xr.DataArray(
                convert_to_dem_ecef(dem=dem),
                dims=["axis", "y", "x"],
                coords={"axis": [0, 1, 2], "x": dem.x.values, "y": dem.y.values},
            )
            dem_ecef.rio.write_crs(4978, inplace=True)

        # Reformat the ECEF DEM for the sarsen function.
        dem_ecef = dem_ecef.drop_vars(["y", "x"])
        dem_ecef = dem_ecef.transpose(*SPATIAL_DIMS_, "axis")

        # --- Core Simulation ---
        # Call the sarsen library's simulation function.
        position = position.assign_coords({"degree": np.arange(position.degree.size)})
        s = simulate_acquisition(
            dem_ecef,
            position,
            include_variables=("slant_range_time", "azimuth_time", "dem_distance"),
        ).chunk(simulation.squeeze().chunks)

        # --- Write Results ---
        # Format the results to match the output structure.
        s = s.expand_dims(dim="acquisition_date", axis=0).drop_vars(set(s.coords).intersection({"spatial_ref", "crs"}))
        s = s.reset_coords(drop=True)
        s = s.drop_vars(s.coords)
        # Write the computed block to the correct region in the Zarr store.
    return s


def get_geo_from_radar_bbox(
    data_ml: dict,
    bbox_radar: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float] | None:
    """
    Convert a bounding box from radar coordinates to geographic coordinates.

    This function uses pre-computed, downsampled ("multilooked") simulation data to
    find the geographic extent (ymin, xmin, ymax, xmax) that corresponds to a given
    bounding box in radar coordinates (azimuth time, slant range time).

    Parameters
    ----------
    data_ml : dict
        A dictionary containing the min and max radar coordinates for each
        coarse geographic grid cell.
    bbox_radar : Tuple[float, float, float, float]
        A tuple defining the bounding box in radar coordinates
        (az_min, rg_min, az_max, rg_max).

    Returns
    -------
    Tuple[float, float, float, float] | None
        A tuple defining the corresponding geographic bounding box, or None if
        there is no overlap.
    """
    # Unpack the radar coordinate bounding box.
    az_min, rg_min, az_max, rg_max = bbox_radar
    # Get the min/max radar coordinates from the multilooked data.
    tree = (
        data_ml["min"].azimuth_time.squeeze().values,
        data_ml["min"].slant_range_time.squeeze().values,
        data_ml["max"].azimuth_time.squeeze().values,
        data_ml["max"].slant_range_time.squeeze().values,
    )
    # Create a mask to find the coarse grid cells that overlap with the radar bounding box.
    mask = ((tree[0] > az_max) | (tree[1] > rg_max) | (tree[2] < az_min) | (tree[3] < rg_min)) == 0
    mask = mask.reshape(data_ml["min"].azimuth_time.shape)
    # --- Calculate Geographic Bounding Box ---
    if np.count_nonzero(mask) > 0:
        # Find the indices of the overlapping cells.
        ind = np.where(mask)
        # Get the min/max geographic coordinates from those cells.
        ymin = data_ml["min"].y[ind[0]].min().values
        xmin = data_ml["min"].x[ind[1]].min().values
        ymax = data_ml["max"].y[ind[0]].max().values
        xmax = data_ml["max"].x[ind[1]].max().values
        bbox_geo = ymin, xmin, ymax, xmax
    else:
        # If no overlap is found, log a message and return None.
        az_min_, rg_min_ = [np.nanmin(data_ml["min"][e].values) for e in ["azimuth_time", "slant_range_time"]]
        az_max_, rg_max_ = [np.nanmax(data_ml["max"][e].values) for e in ["azimuth_time", "slant_range_time"]]

        logger.info(
            f"no points inside requested region {az_min, az_max, rg_min, rg_max}\
             vs {az_min_, az_max_, rg_min_, rg_max_}"
        )

        bbox_geo = None
    return bbox_geo


def get_radar_bbox(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    margin: dict,
) -> Tuple[int | float, int | float, int | float, int | float]:
    """
    Calculate the bounding box in radar coordinates for a given radar chunk.

    Parameters
    ----------
    azimuth_time : xr.DataArray
        The azimuth time coordinate of the radar chunk.
    slant_range_time : xr.DataArray
        The slant range time coordinate of the radar chunk.
    margin : dict
        A dictionary specifying the margin to add to the radar bounding box in both azimuth and range directions
    """

    az = azimuth_time.values
    rg = slant_range_time.values

    # Define the radar time borders, including the margin.
    azimuth_time_margin = margin["azimuth_time"]
    slant_range_time_margin = margin["slant_range_time"]
    az_min, az_max = [
        np.nanmin(az) - azimuth_time_margin,
        np.nanmax(az) + azimuth_time_margin,
    ]
    rg_min, rg_max = [
        np.nanmin(rg) - slant_range_time_margin,
        np.nanmax(rg) + slant_range_time_margin,
    ]

    bbox_radar = az_min, rg_min, az_max, rg_max

    return bbox_radar


def extract_simul_on_radar_chunk(
    simulation: xr.Dataset,
    reference_date: str,
    bbox_radar: Tuple[int | float, int | float, int | float, int | float],
    bbox_geo: Tuple[float, float, float, float],
    simulation_secondary: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Extract the relevant subset of simulation data for a given radar chunk.

    This function determines the geographic area needed to process a specific chunk
    of radar data. It converts the radar chunk's boundaries to geographic
    coordinates and then selects the corresponding subset from the full simulation dataset.

    Parameters
    ----------
    simulation: xr.Dataset
        The full simulation dataset from which to extract the subset.
        `x` and `y` dimensions must be present. No other assumptions are made about its contents.
    data : xr.Dataset | xr.DataArray
        The radar data chunk, used to define the area of interest.
    reference_date : str
        The reference acquisition date.
    bbox_radar : Tuple[int | float, int | float, int | float, int | float]
        The bounding box in radar coordinates (az_min, rg_min, az_max, rg_max).
    bbox_geo : Tuple[float, float, float, float]
        The corresponding bounding box in geographic coordinates (ymin, xmin, ymax, xmax).
    simulation_secondary : xr.Dataset | None, optional
        An optional secondary simulation dataset to also subset.

    Returns
    -------
    xr.Dataset
        The selected subset of the simulation dataset.
    """
    chunks = {k: v[0] for k, v in simulation.chunksizes.items()}
    # Get the azimuth and range time coordinates of the radar chunk.
    # Ensure the coordinate order is correct based on the direction of the axes.
    az_min, rg_min, az_max, rg_max = bbox_radar
    xmin, xmax = (bbox_geo[1], bbox_geo[3]) if simulation.x[0] < simulation.x[-1] else (bbox_geo[3], bbox_geo[1])
    ymin, ymax = (bbox_geo[0], bbox_geo[2]) if simulation.y[0] < simulation.y[-1] else (bbox_geo[2], bbox_geo[0])

    # --- Select Data ---
    # Select the simulation data within the geographic bounding box.
    selection = simulation.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
    # --- Refine Selection with Dilation ---
    # Get the simulation data for the reference geometry.
    reference_sel = selection.sel(acquisition_date=reference_date).squeeze().drop_vars("acquisition_date")
    az = reference_sel.azimuth_time.values
    rg = reference_sel.slant_range_time.values
    # Create a precise mask of pixels within the radar time borders and dilate it.
    mask = xr.DataArray(
        scipy.ndimage.binary_dilation(
            (az >= az_min) & (az <= az_max) & (rg >= rg_min) & (rg <= rg_max),
            structure=scipy.ndimage.generate_binary_structure(2, 2),
        ),
        dims=["y", "x"],
    )
    # If the mask is not empty, use it to make a tighter selection.
    if np.count_nonzero(mask) > 0:
        ind = np.where(mask)
        xmin_idx, xmax_idx = ind[1].min(), ind[1].max() + 1
        ymin_idx, ymax_idx = ind[0].min(), ind[0].max() + 1
        selection = selection.isel(x=slice(xmin_idx, xmax_idx), y=slice(ymin_idx, ymax_idx)).chunk(chunks)
        # If a secondary simulation exists, apply the same tight selection.
        if simulation_secondary is not None:
            selection_secondary = simulation_secondary.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))
            selection_secondary = selection_secondary.isel(
                x=slice(xmin_idx, xmax_idx), y=slice(ymin_idx, ymax_idx)
            ).chunk(chunks)
            selection = xr.concat([selection, selection_secondary], dim="acquisition_date", data_vars="all").chunk(
                chunks
            )
    else:
        raise Exception("Selection is empty.")
    return selection


def extract_simulation_ml(
    simulation: xr.Dataset,
    reference_date: str,
    coarsen_config: dict[str, int]
):
    """
    Create a downsampled ("multilooked") version of the simulation data.

    This function coarsens the simulation data by taking the min and max values
    over blocks corresponding to the chunk size. This creates a low-resolution
    overview that is very fast to load and is used for quickly finding the
    geographic extent of a radar region.

    Parameters
    ----------
    simulation : xr.Dataset
        The full-resolution simulation dataset.
    reference_date : str
        The reference acquisition date to use.
    working_dir : str
        The directory where the multilooked data will be saved.
    recompute : bool, optional
        If True, force recomputation even if the data already exists.
        Defaults to False.

    Returns
    -------
    None
        None. The results are written to disk.
    """
    # --- Compute Min/Max over Chunks ---
    simulation_ml = {
        # For the 'min' dataset, coarsen the data and take the minimum value in each block.
        "min": cast(
            xr.Dataset,
            simulation.sel(acquisition_date=reference_date)
            .drop_vars(["dem_distance", "h", "axis"])
            .astype(float)
            .coarsen(coarsen_config, boundary="pad", coord_func="min"),
        )
        .min(),
        # For the 'max' dataset, do the same but take the maximum value.
        "max": cast(
            xr.Dataset,
            simulation.sel(acquisition_date=reference_date)
            .drop_vars(["dem_distance", "h", "axis"])
            .astype(float)
            .coarsen(coarsen_config, boundary="pad", coord_func="max"),
        )
        .max()
    }
    # Ensure the azimuth_time coordinate retains its correct data type.
    simulation_ml["min"]["azimuth_time"] = simulation_ml["min"].azimuth_time.astype("datetime64[ns]")
    simulation_ml["max"]["azimuth_time"] = simulation_ml["max"].azimuth_time.astype("datetime64[ns]")
    # --- Write Results to Disk ---
    return simulation_ml


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    pos: xr.DataArray,
    azimuth_time: Optional[xr.DataArray] = None,
    include_variables: Container[str] = (),
    **kwargs,
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""
    acquisition = backward_geocode(dem_ecef, pos, t0=azimuth_time)
    slant_range = (acquisition.dem_distance**2).sum(dim="axis") ** 0.5
    slant_range_time = 2.0 / speed_of_light * slant_range

    acquisition["slant_range_time"] = slant_range_time

    for data_var_name in acquisition.data_vars:
        if not isinstance(data_var_name, str):
            raise Exception(f"Expected str for data_var_name, got {type(data_var_name)}")
        if include_variables and data_var_name not in include_variables:
            acquisition = acquisition.drop_vars(data_var_name)

    # drop coordinates that are not associated with any data variable
    for coord_name in acquisition.coords:
        if not isinstance(coord_name, str):
            raise Exception(f"Expected str for data_var_name, got {type(coord_name)}")
        if all(coord_name not in dv.coords for dv in acquisition.data_vars.values()):
            acquisition = acquisition.drop_vars(coord_name)

    return acquisition


def backward_geocode(
    dem_ecef: xr.DataArray,
    pos: xr.DataArray,
    t0: xr.DataArray | np.datetime64 | None = None,
    dim: str = "axis",
    conv_th: float = 1.0e-6,
) -> xr.Dataset:
    if pos.attrs.get("referenceTime", None) is None:
        raise Exception("orbit reference time not defined")
    if pos.attrs.get("scaleTime", None) is None:
        raise Exception("orbit scaling time not defined")
    if t0 is not None:
        if isinstance(t0, xr.DataArray):
            if t0.size != dem_ecef.isel({dim: 0}).size:
                raise Exception("number of guess points different from dem_ecef size")
            t = ((t0 - pos.referenceTime) * pos.scaleTime).astype(float).values
        else:
            t = np.full_like(
                dem_ecef.isel({dim: 0}).drop_vars(dim),
                (t0 - pos.referenceTime) * pos.scaleTime,
                dtype=float,
            )
    else:
        t = np.full_like(dem_ecef.isel({dim: 0}).drop_vars(dim), 0.0, dtype=float)
    mydim = dem_ecef.isel({dim: 0}).drop_vars(dim).dims
    # pass all variables to numpy
    # compute orbit polynomial derivatives
    pos_ = pos.values
    vel_ = polyder(pos_, 1)
    acc_ = polyder(vel_, 1)
    dem = dem_ecef.values
    if dem.ndim == 2:
        dem, t = np.expand_dims(dem, axis=0), t[None, ...]
    direction_ecef = np.zeros_like(dem)
    r = np.zeros_like(dem, dtype=float)
    main_loop(pos_, vel_, acc_, dem, t, direction_ecef, r, conv_th)
    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": xr.DataArray(
                (np.atleast_2d(t.squeeze()) / pos.scaleTime).astype("timedelta64[ns]") + pos.referenceTime,
                dims=mydim,
            ),
            "dem_distance": xr.DataArray(-r.reshape(dem_ecef.shape), dims=dem_ecef.dims),
            "satellite_direction": xr.DataArray(direction_ecef.reshape(dem_ecef.shape), dims=dem_ecef.dims),
        }
    )
    return acquisition
