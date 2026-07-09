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
from scipy.constants import speed_of_light
import copy
import datetime
from typing import Mapping, Union, cast

import mgrs
import numpy as np
import scipy.ndimage
import xarray as xr
from pyproj import CRS, Transformer

import cop_ard_core
from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.dem import (
    GeoEnvPlugin,
    compute_shape_and_off,
    extract_vertical_crs,
    get_resolution_in_meters,
)
from cop_ard_core.computing.griddata import interp_dataset
from cop_ard_core.computing.io import get_id_product, get_storage_options, normpath, read, save_empty_dataset, write
from cop_ard_core.computing.kernels import gaussian_kernel
from cop_ard_core.computing.logger import logger, timeit
from cop_ard_core.computing.simul import extract_simul_on_radar_chunk, get_radar_bbox, get_geo_from_radar_bbox
from cop_ard_core.computing.slc import get_margin
from cop_ard_core.computing.warp import sinc, warp_gslc


@timeit
def stack_geocoding(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    working_directory_static: str,
    working_directory_tmp: str,
    spacing: dict,
    gamma_flattening: bool = False,
    do_reramp: bool = True,
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
    **kwargs,
):
    """
    Geocode a stack of coregistered SLCs in parallel.

    This function is the main entry point for the geocoding phase. It prepares
    the necessary arguments for each burst (or a single scene) and then maps the
    `single_geocode_phase` worker function across a Dask cluster to perform the
    geocoding in parallel. Finally, it assembles the geocoded results into a
    new DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing the coregistered SLC stack (`coreg` group).
    simulation_tree : xr.DataTree
        A DataTree containing the high-resolution simulation data
        (radar geometry on the DEM grid) for each burst.
    simulation_ml_tree : dict
        A dictionary with the downsampled ("multilooked") simulation
        data, used for quickly finding overlaps.
    working_directory_static : str
        The path to a directory for storing persistent static data.
    working_directory_tmp : str
        The path to a directory for temporary processing data.
    gamma_flattening : bool, optional
        If True, apply a radiometric terrain correction to produce
        gamma nought backscatter.
    do_reramp : bool, optional
        If True, re-apply the phase ramp to the geocoded data.
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
    **kwargs
        Additional keyword arguments passed to the processing functions,
        such as `warp_kernel_shape`.

    Returns
    -------
    xr.DataTree
        A new DataTree containing the geocoded SLC (GSLC) data for each burst.
    """
    args = []
    # --- Prepare Arguments for Parallel Mapping ---
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.coreg.drop_nodes("quality")
        unwanted_nodes = [child for child in v["conditions"].children if child != "phase_demod"]
        v["conditions"] = v["conditions"].drop_nodes(unwanted_nodes)
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Geocoding in Parallel ---
    # Map the `single_geocode_phase` worker function over the list of arguments.
    results = cast(
        list,
        map(
            single_geocode_phase,
            args,
            kwargs=dict(
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                gamma_flattening=gamma_flattening,
                do_reramp=do_reramp,
                enable_dask=enable_dask,
                chunks=chunks,
                **kwargs,
            ),
            enable_dask=enable_dask,
            combine=False,
        ),
    )

    # --- Assemble Results ---
    gslcs = xr.DataTree()
    # Reconstruct the DataTree from the list of results returned by the workers.
    for k, gslc in results:
        if multiburst:
            gslcs[f"bursts/{k}"] = gslc
        else:
            gslcs = gslc
    return gslcs


def single_geocode_phase(
    burst_id: str,
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    working_directory_static: str,
    working_directory_tmp: str,
    gamma_flattening: bool,
    spacing: dict,
    do_reramp: bool,
    chunks: dict,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Execute the end-to-end geocoding workflow for a single burst.

    This worker function prepares the interpolation kernel and calls `map_geocode_phase`
    to perform the core geocoding computation. It then merges the result with
    other geometric grid data (like incidence angles) to create the final
    geocoded product for the burst.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data for this burst.
    multiburst : bool
        A boolean flag for multiburst mode.
    working_directory_static : str
        The path for storing persistent static data.
    working_directory_tmp : str
        The path for temporary processing data.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool
        If True, re-apply the phase ramp.
    enable_dask : bool, optional
        If True, enable Dask for sub-tasks.
    **kwargs
        Additional keyword arguments for the warping function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the final geocoded xarray.Dataset for that burst.
    """
    # Define the output paths for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path_static = normpath(f"{working_directory_static}/{path}")
    data_path_tmp = normpath(f"{working_directory_tmp}/{path}")

    # --- Prepare Interpolation Kernel ---
    # Get warp parameters from kwargs or use default values.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = kwargs.get("warp_kernel_shape", np.array([11, 11]))
    kaiser_beta = kwargs.get("warp_kaiser_beta", 6)

    # Pre-compute the sinc kernels for x and y directions.
    sinc_x = sinc(kernel_shape[1], ov_fact, kaiser_beta)
    sinc_y = sinc(kernel_shape[0], ov_fact, kaiser_beta)

    kernel = {
        "kernel_shape": kernel_shape,
        "ov_fact": ov_fact,
        "kaiser_beta": kaiser_beta,
        "sinc_x": sinc_x,
        "sinc_y": sinc_y,
    }

    # --- Run Core Geocoding ---
    # This call orchestrates the block-wise warping from slant range to ground range.
    g = map_geocode_phase(
        cslc,
        simulation,
        simulation_ml,
        data_path_static,
        data_path_tmp,
        kernel,
        gamma_flattening,
        spacing,
        chunks,
        do_reramp,
        enable_dask,
        skip_read=False,  # We need the result back in memory.
        **kwargs,
    )

    # --- Merge with Geometric Data ---
    # Read the pre-computed geometric grid data (e.g., LIA, anf).
    geo_data = read(
        normpath(f"{data_path_static}/geo_grid"),
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )

    gslc = xr.DataTree.from_dict({"measurements": g, "conditions": geo_data})

    return (burst_id, gslc)


def map_geocode_phase(
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path_static: str,
    data_path_tmp: str,
    kernel: dict,
    gamma_flattening: bool,
    spacing: dict,
    chunks: dict,
    do_reramp: bool = True,
    enable_dask: bool = True,
    skip_read: bool = True,
    **kwargs,
) -> xr.Dataset | None:
    """
    Orchestrate the block-wise geocoding of a single coregistered SLC.

    This function initializes an empty Zarr store for the final Geocoded SLC (GSLC)
    product and then uses `map_blocks` to apply the `warp_gslc` function to
    chunks of the simulation grid.

    Parameters
    ----------
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data.
    data_path_static : str
        Path for persistent static data.
    data_path_tmp : str
        Path for temporary processing data.
    coreg_path : str
        Path to the source coregistered data.
    kernel : dict
        A dictionary with parameters for the sinc interpolation kernel.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool, optional
        If True, re-apply the phase ramp. Defaults to True.
    enable_dask : bool, optional
        If True, use Dask for sub-tasks. Defaults to True.
    skip_read : bool, optional
        If True, do not read and return the final dataset. Defaults to True.

    Returns
    -------
    xr.Dataset or None
        The geocoded Dataset if `skip_read` is False, otherwise None.
    """
    # --- Prepare Simulation Subset ---
    annot = cslc.measurements.cstack.attrs
    annot = annot[list(annot.keys())[0]]
    # Calculate grid spacing for margin calculation.
    margin = get_margin(annot)
    # Extract the subset of the simulation grid relevant to this CSLC.
    azimuth_time = cslc.measurements.azimuth_time
    slant_range_time = cslc.measurements.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = (
        (
            extract_simul_on_radar_chunk(
                simulation, simulation.acquisition_date.values[0], bbox_radar, bbox_geo,
            )
        )
        .squeeze()
        .drop_vars(["axis", "dem_distance", "acquisition_date", "h"])
    )

    # Define the output coordinates at pixel centers.
    shape, off = compute_shape_and_off(selection, spacing)
    # Create the coordinate arrays for the output grid.
    x, y = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output GSLC Dataset ---
    # Create the empty Zarr store for the Geocoded SLC.
    image = save_empty_dataset(
        data_vars=["gstack", "phase_model"],
        dims={
            "gstack": ["polarization", "acquisition_date", "y", "x"],
            "phase_model": ["acquisition_date", "y", "x"],
        },
        shape={
            "gstack": [
                cslc.measurements.polarization.size,
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
            "phase_model": [
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
        },
        chunks=chunks,
        dtype={
            "gstack": cslc.measurements.cstack.dtype,
            "phase_model": cslc.measurements.phase_model.dtype,
        },
        coords={
            "polarization": cslc.measurements.polarization.values,
            "acquisition_date": cslc.measurements.acquisition_date.values,
            "y": y,
            "x": x,
        },
        crs=selection.rio.crs.to_string(),
        data_path=normpath(f"{data_path_tmp}/geo_sinc"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        skip_read=False,
    )

    # --- Map the Geocoding Worker Function ---
    # This call applies the `warp_gslc` function to each block of the simulation grid.
    chunksizes = {k: v[0] for k, v in image.chunksizes.items()}
    chunksizes["acquisition_date"] = 10
    chunksizes["polarization"] = image.polarization.size
    radar_path = normpath(f"{data_path_static}/radar_grid")
    radar_data = read(
        radar_path,
        consolidate_metadata=False,
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )
    # Define paths to the different components of the coregistered data.
    measurements_path = normpath(f"{coreg_path}/measurements")
    conditions_path = normpath(f"{coreg_path}/conditions")
    # Read the data needed for warping.
    measurements = read(
        measurements_path,
        consolidate_metadata=False,
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    conditions = read(
        conditions_path,
        consolidate_metadata=False,
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    save_options = {
        "data_path": normpath(f"{data_path_tmp}/geo_sinc"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs, key="storage_options_dynamic"),
        },
    }

    def region_callback(region: Mapping[str, slice], kwargs: dict) -> Mapping[str, slice]:
        out = {}
        for k, v in region.items():
            if k in ["y", "x"]:
                out[k] = slice(v.start, v.stop - 1)
            elif k == "polarization":
                out[k] = slice(0, cslc.measurements.polarization.size)
            elif k == "acquisition_date":
                out[k] = v
            else:
                raise ValueError(f"Unexpected dimension {k}")
        return out

    # Extract metadata from the SLC attributes.
    annot = measurements.cstack.attrs
    annot = annot[list(annot)[0]]
    # Calculate the time interval per pixel in azimuth and range.
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
    scp_idx = annot["ImageData"]["SCPPixel"]["Col"], annot["ImageData"]["SCPPixel"]["Row"]


    map_blocks(
        warp_gslc,
        image,
        args=(
            selection,
            kernel,
            measurements.cstack,
            measurements.phase_model,
            conditions.phase_demod.phase_demod,
            radar_data.anf,
            radar_data.layover,
            radar_data.shadow,
            do_reramp,
            gamma_flattening,
            dtaz,
            dtrg,
            scp_idx,
        ),
        kwargs=kwargs,
        chunksizes=chunksizes,
        combine=False,
        enable_dask=enable_dask,
        region_callback={0: region_callback},
        store_mapping={0: save_options},
    )

    # Optionally read and return the final dataset.
    return (
        read(
            normpath(f"{data_path_tmp}/geo_sinc"),
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        )
        if not skip_read
        else None
    )


@timeit
def stack_interp_nrb(
    gslcs: xr.DataTree,
    utm_zones: dict[str, np.ndarray],
    multiburst: bool,
    working_directory: str,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Interpolate a stack of GSLCs into multiple target UTM zones in parallel.

    This function orchestrates the reprojection of Geocoded Single Look Complex (GSLC)
    data from their native projection into one or more specified UTM zones. This is
    a key step in producing a tiled Normalised Radar Backscatter (NRB) product.
    It maps the `interp_single_nrb` worker function across a Dask cluster to
    process each burst in parallel.

    Parameters
    ----------
    gslcs : xr.DataTree
        The input DataTree containing the geocoded SLC stack.
    utm_zones : dict[str, np.ndarray]
        A dictionary where keys are target UTM zone EPSG codes and
        values are their geographic bounding boxes.
    multiburst : bool
        A boolean flag indicating if the data is in multiburst format.
    working_directory : str
        The root directory for all processing outputs.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
    enable_dask : bool, optional
        If True, use Dask for parallel computation.
    **kwargs
        Additional keyword arguments passed to the processing functions.

    Returns
    -------
    xr.DataTree
        A new DataTree containing the reprojected data, organized by
        UTM zone and then by burst.
    """
    # --- Prepare Arguments for Parallel Mapping ---
    # Create a list of tuples, where each tuple contains the arguments for one call to the worker function.
    args = [(k, gslc.to_dataset(), multiburst) for k, gslc in (gslcs.bursts if multiburst else {"data": gslcs}).items()]
    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Interpolation in Parallel ---
    # Map the `interp_single_nrb` worker function over the list of arguments.
    results = map(
        interp_single_nrb,
        args,
        kwargs=dict(
            utm_zones=utm_zones,
            working_directory=working_directory,
            chunks=chunks,
            enable_dask=enable_dask,
            **kwargs,
        ),
        enable_dask=enable_dask,
        Plugin=GeoEnvPlugin(),
        combine=False,
    )
    if not isinstance(results, list):
        raise Exception(f"Expected list, got {type(results)}")

    # --- Assemble Results into a DataTree ---
    gslcs_tree = xr.DataTree()
    # Iterate through the list of results returned by the workers.
    for k, d in results:
        # Reconstruct the final DataTree, organizing the data by UTM zone and burst ID.
        for u, v in d.items():
            if multiburst:
                gslcs_tree[f"{u}/bursts/{k}"] = v
            else:
                gslcs_tree[u] = v
    return gslcs_tree


def compute_center(gslc: xr.Dataset, crs: str) -> dict:
    """
    Calculate the geometric center of a geocoded dataset in a target CRS.

    Parameters
    ----------
    gslc : xr.Dataset
        The input geocoded dataset (GSLC).
    crs : str
        The EPSG code of the target Coordinate Reference System in which
        to compute the center.

    Returns
    -------
    dict
        A dictionary containing the 'x' and 'y' coordinates of the center.
    """
    # Transform the bounding box of the dataset to the target CRS.
    min_x, min_y, max_x, max_y = gslc.rio.transform_bounds(crs, recalc=True)
    # Calculate the midpoint of the x and y extents.
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    return {"x": center_x, "y": center_y}


def smooth_nrb(gslc: xr.Dataset, size: int = 5, sigma: float = 1.0) -> xr.Dataset:
    """
    Apply a weighted Gaussian blur to the gamma0 backscatter.

    This function smooths the `gamma0` data using a Gaussian kernel. The smoothing
    is weighted by the number of looks (`nlooks`) to account for varying statistical
    reliability of pixels, which is common after terrain correction.

    Parameters
    ----------
    gslc : xr.Dataset
        The input geocoded dataset containing `gamma0` and `nlooks`.
    size : int, optional
        The size of the Gaussian kernel.
    sigma : float, optional
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    xr.Dataset
        The input dataset with the `gamma0` and `nlooks` variables
        updated with the smoothed values.
    """
    # Preserve the original chunking for the output.
    chunks = {k: v[0] for k, v in gslc.chunksizes.items()}
    # Extract data as numpy arrays for processing with scipy.
    gamma0 = gslc.gamma0.values
    nlooks = gslc.nlooks.values
    # Create the Gaussian kernel.
    kernel = gaussian_kernel(size=size, sigma=sigma)

    # --- Weighted Smoothing ---
    # First, smooth the number of looks.
    nlooks = scipy.ndimage.convolve(nlooks, kernel, mode="constant")
    # Then, smooth the gamma0 values, weighted by the original number of looks.
    numerator = scipy.ndimage.convolve(gamma0 * nlooks, kernel[None, None], mode="constant")
    # 2. Perform safe division
    gamma0 = np.divide(numerator, nlooks, out=np.full_like(numerator, 0.0), where=(nlooks != 0))
    # Normalize the smoothed looks.
    nlooks /= kernel.max()

    # --- Update Dataset ---
    # Create new DataArrays with the smoothed data.
    gamma0 = xr.DataArray(gamma0, dims=gslc.gamma0.dims)
    nlooks = xr.DataArray(nlooks, dims=gslc.nlooks.dims)
    # Replace the original variables with the smoothed ones.
    gslc["gamma0"] = gamma0
    gslc["nlooks"] = nlooks
    # Re-apply the original chunking.
    gslc = gslc.chunk(chunks)

    return gslc


def interp_single_nrb(
    burst_id: str,
    gslc: xr.Dataset,
    multiburst: str,
    utm_zones: dict[str, np.ndarray],
    chunks: dict,
    working_directory: str,
    enable_dask: bool = True,
    **kwargs,
):
    """
    Reproject a single GSLC into one or more target UTM zones.

    This worker function handles the reprojection of a single geocoded burst. It
    separates the target UTM zones into two groups: the native zone (which only
    requires clipping) and other zones (which require full interpolation).

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    gslc : xr.Dataset
        The input geocoded SLC (GSLC) dataset for the burst.
    multiburst : str
        A boolean flag for multiburst mode.
    utm_zones : dict[str, np.ndarray]
        A dictionary of target UTM zones and their geographic extents.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    working_directory : str
        The root directory for all outputs.
    enable_dask : bool, optional
        If True, use Dask for parallel sub-tasks.
    **kwargs
        Additional keyword arguments for interpolation and smoothing.

    Returns
    -------
    tuple
        A tuple containing the burst_id and a dictionary where keys are UTM
        zone EPSG codes and values are the corresponding reprojected datasets.
    """
    # --- Extract Parameters ---
    method = kwargs.get("interpolation", {}).get("method", "cubic")
    border = kwargs.get("interpolation", {}).get("border_pixel", 4)
    threshold = kwargs.get("interpolation", {}).get("threshold", 1e-5)

    kernel = {
        "size": kwargs.get("gaussian_blur", {}).get("size", 5),
        "sigma": kwargs.get("gaussian_blur", {}).get("sigma", 0.5),
    }

    # --- Setup CRS and Paths ---
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory}/{path}")
    crs: Union[str, CRS] = CRS(gslc.rio.crs).to_3d()
    if not isinstance(crs, CRS):
        raise Exception(f"Expected CRS, got {type(crs)}")
    horizontal_crs = str(crs.to_2d().to_epsg())
    vertical_crs = extract_vertical_crs(crs)
    vertical_crs = f"+{vertical_crs}" if crs.is_vertical else ""
    # Identify which target UTM zones require interpolation.
    utm_set = set(utm_zones) - {horizontal_crs}

    # --- Prepare for Reprojection ---
    data_vars = [str(k) for k in gslc.data_vars]
    dims = {str(k): cast(list[str], v.dims) for k, v in gslc.data_vars.items()}
    dtype = {str(k): cast(str, v.dtype) for k, v in gslc.data_vars.items()}
    spacing = get_resolution_in_meters(gslc)
    if crs.is_geographic:
        spacing = np.floor(spacing)

    gslc_utm = {}
    # --- Handle Native UTM Zone (Clipping only) ---
    if horizontal_crs in utm_zones:
        minx, miny, maxx, maxy = utm_zones[horizontal_crs]
        # Clip the GSLC to the bounds of its native UTM zone.
        gslc_aoi = gslc.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="4326").chunk(chunks)
        gslc_aoi = gslc_aoi.where(gslc_aoi >= threshold, 0.0)
        gslc_utm[horizontal_crs] = gslc_aoi
        crs_string = f"EPSG:{horizontal_crs}{vertical_crs}" if vertical_crs != "" else f"EPSG:{horizontal_crs}"
        # Store metadata about the center and interpolation status.
        gslc_utm[horizontal_crs].attrs["center"] = compute_center(gslc, crs_string)
        gslc_utm[horizontal_crs].attrs["interpolated"] = False

    # --- Handle Other UTM Zones (Full Interpolation) ---
    for h_crs in utm_set:
        gslc_path = normpath(f"{data_path}/interpolated/gslc_{h_crs}")
        # Get the geographic bounds for the target UTM zone.
        lon_min, lat_min, lon_max, lat_max = utm_zones[h_crs]
        lon, lat = [lon_min, lon_min, lon_max, lon_max], [lat_min, lat_max, lat_min, lat_max]
        # Transform these bounds into the source GSLC's coordinate system to find the area of interest.
        transformer = Transformer.from_crs(4326, gslc.rio.crs, always_xy=True)
        xb, yb = transformer.transform(lon, lat)
        xb_min, xb_max = np.min(xb), np.max(xb)
        yb_min, yb_max = np.min(yb), np.max(yb)
        x_slice = slice(xb_min, xb_max) if spacing[0] > 0 else slice(xb_max, xb_min)
        y_slice = slice(yb_min, yb_max) if spacing[1] > 0 else slice(yb_max, yb_min)

        # Select the relevant portion of the input GSLC.
        gslc_aoi = gslc.sel(x=x_slice, y=y_slice).chunk(chunks)
        if gslc_aoi.x.size == 0 or gslc_aoi.y.size == 0:
            continue

        # --- Define Output Grid for Interpolation ---
        crs_string = f"EPSG:{h_crs}{vertical_crs}" if vertical_crs != "" else f"EPSG:{h_crs}"
        crs = CRS(crs_string)
        transformer = Transformer.from_crs(gslc_aoi.rio.crs, crs, always_xy=True)
        xmin, ymin, xmax, ymax = gslc_aoi.rio.bounds()
        bbox = [[xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax]]
        # Generate the coordinates for the new interpolated grid, aligned to the target spacing.
        x, y = np.round(transformer.transform(bbox[0], bbox[1]))
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
        x1 = x1 - (x1 % abs(spacing[0])) + abs(spacing[0]) / 2
        x2 = x2 + (abs(spacing[0]) - x2 % abs(spacing[0])) - abs(spacing[0]) / 2
        y1 = y1 - (y1 % abs(spacing[1])) + abs(spacing[1]) / 2
        y2 = y2 + (abs(spacing[1]) - y2 % abs(spacing[1])) - abs(spacing[1]) / 2
        xlin = np.arange(x1, x2 + abs(spacing[0]), abs(spacing[0]))
        ylin = np.arange(y1, y2 + abs(spacing[1]), abs(spacing[1]))
        if spacing[0] < 0:
            xlin = xlin[::-1]
        if spacing[1] < 0:
            ylin = ylin[::-1]

        # Define the shape and coordinates for the empty output dataset.
        shape = cast(
            dict[str, list[int]],
            {
                k: (
                    (gslc_aoi.polarization.size, gslc_aoi.acquisition_date.size, ylin.size, xlin.size)
                    if k == "gamma0"
                    else (ylin.size, xlin.size)
                )
                for k in gslc_aoi.data_vars
            },
        )
        coords = {str(k): v.values for k, v in gslc_aoi.drop_vars("spatial_ref").coords.items() if k not in ["y", "x"]}
        coords["y"], coords["x"], center = ylin, xlin, compute_center(gslc, crs_string)

        # --- Initialize Output and Run Interpolation ---
        # Create an empty Zarr store for the interpolated result.
        gslc_empty = save_empty_dataset(
            data_vars=data_vars,
            dims=dims,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            coords=coords,
            crs=crs_string,
            skip_read=False,
            compute=False,
            attrs={"center": center, "interpolated": True},
            data_path=gslc_path,
            storage_options=get_storage_options(kwargs),
        )
        # Map the `interp_nrb` worker function over blocks of the empty output grid.
        map_blocks(
            interp_nrb,
            gslc_empty,
            args=[gslc_aoi, gslc_path, method, kernel, border, threshold],
            kwargs=kwargs,
            enable_dask=enable_dask,
            combine=False,
        )

        # Read the final interpolated result and add it to the output dictionary.
        gslc_interp = read(gslc_path, storage_options=get_storage_options(kwargs))
        gslc_utm[h_crs] = gslc_interp
    return (burst_id, gslc_utm)


def interp_nrb(
    gslc_out: xr.Dataset,
    gslc_in: xr.Dataset,
    data_path: str,
    method: str,
    kernel: dict,
    border: int,
    threshold: float,
    chunk: dict = {},
    **kwargs,
) -> None:
    """
    Interpolate a chunk of a GSLC dataset to a new grid (NRB).

    This is the core worker function for reprojecting a Geocoded SLC (GSLC) to a
    new target grid, typically a UTM zone for a Normalised Radar Backscatter (NRB)
    product. It transforms the coordinates, selects the necessary input data,
    applies smoothing, performs the interpolation, and writes the result to disk.

    Parameters
    ----------
    gslc_out : xr.Dataset
        An empty xarray.Dataset defining the target grid for the output chunk.
    gslc_in : xr.Dataset
        The full input GSLC dataset.
    data_path : str
        The path to the output Zarr store.
    method : str
        The interpolation method to use (e.g., 'cubic', 'linear').
    kernel : dict
        A dictionary with parameters for the Gaussian smoothing kernel.
    border : int
        A border in pixels to add when selecting input data to avoid
        interpolation edge effects.
    threshold : float
        A minimum value threshold to apply to the output, setting
        values below it to zero.
    chunk : dict, optional
        A dictionary describing the specific slice (region) of the output
        grid to be computed.

    Returns
    -------
    None
        None. The function writes the interpolated chunk directly to the Zarr store.
    """
    # --- Coordinate Transformation ---
    # Get the coordinates of the output grid chunk.
    y, x = gslc_out.y.values, gslc_out.x.values
    x_in, y_in = np.meshgrid(x, y)
    # Create a transformer to go from the output CRS to the input CRS.
    transformer = Transformer.from_crs(gslc_out.rio.crs, gslc_in.rio.crs, always_xy=True)
    # Transform the output grid coordinates to find where they land in the input grid.
    x_out, y_out = transformer.transform(x_in, y_in)

    # --- Select Input Data ---
    # Calculate the bounding box of the required input data, including a border.
    xmin, xmax = np.min(x_out), np.max(x_out)
    ymin, ymax = np.min(y_out), np.max(y_out)
    spacing = gslc_in.rio.resolution()
    xmin = xmin - border * abs(spacing[0])
    ymin = ymin - border * abs(spacing[1])
    xmax = xmax + border * abs(spacing[0])
    ymax = ymax + border * abs(spacing[1])

    # Create slices to select the data, handling positive/negative coordinate directions.
    xslice = slice(xmin, xmax) if spacing[0] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing[1] > 0 else slice(ymax, ymin)

    # Select the required portion of the input GSLC for the current date/polarization chunk.
    gslc_sel = gslc_in.sel(x=xslice, y=yslice).isel(
        polarization=chunk["polarization"], acquisition_date=chunk["acquisition_date"]
    )
    # Apply weighted Gaussian smoothing to the selected data.
    gslc_sel = smooth_nrb(gslc_sel, size=kernel["size"], sigma=kernel["sigma"])

    # Optimization: For chunks other than the first, we only need to process gamma0.
    if chunk["polarization"].start != 0 or chunk["acquisition_date"].start != 0:
        gslc_sel = gslc_sel.drop_vars([k for k in gslc_sel.data_vars if k != "gamma0"])

    # --- Perform Interpolation ---
    # Proceed only if the selected input data has a valid size.
    if (gslc_sel.x.size > 3) * (gslc_sel.y.size > 3):
        # Fill any NaNs with 0 before interpolation.
        gslc_sel = gslc_sel.where(np.isfinite(gslc_sel), 0.0)
        # Define the target coordinates for interpolation.
        xi = (y_out, x_out)
        # Call the generic dataset interpolation function.
        gslc_out = interp_dataset(data=gslc_sel, xi=xi, common_dims=("y", "x"), method=method, fill_value=0.0)
        # Apply a threshold to the output, setting small values to zero.
        gslc_out = gslc_out.where(gslc_out >= threshold, 0.0)
        # Binarize the interpolated layover and shadow masks.
        if "mask" in gslc_out:
            gslc_out["mask"] = np.round(gslc_out.mask).astype(int)
    else:
        # If not enough input data is available, fill the output chunk with zeros.
        for k in gslc_out.data_vars:
            gslc_out[k][...] = 0

    # --- Write Result ---
    # Write the computed chunk to the correct region in the output Zarr store.
    write(
        gslc_out.drop_vars(gslc_out.coords).drop_attrs(),
        data_path=normpath(data_path),
        region=chunk,
        consolidated=False,
        skip_read=True,
        storage_options=get_storage_options(kwargs),
    )
    return


def erode_fill_mask(mask: np.ndarray, erosion_step: int) -> np.ndarray:
    """
    Apply morphological operations to clean up a binary mask.

    This function first fills any holes within the mask, then erodes the edges
    to remove small, noisy regions, and finally fills any new holes that may have
    been created by the erosion process. This is useful for creating clean
    data extent masks for mosaicking.

    Parameters
    ----------
    mask : np.ndarray
        The input binary mask as a 2D numpy array.
    erosion_step : int
        The number of iterations for the binary erosion, which
        controls how much the mask is shrunk.

    Returns
    -------
    np.ndarray
        The cleaned binary mask.
    """
    # Fill enclosed holes in the initial mask.
    mask = scipy.ndimage.binary_fill_holes(mask)
    # Pad the mask to avoid edge effects during erosion.
    mask = np.pad(mask, erosion_step, mode="constant", constant_values=1)
    # Erode the mask to remove small features and shrink boundaries.
    mask = scipy.ndimage.binary_erosion(mask, iterations=erosion_step)
    # Remove the padding.
    mask = mask[erosion_step:-erosion_step, erosion_step:-erosion_step]
    # Fill any new holes that might have been created by the erosion.
    mask = scipy.ndimage.binary_fill_holes(mask)
    return mask


def compute_chunk_mosaic(
    nrb: xr.DataArray,
    gslcs: xr.DataTree,
    utm: str,
    spacing: dict,
    working_directory: str,
    erosion_step: int,
    kernel: dict,
    chunk: dict = {},
    **kwargs,
):
    """
    Compute a single chunk of a mosaic from multiple geocoded bursts.

    This is the core worker function for mosaicking. It takes a chunk of the
    final mosaic grid, finds all input bursts that overlap with it, and fills
    the chunk by selecting pixels from the burst closest to the center of its
    original swath.

    Parameters
    ----------
    nrb : xr.DataArray
        An empty xarray.DataArray representing the target chunk of the mosaic.
    gslcs : xr.DataTree
        The DataTree containing all the geocoded input bursts for a single UTM zone.
    utm : str
        The EPSG code of the UTM zone for the mosaic.
    spacing : dict
        A dictionary defining the pixel spacing of the mosaic.
    working_directory : str
        The root directory for outputs.
    erosion_step : int
        The number of erosion iterations for cleaning burst masks.
    kernel : dict
        A dictionary with parameters for Gaussian smoothing.
    chunk : dict, optional
        A dictionary describing the specific slice (region) of the mosaic
        grid to be computed.

    Returns
    -------
    None
        None. The function writes the computed mosaic chunk directly to a Zarr store.
    """
    # Flag to indicate if this chunk should compute all layers (gamma0, lia, anf, etc.).
    # Only the first chunk (in date and polarization) does this to avoid redundant computation.
    compute_all = (chunk["acquisition_date"].start == 0) and (chunk["polarization"].start == 0)

    # --- Define Chunk Geometry ---
    x, y = nrb.x.values, nrb.y.values
    # Define the exact boundaries of the current chunk at pixel edges.
    xmin, ymin = x.min() - abs(spacing["x"]) / 2.0, y.min() - abs(spacing["y"]) / 2.0
    xmax, ymax = x.max() + abs(spacing["x"]) / 2.0, y.max() + abs(spacing["y"]) / 2.0

    # Create slices for selecting data within the chunk's extent.
    xslice = slice(xmin, xmax) if spacing["x"] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing["y"] > 0 else slice(ymax, ymin)

    # --- Initialize Output Arrays ---
    multibursts = True if "bursts" in gslcs else False
    # Get the bounding boxes of all input bursts.
    coords = np.array([v.gamma0.rio.bounds() for v in (gslcs.bursts if multibursts else {"data": gslcs}).values()])

    shape = nrb.gamma0.shape
    # Array to hold the final gamma0 values.
    total_geo_ap = np.zeros(shape)
    # Array to keep track of the minimum distance to a burst center for each pixel.
    dist_min = np.ones((y.size, x.size)) * 1e9
    # Initialize arrays for other layers if this is the first chunk.
    if compute_all:
        total_lia = np.zeros((y.size, x.size))
        total_anf = np.zeros((y.size, x.size))
        total_nlooks = np.zeros((y.size, x.size))
        total_mask = np.zeros((y.size, x.size), dtype=int)
        total_index = np.full((y.size, x.size), fill_value=-1, dtype=int)

    # --- Main Mosaicking Loop ---
    # Iterate through each input burst.
    for en, ((gslc), c) in enumerate(zip((gslcs.bursts if multibursts else {"data": gslcs}).values(), coords)):
        # Coarse check: process only bursts that overlap with the current chunk's bounding box.
        if (c[0] <= xmax) and (c[1] <= ymax) and (c[2] >= xmin) and (c[3] >= ymin):
            # Get the center of the burst.
            xc, yc = gslc.attrs["center"]["x"], gslc.attrs["center"]["y"]
            # Select the portion of the burst that overlaps with the current chunk.
            gslc = (
                gslc.to_dataset()
                .sel(y=yslice, x=xslice)
                .drop_vars("spatial_ref")
                .isel(polarization=chunk["polarization"], acquisition_date=chunk["acquisition_date"])
            )
            xx, yy = gslc.x.values, gslc.y.values
            # Skip if the selected portion is empty.
            if xx.size == 0 or yy.size == 0:
                continue

            # If the burst was not interpolated, smooth it now.
            if gslc.attrs["interpolated"] is False:
                gslc = smooth_nrb(gslc, size=kernel["size"], sigma=kernel["sigma"])

            # --- Seamline Algorithm ---
            # Find the indices in the output chunk grid that correspond to the current burst's grid.
            x1, x2 = np.argmin(abs(x - xx[0])), np.argmin(abs(x - xx[-1]))
            y1, y2 = np.argmin(abs(y - yy[0])), np.argmin(abs(y - yy[-1]))
            xx, yy = np.meshgrid(xx, yy)
            # Create a clean data mask for the burst.
            mask = (gslc.gamma0 > 0).prod(dim=["polarization", "acquisition_date"]).values
            mask = erode_fill_mask(mask, erosion_step)
            # Calculate the distance of each pixel in the burst from its center.
            dist = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
            # Find the pixels where this burst is closer to its center than any previous burst.
            ind = np.where((dist < dist_min[y1 : y2 + 1, x1 : x2 + 1]) * mask)

            # If there are such pixels, update the mosaic with data from this burst.
            if ind[0].size > 0:
                g = gslc.gamma0.values
                # Map the local indices within the burst to global indices in the chunk.
                ind_y = ind[0] + y1
                ind_x = ind[1] + x1
                # Update the minimum distance array.
                dist_min[ind_y, ind_x] = dist[ind]
                # Copy the gamma0 values from this burst to the final mosaic.
                total_geo_ap[:, :, ind_y, ind_x] = g[:, :, ind[0], ind[1]]

                # If this is the first chunk, also copy all other data layers.
                if compute_all:
                    nl = gslc.nlooks.values
                    lia = gslc.lia.values
                    anf = gslc.anf.values
                    ml = gslc.mask.values
                    total_lia[ind_y, ind_x] = lia[ind]
                    total_anf[ind_y, ind_x] = anf[ind]
                    total_nlooks[ind_y, ind_x] = nl[ind]
                    total_mask[ind_y, ind_x] = ml[ind]
                    # Record which burst this pixel came from.
                    total_index[ind_y, ind_x] = en

    # --- Finalize and Write Chunk ---
    # After checking all bursts, construct the final result dataset for this chunk.
    if compute_all:
        result = xr.Dataset(
            data_vars={
                "gamma0": (["polarization", "acquisition_date", "y", "x"], total_geo_ap),
                "nlooks": (["y", "x"], total_nlooks),
                "lia": (["y", "x"], total_lia),
                "anf": (["y", "x"], total_anf),
                "mask": (["y", "x"], total_mask),
                "burst_id": (["y", "x"], total_index),
            },
        )
    else:
        # For other chunks, only write the gamma0 values.
        result = xr.Dataset(
            data_vars={
                "gamma0": (["polarization", "acquisition_date", "y", "x"], total_geo_ap),
            },
        )
    # Write the computed chunk to the correct region in the output Zarr store.
    write(
        result,
        normpath(f"{working_directory}/nrb_{utm}"),
        region=chunk,
        consolidated=False,
        mode="r+",
        skip_read=True,
        storage_options=get_storage_options(kwargs),
    )

    return


@timeit
def stack_mosaic(
    gslcs_tree: xr.Dataset,
    working_directory: str,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Orchestrate the mosaicking of GSLC data for multiple UTM zones in parallel.

    This function takes a DataTree where each top-level node represents a UTM zone
    containing one or more reprojected GSLC bursts. It maps the `mosaic` function
    across each UTM zone to generate a seamless NRB product for that zone in parallel.

    Parameters
    ----------
    gslcs_tree : xr.DataTree
        The input DataTree where each node is a UTM zone containing
        the reprojected GSLC bursts.
    working_directory : str
        The root directory for storing the final mosaicked
        NRB products.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
    enable_dask : bool, optional
        If True, use Dask for parallel computation.
    **kwargs
        Additional keyword arguments passed to the `map` function.

    Returns
    -------
    xr.DataTree
        A new DataTree where each node is a complete, mosaicked NRB
        product for a single UTM zone.
    """
    # --- Prepare Arguments for Parallel Mapping ---
    # Create a list of tuples, where each tuple contains the arguments for one call to the `mosaic` function.
    # The arguments are the UTM zone ID (u) and the DataTree of GSLCs for that zone (gslcs).
    args = [(u, gslcs) for u, gslcs in gslcs_tree.items()]
    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Mosaicking in Parallel ---
    # Map the `mosaic` worker function over the list of arguments.
    results = map(
        mosaic,
        args,
        kwargs=dict(chunks=chunks, working_directory=working_directory, enable_dask=enable_dask, **kwargs),
        enable_dask=enable_dask,
        combine=False,
    )
    if not isinstance(results, list):
        raise Exception(f"Expected list, got {type(results)}")

    # --- Assemble Final DataTree ---
    nrb_tree = xr.DataTree()
    # Iterate through the list of results returned by the workers.
    for u, nrb in results:
        # Reconstruct the final DataTree, with each node representing a mosaicked UTM zone.
        nrb_tree[u] = nrb
    return nrb_tree


def mosaic(utm: str, gslcs: xr.DataTree, chunks: dict, working_directory: str, enable_dask: bool = True, **kwargs):
    """
    Create a seamless mosaic from multiple GSLC bursts for a single UTM zone.

    This worker function takes all geocoded bursts for a specific UTM zone,
    calculates their total spatial extent, initializes an empty Zarr store for
    the final mosaic, and then orchestrates the block-wise computation of the
    mosaic by mapping the `compute_chunk_mosaic` function.

    Parameters
    ----------
    utm : str
        The EPSG code of the UTM zone for which the mosaic is being created.
    gslcs : xr.DataTree
        The DataTree containing all the reprojected GSLC bursts for this UTM zone.
    chunks : dict
        A dictionary defining the chunk sizes for the output mosaic.
    working_directory : str
        The root directory for storing the final mosaicked NRB product.
    enable_dask : bool, optional
        If True, use Dask for parallel sub-tasks.
    **kwargs
        Additional keyword arguments, such as `erosion_step` and kernel parameters.

    Returns
    -------
    tuple
        A tuple containing the UTM zone ID (str) and the final,
        computed mosaicked NRB dataset (xr.Dataset), read back from disk.
    """
    # --- Extract Parameters ---
    erosion_step = kwargs.get("erosion_step", 5)
    kernel = {"size": kwargs.get("kernel_size", 5), "sigma": kwargs.get("kernel_sigma", 0.5)}

    # --- Determine Mosaic Extent ---
    multibursts = True if "bursts" in gslcs else False
    # Get the bounding box of all input bursts to determine the total extent of the mosaic.
    coords = np.array([v.gamma0.rio.bounds() for v in (gslcs.bursts if multibursts else {"data": gslcs}).values()])
    xmin = coords[:, 0].min()
    ymin = coords[:, 1].min()
    xmax = coords[:, 2].max()
    ymax = coords[:, 3].max()

    # --- Define Output Grid ---
    # Get a sample DataArray to determine dimensions and CRS.
    for v in (gslcs.bursts if multibursts else {"data": gslcs}).values():
        g = v.gamma0
        break
    # Get the pixel spacing (resolution) from the sample.
    spacing = {"x": g.rio.resolution()[0], "y": g.rio.resolution()[1]}
    # Calculate the size of the output grid in pixels.
    size = ((xmax - xmin) // abs(spacing["x"]) + 1).astype(int)
    # Create the x-coordinate array for the mosaic, representing pixel centers.
    x = (
        np.array([xmin + spacing["x"] * i for i in range(size)])
        if spacing["x"] > 0
        else np.array([xmax + spacing["x"] * i for i in range(size)])
    )
    x = (x[:-1] + x[1:]) / 2.0
    # Create the y-coordinate array.
    size = ((ymax - ymin) // abs(spacing["y"]) + 1).astype(int)
    y = (
        np.array([ymin + spacing["y"] * i for i in range(size)])
        if spacing["y"] > 0
        else np.array([ymax + spacing["y"] * i for i in range(size)])
    )
    y = (y[:-1] + y[1:]) / 2.0
    # Define the final shape of the multi-dimensional gamma0 array.
    shape = [g.polarization.size, g.acquisition_date.size, y.size, x.size]

    # --- Prepare Metadata and Initialize Output ---
    crs: Union[str, CRS] = CRS(g.rio.crs).to_3d()
    if not isinstance(crs, CRS):
        raise Exception(f"Expected CRS, got {type(crs)}")
    horizontal_crs = str(crs.to_2d().to_epsg())
    vertical_crs = extract_vertical_crs(crs)
    vertical_crs = f"+{vertical_crs}" if crs.is_vertical else ""
    crs_string = f"{horizontal_crs}{vertical_crs}" if vertical_crs != "" else horizontal_crs
    # Create attributes for the output mosaic, including a legend mapping burst IDs to indices.
    attrs = {
        "legend": {en: name for en, name in enumerate((gslcs.bursts if multibursts else {"0": "data"}).keys())},
        "CRS_EPSG": crs_string,
        "CRS_WKT": crs.to_wkt(),
    }

    # Create an empty Zarr store on disk with the full structure of the final NRB product.
    nrb = save_empty_dataset(
        data_vars=["gamma0", "nlooks", "lia", "anf", "mask", "burst_id"],
        dims={
            "gamma0": ["polarization", "acquisition_date", "y", "x"],
            "nlooks": ["y", "x"],
            "lia": ["y", "x"],
            "anf": ["y", "x"],
            "mask": ["y", "x"],
            "burst_id": ["y", "x"],
        },
        shape={
            "gamma0": shape,
            "nlooks": shape[2:],
            "lia": shape[2:],
            "anf": shape[2:],
            "mask": shape[2:],
            "burst_id": shape[2:],
        },
        chunks=chunks,
        dtype={
            "gamma0": "float32",
            "nlooks": "float32",
            "lia": "float32",
            "anf": "float32",
            "mask": "int32",
            "burst_id": "int32",
        },
        attrs=attrs,
        crs=g.rio.crs,
        coords={"polarization": g.polarization.values, "acquisition_date": g.acquisition_date.values, "y": y, "x": x},
        data_path=normpath(f"{working_directory}/nrb_{utm}"),
        skip_read=False,
        **kwargs,
    )

    # --- Map the Mosaicking Worker Function ---
    # This call applies the `compute_chunk_mosaic` function to each block of the empty `nrb` grid.
    map_blocks(
        compute_chunk_mosaic,
        nrb,
        args=[gslcs, utm, spacing, working_directory, erosion_step, kernel],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
    )

    # --- Read and Return Final Mosaic ---
    # After all chunks have been computed and written, read the final consolidated mosaic from disk.
    nrb = read(
        normpath(f"{working_directory}/nrb_{utm}"),
        mask_and_scale=False,
        storage_options=get_storage_options(kwargs),
    )
    return (utm, nrb)


def map_tiles(
    utm: str,
    tile_name: str,
    tile_dataset: xr.Dataset,
    nrb: xr.Dataset,
    working_directory: str,
    enable_dask: bool,
    chunks: dict,
    **kwargs,
):
    spacing = {"x": (nrb.x[1] - nrb.x[0]).values, "y": (nrb.y[1] - nrb.y[0]).values}
    # Create a transformer from geographic coordinates (lat/lon) to this UTM zone.
    transformer = Transformer.from_crs("4326", utm, always_xy=True)
    m = mgrs.MGRS()
    # --- Sanity Check and Coordinate Transformation ---
    u, h, _, _ = m.MGRSToUTM(tile_name)
    if str(u) != utm[-2:]:
        raise Exception(f"Expected UTM zone {u}, got {utm[-2:]}")
    # Transform the geographic bounds of the MGRS tile into the UTM coordinate system.
    bounds = tile_dataset["bounds"].values
    x, y = np.round(transformer.transform(bounds[:, 0], bounds[:, 1]))
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # --- Select and Create Tile ---
    # Define slices to select the data for this tile from the mosaic.
    xslice = slice(xmin, xmax) if spacing["x"] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing["y"] > 0 else slice(ymax, ymin)
    # Select the data from the mosaic.
    nrb_sel = nrb.sel(y=yslice, x=xslice)

    nrb_tile = None
    # Skip tiles that contain no valid data.
    if np.all(nrb_sel.gamma0 == 0):
        logger.info(f"No valid point for {tile_name} tile")
    else:
        # --- Create Tile Grid and Copy Data ---
        # Define the precise grid for the new tile, ensuring alignment.
        off_x = np.array((nrb.x.min() - xmin) % abs(spacing["x"]))
        off_y = np.array((nrb.y.min() - ymin) % abs(spacing["y"]))
        xlin = np.arange(xmin, xmax, abs(spacing["x"])) + off_x
        ylin = np.arange(ymin, ymax, abs(spacing["y"])) + off_y
        if spacing["x"] < 0:
            xlin = xlin[::-1]
        if spacing["y"] < 0:
            ylin = ylin[::-1]

        # Create an empty dataset for the new tile.
        data_vars = ["burst_id", "lia", "anf", "mask", "gamma0", "nlooks"]
        dims = {k: (["polarization", "acquisition_date", "y", "x"] if k == "gamma0" else ["y", "x"]) for k in data_vars}
        sizes = dict(nrb.sizes)
        sizes["y"] = int(ylin.size)
        sizes["x"] = int(xlin.size)
        shape = {
            k: (
                [sizes["polarization"], sizes["acquisition_date"], sizes["y"], sizes["x"]]
                if k == "gamma0"
                else [sizes["y"], sizes["x"]]
            )
            for k in data_vars
        }
        dtype = {k: nrb[k].dtype.name for k in data_vars}
        coords = {
            "polarization": nrb.polarization.values,
            "acquisition_date": nrb.acquisition_date.values,
            "y": ylin,
            "x": xlin,
        }
        attrs = {attr_k: attr_v for attr_k, attr_v in nrb.attrs.items() if attr_k not in ["store_path", "obj_type"]}
        tile_path = normpath(f"{working_directory}/tiles/{tile_name}")
        nrb_tile = save_empty_dataset(
            data_vars=data_vars,
            dims=dims,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            coords=coords,
            attrs=attrs,
            crs=nrb.rio.crs,
            data_path=tile_path,
            storage_options=get_storage_options(kwargs),
            skip_read=False,
        )
        map_blocks(
            compute_tile,
            obj=nrb_tile,
            args=(nrb_sel, tile_path),
            kwargs=kwargs,
            combine=False,
            enable_dask=enable_dask,
        )
        nrb_tile = read(tile_path, as_type="xr.Dataset", storage_options=get_storage_options(kwargs))
        # Add the completed tile to the output DataTree.
    return (tile_name, nrb_tile)


def compute_tile(nrb_tile: xr.Dataset, nrb: xr.Dataset, tile_path: str, chunk: dict = {}, **kwargs):
    xlin = nrb_tile.x.values
    ylin = nrb_tile.y.values
    xmin, xmax = nrb_tile.x.min().values, nrb_tile.x.max().values
    ymin, ymax = nrb_tile.y.min().values, nrb_tile.y.max().values

    xslice = slice(xmin, xmax)
    yslice = slice(ymin, ymax)
    nrb = nrb.isel(acquisition_date=chunk["acquisition_date"], polarization=chunk["polarization"]).sel(
        y=yslice, x=xslice
    )
    for k in nrb_tile.data_vars:
        if k == "burst_id":
            nrb_tile[k][...] = -1
        else:
            nrb_tile[k][...] = 0.0

    # --- Copy Data from Mosaic to Tile ---
    # Find the indices in the new tile grid that correspond to the selected mosaic data.
    if nrb.x.size != 0 and nrb.y.size != 0:
        xx, yy = nrb.x.values, nrb.y.values
        x1, x2 = np.argmin(abs(xlin - xx[0])), np.argmin(abs(xlin - xx[-1]))
        y1, y2 = np.argmin(abs(ylin - yy[0])), np.argmin(abs(ylin - yy[-1]))
        # Copy the data for each variable from the selection to the new tile.
        for data_var in nrb_tile.data_vars:
            if data_var == "gamma0":
                nrb_tile[data_var][:, :, y1 : y2 + 1, x1 : x2 + 1] = nrb[data_var].values
            else:
                nrb_tile[data_var][y1 : y2 + 1, x1 : x2 + 1] = nrb[data_var].values

    write(
        nrb_tile.drop_vars(nrb_tile.coords).drop_attrs(),
        data_path=tile_path,
        region=chunk,
        consolidated=False,
        skip_read=True,
        storage_options=get_storage_options(kwargs),
    )
    return


@timeit
def nrb_mgrs_tiles(
    nrb_tree: xr.DataTree,
    mgrs_tiles: xr.DataTree,
    utm_zones: dict,
    epsg_to_tiles: dict,
    working_directory: str,
    enable_dask: bool = True,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
    **kwargs,
) -> xr.DataTree:
    """
    Tile the mosaicked NRB products into standard MGRS grid cells.

    This function takes the seamless NRB mosaics (one for each UTM zone) and
    clips them into individual tiles based on the MGRS grid definition.

    Parameters
    ----------
    nrb_tree : xr.DataTree
        A DataTree where each node is a complete, mosaicked NRB product
        for a single UTM zone.
    mgrs_tiles : xr.DataTree
        A DataTree defining the geographic boundaries of the target MGRS tiles.
    utm_zones : dict
        A dictionary mapping UTM zones to the MGRS tiles they contain.

    Returns
    -------
    xr.DataTree
        A new DataTree where each node is a final NRB product for a
        single MGRS tile.
    """

    # Iterate over each UTM zone that contains data.
    args = [
        (utm, k, v, nrb_tree[utm].to_dataset())
        for k, v in mgrs_tiles.items()
        for utm in utm_zones
        if k in epsg_to_tiles[utm]
    ]
    args = list(zip(*args))

    results = cast(
        list,
        map(
            map_tiles,
            args,
            kwargs=dict(
                working_directory=working_directory,
                enable_dask=enable_dask,
                chunks=chunks,
                **kwargs,
            ),
            enable_dask=enable_dask,
            combine=False,
        ),
    )

    nrb_tiles = xr.DataTree()
    for k, v in results:
        if v is not None:
            nrb_tiles[k] = v
    return nrb_tiles


def update_gslcs(gslcs: xr.DataTree, slcs: xr.DataTree, external_metadata: dict = {}):
    """
    Finalize the GSLC product by adding product-level metadata.

    This function iterates through the processed bursts (or single scene),
    restructures the data into the final product format (e.g., creating
    `measurements`, `conditions`, `quality` groups), and assigns a comprehensive
    set of attributes to conform to the GSLC product specification.

    Parameters
    ----------
    gslcs : xr.DataTree
        The DataTree containing all the processed geocoded bursts.
    external_metadata : dict, optional
        A dictionary of external metadata to be included in
        the final product attributes (e.g., processing center, history).

    Returns
    -------
    xr.DataTree
        A new DataTree, restructured and fully populated with final
        GSLC product metadata.
    """
    processing_start_time = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    bursts = xr.DataTree()
    multiburst = True if "bursts" in gslcs else False
    # --- Iterate Over Each Burst/Scene ---
    for k, gslc in (gslcs.bursts if multiburst else {"data": gslcs}).items():
        gstack = gslc.measurements.ds
        name = k if multiburst else "data"
        stac, processing_history = get_stac_and_history_gslc(
            name,
            gstack,
            slcs,
            processing_start_time=processing_start_time,
        )
        other_metadata = get_other_metadata_gslc(external_metadata)
        bursts[stac["id"]] = clean_gslc(gslc)
        bursts[stac["id"]].attrs = {
            "stac_discovery": stac,
            "processing_history": processing_history,
            "other_metadata": other_metadata,
        }
    stac_group, history_group = update_container(bursts, "GSLC", slcs.attrs["stac_discovery"]["id"])
    # Apply any final, global attribute fixes.
    fix_attributes(bursts, ATTRS_GEO)
    bursts.attrs = {"stac_discovery": stac_group, "processing_history": history_group}
    return bursts


def clean_gslc(gslc):
    # --- Restructure the DataTree for Final Product Format ---
    gslc_fix = xr.DataTree()
    aq_size = gslc.measurements.acquisition_date.size
    gslc_fix["measurements/gslc"] = (
        gslc.measurements.gstack.squeeze("acquisition_date") if aq_size == 1 else gslc.measurements.gstack
    )
    gslc_fix["measurements/phase_model"] = (
        gslc.measurements.phase_model.squeeze("acquisition_date") if aq_size == 1 else gslc.measurements.phase_model
    )
    gslc_fix["conditions/local_incidence_angle"] = gslc.conditions.lia
    gslc_fix["conditions/anf"] = gslc.conditions.anf
    gslc_fix["quality/mask"] = gslc.conditions.mask
    gslc_fix["quality/nlooks"] = gslc.conditions.nlooks

    return gslc_fix


def get_stac_and_history_gslc(burst_id, gslc, slcs, processing_start_time):
    multiburst = True if "bursts" in slcs else False
    attrs = copy.deepcopy((slcs.bursts[burst_id] if multiburst else slcs)).attrs
    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]

    crs = CRS(gslc.rio.crs).to_3d()
    horizontal_crs = str(crs.to_2d().to_epsg())
    vertical_crs = extract_vertical_crs(crs)
    vertical_crs = f"+{vertical_crs}" if crs.is_vertical else ""
    product_wkt = crs.to_wkt()
    product_epsg = f"{horizontal_crs}{vertical_crs}" if vertical_crs != "" else horizontal_crs
    lines_number, pixels_number = gslc.y.size, gslc.x.size
    pixel_spacing, line_spacing = gslc.rio.resolution()
    gsd = max(pixel_spacing, line_spacing)
    bbox = [round(i, 6) for i in gslc.rio.transform_bounds("EPSG:4326")]
    coordinates = [
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[3]],
        [bbox[2], bbox[1]],
        [bbox[0], bbox[1]],
    ]
    stac.pop("id")
    stac["properties"].pop("sat:anx_datetime", None)
    stac["properties"].pop("sar:pixel_spacing_azimuth", None)
    stac["properties"].pop("sar:pixel_spacing_range", None)
    stac["properties"].pop("view:incidence_angle", None)
    mode = stac["properties"]["sar:instrument_mode"]
    stac["stac_extensions"].append("https://stac-extensions.github.io/projection/v1.1.0/schema.json")
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coordinates]
    stac["properties"]["title"] = "Sentinel-1 ARD GSLC Product"
    stac["properties"]["created"] = processing_start_time
    stac["properties"]["processing:software"] = {"Sentinel-1 ARD CORE": f"{cop_ard_core.__version__}"}
    stac["properties"]["processing:version"] = f"{cop_ard_core.__version__}"
    stac["properties"]["processing:datetime"] = processing_start_time
    stac["properties"]["product:type"] = f"S01S{mode}GSL"
    stac["properties"]["gsd"] = gsd
    stac["properties"]["proj:epsg"] = product_epsg
    stac["properties"]["proj:wkt2"] = product_wkt
    stac["properties"]["proj:shape"] = [pixels_number, lines_number]
    stac["properties"]["proj:transform"] = list(gslc.rio.transform())[:6]
    stac["assets"] = {
        "measurements": {
            "href": "./measurements",
            "title": "GSLC Measurements",
            "description": "Zarr group containing the geocoded SAR image data",
            "type": "application/vnd+zarr",
            "roles": ["data"],
        },
        "conditions": {
            "href": "./conditions",
            "title": "GSLC Conditions",
            "description": "Zarr group containing measurement conditions",
            "type": "application/vnd+zarr",
            "roles": ["metadata"],
        },
        "quality": {
            "href": "./quality",
            "title": "GSLC Quality Metrics",
            "description": "Zarr group containing quality assessment and accuracy metrics",
            "type": "application/vnd+zarr",
            "roles": ["qa", "metadata"],
        },
    }
    id = get_id_product(stac, burst_id=burst_id)
    stac["id"] = id

    processing_history["Sentinel-1 ARD GSLC Product"] = copy.deepcopy(processing_history["Sentinel-1 ARD CSLC Product"])
    inputs = processing_history["Sentinel-1 ARD CSLC Product"][0]["outputs"]
    processing_history["Sentinel-1 ARD GSLC Product"][0]["inputs"] = inputs
    processing_history["Sentinel-1 ARD GSLC Product"][0]["outputs"] = id
    processing_history["Sentinel-1 ARD GSLC Product"][0]["time"] = processing_start_time
    processing_history["Sentinel-1 ARD GSLC Product"][0]["version"] = f"{cop_ard_core.__version__}"

    return stac, processing_history


def get_other_metadata_gslc(external_metadata):
    attrs = {
        "product_data_access": external_metadata["product_data_access"],
        "product_geometry": "projected",
        "product_pixel_coordinate_convention": external_metadata["product_pixel_coordinate_convention"],
        "noise_correction_flag": external_metadata["noise_removal"],
        "gslc_status": external_metadata["gslc_status"],
        "product_family_specification": external_metadata["product_family_specification"],
        "measurement_type": external_metadata["measurement_type"],
        "product_resampling_method": "sinc interpolator",
        "product_filtering": external_metadata["product_filtering"],
        "eastern_geometric_accuracy": external_metadata["eastern_geometric_accuracy"],
        "northern_geometric_accuracy": external_metadata["northern_geometric_accuracy"],
        "geometric_accuracy_radial_rmse": external_metadata["geometric_accuracy_radial_rmse"],
    }
    return attrs


def update_tiles(nrb_tiles: xr.DataTree, slcs, external_metadata: dict = {}):
    """
    Finalize the NRB tiles by adding product-level metadata.

    This function iterates through the processed MGRS tiles, restructures the
    data into the final product format (e.g., creating `measurements`,
    `conditions`, `quality` groups), and assigns a comprehensive set of
    attributes to conform to the NRB product specification.

    Parameters
    ----------
    nrb_tiles : xr.DataTree
        The DataTree containing all the final, tiled NRB data.
    external_metadata : dict, optional
        A dictionary of external metadata to be included in
        the final product attributes.

    Returns
    -------
    xr.DataTree
        A new DataTree, restructured and fully populated with final
        NRB product metadata.
    """
    processing_start_time = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    nrb_fix = xr.DataTree()
    for tile_name, tile_ in nrb_tiles.items():
        tile = tile_.to_dataset()

        stac, processing_history = get_stac_and_history_nrb(tile_name, tile, slcs, processing_start_time)
        other_metadata = get_other_metadata_nrb(tile_name, tile, external_metadata)
        nrb_fix[stac["id"]] = clean_nrb(tile)
        nrb_fix[stac["id"]].attrs = {
            "stac_discovery": stac,
            "processing_history": processing_history,
            "other_metadata": other_metadata,
        }
    stac_group, history_group = update_container(nrb_fix, "NRB", slcs.attrs["stac_discovery"]["id"])
    fix_attributes(nrb_fix, ATTRS_GEO)
    nrb_fix.attrs = {"stac_discovery": stac_group, "processing_history": history_group}
    return nrb_fix


def update_container(dt: xr.DataTree, dt_name, cslc_id):
    attrs = copy.deepcopy(next(iter(dt.children.values())).attrs)
    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]
    stac.pop("id", None)
    start = np.datetime64(stac["properties"]["start_datetime"])
    end = np.datetime64(stac["properties"]["end_datetime"])
    bbox = stac["bbox"]
    coords = stac["geometry"]["coordinates"][0][:-1]
    assets = {}
    for k, v in dt.items():
        stac_product = v.attrs["stac_discovery"]
        s = np.datetime64(stac_product["properties"]["start_datetime"])
        e = np.datetime64(stac_product["properties"]["end_datetime"])
        b = stac_product["bbox"]
        c = stac_product["geometry"]["coordinates"][0][:-1]
        bbox = [min(b[0], bbox[0]), min(b[1], bbox[1]), max(b[2], bbox[2]), max(b[3], bbox[3])]
        start = np.minimum(s, start)
        end = np.maximum(e, end)
        for i in range(4):
            i1, i2 = i // 2, i % 2
            lon_f = min if i1 == 0 else max
            lat_f = min if i2 == 0 else max
            coords[i] = [lon_f(coords[i][0], c[i][0]), lat_f(coords[i][1], c[i][1])]
        assets[k] = {"href": f"./{k}", "type": "application/vnd+zarr"}
    coords.append(coords[0])

    datetime = start + round((end.astype(int) - start.astype(int)) / 2)
    start = str(start) + "Z"
    end = str(end) + "Z"
    datetime = str(datetime) + "Z"
    stac["properties"].pop("proj:epsg", None)
    stac["properties"].pop("proj:wkt2", None)
    stac["properties"].pop("proj:shape", None)
    stac["properties"].pop("proj:transform", None)
    stac["properties"]["datetime"] = datetime
    stac["properties"]["start_datetime"] = start
    stac["properties"]["end_datetime"] = end
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coords]
    stac["properties"]["sar:beam_ids"] = beam_ids[stac["properties"]["sar:instrument_mode"]]
    stac["assets"] = assets
    stac["stac_extensions"] = [i for i in stac["stac_extensions"] if i.split("/")[-3] not in ["view", "projection"]]
    id = get_id_product(stac)
    stac["id"] = id
    processing_history[f"Sentinel-1 ARD {dt_name} Product"][0]["inputs"] = cslc_id
    processing_history[f"Sentinel-1 ARD {dt_name} Product"][0]["outputs"] = id

    return stac, processing_history


def get_other_metadata_nrb(tile_name, tile, external_metadata):
    attrs = {
        "product_data_access": external_metadata["product_data_access"],
        "product_geometry": "projected",
        "mgrs": {"UTM_zone": tile_name[:2], "latitude_band": tile_name[2], "grid_square": tile_name[3:]},
        "product_pixel_coordinate_convention": external_metadata["product_pixel_coordinate_convention"],
        "noise_correction_flag": external_metadata["noise_removal"],
        "nrb_status": external_metadata["nrb_status"],
        "product_family_specification": external_metadata["product_family_specification"],
        "measurement_type": "gamma0",
        "product_resampling_method": "area_projection",
        "product_filtering": external_metadata["product_filtering"],
        "eastern_geometric_accuracy": external_metadata["eastern_geometric_accuracy"],
        "northern_geometric_accuracy": external_metadata["northern_geometric_accuracy"],
        "geometric_accuracy_radial_rmse": external_metadata["geometric_accuracy_radial_rmse"],
        "burst_id_mapping": tile.attrs["legend"],
    }
    return attrs


def clean_nrb(tile):
    nrb_tile = xr.DataTree()
    nrb_tile["measurements/gamma0"] = (
        tile.gamma0.squeeze("acquisition_date") if tile.acquisition_date.size == 1 else tile.gamma0
    )
    nrb_tile["conditions/local_incidence_angle"] = tile.lia
    nrb_tile["conditions/anf"] = tile.anf
    nrb_tile["conditions/burst_index"] = tile.burst_id
    nrb_tile["quality/nlooks"] = tile.nlooks
    nrb_tile["quality/mask"] = tile.mask

    return nrb_tile


def get_stac_and_history_nrb(tile_name, tile, slcs, processing_start_time):
    multiburst = True if "bursts" in slcs else False
    # 1. Extract the raw attributes
    if multiburst:
        first_burst = next(iter(slcs.bursts.children.values()))
        raw_attrs = first_burst.attrs
    else:
        raw_attrs = slcs.attrs

    # 2. Safely copy the nested dictionary
    attrs = copy.deepcopy(raw_attrs)

    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]
    burst_idx = np.unique(tile.burst_id).astype(str)
    legend = tile.attrs["legend"]
    legend = [legend[i] for i in burst_idx if i in legend]
    beam_ids = []
    inputs = []
    counter = 0
    for bid, slc in (slcs.bursts if multiburst else {"0": slcs}).items():
        if bid not in legend:
            continue

        inputs.append(slc.attrs["processing_history"]["Sentinel-1 ARD CSLC Product"][0]["outputs"])

        stac_product = slc.attrs["stac_discovery"]
        beam_ids.extend(stac_product["properties"]["sar:beam_ids"])
        if counter == 0:
            start = np.datetime64(stac_product["properties"]["start_datetime"])
            end = np.datetime64(stac_product["properties"]["end_datetime"])
        else:
            s = np.datetime64(stac_product["properties"]["start_datetime"])
            e = np.datetime64(stac_product["properties"]["end_datetime"])
            start = min(start, s)
            end = max(end, e)
        counter += 1
    beam_ids = [str(i) for i in np.unique(beam_ids)]
    datetime_ = start + round((end.astype(int) - start.astype(int)) / 2)
    start = str(start) + "Z"
    end = str(end) + "Z"
    datetime_ = str(datetime_) + "Z"
    product_epsg, product_wkt = [tile.attrs[attr_k] for attr_k in ["CRS_EPSG", "CRS_WKT"]]
    lines_number, pixels_number = tile.y.size, tile.x.size
    pixel_spacing, line_spacing = tile.rio.resolution()
    gsd = max(pixel_spacing, line_spacing)
    bbox = [round(i, 6) for i in tile.rio.transform_bounds("EPSG:4326")]
    coordinates = [
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[3]],
        [bbox[2], bbox[1]],
        [bbox[0], bbox[1]],
    ]
    stac.pop("id", None)
    stac["properties"].pop("sat:anx_datetime", None)
    stac["properties"].pop("sar:pixel_spacing_azimuth", None)
    stac["properties"].pop("sar:pixel_spacing_range", None)
    stac["properties"].pop("view:incidence_angle", None)
    mode = stac["properties"]["sar:instrument_mode"]
    stac["stac_extensions"].append("https://stac-extensions.github.io/projection/v1.1.0/schema.json")
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coordinates]
    stac["properties"]["title"] = "Sentinel-1 ARD NRB Product"
    stac["properties"]["start_datetime"] = start
    stac["properties"]["end_datetime"] = end
    stac["properties"]["datetime"] = datetime_
    stac["properties"]["created"] = processing_start_time
    stac["properties"]["processing:software"] = {"Sentinel-1 ARD CORE": f"{cop_ard_core.__version__}"}
    stac["properties"]["processing:version"] = f"{cop_ard_core.__version__}"
    stac["properties"]["processing:datetime"] = processing_start_time
    stac["properties"]["product:type"] = f"S01S{mode}NRB"
    stac["properties"]["sar:beam_ids"] = beam_ids
    stac["properties"]["gsd"] = gsd
    stac["properties"]["proj:epsg"] = product_epsg
    stac["properties"]["proj:wkt2"] = product_wkt
    stac["properties"]["proj:shape"] = [pixels_number, lines_number]
    stac["properties"]["proj:transform"] = list(tile.rio.transform())[:6]
    stac["assets"] = {
        "measurements": {
            "href": "./measurements",
            "title": "NRB Measurements",
            "description": "Zarr group containing the intensity of the geocoded SAR image data",
            "type": "application/vnd+zarr",
            "roles": ["data"],
        },
        "conditions": {
            "href": "./conditions",
            "title": "NRB Conditions",
            "description": "Zarr group containing measurement conditions",
            "type": "application/vnd+zarr",
            "roles": ["metadata"],
        },
        "quality": {
            "href": "./quality",
            "title": "NRB Quality Metrics",
            "description": "Zarr group containing quality assessment and accuracy metrics",
            "type": "application/vnd+zarr",
            "roles": ["qa", "metadata"],
        },
    }
    id = get_id_product(stac, tile_name=tile_name)
    stac["id"] = id

    processing_history["Sentinel-1 ARD NRB Product"] = copy.deepcopy(processing_history["Sentinel-1 ARD CSLC Product"])
    processing_history["Sentinel-1 ARD CSLC Product"][0]["outputs"] = inputs
    processing_history["Sentinel-1 ARD NRB Product"][0]["inputs"] = inputs
    processing_history["Sentinel-1 ARD NRB Product"][0]["outputs"] = id
    processing_history["Sentinel-1 ARD NRB Product"][0]["time"] = processing_start_time
    processing_history["Sentinel-1 ARD NRB Product"][0]["version"] = f"{cop_ard_core.__version__}"

    return stac, processing_history


def fix_attributes(data: Union[xr.DataTree, xr.DataArray], attrs):
    """
    Recursively clean up and standardize attributes in a DataTree or DataArray.

    This utility function applies a standard set of attributes to dimension
    coordinates, removes unwanted attributes and encoding keys from datasets,
    and standardizes the attributes of data variables based on a provided template.
    It operates recursively on DataTree objects.

    Parameters
    ----------
    data : Union[xr.DataTree, xr.DataArray]
        The xarray.DataTree or xarray.DataArray to be cleaned.
    attrs
        A dictionary template where keys are dimension or variable names
        and values are the standard attributes to be assigned.

    Returns
    -------
    xr.DataTree or xr.DataArray
        The modified `data` object with cleaned and standardized attributes.
    """
    # --- Standardize Dimension Attributes ---
    # Iterate through all dimensions in the data object.
    for k in data.dims:
        if not isinstance(k, str):
            raise Exception(f"Expected str, got {type(k)}")
        # If the dimension name is in the attribute template, apply the standard attributes.
        if k in attrs:
            if not isinstance(k, str):
                raise Exception(f"Expected str, got {type(k)}")
            data[k].attrs = attrs[k]
            data[k].attrs.update({"short_name": k})

    # --- Clean Top-Level Attributes and Encoding ---
    # Remove any unwanted attributes from the top level of the object.
    for k in DEL_ATTRS:
        if k in data.attrs:
            del data.attrs[k]
    # Remove any unwanted keys from the object's encoding.
    for k in DEL_ENCODING:
        if k in data.encoding:
            del data.encoding[k]

    # --- Standardize Data Variable Attributes ---
    # If the object is a DataArray and its name is in the attribute template...
    if data.name in attrs and isinstance(data, xr.DataArray):
        if not isinstance(data.name, str):
            raise Exception(f"Expected str, got {type(data.name)}")
        # ...update its attributes with the standard template.
        data.attrs.update({**attrs[data.name], "short_name": data.name})

    # --- Recurse for DataTree ---
    # If the object is a DataTree, apply this function recursively to all its nodes.
    if isinstance(data, xr.DataTree):
        for k, v in data.items():
            fix_attributes(v, attrs)
    return data


DEL_ATTRS = ["store_path", "obj_type", "_FillValue", "etad_file", "coregistration_reference_date"]
DEL_ENCODING = ["coordinates"]
ATTRS_GEO = {
    "gamma0": {
        "description": "Radiometric Terrain Correction gamma naught backscatter linear scaling image",
        "long_name": "Gamma nought",
        "units": "m^2 / m^2",
    },
    "phase_model": {
        "description": "Projected topographic phase model, in rad, removed from the original interferometric phase."
        "The interferometric phase is evaluated wrt the reference orbit, used for co-registration process.",
        "long_name": "Interferometric phase model",
        "units": "rad",
    },
    "polarization": {
        "description": "acquired polarizations",
        "long_name": "Polarization",
        "units": "n/a",
    },
    "acquisition_date": {
        "description": "Image acquisition date, espressed as YYYY-MM-DD, data type is string",
        "long_name": "Acquisition date",
        "units": "n/a",
    },
    "y": {
        "description": "UTM y coordinates",
        "long_name": "North coordinates",
        "units": "m",
    },
    "x": {
        "description": "UTM x coordinates",
        "long_name": "East coordinates",
        "units": "m",
    },
    "gslc": {
        "description": "Projected Co-registered SLC, with module equal to gamma0 or beta0 coefficients and\
                            phase equal to the projected CSLC phase",
        "long_name": "Geocoded Single Look Complex",
        "units": "n/a",
    },
    "anf": {
        "description": "Optional area normalization factor, used to obtain the gamma0 coefficient",
        "long_name": "terrain simulated radar backscattering",
        "units": "m^2",
    },
    "local_incidence_angle": {
        "description": "DEM-based local Incidence angle image",
        "long_name": "local incidence angle",
        "units": "deg",
    },
    "mask": {
        "description": "layover=1, shadow=2, river=3, lake=4, ocean=5 and valid_point=0",
        "long_name": "mask of valid points",
        "units": "n/a",
    },
    "nlooks": {
        "description": "Number of averaged radar cells",
        "long_name": "Number of looks",
        "units": "n/a",
    },
    "nrb": {
        "description": "Terrain corrected gamma nought",
        "long_name": "terrain corrected gamma nought",
        "units": "m^2 / m^2",
    },
}

beam_ids = {
    "IW": ["IW1", "IW2", "IW3"],  # Interferometric Wide (3 sub-swaths)
    "EW": ["EW1", "EW2", "EW3", "EW4", "EW5"],  # Extra Wide (5 sub-swaths)
    "SM": ["S1", "S2", "S3", "S4", "S5", "S6"],  # Stripmap (6 overlapping beams)
}
