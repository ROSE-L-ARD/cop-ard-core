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
}

beam_ids = {
    "IW": ["IW1", "IW2", "IW3"],  # Interferometric Wide (3 sub-swaths)
    "EW": ["EW1", "EW2", "EW3", "EW4", "EW5"],  # Extra Wide (5 sub-swaths)
    "SM": ["S1", "S2", "S3", "S4", "S5", "S6"],  # Stripmap (6 overlapping beams)
}
