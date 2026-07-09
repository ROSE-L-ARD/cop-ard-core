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
import os
from typing import Any, Dict, Mapping, Sequence, cast

import dask.distributed
import defusedxml.ElementTree as ET
import mgrs
import numpy as np
import pyproj
import rasterio
import rioxarray
import xarray as xr
from pyproj import CRS, Transformer, datadir
from pyproj.transformer import TransformerGroup

from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.griddata import interp_dataset
from cop_ard_core.computing.io import exists, get_storage_options, normpath, read, save_empty_dataset, write
from cop_ard_core.computing.logger import logger, timeit

rioxarray.__version__


def extract_vertical_crs(crs: CRS) -> str:
    if not isinstance(crs, CRS):
        raise ValueError(f"Expected type CRS, got {type(crs)}")
    vcrs = crs.sub_crs_list[1].to_string().split(":")[1] if crs.is_vertical else ""
    return vcrs


def validate_vertical_transformation(data: xr.Dataset | xr.DataArray, outproj: str) -> None:
    """
    Validates that the PROJ pipeline successfully applies a vertical datum shift.

    This extracts the center coordinates of the dataset and runs a mock
    transformation to ensure the Z value physically changes, proving the
    geoid grid is actively being used.
    """
    CRS_EPSG = f"EPSG:{outproj}" if not str(outproj).startswith("EPSG:") else outproj
    in_crs, out_crs = CRS(data.rio.crs).to_3d(), CRS(CRS_EPSG).to_3d()

    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)

    # Get the center of the dataset to hit a valid geographic area
    bounds = data.rio.bounds()
    test_x = (bounds[0] + bounds[2]) / 2.0
    test_y = (bounds[1] + bounds[3]) / 2.0
    test_z = 50.0  # Arbitrary test elevation

    # errcheck=True forces a crash if PROJ tries to return 'inf'
    new_x, new_y, new_z = transformer.transform(test_x, test_y, test_z, errcheck=True)

    # If Z didn't change (within a tiny floating-point margin), the grid shift failed
    if abs(new_z - test_z) < 1e-5:
        error_msg = (
            f"Transformation validation failed! Z coordinate did not change. "
            f"Input Z: {test_z}, Output Z: {new_z}. The geoid grid is likely missing or ignored."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return


def check_available_transformation(data: xr.Dataset | xr.DataArray, outproj: str) -> None:
    """
    Check if a valid PROJ transformation is available between two CRS.

    This function inspects the PROJ transformation pipeline between the input data's
    CRS and a target CRS. It logs the best available transformation and, if
    the best one is not available, it attempts to download the required grid files.
    Finally, it triggers a mathematical validation of the vertical shift.

    Parameters
    ----------
    data
        The input xarray object with a defined CRS in its `rio` accessor.
    outproj
        The EPSG code of the target Coordinate Reference System.

    Returns
    -------
    None
        The function will raise an exception if a transformation grid is
        missing and cannot be downloaded, or if the math validation fails.
    """
    # Ensure the output projection is in the standard "EPSG:XXXX" format.
    CRS_EPSG = f"EPSG:{outproj}" if not str(outproj).startswith("EPSG:") else outproj
    if not isinstance(CRS_EPSG, str):
        raise Exception(f"Expected str, got {type(CRS_EPSG)}")
    # Create 3D CRS objects for both input and output to ensure height is handled.
    in_crs, out_crs = CRS(data.rio.crs).to_3d(), CRS(CRS_EPSG).to_3d()
    # Create a TransformerGroup to inspect the available transformation pipelines.
    tg = TransformerGroup(in_crs, out_crs)
    # If the best transformation requires a grid file that is not present, try to download it.
    if tg.best_available is False:
        # Safely extract the first path if there are multiple directories
        pyproj_dir = datadir.get_data_dir()
        try:
            tg.download_grids(directory=pyproj_dir, verbose=True)
            datadir.set_data_dir(datadir.get_data_dir())
            logger.info("PROJ cache successfully reloaded mid-script.")
        except Exception as e:
            logger.critical(f"Type of error is: {e.__class__.__name__}")
            # Log information about the transformation.
            logger.critical(f"Input crs : {in_crs.to_string()}")
            logger.critical(f"Output crs: {out_crs.to_string()}")
            logger.error(f"Best available transformation: {tg.best_available}")
            raise Exception(f"Missing transformation file in {pyproj_dir}") from e

    in_vertical_crs = extract_vertical_crs(in_crs)
    out_vertical_crs = extract_vertical_crs(out_crs)
    # Trigger the separate validation function to confirm the math works
    if in_vertical_crs != out_vertical_crs:
        validate_vertical_transformation(data, outproj)
    return


def get_default_epsg(dem: xr.DataArray | xr.Dataset) -> str:
    """
    Determine the default UTM or UPS projection for a given geographic area.

    This function calculates the center latitude and longitude of the input DEM and
    determines the most appropriate Universal Transverse Mercator (UTM) or

    Universal Polar Stereographic (UPS) projection for that location.

    Parameters
    ----------
    dem : xarray.DataArray or xarray.Dataset
        An xarray DataArray or Dataset with a defined CRS.

    Returns
    -------
    str
        A string representing the EPSG code for the determined UTM or UPS zone.
    """
    # Create CRS objects for the source and target (WGS84 lat/lon).
    source = CRS.from_user_input(dem.rio.crs)
    target = CRS.from_epsg(4326)

    # Create a transformer to convert from the source CRS to lat/lon.
    transformer = Transformer.from_crs(source, target, always_xy=True)

    # Transform the center coordinates of the DEM to get its central longitude and latitude.
    lon, lat = transformer.transform(dem.x.mean(), dem.y.mean())

    # --- Determine the appropriate projection based on latitude ---
    if lat >= 84:
        # Use UPS North for the Arctic region.
        epsg = "EPSG:32661"
    elif lat <= -80:
        # Use UPS South for the Antarctic region.
        epsg = "EPSG:32761"
    else:
        # For mid-latitudes, determine the UTM zone.
        m = mgrs.MGRS()
        mgrs_tile = m.toMGRS(lat, lon)
        # Convert the MGRS tile to UTM to get the zone number and hemisphere.
        u, h, _, _ = m.MGRSToUTM(mgrs_tile)
        # Construct the UTM EPSG code (326xx for North, 327xx for South).
        epsg = f"EPSG:32{6 if h == 'N' else 7}{u}"
    return epsg


def download_copdem30m(minx, miny, maxx, maxy):
    """
    Download a subset of the Copernicus GLO-30 DEM from a cloud-based Zarr store.

    Parameters
    ----------
    minx
        The minimum longitude of the bounding box.
    miny
        The minimum latitude of the bounding box.
    maxx
        The maximum longitude of the bounding box.
    maxy
        The maximum latitude of the bounding box.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the clipped DEM data.
    """
    # Open the remote Copernicus DEM Zarr store.
    dem = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
        storage_options={"client_kwargs": {"trust_env": True}},
        chunks={},
        engine="zarr",
        decode_coords="all",
        mask_and_scale=False,
    )
    if not isinstance(dem.rio.crs, rasterio.crs.CRS):
        raise Exception(f"Expected rasterio.crs.CRS, got {type(dem.rio.crs)}")
    # Clip the global DEM to the specified bounding box.
    dem = dem.rio.clip_box(minx, miny, maxx, maxy, crs="EPSG:4326")
    # Rename coordinates to the standard 'y' and 'x'.
    dem = dem.rename({"lat": "y", "lon": "x"})
    return dem


def import_dem(
    bbox: Sequence[float],
    working_dir: str | None = None,
    dem_type: str = "Copernicus",
    chunks: Mapping[Any, int] | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Import and preprocess a Digital Elevation Model (DEM).

    This function handles the import of a DEM from a specified source (e.g.,
    Copernicus GLO-30, or a user-provided file path). It clips the DEM to a
    given bounding box, applies chunking, handles NoData values, and saves
    the result to a Zarr store.
    """
    border = 0.05

    # Expand the bounding box slightly to avoid edge effects during reprojection.
    minx, miny, maxx, maxy = bbox
    minx = minx - border
    miny = miny - border
    maxx = maxx + border
    maxy = maxy + border

    # --- DEM Source Handling ---
    if dem_type == "Copernicus":
        # Download the Copernicus DEM for the specified bounding box.
        dem = download_copdem30m(minx, miny, maxx, maxy)
    else:
        # Open a user-specified DEM from a local or remote path.
        dem = xr.open_dataset(
            dem_type,
            chunks={},
            engine="zarr",
            decode_coords="all",
            storage_options=get_storage_options(kwargs, key="storage_options_dem"),
        )
        if not dem.rio.crs:
            raise Exception(f"Input DEM projection not defined {dem} {dem.rio.crs}")

        # Standardize coordinate names if the CRS is geographic.
        if dem.rio.crs.is_geographic:
            if all([i in dem.coords for i in ["lon", "lat"]]):
                dem = dem.rename({"lat": "y", "lon": "x"})

        # Preserve original chunking for efficient clipping.
        chunksize = {e: dem.chunksizes[e][0] for e in dem.coords if e in ["x", "y", "lon", "lat"]}

        # Clip the DEM to the area of interest.
        dem = dem.rio.clip_box(minx, miny, maxx, maxy, crs="EPSG:4326")

        # Re-apply chunking after clipping.
        dem = dem.chunk(chunksize)

    # Apply user-specified chunking if provided.
    if chunks:
        dem = dem.chunk(chunks)
        # Drop encoding to prevent issues when writing re-chunked data to Zarr
        dem = dem.drop_encoding()

    # Convert the Dataset to a DataArray named 'z'.
    dem = dem.to_array(name="z").squeeze(drop=True)

    # --- NaN Handling ---
    # Ensure spatial dimensions are strictly recognized
    dem.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Write the processed DEM to a Zarr store in the working directory.
    if (dem_type == "Copernicus") or dem_type.startswith("https://"):
        dem = write(dem, normpath(f"{working_dir}/dem"), storage_options=get_storage_options(kwargs))

    return dem


def compute_shape_and_off(
    data: xr.DataArray | xr.Dataset, spacing: Dict[str, Any], out_crs: str | int | None = None
) -> tuple[Dict[str, int], Dict[str, float]]:
    """
    Compute the output grid shape and origin offset for a reprojection.

    This function calculates the dimensions (shape) and top-left corner
    coordinates (offset) of a new grid in a target CRS that fully encompasses
    the extent of the input DEM. The new grid is aligned to multiples of the
    specified spacing.

    Parameters
    ----------
    dem
        The input Digital Elevation Model.
    out_crs
        The EPSG code of the target Coordinate Reference System.
    spacing : dict
        A dictionary defining the grid spacing (resolution) in the
        target CRS units for the 'x' and 'y' dimensions.

    Returns
    -------
    shape : dict
        The number of grid cells in the 'x' and 'y' dimensions.
    off : dict
        The coordinates of the grid origin (top-left corner).
    """
    # Use absolute spacing for calculations.
    abs_spacing = {k: abs(v) for k, v in spacing.items()}
    # Transform the bounds of the input DEM to the output CRS.
    if out_crs:
        xmin, ymin, xmax, ymax = data.rio.transform_bounds(out_crs, recalc=True)
    else:
        xmin, ymin, xmax, ymax = data.x.min().values, data.y.min().values, data.x.max().values, data.y.max().values

    # --- Align Grid to Spacing ---
    # Adjust the bounds to be integer multiples of the spacing to ensure grid alignment.
    xmin, xmax = (
        np.ceil(xmin / abs_spacing["x"]) * abs_spacing["x"],
        np.floor(xmax / abs_spacing["x"]) * abs_spacing["x"],
    )
    ymin, ymax = (
        np.ceil(ymin / abs_spacing["y"]) * abs_spacing["y"],
        np.floor(ymax / abs_spacing["y"]) * abs_spacing["y"],
    )

    # --- Calculate Shape and Offset ---
    # Calculate the number of pixels required in each dimension.
    shape = {
        "x": (np.round(abs(xmax - xmin) / abs_spacing["x"]) + 1).astype(int),
        "y": (np.round(abs(ymax - ymin) / abs_spacing["y"]) + 1).astype(int),
    }
    # Determine the origin offset based on the direction of the spacing (positive/negative).
    off = {"x": xmin if spacing["x"] > 0 else xmax, "y": ymin if spacing["y"] > 0 else ymax}
    return shape, off


@timeit
def dem_reproject(
    dem: xr.DataArray,
    spacing: Dict[str, Any],
    outproj: str | int,
    data_path: str,
    chunks: dict = {"x": 2048, "y": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Reproject a Digital Elevation Model (DEM) to a specified CRS.

    This function orchestrates the reprojection of a DEM. It defines the output
    grid, initializes an empty Zarr store for the result, and then maps the
    `interp_dem` function over blocks to perform the interpolation in parallel.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    spacing
        The desired spatial resolution (spacing) in the output CRS units.
    outproj
        The target EPSG code for the output DEM.
    data_path
        The path where the reprojected DEM Zarr store will be created.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    enable_dask : bool
        If True, use Dask for parallel computation.
    **kwargs
        Additional keyword arguments, such as `method` for interpolation.

    Returns
    -------
    xarray.Dataset
        The reprojected DEM as an xarray.Dataset, read back from the Zarr store.
    """
    method = kwargs.pop("method", "cubic")
    # Ensure the input DEM has 'x' and 'y' coordinates.
    if (all([i in dem.coords for i in ["x", "y"]])) is False:
        raise Exception('DEM dataset coordinates must contain "x" and "y"')
    dem_ds = dem.to_dataset(name="z")

    # --- CRS Handling ---
    CRS_EPSG = f"EPSG:{outproj}" if not str(outproj).startswith("EPSG:") else outproj
    if not isinstance(CRS_EPSG, str):
        raise Exception(f"Expected str, got {type(CRS_EPSG)}")
    # Create 3D CRS objects for robust transformation.
    in_crs, out_crs = CRS(dem.rio.crs).to_3d(), CRS(CRS_EPSG).to_3d()
    dem_ds.rio.write_crs(in_crs, inplace=True)
    # Construct a composite CRS that preserves the vertical datum.
    vertical_crs = extract_vertical_crs(in_crs)
    crs_2d = out_crs.to_2d().to_string()
    MID_EPSG = f"{crs_2d}+{vertical_crs}" if vertical_crs != "" else crs_2d
    mid_crs = CRS(MID_EPSG).to_3d()

    # --- Define Output Grid ---
    shape, off = compute_shape_and_off(dem, spacing, CRS_EPSG)
    # Create the coordinate arrays for the output grid.
    x_lin, y_lin = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output ---
    # Create an empty Zarr store for the reprojected DEM.
    dem_out = save_empty_dataset(
        data_vars=["z"],
        dims={"z": ["y", "x"]},
        shape={"z": [y_lin.size, x_lin.size]},
        dtype={"z": dem_ds.z.dtype},
        chunks=chunks,
        crs=out_crs,
        coords={"y": y_lin, "x": x_lin},
        data_path=normpath(f"{data_path}/dem_resampled"),
        **kwargs,
    )

    # --- Run Reprojection in Parallel ---
    # Map the interpolation function over blocks of the output grid.
    save_options = {
        "data_path": normpath(f"{data_path}/dem_resampled"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs),
        },
    }
    map_blocks(
        interp_dem,
        dem_out.z,
        args=[dem_ds.z, mid_crs, method],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
        store_mapping={0: save_options},
    )
    # Read and return the final reprojected dataset.
    return read(
        normpath(f"{data_path}/dem_resampled"), as_type="xr.Dataset", storage_options=get_storage_options(kwargs)
    )


def interp_dem(
    dem_out: xr.DataArray,
    dem_in: xr.DataArray,
    mid_crs: CRS,
    method: str = "cubic",
    border: int = 10,
    **kwargs,
):
    """
    Interpolate a chunk of the input DEM to the output grid.

    This is the core worker function for DEM reprojection. It takes a chunk of the
    output grid, transforms its coordinates back to the input DEM's CRS, reads the
    necessary portion of the input DEM, and performs the interpolation.

    Parameters
    ----------
    dem_out
        The output grid chunk with coordinates in the target CRS.
    dem_in
        The full input DEM dataset.
    mid_crs
        The composite intermediate CRS.
    data_path
        The path where the result will be written.
    method
        The interpolation method to use (e.g., 'cubic', 'linear').
    border
        A border in pixels to add when reading the input DEM to avoid
        interpolation edge effects.
    chunk : dict
        A dictionary describing the specific output slice (region) to process.

    Returns
    -------
    None
        None. The result is written directly to the output Zarr store.
    """
    # --- Coordinate Transformation ---
    y, x = dem_out.y.values, dem_out.x.values
    x_in, y_in = np.meshgrid(x, y)
    # Create a transformer to go from the output CRS to the input CRS.
    transformer = Transformer.from_crs(dem_out.rio.crs, dem_in.rio.crs, always_xy=True)
    # Transform the output grid coordinates to find where they land in the input grid.
    x_out, y_out = transformer.transform(x_in, y_in)

    # --- Select Input Data ---
    # Calculate the bounding box of the required input data, including a border.
    xmin, xmax = np.min(x_out), np.max(x_out)
    ymin, ymax = np.min(y_out), np.max(y_out)
    spacing = dem_in.rio.resolution(recalc=True)
    xmin = xmin - border * abs(spacing[0])
    ymin = ymin - border * abs(spacing[1])
    xmax = xmax + border * abs(spacing[0])
    ymax = ymax + border * abs(spacing[1])
    # Create slices to select the data, handling positive/negative coordinate directions.
    xslice = slice(xmin, xmax) if spacing[0] > 0 else slice(xmax, xmin)
    yslice = slice(ymin, ymax) if spacing[1] > 0 else slice(ymax, ymin)

    # Read the selected portion of the input DEM.
    dem_sel = dem_in.sel(x=xslice, y=yslice).fillna(0)

    # --- Interpolation ---
    if (dem_sel.x.size > 3) and (dem_sel.y.size > 3):
        # Perform the interpolation if enough input data is available.
        dem_interp = interp_dataset(
            data=dem_sel.to_dataset(name="z"),
            xi=(y_out, x_out),  # The coordinates to interpolate at.
            common_dims=("y", "x"),
            method=method,
            fill_value=float(dem_sel.mean(skipna=True).values),  # Use mean for fill value.
        )
        # Apply the vertical datum transformation.
        height_transform = Transformer.from_crs(mid_crs, dem_out.rio.crs, always_xy=True)
        _, _, dem_interp.z[...] = height_transform.transform(x_in, y_in, dem_interp.z.values)
    else:
        # If not enough data, fill the output chunk with zeros.
        dem_interp = xr.DataArray(np.zeros((y.size, x.size)), dims=["y", "x"]).to_dataset(name="z")

    return dem_interp


@timeit
def stack_dem(
    dem: xr.Dataset,
    slcs: xr.DataTree,
    multiburst: bool,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    recompute: bool = False,
    **kwargs,
):
    """
    Reproject a DEM for each burst or scene in a stack.

    This function orchestrates the reprojection of a DEM for an entire stack.
    It maps the `single_dem` function over each item in the `slcs` DataTree,
    processing each one in parallel, and then assembles the results into a new
    DataTree.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    slcs : xarray.DataTree
        An xarray.DataTree containing the SAR data, used to define the
        geographic extent for each burst.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    spacing
        The desired spatial resolution of the output DEMs.
    working_directory
        The root directory for all processing outputs.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    **kwargs
        Additional keyword arguments passed to the reprojection function.

    Returns
    -------
    xarray.DataTree
        An xarray.DataTree containing the reprojected DEM for each burst or scene.
    """
    # --- Determine Target CRS ---
    if crs is None:
        # If no CRS is specified, determine the default UTM/UPS zone.
        epsg = get_default_epsg(dem)
    else:
        epsg = crs
    # Check if the required transformation is available.
    check_available_transformation(dem, epsg)

    # --- Prepare Arguments and Run in Parallel ---
    # Create a list of arguments for each burst/scene to be processed.
    args = [
        (k, [o.measurements.slc.attrs for o in s.orig.values()], multiburst)
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
    ]
    args = list(zip(*args))
    results = cast(
        list,
        map(
            single_dem,
            args,
            # Pass common arguments via kwargs.
            kwargs=dict(
                dem=dem,
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                chunks=chunks,
                crs=crs,
                enable_dask=enable_dask,
                recompute=recompute,
                **kwargs,
            ),
            combine=False,
            enable_dask=enable_dask,
            Plugin=GeoEnvPlugin(in_proj=dem.rio.crs, out_proj=epsg),
        ),
    )

    # --- Assemble Results ---
    dem_tree = xr.DataTree()
    for k, d in results:
        # Reconstruct the DataTree structure from the list of results.
        if multiburst:
            dem_tree[f"bursts/{k}"] = d
        else:
            dem_tree = xr.DataTree(d)
    return dem_tree


def single_dem(
    burst_id: str,
    slc_attrs: list[dict],
    multiburst: bool,
    dem: xr.Dataset,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    border: float = 10.0 / 3600.0,  # Approx. 300m border in degrees
    recompute=False,
    **kwargs,
):
    """
    Reproject a DEM for a single burst or scene.

    This worker function calculates the geographic bounding box of a single SAR
    burst, clips the input DEM to that area, and then calls `dem_reproject` to
    perform the reprojection.

    Parameters
    ----------
    burst_id
        The identifier for the burst or scene.
    slc
        The DataTree for the specific burst, used to find its footprint.
    multiburst : bool
        A boolean flag for multiburst mode.
    dem
        The full input Digital Elevation Model.
    spacing
        The desired spatial resolution of the output DEM.
    working_directory
        The root directory for outputs.
    chunks : dict
        A dictionary defining chunk sizes for processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    enable_dask : bool
        If True, use Dask for parallel computation.
    border
        A border in degrees to add to the burst's bounding box.
    **kwargs
        Additional keyword arguments for the reprojection function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the reprojected DEM Dataset.
    """
    # Define the output path for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory_tmp}/{path}")
    simulation_path = f"{working_directory_static}/processed_data/{path}/simulation.ok"
    storage_options_static = get_storage_options(kwargs, key="storage_options_static")
    # --- Check for Existing Results ---
    # Decide whether to run the computation or use a cached result.
    if exists(simulation_path, storage_options=storage_options_static) is False or recompute:
        # --- Calculate Bounding Box ---
        lat = []
        lon = []
        # Collect the corner coordinates from all original images in this burst.
        for attrs in slc_attrs:
            annot = attrs[list(attrs)[0]]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])
        # Determine the min/max lat/lon and add a border.
        lat_min, lat_max = min(lat) - border, max(lat) + border
        lon_min, lon_max = min(lon) - border, max(lon) + border

        # --- Clip and Reproject DEM ---
        # Clip the global DEM to the specific area of this burst.
        dem_aoi = dem.rio.clip_box(lon_min, lat_min, lon_max, lat_max, crs="EPSG:4326").chunk(chunks)
        # Determine the target CRS if not provided.
        if crs is None:
            crs = get_default_epsg(dem_aoi)
        # Call the main reprojection function.
        dem_resampled = dem_reproject(
            dem_aoi, spacing, crs, chunks=chunks, data_path=data_path, enable_dask=enable_dask, **kwargs
        )
    else:
        logger.info(f"Found existing DEM file for {burst_id}")
        dem_resampled = read(
            normpath(f"{working_directory_static}/{path}/simulation"),
            as_type="xr.Dataset",
            storage_options=storage_options_static,
        ).h.to_dataset(name="z")

    return (burst_id, dem_resampled)


def parse_kml(path: str, bbox: list) -> xr.DataTree:
    """
    Parse a KML file containing MGRS tile definitions.

    This function reads a KML file that defines the boundaries of MGRS tiles,
    filters the tiles that intersect with a given bounding box, and returns
    their information in a DataTree.

    Parameters
    ----------
    path : str
        The file path to the MGRS KML file.
    bbox : list
        The geographic bounding box [min_lon, min_lat, max_lon, max_lat]
        to filter tiles.

    Returns
    -------
    xarray.DataTree
        An xarray.DataTree where each node is an MGRS tile that intersects the
        bounding box, containing the tile's point and boundary coordinates.
    """
    # Parse the XML structure of the KML file.
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = {"": "http://www.opengis.net/kml/2.2"}

    # Find all Placemark names and geometries.
    names = root.findall(".//Placemark/name", namespace)
    geometries = root.findall(".//Placemark/MultiGeometry", namespace)

    mgrs_tiles = xr.DataTree()
    for name, geometry in zip(names, geometries):
        n = name.text
        # Extract the center point and boundary coordinates for each tile.
        point = geometry.find("Point/coordinates", namespace)
        point = np.array(point.text.replace("\t", "").strip().split(",")[:-1]).astype(float)
        bounds = geometry.findall("Polygon/outerBoundaryIs/LinearRing/coordinates", namespace)
        for b in bounds:
            coords = (b.text.replace("\t", "").strip()).split(",0 ")[:-1]
            coords = np.array([np.array(i.split(",")).astype(float) for i in coords])
            # Select the correct boundary coordinates.
            if np.sign(coords[0, 0]) == np.sign(point[0]):
                break

        # --- Bounding Box Intersection Test ---
        # Check if the tile's bounding box intersects with the user's bbox.
        if (
            (coords[:, 0].min() <= bbox[2])
            & (coords[:, 0].max() >= bbox[0])
            & (coords[:, 1].min() <= bbox[3])
            & (coords[:, 1].max() >= bbox[1])
        ):
            # If they intersect, add the tile to the output DataTree.
            mgrs_tiles[n] = xr.Dataset(data_vars={"point": point, "bounds": (["corners", "point"], coords)})
    return mgrs_tiles


def get_utm_zones(mgrs_tiles: xr.DataTree, bbox: list) -> tuple[dict, dict]:
    """
    Determine the UTM zones covering an area and their respective bounding boxes.

    This function takes a set of MGRS tiles that intersect an area of interest
    and groups them by their corresponding UTM zone. For each UTM zone, it
    calculates the total geographic extent covered by the tiles within that zone.

    Parameters
    ----------
    mgrs_tiles : xr.DataTree
        A DataTree of MGRS tiles, typically from `parse_kml`.
    bbox : list
        The overall geographic bounding box of the area of interest.

    Returns
    -------
    dict
        A dictionary where keys are UTM zone EPSG codes (str) and values are the
        bounding boxes (np.ndarray) covered by that zone.
    """
    utm_zones = {}
    m = mgrs.MGRS()
    epsg_to_tiles: dict[str, set] = {}
    # Iterate through each MGRS tile.
    for k, v in mgrs_tiles.items():
        # Convert the MGRS tile name to a UTM zone.
        u, h, _, _ = m.MGRSToUTM(k)
        epsg = f"32{6 if h == 'N' else 7}{u}"
        bounds = cast(np.ndarray, v["bounds"].values)
        lon_min, lat_min = np.min(bounds, axis=0)
        lon_max, lat_max = np.max(bounds, axis=0)
        # --- Group by UTM Zone ---
        # If this is the first tile for this UTM zone, initialize its extent.
        if epsg not in utm_zones:
            utm_zones[epsg] = np.array([lon_min, lat_min, lon_max, lat_max])
            epsg_to_tiles[epsg] = set()
        # Otherwise, expand the existing extent to include this tile.
        else:
            lon_min = np.minimum(lon_min, utm_zones[epsg][0])
            lat_min = np.minimum(lat_min, utm_zones[epsg][1])
            lon_max = np.maximum(lon_max, utm_zones[epsg][2])
            lat_max = np.maximum(lat_max, utm_zones[epsg][3])
            utm_zones[epsg] = np.array([lon_min, lat_min, lon_max, lat_max])
        epsg_to_tiles[epsg].add(k)

    # --- Clip Extents to Overall BBox ---
    # Finally, clip the extent of each UTM zone to the overall area of interest.
    for k, v in utm_zones.items():
        lon_min = np.maximum(v[0], bbox[0])
        lat_min = np.maximum(v[1], bbox[1])
        lon_max = np.minimum(v[2], bbox[2])
        lat_max = np.minimum(v[3], bbox[3])
        utm_zones[k] = np.array([lon_min, lat_min, lon_max, lat_max])
    return utm_zones, epsg_to_tiles


def get_resolution_in_meters(data: xr.DataArray | xr.Dataset) -> np.ndarray:
    """
    Calculate the spatial resolution of a dataset in meters.

    If the dataset's CRS is geographic (in degrees), this function transforms
    a sample of the grid to a suitable UTM projection to estimate the approximate
    resolution in meters. If the CRS is already projected, it returns the
    resolution directly.

    Parameters
    ----------
    data
        The input xarray object with a defined CRS.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the resolution in [x, y] meters.
    """
    crs = CRS(data.rio.crs).to_3d()

    # If the CRS is in degrees, we need to reproject to get meters.
    if crs.is_geographic:
        # Find the appropriate local UTM zone.
        default_epsg = get_default_epsg(data)
        # Create a transformer from geographic to UTM.
        transformer = Transformer.from_crs(data.rio.crs, default_epsg, always_xy=True)
        # Take a sample of the coordinates to transform.
        x, y = data.x[0:1000].values, data.y[0:1000].values
        x, y = np.meshgrid(x, y)
        # Transform the grid.
        x, y = transformer.transform(x, y)
        # Calculate the mean difference between adjacent points in meters.
        resolution = (np.diff(x, axis=1).mean(), np.diff(y, axis=0).mean())
    else:
        # If the CRS is already projected, the resolution is in meters.
        resolution = data.rio.resolution()
    return np.array(resolution)


def convert_to_dem_ecef(dem: xr.DataArray) -> np.ndarray:
    """
    Converts a Digital Elevation Model (DEM) to Earth-Centered, Earth-Fixed (ECEF) coordinates.

    This function reprojects the spatial coordinates (x, y) and elevation values (z)
    of the input DEM from its native CRS to the WGS 84 / ECEF coordinate system (EPSG:4978).

    Args:
        dem (xr.DataArray): Input DEM data containing 'x' and 'y' coordinates and
                            elevation values. Must have a valid CRS accessible via `.rio.crs`.

    Returns:
        np.ndarray: A numpy array of shape (3, H, W) containing the transformed
                    X, Y, Z coordinates in the ECEF frame.
    """
    # Define the Source CRS (from input) and Target CRS (EPSG:4978 = WGS 84 ECEF).
    # .to_3d() is used to ensure the CRS is treated as a 3D system (Lat/Lon/Height).
    source = CRS(dem.rio.crs).to_3d()
    target = CRS(4978)

    # Initialize the transformer.
    # always_xy=True ensures the order is (x, y) or (lon, lat) rather than (lat, lon).
    transformer = Transformer.from_crs(source, target, always_xy=True)

    # Extract 1D coordinate arrays (x, y) and the 2D elevation grid (z)
    x, y, z = dem.x.values, dem.y.values, dem.values

    # Create 2D coordinate grids for x and y to match the shape of the z (elevation) array.
    # This aligns every pixel's height with its geospatial location.
    x_grid, y_grid = np.meshgrid(x, y)

    # Perform the coordinate transformation on the full grid.
    # Returns X, Y, Z arrays in ECEF coordinates.
    X, Y, Z = transformer.transform(x_grid, y_grid, z)

    # Stack result into a numpy array
    return np.array([X, Y, Z])


class GeoEnvPlugin(dask.distributed.WorkerPlugin):
    name = "patch_gdal_proj_env"

    def __init__(
        self, in_proj: str = "EPSG:9518", out_proj: str = "EPSG:4979", test_coords: tuple = (12.5, 41.9, 50.0)
    ):
        """
        Initializes the plugin.
        """
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.test_coords = test_coords

    def setup(self, worker):
        """Runs on the worker immediately when it starts."""

        # 1. Base Environment Setup (Locking paths)
        pyproj_dir = datadir.get_data_dir()
        datadir.set_data_dir(pyproj_dir)

        # STRICTLY OFFLINE: Workers are not allowed to use the internet
        pyproj.network.set_network_enabled(False)

        rasterio_base = os.path.dirname(rasterio.__file__)
        rasterio_gdal_dir = os.path.join(rasterio_base, "gdal_data")
        if os.path.exists(rasterio_gdal_dir):
            os.environ["GDAL_DATA"] = rasterio_gdal_dir

        # 2. Strict Metadata Check
        in_crs = CRS(self.in_proj).to_3d()
        out_crs = CRS(self.out_proj).to_3d()
        tg = TransformerGroup(in_crs, out_crs)

        if tg.best_available is False:
            error_msg = (
                f"Worker {worker.address} is missing required PROJ grids for {self.in_proj} -> {self.out_proj}. "
                "Ensure the environment was provisioned properly before starting the worker."
            )
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        in_vertical_crs = extract_vertical_crs(in_crs)
        out_vertical_crs = extract_vertical_crs(out_crs)
        if in_vertical_crs != out_vertical_crs:
            # 3. Strict Mathematical Validation
            transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)
            test_x, test_y, test_z = self.test_coords

            # Execute the transformation; errcheck=True catches missing grid errors
            _, _, new_z = transformer.transform(test_x, test_y, test_z, errcheck=True)

            if np.allclose(test_z, new_z, atol=1e-5):
                error_msg = (
                    f"Worker {worker.address} validation failed! Z coordinate did not change. "
                    "The geoid grid is missing or being ignored."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
