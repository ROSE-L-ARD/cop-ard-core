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
from typing import Any, Dict, Mapping, Sequence

import dask.distributed
import mgrs
import numpy as np
import pyproj
import rasterio
import rioxarray
import xarray as xr
from pyproj import CRS, Transformer, datadir
from pyproj.transformer import TransformerGroup

from cop_ard_core.computing.griddata import interp_dataset
from cop_ard_core.computing.io import get_storage_options, normpath, write
from cop_ard_core.computing.logger import logger

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
