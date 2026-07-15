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
import itertools
from typing import Dict

import numpy as np
import xarray as xr
from scipy.constants import speed_of_light
from scipy.ndimage import median_filter

from cop_ard_core.computing.dem import convert_to_dem_ecef
from cop_ard_core.computing.griddata import init_interpolate, interpolate
from cop_ard_core.computing.io import (
    build_sparse_array,
    exists,
    get_storage_options,
    normpath,
    read,
)
from cop_ard_core.computing.kernels import (
    area_proj_triangles,
    compute_incidence_angle_ground,
    computeGammaNaughtArea,
    csr_times_dense,
    update_dense_array,
)
from cop_ard_core.computing.simul import extract_simul_on_radar_chunk, get_geo_from_radar_bbox, get_radar_bbox


def build_sparse_array(
    data: np.ndarray,
    col: np.ndarray,
    row: np.ndarray,
    shape: np.ndarray,
    format: str,
    chunks: int,
    attrs: dict = {},
) -> xr.Dataset:
    """
    Save a sparse matrix in COO or CSR format to a Zarr store.

    This function takes the components of a sparse matrix (data, indices, etc.),
    packages them into an xarray.Dataset, and writes them to a Zarr store.
    This allows for efficient storage of sparse data.

    Parameters
    ----------
    data : np.ndarray
        The non-zero values of the sparse matrix.
    col : np.ndarray
        For "COO", the column indices for each value. For "CSR",
        this corresponds to the `indices` array.
    row : np.ndarray
        For "COO", the row indices for each value. For "CSR",
        this corresponds to the `indptr` array.
    shape : np.ndarray
        The shape of the equivalent dense matrix.
    format : str
        The sparse format, which must be either "COO" or "CSR".
    chunks : int
        The chunk size to be used for the Dask arrays in the dataset.
    data_path : str
        The output path for the Zarr store.
    skip_read : bool, optional
        If False, the function will read the data back
        from disk and return it. Defaults to True.
    attrs : dict, optional
        Additional attributes to save with the dataset.

    Returns
    -------
    xr.Dataset or None
        The saved dataset if `skip_read` is False, otherwise None.

    Raises
    ------
    Exception
        If the specified format is not "COO" or "CSR".
    """
    # Validate that the specified sparse format is supported.
    if format not in ["COO", "CSR"]:
        raise Exception(f"Expected COO or CSR format, got {format}")
    # Initialize a dictionary to hold the data variables for the xarray Dataset.
    data_vars = {
        "data": ("point", data),
        "nnz": data.shape[0],
        "shape": shape,
    }
    # Add format-specific variables.
    if format == "COO":
        # For Coordinate format, store column and row index arrays.
        data_vars["col"] = ("point", col)
        data_vars["row"] = ("point", row)
    elif format == "CSR":
        # For Compressed Sparse Row format, store indices and index pointer arrays.
        data_vars["indices"] = ("point", col)
        data_vars["indptr"] = ("indptr", row)

    # Create the xarray Dataset from the prepared data variables and attributes.
    ds = xr.Dataset(
        data_vars=data_vars,
        attrs={"format": format, **attrs},
    ).chunk(chunks)
    return ds


def triangulate_regular(ysize: int, xsize: int) -> np.ndarray:
    """
    Generate vertex indices for a standard triangulation of a regular grid.

    This function creates a list of triangles covering a regular grid. Each
    rectangular cell in the grid is divided into two triangles. The output is
    an array where each row represents a triangle, and the columns contain the
    linear indices of its three vertices in a flattened grid.

    Parameters
    ----------
    ysize : int
        The number of grid cells (height - 1) along the y-axis.
    xsize : int
        The number of grid cells (width - 1) along the x-axis.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape ``(2 * ysize * xsize, 3)`` containing
        the linear indices of the vertices for each triangle.
    """
    # `aa` defines the triangles for the first row of cells.
    aa = np.vstack(
        [
            np.array(
                [
                    [i, i + 1, xsize + i],
                    [i, i + 1, xsize + i + 1],
                    [i, xsize + i + 1, xsize + i],
                    [i + 1, xsize + i + 1, xsize + i],
                ]
            )
            # Iterate through columns (xsize - 1 cells horizontally)
            for i in range(xsize - 1)
        ]
    )
    # Stack the triangle definitions vertically for all rows.
    # Add `i * xsize` to shift the indices down for each subsequent row.
    # The total number of rows of cells is `ysize - 1`.
    return np.vstack([aa + i * xsize for i in range(ysize - 1)])


def compute_rasterization(
    simulation: xr.DataArray,
    xmin_global: int,
    ymin_global: int,
    xsize_global: int,
    radar_grid_pixel_area: float,
    bbox: np.ndarray,
    resolution: tuple[float, float],
    block_id: str,
) -> tuple[xr.DataArray, xr.Dataset | None, list[tuple[str, xr.Dataset]] | None]:
    """
    Compute rasterization weights and the accumulated gamma-naught area (RTC-AP)
    for a block of the radar data simulation.

    This function performs triangulation on the geogrid, computes weights
    for mapping from radar grid to geogrid, and then calculates the
    Gamma Naught Area (GNA), local incidence angle (lia), and number of looks (nlooks).

    Parameters
    ----------
    simulation : xr.DataArray
        The radar data simulation chunk, containing
        coordinates (az_px, rg_px) and DEM height (h).
    data_path_static : str
        Path to the directory where static results
        (geo_grid, weights) will be saved.
    data_path_tmp : str
        Path to the directory where temporary results
        (sparse layover/area blocks) will be saved.
    xmin_global : int
        Global minimum of the radar x-coordinate (range).
    ymin_global : int
        Global minimum of the radar y-coordinate (azimuth).
    xsize_global : int
        Total radar grid size along the x-axis (range).
    dim_x_global : int
        Total geogrid size along the x-axis.
    radar_grid_pixel_area : float
        The area of a single radar grid pixel.
    normal_order : bool
        The order needed to compute the surface normal correctly.
    bbox : np.ndarray
        Bounding box coordinates of the full radar scene
        [min_rg, min_az, max_rg, max_az].
    chunk_length : int
        Size (dimension) of a square geogrid chunk.
    resolution : tuple[float, float]
        Output resolution (az_res, rg_res)
        used to calculate expected number of looks.
    job_id : int or None, optional
        Identifier for the job, used to determine the
        position of this chunk in the global grid.

    Returns
    -------
    tuple[xr.DataArray, xr.Dataset | None, list[tuple[str, xr.Dataset]] | None]
        A tuple containing nlooks and the temporary sparse area matrices.
        Layover matrices are returned as a list of bbox/matrix tuples
    """
    # Calculate the expected number of looks based on desired resolution and radar pixel area
    expected_nlooks = np.ceil(np.abs(resolution).prod() / radar_grid_pixel_area).astype("int64")

    # Get radar grid coordinates (azimuth and range pixel indices) from the simulation
    az_px, rg_px = simulation.az_px, simulation.rg_px

    # Get local geogrid dimensions (number of cells per row and per column)
    # The size is the number of points, so dimensions are size - 1
    dim_x, dim_y = simulation.x.size - 1, simulation.y.size - 1

    # Create a regular grid of triangles from the geogrid dimensions
    triang = triangulate_regular(dim_y + 1, dim_x + 1)

    # Find the local minimum coordinates (floor) in the radar grid for this chunk
    ymin, xmin = np.floor(az_px.min().values).astype(int), np.floor(rg_px.min().values).astype(int)

    # Adjust the bounding box to be relative to the local chunk's minimum coordinates
    bbox_chunk = bbox.copy()
    bbox_chunk = np.array([bbox_chunk[0] - xmin, bbox_chunk[1] - ymin, bbox_chunk[2] - xmin, bbox_chunk[3] - ymin])

    # Translate radar grid coordinates to the local reference system (starting from 0,0) and flatten
    x_, y_ = (rg_px - xmin).squeeze().values.ravel(), (az_px - ymin).squeeze().values.ravel()

    # Compute rasterization weights (w) and number of looks (nl)
    w, nl, layover = get_w(
        x=x_,
        y=y_,
        triang=triang,
        ymin=ymin,
        ymin_global=ymin_global,
        xmin=xmin,
        xmin_global=xmin_global,
        xsize_global=xsize_global,
        bbox=bbox_chunk,
        expected_nlooks=expected_nlooks,
    )

    # Proceed only if weights were successfully computed
    if w is not None:
        # Validate the type of number of looks array
        if not isinstance(nl, np.ndarray):
            raise Exception(f"Expected np.ndarray, got {type(nl)}")
        # Reshape nlook: (dim_y, dim_x, 4) -> sum over 4 triangles per quad, div by 2 (2 triangles per geogrid cell)
        nlook = xr.DataArray(nl.reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]) / 2
    else:
        # If no weights were computed (e.g., empty chunk), create empty arrays
        nlook = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])

    return nlook, w, layover


def compute_rtc_ap(
    csr_matrix: xr.Dataset,
    simulation: xr.DataArray,
    radar_grid_pixel_area: float,
    normal_order: bool,
    bbox: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray, xr.Dataset | None]:
    """
    Compute rasterization weights and the accumulated gamma-naught area (RTC-AP)
    for a block of the radar data simulation.

    This function performs triangulation on the geogrid, computes weights
    for mapping from radar grid to geogrid, and then calculates the
    Gamma Naught Area (GNA), local incidence angle (lia), and number of looks (nlooks).

    Parameters
    ----------
    simulation : xr.DataArray
        The radar data simulation chunk, containing
        coordinates (az_px, rg_px) and DEM height (h).
    data_path_static : str
        Path to the directory where static results
        (geo_grid, weights) will be saved.
    data_path_tmp : str
        Path to the directory where temporary results
        (sparse layover/area blocks) will be saved.
    xmin_global : int
        Global minimum of the radar x-coordinate (range).
    ymin_global : int
        Global minimum of the radar y-coordinate (azimuth).
    xsize_global : int
        Total radar grid size along the x-axis (range).
    dim_x_global : int
        Total geogrid size along the x-axis.
    radar_grid_pixel_area : float
        The area of a single radar grid pixel.
    normal_order : bool
        The order needed to compute the surface normal correctly.
    bbox : np.ndarray
        Bounding box coordinates of the full radar scene
        [min_rg, min_az, max_rg, max_az].
    chunk_length : int
        Size (dimension) of a square geogrid chunk.
    resolution : tuple[float, float]
        Output resolution (az_res, rg_res)
        used to calculate expected number of looks.
    job_id : int or None, optional
        Identifier for the job, used to determine the
        position of this chunk in the global grid.

    Returns
    -------
    tuple[xr.Dataset, xr.DataTree | None, xr.DataTree | None, xr.DataTree | None]
        A tuple containing the computed geogrid data (nlooks, anf, lia) and the temporary sparse area matrices.
    """
    # --- Start Radiometric Terrain Correction (RTC) Area Projection (AP) ---
    # Get DEM points in ECEF coordinates: (axis, y, x) -> (y, x, axis) -> (n_points, 3)
    points = convert_to_dem_ecef(dem=simulation.h.compute())
    points = np.transpose(points, (1, 2, 0)).reshape(-1, 3)
    dim_x, dim_y = simulation.x.size - 1, simulation.y.size - 1

    # Create a regular grid of triangles from the geogrid dimensions
    triang = triangulate_regular(dim_y + 1, dim_x + 1)
    # Get the ECEF coordinates for each vertex of each triangle
    # Dimensions: (number_of_triangles, 3 vertices, 3 axes)
    triang_points = points[triang]

    # Compute the mean radar distance for the barycenter of each triangle
    radar_distance = np.mean(
        simulation.dem_distance.values.reshape(-1, 3)[triang],
        axis=1,
    )

    # Initialize output arrays for Gamma Naught Area (GNA) and cosine of local incidence angle
    GNA, cos_theta = np.zeros(triang.shape[0], dtype="float64"), np.zeros(triang.shape[0], dtype="float64")

    # Initialize arrays for building the sparse area matrix (CSR format)
    dense, data = np.zeros(csr_matrix.shape.values[1], dtype="float64"), np.zeros(csr_matrix.nnz.values, dtype="float64")
    indices, indptr = np.zeros(csr_matrix.nnz.values, dtype="int64"), np.zeros(bbox[3] + 1, dtype="int64")

    # Define the bounding box for the full radar grid (used by the kernel)
    bbox_radar = np.array([bbox[3], 0, bbox[2], 0])
    # Call the high-performance kernel to compute GNA and the sparse area matrix 'A'
    nnz = computeGammaNaughtArea(
        triang_points=triang_points.ravel(),
        radar_distance=radar_distance.ravel(),
        radar_grid_pixel_area=radar_grid_pixel_area,
        num_triangles=triang.shape[0],
        normal_order=normal_order,
        GNA=GNA,
        lia=cos_theta,
        data_w=csr_matrix.data.values,
        indptr_w=csr_matrix.indptr.values,
        indices_w=csr_matrix.indices.values,
        n_rows=csr_matrix.shape.values[0],
        n_cols=csr_matrix.shape.values[1],
        dense=dense,
        data=data,
        indices=indices,
        indptr=indptr,
        width=bbox[2],
        length=bbox[3],
        bbox=bbox_radar,
    )

    # Trim the sparse matrix arrays to the actual number of non-zero elements
    data, indices = data[:nnz].astype("float32"), indices[:nnz]

    # If the sparse matrix is not empty, save it to the temporary directory
    a_block = None
    if nnz != 0:
        a_block = build_sparse_array(
            data=data,
            row=indptr,
            col=indices,
            shape=np.array([bbox[3], bbox[2]]),  # Shape of the full radar grid
            format="CSR",
            chunks=triang.shape[0] // 4,
        )

    # Compute Area Normalization Factor (ANF) by summing GNA contributions
    anf = xr.DataArray(GNA.reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]).astype("float32") / 2.0

    # Compute local incidence angle (lia) from the cosine
    lia = (
        xr.DataArray(np.arccos(cos_theta).reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]).astype("float32")
        / 4.0
    )

    return anf, lia, a_block


def get_w(
    x: np.ndarray,
    y: np.ndarray,
    triang: np.ndarray,
    ymin: int,
    ymin_global: int,
    xmin: int,
    xmin_global: int,
    xsize_global: int,
    bbox: np.ndarray,
    expected_nlooks: int,
    plane_orientation: int = 1,
) -> tuple[xr.Dataset, xr.DataArray, list[tuple[str, xr.Dataset]]] | tuple[None, None, None]:
    """
    Compute the rasterization weights for all triangles in the geogrid chunk.

    This function calls a kernel to perform area projection for triangles onto
    the radar grid, generating sparse weights (w), layover masks (l1, l2),
    and the number of looks (nlook).

    Parameters
    ----------
    x : np.ndarray
        Flattened radar x-coordinates (range) in the local chunk system.
    y : np.ndarray
        Flattened radar y-coordinates (azimuth) in the local chunk system.
    triang : np.ndarray
        Triangle definitions (indices of vertices)
        for the geogrid.
    ymin : int
        The minimum y-coordinate (azimuth) in the local radar grid.
    ymin_global : int
        The minimum y-coordinate (azimuth) in the global radar grid.
    xmin : int
        The minimum x-coordinate (range) in the local radar grid.
    xmin_global : int
        The minimum x-coordinate (range) in the global radar grid.
    xsize_global : int
        The total size of the global radar grid in the x-direction (range).
    bbox : np.ndarray
        Bounding box defining the region of interest
        relative to the local chunk's minimum coordinates.
    expected_nlooks : int
        The expected number of looks for normalization.
    data_path_static : str
        Path to save the static weights file.
    data_path_tmp : str
        Path to save temporary layover mask files.
    block_id : str
        A string identifier for this chunk (e.g., "row.col").
    plane_orientation : int, optional
        Flag indicating the orientation
        of the plane. Defaults to 1.

    Returns
    -------
    tuple[xr.Dataset, xr.DataArray] or tuple[None, None]
        A tuple containing:

        - **xr.Dataset**: A sparse dataset (read from disk) containing the weights (w).
        - **xr.DataArray**: A numpy array containing the number of looks (nlook).

        Returns ``(None, None)`` if no non-zero weights are computed.
    """
    # Calculate the size of the local radar grid bounding box
    xsize = (np.ceil(x.max()) - np.floor(x.min())).astype(int)
    ysize = (np.ceil(y.max()) - np.floor(y.min())).astype(int)

    # Define vertex order for triangle traversal
    vertexes = np.array([0, 1, 2, 0], dtype=np.int64)
    number_of_triang = triang.shape[0]

    # Calculate a unique 1D index for the start (top-left) of this chunk
    # in the global radar grid
    linear_global_index = (ymin - ymin_global) * xsize_global + (xmin - xmin_global)

    # Initialize bounding box arrays to be populated by the kernel
    # These will define the extent of the sparse layover/weight matrices
    bbox_l1 = np.array([bbox[3] + ymin, 0, bbox[2] + xmin, 0]).astype("int64")
    bbox_l2 = np.array([bbox[3] + ymin, 0, bbox[2] + xmin, 0]).astype("int64")
    bbox_w = np.array([bbox[3] + ymin, 0, bbox[2] + xmin, 0]).astype("int64")

    # Call the high-performance kernel to compute weights, layover, and nlooks
    w, l1, l2, nlook = area_proj_triangles(
        y=y,
        x=x,
        triangs=triang.ravel(),
        vertexes=vertexes,
        bbox=bbox,
        linear_global_index=linear_global_index,
        length=ysize,
        width=xsize,
        xsize_global=xsize_global,
        xmin_global=xmin_global,
        ymin_global=ymin_global,
        xmin_chunk=xmin,
        ymin_chunk=ymin,
        radar_grid_ysize=bbox[3] + ymin,
        plane_orientation=plane_orientation,
        bbox_l1=bbox_l1,
        bbox_l2=bbox_l2,
        bbox_w=bbox_w,
        expected_nlooks=expected_nlooks,
    )

    # Get the number of non-zero elements in the computed weights
    nnz = w[0].size

    if nnz > 0:
        # If weights exist, save them as a sparse array dataset
        w = build_sparse_array(
            data=w[0].astype("float32"),  # data
            row=w[1],  # indptr
            col=w[2],  # indices
            shape=np.array([number_of_triang, (bbox[2] + xmin) * (bbox[3] + ymin)]),
            format="CSR",
            chunks=number_of_triang // 4,
            attrs={"radar_bbox": bbox_w},  # Store the bounding box as metadata
        )

        # Save the layover masks (l1 and l2) to the temporary directory
        # File name is based on the sparse block's bounding box
        # data_path=f"{data_path_tmp}/sparse_dir/layover_block_{i}/{'.'.join([str(j) for j in b])}",
        # skip_read=True,  # We don't need to read these back
        # storage_options=get_storage_options(kwargs, key="storage_options_static"),
        layover: list[tuple[str, xr.Dataset]] = []
        for i, (ll, b) in enumerate([(l1, bbox_l1), (l2, bbox_l2)]):
            ds = build_sparse_array(
                data=ll[0].astype("float32"),
                row=ll[1],
                col=ll[2],
                shape=np.array([(bbox[3] + ymin), (bbox[2] + xmin)]),
                format="CSR",
                chunks=number_of_triang // 4,
            )
            layover.append(('.'.join([str(j) for j in b]), ds))

    # Return the weights dataset and the nlook array
    # If no weights were computed, return None
    return (w, nlook.astype("float32"), layover) if nnz > 0 else (None, None, None)


def compute_geocoded_mask(
    csr_matrix: xr.Dataset,
    layover: xr.DataArray,
    shadow: xr.DataArray,
    xsize_global: int,
    bbox_radar: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    threshold: float = 0.3,
) -> np.ndarray:
    """
    Computes the geocoded Gamma Naught for a single chunk.

    This is the worker function called by `map_blocks`. It loads the
    pre-computed sparse weights (`w`) for this chunk, reads the corresponding
    section of the SLC and radar-grid ANF, normalizes the SLC intensity by
    the ANF, performs the sparse matrix multiplication (geocoding), and
    writes the resulting geocoded chunk to the 'geo_grid' Zarr store.

    Parameters
    ----------
    csr_matrix : xr.Dataset
        CSR format sparse matrix components
    layover : xr.DataArray
        Layover boolean mask
    shadow : xr.DataArray
        Shadow boolean mask
    xsize_global : int
        The total global size of the radar grid along
        the x-axis (range).
    bbox_radar : tuple[int, int, int, int]
        Bounding box of the mask in radar coordinates
    output_shape : tuple[int, int]
        Output shape of the mask
    threshold : float, optional
        Minimum ANF value; pixels below this
        are set to 0 to avoid division by zero.
        Defaults to 1e-3.

    Returns
    -------
    xr.Dataset
        Geocoded mask data for this chunk.
    """
    dim_y, dim_x = output_shape

    # Read sparse matrix components (CSR format)
    data = csr_matrix.data.values
    indices = csr_matrix.indices.values
    indptr = csr_matrix.indptr.values
    shape = csr_matrix.shape.values
    
    mask_layover = layover.values.astype("float32")
    mask_shadow = shadow.values.astype("float32")
    # Stack them into a (2, az, rg) array
    radar_mask = np.stack([mask_layover, mask_shadow], axis=0)

    # Initialize the output array for the geocoded mask (on triangle grid)
    geocoded_mask = np.zeros((radar_mask.shape[0], shape[0]), dtype="float32")

    # Perform the sparse matrix multiplication: geocoded_mask = w * radar_mask
    csr_times_dense(
        n_dims=radar_mask.shape[0],  # 2 dimensions (layover, shadow)
        n_rows=shape[0],
        data=data,
        indices=indices,
        indptr=indptr,
        radar_image=radar_mask.ravel(),
        geocoded_image=geocoded_mask.ravel(),
        ystart=bbox_radar[0],
        xstart=bbox_radar[2],
        xsize_global=xsize_global,
        ysize_block=radar_mask.shape[1],
        xsize_block=radar_mask.shape[2],
    )

    # Reshape and average from triangle-grid to cell-grid
    geocoded_mask = geocoded_mask.reshape(radar_mask.shape[0], dim_y, dim_x, 4).sum(axis=-1) / 4.0
    # Convert from fractional mask to boolean mask using a threshold
    geocoded_mask = geocoded_mask > threshold
    geocoded_mask = geocoded_mask[0].astype("int32") + 2 * geocoded_mask[1].astype("int32")
    return geocoded_mask


def compute_shadow(
    radar_grid: xr.DataArray,
    simulation_h: xr.DataArray,
    simulation_dem_distance: xr.DataArray,
    simulation_azimuth_time: xr.DataArray,
    simulation_slant_range_time: xr.DataArray,
    simulation_ml: dict,
    local_incidence_angle: xr.DataArray,
    eps: float,
    dt: dict,
    margin: dict,
    **kwargs,
) -> xr.Dataset:  # TODO: fix docstring
    """
    Computes the shadow mask for a single chunk of the radar grid.

    This function interpolates the ground incidence angle and the geogrid
    local incidence angle (LIA) onto the radar grid. It then identifies shadow
    pixels based on two criteria:
    1. The cumulative maximum of the ground incidence angle (detecting occlusions).
    2. A grazing angle (LIA) threshold.

    Parameters
    ----------
    A : xr.DataArray
        The template Dask chunk (defines the radar grid
        region for this computation).
    simulation : xr.Dataset
        The radar simulation dataset.
    simulation_ml : dict
        The multi-looked simulation data.
    eps : float
        Epsilon value for the cumulative maximum comparison.
    dt : dict
        Dictionary of time differentials for az/rg.
    margin : dict
        Dictionary of time margins for simulation extraction.
    data_path : str
        Path to the directory where results are saved and
        'geo_grid' LIA is read from.
    chunk : dict, optional
        A dictionary injected by `map_blocks`
        defining the slices for this chunk.

    Returns
    -------
    xr.Dataset
        The computed shadow mask as an xr.Dataset.
    """
    init_interpolate()
    simulation = xr.Dataset(
        data_vars={
            "h": simulation_h,
            "dem_distance": simulation_dem_distance,
            "azimuth_time": simulation_azimuth_time,
            "slant_range_time": simulation_slant_range_time,
        },
    )

    azimuth_time = radar_grid.azimuth_time
    slant_range_time = radar_grid.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    # Extract the relevant simulation data corresponding to this radar chunk + margin
    selection = extract_simul_on_radar_chunk(
        simulation, simulation.acquisition_date.values[0], bbox_radar, bbox_geo
    ).squeeze()

    # define slices
    xslice = slice(selection.x[0].values, selection.x[-1].values)
    yslice = slice(selection.y[0].values, selection.y[-1].values)

    # Read the Local Incidence Angle (LIA) from the geogrid
    lia = local_incidence_angle.sel(x=xslice, y=yslice)
    resolution = lia.rio.resolution()
    x = lia.x.values - resolution[0] / 2.0
    y = lia.y.values - resolution[1] / 2.0
    # define slices to align simulation with lia since lia could be smaller
    xslice = slice(x[0], x[-1])
    yslice = slice(y[0], y[-1])

    selection = selection.sel(x=xslice, y=yslice)
    lia = lia.values
    # --- Calculate Ground Incidence Angle ---
    # Convert DEM points to ECEF
    points = convert_to_dem_ecef(dem=selection.h.compute())
    points = np.transpose(points, (1, 2, 0)).reshape(-1, 3)
    # Get sensor-to-target distance vectors
    radar_distance = selection.dem_distance.values.reshape(-1, 3)
    incidence_angle_ground = np.zeros(points.shape[0], dtype=float)
    compute_incidence_angle_ground(
        radar_distance=radar_distance.ravel(),
        points=points.ravel(),
        incidence_angle_ground=incidence_angle_ground,
        num_points=incidence_angle_ground.shape[0],
    )

    incidence_angle_ground = incidence_angle_ground.reshape(selection.y.size, selection.x.size)

    # --- Interpolate to Radar Grid ---
    # Get reference start times for the chunk
    az_ref, rg_ref = radar_grid.azimuth_time[0].values, radar_grid.slant_range_time[0].values

    # Create a 2D array of (az, rg) pixel coordinates for the simulation points
    points = np.array(
        [
            (((selection.azimuth_time - az_ref).astype(float)) / (dt["azimuth_time"] * 1e9)).values,
            ((selection.slant_range_time - rg_ref).astype(float) / (dt["slant_range_time"])).values,
        ]
    ).transpose(2, 1, 0)

    # Stack the values to be interpolated (Ground Incidence Angle, LIA)
    z = np.stack([incidence_angle_ground.T, lia.T])

    # Define the output grid bounding box (in pixel coordinates)
    bbox = np.array([0, 0, radar_grid.slant_range_time.size, radar_grid.azimuth_time.size])

    # Interpolate values from simulation points (on geogrid) to the radar grid
    m = interpolate(points, z, bbox)
    incidence_angle_radar = m[0, :, :]
    cos_theta = m[1, :, :]  # This is interpolated cos(LIA)

    # --- Create Shadow Mask ---
    # 1. Apply median filter ONLY along axis 1 (range).
    smoothed_angle = median_filter(incidence_angle_radar, size=(1, 5))
    # 2. Cumulative maximum mask (detects occlusions)
    # Scan along range (axis=1) and find pixels where the angle *decreases*
    cumulative_max = np.maximum.accumulate(smoothed_angle, axis=1)
    # 3. Create the mask
    mask_max = abs(cumulative_max - smoothed_angle) >= eps
    # 2. Grazing angle mask (where local slope is parallel to radar beam)
    mask_cos = cos_theta <= 0.05
    # Combine masks (shadow is True if *either* condition is met)
    mask_shadow = mask_max + mask_cos
    mask_shadow = xr.DataArray(mask_shadow, dims=["azimuth_time", "slant_range_time"]).to_dataset(name="shadow")
    # Write the shadow mask chunk to the 'radar_grid' Zarr store
    return mask_shadow
