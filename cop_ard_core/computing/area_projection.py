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
from typing import Dict, cast

import botocore.exceptions
import dask.array as da
import numpy as np
import xarray as xr
from scipy.constants import speed_of_light
from scipy.ndimage import median_filter

from cop_ard_core.computing.dem import GeoEnvPlugin, convert_to_dem_ecef, get_resolution_in_meters
from cop_ard_core.computing.griddata import init_interpolate, interpolate
from cop_ard_core.computing.io import (
    delete,
    exists,
    get_storage_options,
    glob,
    makedirs,
    normpath,
    read,
    save_empty_dataset,
    save_sparse_array,
    touch,
    write, build_sparse_array,
)
from cop_ard_core.computing.kernels import (
    area_proj_triangles,
    compute_incidence_angle_ground,
    computeGammaNaughtArea,
    csr_times_dense,
    update_dense_array,
)
from cop_ard_core.computing.logger import logger, timeit
from cop_ard_core.computing.simul import extract_simul_on_radar_chunk, get_radar_bbox, get_geo_from_radar_bbox
from cop_ard_core.computing.slc import get_margin


def initialize_simulation(
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    chunk_length: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Initialize and subset simulation data based on SLC extent and calculate global parameters.

    This function determines the relevant geographic subset of the simulation data
    that corresponds to the spatial extent of a given Single Look Complex (SLC)
    image. It calculates various global parameters needed for subsequent rasterization
    and RTC processing, such as grid dimensions, pixel area, and bounding boxes.

    Parameters
    ----------
    slc : xr.DataArray
        The input SLC image DataArray, used to define the
        radar grid extent.
    simulation : xr.Dataset
        The full simulation dataset containing radar geometry
        (azimuth_time, slant_range_time) mapped onto the geographic grid.
    simulation_ml : dict
        A dictionary containing the downsampled ("multilooked")
        simulation data, used for quickly finding the geographic bounds.
    chunk_length : int
        The desired chunk size for the spatial dimensions (x, y)
        of the processed simulation subset.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        A tuple containing:

        - **selection** (`xr.Dataset`): The subset of the simulation data relevant to the
          SLC, re-chunked and with added radar pixel coordinates (``az_px``, ``rg_px``).
        - **global_values** (`xr.Dataset`): A dataset containing important global parameters
          and coordinate vectors derived from the SLC and simulation data.
    """
    # --- Extract Metadata and Calculate Parameters ---
    # Get metadata from the first polarization's attributes.
    annot = slc.attrs[list(slc.attrs.keys())[0]]
    # Calculate azimuth time step (in seconds) and slant range time step.
    dt_az = annot["Grid"]["Col"]["SS"] * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1]
    dt_rg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
    # Get reference times for coordinate conversion.
    az_ref, rg_ref = slc.azimuth_time[0].values, slc.slant_range_time[0].values
    # Calculate the area of a single pixel in the radar grid.
    radar_grid_pixel_area = annot["Grid"]["Col"]["SS"] * annot["Grid"]["Row"]["SS"]
    # Calculate time steps as timedelta64 and float for margin calculation.
    margin = get_margin(annot)

    # --- Select Relevant Simulation Subset ---
    # Use the helper function to extract the simulation data covering the SLC's footprint plus margin.
    azimuth_time = slc.azimuth_time
    slant_range_time = slc.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = extract_simul_on_radar_chunk(
        simulation, simulation.acquisition_date.values[0], bbox_radar, bbox_geo
    ).squeeze()  # Remove singleton dimensions.
    # Get the geographic bounding box of the selected simulation subset.
    bbox_geo = [selection.x.min().values, selection.y.min().values, selection.x.max().values, selection.y.max().values]

    # --- Calculate Radar Pixel Coordinates ---
    # Convert the simulation's azimuth and range times into pixel coordinates relative to the SLC grid origin.
    az = (selection.azimuth_time - az_ref).astype(float) / (dt_az * 1e9)
    rg = (selection.slant_range_time - rg_ref).astype(float) / (dt_rg)

    # --- Clean Up and Rechunk Selection ---
    # Keep only essential coordinates and variables in the selection.
    keep_coords = ["x", "y"]
    keep_vars = ["dem_distance", "h"]
    selection = selection.drop_vars([i for i in selection.coords if i not in keep_coords])
    selection = selection.drop_vars([i for i in selection.data_vars if i not in keep_vars])
    # Assign the calculated radar pixel coordinates.
    selection = selection.assign(az_px=az, rg_px=rg)
    # Apply the specified chunking to the selected subset.
    selection = selection.chunk(chunk_length)

    # --- Calculate Global Grid Dimensions ---
    # Get geographic grid spacing.
    spacing_x = abs(selection.x[1] - selection.x[0]).values
    spacing_y = abs(selection.y[1] - selection.y[0]).values
    # Calculate the total size of the geographic grid covering the selection.
    dim_x_global = np.round((bbox_geo[2] - bbox_geo[0]) / spacing_x).astype(int)
    dim_y_global = np.round((bbox_geo[3] - bbox_geo[1]) / spacing_y).astype(int)

    # Determine the extent of the radar grid covered by the selection in pixel coordinates.
    az_px = selection.az_px
    rg_px = selection.rg_px
    ymin_global, xmin_global = np.floor(az_px.min().values).astype(int), np.floor(rg_px.min().values).astype(int)
    ymax_global, xmax_global = np.ceil(az_px.max().values).astype(int), np.ceil(rg_px.max().values).astype(int)
    # Calculate the total size of the relevant radar grid portion.
    xsize_global = np.ceil(xmax_global - xmin_global).astype(int)
    ysize_global = np.ceil(ymax_global - ymin_global).astype(int)

    # --- Assemble Global Values Dataset ---
    # Store all calculated global parameters and coordinate vectors in a separate dataset.
    global_values = {
        "radar_grid_pixel_area": {"dims": ("point"), "data": [radar_grid_pixel_area]},
        "chunk_length": {"dims": ("point"), "data": [chunk_length]},
        "chunks_slc": {  # Store original SLC chunk sizes
            "dims": ("chunk"),
            "data": [slc.chunksizes["azimuth_time"][0], slc.chunksizes["slant_range_time"][0]],
        },
        "radar_xsize": {"dims": ("point"), "data": [xsize_global]},
        "radar_ysize": {"dims": ("point"), "data": [ysize_global]},
        "radar_xmin": {"dims": ("point"), "data": [xmin_global]},
        "radar_ymin": {"dims": ("point"), "data": [ymin_global]},
        "radar_xmax": {"dims": ("point"), "data": [xmax_global]},  # Max extent needed?
        "geo_xsize": {"dims": ("point"), "data": [dim_x_global]},
        "geo_ysize": {"dims": ("point"), "data": [dim_y_global]},
        "bbox": {"dims": ("values"), "data": [0, 0, slc.slant_range_time.size, slc.azimuth_time.size]},  # Full SLC bbox
        "x_coords": {"dims": ("x"), "data": selection.x.values},
        "y_coords": {"dims": ("y"), "data": selection.y.values},
        "rg_coords": {"dims": ("slant_range_time"), "data": slc.slant_range_time.values},
        "az_coords": {"dims": ("azimuth_time"), "data": slc.azimuth_time.values},
    }
    global_values = xr.Dataset.from_dict(global_values).rio.write_crs(simulation.rio.crs)
    return selection, global_values


def compute_chunks(chunks: dict, overlap: dict = {"y": 1, "x": 1}) -> Dict:
    """
    Compute overlapping chunk boundaries for use with `map_blocks`.

    Given a dictionary defining the chunk sizes along each dimension, this function
    calculates the start and end indices for each chunk, adding a specified overlap
    between adjacent chunks. This is necessary for algorithms that require access
    to neighboring data during block processing.

    Parameters
    ----------
    chunks : dict
        A dictionary mapping dimension names to tuples representing
        the size of each chunk along that dimension (e.g., ``{'y': (1024, 1024, 512)}``).
    overlap : dict, optional
        A dictionary mapping dimension names (``'x'``, ``'y'``) to
        the desired overlap size in pixels. Defaults to ``{"y": 1, "x": 1}``.

    Returns
    -------
    Dict
        A dictionary where keys are dimension names and values are lists of
        ``[start, end]`` index pairs for each overlapping chunk along that dimension.
        This output format is suitable for the `chunks` argument of `map_blocks`.

    Raises
    ------
    Exception
        If the requested overlap is larger than the chunk size.
    """
    # --- Input Validation ---
    if (overlap["x"] < chunks["x"][0]) is False:
        raise Exception(f"Overlap {overlap['x']} is bigger than chunk length {chunks['x'][0]}")
    if (overlap["y"] < chunks["y"][0]) is False:
        raise Exception(f"Overlap {overlap['y']} is bigger than chunk length {chunks['y'][0]}")

    ch = {}
    # Iterate through each dimension specified in the input chunks.
    for k, v in chunks.items():
        # Calculate the cumulative chunk boundaries.
        li = list(v)
        li.insert(0, 0)
        # Check if overlap should be applied to this dimension and if there's more than one chunk.
        if k in overlap and len(v) != 1:
            delta = overlap[k]
            # Calculate [start, end + overlap] for all chunks except the last one.
            ch[k] = [[en * s, en * s + e + delta] for en, (s, e) in enumerate(zip(li[0:-2], li[1:-1]))]
            # Handle the last chunk separately: [previous_end - overlap, previous_end + last_chunk_size].
            if chunks[k][-1] != 1:  # Avoid issues if the last chunk has size 1
                ch[k].append([ch[k][-1][1] - delta, ch[k][-1][1] - delta + chunks[k][-1]])
        else:
            # If no overlap or only one chunk, calculate simple [start, end] boundaries.
            ch[k] = [[en * s, en * s + e] for en, (s, e) in enumerate(zip(li[:-1], li[1:]))]

    # --- Generate Product of Chunk Boundaries ---
    # Use itertools.product to create all combinations of chunk boundaries across dimensions.
    r = np.array(list(itertools.product(*[v for v in ch.values()])))
    # Restructure the result into the dictionary format required by `map_blocks`.
    return {k: r[:, en].tolist() for en, k in enumerate(chunks.keys())}


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


def compute_rasterization_and_rtc_ap(
    simulation: xr.DataArray,
    xmin_global: int,
    ymin_global: int,
    xsize_global: int,
    dim_x_global: int,
    radar_grid_pixel_area: float,
    normal_order: bool,
    bbox: np.ndarray,
    chunk_length: int,
    resolution: tuple[float, float],
    job_id=None,
) -> tuple[xr.Dataset, xr.DataTree | None, xr.DataTree | None, xr.DataTree | None]:
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

    # Determine the chunk's row and column position in the global grid using the job_id
    row_number = job_id // (np.ceil(dim_x_global / chunk_length).astype(int))
    column_number = job_id % (np.ceil(dim_x_global / chunk_length).astype(int))

    # Define the geogrid region (slice) this chunk corresponds to, for writing output
    region = {
        "y": slice(row_number * chunk_length, row_number * chunk_length + dim_y),
        "x": slice(column_number * chunk_length, column_number * chunk_length + dim_x),
    }

    # Translate radar grid coordinates to the local reference system (starting from 0,0) and flatten
    x_, y_ = (rg_px - xmin).squeeze().values.ravel(), (az_px - ymin).squeeze().values.ravel()

    # Compute rasterization weights (w) and number of looks (nl)
    w, nl, w_dt, layover_dt = get_w(
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
        block_id=f"{row_number}.{column_number}",
    )

    # Proceed only if weights were successfully computed
    ablock_dt = None
    if w is not None:
        # Validate the type of number of looks array
        if not isinstance(nl, np.ndarray):
            raise Exception(f"Expected np.ndarray, got {type(nl)}")

        # Reshape nlook: (dim_y, dim_x, 4) -> sum over 4 triangles per quad, div by 2 (2 triangles per geogrid cell)
        nlook = xr.DataArray(nl.reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]) / 2

        # --- Start Radiometric Terrain Correction (RTC) Area Projection (AP) ---

        # Get DEM points in ECEF coordinates: (axis, y, x) -> (y, x, axis) -> (n_points, 3)
        points = convert_to_dem_ecef(dem=simulation.h.compute())
        points = np.transpose(points, (1, 2, 0)).reshape(-1, 3)

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
        dense, data = np.zeros(w.shape.values[1], dtype="float64"), np.zeros(w.nnz.values, dtype="float64")
        indices, indptr = np.zeros(w.nnz.values, dtype="int64"), np.zeros(bbox[3] + 1, dtype="int64")

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
            nlook=nl,
            data_w=w.data.values,
            indptr_w=w.indptr.values,
            indices_w=w.indices.values,
            n_rows=w.shape.values[0],
            n_cols=w.shape.values[1],
            dense=dense,
            data=data,
            indices=indices,
            indptr=indptr,
            width=bbox[2],
            length=bbox[3],
            bbox=bbox_radar,
        )
        w = None

        # Trim the sparse matrix arrays to the actual number of non-zero elements
        data, indices = data[:nnz].astype("float32"), indices[:nnz]

        # If the sparse matrix is not empty, save it to the temporary directory
        if nnz != 0:
            a_block = build_sparse_array(
                data=data,
                row=indptr,
                col=indices,
                shape=np.array([bbox[3], bbox[2]]),  # Shape of the full radar grid
                format="CSR",
                chunks=triang.shape[0] // 4,
                # Save with a name corresponding to its radar grid bounding box
            )
            ablock_dt = xr.DataTree(children={'.'.join([str(i) for i in bbox_radar]): xr.DataTree(dataset=a_block)})

        # Compute Area Normalization Factor (ANF) by summing GNA contributions
        ANF = xr.DataArray(GNA.reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]).astype("float32") / 2.0

        # Compute local incidence angle (lia) from the cosine
        lia = (
            xr.DataArray(np.arccos(cos_theta).reshape(dim_y, dim_x, 4).sum(axis=2), dims=["y", "x"]).astype("float32")
            / 4.0
        )
    else:
        # If no weights were computed (e.g., empty chunk), create empty arrays
        nlook = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])
        ANF = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])
        lia = xr.DataArray(np.zeros((dim_y, dim_x), dtype="float32"), dims=["y", "x"])

    # Combine geogrid results into a single dataset
    geo_data = xr.Dataset(data_vars={"nlooks": nlook, "anf": ANF, "lia": lia})

    return geo_data, w_dt, layover_dt, ablock_dt


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
    block_id: str,
    plane_orientation: int = 1,
) -> tuple[xr.Dataset, xr.DataArray, xr.DataTree, xr.DataTree] | tuple[None, None, None, None]:
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
        w_dt = xr.DataTree(children={block_id: xr.DataTree(dataset=w)})

        # Save the layover masks (l1 and l2) to the temporary directory
        # File name is based on the sparse block's bounding box
        # data_path=f"{data_path_tmp}/sparse_dir/layover_block_{i}/{'.'.join([str(j) for j in b])}",
        # skip_read=True,  # We don't need to read these back
        # storage_options=get_storage_options(kwargs, key="storage_options_static"),
        layover_dt = xr.DataTree(children={
            f"layover_block_{i}": xr.DataTree()
            for i in range(2)}
        )
        for i, (ll, b) in enumerate([(l1, bbox_l1), (l2, bbox_l2)]):
            ds = build_sparse_array(
                data=ll[0].astype("float32"),
                row=ll[1],
                col=ll[2],
                shape=np.array([(bbox[3] + ymin), (bbox[2] + xmin)]),
                format="CSR",
                chunks=number_of_triang // 4,
            )
            layover_dt[f"layover_block_{i}/{'.'.join([str(j) for j in b])}"] = xr.DataTree(dataset=ds)
        
    # Return the weights dataset and the nlook array
    # If no weights were computed, return None
    return (w, nlook.astype("float32"), w_dt, layover_dt) if nnz > 0 else (None, None, None, None)


def sparse_to_dense(A: xr.Dataset, files_A, files_l1, files_l2, data_path: str, chunk: dict = {}, **kwargs) -> xr.Dataset:
    """
    Converts and accumulates sparse matrix blocks into a single dense xr.DataArray
    for a specified chunk of the radar grid.

    This function is intended to be used with a `map_blocks` operation. It reads
    multiple sparse files (Area, Layover1, Layover2), finds those that
    overlap with the target `chunk`, and sums them into a dense array.

    Parameters
    ----------
    A : xr.Dataset
        An empty xr.Dataset, required by the `map_blocks`
        framework to determine the `chunk` argument.
    files_A : list of str
        A list of file paths to the sparse 'A' (Area) blocks.
    files_l1 : list of str
        A list of file paths to the sparse 'l1' (layover) blocks.
    files_l2 : list of str
        A list of file paths to the sparse 'l2' (layover) blocks.
    data_path : str
        The directory where the resulting dense radar grid
        dataset will be saved.
    chunk : dict or None, optional
        The chunk definition (e.g., {'azimuth_time': slice(...),
        'slant_range_time': slice(...)}) associated with this job.

    Returns
    -------
    xr.Dataset
        The accumulated dense 'anf' and 'layover' arrays as an xr.Dataset.
    """
    # Extract the radar grid boundaries for this target dense chunk
    rg_min = chunk["slant_range_time"].start
    rg_max = chunk["slant_range_time"].stop
    az_min = chunk["azimuth_time"].start
    az_max = chunk["azimuth_time"].stop

    # Initialize empty dense arrays to accumulate results
    A_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")
    l1_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")
    l2_blocks = np.zeros((az_max - az_min, rg_max - rg_min), dtype="float32")

    # Loop over the three file lists and their corresponding destination arrays
    for list_files, dense_array in zip([files_A, files_l1, files_l2], [A_blocks, l1_blocks, l2_blocks]):
        # Iterate through each sparse file
        for f in list_files:
            file = normpath(f)
            # Parse the sparse block's bounding box from its file name
            # Format: .../{az_min}.{az_max}.{rg_min}.{rg_max}
            line = file.split("/")[-1].split(".")
            block_az_min = int(line[0])
            block_az_max = int(line[1])
            block_rg_min = int(line[2])
            block_rg_max = int(line[3])

            # Check if the sparse block overlaps with the target dense chunk
            if (
                (block_az_max >= az_min)
                & (block_az_min <= az_max)
                & (block_rg_max >= rg_min)
                & (block_rg_min <= rg_max)
            ):
                # If it overlaps, read the sparse data
                ds = read(
                    file,
                    consolidated=False,
                    storage_options=get_storage_options(kwargs, "storage_options_dynamic"),
                )
                data = ds.data.values
                indices = ds.indices.values
                indptr = ds.indptr.values

                # Call the kernel to efficiently update the dense array with sparse data
                # This handles adding values to the correct sub-region
                update_dense_array(
                    array=dense_array.ravel(),
                    data=data,
                    indices=indices,
                    indptr=indptr,
                    xmin=rg_min,
                    xmax=rg_max,
                    ymin=az_min,
                    ymax=az_max,
                )

    # Convert layover contribution counts to boolean masks
    # Any pixel with a contribution > 1 is considered layover
    l1_blocks = l1_blocks > 1 + 1e-7  # Use epsilon for float comparison
    l2_blocks = l2_blocks > 1 + 1e-7

    # Convert the dense ANF array to an xr.DataArray with correct dims
    A_blocks = xr.DataArray(A_blocks, dims=["azimuth_time", "slant_range_time"])
    # Combine the two layover masks
    layover_blocks = xr.DataArray(l1_blocks + l2_blocks, dims=["azimuth_time", "slant_range_time"])

    # Create the final dataset for the radar grid chunk
    radar_data = xr.Dataset(data_vars={"anf": A_blocks, "layover": layover_blocks})

    # Write the dense dataset to disk for the specified chunk region
    return radar_data


def compute_geocoded_mask(
    empty_data: xr.Dataset,
    chunk_length: int,
    xsize_global: int,
    data_path_static: str,
    threshold: float = 0.3,
    chunk: dict = {},
    **kwargs,
) -> xr.Dataset:
    """
    Computes the geocoded Gamma Naught for a single chunk.

    This is the worker function called by `map_blocks`. It loads the
    pre-computed sparse weights (`w`) for this chunk, reads the corresponding
    section of the SLC and radar-grid ANF, normalizes the SLC intensity by
    the ANF, performs the sparse matrix multiplication (geocoding), and
    writes the resulting geocoded chunk to the 'geo_grid' Zarr store.

    Parameters
    ----------
    empty_data : xr.Dataset
        The template chunk (used by `map_blocks`
        to determine dimensions).
    slc : xr.DataArray
        The *entire* SLC DataArray. This function will
        slice the required portion.
    chunk_length : int
        The edge size of a square geogrid chunk.
    xsize_global : int
        The total global size of the radar grid along
        the x-axis (range).
    data_path_static : str
        Path to static data (weights 'w', 'radar_grid').
    data_path_tmp : str
        Path to the output 'geo_grid' Zarr store.
    recompute : bool, optional
        If True, triggers re-computation of
        geogrid masks (layover, shadow).
        Defaults to False.
    threshold : float, optional
        Minimum ANF value; pixels below this
        are set to 0 to avoid division by zero.
        Defaults to 1e-3.
    chunk : dict or None, optional
        A dictionary injected by `map_blocks`
        defining the slices for this chunk.

    Returns
    -------
    xr.Dataset
        Geocoded mask data for this chunk.
    """
    so = get_storage_options(kwargs)
    # Get the geogrid dimensions for this chunk (e.g., 512x512)
    dim_x, dim_y = empty_data.x.size - 1, empty_data.y.size - 1

    # Determine the chunk's row/column index from its start position
    row_number, column_number = chunk["y"].start // chunk_length, chunk["x"].start // chunk_length

    # Load the sparse weights file corresponding to this chunk
    f = normpath(f"{data_path_static}/w/{row_number}.{column_number}")
    if exists(f, storage_options=so):
        # Read sparse matrix components (CSR format)
        ds = read(f, as_type="xr.Dataset", consolidated=False, storage_options=so)
        data = ds.data.values
        indices = ds.indices.values
        indptr = ds.indptr.values
        shape = ds.shape.values
        # Get the bounding box of the radar pixels this chunk depends on
        bbox_radar = ds.attrs["radar_bbox"]
        radar_data = read(
            normpath(f"{data_path_static}/radar_grid"),
            mask_and_scale=False,
            storage_options=so,
        ).isel(
            azimuth_time=slice(bbox_radar[0], bbox_radar[1] + 1),
            slant_range_time=slice(bbox_radar[2], bbox_radar[3] + 1),
        )
        mask_layover = radar_data.layover.values.astype("float32")
        mask_shadow = radar_data.shadow.values.astype("float32")
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
        geocoded_mask = xr.Dataset(data_vars={"mask": (["y", "x"], geocoded_mask)})
    else:
        # If no weights were provided, create an empty boolean mask
        size = {k: v.stop - v.start for k, v in chunk.items()}
        geocoded_mask = xr.Dataset(
            data_vars={"mask": (["y", "x"], np.zeros((size["y"] - 1, size["x"] - 1), dtype="int32"))}
        )

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
