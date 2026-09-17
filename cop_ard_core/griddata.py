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
"""
Grid data interpolation: triangulation, barycentric interpolation kernels, and
xarray Dataset interpolation onto regular or arbitrary grids.
"""

# functions decorated with njit are not considered by pytest --cov
from typing import Any, Union, cast

import numpy as np
import scipy.interpolate
import xarray as xr
from numba import njit


@njit(cache=True)
def _swap(edge):
    """
    Ensure a line segment (edge) is sorted by its y-coordinate.

    This helper function takes a 2x2 numpy array representing an edge with
    two points [x, y] and ensures the point with the smaller y-coordinate
    comes first. This is a prerequisite for the scanline intersection algorithm.

    Parameters
    ----------
    edge : np.ndarray
        A 2x2 numpy array where each row is a point [x, y].

    Returns
    -------
    np.ndarray
        The edge, sorted vertically (by the second column).
    """
    # Compare the y-coordinates of the two points.
    if edge[0, 1] > edge[1, 1]:
        # If the first point is higher than the second, swap them.
        return np.array([[edge[1, 0], edge[1, 1]], [edge[0, 0], edge[0, 1]]], dtype=np.float64)
    else:
        # Otherwise, return the edge as is.
        return edge


@njit(cache=True)
def _intersection(y, x0, y0, x1, y1):
    """
    Calculate the intersection of a horizontal scanline with a line segment.

    This function finds the x-coordinate(s) where a horizontal line at a given
    `y` value intersects a line segment defined by points (x0, y0) and (x1, y1).
    It handles oblique, vertical, and horizontal segments.

    Parameters
    ----------
    y : float
        The y-coordinate of the horizontal scanline.
    x0 : float
        The x-coordinate of the first point of the segment.
    y0 : float
        The y-coordinate of the first point of the segment.
    x1 : float
        The x-coordinate of the second point of the segment.
    y1 : float
        The y-coordinate of the second point of the segment.

    Returns
    -------
    xtype : int
        An integer indicating the type of intersection:
        0: No intersection.
        1: A single point intersection (oblique or vertical segment).
        2: A horizontal line segment intersection.
    x_i : np.ndarray
        A numpy array containing the x-coordinate(s) of the intersection.
    """
    # Check if the scanline is outside the vertical range of the segment.
    if (y < y0) or (y > y1):  # out of triangle
        xtype = 0
        x_i = np.empty(0, dtype=np.float64)
    # Handle an oblique (diagonal) segment.
    elif (x0 != x1) and (y0 != y1):  # oblique segment
        xtype = 1
        # Calculate the x-intersection using linear interpolation.
        x_val = (y - y0) / (y1 - y0) * (x1 - x0) + x0
        x_i = np.array([x_val], dtype=np.float64)
    # Handle a vertical segment.
    elif x0 == x1:  # vertical segment
        xtype = 1
        x_i = np.array([x0], dtype=np.float64)
    # Handle a horizontal segment that coincides with the scanline.
    elif y == y1:  # horizontal segment
        xtype = 2
        x_min = min(x0, x1)
        x_max = max(x0, x1)
        # The intersection is all integer points along the segment.
        x_i = np.arange(np.ceil(x_min), np.floor(x_max) + 1, dtype=np.float64)
    # Default case for no intersection.
    else:
        xtype = 0
        x_i = np.empty(0, dtype=np.float64)

    return xtype, x_i


@njit(cache=True)
def _compute_interpolation_matrix(triangle):
    """
    Compute the inverse matrix for barycentric coordinate interpolation.

    This function sets up and inverts a 3x3 matrix derived from the triangle's
    vertex coordinates. This inverse matrix is then used to efficiently calculate
    the barycentric weights for any point inside the triangle.

    Parameters
    ----------
    triangle : np.ndarray
        A 3x3 numpy array where each row is a vertex [x, y, z].
        Only the x and y columns are used.

    Returns
    -------
    np.ndarray
        The 3x3 inverse matrix used for interpolation.
    """
    # Create the matrix [x, y, 1] for the three vertices and compute its inverse.
    inv_matrix = np.linalg.inv(np.hstack((triangle[:, :2], np.ones((3, 1)))))
    return np.ascontiguousarray(inv_matrix)


@njit(cache=True)
def _compute_interpolation_vector(interpolation_matrix: np.ndarray, x_out: float, y_out: float):
    """
    Compute the barycentric weight vector for a point within a triangle.

    This vector, when multiplied by the z-values at the triangle's vertices,
    gives the interpolated z-value at the point (x_out, y_out).

    Parameters
    ----------
    interpolation_matrix : np.ndarray
        The pre-computed inverse matrix from
        `_compute_interpolation_matrix`.
    x_out : float
        The x-coordinate of the point to interpolate.
    y_out : float
        The y-coordinate of the point to interpolate.

    Returns
    -------
    np.ndarray
        A 1D numpy array of three barycentric weights.
    """
    # Calculate the weights by multiplying the point's homogeneous coordinate vector with the inverse matrix.
    return np.array([x_out, y_out, 1], dtype=np.float64) @ interpolation_matrix


@njit(cache=True)
def _interpolate_single_triangle(triangle, interpolated_values):
    """
    Rasterize a single triangle and interpolate values at integer grid points inside it.

    This function implements a scanline rasterization algorithm. It iterates
    vertically over the triangle's y-range, finds the x-intersections of the
    scanline with the triangle's edges, and then interpolates the z-values for
    each integer grid point between the intersections.

    Parameters
    ----------
    triangle : np.ndarray
        A 3xN numpy array where each row represents a vertex
        [x, y, z0, z1, ...].
    interpolated_values : np.ndarray
        A pre-allocated array to store the results.
    n_valid_points : int
        The current count of interpolated points, used as an
        index for storing new results.

    Returns
    -------
    int
        The updated count of total interpolated points.
    """
    # Determine the vertical range (scanlines) to iterate over.
    y_min = np.min(triangle[:, 1])
    y_max = np.max(triangle[:, 1])
    y_s = np.arange(np.ceil(y_min), np.floor(y_max) + 1, dtype=np.int32)

    # Define the three edges of the triangle and sort them by y-coordinate.
    edge_0 = _swap(
        np.array([[triangle[0, 0], triangle[0, 1]], [triangle[1, 0], triangle[1, 1]]], dtype=np.float64),
    )
    edge_1 = _swap(
        np.array([[triangle[1, 0], triangle[1, 1]], [triangle[2, 0], triangle[2, 1]]], dtype=np.float64),
    )
    edge_2 = _swap(
        np.array([[triangle[2, 0], triangle[2, 1]], [triangle[0, 0], triangle[0, 1]]], dtype=np.float64),
    )

    # Compute the interpolation matrix once for the entire triangle.
    i_matrix = _compute_interpolation_matrix(triangle)

    # --- Scanline Algorithm ---
    # Iterate over each horizontal scanline that crosses the triangle.
    for y in y_s:
        j = 0
        x_i = np.zeros(2, dtype=np.float64)
        x_s = np.empty(0, dtype=np.float64)

        # Find the intersection points of the current scanline `y` with each edge.
        for e in [edge_0, edge_1, edge_2]:
            x0, y0, x1, y1 = e[0, 0], e[0, 1], e[1, 0], e[1, 1]
            xtype, x = _intersection(y, x0, y0, x1, y1)
            # If it's a point intersection, store it.
            if xtype == 1:
                x_i[j] = x[0]
                j += 1
                # Once two intersections are found, we have the span for this scanline.
                if j == 2:
                    x_min = min(x_i[0], x_i[1])
                    x_max = max(x_i[0], x_i[1])
                    # Get all integer x-coordinates within the span.
                    x_s = np.arange(np.ceil(x_min), np.floor(x_max) + 1, dtype=np.float64)
                    break
            # If it's a horizontal edge intersection, that defines the entire span.
            elif xtype == 2:
                x_s = x
                break

        # --- Interpolate and Store Results ---
        # For each integer point on the scanline span, perform interpolation.
        for i in range(x_s.shape[0]):
            # Store the integer grid coordinates (x, y).
            if x_s[i] >= 0 and x_s[i] < interpolated_values.shape[1] and y >= 0 and y < interpolated_values.shape[0]:
                interpolated_values[y, int(x_s[i]), :] = (
                    _compute_interpolation_vector(i_matrix, x_s[i], y) @ triangle[:, 2:]
                )

    return


@njit(cache=True)
def _interpolate(triangles, result, bbox):
    """
    Interpolate values for a set of triangles within a given bounding box.

    This function iterates through a list of triangles, performs interpolation
    for each one that overlaps with the bounding box using `_interpolate_single_triangle`,
    and finally filters the results to include only points strictly inside the box.

    Parameters
    ----------
    triangles : np.ndarray
        A 3D numpy array where each slice is a triangle
        with vertices and associated z-values.
    bbox : np.ndarray
        A 4-element array defining the bounding box
        [xmin, ymin, xmax, ymax].

    Returns
    -------
    np.ndarray
        A 2D array where each row is an interpolated point
        [x, y, z0_interp, z1_interp, ...].
    """
    # --- Main Interpolation Loop ---
    # Iterate through each triangle in the input array.
    for i in range(triangles.shape[0]):
        # Reshape the data for a single triangle.
        triangle = np.ascontiguousarray(triangles[i].reshape(-1, 3).T)
        # Get the bounding box of the current triangle.
        min_0 = triangle[:, 0].min()
        min_1 = triangle[:, 1].min()
        max_0 = triangle[:, 0].max()
        max_1 = triangle[:, 1].max()

        # Coarse check: process only triangles that overlap with the target bounding box.
        if (min_0 < bbox[2]) & (min_1 < bbox[3]) & (max_0 >= bbox[0]) & (max_1 >= bbox[1]):
            # Call the single-triangle interpolation function.
            _interpolate_single_triangle(triangle, result)

    return result


def interpolate(points: np.ndarray, values: np.ndarray, bbox: np.ndarray):
    """
    Perform irregular grid interpolation using a custom triangle-based rasterizer.

    This function takes data on an irregular grid (`points` and `values`),
    triangulates the grid, interpolates the values onto a regular integer grid
    defined by a bounding box (`bbox`), and reconstructs the final output maps.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (m, n, 2) representing the
        (y, x) coordinates of the irregular grid.
    values : np.ndarray
        An array of shape (D, m, n) containing the D
        different value layers to be interpolated.
    bbox : np.ndarray
        An array with values (xmin, ymin, xmax, ymax)
        defining the target regular grid.

    Returns
    -------
    np.ndarray
        An array of interpolated values for each map. The output
        shape is (D, ymax-ymin, xmax-xmin).
    """
    # --- Prepare Triangles ---
    xsize = points.shape[0] - 1
    ysize = points.shape[1] - 1
    # Generate a standard triangulation for a rectangular grid of this size.
    triangles_grid = generate_triangles(ysize, xsize).astype("int32")
    # Extract the y and x coordinates from the input points.
    y = points[:, :, 0]
    x = points[:, :, 1]

    # --- Prepare Values for Interpolation ---
    # Use an offset to improve numerical precision during interpolation.
    offset = values[:, 0, 0]
    z = values - offset[:, np.newaxis, np.newaxis]
    # Get the x, y, and z coordinates for the vertices of each triangle.
    x_c = x[triangles_grid[:, :, 0], triangles_grid[:, :, 1]]
    y_c = y[triangles_grid[:, :, 0], triangles_grid[:, :, 1]]
    z_c = z[:, triangles_grid[:, :, 0], triangles_grid[:, :, 1]]
    # Handle multiple z-layers.
    z_c = [z_c[j] for j in range(z_c.shape[0])] if z_c.shape[0] > 1 else [z_c[0]]
    # Combine x, y, and all z-layers into a single array for the Numba function.
    triangles = np.hstack((x_c, y_c, *z_c))

    # --- Call Numba Interpolation Core ---
    results = np.zeros((bbox[3], bbox[2], (triangles.shape[1] - 6) // 3), dtype=np.float64)
    results = _interpolate(triangles, results, bbox)
    results = results.transpose(2, 0, 1)
    results += offset[:, np.newaxis, np.newaxis]
    return results


def generate_triangles(ysize: int, xsize: int):
    """
    Generate a mesh of triangles for a regular grid.

    This function creates a standard triangulation for a grid of size
    (ysize, xsize), where each rectangular cell is split into two triangles.

    Parameters
    ----------
    ysize : int
        The number of cells in the y-dimension.
    xsize : int
        The number of cells in the x-dimension.

    Returns
    -------
    np.ndarray
        A 3D numpy array containing the vertex indices for all triangles.
    """
    # Create flattened arrays of grid indices.
    yy, xx = [j.ravel() for j in np.mgrid[0:ysize, 0:xsize]]
    # Define the vertices for the "lower-left" triangles of each grid cell.
    t1 = np.vstack((xx, yy)).T.reshape(xx.size, 1, 2) + np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    # Define the vertices for the "upper-right" triangles of each grid cell.
    t2 = np.vstack((xx, yy)).T.reshape(xx.size, 1, 2) + np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
    # Combine both sets of triangles and add a placeholder column for z-values.
    return np.append(np.vstack((t1, t2)), np.zeros((t1.shape[0] * 2, 3, 1)), axis=2)


def init_interpolate():
    """
    Pre-compile the Numba-jitted interpolation functions.

    This function serves as a "warm-up" call. By executing the main Numba
    function `_interpolate` with small, dummy inputs, it triggers the
    Just-In-Time (JIT) compilation process. This ensures that the first
    real call to the interpolation functions will be fast, as they will
    already be compiled.

    Parameters
    ----------
    None.

    Returns
    -------
    None.
    """
    # Define minimal dummy inputs to trigger compilation.
    bbox = np.array([0, 0, 2, 2])
    triangles = np.array([[0, 1.5, 2.0, 0, 1.5, 0, 0, 1, 2]])
    results = np.zeros((bbox[3], bbox[2], (triangles.shape[1] - 6) // 3), dtype=np.float64)
    # Call the main Numba function to compile it and its dependencies.
    _interpolate(triangles, results, bbox)
    return


def interp_dataset(
    data: xr.Dataset,
    xi: tuple,
    common_dims: tuple,
    method: str = "cubic",
    bounds_error: bool = False,
    fill_value: float = np.nan,
    points: tuple | None = None,
    indices: tuple | None = None,
    common_dims_shape: tuple | None = None,
    rename: dict = {},
    target_coords: dict | None = None,
) -> xr.Dataset:
    """
    Interpolate all variables in an xarray.Dataset using scipy.interpolate.interpn.

    This is a flexible wrapper around `scipy.interpolate.interpn` that can
    handle multi-dimensional xarray.Dataset objects. It stacks all data variables
    into a single large numpy array, performs the interpolation once, and then
    unpacks the results back into a new xarray.Dataset with the correct
    dimensions and variable names.

    Parameters
    ----------
    data : xr.Dataset
        The input xarray.Dataset containing the data on a regular grid.
    xi : tuple
        A tuple of arrays representing the coordinates of the points at which
        to interpolate the data.
    common_dims : tuple
        A tuple of dimension names that are common to all variables
        and represent the interpolation grid (e.g., ('y', 'x')).
    method : str, optional
        The interpolation method to use (e.g., 'linear', 'cubic', 'nearest').
    bounds_error : bool, optional
        If True, raise an error if interpolation is attempted
        outside the bounds of the input data.
    fill_value : float, optional
        The value to use for points outside the interpolation bounds.
    points : tuple or None, optional
        A tuple of coordinate arrays for the input data grid. If None,
        they are taken from the `data` dataset.
    indices : tuple or None, optional
        If the output is a sparse array, these are the indices to fill.
    common_dims_shape : tuple or None, optional
        The shape of the output grid if `indices` are used.
    rename : dict, optional
        A dictionary to rename dimensions in the output dataset.

    Returns
    -------
    xr.Dataset
        An xarray.Dataset containing all variables interpolated to the new grid.
    """
    stack_data = []
    # If the input grid points are not provided, extract them from the dataset.
    if points is None:
        points = tuple(data[k].values for k in common_dims)

    # --- Stack Data Variables for Interpolation ---
    # Separate variables that only have common_dims from those with extra dimensions.
    dv = [k for k in data.data_vars if set(data[k].dims) != set(common_dims)]
    da: Union[xr.DataArray, xr.Dataset] = data.drop_vars(dv)
    slices: dict[str, Any] = {}
    counter = len(da.data_vars.keys())
    if counter != 0:
        da = da.to_dataarray().transpose(*common_dims, "variable")
        # Keep track of which slice in the final stack belongs to which variable.
        slices = {var: en for en, var in enumerate(da["variable"].values)}
        stack_data.append(da.values)
    for k in dv:
        var = data[k]
        var = var.stack(point=[k for k in var.dims if k not in common_dims])
        stack_data.append(var.values)
        # Record the start and end slice indices for this variable.
        slices[cast(str, k)] = (counter, counter + var.point.size)
        counter += var.point.size

    # Concatenate all data into a single numpy array for interpolation.
    stacked_data = np.concat(stack_data, axis=len(common_dims)).astype(float)

    # --- Perform Interpolation ---
    interpolated_data = scipy.interpolate.interpn(
        points,
        stacked_data,
        xi,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )
    # Move the 'variable' axis to the front for easier slicing.
    interpolated_data = np.moveaxis(interpolated_data, -1, 0)

    # --- Reconstruct Output ---
    # If interpolating to a sparse grid, place the results into a full array.
    if (isinstance(indices, tuple)) and (isinstance(common_dims_shape, tuple)):
        filled_array = np.full((interpolated_data.shape[0], *common_dims_shape), fill_value=fill_value)
        filled_array[:, *indices] = interpolated_data
        interpolated_data = filled_array
    else:
        common_dims_shape = xi[0].shape

    # --- Reconstruct Output Dataset ---
    # Create an empty xarray Dataset to hold the final interpolated results.
    interpolated_dataset = xr.Dataset()
    # Iterate through the dictionary that maps variable names to their slice indices.
    for k in slices:
        # Get the original dimension names for the current variable and apply any renaming.
        dims = [rename.get(d, d) for d in data[k].dims]
        # Check if the variable was a simple 2D variable (slice is an int).
        if isinstance(slices[k], int):
            # If so, simply extract its data slice and create a new DataArray.
            interpolated_dataset[k] = xr.DataArray(interpolated_data[slices[k]], dims=dims)
        # Handle variables that had extra dimensions and were stacked.
        else:
            # Determine the shape of the extra (non-common) dimensions.
            shape = [data[d].size for d in data[k].dims if d not in common_dims]
            # Extract the data for this variable using its start and end slice indices.
            # Reshape the flattened data back to its original multi-dimensional structure.
            interpolated_dataset[k] = xr.DataArray(
                interpolated_data[slices[k][0] : slices[k][1], ...].reshape(*shape, *common_dims_shape),
                dims=dims,
            )

    # --- Assign Coordinates ---
    # 1. Preserve original coordinates for non-common extra dimensions (e.g., polarization)
    for dim in data.dims:
        if dim not in common_dims and dim in data.coords:
            new_dim_name = rename.get(dim, dim)
            interpolated_dataset = interpolated_dataset.assign_coords({new_dim_name: data[dim]})

    # 2. Attach user-provided target coordinates for the interpolated dimensions
    if target_coords is not None:
        # Apply dimension renaming to target_coords if applicable
        renamed_target_coords = {rename.get(k, k): v for k, v in target_coords.items()}
        interpolated_dataset = interpolated_dataset.assign_coords(renamed_target_coords)

    return interpolated_dataset


def interp_dataset_scaled(
    data: xr.Dataset, xi: tuple, common_dims: tuple, method: str = "cubic", target_coords: dict | None = None, **kwargs
) -> xr.Dataset:
    """
    Interpolate a Dataset after min-max scaling each variable to [0, 1] for
    numerical stability, rescaling the results back to the original range.
    """
    data_scaled = data.copy()
    scales = {}

    for var in data_scaled.data_vars:
        v_min = float(data_scaled[var].min())
        v_max = float(data_scaled[var].max())
        v_range = v_max - v_min

        if v_range != 0:
            scales[var] = (v_min, v_range)
            data_scaled[var] = (data_scaled[var] - v_min) / v_range
        else:
            scales[var] = (v_min, 1.0)

    ds_interp = interp_dataset(
        data=data_scaled, xi=xi, common_dims=common_dims, method=method, target_coords=target_coords, **kwargs
    )

    for var, (v_min, v_range) in scales.items():
        ds_interp[var] = (ds_interp[var] * v_range) + v_min

    return ds_interp
