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
Numba-accelerated kernels for radar geocoding: area projection integration,
satellite state propagation, and slant-range geolocation (iterative
radar-to-ground point mapping).
"""

import math

import numba
import numpy as np

# Define a maximum iteration count to prevent infinite loops in edge cases
ITERMAX = 1000
MAXITER = 10


@numba.njit(cache=True)
def area_proj_integrate_segment(
    y1: float,
    y2: float,
    x1: float,
    x2: float,
    length: int,
    width: int,
    w_arr: np.ndarray,
    plane_orientation: int,
):
    """
    Integrate the projected area under a line segment onto a grid using scanline.

    This Numba-accelerated function calculates the contribution of a single line
    segment (part of a polygon edge) to the projected area within each grid cell
    it intersects. It uses a scanline approach, integrating trapezoidal areas
    under the segment within each vertical column of pixels. The results are
    accumulated directly into the `w_arr`.

    Parameters
    ----------
    y1 : float
        Y-coordinate of the line segment's start point.
    y2 : float
        Y-coordinate of the line segment's end point.
    x1 : float
        X-coordinate of the line segment's start point.
    x2 : float
        X-coordinate of the line segment's end point.
    length : int
        The height (number of rows) of the grid.
    width : int
        The width (number of columns) of the grid.
    w_arr : np.ndarray
        A 1D NumPy array representing the flattened 2D grid
        (row-major order). This array is modified in-place to accumulate areas.
    plane_orientation : int
        A multiplier (typically 1 or -1) indicating the
        orientation of the polygon edge relative to the winding order. This
        determines whether the area is added or subtracted.

    Returns
    -------
    None
        The function modifies the `w_arr` in-place.
    """
    # --- Initial Checks & Early Exit ---
    # Perform quick checks to see if the line segment can be skipped entirely.
    if (
        # Check if the line is nearly vertical (slope is too high).
        abs(x2 - x1) < 1e-5
        # Check if the line is completely outside the grid horizontally.
        or (x1 < 0 and x2 < 0)
        or (x1 > width and x2 > width)
        # Check if the line is completely outside the grid vertically.
        or (y1 < 0 and y2 < 0)
        or (y1 > length and y2 > length)
        # Check for NaN coordinates which indicate invalid geometry.
        or math.isnan(x1)
        or math.isnan(x2)
        or math.isnan(y1)
        or math.isnan(y2)
    ):
        # If any check passes, the segment contributes no area, so return early.
        return

    # --- Line Equation Setup ---
    # Calculate the slope and y-intercept (offset) of the line segment.
    slope = (y2 - y1) / (x2 - x1)
    offset = y1 - slope * x1

    # --- Integration Direction Setup ---
    # Determine the direction of integration along the x-axis based on the segment's orientation.
    if x2 > x1:
        # Integrate from left to right.
        x_start = x1
        x_end = x2
        # Use the provided plane orientation directly.
        segment_multiplier = float(plane_orientation)
    else:
        # Integrate from right to left (effectively).
        x_start = x2
        x_end = x1
        # Invert the plane orientation because we are integrating in the opposite x-direction.
        segment_multiplier = float(-plane_orientation)

    # Clip the starting x-coordinate to the left edge of the grid (x=0).
    if x_start < 0:
        x_start = 0.0

    # Small margin to prevent floating point issues at pixel boundaries.
    x_increment_margin = 1e-6
    # Counter to prevent potential infinite loops.
    count = 0

    # --- Main Integration Loop (Scanline Algorithm) ---
    # Iterate along the line segment's projection onto the x-axis.
    # The `compute_weights` function handles the calculation for each step
    # between pixel boundary crossings and updates x_start.
    while (x_start < x_end) and (count < ITERMAX):
        # Call the helper function to compute weights for the current step.
        x_start, count, stop = compute_weights(
            x_start,
            x_end,
            slope,
            offset,
            length,
            width,
            x_increment_margin,
            segment_multiplier,
            w_arr,
            count,
        )
        # If the helper function signals to stop (e.g., reached end or no progress), break the loop.
        if stop:
            break


@numba.njit(cache=True)
def compute_weights(
    x_start: float,
    x_end: float,
    slope: float,
    offset: float,
    length: int,
    width: int,
    x_increment_margin: float,
    segment_multiplier: float,
    w_arr: np.ndarray,
    count: int,
):
    """
    Compute and accumulate area weights for one step of the integration.

    This function calculates the projected area contribution for the portion of
    the line segment between `x_start` and the next pixel boundary crossing (`x_next`).
    It updates the corresponding pixel(s) in `w_arr` and returns the new `x_start`
    for the next iteration.

    Parameters
    ----------
    x_start : float
        The starting x-coordinate for this integration step.
    x_end : float
        The final x-coordinate of the entire line segment.
    slope : float
        The slope of the line segment.
    offset : float
        The y-intercept of the line segment.
    length : int
        The height (number of rows) of the grid.
    width : int
        The width (number of columns) of the grid.
    x_increment_margin : float
        A small margin for calculating y_index.
    segment_multiplier : float
        The multiplier incorporating plane orientation
        and integration direction.
    w_arr : np.ndarray
        The 1D flattened grid array to accumulate weights into.
    count : int
        The current iteration count (for loop breaking).

    Returns
    -------
    tuple[float, int, bool]
        - The updated `x_start` coordinate for the next step (equals `x_next`).
        - The updated iteration `count`.
        - A boolean `stop` flag, indicating whether the main loop should terminate.
    """
    # Increment iteration counter.
    count += 1
    # Calculate the y-coordinate at the start of this step.
    y_start = slope * x_start + offset
    # Calculate y slightly ahead to determine the correct y_index robustly.
    y_start_next = slope * (x_start + x_increment_margin) + offset
    # Get the integer indices of the current pixel column and row.
    x_index = int(math.floor(x_start))
    y_index = int(math.floor(y_start_next))

    # --- Vertical Boundary Checks ---
    # If the line is currently below the grid (y_index < 0)...
    if y_index < 0:
        # ...calculate the x-coordinate where it enters the grid (y=0) and jump there.
        x_start = -offset / slope
        return (x_start, count, False)  # Continue loop from the new x_start.
    # If the line is currently above the grid (y_index >= length)...
    if y_index > length - 1:
        # ...calculate the x-coordinate where it exits the grid (y=length-1) and jump there.
        # Note: using length-1 because y_index represents the row *below* the line at y=length.
        x_start = (length - 1 - offset) / slope
        return (x_start, count, False)  # Continue loop from the new x_start.

    # --- Calculate Next Intersection Point (x_next) ---
    # Find the x-coordinate where the line segment next intersects either a
    # vertical pixel boundary (x = x_index + 1) or a horizontal pixel boundary
    # (y = y_index or y = y_index + 1, depending on slope).
    if abs(slope) < 1e-9:
        # If the slope is near zero (horizontal line), the next intersection is simply the next vertical grid line.
        x_next = float(x_index + 1)
    elif slope > 0:
        # Positive slope: next intersection is the minimum of the next vertical line
        # or the horizontal line *above* the current pixel (y = y_index + 1).
        x_next = min((y_index + 1 - offset) / slope, float(x_index + 1))
    else:  # slope < 0
        # Negative slope: next intersection is the minimum of the next vertical line
        # or the horizontal line *at the bottom* of the current pixel (y = y_index).
        x_next = min((y_index - offset) / slope, float(x_index + 1))

    # Ensure x_next does not go beyond the end of the entire line segment.
    x_next = min(x_next, x_end)

    # --- Termination Checks ---
    # Check if we've made negligible progress or moved outside the grid horizontally.
    if abs(x_start - x_next) < 1e-9 or x_index >= width:
        # If so, signal the main loop to stop.
        return (x_start, count, True)

    # --- Area Calculation and Accumulation ---
    # Calculate the y-coordinate at x_next.
    y_next = slope * x_next + offset

    # Calculate the area of the trapezoid under the line segment within the current pixel (x_index, y_index).
    # y_center represents the average height of the segment within the pixel relative to the pixel's
    # bottom edge (y_index).
    y_center = (y_next + y_start - 2 * y_index) / 2.0
    area = segment_multiplier * (x_next - x_start) * y_center
    # Add this area to the corresponding element in the flattened weight array.
    w_arr[y_index * width + x_index] += area

    # For all pixels *below* the current one in the same column, the segment covers their full height (1.0).
    # Add the rectangular area contribution to these pixels.
    area_below = segment_multiplier * (x_next - x_start)  # Area = width * height = (x_next - x_start) * 1.0
    temp_y_index = y_index - 1
    # Loop downwards from the pixel below the current one.
    while temp_y_index >= 0:
        # Add the area contribution.
        w_arr[temp_y_index * width + x_index] += area_below
        temp_y_index -= 1

    # --- Advance to Next Step ---
    # The end of this step becomes the start of the next step.
    x_start = x_next
    # Signal the main loop to continue.
    return (x_start, count, False)


@numba.njit(cache=True)
def get_workspace(
    y: np.ndarray,
    x: np.ndarray,
    triangs: np.ndarray,
    vertexes: np.ndarray,
    bbox: np.ndarray,
):
    """
    Calculates the maximum required workspace (width and height) for a set of
    polygons that intersect a given bounding box.

    Parameters
    ----------
    y : np.ndarray
        1D NumPy array of y-coordinates (float64).
    x : np.ndarray
        1D NumPy array of x-coordinates (float64).
    triangs : np.ndarray
        1D array defining the polygons as a flattened list of
        vertex indices (int64).
    vertexes : np.ndarray
        The vertex indices to use for each polygon, e.g.,
        [0, 1, 2, 0] for triangles (int64).
    bbox : np.ndarray
        The global bounding box [xmin, ymin, xmax, ymax]
        (int64).

    Returns
    -------
    tuple[int, int, int]
        A tuple containing:
        - max_w: The maximum width of any relevant polygon.
        - max_h: The maximum height of any relevant polygon.
        - workspace: The product max_w * max_h.
    """
    # Get the number of vertices per polygon and the number of polygons
    nv = len(vertexes) - 1
    ntr = len(triangs) // nv
    # Initialize max width and height. These were pointers in the C code.
    max_w = 0
    max_h = 0

    # Loop over each polygon
    for t in range(ntr):
        # --- Find the bounding box of the current polygon ---
        # Initialize with the coordinates of the first vertex
        first_vertex_idx = triangs[nv * t + vertexes[0]]
        ymin = y[first_vertex_idx]
        ymax = y[first_vertex_idx]
        xmin = x[first_vertex_idx]
        xmax = x[first_vertex_idx]
        # Find the min/max coordinates among the other vertices
        for vi in range(1, nv):
            vertex_idx = triangs[nv * t + vertexes[vi]]
            ymin = min(ymin, y[vertex_idx])
            ymax = max(ymax, y[vertex_idx])
            xmin = min(xmin, x[vertex_idx])
            xmax = max(xmax, x[vertex_idx])
        # Floor the results to get integer grid coordinates
        ymin = math.floor(ymin)
        ymax = math.floor(ymax)
        xmin = math.floor(xmin)
        xmax = math.floor(xmax)
        # --- Check for intersection with the global bounding box ---
        # If the polygon's box is completely outside the global box, skip it
        if (xmax < bbox[0]) or (xmin > bbox[2]) or (ymax < bbox[1]) or (ymin > bbox[3]):
            continue
        # --- Update the maximum workspace dimensions ---
        max_w = max(max_w, xmax - xmin + 1)
        max_h = max(max_h, ymax - ymin + 1)

    # Return the results
    workspace = int(max_w * max_h)
    return int(max_w), int(max_h), workspace


@numba.njit(cache=True)
def aggregate_dense_to_sparse(
    w_arr: np.ndarray,
    max_h: int,
    max_w: int,
    xmin: int,
    ymin: int,
    width: int,
    linear_global_index: int,
    xsize_global: int,
    xmin_global: int,
    ymin_global: int,
    xmin_chunk: int,
    ymin_chunk: int,
    bbox: np.ndarray,
    bbox_w: np.ndarray,
    data: np.ndarray,
    col: np.ndarray,
    ind_ptr: np.ndarray,
    nlook: np.ndarray,
    l1_aux: np.ndarray,
    l2_aux: np.ndarray,
    t: int,
    counter: int,
):
    """
    Aggregate weights from a dense workspace array into sparse matrix components.

    This Numba-accelerated function iterates through a dense 2D workspace array
    (`w_arr`), identifies non-zero weights, maps their local coordinates to
    global coordinates, and appends the weight and its global column index to
    the sparse matrix arrays (`data`, `col`). It also updates the row pointer
    array (`ind_ptr`) and accumulates weights into auxiliary buffers (`l1_aux`,
    `l2_aux`) and a total look count (`nlook`).

    Parameters
    ----------
    w_arr : np.ndarray
        The dense 1D workspace array (flattened 2D) containing weights.
    max_h : int
        The height of the relevant portion within `w_arr`.
    max_w : int
        The width of the relevant portion within `w_arr`.
    xmin : int
        The starting column index (offset) within the full grid for `w_arr`.
    ymin : int
        The starting row index (offset) within the full grid for `w_arr`.
    width : int
        The total width of the full (non-workspace) grid.
    linear_global_index : int
        A global linear index offset.
    xsize_global : int
        The global width of the output grid.
    xmin_global : int
        The global minimum x-coordinate offset.
    ymin_global : int
        The global minimum y-coordinate offset.
    xmin_chunk : int
        The starting x-coordinate of the current processing chunk.
    ymin_chunk : int
        The starting y-coordinate of the current processing chunk.
    bbox : np.ndarray
        The bounding box [xmin, ymin, xmax, ymax] of the full output grid.
    bbox_w : np.ndarray
        A bounding box array [ymin, ymax, xmin, xmax] updated with the extent
        of non-zero weights found.
    data : np.ndarray
        The array to store non-zero sparse matrix values. Modified in-place.
    col : np.ndarray
        The array to store column indices for the sparse matrix. Modified in-place.
    ind_ptr : np.ndarray
        The CSR index pointer array. Modified in-place.
    nlook : np.ndarray
        An array accumulating the total weight (looks) for each row (polygon). Modified in-place.
    l1_aux : np.ndarray
        An auxiliary dense buffer for accumulating weights (strategy 1). Modified in-place.
    l2_aux : np.ndarray
        An auxiliary dense buffer for accumulating weights (strategy 2). Modified in-place.
    t : int
        The current row index (polygon index) being processed.
    counter : int
        The current index (count) for appending to `data` and `col` arrays.

    Returns
    -------
    int
        The updated `counter` value.
    """
    # Iterate through the relevant portion of the dense workspace array (height x width).
    for i in range(int(max_h)):
        for j in range(int(max_w)):
            # Calculate the linear index `k` in the flattened `w_arr`.
            k = j + i * int(max_w)
            # Get the absolute weight value.
            weight = abs(w_arr[k])
            # Process only if the weight is significantly non-zero.
            if weight > 1e-12:
                # --- Coordinate Mapping ---
                # Calculate the linear index `index` in the full (non-workspace) grid.
                index = (j + xmin) + (i + ymin) * width
                # Calculate the remapped linear index `reindex` in the global output grid.
                reindex = linear_global_index + (index // width) * xsize_global + (index % width)
                # Convert the global linear index `reindex` back to 2D global coordinates (col_x, col_y).
                col_x = reindex % xsize_global + xmin_global
                col_y = reindex // xsize_global + ymin_global

                # --- Bounding Box Check and Sparse Matrix Update ---
                # Ensure the calculated global coordinates are within the target bounding box for the current chunk.
                if (col_x >= 0) and (col_x < bbox[2] + xmin_chunk) and (col_y >= 0) and (col_y < bbox[3] + ymin_chunk):
                    # Update the bounding box `bbox_w` to track the extent of non-zero weights.
                    bbox_w[0] = min(col_y, bbox_w[0])  # min y
                    bbox_w[1] = max(col_y, bbox_w[1])  # max y
                    bbox_w[2] = min(col_x, bbox_w[2])  # min x
                    bbox_w[3] = max(col_x, bbox_w[3])  # max x

                    # --- Update CSR Sparse Matrix Components ---
                    # Increment the count of non-zero elements for the next row's pointer start.
                    ind_ptr[t + 1] += 1
                    # Store the weight value.
                    data[counter] = weight
                    # Store the global linear column index for the CSR matrix.
                    col[counter] = col_x + col_y * (bbox[2] + xmin_chunk)
                    # Accumulate the total weight (number of looks) for the current polygon `t`.
                    nlook[t] += weight

                    # --- Accumulate into Auxiliary Buffers ---
                    # Distribute weights into two auxiliary buffers based on the polygon index `t`.
                    # This seems to be part of a specific aggregation strategy.
                    if t % 4 == 0 or t % 4 == 3:
                        l1_aux[int(index)] += weight
                    else:
                        l2_aux[int(index)] += weight

                    # Increment the counter for the next non-zero element.
                    counter += 1
    # Finalize the index pointer for the current row `t` by adding the previous row's end pointer.
    ind_ptr[t + 1] += ind_ptr[t]
    # Return the updated counter for non-zero elements.
    return counter


@numba.njit(cache=True)
def area_proj_triangles(
    y: np.ndarray,
    x: np.ndarray,
    triangs: np.ndarray,
    vertexes: np.ndarray,
    bbox: np.ndarray,
    linear_global_index: int,
    length: int,
    width: int,
    xsize_global: int,
    xmin_global: int,
    ymin_global: int,
    xmin_chunk: int,
    ymin_chunk: int,
    radar_grid_ysize: int,
    plane_orientation: int,
    bbox_l1: np.ndarray,
    bbox_l2: np.ndarray,
    bbox_w: np.ndarray,
    expected_nlooks: int,
):
    """
    Rasterize polygons (defined by triangles) onto a grid using area projection.

    This Numba-accelerated function takes a set of polygons, defined by their
    vertices and a triangulation scheme, and calculates the projected area
    contribution of each polygon onto a target grid. It outputs the results in
    multiple sparse matrix formats (CSR) suitable for different downstream
    processing steps.

    Parameters
    ----------
    y : np.ndarray
        Array of y-coordinates for all vertices.
    x : np.ndarray
        Array of x-coordinates for all vertices.
    triangs : np.ndarray
        Flattened array defining the triangles (indices into x, y).
    vertexes : np.ndarray
        Array defining the vertices for each polygon edge loop.
    bbox : np.ndarray
        The bounding box [xmin, ymin, xmax, ymax] of the processing chunk.
    linear_global_index : int
        Global linear index offset for coordinate mapping.
    length : int
        Total height of the full grid.
    width : int
        Total width of the full grid.
    xsize_global : int
        Global width of the output grid.
    xmin_global : int
        Global minimum x-coordinate offset.
    ymin_global : int
        Global minimum y-coordinate offset.
    xmin_chunk : int
        Starting x-coordinate of the current chunk.
    ymin_chunk : int
        Starting y-coordinate of the current chunk.
    radar_grid_ysize : int
        The total height of the final radar grid (used for ind_ptr size).
    plane_orientation : int
        Orientation multiplier (1 or -1).
    bbox_l1 : np.ndarray
        Bounding box for the l1 auxiliary sparse matrix. Updated in-place.
    bbox_l2 : np.ndarray
        Bounding box for the l2 auxiliary sparse matrix. Updated in-place.
    bbox_w : np.ndarray
        Bounding box for the main sparse matrix. Updated in-place.
    expected_nlooks : int
        An estimate of the average number of non-zero elements
        per row, used for pre-allocating sparse arrays.

    Returns
    -------
    tuple
        - (data, ind_ptr, col): CSR components for the main weight matrix.
        - (l1, ind_ptr_l1, col_l1): CSR components for the first auxiliary matrix.
        - (l2, ind_ptr_l2, col_l2): CSR components for the second auxiliary matrix.
        - nlook (np.ndarray): Array of total looks per polygon.
    """
    # --- Initialization ---
    nv = len(vertexes) - 1  # Number of vertices per polygon
    ntr = len(triangs) // nv  # Total number of triangles/polygons

    # --- 1. Workspace Calculation and Allocation ---
    # Determine the required size of the temporary dense workspace array.
    max_w, max_h, work_size = get_workspace(y, x, triangs, vertexes, bbox)

    # --- 2. Allocate Output Arrays ---
    # Proceed only if a valid workspace size is determined (i.e., polygons are within bounds).
    if work_size > 0:
        # Allocate the temporary dense workspace array.
        w_arr = np.zeros(work_size, dtype="float64")
        # Allocate auxiliary dense accumulation buffers.
        l1_aux, l2_aux = np.zeros(length * width, dtype="float64"), np.zeros(length * width, dtype="float64")
        # Pre-allocate sparse matrix component arrays with an estimated size.
        data = np.empty(ntr * (expected_nlooks + 5), dtype="float64")
        col = np.empty(ntr * (expected_nlooks + 5), dtype="int64")
        ind_ptr, nlook = np.zeros(ntr + 1, dtype="int64"), np.zeros((ntr), dtype="float64")
        counter = 0  # Counter for non-zero elements in the main sparse matrix.

        # --- 3. Main Loop: Iterate Over Polygons ---
        for t in range(ntr):
            # Calculate the integer bounding box (ymin, ymax, xmin, xmax) for the current polygon.
            first_idx = triangs[nv * t + vertexes[0]]
            ymin, ymax = y[first_idx], y[first_idx]
            xmin, xmax = x[first_idx], x[first_idx]
            for vi in range(1, nv):
                idx = triangs[nv * t + vertexes[vi]]
                ymin, ymax = min(ymin, y[idx]), max(ymax, y[idx])
                xmin, xmax = min(xmin, x[idx]), max(xmax, x[idx])
            # Convert float bounds to integer bounds covering the polygon.
            ymin, ymax = math.floor(ymin), math.floor(ymax)
            xmin, xmax = math.floor(xmin), math.floor(xmax)

            # Skip this polygon if its bounding box is completely outside the processing chunk's bbox.
            if xmax < bbox[0] or xmin > bbox[2] or ymax < bbox[1] or ymin > bbox[3]:
                # Ensure ind_ptr is correctly incremented even for skipped rows.
                ind_ptr[t + 1] += ind_ptr[t]
                continue

            # --- 4. Rasterize Polygon into the temporary workspace `w_arr` ---
            # Iterate through the edges of the polygon.
            for vi in range(nv):
                # Get the indices of the start and end vertices of the current edge.
                idx1, idx2 = triangs[nv * t + vertexes[vi]], triangs[nv * t + vertexes[vi + 1]]
                # Call the segment integration function to rasterize this edge into w_arr.
                # Coordinates are relative to the polygon's integer bounding box (xmin, ymin).
                area_proj_integrate_segment(
                    y[idx1] - ymin,
                    y[idx2] - ymin,
                    x[idx1] - xmin,
                    x[idx2] - xmin,
                    int(max_h),
                    int(max_w),
                    w_arr,
                    plane_orientation,
                )

            # --- 5. Aggregate Results from `w_arr` into Sparse Format ---
            # Transfer the weights from the dense workspace `w_arr` to the sparse matrix arrays.
            counter = aggregate_dense_to_sparse(
                w_arr,
                max_h,
                max_w,
                xmin,
                ymin,
                width,
                linear_global_index,
                xsize_global,
                xmin_global,
                ymin_global,
                xmin_chunk,
                ymin_chunk,
                bbox,
                bbox_w,
                data,
                col,
                ind_ptr,
                nlook,
                l1_aux,
                l2_aux,
                t,
                counter,
            )
            # Reset the workspace array for the next polygon.
            w_arr.fill(0)

        # --- 6. Post-Loop Processing: Build l1/l2 Sparse Data ---
        # Allocate arrays for the auxiliary sparse matrices (l1, l2).
        l1, l2 = np.empty(length * width, dtype="float64"), np.empty(length * width, dtype="float64")
        col_l1, col_l2 = np.empty(length * width, dtype="int64"), np.empty(length * width, dtype="int64")
        # Initialize index pointer arrays for the auxiliary CSR matrices.
        ind_ptr_l1, ind_ptr_l2 = (
            np.zeros(bbox[3] + ymin_chunk + 1, dtype="int64"),
            np.zeros(bbox[3] + ymin_chunk + 1, dtype="int64"),
        )
        counter_l1, counter_l2 = 0, 0
        # Iterate through the dense auxiliary buffers (l1_aux, l2_aux).
        for i in range(length * width):
            # Map the linear index `i` back to global 2D coordinates.
            reindex = linear_global_index + (i // width) * xsize_global + (i % width)
            col_x, col_y = reindex % xsize_global + xmin_global, reindex // xsize_global + ymin_global
            # If there's a non-zero weight in l1_aux, add it to the l1 sparse components.
            if l1_aux[i] != 0:
                l1[counter_l1] = l1_aux[i]
                col_l1[counter_l1] = col_x
                ind_ptr_l1[col_y + 1] += 1  # Increment count for this row.
                # Update the bounding box for the l1 matrix.
                bbox_l1[0], bbox_l1[1] = min(col_y, bbox_l1[0]), max(col_y, bbox_l1[1])
                bbox_l1[2], bbox_l1[3] = min(col_x, bbox_l1[2]), max(col_x, bbox_l1[3])
                counter_l1 += 1
            # If there's a non-zero weight in l2_aux, add it to the l2 sparse components.
            if l2_aux[i] != 0:
                l2[counter_l2] = l2_aux[i]
                col_l2[counter_l2] = col_x
                ind_ptr_l2[col_y + 1] += 1  # Increment count for this row.
                # Update the bounding box for the l2 matrix.
                bbox_l2[0], bbox_l2[1] = min(col_y, bbox_l2[0]), max(col_y, bbox_l2[1])
                bbox_l2[2], bbox_l2[3] = min(col_x, bbox_l2[2]), max(col_x, bbox_l2[3])
                counter_l2 += 1
        # Compute the cumulative sum for the index pointer arrays (final step of CSR construction).
        for i in range(radar_grid_ysize):
            ind_ptr_l1[i + 1] += ind_ptr_l1[i]
            ind_ptr_l2[i + 1] += ind_ptr_l2[i]

        # --- 7. Finalize and Return ---
        # Trim the pre-allocated sparse arrays to their actual size.
        data, col = data[:counter], col[:counter]
        l1, col_l1 = l1[:counter_l1], col_l1[:counter_l1]
        l2, col_l2 = l2[:counter_l2], col_l2[:counter_l2]
    else:
        # If work_size was 0, return empty arrays.
        data, col, ind_ptr = np.empty(0, dtype="float64"), np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
        l1, col_l1, ind_ptr_l1 = np.empty(0, dtype="float64"), np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
        l2, col_l2, ind_ptr_l2 = np.empty(0, dtype="float64"), np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
        nlook = np.zeros((ntr), dtype="float64")  # Ensure nlook is returned even if empty

    # Return the components of the three sparse matrices and the total looks per polygon.
    return (data, ind_ptr, col), (l1, ind_ptr_l1, col_l1), (l2, ind_ptr_l2, col_l2), nlook


@numba.njit(cache=True)
def computeGammaNaughtArea(
    triang_points: np.ndarray,
    radar_distance: np.ndarray,
    radar_grid_pixel_area: float,
    num_triangles: int,
    normal_order: bool,
    GNA: np.ndarray,
    lia: np.ndarray,
    data_w: np.ndarray,
    indptr_w: np.ndarray,
    indices_w: np.ndarray,
    n_rows: int,
    n_cols: int,
    dense: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    width: int,
    length: int,
    bbox: np.ndarray,
):
    """
    Compute Gamma Naught Area (GNA) for DEM triangles and aggregate results.

    This Numba-accelerated function calculates the Gamma Naught Area normalization
    factor for each DEM triangle based on its surface normal, the radar viewing
    geometry, and the radar pixel area. It then distributes these GNA values onto
    a dense grid weighted by the projected area contribution of each triangle
    (provided as a sparse matrix `W`). Finally, it converts the resulting dense
    GNA map back into a sparse CSR matrix format.

    Parameters
    ----------
    triang_points : np.ndarray
        Flattened array of triangle vertex coordinates
        (3 vertices * 3 coordinates = 9 values per triangle).
    radar_distance : np.ndarray
        Flattened array of radar-to-vertex distance vectors
        (3 values per triangle, representing the vector to one vertex).
    radar_grid_pixel_area : float
        The area of a single pixel in the radar grid.
    num_triangles : int
        The total number of triangles to process.
    normal_order : bool
        If True, computes normal using `cross(e0, e1)`, else `cross(e1, e0)`.
    GNA : np.ndarray
        1D array to store the computed Gamma Naught Area for each triangle. Modified in-place.
    lia : np.ndarray
        1D array to store the cosine of the local incidence angle for each triangle.
        Modified in-place.
    data_w : np.ndarray
        Non-zero values of the sparse weight matrix `W` (triangle projected area).
    indptr_w : np.ndarray
        Index pointer array for the sparse weight matrix `W` (CSR format).
    indices_w : np.ndarray
        Column indices array for the sparse weight matrix `W` (CSR format).
    n_rows : int
        Number of rows in the sparse weight matrix `W` (number of triangles).
    n_cols : int
        Number of columns in the sparse weight matrix `W` (flattened radar grid size).
    dense : np.ndarray
        A pre-allocated dense 1D array (workspace) to accumulate the weighted GNA values.
        Modified in-place.
    data : np.ndarray
        Pre-allocated array to store the non-zero values of the final sparse GNA matrix.
        Modified in-place.
    indices : np.ndarray
        Pre-allocated array to store the column indices of the final sparse GNA matrix.
        Modified in-place.
    indptr : np.ndarray
        Pre-allocated array for the index pointer of the final sparse GNA matrix.
        Modified in-place.
    width : int
        Width of the radar grid.
    length : int
        Height of the radar grid.
    bbox : np.ndarray
        Bounding box array [ymin, ymax, xmin, xmax] updated with the extent of non-zero GNA values.

    Returns
    -------
    int
        The number of non-zero elements (`count`) in the final sparse GNA matrix.
    """
    # Pre-allocate arrays for edge vectors.
    e0 = np.zeros(3, dtype=float)
    e1 = np.zeros(3, dtype=float)

    # --- Step 1: Compute GNA and LIA per Triangle ---
    for i in range(num_triangles):
        # Extract the coordinates for the three vertices of the current triangle.
        # Compute the two edge vectors (e0, e1) originating from the first vertex.
        for j in range(3):
            e0[j] = triang_points[9 * i + 3 + j] - triang_points[9 * i + j]  # Vertex 2 - Vertex 1
            e1[j] = triang_points[9 * i + 6 + j] - triang_points[9 * i + j]  # Vertex 3 - Vertex 1

        # Compute the surface normal vector to the triangle using the cross product.
        # The order depends on `normal_order` to ensure consistent orientation.
        normal = np.cross(e0, e1) if normal_order else np.cross(e1, e0)
        # Calculate the magnitude (length) of the normal vector (proportional to triangle area).
        normal_module = np.linalg.norm(normal)
        # Calculate the magnitude of the radar distance vector (approx. slant range).
        radar_distance_module = np.linalg.norm(radar_distance[3 * i : 3 * i + 3])

        # Compute the dot product between the radar distance vector and the normal vector.
        s = 0
        for j in range(3):
            s += radar_distance[3 * i + j] * normal[j]

        # Calculate the cosine of the local incidence angle (theta).
        cos_theta = s / (radar_distance_module * normal_module)
        # Store the cosine of the local incidence angle.
        lia[i] = cos_theta

        # Calculate Gamma Naught Area (GNA) only if the angle is reasonable (not grazing).
        # GNA = (Projected Area in Slant Range) / (Ground Area) * Normalization
        if cos_theta > 0.05:  # Threshold to avoid division by zero or large values at grazing angles.
            GNA[i] = (cos_theta * normal_module) / (2.0 * radar_grid_pixel_area)
            # Note: normal_module / 2.0 is the triangle area in 3D space.
            # cos_theta * (normal_module / 2.0) is the area projected onto the slant range plane.
            # Dividing by radar_grid_pixel_area normalizes by the radar pixel size.

    # --- Step 2: Distribute GNA onto Dense Grid using Sparse Weights W ---
    # Iterate through each row (triangle) of the sparse weight matrix W.
    for i in range(n_rows):
        nlook = 0.0
        for j in range(indptr_w[i], indptr_w[i + 1]):
            nlook += data_w[j]

        if nlook > 0.0:
            # Iterate through the non-zero elements (projected areas) for this triangle.
            for k in range(indptr_w[i], indptr_w[i + 1]):
                # `indices_w[k]` is the linear index of the radar grid pixel.
                # `data_w[k]` is the projected area of triangle `i` onto pixel `indices_w[k]`.
                # `nlook` is the total projected area of triangle `i`.
                # We distribute the triangle's GNA value proportionally to its projected area in each pixel.
                # The division by 2.0 might be related to specific definitions or conventions.
                dense[indices_w[k]] += GNA[i] * (data_w[k] / nlook) / 2.0

    # --- Step 3: Convert Dense GNA Map to Sparse CSR Format ---
    count = 0  # Counter for non-zero elements in the output sparse matrix.
    # Iterate through each pixel index `i` of the dense GNA map.
    for i in range(n_cols):
        # If the accumulated GNA value in the dense array is significant...
        if dense[i] > 1e-13:
            # ...store the value and its coordinates in the sparse matrix components.
            data[count] = dense[i]
            # Calculate the 2D column index (`indices[count]`) and row index (`row`).
            indices[count] = i % width  # Column index (x)
            row = i // width  # Row index (y)
            # Increment the count for the next row's starting pointer in the CSR format.
            indptr[row + 1] += 1
            # Update the bounding box `bbox` to track the extent of non-zero values.
            bbox[0] = np.minimum(row, bbox[0])  # min y
            bbox[1] = np.maximum(row, bbox[1])  # max y
            bbox[2] = np.minimum(indices[count], bbox[2])  # min x
            bbox[3] = np.maximum(indices[count], bbox[3])  # max x
            # Increment the non-zero element counter.
            count += 1

    # Compute the final cumulative sum for the CSR index pointer array.
    for i in range(length):
        indptr[i + 1] += indptr[i]

    # Return the total count of non-zero elements found.
    return count


@numba.njit(cache=True)
def update_dense_array(
    array: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
):
    """
    Update a dense array by adding values from a sparse CSR matrix within a bounding box.

    This Numba-accelerated function efficiently adds the contributions from a sparse
    matrix (representing, e.g., projected area or GNA) to a dense 2D array, but
    only considers the elements within a specified rectangular region (bounding box).

    Parameters
    ----------
    array : np.ndarray
        The 1D flattened dense array to be updated in-place.
    data : np.ndarray
        The non-zero values of the sparse CSR matrix.
    indices : np.ndarray
        The column indices of the sparse CSR matrix.
    indptr : np.ndarray
        The index pointer array of the sparse CSR matrix.
    xmin : int
        The minimum column index (inclusive) of the update region.
    xmax : int
        The maximum column index (exclusive) of the update region.
    ymin : int
        The minimum row index (inclusive) of the update region.
    ymax : int
        The maximum row index (exclusive) of the update region.

    Returns
    -------
    None
        The `array` is modified in-place.
    """
    # Calculate the width of the update region.
    width = xmax - xmin
    # Iterate through the rows within the specified bounding box [ymin, ymax).
    for i in range(ymin, ymax):
        # Iterate through the non-zero elements in the current row `i` of the sparse matrix.
        for j in range(indptr[i], indptr[i + 1]):
            # Get the column index `col` for the current non-zero element.
            col = indices[j]
            # Check if the column index falls within the horizontal bounds [xmin, xmax).
            if (col >= xmin) and (col < xmax):
                # If within bounds, calculate the linear index in the dense `array`
                # (relative to the bounding box origin) and add the sparse value.
                array[(i - ymin) * width + (col - xmin)] += data[j]
    return


@numba.njit(cache=True)
def csr_times_dense(
    n_dims: int,
    n_rows: int,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    radar_image: np.ndarray,
    geocoded_image: np.ndarray,
    ystart: int,
    xstart: int,
    xsize_global: int,
    ysize_block: int,
    xsize_block: int,
):
    """
    Perform a sparse (CSR) matrix multiplication with a dense vector/image.

    This Numba-accelerated function implements the core resampling operation, effectively
    multiplying a sparse matrix `W` (representing the area projection weights)
    by a dense vector `R` (the flattened radar image) to produce a dense vector `G`
    (the flattened geocoded image): `G = W @ R`. The operation is optimized for
    block processing.

    Parameters
    ----------
    n_dims : int
        Number of dimensions (e.g., polarizations) in the radar/geocoded images.
    n_rows : int
        Number of rows in the sparse matrix (number of geocoded pixels).
    data : np.ndarray
        Non-zero values of the sparse weight matrix `W`.
    indices : np.ndarray
        Column indices of `W` (linear index into the radar image).
    indptr : np.ndarray
        Index pointer array of `W`.
    radar_image : np.ndarray
        The flattened dense input radar image block `R`.
    geocoded_image : np.ndarray
        The flattened dense output geocoded image block `G`. Modified in-place.
    ystart : int
        Starting row index of the current block in the global geocoded grid.
    xstart : int
        Starting column index of the current block in the global geocoded grid.
    xsize_global : int
        Total width of the global radar grid.
    ysize_block : int
        Height of the current processing block.
    xsize_block : int
        Width of the current processing block.

    Returns
    -------
    None
        The `geocoded_image` array is modified in-place.
    """
    # Iterate through each row `i` of the sparse matrix (each pixel in the output geocoded image block).
    for i in range(n_rows):
        # --- Calculate Normalization Factor (Total Weight) ---
        # Sum the weights (projected areas) for the current output pixel `i`.
        nlook = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            nlook += data[j]

        # --- Weighted Summation (Matrix Multiplication) ---
        # Iterate again through the non-zero weights for the current output pixel `i`.
        for j in range(indptr[i], indptr[i + 1]):
            # `idx` is the linear index into the *global* radar image grid.
            idx = indices[j]
            # Convert the global radar index `idx` to local block coordinates (x, y).
            x = idx % xsize_global - xstart
            y = idx // xsize_global - ystart
            # Calculate the linear index `linear_coord` within the *local* radar image block.
            linear_coord = y * xsize_block + x
            # Calculate the normalized weight (contribution of this radar pixel to the geocoded pixel).
            val = data[j] / nlook

            # --- Accumulate Result for Each Dimension ---
            # Iterate through each dimension (e.g., polarization).
            for k in range(n_dims):
                # Calculate the starting linear index for dimension `k` in the flattened images.
                index_1 = k * n_rows  # Offset in the geocoded image array
                index_2 = k * xsize_block * ysize_block  # Offset in the radar image array
                # Perform the core operation: geocoded_pixel += weight * radar_pixel.
                geocoded_image[index_1 + i] += val * radar_image[index_2 + linear_coord]
    return


@numba.njit(cache=True)
def compute_incidence_angle_ground(
    radar_distance: np.ndarray,
    points: np.ndarray,
    incidence_angle_ground: np.ndarray,
    num_points: int,
):
    """
    Compute the ground incidence angle at each ground point from the satellite
    position, storing results in the provided output array.
    """
    pos_satellite = np.zeros(3, dtype=float)
    for i in range(num_points):
        for j in range(3):
            pos_satellite[j] = radar_distance[3 * i + j] - points[3 * i + j]
        s = 0
        for j in range(3):
            s += radar_distance[3 * i + j] * pos_satellite[j]

        radar_distance_module = np.linalg.norm(radar_distance[3 * i : 3 * i + 3])
        pos_satellite_module = np.linalg.norm(pos_satellite)
        incidence_angle_ground[i] = np.arccos(s / (radar_distance_module * pos_satellite_module))

    return


def gaussian_kernel(size: int = 5, sigma: float = 1.0):
    """
    Generates a 2D Gaussian kernel for image smoothing.

    Parameters
    ----------
    size : int, optional
        The size (width and height) of the kernel. Should be odd. Defaults to 5.
    sigma : float, optional
        The standard deviation of the Gaussian distribution. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the normalized Gaussian kernel.
    """
    # Create a 1D array of coordinates centered around 0.
    x = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    # Calculate the 1D Gaussian function values.
    gauss = np.exp(-0.5 * x**2 / sigma**2)
    # Create the 2D kernel by taking the outer product of the 1D Gaussian with itself.
    kernel = np.outer(gauss, gauss)
    # Normalize the kernel so that its elements sum to 1.
    return kernel / np.sum(kernel)


@numba.njit(cache=True)
def polyval(t: float, coeffs: np.ndarray):
    """Evaluate a polynomial at a given set of values t."""
    result = np.zeros(coeffs.shape[1])  # Initialize result as a scalar
    for i in range(coeffs.shape[0] - 1, -1, -1):  # Iterate over all coefficients
        result = result * t + coeffs[i]  # Accumulate polynomial evaluation

    return result


@numba.njit(cache=True)
def numba_dot(a, b):
    """
    Compute the dot product of two 1D arrays (Numba JIT-compiled).
    """
    result = 0.0
    for j in range(a.size):
        result += a[j] * b[j]

    return result


@numba.njit(cache=True)
def main_loop(pos, vel, acc, dem, t, direction_ecef, r, conv_th):
    """
    Iteratively solve the slant-range geolocation equations per pixel,
    converging on the azimuth time at which the satellite-to-ground range
    matches, updating time, range vector, and velocity direction outputs.
    """
    corr = 0
    tsol = t[t.shape[0] // 2, t.shape[1] // 2]
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            torig = tsol
            t[i, j] = tsol + corr
            for k in range(MAXITER):
                # compute start point

                p = polyval(t[i, j], pos)
                v = polyval(t[i, j], vel)
                a = polyval(t[i, j], acc)

                # compute range vector
                r[i, j, :] = p - dem[i, j, :]

                # update time
                delta = numba_dot(v, r[i, j, :]) / (numba_dot(a, r[i, j, :]) + numba_dot(v, v))

                t[i, j] = t[i, j] - delta
                direction_ecef[i, j, :] = v / numba_dot(v, v) ** 0.5

                maxcorr = np.abs(delta)
                if maxcorr < conv_th:
                    corr = t[i, j] - torig
                    tsol = t[i, j]
                    break
