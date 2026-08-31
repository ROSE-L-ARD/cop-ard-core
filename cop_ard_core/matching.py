from typing import Literal, Sequence

import numpy as np
import xarray as xr
from scipy.signal import convolve2d

from cop_ard_core.logger import logger


def get_corr_sub_pix(
    corr: np.ndarray, dim: int | Sequence[int] | None = None, epsylon: float = 1e-14, epsylon_2: float = 1e-5
) -> np.ndarray:
    """Compute sub-pixel offset interpolating the correlation matrix around the maximum

    Parameters
    - corr: correlation matrx (3D data cube)
    - dim: size of the box to be considered around the maximum to interpolate the correlation matrix

    Output:
    - sol:
        + Position of the peak
            - (dy, dx)
        + Quality coefficient:
            - coherence, ellipse axes size
    """

    def get_model_matrix(dim: Sequence[int]) -> np.ndarray | None:
        """
        Compute the model matrix as polinomial function with linear combination of the positions x

        Parameters:
        - dim: size of the patch to be considered to evaluate the sub pixel offset


        Returns:
        - M: array: model matrix
        """
        n = np.sum([e > 0 for e in dim])
        x = np.vstack(
            [
                e.ravel()
                for e in np.meshgrid(
                    np.arange(2 * dim[0] + 1) - dim[0],
                    np.arange(2 * dim[1] + 1) - dim[1],
                )
            ]
        ).T
        M = np.ones(((2 * dim[0] + 1) * (2 * dim[1] + 1), 3 * n), dtype="float64")

        if n == 2:
            M[:, 0] = x[:, 0] ** 2
            M[:, 1] = x[:, 0] * x[:, 1]
            M[:, 2] = x[:, 1] ** 2
            M[:, 3] = x[:, 0]
            M[:, 4] = x[:, 1]
        elif n == 1:
            i = 0 if dim[0] > 0 else 1
            M[:, 0] = x[:, i] ** 2
            M[:, 1] = x[:, i]
        else:
            M = None
        return M

    if dim is None:
        dim = [1, 1]

    if isinstance(dim, int):
        dim = [dim, dim]

    M = get_model_matrix(dim)
    sol = np.zeros((corr.shape[0], 5), dtype=float)

    for i in range(corr.shape[0]):
        imax = np.argmax(corr[i, :, :])
        ymax = imax // corr.shape[2]
        xmax = imax % corr.shape[2]
        if (corr[i, ymax, xmax] > (1 - epsylon_2)) or (M is None):
            sol[i, :] = np.array(
                [
                    ymax - corr.shape[1] // 2,
                    xmax - corr.shape[2] // 2,
                    corr[i, ymax, xmax],
                    0,
                    0,
                ]
            )
            continue

        if (ymax < dim[0]) or (ymax >= corr.shape[1] - dim[0]) or (xmax < dim[1]) or (xmax >= corr.shape[2] - dim[1]):
            continue

        # select points around the integer maximum position
        cc = corr[i, ymax - dim[0] : ymax + dim[0] + 1, xmax - dim[1] : xmax + dim[1] + 1].T.ravel()

        # fit a paraboloid
        c = np.linalg.pinv(M * cc[..., None]).dot(cc**2)

        # compute the position of the paraboloid vertex
        den = 4 * c[0] * c[2] - c[1] ** 2

        # compute semi-axes
        sy = np.sqrt((c[2] - c[0]) ** 2 + c[1] ** 2) / 2
        sx = -(c[2] + c[0]) / 2

        # check if fit is ok
        if (den > 0) and (c[2] <= 0):
            # compute sub-pixel offset
            if c[2] == 0:
                dy = -c[3] / (2 * c[0])
            else:
                dy = (-2 * c[2] * c[3] + c[1] * c[4]) / den

            if c[0] == 0:
                dx = -c[4] / (2 * c[2])
            else:
                dx = (-2 * c[0] * c[4] + c[1] * c[3]) / den

            if (np.abs(dy) > 1) or (np.abs(dx) > 1):
                continue

            # return dy, dx, coemax, paraboloid eigenvalues
            sol[i, :] = np.array(
                [
                    ymax - corr.shape[1] // 2 + dy,
                    xmax - corr.shape[2] // 2 + dx,
                    cc[cc.size // 2],
                    1 / np.sqrt(sx - sy),
                    1 / np.sqrt(sx + sy),
                ]
            )
        elif np.abs(den) < epsylon:
            if c[2] == 0:
                dy = -c[3] / (2 * c[0])
                dx = 0
            elif c[0] == 0:
                dx = -c[4] / (2 * c[2])
                dy = 0
            else:
                dx = 0
                dy = 0

            sol[i, :] = np.array(
                [
                    ymax - corr.shape[1] // 2 + dy,
                    xmax - corr.shape[2] // 2 + dx,
                    cc[cc.size // 2],
                    1 / np.sqrt(sx - sy),
                    1 / np.sqrt(sx + sy),
                ]
            )
    return sol


def forward_flatten(
    stack: np.ndarray,
    reference: np.ndarray,
    wsearch: list[int],
    npt: int,
    nco: int,
    nim: int,
    shp: list[int],
):
    """
    i1                      i2

            |  0  |                     |  0  |  1  | ...... |  npt-1  |
    --------|-----|         ------------|-----|-----|        |---------|
    p(0)    |     |         0           |     |     |        |         |
    --------|-----|         ------------|-----|-----|        |---------|
    p(1)    |     |         1           |     |     |        |         |
    --------|-----|         ------------|-----|-----|        |---------|
    p(2)    |     |         2           |     |     |        |         |
    --------|-----|         ------------|-----|-----|        |---------|
    .       |     |         .           |     |     |        |         |
    .       |     |         .           |     |     |        |         |
    .       |     |         .           |     |     |        |         |
    --------|-----|         ------------|-----|-----|        |---------|
    p(npt-1)|     |         nco x nim -1|     |     |        |         |
    --------|-----|         ------------|-----|-----|        |---------|
    """
    i1 = np.empty((npt, 1), dtype=stack.dtype)
    i2 = np.empty((nco * nim, npt), dtype=stack.dtype)

    # flatten reference image
    i1[:, 0] = reference[
        wsearch[0] : (-wsearch[0] if wsearch[0] > 0 else None),
        wsearch[1] : (-wsearch[1] if wsearch[1] > 0 else None),
    ].ravel()

    # flatten secondary images
    for i_z in range(stack.shape[2]):
        for count in range(nco):
            i_y = count // (2 * wsearch[1] + 1)
            i_x = count % (2 * wsearch[1] + 1)
            i2[count + i_z * nco, :] = stack[i_y : i_y + shp[0], i_x : i_x + shp[1], i_z].ravel()

    return i1, i2


def backward_flatten(
    stack: np.ndarray,
    reference: np.ndarray,
    wsearch: list[int],
    npt: int,
    nco: int,
    nim: int,
    shp: list[int],
):
    i1 = np.empty((npt, nco), dtype=stack.dtype)
    i2 = np.empty((nim, npt), dtype=stack.dtype)

    # flatten secondary images
    for count in range(nco):
        i_y = count // (2 * wsearch[1] + 1)
        i_x = count % (2 * wsearch[1] + 1)
        i1[:, count] = reference[i_y : i_y + shp[0], i_x : i_x + shp[1]].ravel()

    # flatten reference image
    for i_z in range(stack.shape[2]):
        i2[i_z, :] = stack[
            wsearch[0] : (-wsearch[0] if wsearch[0] > 0 else None),
            wsearch[1] : (-wsearch[1] if wsearch[1] > 0 else None),
            i_z,
        ].ravel()

    return i1, i2


def shift_update(
    i: xr.DataArray,
    i0: xr.DataArray,
    x: xr.DataArray,
    y: xr.DataArray,
    kernel: np.ndarray,
) -> xr.DataArray:

    # Remove mean, convert to float64, and remove degenerate dimensions
    o = (i - i.mean()).astype("float64").squeeze()

    # Perform 2D convolution using scipy
    # mode='same' ensures the output shape matches the input 'o'
    # boundary='symm' (symmetrical) minimizes edge artifacts compared to zero-padding
    # .fillna(0) prevents any existing NaNs in the SAR grid from bleeding into the convolution
    o_np_conv = convolve2d(o.fillna(0).values, kernel, mode="same", boundary="symm")

    # Reconstruct the DataArray to utilize xarray's advanced nearest-neighbor selection
    o_conv = xr.DataArray(o_np_conv, coords=o.coords, dims=o.dims)

    # Sample at the specified azimuth and slant range coordinates
    o_sampled = o_conv.sel(azimuth_time=y, slant_range_time=x, method="nearest")

    # Split the sampled array in half along the point dimension
    half_idx = o_sampled.point.size // 2
    o1 = o_sampled.isel(point=slice(half_idx))
    o2 = o_sampled.isel(point=slice(half_idx, None))

    diff_data = o1.data - o2.data

    # Cast back to the original datatype (e.g., timedelta64[ns]) and wrap in a DataArray
    diff = xr.DataArray(diff_data, coords=o1.coords, dims=o1.dims).astype(i0.dtype).compute()
    # Return the final shifted array
    return diff - i0


def fix_disparity(
    burst_id: str,
    acquisition_id: str,
    disparity: xr.DataArray,
    drg: xr.DataArray,
    daz: xr.DataArray,
    cohe_th: float = 0.3,
    kernel_shape: list[int] = [15, 15],
    **kwargs,
):
    # kernel creation for drg and daz multilooking #
    kernel = np.ones(kernel_shape) / np.prod(kernel_shape)

    # image position correction #
    disparity = disparity.squeeze().compute()
    uv = disparity.groupby(disparity.sel(variable="coherence") >= cohe_th)[True]
    dy, dx = (uv.sel(variable=["dy", "dx"])).values.T
    dy = dy.astype("timedelta64[ns]")
    x = uv.x
    y = uv.y.astype("datetime64[ns]")
    m0_x = x
    m0_y = y
    m1_x = x + dx
    m1_y = y + dy

    x = xr.concat((e.drop_vars("point", errors="ignore") for e in (m0_x, m1_x)), dim="point")
    y = xr.concat((e.drop_vars("point", errors="ignore") for e in (m0_y, m1_y)), dim="point")

    # shift in the x direction updating #
    dx = shift_update(drg, dx, x, y, kernel)
    dy = shift_update(daz, dy, x, y, kernel)

    ds = xr.Dataset(
        data_vars={
            "m1_x": ("point", m1_x.values),
            "m1_y": ("point", m1_y.values),
            "dx": ("point", dx.values),
            "dy": ("point", dy.values),
            "coherence": ("point", uv.sel(variable="coherence").values),
        },
        coords=m1_x.coords,
    )

    return burst_id, acquisition_id, ds


def initialize_patches(
    y_size: int,
    y_width: int,
    y_step: int,
    x_size: int,
    x_width: int,
    x_step: int,
) -> xr.Dataset:
    """

    Create a new patches dataset.
    This dataset is used for Pearson coefficient computation.

    :param x_size: total size in the x-direction of the area to be covered by patches.
    :param x_width: width of each patch along the x-direction.
    :param x_step: distance in pixels between two consecutive patches in the x-direction.
    :param y_size: total size in the y-direction of the area to be covered by patches.
    :param y_width: width of each patch along the y-direction.
    :param y_step: distance in pixels between two consecutive patches in the y-direction.

    """
    y_off = 1 if (y_size % y_step != 0) else 0
    x_off = 1 if (x_size % x_step != 0) else 0
    y_elements = y_size // y_step + y_off
    x_elements = x_size // x_step + x_off
    y_start = np.repeat(np.arange(0, y_step * y_elements, y_step), x_elements)
    x_start = np.tile(np.arange(0, x_step * x_elements, x_step), y_elements)
    n_elements, y_end, x_end = (
        y_elements * x_elements,
        y_start + y_width - 1,
        x_start + x_width - 1,
    )
    coords = {"patch": np.arange(n_elements)}
    data_vars = {}
    for k, v in zip(["x_start", "x_end", "y_start", "y_end"], [x_start, x_end, y_start, y_end]):
        data_vars[k] = (["patch"], v)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds


def compute_pearson_coefficient(
    patch: xr.Dataset,
    stack: xr.DataArray,
    reference: xr.DataArray,
    pixel_to_time: np.ndarray,
    peak_window_size: Sequence[int],
    y_dim: str,
    x_dim: str,
    correlation_kernel_size: Sequence[int] | None,
):

    # Number of images and number of points
    shp = np.array(stack.shape[0:2]) - 2 * np.array(peak_window_size)
    nim, npt = stack.shape[2], np.prod(shp)

    # Check that the search window is not too large
    if npt < 0:
        logger.warning(f"Search window {peak_window_size} is too large for the input image of size {stack.shape[0:2]}.")
        dy, dx, coherence, sy, sx = np.zeros((5, nim), dtype="float32")
    else:
        # Correlation window dimension
        nco = (2 * peak_window_size[0] + 1) * (2 * peak_window_size[1] + 1)
        i1, i2 = backward_flatten(stack.values, reference.values, list(peak_window_size), npt, nco, nim, shp)

        # Remove bias
        i1 = i1 - i1.mean(axis=0)
        i2 = i2 - i2.mean(axis=1)[..., None]

        # Scale to norm 1
        with np.errstate(divide="ignore", invalid="ignore"):
            i1 = i1 / np.linalg.norm(i1, axis=0)
            i2 = i2 / np.linalg.norm(i2, axis=1)[..., None]

        # Compute Pearson correlation coefficient
        corr = np.abs(i2 @ i1.conj())
        corr = corr.reshape([nim, 2 * peak_window_size[0] + 1, 2 * peak_window_size[1] + 1])

        dy, dx, coherence, sy, sx = get_corr_sub_pix(corr, correlation_kernel_size).T
    return xr.DataArray(
        data=np.vstack((*np.vstack((dy, dx)) * pixel_to_time, coherence, sy, sx)).T.reshape(nim, -1, 5),
        dims=["acquisition_id", "point", "variable"],
        coords={
            "acquisition_id": stack.z.values,
            "point": [patch.patch.values],
            "y": (["point"], [stack[y_dim].mean().values]),
            "x": (["point"], [stack[x_dim].mean().values]),
            "variable": ["dy", "dx", "coherence", "s_y", "s_x"],
        },
        attrs={"pixel_to_time": pixel_to_time.flatten()},
    )
