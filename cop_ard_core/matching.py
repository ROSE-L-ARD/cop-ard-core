from typing import Literal, Sequence

import numpy as np
import xarray as xr
from scipy.constants import speed_of_light
from xrspatial.convolution import convolution_2d

from cop_ard_core.detrend import estimate_global_surface
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


def polyfit(
    disparity: xr.DataArray,
    drg: xr.DataArray,
    daz: xr.DataArray,
    annot: dict,
    cohe_th: float = 0.3,
    deg: int = 3,
    kernel_shape: list[int] = [15, 15],
    **kwargs,
):
    # kernel creation for drg and daz multilooking #
    kernel = np.ones(kernel_shape) / np.prod(kernel_shape)

    def to_polygrid2d(c):
        j = np.insert(np.cumsum(deg + 1 - np.arange(0, deg + 1)), 0, 0)

        return np.flip(
            np.sum([np.diag(c[j[k] : j[k + 1]], k) for k in np.arange(0, deg + 1)], axis=0),
            axis=1,
        )

    def shift_update(
        i: xr.DataArray,
        i0: xr.DataArray,
        scale_fact: float,
        x: xr.DataArray,
        y: xr.DataArray,
    ) -> xr.DataArray:
        o = (i - i.mean()).astype("float64").squeeze()
        o = convolution_2d(o, kernel).isel(azimuth_time=y, slant_range_time=x)

        o1 = o.isel(point=slice(o.point.size // 2))
        o2 = o.isel(point=slice(o.point.size // 2, None))
        return (o1 - o2) / scale_fact - i0

    def poly_estimate(
        v: np.ndarray,
        x: np.ndarray,
        name: str,
        attrs: dict,
    ):
        output = estimate_global_surface(x, v, degree=deg, **kwargs)

        poly = xr.DataArray(
            data=to_polygrid2d(output.x),
            dims=["i", "j"],
            name=name,
            coords={k: np.arange(deg + 1) for k in ["i", "j"]},
            attrs=attrs,
        )
        return poly.to_dataset().assign_attrs({"type": disparity.type})

    def create_attrs(offset, scale_fact):
        y0, x0 = offset.values
        s_y, s_x = scale_fact.values * disparity.pixel_to_time
        if disparity.type == "times":
            y0 = daz.azimuth_time[0] + np.round((disparity.pixel_to_time[0] * y0)).astype("timedelta64[ns]")
            x0 = drg.slant_range_time[0] + x0 * disparity.pixel_to_time[1]
            y0, x0 = [e.values for e in [y0, x0]]
            s_y = np.round(s_y).astype("timedelta64[ns]")
        return {"deg": deg, "offset": (y0, x0), "scale_fact": (s_y, s_x)}

    try:
        # image position correction #
        disparity = disparity.squeeze()
        uv = disparity.groupby(disparity.sel(variable="cohe") >= cohe_th)[True]
        dy, dx = (uv.sel(variable=["dy", "dx"]) / disparity.pixel_to_time).values.astype("float64").T

        if disparity.type == "times":
            x = uv.x - drg.slant_range_time[0]
            y = uv.y.astype("datetime64[ns]") - daz.azimuth_time[0]
            m0 = xr.concat(
                (v.astype("float64") / disparity.pixel_to_time[k] for k, v in enumerate([y, x])),
                dim="variable",
            )
        else:
            m0 = xr.concat((uv.y, uv.x), dim="variable")

        # from here on all coordinates are in pixels #
        m0 = m0.assign_coords({"variable": ["y", "x"]})
        m1 = m0 + np.vstack((dy, dx))
    except Exception as e:
        logger.critical(f"Error: {type(e).__name__} - {e}")
        raise RuntimeError("No valid points found")

    # concatenation of points (m0, m1) #
    y, x = np.round(xr.concat((e.drop_vars("point", errors="ignore") for e in (m0, m1)), dim="point")).astype("int32")

    # shift in the x direction updating #
    dx = shift_update(drg, dx, (2 * annot["Grid"]["Row"]["SS"] / speed_of_light), x, y)

    # shift in the y direction updating #
    dy = shift_update(
        daz,
        dy,
        (annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9),
        x,
        y,
    )
    # coordinates rescaling to avoid numeric instability issues #
    offset = m1.mean(dim="point")
    scale_fact = (m1.max(dim="point") - m1.min(dim="point")) / 2.0
    m1 = ((m1 - offset) / scale_fact).values.T

    # polynomial (both for x and y) estimation #
    name = str(disparity.acquisition_date.values)
    attrs = create_attrs(offset, scale_fact)
    print("-" * 50)
    return list(poly_estimate(v.values, m1, name, attrs) for v in (dy, dx))


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
    forward: bool,
    y_dim: str,
    x_dim: str,
    correlation_kernel_size: Sequence[int] | None,
    data_type: Literal["times", "pixels"] = "times",
):

    # Number of images and number of points
    shp = np.array(stack.shape[0:2]) - 2 * np.array(peak_window_size)
    nim, npt = stack.shape[2], np.prod(shp)

    # Check that the search window is not too large
    if npt < 0:
        logger.warning(f"Search window {peak_window_size} is too large for the input image of size {stack.shape[0:2]}.")
        dy, dx, cohe, sy, sx = np.zeros((5, nim), dtype="float32")
    else:
        # Correlation window dimension
        nco = (2 * peak_window_size[0] + 1) * (2 * peak_window_size[1] + 1)
        i1, i2 = (
            forward_flatten(stack.values, reference.values, list(peak_window_size), npt, nco, nim, shp)
            if forward
            else backward_flatten(stack.values, reference.values, list(peak_window_size), npt, nco, nim, shp)
        )

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

        dy, dx, cohe, sy, sx = get_corr_sub_pix(corr, correlation_kernel_size).T

    return xr.DataArray(
        data=np.vstack((*np.vstack((dy, dx)) * pixel_to_time, cohe, sy, sx)).T.reshape(nim, -1, 5),
        dims=["acquisition_date", "point", "variable"],
        coords={
            "acquisition_date": stack.z.values,
            "point": patch.patch.values,
            "y": (["point"], [stack[y_dim].mean().values]),
            "x": (["point"], [stack[x_dim].mean().values]),
            "variable": ["dy", "dx", "cohe", "s_y", "s_x"],
        },
        attrs={"type": data_type, "pixel_to_time": pixel_to_time.flatten()},
    )
