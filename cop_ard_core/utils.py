import numpy as np
import xarray as xr
from scipy.constants import speed_of_light


def get_interf(a: xr.DataArray, b: xr.DataArray, ml_factor: dict[str, int]):
    return (a * b.conj()).coarsen(ml_factor, boundary="trim").mean()  # type: ignore mean is injected dynamically


def get_normalized_interf(
    a: xr.DataArray,
    b: xr.DataArray,
    ml_factor: dict[str, int],
) -> xr.DataArray:
    return (get_interf(a, b, ml_factor) / np.sqrt(get_interf(a, a, ml_factor) * get_interf(b, b, ml_factor))).fillna(0j)


def compute_interf(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    # complex multiplication and average #
    return (reference.conj() * secondary).coarsen(multilook, boundary="trim").mean()  # type: ignore  mean is injected dynamically


def get_power(slc: xr.DataArray, multilook: dict):
    # compute the image power and average
    return (slc * slc.conj()).real.coarsen(multilook, boundary="trim").mean()  # type: ignore mean is injected dynamically


def compute_interf_norm(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    multilook: dict,
) -> xr.DataArray:
    # compute interferogram #
    num = compute_interf(reference, secondary, multilook)

    # compute power of the images
    pow_ref = get_power(reference, multilook)
    pow_sec = get_power(secondary, multilook)

    # compute the normalization factor
    den = np.sqrt(pow_ref * pow_sec)

    # fill invalid values with 0 (null-coherence)
    return (num / den).fillna(0j)


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    w_ = w / 2.0
    return ((v + w_) % w) - w_


def get_coordinates(y, x, alpha, beta):
    """
    Calculates coordinates for polynomial evaluation based on offsets and scaling.
    """
    return (((e - alpha[i]).astype("float64") / beta[i].astype("float64")).values for i, e in enumerate([y, x]))


# @timeit
def get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact):
    c1, c2 = get_coordinates(azimuth_time, slant_range_time, offset, scale_fact)
    return np.polynomial.polynomial.polygrid2d(c1, c2, y_poly)


def get_coeff_demod(dkcoa_polynomial_coeffs: np.ndarray):
    return (2 * np.pi) * np.polynomial.polynomial.polyint(dkcoa_polynomial_coeffs, axis=1)


# @timeit
def get_phramp_diff(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    drg: xr.DataArray,
    daz: xr.DataArray,
    timeca_polynomial_coeffs: np.ndarray,
    azimuth_grid_spacing: float,
    dkcoa_polynomial_coeffs: np.ndarray,
    scp_pixel_azimuth_time: np.datetime64,
    scp_pixel_slant_range_time: np.float64,
    y_poly: xr.DataArray,
):

    # @timeit
    def poly_combine(rg, az_off, az_ori, coeff_demod):
        return np.polynomial.polynomial.polyval2d(rg, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
            rg, az_ori, coeff_demod
        )

    # extract demodulation coefficients
    coeff_demod = get_coeff_demod(dkcoa_polynomial_coeffs)

    # compute spacing and start times
    dtaz = timeca_polynomial_coeffs[1] * azimuth_grid_spacing * 1e9
    scale_daz = 1 / (timeca_polynomial_coeffs[1] * 1e9)

    # compute azimuth correction
    daz_corr = get_daz_corr(azimuth_time, slant_range_time, y_poly.values, y_poly.offset, y_poly.scale_fact)

    # Corrected slant range and azimuth positions
    daz_corr = daz + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

    # compute the position in the secondary image corresponding to the reference one with and without corrections in meters wrt the scene centre
    az_off = ((azimuth_time + daz_corr - scp_pixel_azimuth_time).astype(float) * scale_daz).values
    az_ori = ((azimuth_time + daz - scp_pixel_azimuth_time).astype(float) * scale_daz).values

    # range in meters wrt the burst center
    rg = ((slant_range_time + drg - scp_pixel_slant_range_time).T * (speed_of_light / 2)).values

    # recontruct polynomial function
    ph_ramp_diff_da = xr.DataArray(
        poly_combine(rg, az_off, az_ori, coeff_demod),
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "slant_range_time": slant_range_time,
            "azimuth_time": azimuth_time,
        },
    )

    return ph_ramp_diff_da


def time_to_pixel(
    stack: xr.Dataset | xr.DataArray, y0: float | None = None, x0: float | None = None
) -> xr.Dataset | xr.DataArray:
    """

    Convert from times to pixel dimensions

    """

    if y0 is None:
        y0 = stack.azimuth_time[0].values
    if x0 is None:
        x0 = stack.slant_range_time[0].values

    # spacing computation #
    y_spacing, x_spacing = [(stack[e][1] - stack[e][0]).values for e in ["azimuth_time", "slant_range_time"]]

    # times to pixel coordinates #
    x = np.round((stack.slant_range_time - x0) / x_spacing).values.astype("int32")
    y = np.round((stack.azimuth_time - y0) / y_spacing).values.astype("int32")

    stack = stack.assign_coords({"y": (["azimuth_time"], y), "x": (["slant_range_time"], x)})
    return stack.swap_dims(azimuth_time="y", slant_range_time="x")
