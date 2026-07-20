import numpy as np
import xarray as xr
from scipy.constants import speed_of_light


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
