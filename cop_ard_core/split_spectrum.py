import numpy as np
import xarray as xr

from cop_ard_core.detrend import fit, fit_2d_poly, get_model_matrix, to_polygrid2d

from cop_ard_core.unwrap import unwrap_phase
from cop_ard_core.logger import timeit


def joint_unwrap(
    mosaic_l_coherence: xr.DataArray,
    mosaic_h_coherence: xr.DataArray,
    mosaic_l_phase: xr.DataArray,
    mosaic_h_phase: xr.DataArray,
    cohe_th: float = 0.3,
):

    # extract the average coherence between the two sub-bands
    cohe_mask = xr.ufuncs.sqrt(mosaic_h_coherence * mosaic_l_coherence)

    # compute the average phase between the two sub-bands
    ph_mean = xr.ufuncs.angle(np.exp(1j * mosaic_h_phase) + np.exp(1j * mosaic_l_phase))
    ph_mean_unw = unwrap_phase(ph_mean, cohe_mask, threshold=cohe_th)

    # unwrap sub-band residues and add back the average phase
    ph_l_unw = unwrap_phase(mosaic_l_phase - ph_mean_unw, cohe_mask, threshold=cohe_th) + ph_mean_unw
    ph_h_unw = unwrap_phase(mosaic_h_phase - ph_mean_unw, cohe_mask, threshold=cohe_th) + ph_mean_unw
    return ph_l_unw, ph_h_unw, cohe_mask


def split_spectrum_core(
    phase_unw_l: xr.DataArray,
    phase_unw_h: xr.DataArray,
    f0: float,
    f_l_tot: float,
    f_h_tot: float,
) -> tuple[xr.DataArray, xr.DataArray]:

    # dispersive component
    phase_iono = (
        (phase_unw_l * (f_h_tot / f0) - phase_unw_h * (f_l_tot / f0)) * (f_l_tot * f_h_tot) / (f_h_tot**2 - f_l_tot**2)
    )

    # non-dispersive component
    phase_nd = (phase_unw_h * (f_h_tot / f0) - phase_unw_l * (f_l_tot / f0)) * f0 / (2 * (f_h_tot - f_l_tot))

    return phase_iono, phase_nd


def scale_azimuth_time(
    azimuth_time_raw: xr.DataArray,
    azimuth_time: xr.DataArray,
    H_orb: xr.DataArray | float,
    H_target: xr.DataArray | float,
):

    # compute difference between azimuth time on the ground and at orbit level
    az_time = (azimuth_time_raw - azimuth_time).astype(float)

    # rescale with the elevation of ionosphere
    az_time = (az_time * (H_target / H_orb)).astype("timedelta64[ns]")

    # compute the ablote time
    az_time = az_time + azimuth_time

    return az_time


@timeit
def fit_ionosphere(
    phase_iono: xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    H_iono: float,
    H_orb: float,
    degree: int = 1,
    norm: int = 1,
    weights: xr.DataArray | None = None,
) -> xr.DataArray:

    y_ = scale_azimuth_time(azimuth_time_raw, phase_iono.azimuth_time, H_orb, H_iono).values

    # compute minimization axis
    x_ = np.tile(phase_iono.slant_range_time.values, phase_iono.azimuth_time.size).ravel()
    off_y = y_[0, 0]
    y_ = (y_ - off_y).astype(float).ravel()
    data = phase_iono.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values
    w_stack = (
        weights.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values if weights is not None else None
    )

    # compute polynomial fit coefficients
    coeffs, x_sc, x_0 = fit_2d_poly(x_, y_, data, degree=degree, norm=norm, weights=w_stack)

    attrs = {"deg": degree, "offset": x_0, "scale_fact": x_sc, "off_y": str(off_y)}
    # costruct dataarray with fit coefficients and attributes for rescaling and centering the coordinates
    poly_corr = xr.DataArray(
        data=to_polygrid2d(coeffs.x, degree),
        dims=["i", "j"],
        name="poly_corr",
        coords={key: np.arange(degree + 1) for key in ["i", "j"]},
        attrs=attrs,
    )

    # return coefficients and attributes for rescaling and centering the coordinates
    return poly_corr


def reconstruct_iono_surface(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    poly_corr: xr.DataArray,
    H_iono: xr.DataArray | float,
    H_orb: xr.DataArray | float,
) -> xr.DataArray:


    y_ = scale_azimuth_time(azimuth_time_raw, azimuth_time, H_orb, H_iono).values

    # compute minimization axis
    x_ = np.tile(slant_range_time.values, azimuth_time.size).ravel()
    off_y = np.datetime64(poly_corr.attrs["off_y"], "ns")
    y_ = (y_ - off_y).astype(float).ravel()

    x_full = np.vstack([y_, x_]).T

    x_full = (x_full - poly_corr.attrs["offset"]) / poly_corr.attrs["scale_fact"]


    # reconstruct the surface in correspondence of the coordinates x_full
    surf = np.polynomial.polynomial.polyval2d(x_full[:, 1], x_full[:, 0], poly_corr.values)
    surf = surf.reshape(azimuth_time.size, slant_range_time.size)
    surf = xr.DataArray(
        surf,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": azimuth_time.values, "slant_range_time": slant_range_time.values},
    )

    # return coefficients and attributes for rescaling and centering the coordinates
    return surf
