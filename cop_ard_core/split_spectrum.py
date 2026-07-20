from cop_ard_core.utils import wrap
from cop_ard_core.unwrap import unwrap_phase
import numpy as np
import xarray as xr

from cop_ard_core.detrend import fit, fit_2d_poly, get_model_matrix


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
    ph_mean = xr.ufuncs.angle(np.exp(1j * (mosaic_h_phase) + np.exp(1j * mosaic_l_phase)))
    ph_mean_unw = unwrap_phase(ph_mean, cohe_mask, threshold=cohe_th)

    # unwrap sub-band residues and add back the average phase
    ph_h_unw = unwrap_phase(wrap(mosaic_h_phase - ph_mean_unw), cohe_mask, threshold=cohe_th) + ph_mean_unw
    ph_l_unw = unwrap_phase(wrap(mosaic_l_phase - ph_mean_unw), cohe_mask, threshold=cohe_th) + ph_mean_unw

    return ph_l_unw, ph_h_unw


def split_spectrum_core(
    phase_unw_l: xr.DataArray, phase_unw_h: xr.DataArray, f0: float, f_l_tot: float, f_h_tot: float
) -> tuple[xr.DataArray, xr.DataArray]:

    # dispersive component
    phase_iono = (
        (phase_unw_l * (f_h_tot / f0) - phase_unw_h * (f_l_tot / f0)) * (f_l_tot * f_h_tot) / (f_h_tot**2 - f_l_tot**2)
    )

    # non-dispersive component
    phase_nd = (phase_unw_h * (f_h_tot / f0) - phase_unw_l * (f_l_tot / f0)) * f0 / (2 * (f_h_tot - f_l_tot))

    return phase_iono, phase_nd


def scale_azimuth_time(azimuth_time_raw: xr.DataArray, azimuth_time: xr.DataArray, H_orb: float, H_target: float):

    # compute difference between azimuth time on the ground and at orbit level
    az_time = (azimuth_time_raw - azimuth_time).astype(float)

    # rescale with the elevation of ionosphere
    az_time = (az_time * (H_target / H_orb)).astype("timedelta64[ns]")

    # compute the ablote time
    az_time = az_time + azimuth_time

    return az_time


def fit_ionosphere(
    phase_iono: xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    H_iono: float,
    H_orb: float,
    degree: int = 1,
    norm: int = 1,
) -> xr.DataArray:

    y_ = scale_azimuth_time(azimuth_time_raw, phase_iono.azimuth_time, H_orb, H_iono)

    # compute minimization axis
    x_ = np.tile(phase_iono.slant_range_time.values, phase_iono.azimuth_time.size).reshape(
        phase_iono.azimuth_time_raw.shape
    )
    y_ = (y_ - y_[0, 0]).astype(float).values
    data = phase_iono.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values
    # compute polynomial fit coefficients
    coeffs, x_0, x_sc = fit_2d_poly(x_, y_, data, degree=degree, norm=norm)

    # costruct dataarray with fit coefficients and attributes for rescaling and centering the coordinates
    coeffs = xr.DataArray(coeffs.x, dims=["degree"], coords={"degree": np.arange(coeffs.x.size)})
    coeffs.attrs["offset"] = x_0
    coeffs.attrs["scale_fact"] = x_sc

    # return coefficients and attributes for rescaling and centering the coordinates
    return coeffs


def reconstruct_iono_surface(
    interf: xr.Dataset | xr.DataArray,
    azimuth_time_raw: xr.DataArray,
    coeffs: xr.DataArray,
    H_iono: float,
    H_orb: float,
    degree: int = 1,
) -> xr.DataArray:

    y_ = scale_azimuth_time(azimuth_time_raw, interf.azimuth_time, H_orb, H_iono)

    # compute minimization axis
    x_ = np.tile(interf.slant_range_time.values, interf.azimuth_time.size).reshape(azimuth_time_raw.shape)
    y_ = (y_ - y_[0, 0]).astype(float).values
    x_full = np.vstack([x_.ravel(), y_.ravel()]).T
    x_full = (x_full - coeffs.attrs["offset"]) / coeffs.attrs["scale_fact"]

    # reconstruct the surface in correspondence of the coordinates x_full
    surf = fit(get_model_matrix(x_full, degree=degree), coeffs.values)
    surf = surf.reshape(azimuth_time_raw.shape)
    surf = xr.DataArray(surf, dims=["azimuth_time", "slant_range_time"])

    # return coefficients and attributes for rescaling and centering the coordinates
    return surf
