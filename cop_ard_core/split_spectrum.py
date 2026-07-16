import os

import numpy as np
import xarray as xr

import cop_ard_core.detrend
import cop_ard_core.esd
import cop_ard_core.interf
import cop_ard_core.unwrap
import cop_ard_core.utils
from cop_ard_core.computing.computation import map
from cop_ard_core.computing.io import Path, read, write
from cop_ard_core.computing.logger import logger


def joint_unwrap(mosaic_l: xr.Dataset, mosaic_h: xr.Dataset, cohe_th: float = 0.3):

    cohe_th = 0.3

    # TO BE MANAGED MULTIPLE POLARIZATIONS AND ACQUISITION DATES
    mosaic_h_sel = mosaic_h.isel(polarization=0, acquisition_date=1)
    mosaic_l_sel = mosaic_l.isel(polarization=0, acquisition_date=1)

    # extract the average coherence between the two sub-bands
    cohe_mask = xr.apply_ufunc(np.sqrt, (mosaic_h_sel.cohe) * (mosaic_l_sel.cohe), dask="parallelized")

    # compute the average phase between the two sub-bands
    ph_mean = xr.apply_ufunc(
        np.angle, np.exp(1j * (mosaic_h_sel.phase) + np.exp(1j * mosaic_l_sel.phase)), dask="parallelized"
    )
    ph_mean_unw = cop_ard_core.unwrap.unwrap_phase(ph_mean, cohe_mask, threshold=cohe_th)

    # unwrap sub-band residues and add back the average phase
    ph_h_unw = (
        cop_ard_core.unwrap.unwrap_phase(
            cop_ard_core.utils.wrap(mosaic_h_sel.phase - ph_mean_unw), cohe_mask, threshold=cohe_th
        )
        + ph_mean_unw
    )
    ph_l_unw = (
        cop_ard_core.unwrap.unwrap_phase(
            cop_ard_core.utils.wrap(mosaic_l_sel.phase - ph_mean_unw), cohe_mask, threshold=cohe_th
        )
        + ph_mean_unw
    )

    return ph_l_unw, ph_h_unw


def split_spectrum_core(
    phase_unw_l: xr.DataArray, phase_unw_h: xr.DataArray, f0: float, f_l_tot: float, f_h_tot: float
) -> [xr.DataArray, xr.DataArray]:

    # dispersive component
    phase_iono = (
        (phase_unw_l * (f_h_tot / f0) - phase_unw_h * (f_l_tot / f0)) * (f_l_tot * f_h_tot) / (f_h_tot**2 - f_l_tot**2)
    )

    # non-dispersive component
    phase_nd = (phase_unw_h * (f_h_tot / f0) - phase_unw_l * (f_l_tot / f0)) * f0 / (2 * (f_h_tot - f_l_tot))

    return phase_iono, phase_nd


def split_spectrum(
    slcs: xr.DataTree,
    reference_date: str,
    frequency_offset: float,
    bandwidth: float,
    cohe_th: float,
    bbox: tuple,
    y_poly: xr.Dataset,
    multilook: dict | None = None,
    wdir_path: Path = Path("/tmp"),
):

    # define center band frequencies
    f0 = cop_ard_core.utils.get_carrier_frequency(slcs.bursts, "coreg/measurements/cstack")
    f_l_tot = f0 - frequency_offset
    f_h_tot = f0 + frequency_offset

    # lower band interferogram
    mosaic_l = cop_ard_core.interf.get_mosaic_interf(
        slcs, reference_date, f0=-frequency_offset, bw=bandwidth, bbox=bbox, multilook=multilook, y_poly=y_poly
    )

    logger.debug(f"save mosaic_l to file {os.path.join(wdir_path, 'mosaic_l')}")
    mosaic_l = write(mosaic_l, data_path=os.path.join(wdir_path, "mosaic_l"), mode="w")

    # upper band interferogram
    mosaic_h = cop_ard_core.interf.get_mosaic_interf(
        slcs, reference_date, f0=frequency_offset, bw=bandwidth, bbox=bbox, multilook=multilook, y_poly=y_poly
    )

    logger.debug(f"save mosaic_h to file {os.path.join(wdir_path, 'mosaic_h')}")
    mosaic_h = write(mosaic_h, data_path=os.path.join(wdir_path, "mosaic_h"), mode="w")

    # joint interferogram phase unwrapping
    phase_unw_l, phase_unw_h = joint_unwrap(mosaic_l, mosaic_h, cohe_th=cohe_th)
    phase_tot_l = phase_unw_l + mosaic_l.phase_ramp if "phase_ramp" in mosaic_l.data_vars else phase_unw_l
    phase_tot_h = phase_unw_h + mosaic_h.phase_ramp if "phase_ramp" in mosaic_h.data_vars else phase_unw_h

    # separate dispersive and non-dispersive components
    phase_iono, phase_nd = split_spectrum_core(phase_tot_l, phase_tot_h, f0, f_l_tot, f_h_tot)

    # assign azimuth time of raw image in order to enable interpolation
    phase_iono["azimuth_time_raw"] = mosaic_l["azimuth_time_raw"]

    return phase_iono, phase_nd


def scale_azimuth_time(azimuth_time_raw: xr.DataArray, azimuth_time: xr.DataArray, H_orb: float, H_target: float):

    # compute difference between azimuth time on the ground and at orbit level
    az_time = (azimuth_time_raw - azimuth_time).astype(float)

    # rescale with the elevation of ionosphere
    az_time = (az_time * (H_target / H_orb)).astype("timedelta64[ns]")

    # compute the ablote time
    az_time = az_time + azimuth_time

    return az_time


def fit_ionosphere(phase_iono: xr.Dataset, H_iono: float, H_orb: float, degree: int = 1, norm: int = 1) -> xr.DataArray:

    y_ = scale_azimuth_time(phase_iono.azimuth_time_raw, phase_iono.azimuth_time, H_orb, H_iono)

    # compute minimization axis
    x_ = np.tile(phase_iono.slant_range_time.values, phase_iono.azimuth_time.size).reshape(phase_iono.azimuth_time_raw.shape)
    y_ = (y_ - y_[0,0]).astype(float).values
    data = phase_iono.squeeze().stack(point=['azimuth_time', 'slant_range_time']).values
    # compute polynomial fit coefficients
    coeffs, x_0, x_sc = cop_ard_core.detrend.fit_2d_poly(x_, y_, data, degree=degree, norm=norm)

    # costruct dataarray with fit coefficients and attributes for rescaling and centering the coordinates
    coeffs = xr.DataArray(coeffs.x, dims=["degree"], coords={"degree": np.arange(coeffs.x.size)})
    coeffs.attrs["offset"] = x_0
    coeffs.attrs["scale_fact"] = x_sc

    # return coefficients and attributes for rescaling and centering the coordinates
    return coeffs


def reconstruct_iono_surface(interf: xr.Dataset, coeffs: xr.DataArray, H_iono: float, H_orb: float, degree: int=1) -> xr.DataArray:

    y_ = scale_azimuth_time(interf.azimuth_time_raw, 
                            interf.azimuth_time,
                            H_orb,
                            H_iono)

    # compute minimization axis
    x_ = np.tile(interf.slant_range_time.values, interf.azimuth_time.size).reshape(interf.azimuth_time_raw.shape)
    y_ = (y_ - y_[0,0]).astype(float).values    
    x_full = np.vstack([x_.ravel(), y_.ravel()]).T 
    x_full = (x_full - coeffs.attrs['offset']) / coeffs.attrs['scale_fact']

    # reconstruct the surface in correspondence of the coordinates x_full
    surf = cop_ard_core.detrend.fit(cop_ard_core.detrend.get_model_matrix(x_full, degree=degree), coeffs.values)
    surf = surf.reshape(interf.azimuth_time_raw.shape)
    surf = xr.DataArray(surf, dims=['azimuth_time', 'slant_range_time'])

    # return coefficients and attributes for rescaling and centering the coordinates
    return surf


def remove_iono_surface_single_burst(burst_id: str,
                                     burst: xr.DataTree, coeffs: xr.DataArray, 
                        H_iono: float, H_orb: float, 
                        outfolder: Path,
                        degree: int=1, 
                        ) -> tuple[str, xr.DataTree]:

    # recontruct the surface over all points
    cstack = burst["coreg/measurements/cstack"]

    if "2016-02-14" not in cstack.acquisition_date.values:
        return

    # compute raw  azimuth time
    azimuth_time_raw = cop_ard_core.utils.get_azimuth_time_raw(cstack)

    # rescale the time to the ionospheric height
    azimuth_time_iono = scale_azimuth_time(azimuth_time_raw, cstack.azimuth_time, H_orb, H_iono)
    # compute minimization axis
    x_ = np.tile(cstack.slant_range_time.values, cstack.azimuth_time.size).reshape(azimuth_time_raw.shape)
    y_ = (azimuth_time_iono - azimuth_time_iono[0, 0]).astype(float).values
    x_full = np.vstack([x_.ravel(), y_.ravel()]).T
    x_full = (x_full - coeffs.attrs["offset"]) / coeffs.attrs["scale_fact"]

    # reconstruct the surface in correspondence of the coordinates x_full
    surf = cop_ard_core.detrend.fit(cop_ard_core.detrend.get_model_matrix(x_full, degree=degree), coeffs.values)
    surf = surf.reshape(azimuth_time_raw.shape)
    surf = xr.DataArray(surf, dims=["azimuth_time", "slant_range_time"])

    burst["coreg/measurements/phase_iono"] = surf

    # remove ionosphere from slc
    burst["coreg/measurements/cstack"].loc[{"acquisition_date": "2016-02-14"}] *= np.exp(
        -1j * burst.coreg.measurements.phase_iono
    )

    burst = write(burst, f"{outfolder}/bursts/{burst_id}", mode='w')
    return burst_id, burst

