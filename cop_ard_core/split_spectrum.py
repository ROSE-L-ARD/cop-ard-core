import numpy as np
import xarray as xr
from s1_ard_core.computing.logger import logger, timeit

import cop_ard_core.utils
import cop_ard_core.interf
import cop_ard_core.detrend

def joint_unwrap(mosaic_l: xr.Dataset, mosaic_h: xr.Dataset, cohe_th: float=0.3):

    cohe_th = 0.3

    # TO BE MANAGED MULTIPLE POLARIZATIONS AND ACQUISITION DATES
    mosaic_h_sel = mosaic_h.isel(polarization=1, acquisition_date=1)
    mosaic_l_sel = mosaic_l.isel(polarization=1, acquisition_date=1)

    # extract the average coherence between the two sub-bands
    cohe_mask = xr.apply_ufunc(np.sqrt, (mosaic_h_sel.cohe) * (mosaic_l_sel.cohe), dask='parallelized')

    # compute the average phase between the two sub-bands
    ph_mean = xr.apply_ufunc(np.angle, np.exp(1j*(mosaic_h_sel.phase)+np.exp(1j*mosaic_l_sel.phase)), dask='parallelized')
    ph_mean_unw = cop_ard_core.unwrap.unwrap_phase(ph_mean, cohe_mask, threshold=cohe_th)

    # unwrap sub-band residues and add back the average phase
    ph_h_unw = cop_ard_core.unwrap.unwrap_phase(cop_ard_core.utils.wrap(mosaic_h_sel.phase-ph_mean_unw), cohe_mask, threshold=cohe_th)+ph_mean_unw
    ph_l_unw = cop_ard_core.unwrap.unwrap_phase(cop_ard_core.utils.wrap(mosaic_l_sel.phase-ph_mean_unw), cohe_mask, threshold=cohe_th)+ph_mean_unw

    return ph_l_unw, ph_h_unw

def split_spectrum_core(phase_unw_l: xr.DataArray, phase_unw_h: xr.DataArray, f0: float, f_l_tot: float, f_h_tot: float) -> [xr.DataArray, xr.DataArray]:

    # dispersive component
    phase_iono = (phase_unw_l * (f_h_tot/f0) - phase_unw_h * (f_l_tot/f0)) * (f_l_tot * f_h_tot) / (f_h_tot**2 - f_l_tot**2)

    # non-dispersive component
    phase_nd = (phase_unw_h * (f_h_tot/f0) - phase_unw_l * (f_l_tot/f0)) * f0 / (2*(f_h_tot - f_l_tot))

    return phase_iono, phase_nd


def split_spectrum(slcs: xr.DataTree, reference_date: str, 
                   frequency_offset: float, bandwidth: float, 
                   cohe_th: float, bbox: tuple,
                   y_poly: xr.Dataset,
                   multilook: dict | None = None):

    # define center band frequencies
    f0 = cop_ard_core.utils.get_carrier_frequency(slcs.bursts, 'coreg/measurements/cstack')
    f_l_tot = f0 - frequency_offset
    f_h_tot = f0 + frequency_offset

    # lower band interferogram
    mosaic_l = cop_ard_core.interf.get_mosaic_interf(slcs, reference_date, f0=-frequency_offset, bw=bandwidth, bbox=bbox, multilook=multilook, y_poly=y_poly)

    # upper band interferogram
    mosaic_h = cop_ard_core.interf.get_mosaic_interf(slcs, reference_date, f0=frequency_offset, bw=bandwidth, bbox=bbox, multilook=multilook, y_poly=y_poly)

    # joint interferogram phase unwrapping
    phase_unw_l, phase_unw_h = joint_unwrap(mosaic_l, mosaic_h, cohe_th = cohe_th)
    phase_tot_l = phase_unw_l + mosaic_l.phase_ramp
    phase_tot_h = phase_unw_h + mosaic_h.phase_ramp

    # separate dispersive and non-dispersive components
    phase_iono, phase_nd = split_spectrum_core(phase_tot_l, phase_tot_h, f0, f_l_tot, f_h_tot)

    # assign azimuth time of raw image in order to enable interpolation
    phase_iono['azimuth_time_raw'] = mosaic_l['azimuth_time_raw']
    
    return phase_iono, phase_nd 


def scale_azimuth_time(azimuth_time_raw: xr.DataArray, 
                       azimuth_time: xr.DataArray,
                       H_orb: float,
                       H_target: float):

    # compute difference between azimuth time on the ground and at orbit level
    az_time = (azimuth_time_raw - azimuth_time).astype(float)

    # rescale with the elevation of ionosphere
    az_time = (az_time * (H_target / H_orb)).astype('timedelta64[ns]')
    
    # compute the ablote time
    az_time = az_time + azimuth_time

    return az_time


def fit_ionosphere(phase_iono: xr.Dataset, H_iono: float, H_orb: float, degree: int=1, norm: int=1):

    y_ = scale_azimuth_time(phase_iono.azimuth_time_raw, 
                            phase_iono.azimuth_time,
                            H_orb,
                            H_iono)

    # compute minimization axis
    x_ = np.tile(phase_iono.slant_range_time.values, phase_iono.azimuth_time.size).reshape(phase_iono.azimuth_time_raw.shape)
    y_ = (y_ - y_[0,0]).astype(float).values
    data = phase_iono.squeeze().stack(point=['azimuth_time', 'slant_range_time']).values

    # compute polynomial fit coefficients
    return cop_ard_core.detrend.fit_2d_poly(x_, y_, data, degree=degree, norm=norm)


def remove_iono_surface(burst: xr.Dataset, coeffs, 
                        x_0: tuple, x_sc: tuple, 
                        H_iono: float, H_orb: float, 
                        degree: int=1):

    # recontruct the surface over all points

    cstack = burst['coreg/measurements/cstack']

    # compute raw  azimuth time 
    azimuth_time_raw = cop_ard_core.utils.get_azimuth_time_raw(cstack)

    # rescale the time to the ionospheric height
    azimuth_time_iono = scale_azimuth_time(azimuth_time_raw, 
                            cstack.azimuth_time,
                            H_orb,
                            H_iono)
    # compute minimization axis
    x_ = np.tile(cstack.slant_range_time.values, cstack.azimuth_time.size).reshape(azimuth_time_raw.shape)
    y_ = (azimuth_time_iono - azimuth_time_iono[0,0]).astype(float).values
    x_full = np.vstack([x_.ravel(), y_.ravel()]).T 
    x_full = (x_full - x_0) / x_sc

    # reconstruct the surface in correspondence of the coordinates x_full
    surf = cop_ard_core.detrend.fit(cop_ard_core.detrend.get_model_matrix(x_full, regions=None, degree=degree), coeffs.x)
    surf = surf.reshape(azimuth_time_raw.shape)
    surf = xr.DataArray(surf, dims=['azimuth_time', 'slant_range_time'])
    
    # return the ionospheric layer
    burst['coreg/measurements/phase_iono'] = surf
    
    return burst


def split_spectrum_flow(slcs: xr.DataTree, reference_date: str, 
                        multilook: dict, frequency_offset: float, bandwidth: float,
                        H_iono: float, H_orb: float, degree: int=1, norm: int=1,
                        wdir_path: str = "/tmp"):

    # evaluate bounding box
    bbox = cop_ard_core.utils.get_bounding_box(slcs.bursts, 'coreg/measurements')

    # extract full band interferogram
    phase_full = cop_ard_core.interf.get_mosaic_interf(slcs, reference_date, bbox=bbox, multilook=multilook)

    # estimate ESD at full-band
    y_poly = esd_module.estimate(slcs, wdir_path, reference_date, poly_degree=2, enable_dask=True)
    
    # separate dispersive and non-dispersive components
    phase_iono, phase_nd = split_spectrum(slcs, reference_date, frequency_offset, bandwidth=bandwidth, cohe_th=cohe_th, y_poly=y_poly)

    # fit surface at ionospheric height
    coeffs, x_sc, x_0 = fit_ionosphere(phase_iono, H_iono, H_orb, degree=degree, norm=norm)
