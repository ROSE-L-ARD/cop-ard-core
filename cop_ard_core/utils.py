from typing import Any

import numpy as np
import xarray as xr
import xrft
from scipy.constants import speed_of_light


def extract_subband(frange, bw, f0):

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)
    sel = frange * mask

    return xrft.ifft(
        (sel).chunk(freq_slant_range_time=-1), dim=["freq_slant_range_time"], true_amplitude=True, true_phase=True
    )


# def unwrap_phase(ph: xr.DataArray, cohe: xr.DataArray, threshold: float = 1e-3):

#     # stack x,y coordinates
#     x_, y_ = np.meshgrid(np.arange(ph.slant_range_time.size), np.arange(ph.azimuth_time.size))
#     ph_f_lin = ph.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values
#     cohe_f_lin = cohe.squeeze().stack(point=["azimuth_time", "slant_range_time"]).values

#     # apply coherence threshold
#     mask = cohe_f_lin > threshold

#     # select points to be unwrapped
#     ph_sel, cohe_sel = ph_f_lin[mask], cohe_f_lin[mask]
#     x, y = x_.ravel()[mask], y_.ravel()[mask]

#     # call MCF solver
#     print(f"call mcf on {x.size} points")
#     exit_status, total_cost, sol_ = mcf.integ_mcf(x, y, ph_sel, cohe_sel, dict())
#     print(f"Exit status: {exit_status}, Total cost: {total_cost}")

#     # reconstruct unwrapped phase 2D dataarray
#     ph_f = ph.copy()
#     # sol_ = ph_sel
#     ph_f_lin = ph_f_lin * np.nan
#     ph_f_lin[mask] = sol_
#     ph_f[:, :] = ph_f_lin.reshape(ph.shape)

#     return ph_f


def get_bounding_box(slcs: xr.DataTree, path: str) -> list:
    """
    Compute the bounding box of a burst stack in azimuth time and slant range time.

    This function iterates over all bursts in a stack of SLCs and extracts the minimum and maximum
    values of `azimuth_time` and `slant_range_time` across all bursts. The result is a bounding box
    that encompasses the full spatial and temporal extent of the merged image.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    list
        A list of four elements: `[az_min, az_max, rg_min, rg_max]`, representing the bounding box
        in azimuth time and slant range time.
    """
    azimuth_time, slant_range_time = [], []

    for burst in slcs.values():
        if burst[path].azimuth_time.size > 0:
            azimuth_time += [burst[path].azimuth_time.min().values, burst[path].azimuth_time.max().values]
        if burst[path].slant_range_time.size > 0:
            slant_range_time += [burst[path].slant_range_time.min().values, burst[path].slant_range_time.max().values]

    if len(azimuth_time) <= 0 or len(slant_range_time) <= 0:
        raise ValueError("Null slc in input")

    return [
        np.min(azimuth_time),
        np.max(azimuth_time),
        np.min(slant_range_time),
        np.max(slant_range_time),
    ]


def get_spacing(slcs: xr.DataTree, path: str) -> list:
    """
    Compute the spacing of a burst stack in azimuth time and slant range time.

    This function iterates over all bursts in a stack of SLCs and extracts the minimum and maximum
    values of `azimuth_time` and `slant_range_time` across all bursts and derive the corresponding spacing.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    list
        A list of two elements: `[dt_az, dt_rg]`, representing the spacing in azimuth time and slant range time.
    """
    azimuth_time, slant_range_time = [], []

    for burst in slcs.values():
        if burst[path].azimuth_time.size > 0:
            azimuth_time += [
                (burst[path].azimuth_time.max().values - burst[path].azimuth_time.min().values).astype(float)
                / (burst[path].azimuth_time.size - 1)
            ]
        if burst[path].slant_range_time.size > 0:
            slant_range_time += [
                (burst[path].slant_range_time.max().values - burst[path].slant_range_time.min().values)
                / (burst[path].slant_range_time.size - 1)
            ]

    if len(azimuth_time) <= 0 or len(slant_range_time) <= 0:
        raise ValueError("Null slc in input")

    return [
        np.mean(azimuth_time),
        np.mean(slant_range_time),
    ]


def get_carrier_frequency(slcs: xr.DataTree, path: str, pol: str) -> float:
    """
    Compute the mean carrier frequency of the bursts.

    This function iterates over all bursts in a stack of SLCs and extracts the carrier frequency attribute.

    Parameters
    ----------
    slcs : xr.DataTree
        A DataTree containing multiple SAR bursts, each with `azimuth_time` and `slant_range_time` coordinates.

    Returns
    -------
    float
        Average carrier frequency.
    """
    f0 = []

    for burst in slcs.values():
        annot = burst[path].attrs[pol]
        f0.append((annot["RadarCollection"]["TxFrequency"]["Max"] + annot["RadarCollection"]["TxFrequency"]["Min"]) / 2)

    if len(f0) <= 0:
        raise ValueError("Null slc in input")

    return np.mean(f0)


def get_coeff_demod(annot: dict[str, Any]):

    return (2 * np.pi) * np.polynomial.polynomial.polyint(annot["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"], axis=1)


def get_poly(poly, poly_type: str, dd: str):

    poly.attrs["offset"][0] = np.datetime64(poly.attrs[dd]["offset"][0], "ns")
    poly.attrs[dd]["offset"][1] = np.array(poly.attrs[dd]["offset"][1])
    poly.attrs[dd]["scale_fact"][0] = np.array(int(poly.attrs[dd]["scale_fact"][0].split()[0]))
    poly.attrs[dd]["scale_fact"][1] = np.array(poly.attrs[dd]["scale_fact"][1])

    return poly


def get_coordinates(coreg, alpha, beta, coord_type: str = "pixels"):
    """
    Calculates coordinates for polynomial evaluation based on offsets and scaling.
    """
    y, x = coreg.azimuth_time, coreg.slant_range_time
    return (((e - alpha[i]).astype("float64") / beta[i].astype("float64")).values for i, e in enumerate([y, x]))


def get_phramp_diff(burst: xr.Dataset, slc: xr.Dataset, y_poly, dd: str):

    annot = slc.attrs[list(slc.attrs)[0]]

    # y_poly = get_poly(poly, 'y_poly', dd)
    # x_poly = get_poly(burst, 'x_poly', dd)
    coeff_demod = get_coeff_demod(annot)

    # compute spacing and start times
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    # dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light

    # compute azimuth correction
    # rg = (burst.slant_range_time - y_poly[dd].attrs['offset'][1]) / y_poly[dd].attrs['scale_fact'][1]
    # az = (burst.azimuth_time - y_poly[dd].attrs['offset'][0]).astype(float) / y_poly[dd].attrs['scale_fact'][0]

    c1, c2 = get_coordinates(burst, y_poly[dd].offset, y_poly[dd].scale_fact, y_poly.type)
    daz_corr = np.polynomial.polynomial.polygrid2d(c1, c2, y_poly[dd].values)

    # if y_poly is None:
    #     daz_corr = np.zeros((burst.daz.shape[1], burst.daz.shape[2]))
    # else:
    #     daz_corr = polyval2d(*np.meshgrid(az, rg), y_poly.values).T

    # if x_poly is None:
    # drg_corr = np.zeros((burst.daz.shape[1], burst.daz.shape[2]))
    # else:
    #     drg_corr = polyval2d(*np.meshgrid(az, rg), x_poly.values).T

    # Corrected slant range and azimuth positions
    drg = burst.drg[1, :, :]  # + drg_corr * dtrg
    daz = burst.daz[1, :, :] + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

    rg_mid = slc.isel(slant_range_time=annot["ImageData"]["SCPPixel"]["Row"]).slant_range_time.values
    az_mid = slc.isel(azimuth_time=annot["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values

    # compute the position in the secondary image corresponding to the reference one with and without corrections
    az_off = (burst.azimuth_time + daz - az_mid).astype(float) / dtaz * annot["Grid"]["Col"]["SS"]
    az_ori = (burst.azimuth_time + burst.daz[1, :, :] - az_mid).astype(float) / dtaz * annot["Grid"]["Col"]["SS"]
    rg_off = (burst.slant_range_time + drg - rg_mid).T * speed_of_light / 2
    rg_ori = (burst.slant_range_time + burst.drg[1, :, :] - rg_mid).T * speed_of_light / 2

    ph_ramp_diff = np.polynomial.polynomial.polyval2d(rg_off, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
        rg_ori, az_ori, coeff_demod
    )

    return xr.DataArray(
        ph_ramp_diff,
        dims=["azimuth_time", "slant_range_time"],
        coords={"slant_range_time": burst.slant_range_time, "azimuth_time": burst.azimuth_time},
    )


# def flat_single_burst(crg: xr.DataTree, slc_orig: xr.DataTree, reference: str, dd: str):

#     for k, burst in crg["burst"].items():
#         # iterate over each burst
#         slc = slc_orig[f"burst/{k}/orig"][dd].slc

# ph_ramp_diff = get_phramp_diff(burst, slc, dd)


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    w_ = w / 2.0
    return ((v + w_) % w) - w_
