import xarray as xr
import xrft
from scipy.constants import speed_of_light
from s1_ardcore.computing.io import write
import numpy as np


def extract_subband(frange: xr.DataArray, bw: float, f0: float) -> xr.DataArray:

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)
    sel = frange * mask

    return xrft.ifft(
        (sel).chunk(freq_slant_range_time=-1), dim=["freq_slant_range_time"], true_amplitude=True, true_phase=True
    )


def subband_mapping(burst: str, cslc: xr.DataTree, f_c: float, bw: float, id: str, wdir_path: str):
    chunks_burst = {k: v[0] for k, v in cslc.chunksizes.items()}
    if cslc.acquisition_date.size != 2:
        print(f"Skipping burst {k} with {cslc.acquisition_date.size} acquisitions")
        return None, None

    print(50 * "=")
    # select area according to the output multilook grid
    print("Select area ...")

    # compute full range spectrum
    print("Computing full range spectrum...")
    frange = xrft.fft(
        cslc.drop_vars(["slant_range_m", "azimuth_m"]).chunk(slant_range_time=-1),
        true_amplitude=True,
        true_phase=True,
        dim="slant_range_time",
    ).persist()

    # extract low and high freq subbands
    print("Extracting subbands...")

    # extract slc sub band
    cslc_sub_band = (
        extract_subband(frange, bw, f_c)
        .to_dataset(name="cstack")
        .compute()
        .chunk(chunks_burst)
        .assign_coords(cslc.coords)
        .assign_attrs(cslc.attrs)
    )
    write(cslc_sub_band, f"{wdir_path}/{id}/bursts/{burst}/coreg/measurements", mode="w")


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


def get_coeff_demod(annot: dict):

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


def get_phramp_diff(burst: xr.Dataset, orig: xr.DataTree, y_poly: xr.Dataset):
    ph_ramp_diff_list = []

    for k, dd in enumerate(burst.acquisition_date.values):
        slc = orig[f"{dd}/measurements/slc"]
        annot = slc.attrs[list(slc.attrs)[0]]

        # extract demodulation coefficients
        coeff_demod = get_coeff_demod(annot)

        # compute spacing and start times
        dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9

        # compute azimuth correction
        c1, c2 = get_coordinates(burst, y_poly[dd].offset, y_poly[dd].scale_fact, y_poly.type)
        daz_corr = np.polynomial.polynomial.polygrid2d(c1, c2, y_poly[dd].values)

        # Corrected slant range and azimuth positions
        drg = burst.drg.sel(acquisition_date=dd)
        daz = burst.daz.sel(acquisition_date=dd) + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

        # centre scene coordinates
        rg_mid = slc.isel(slant_range_time=annot["ImageData"]["SCPPixel"]["Row"]).slant_range_time.values
        az_mid = slc.isel(azimuth_time=annot["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values

        # compute the position in the secondary image corresponding to the reference one with and without corrections
        az_off = (burst.azimuth_time + daz - az_mid).astype(float) / dtaz * annot["Grid"]["Col"]["SS"]
        az_ori = (
            (burst.azimuth_time + burst.daz.sel(acquisition_date=dd) - az_mid).astype(float)
            / dtaz
            * annot["Grid"]["Col"]["SS"]
        )
        rg = (burst.slant_range_time + drg - rg_mid).T * speed_of_light / 2

        ph_ramp_diff = np.polynomial.polynomial.polyval2d(rg, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
            rg, az_ori, coeff_demod
        )
        ph_ramp_diff_list.append(
            xr.DataArray(
                ph_ramp_diff,
                dims=["azimuth_time", "slant_range_time"],
                coords={"slant_range_time": burst.slant_range_time, "azimuth_time": burst.azimuth_time},
            )
        )

    return xr.concat(ph_ramp_diff_list, dim="acquisition_date").assign_coords(
        {"acquisition_date": burst.acquisition_date}
    )


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    w_ = w / 2.0
    return ((v + w_) % w) - w_