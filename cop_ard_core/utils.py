import numpy as np
import xarray as xr
import xrft
from s1_ard_core.computing.io import write
from s1_ard_core.computing.logger import logger, timeit
from scipy.constants import speed_of_light


@timeit
def extract_subband(cslc: xr.DataArray, bw: float, f0: float) -> xr.DataArray:
    frange = xrft.fft(
        cslc.drop_vars(["slant_range_m", "azimuth_m"]).chunk(slant_range_time=-1),
        true_amplitude=True,
        true_phase=True,
        dim="slant_range_time",
    )

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)

    return xrft.ifft(
        (frange * mask).chunk(freq_slant_range_time=-1),
        dim=["freq_slant_range_time"],
        true_amplitude=True,
        true_phase=True,
    )


def subband_mapping(burst: str, cslc: xr.DataTree, f_c: float, bw: float, id: str, wdir_path: str):
    chunks_burst = {k: v[0] for k, v in cslc.chunksizes.items()}
    if cslc.acquisition_date.size != 2:
        logger.info(f"Skipping burst {burst} with {cslc.acquisition_date.size} acquisitions")
        return None, None

    frange = xrft.fft(
        cslc.drop_vars(["slant_range_m", "azimuth_m"]).chunk(slant_range_time=-1),
        true_amplitude=True,
        true_phase=True,
        dim="slant_range_time",
    ).persist()

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
            azimuth_time += [
                burst[path].azimuth_time.min().values,
                burst[path].azimuth_time.max().values,
            ]
        if burst[path].slant_range_time.size > 0:
            slant_range_time += [
                burst[path].slant_range_time.min().values,
                burst[path].slant_range_time.max().values,
            ]

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


@timeit
def get_daz_corr(burst, y_poly, dd):
    c1, c2 = get_coordinates(burst, y_poly[dd].offset, y_poly[dd].scale_fact, y_poly.type)
    return np.polynomial.polynomial.polygrid2d(c1, c2, y_poly[dd].values)


@timeit
def get_phramp_diff(burst: xr.Dataset, orig: xr.DataTree, y_poly: xr.Dataset):
    ph_ramp_diff_list = []

    @timeit
    def poly_combine(rg, az_off, az_ori, coeff_demod):
        return np.polynomial.polynomial.polyval2d(rg, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
            rg, az_ori, coeff_demod
        )

    for k, dd in enumerate(burst.acquisition_date.values):
        slc = orig[f"{dd}/measurements/slc"]
        annot = slc.attrs[list(slc.attrs)[0]]

        # extract demodulation coefficients
        coeff_demod = get_coeff_demod(annot)

        # compute spacing and start times
        dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
        scale_daz = 1 / (annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * 1e9)

        # compute azimuth correction
        daz_corr = get_daz_corr(burst, y_poly, dd)

        # Corrected slant range and azimuth positions
        drg = burst.drg.sel(acquisition_date=dd)
        daz = burst.daz.sel(acquisition_date=dd) + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

        # centre scene coordinates
        rg_mid = slc.isel(slant_range_time=annot["ImageData"]["SCPPixel"]["Row"]).slant_range_time.values
        az_mid = slc.isel(azimuth_time=annot["ImageData"]["SCPPixel"]["Col"]).azimuth_time.values

        # compute the position in the secondary image corresponding to the reference one with and without corrections
        az_off = ((burst.azimuth_time + daz - az_mid).astype(float) * scale_daz).values
        az_ori = ((burst.azimuth_time + burst.daz.sel(acquisition_date=dd) - az_mid).astype(float) * scale_daz).values

        rg = ((burst.slant_range_time + drg - rg_mid).T * (speed_of_light / 2)).values

        # recontruct polynomial function
        ph_ramp_diff_da = xr.DataArray(
            poly_combine(rg, az_off, az_ori, coeff_demod),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "slant_range_time": burst.slant_range_time,
                "azimuth_time": burst.azimuth_time,
            },
        )

        ph_ramp_diff_list.append(ph_ramp_diff_da)

    # return the concatenated dataarray
    return xr.concat(ph_ramp_diff_list, dim="acquisition_date").assign_coords(
        {"acquisition_date": burst.acquisition_date}
    )


def wrap(v: np.ndarray, w: float = 2 * np.pi):
    w_ = w / 2.0
    return ((v + w_) % w) - w_


def sparse_multilook(data: xr.Dataset, cohe_th: float, multilook: dict):
    coords = list(multilook.keys())
    if not np.all([e in data.dims for e in coords]):
        raise Exception(f"Not all {coords} in {data.dims}")

    # set temporary linear coords
    azimuth_time = data.azimuth_time.values
    slant_range_time = data.slant_range_time.values
    az0 = azimuth_time[0]
    rg0 = slant_range_time[0]
    dtaz = (azimuth_time[-1] - azimuth_time[0]).astype(float) / (azimuth_time.size - 1)
    dtrg = (slant_range_time[-1] - slant_range_time[0]) / (slant_range_time.size - 1)
    y_ = (azimuth_time - az0).astype(float) / dtaz
    x_ = (slant_range_time - rg0) / dtrg

    data_copy = data.copy(deep=True)
    data_copy = data_copy.assign_coords({"y_": ("azimuth_time", y_), "x_": ("slant_range_time", x_)})
    data_copy = data_copy.swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"})

    # pewrform multilook considering temporary coords
    multilook_yx = {
        "y_": multilook["azimuth_time"],
        "x_": multilook["slant_range_time"],
    }
    data_ml = sparse_multilook_core(data_copy, cohe_th=cohe_th, multilook=multilook_yx)

    # restore original coords
    az_ml = (data_ml.y_.data * dtaz).astype("timedelta64[ns]") + az0
    rg_ml = data_ml.x_.data * dtrg + rg0
    data_ml = data_ml.assign_coords({"azimuth_time": ("y_", az_ml), "slant_range_time": ("x_", rg_ml)})
    data_ml = data_ml.swap_dims({"y_": "azimuth_time", "x_": "slant_range_time"})
    data_ml = data_ml.drop_vars(["y_", "x_"])

    return data_ml


def sparse_multilook_core(data: xr.Dataset, cohe_th: float, multilook: dict):
    coords = list(multilook.keys())

    x_size_ml = data[coords[1]].size // multilook[coords[1]] + 1

    data_stack = data.stack(points=coords)
    data_stack = data_stack.where((data_stack.cohe > cohe_th).compute(), drop=True)

    x_ml = np.round(data_stack[coords[1]] // multilook[coords[1]]).astype(int)
    y_ml = np.round(data_stack[coords[0]] // multilook[coords[0]]).astype(int)
    ind = (x_ml + y_ml * x_size_ml).astype(int)
    data_stack = data_stack.assign_coords(ind=ind)

    v_mean = data_stack.groupby(["ind"]).mean()

    coords_ml = {e: ("points", data_stack[e].astype(float).groupby(["ind"]).mean().data) for e in coords}

    data_ml = xr.Dataset(
        v_mean,
        coords=coords_ml,
    )

    return data_ml


def griddata_interp(data: xr.Dataset, out_coords):
    from scipy.interpolate import griddata

    # set temporary linear coords
    azimuth_time = data.azimuth_time.values
    slant_range_time = data.slant_range_time.values
    az0 = azimuth_time[0]
    rg0 = slant_range_time[0]
    dtaz = (azimuth_time[-1] - azimuth_time[0]).astype(float) / (azimuth_time.size - 1)
    dtrg = (slant_range_time[-1] - slant_range_time[0]) / (slant_range_time.size - 1)
    y_ = (azimuth_time - az0).astype(float) / dtaz
    x_ = (slant_range_time - rg0) / dtrg
    points = np.vstack([x_, y_]).T

    data_copy = data.copy(deep=True)
    data_copy = data_copy.assign_coords({"y_": ("azimuth_time", y_), "x_": ("slant_range_time", x_)})
    data_copy = data_copy.swap_dims({"azimuth_time": "y_", "slant_range_time": "x_"})

    gridx, gridy = out_coords["slant_range_time"], out_coords["azimuth_time"]
    gridx = (gridx - rg0) / dtrg
    gridy = (gridy - az0).astype(float) / dtaz
    gridxx, gridyy = np.meshgrid(gridx, gridy)
    out_points = np.vstack((gridxx.ravel(), gridyy.ravel())).T

    print(points.shape)
    print(out_points.shape)

    res = griddata(points, data.phase.values, out_points, method="cubic")

    return xr.Dataset(
        data_vars={"phase": (["azimuth_time", "slant_range_time"], res.reshape(gridxx.shape))},
        coords=out_coords,
    )
