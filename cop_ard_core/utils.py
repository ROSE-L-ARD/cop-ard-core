import numpy as np
import xarray as xr
import scipy.fft
from s1_ard_core.computing.io import write, path_to_dict
from s1_ard_core.computing.logger import logger, timeit
from scipy.constants import speed_of_light


# @timeit
def extract_subband(cslc: xr.DataArray, bw: float, f0: float, dim_transf: str = 'slant_range_time') -> xr.DataArray:

    dim = cslc.dims
    freq_dim = [f"freq_{e}" if e == dim_transf else e for e in cslc.dims] 
    axis = cslc.get_axis_num(dim_transf)
    N = cslc.sizes[dim_transf]
    t_coords = cslc[dim_transf].values
    coord_trasf = (np.arange(N) - N // 2)  / (t_coords[-1] - t_coords[0])
    # coord_trasf = (np.arange(cslc[dim_trasf].size) - cslc[dim_trasf].size // 2)  / (cslc[dim_trasf][-1] - cslc[dim_trasf][0]).values
    freq_coords = {
        (f"freq_{e}" if e == dim_transf else e): (coord_trasf if e == dim_transf else cslc.coords[e].values)
        for e in cslc.dims
    }
 
    frange = xr.DataArray(scipy.fft.fftshift(scipy.fft.fft(cslc.values, norm="ortho", axis=axis), axes=(axis,)), dims=freq_dim, coords=freq_coords)

    mask = (frange.freq_slant_range_time > f0 - bw / 2) * (frange.freq_slant_range_time < f0 + bw / 2)

    return xr.DataArray(scipy.fft.ifft((frange * mask).values, norm='ortho', axis=axis), dims=dim, coords=cslc.drop_vars(["slant_range_m", "azimuth_m"]).coords)


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

def get_burst_duration(slcs: xr.DataTree, path: str) -> list:

    azimuth_time, azimuth_time_raw = [], []

    for burst in slcs.values():
        if burst[path].azimuth_time.size > 0:
            _, annot = next(iter(burst[path].attrs.items()))
            
            # focsed zero-Doppler azimuth time
            azimuth_time += [
                (burst[path].azimuth_time.max().values - burst[path].azimuth_time.min().values).astype(float) / 1e9
            ]

            # raw azimuth time
            slant_range_m = burst[path].slant_range_m.values[::10]
            azimuth_m = burst[path].azimuth_m.values[::10]

            coeffs = annot['Grid']['TimeCOAPoly']['Coefs']
            
            TimeCOA = np.polynomial.polynomial.polygrid2d(slant_range_m, azimuth_m, coeffs ).T.mean(axis=1)
            azimuth_time_raw += [np.nanmax(TimeCOA) - np.nanmin(TimeCOA)]
            
    if len(azimuth_time) <= 0:
        raise ValueError("Null slc in input")

    return [
        np.mean(azimuth_time), np.mean(azimuth_time_raw)
    ]


def get_azimuth_time_raw(burst: xr.DataTree, multilook: dict| None=None) -> xr.DataArray:

    # retrieve image annotation
    # the polarization is equivalent
    _, annot = next(iter(burst.attrs.items()))

    if multilook is not None:   
        az_slice = slice(multilook['azimuth_time']//2,None,multilook['azimuth_time'])
        rg_slice = slice(multilook['slant_range_time']//2,None,multilook['slant_range_time'])
    else:
        az_slice = slice(None, None, None)
        rg_slice = slice(None, None, None)

    burst_ = burst.isel(azimuth_time=az_slice, slant_range_time=rg_slice)
    
    # retrieve axis and coefficients
    slant_range_m = burst_.slant_range_m.values
    azimuth_m = burst_.azimuth_m.values
    coeffs = annot['Grid']['TimeCOAPoly']['Coefs']

    # compute the relative azimuth time
    timeCOA = np.polynomial.polynomial.polygrid2d(slant_range_m, azimuth_m, coeffs ).T
    
    # convert to timedelta in nanoseconds
    timeCOA = (timeCOA*1e9).astype('timedelta64[ns]')

    # compute absolute time
    timeCOA = timeCOA + np.datetime64(annot["Timeline"]["CollectStart"][:-1])

    # return as dataarray
    return xr.DataArray(timeCOA, dims=['azimuth_time', 'slant_range_time'], 
                        coords={'azimuth_time': burst_.azimuth_time.values, 
                                'slant_range_time': burst_.slant_range_time.values})


def get_carrier_frequency(slcs: xr.DataTree, path: str) -> float:
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
        _, annot = next(iter(burst[path].attrs.items()))
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


def get_coordinates(y, x, alpha, beta):
    """
    Calculates coordinates for polynomial evaluation based on offsets and scaling.
    """
    return (((e - alpha[i]).astype("float64") / beta[i].astype("float64")).values for i, e in enumerate([y, x]))


# @timeit
def get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact):
    c1, c2 = get_coordinates(azimuth_time, slant_range_time, offset, scale_fact)
    return np.polynomial.polynomial.polygrid2d(c1, c2, y_poly)


def get_az_rg_meters(annot):
    """
    Compute azimuth and slant range distance arrays (in meters) from SICD metadata.
    """

    try:
        # Access metadata fields directly from the Sarpy object.
        daz = annot.Grid.Col.SS
        azimuth_m = (np.arange(annot.ImageData.FullImage.NumCols) - annot.ImageData.SCPPixel.Col) * daz
        slant_range_m = (
            np.arange(annot.ImageData.FullImage.NumRows) - annot.ImageData.SCPPixel.Row
        ) * annot.Grid.Row.SS
    except AttributeError:
        # Fallback to dictionary-based access.
        daz = path_to_dict(annot, "Grid.Col.SS")
        drg = path_to_dict(annot, "Grid.Row.SS")
        num_cols = path_to_dict(annot, "ImageData.FullImage.NumCols")
        scp_col = path_to_dict(annot, "ImageData.SCPPixel.Col")
        num_rows = path_to_dict(annot, "ImageData.FullImage.NumRows")
        scp_row = path_to_dict(annot, "ImageData.SCPPixel.Row")
        azimuth_m = (np.arange(num_cols) - scp_col) * daz
        slant_range_m = (np.arange(num_rows) - scp_row) * drg
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_m, slant_range_m = None, None
        raise
    return azimuth_m, slant_range_m

# @timeit
def get_az_rg_times(annot):
    """
    Compute azimuth and slant range time coordinate arrays from SICD metadata.
    """
    import pandas as pd

    try:
        # Access metadata fields directly from the Sarpy object.
        num_cols = annot.ImageData.FullImage.NumCols
        num_rows = annot.ImageData.FullImage.NumRows
        scp_col = annot.ImageData.SCPPixel.Col
        scp_row = annot.ImageData.SCPPixel.Row
        time_ca_poly = annot.RMA.INCA.TimeCAPoly.Coefs
        collect_start = annot.Timeline.CollectStart
        col_ss = annot.Grid.Col.SS
        row_ss = annot.Grid.Row.SS
        r_ca_scp = annot.RMA.INCA.R_CA_SCP
    except AttributeError:
        # Fallback to dictionary-based access if the input is a dict.
        num_cols = path_to_dict(annot, "ImageData.FullImage.NumCols")
        num_rows = path_to_dict(annot, "ImageData.FullImage.NumRows")
        scp_col = path_to_dict(annot, "ImageData.SCPPixel.Col")
        scp_row = path_to_dict(annot, "ImageData.SCPPixel.Row")
        time_ca_poly = path_to_dict(annot, "RMA.INCA.TimeCAPoly.Coefs")
        raw_time = path_to_dict(annot, "Timeline.CollectStart")
        collect_start = pd.to_datetime(raw_time, utc=True).tz_localize(None).to_numpy()
        col_ss = path_to_dict(annot, "Grid.Col.SS")
        row_ss = path_to_dict(annot, "Grid.Row.SS")
        r_ca_scp = path_to_dict(annot, "RMA.INCA.R_CA_SCP")
    except Exception as e:
        logger.critical(f"Caught an unexpected error: {type(e).__name__} - {e}")
        azimuth_time, slant_range_time = None, None
        raise
    finally:
        # Calculate azimuth time based on the TimeCAPoly polynomial.
        dtaz = time_ca_poly[1] * col_ss
        azimuth_time_float = (np.arange(num_cols, dtype="float64") - scp_col) * dtaz + time_ca_poly[0]
        azimuth_time = (np.round(azimuth_time_float * 1e9)).astype("timedelta64[ns]") + collect_start
        # Calculate slant range time based on the distance from the scene center.
        slant_range_time = 2 * ((np.arange(num_rows) - scp_row) * row_ss + r_ca_scp) / speed_of_light
    return azimuth_time, slant_range_time

# @timeit
def get_phramp_diff_(azimuth_time: xr.DataArray,
                     slant_range_time: xr.DataArray, 
                     drg: xr.DataArray, 
                     daz: xr.DataArray, 
                     annot: dict, 
                     y_poly: xr.DataArray,
                     multilook: dict):

    # @timeit
    def poly_combine(rg, az_off, az_ori, coeff_demod):
        return np.polynomial.polynomial.polyval2d(rg, az_off, coeff_demod) - np.polynomial.polynomial.polyval2d(
            rg, az_ori, coeff_demod
        )
        
    # extract demodulation coefficients
    coeff_demod = get_coeff_demod(annot)

    # compute spacing and start times
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    scale_daz = 1 / (annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * 1e9)

    # compute azimuth correction
    daz_corr = get_daz_corr(azimuth_time, slant_range_time, y_poly.values, y_poly.offset, y_poly.scale_fact)

    # Corrected slant range and azimuth positions
    daz_corr = daz + np.round(daz_corr * dtaz).astype("timedelta64[ns]")

    # centre scene coordinates
    az_time, rg_time = get_az_rg_times(annot)

    rg_mid = rg_time[annot["ImageData"]["SCPPixel"]["Row"]]
    az_mid = az_time[annot["ImageData"]["SCPPixel"]["Col"]]

    # compute the position in the secondary image corresponding to the reference one with and without corrections in meters wrt the scene centre
    az_off = ((azimuth_time + daz_corr - az_mid).astype(float) * scale_daz).values
    az_ori = ((azimuth_time + daz - az_mid).astype(float) * scale_daz).values

    # range in meters wrt the burst center
    rg = ((slant_range_time + drg - rg_mid).T * (speed_of_light / 2)).values 

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

# @timeit
def get_phramp_diff(interf: xr.Dataset, drg_full: xr.DataArray, daz_full: xr.DataArray, annot: dict, y_poly: xr.Dataset, multilook: dict):
    ph_ramp_diff_list = []

    for k, dd in enumerate(interf.acquisition_date.values):
        #  call core function
        y_poly[dd].attrs['type'] = y_poly.type
        res = get_phramp_diff_(interf.azimuth_time,
                               interf.slant_range_time, 
                               drg_full.sel(acquisition_date=dd).coarsen(multilook, boundary='trim').mean(), 
                               daz_full.sel(acquisition_date=dd).coarsen(multilook, boundary='trim').mean(),
                               annot[dd], y_poly[dd], multilook)
        ph_ramp_diff_list.append(res)

    # return the concatenated dataarray
    return xr.concat(ph_ramp_diff_list, dim="acquisition_date").assign_coords(
        {"acquisition_date": interf.acquisition_date}
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
    data_stack = data_stack.assign_coords(ind=ind).compute()

    v_mean = data_stack.groupby(["ind"]).median()

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

    res = griddata(points, data.phase.values, out_points, method="cubic")

    return xr.Dataset(
        data_vars={"phase": (["azimuth_time", "slant_range_time"], res.reshape(gridxx.shape))},
        coords=out_coords,
    )


def encode_poly(v_poly: xr.Dataset, name: str, stack: xr.Dataset) -> xr.Dataset:
    """
    Return a version of input v_poly object so that it can be saved.

    Merges separate variables in v_poly into a single variable aligned
    with stack.acquisition_date.
    """

    def _safe_encode(obj):
        """Recursively encode attributes for serialization."""
        if isinstance(obj, (list, tuple)):
            return [_safe_encode(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _safe_encode(v) for k, v in obj.items()}
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()  # Convert numpy scalars to python scalars
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        else:
            return obj

    # 1. Extract attributes BEFORE transforming the dataset
    # We capture attributes from variables that match the stack's dates
    poly_attrs = {}

    # We convert stack dates to the same type as v_poly keys (usually strings) for checking
    stack_date_lookup = set(stack.acquisition_date.astype(str).values)

    for var_name, data_array in v_poly.data_vars.items():
        # Check if this variable name corresponds to a date in the stack
        if str(var_name) in stack_date_lookup:
            poly_attrs[str(var_name)] = _safe_encode(data_array.attrs)

    # 2. Transform: Convert variables (dates) into a dimension
    da_poly = v_poly.to_array(dim="acquisition_date", name=name)

    # 3. Ensure Type Consistency (Crucial step)
    # to_array usually results in string/object coords. Cast to datetime to match stack.
    try:
        da_poly["acquisition_date"] = da_poly.acquisition_date.astype("datetime64[ns]")
    except ValueError:
        # Fallback if keys weren't date strings
        pass

    # 4. Filter and Assign Attributes
    da_poly = (
        da_poly.sel(acquisition_date=stack.acquisition_date)  # Filter dates
        .assign_attrs(poly_attrs)  # Add the attrs we extracted earlier
        .assign_attrs(_safe_encode(stack.attrs))  # Add stack global attrs
    )

    # 5. Convert back to Dataset
    ds_out = da_poly.to_dataset()

    # 6. Chunking
    # We use .get() to avoid KeyErrors if dims are missing or not chunked
    chunk_plan = {"acquisition_date": stack.chunksizes.get("acquisition_date", "auto"), "i": 1, "j": 1}

    # Only chunk dimensions that actually exist in the output
    final_chunks = {k: v for k, v in chunk_plan.items() if k in ds_out.dims}

    return ds_out.chunk(final_chunks)



def decode_poly(v_poly: xr.Dataset, name: str, stack: xr.Dataset) -> xr.Dataset:
    """
    Read and reformat x_ or y_poly from the slcs structure.

    Splits the 'acquisition_date' dimension back into separate variables.
    """

    def _safe_decode(obj):
        """Recursively decode attributes, handling strings to datetime/timedelta."""
        if isinstance(obj, dict):
            return {k: _safe_decode(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_safe_decode(x) for x in obj]
        elif isinstance(obj, str):
            # 1. Try Timedelta (specific format from encode)
            if "nanoseconds" in obj:
                try:
                    # Extracts "12345" from "12345 nanoseconds"
                    val = float(obj.split()[0])
                    return np.timedelta64(int(val), "ns")
                except (ValueError, IndexError):
                    pass

            # 2. Try Datetime
            # Only attempt conversion if it looks like a date (ISO-ish) to avoid false positives
            if "-" in obj and ":" in obj:
                try:
                    return np.datetime64(obj)
                except ValueError:
                    pass

            # Return original string if no conversion applied
            return obj

        # Return numbers/arrays as is
        return obj

    # Dictionary to hold the new variables
    data_vars = {}

    # We iterate over the dates in the stack
    # We assume stack["acquisition_date"] exists
    dates = stack["acquisition_date"].values

    for date in dates:
        # Convert date to string to find the matching attribute key
        # (Assuming encode_poly saved keys as strings)
        date_key = str(date)

        # 1. Select data lazily
        # drop=True removes 'acquisition_date' so it doesn't persist as a scalar coord
        da_slice = v_poly[name].sel(acquisition_date=date, drop=True)

        # 2. Decode specific attributes for this date
        # We use .get() in case the attribute is missing
        raw_attrs = da_slice.attrs.get(date_key, {})
        da_slice.attrs = _safe_decode(raw_attrs)

        # 3. Add to dictionary
        # We use the string representation of the date as the variable name
        data_vars[date_key] = da_slice

    # Create the final dataset
    decoded_ds = xr.Dataset(data_vars)

    # Preserve the specific 'type' attribute from the input
    decoded_ds = decoded_ds.assign_attrs({"type": v_poly.attrs.get("type", "times")})

    return decoded_ds


def get_interf(a,b,ml_factor):
    return (a*b.conj()).coarsen(ml_factor, boundary='trim').mean()

# def get_normalized_interf(a, b, ml_factor, rel_thresh=None, abs_thresh=1e-6):
#     cross = get_interf(a, b, ml_factor)
#     p_a = get_interf(a, a, ml_factor)
#     p_b = get_interf(b, b, ml_factor)
    
#     coh = cross / np.sqrt(p_a * p_b)
    
#     valid_mask = (p_a > abs_thresh) & (p_b > abs_thresh)
    
#     if rel_thresh is not None:
#         is_actively_updated = p_a > (rel_thresh * p_b)
#         valid_mask = valid_mask & is_actively_updated
        
#     return coh.where(valid_mask, 0j).fillna(0j)



def get_normalized_interf(a,b,ml_factor):
    return (get_interf(a,b,ml_factor) / np.sqrt(get_interf(a,a,ml_factor) * get_interf(b,b,ml_factor))).fillna(0j)


def time_to_pixel(stack: xr.Dataset, y0: float | None = None, x0: float | None = None) -> xr.Dataset:
    """

    Convert from times to pixel dimensions of the input stack

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