import datetime
import warnings

import numpy as np
import xarray as xr
import xrspatial.convolution as conv

import s1_ard_core.computing.computation
from s1_ard_core.computing.io import Path, normpath, read, write
from s1_ard_core.computing.logger import logger


def to_dataset(X, acquisition_date, poly_degree, t0, r0, s_y, s_x):

    def to_polygrid2d(c):
        m = np.zeros((poly_degree + 1, poly_degree + 1), dtype="float64")
        m[:, 0] = np.flip(c)
        return m

    axis = np.arange(poly_degree + 1)
    da = [
        xr.DataArray(
            name=str(dd),
            data=to_polygrid2d(X[i, :]),
            dims=["i", "j"],
            coords={"i": axis, "j": axis},
            attrs={"deg": poly_degree, "offset": (t0, r0), "scale_fact": (s_y, s_x)},
        )
        for i, dd in enumerate(acquisition_date.values)
    ]
    return xr.Dataset({e.name: e for e in da})


def find_global_shift(
    esd: xr.Dataset,
    poly_degree: int = 0,
):
    """

    Starting from the shifts estimated by applying the ESD between
    different bursts and (if requested) by using multiple references,
    a system of linear equation is constructed and solved to obtain
    a polynomial function (of degree poly_degree) that describes the azimuth
    residual shift between each image of the stack and a reference date

    """
    import scipy.sparse
    import scipy.sparse.linalg

    # number of vars and eqs #
    n_eq = esd.acquisition_date.size
    n_vars = n_eq * (poly_degree + 1)

    # reference times both in azimuth and range #
    acquisition_date = esd.acquisition_date
    r0, t0 = [esd[e].mean() for e in ["slant_range_time", "azimuth_time"]]
    s_x, s_y = [(esd[e].max() - esd[e].min()).astype("float64") / 2.0 for e in ["slant_range_time", "azimuth_time"]]
    (i_mc,) = np.where(acquisition_date == esd.coregistration_reference_date)

    def create_values(y) -> np.ndarray:
        return np.vstack([y**i for i in range(poly_degree, -1, -1)]).T

    def create_matrix(c_esd):

        # current reference date index #
        (i_m,) = np.where(acquisition_date == c_esd.reference_date)
        # values to be stored into D and M matrices #
        v = create_values(((c_esd.azimuth_time - t0).astype("float64") / s_y).values)
        v = (v.repeat(n_eq, axis=0) * c_esd.coherence.values[..., None]).flatten()
        # diagonal matrix for all the slaves #
        col_ids = (
            np.arange(n_eq)[..., None] * (poly_degree + 1)
            + (np.tile(np.arange(poly_degree + 1), n_eq)).reshape(-1, poly_degree + 1)
        ).flatten()
        row_ids = np.repeat(np.arange(n_eq), poly_degree + 1)
        D = scipy.sparse.coo_matrix((v.flatten(), (row_ids, col_ids)), shape=(n_eq, n_vars))

        # reference matrix where only one column has values different from zero #
        col_ids = (
            np.repeat(i_m * (poly_degree + 1), n_eq)[..., None]
            + np.tile(np.arange(poly_degree + 1), n_eq).reshape(-1, poly_degree + 1)
        ).flatten()
        M = scipy.sparse.coo_matrix((v.flatten(), (row_ids, col_ids)), shape=(n_eq, n_vars))

        # removing from A and B the all invalid lines #
        i_v = ((np.isfinite(c_esd["shift"])) & (c_esd["shift"].acquisition_date != c_esd.reference_date)).values

        A = (D - M)[i_v, :]
        B = (c_esd["shift"].values * c_esd["coherence"].values)[i_v]
        return (A, B)

    A, B = zip(*[create_matrix(esd.sel(stacked=(id, date))) for (id, date) in esd.stacked.values])
    # global least square problem solving #
    A, B = scipy.sparse.vstack(A), np.hstack(B)
    logger.debug(f"A : {A.toarray().shape}")
    logger.debug(f"B : {B.shape}")

    X, istop, itn, r1norm, r2norm, _, _, _, _, _ = scipy.sparse.linalg.lsqr(A, B)

    # solution to y_poly structure #
    # When there is only one sub-swath s_x would be #
    # zero; in such a case I override it with one   #
    X = np.atleast_2d(X).reshape(n_eq, -1)

    X = to_dataset(
        X - X[i_mc, :],
        acquisition_date,
        poly_degree,
        t0.values,
        r0.values,
        np.round(s_y).astype("timedelta64[ns]").values,
        s_x.values if s_x.values > 0 else np.array(1.0),
    )

    return X.assign_attrs({"type": "times"})


def compute_azimuth_shifts(
    burst0: xr.Dataset,
    burst1: xr.Dataset,
    reference0: xr.Dataset,
    reference1: xr.Dataset,
    dtaz: float,
    eta: float,
    reference_dates: list[str],
    kernel_shape: list[int] = [3, 3],
    coherence_th: float = 0.7,
):

    def compute_coherence(
        interf: xr.DataArray,
        power_s: xr.DataArray,
        power_m: xr.DataArray,
    ) -> xr.DataArray:

        # kernel creation #
        k = np.ones(kernel_shape) / np.prod(kernel_shape)

        # numerator computation #
        num = np.abs(conv.convolution_2d(interf.real.squeeze(), k) + 1j * conv.convolution_2d(interf.imag.squeeze(), k))
        # denominator computation #
        den = np.sqrt(conv.convolution_2d(power_s.squeeze(), k) * conv.convolution_2d(power_m.squeeze(), k))
        return (num / den).expand_dims("acquisition_date").compute()

    def compute_shift(reference_date: str, f_d: float, dtaz: float) -> tuple[xr.DataArray]:

        if all(reference_date in e.acquisition_date for e in [reference0, reference1]):

            # interferograms and power computation #
            pow0 = (burst0 * burst0.conj()).real
            interf0 = burst0 * reference0.sel(acquisition_date=reference_date).conj()
            interf1 = burst1 * reference1.sel(acquisition_date=reference_date).conj()

            # coherence computation #
            coherence = compute_coherence(
                interf0, pow0, (reference0 * reference0.conj()).real.sel(acquisition_date=reference_date)
            )

            # coherence value lower than coherence_th will be ignored  #
            coherence = coherence.where(coherence >= coherence_th)

            # phase mean in the overlap area #
            interf01 = (interf0 * interf1.conj() * coherence).mean(dim=["slant_range_time", "azimuth_pixel"])
            mean_phase = xr.apply_ufunc(np.angle, interf01.compute())
            mean_phase = xr.DataArray(
                data=mean_phase, dims=["acquisition_date"], coords={"acquisition_date": interf01.acquisition_date}
            )

            # phase to shift convertion #
            shifts = (mean_phase / (2.0 * np.pi * f_d * dtaz)).rename("shifts")

            # TODO: check if this version of the formula is right. In the previous
            #      implementation it was: wrap_cycle = 1.0 / (2 * f_d * dtaz)
            #      I think it was wrong, the divide by 2 is not required
            #      wrap cycle to be used to unwrap estimated shifts
            wrap_cycle = 1.0 / (f_d * dtaz)

            # shifts simple unwrapping attempt #
            shifts = shifts - wrap_cycle * np.round((shifts - shifts.median()) / wrap_cycle)

            # return the obtained shifts and the corresponding mean coherences #
            coherence = coherence.mean(dim=["azimuth_pixel", "slant_range_time"])
        else:

            # if the reference_date is not in the burst nan shift and coherence will be returned #
            coords = burst0.acquisition_date.coords
            data = np.full(burst0.acquisition_date.size, np.nan)
            shifts = xr.DataArray(name="shifts", data=data, coords=coords)
            coherence = xr.DataArray(name="coherence", data=data, coords=coords)

        return (reference_date, shifts.compute(), coherence.rename("coherence").compute())

    # Computation of doppler frequency at the edge of the burst0 #
    annot = burst0.attrs[list(burst0.attrs)[0]]
    coeffs = annot["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]
    sm, am = np.meshgrid(burst0.slant_range_m, burst0.azimuth_m[-1:])
    f_d = np.polynomial.polynomial.polyval2d(sm, am, np.polynomial.polynomial.polyint(coeffs, axis=1))
    f_d = np.mean(f_d * (4.0 / eta))
    with warnings.catch_warnings():
        # Silence warning related to divide and multiply
        warnings.filterwarnings("ignore", message=".*invalid value encountered in.*")
        warnings.filterwarnings("ignore", message=".*overflow encountered in multiply.*")

        # shifts estimation for each reference_date that is also a date of burst0
        reference, shift, cohe = zip(*[compute_shift(e, f_d, dtaz) for e in reference_dates])

    # obtained shifts returned as a xr.Dataset #
    return xr.Dataset(
        {
            "reference_date": list(reference),
            "coherence": xr.concat(cohe, dim="reference_date"),
            "shift": xr.concat(shift, dim="reference_date"),
        }
    )


def esd_single_burst(
    burst_id: str,
    working_directory: Path,
    burst_list: list[str],
    reference_dates: list[str],
    kernel_shape: list[int] = [3, 3],
    coherence_th: float = 0.7,
    **kwargs,
):
    """

    Compute the residual shift in the azimuth direction applying the ESD procedure.
    It returns the estimeted shift (in pixel), the corresponding reliability index
    and the position of the overlap area used to perform the estimate
    (i.e the azimuth and slant_range coordinates of the baricenter of the area
    of overlap between the two consecutive bursts used to perform the ESD)

    :param slcs_path: path to the input datatree
    :param burst_list: list of burst identifier the datatree contains
    :param c_id: identifier (e.g iw1_254) of the burst for which the ESD must be computed
    :param reference_dates: a list or a single date that corresponds to the image(s) to be
                            used as reference by the shift estimate procedure. In case of a
                            list of images, the shifts will be estimated with respect to
                            all the images in the list

    :return: dictionary with the following structure:
             {
               (reference_date_0, x, y): shifts_0, mean_coherence_0,
               (reference_date_1, x, y): shifts_1, mean_coherence_1,
               ...
               (reference_date_n-1, x, y): shifts_n-1, mean_coherence_n-1,
             }
             where shifts_i and mean_coherence_i are the dataarrays of shifts
             and of mean coherence values, respectively, obtained using the
             reference_date_i image as ESD reference

    """

    def get_next_burst_id():
        swath, id = burst_id.split("_")
        next_burst_id = f"{swath}_{int(id)+1}"
        return next_burst_id

    def time_to_pixel(stack: xr.Dataset, dtaz: float, reference_time: datetime.datetime | None = None):

        if reference_time is None:
            reference_time = stack.azimuth_time[0].values
        azimuth_pixel = np.round((stack.azimuth_time - reference_time).astype("float64") / 1e9 / dtaz).astype("int32")
        stack = stack.assign_coords({"azimuth_pixel": azimuth_pixel})
        return stack.swap_dims(azimuth_time="azimuth_pixel"), reference_time

    # current coregistered burst extraction #
    current_burst = read(working_directory.append(f"bursts/{burst_id}/coreg/measurements"), consolidate_metadata=False)

    next_burst_id = get_next_burst_id()
    if next_burst_id in burst_list:

        # next coregistered burst extraction #
        logger.info(f"Processing overlap area between bursts: {burst_id} and {next_burst_id}")
        next_burst = read(
            working_directory.append(f"bursts/{next_burst_id}/coreg/measurements"), consolidate_metadata=False
        )
        annot = current_burst.cstack.attrs
        annot = annot[list(annot)[0]]
        dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"]
        eta = (current_burst.azimuth_time[-1] - current_burst.azimuth_time[0]).values.astype("float64") / 1e9 / 2.0
        # swap azimuth_time dimension with a new azimuth time in pixel #
        current_burst, reference_time = time_to_pixel(current_burst, dtaz)
        next_burst, _ = time_to_pixel(next_burst, dtaz, reference_time)

        # now the overlap area between burst can be easily computed #
        sel = next_burst.drop_vars(next_burst.data_vars) + current_burst.drop_vars(current_burst.data_vars)

        v_dims = [j for j in sel.dims if j not in ["i", "j"]]

        available_pols = set(current_burst.polarization.values)
        priorities = ["V:V", "H:H", "V:H", "H:V"]
        pol = next((p for p in priorities if p in available_pols), None)
        if pol is None:
            raise Exception(f"No polarization has been found {available_pols}")
        # overlap area extraction from current and next bursts #
        next_burst = next_burst.sel({k: sel.coords[k] for k in v_dims}).cstack.sel(polarization=pol)
        current_burst = current_burst.sel({k: sel.coords[k] for k in v_dims}).cstack.sel(polarization=pol)

        # returns the estimated azimuth shift and the corresponding quality index #
        logger.info(f"Computing of azimuth shifts")

        next_reference = next_burst.sel(acquisition_date=reference_dates)
        current_reference = current_burst.sel(acquisition_date=reference_dates)

        esd = s1_ard_core.computing.computation.map_blocks(
            compute_azimuth_shifts,
            (current_burst, next_burst),
            args=(current_reference, next_reference),
            kwargs={
                "dtaz": dtaz,
                "eta": eta,
                "reference_dates": reference_dates,
                "kernel_shape": kernel_shape,
                "coherence_th": coherence_th,
            },
            chunksizes={"acquisition_date": 1},
            enable_dask=True,
        )

        # new dimension id and corresponding coords added to the shifts #
        coords = {k: (["id"], [current_burst[k].mean().values]) for k in ["azimuth_time", "slant_range_time"]}
        esd = esd.expand_dims({"id": [burst_id]}).assign_coords(coords)
    else:
        logger.info(f"ESD computation skipped: burst ({next_burst_id}) does not exist")
        esd = None

    return esd


def estimate(
    slcs: xr.DataTree,
    working_directory: str | Path,
    reference_dates: list[str] | str,
    poly_degree: int = 0,
    **kwargs,
):
    key = "coregistration_reference_date"
    if not isinstance(reference_dates, (list, tuple)):
        reference_dates = [reference_dates]
    if isinstance(working_directory, str):
        working_directory = Path(working_directory)
    # Residual azimuth shifts estimation for each bursts
    esd = s1_ard_core.computing.computation.map(
        esd_single_burst,
        args=list(slcs.bursts),
        kwargs={
            "working_directory": working_directory,
            "burst_list": list(slcs.bursts),
            "reference_dates": reference_dates,
        },
        combine=False,
        **kwargs,
    )

    # Structure re-formatting in order to solve the global problem
    esd = xr.merge([e for e in esd if e is not None], compat="no_conflicts", combine_attrs="drop", join="outer").stack(
        stacked=("id", "reference_date")
    )

    # Auxiliary data saved as attributes of the dataset
    esd = esd.assign_attrs({key: list(slcs.bursts.children.values())[0].coreg.measurements.attrs[key]})
    return find_global_shift(esd, poly_degree=poly_degree)
