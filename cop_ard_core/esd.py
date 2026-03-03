import datetime
import warnings
from typing import cast

import numpy as np
import s1_ard_core.computing.computation
import scipy.optimize
import xarray as xr
import xrspatial.convolution as conv
from numba import njit
from s1_ard_core.computing.io import Path, read
from s1_ard_core.computing.logger import logger, timeit

import cop_ard_core.utils


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
        i_v = (
            (np.isfinite(c_esd["mean_shift"])) & (c_esd["mean_shift"].acquisition_date != c_esd.reference_date)
        ).values

        A = (D - M)[i_v, :]
        B = (c_esd["mean_shift"].values * c_esd["coherence"].values)[i_v]
        return (A, B)

    A, B = zip(*[create_matrix(esd.sel(stacked=(id, date))) for (id, date) in esd.stacked.values])
    # global least square problem solving #
    A, B = scipy.sparse.vstack(A), np.hstack(B)

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
                interf0,
                pow0,
                (reference0 * reference0.conj()).real.sel(acquisition_date=reference_date),
            )

            # coherence value lower than coherence_th will be ignored  #
            coherence = coherence.where(coherence >= coherence_th)

            # phase mean in the overlap area #
            interf01 = (interf0 * interf1.conj() * coherence).mean(dim=["slant_range_time", "azimuth_pixel"])
            mean_phase = xr.apply_ufunc(np.angle, interf01.compute())
            mean_phase = xr.DataArray(
                data=mean_phase,
                dims=["acquisition_date"],
                coords={"acquisition_date": interf01.acquisition_date},
                name="mean_phase",
            )

            coherence = coherence.mean(dim=["azimuth_pixel", "slant_range_time"])
        else:
            # if the reference_date is not in the burst nan shift and coherence will be returned #
            coords = burst0.acquisition_date.coords
            data = np.full(burst0.acquisition_date.size, np.nan)
            mean_phase = xr.DataArray(name="mean_phase", data=data, coords=coords)
            coherence = xr.DataArray(name="coherence", data=data, coords=coords)

        return (
            reference_date,
            mean_phase.compute(),
            coherence.rename("coherence").compute(),
        )

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
        reference, mean_phase, cohe = zip(*[compute_shift(e, f_d, dtaz) for e in reference_dates])

        wrap_cycle = tuple([xr.full_like(s, 1 / (f_d * dtaz)) for s in mean_phase])

    # obtained shifts returned as a xr.Dataset #
    cohe = xr.concat(cohe, dim="reference_date")
    mean_phase = xr.concat(mean_phase, dim="reference_date")
    wrap_cycle = xr.concat(wrap_cycle, dim="reference_date")
    mean_shift = xr.zeros_like(mean_phase)

    esd = xr.Dataset(
        {
            "reference_date": list(reference),
            "coherence": cohe,
            "mean_phase": mean_phase,
            "mean_shift": mean_shift,
            "wrap_cycle": wrap_cycle,
        }
    )

    return esd


def esd_single_burst(
    burst_id: str,
    working_directory: Path,
    burst_list: list[str],
    reference_dates: list[str],
    kernel_shape: list[int] = [3, 3],
    coherence_th: float = 0.7,
    freq: float | None = None,
    bandwidth: float | None = None,
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
        next_burst_id = f"{swath}_{int(id) + 1}"
        return next_burst_id

    def time_to_pixel(stack: xr.Dataset, dtaz: float, reference_time: datetime.datetime | None = None):
        if reference_time is None:
            reference_time = stack.azimuth_time[0].values
        azimuth_pixel = np.round((stack.azimuth_time - reference_time).astype("float64") / 1e9 / dtaz).astype("int32")
        stack = stack.assign_coords({"azimuth_pixel": azimuth_pixel})
        return stack.swap_dims(azimuth_time="azimuth_pixel"), reference_time

    # current coregistered burst extraction #
    current_burst = read(
        working_directory.append(f"bursts/{burst_id}/coreg/measurements"),
        consolidate_metadata=False,
    )

    next_burst_id = get_next_burst_id()
    if next_burst_id in burst_list:
        # next coregistered burst extraction #
        logger.info(f"Processing overlap area between bursts: {burst_id} and {next_burst_id}")
        next_burst = read(
            working_directory.append(f"bursts/{next_burst_id}/coreg/measurements"),
            consolidate_metadata=False,
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
        chunks_burst = {k: v[0] for k, v in current_burst.chunksizes.items()}

        if (freq is not None) and (bandwidth is not None):
            logger.info(f"Selecting frequency {freq}")
            logger.info(f"Selecting bandwidth {bandwidth}")

            current_burst = (
                cop_ard_core.utils.extract_subband(current_burst, bandwidth, freq)
                .compute()
                .chunk(chunks_burst)
                .assign_coords(current_burst.coords)
                .assign_attrs(current_burst.attrs)
            )
            next_burst = (
                cop_ard_core.utils.extract_subband(next_burst, bandwidth, freq)
                .compute()
                .chunk(chunks_burst)
                .assign_coords(next_burst.coords)
                .assign_attrs(next_burst.attrs)
            )

        # returns the estimated azimuth shift and the corresponding quality index #
        logger.info("Computing of azimuth shifts")

        next_reference = next_burst.sel(acquisition_date=reference_dates)
        current_reference = current_burst.sel(acquisition_date=reference_dates)
        # new dimension id and corresponding coords added to the shifts #
        coords = {k: (["id"], [current_burst[k].mean().values]) for k in ["azimuth_time", "slant_range_time"]}

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
        esd = esd.expand_dims({"id": [burst_id]}).assign_coords(coords)
    else:
        logger.info(f"ESD computation skipped: burst ({next_burst_id}) does not exist")
        esd = None

    return esd


@timeit
def estimate(
    slcs: xr.DataTree,
    working_directory: str | Path,
    reference_dates: list[str] | str,
    poly_degree: int = 0,
    freq: float | None = None,
    bandwidth: float | None = None,
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
            "freq": freq,
            "bandwidth": bandwidth,
        },
        combine=False,
        **kwargs,
    )

    # Structure re-formatting in order to solve the global problem
    esd = xr.merge(
        [e for e in esd if e is not None],
        compat="no_conflicts",
        combine_attrs="drop",
        join="outer",
    )

    args = [
        esd.sel(acquisition_date=d, reference_date=rd)
        for d in esd.acquisition_date.values
        for rd in reference_dates
        if d != rd
    ]

    esd_opt = cast(
        list,
        s1_ard_core.computing.computation.map(
            optimize_curvature,
            args=args,
            kwargs={"poly_degree": poly_degree},
            combine=False,
            **kwargs,
        ),
    )
    for e in esd_opt:
        d, rd = e.acquisition_date.values, e.reference_date.values
        before = esd["mean_phase"].loc[:, rd, d].copy(True).values
        esd["mean_phase"].loc[:, rd, d] = e.mean_phase
        esd["mean_shift"].loc[:, rd, d] = e.mean_shift
        after = esd["mean_phase"].loc[:, rd, d].values
        print(after - before)

    esd = esd.stack(stacked=("id", "reference_date")).assign_attrs(
        {key: list(slcs.bursts.children.values())[0].coreg.measurements.attrs[key]}
    )
    # Auxiliary data saved as attributes of the dataset
    return esd, find_global_shift(esd, poly_degree=poly_degree)


def optimize_curvature(esd, poly_degree: int):
    ind = np.where(np.isfinite(esd["mean_phase"].values))[0]  # * esd.id.str.startswith(swath))[0]
    y = esd["mean_phase"].values[ind]
    w = esd["wrap_cycle"].values[ind]
    az = esd["azimuth_time"].values[ind]
    t = (az - az.min()).astype(float) / 1e9
    if poly_degree != 0:
        A = np.vstack([t**i for i in range(1, poly_degree + 1)]) * 2 * np.pi / w
        scale = np.ptp(A, axis=1)
        A = 2 * A / scale[:, None] - 1
        params = np.array([np.arange(-3 * np.pi, 3 * np.pi, np.pi / 100) for i in range(poly_degree)])

        opt_param = optimize_phase(params, y, A)
        opt_param = scipy.optimize.minimize(scipy_optimize_phase, opt_param, args=(y, A), method="BFGS").x
        cost = angle_phase(opt_param, y, A)
        y_model = opt_param @ A + cost
    else:
        y_model = angle_phase(np.zeros(1), y, np.zeros(1))

    phase = cop_ard_core.utils.wrap(y - y_model) + y_model

    esd_opt = esd.copy(deep=True)
    esd_opt["mean_phase"][ind] = phase
    esd_opt["mean_shift"][:] = np.nan
    esd_opt["mean_shift"][ind] = phase / (2 * np.pi) * esd_opt["wrap_cycle"][ind]

    return esd_opt


@njit(cache=True)
def compute_phase(param, y, A):
    return np.abs(np.mean(np.exp(1j * (y - param @ A))))


@njit(cache=True)
def angle_phase(param, y, A):
    return np.angle(np.mean(np.exp(1j * (y - param @ A))))


@njit(cache=True)
def optimize_phase(params, y, A):
    n_rows = params.shape[0]
    n_cols = params.shape[1]
    total_combinations = n_cols**n_rows
    opt_res = 0
    opt_param = np.zeros(n_rows, dtype=params.dtype)
    for i in range(total_combinations):
        index = i
        param = np.zeros(n_rows, dtype=params.dtype)
        for row in range(n_rows - 1, -1, -1):
            col = index % n_cols
            param[row] = params[row, col]
            index = index // n_cols
        res = compute_phase(param, y, A)
        if res > opt_res:
            opt_res = res
            opt_param = param

    return opt_param


def scipy_optimize_phase(param, y, A) -> float:
    """
    Compute the objective function value for a given parameters x.
    This function is a wrapper to be used with Scipy optimizers.
    """
    return -compute_phase(param, y, A)
