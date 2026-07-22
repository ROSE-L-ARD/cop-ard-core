import numpy as np
import scipy.sparse
import xarray as xr
from numba import njit

from cop_ard_core.utils import compute_interf_norm, wrap


def _to_dataset(X, acquisition_date, poly_degree, t0, r0, s_y, s_x):
    def to_polygrid2d(c):
        m = np.zeros((poly_degree + 1, poly_degree + 1), dtype="float64")
        m[:, 0] = np.flip(c)
        return m

    axis = np.arange(poly_degree + 1)
    data_vars = {
        str(dd): xr.DataArray(
            data=to_polygrid2d(X[i, :]),
            dims=["i", "j"],
            coords={"i": axis, "j": axis},
            attrs={"deg": poly_degree, "offset": (t0, r0), "scale_fact": (s_y, s_x)},
        )
        for i, dd in enumerate(acquisition_date.values)
    }
    return xr.Dataset(data_vars)


def compute_shift(
    burst0: xr.DataArray,
    burst1: xr.DataArray,
    reference0: xr.DataArray,
    reference1: xr.DataArray,
    acquisition_date: xr.DataArray,
    reference_date: str,
    multilook: dict,
    coherence_threshold: float,
) -> tuple[str, xr.DataArray, xr.DataArray]:

    coords = acquisition_date.coords
    data = np.full(acquisition_date.size, np.nan)
    mean_phase = xr.DataArray(name="mean_phase", data=data, coords=coords)
    coherence = xr.DataArray(name="coherence", data=data, coords=coords)

    if all(reference_date in e.acquisition_date for e in [reference0, reference1]):
        # interferogram multilooked and normalized #
        interf0 = compute_interf_norm(reference0.sel(acquisition_date=reference_date).squeeze(), burst0, multilook)
        interf1 = compute_interf_norm(reference1.sel(acquisition_date=reference_date).squeeze(), burst1, multilook)

        # coherence computation #
        interf_prod = interf1 * interf0.conj()
        coherence_ = abs(interf0)  # based only on one burst

        # mask interferogram
        mask = coherence_ >= coherence_threshold
        n_valid_points = np.count_nonzero(mask)

        interf_prod = (interf_prod).where(mask)
        coherence = coherence_.where(mask, 0).sum(dim=["azimuth_pixel", "slant_range_time"]) / np.sqrt(n_valid_points)

        # phase mean in the overlap area #
        mean_phase = xr.ufuncs.angle(interf_prod.sum(dim=["slant_range_time", "azimuth_pixel"]))
        mean_phase = xr.DataArray(
            data=mean_phase,
            dims=["acquisition_date"],
            coords={"acquisition_date": interf_prod.acquisition_date},
            name="mean_phase",
        )
        mean_phase = mean_phase.where(coherence > 0)
    return (
        reference_date,
        mean_phase,
        coherence.rename("coherence"),
    )


def compute_azimuth_shifts(
    burst0: xr.DataArray,
    burst1: xr.DataArray,
    reference0: xr.DataArray,
    reference1: xr.DataArray,
    reference_dates: list[str],
    multilook: dict = {"azimuth_pixel": 4, "slant_range_time": 16},
    coherence_th: float = 0.7,
):

    # @timeit

    # Computation of doppler frequency at the burst overlap #
    annot0 = burst0.attrs[list(burst0.attrs)[0]]
    coeffs0 = annot0["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]
    dazm0 = annot0["Grid"]["Col"]["SS"]
    annot1 = burst1.attrs[list(burst1.attrs)[0]]
    coeffs1 = annot1["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]
    dazm1 = annot0["Grid"]["Col"]["SS"]
    dazm = (dazm1 + dazm0) / 2.0

    # compute common area
    sm0, am0 = np.meshgrid(burst0.slant_range_m, burst0.azimuth_m)
    sm1, am1 = np.meshgrid(burst1.slant_range_m, burst1.azimuth_m)
    f_d_0 = np.polynomial.polynomial.polyval2d(sm0, am0, coeffs0)
    f_d_1 = np.polynomial.polynomial.polyval2d(sm1, am1, coeffs1)
    with np.errstate(
        invalid="ignore",
        over="ignore",
    ):
        # shifts estimation for each reference_date that is also a date of burst0
        reference, mean_phase, cohe = zip(
            *[
                compute_shift(
                    burst0=burst0,
                    burst1=burst1,
                    reference0=reference0,
                    reference1=reference1,
                    acquisition_date=burst0.acquisition_date,
                    reference_date=e,
                    multilook=multilook,
                    coherence_threshold=coherence_th,
                )
                for e in reference_dates
            ]
        )
        # wrap_cycle = tuple([1 / ((f_d_1 - f_d_0) * dazm) for s in mean_phase])
        wrap_cycle = tuple([xr.full_like(s, 1 / ((f_d_1 - f_d_0).mean() * dazm)) for s in mean_phase])

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

    X = _to_dataset(
        X - X[i_mc, :],
        acquisition_date,
        poly_degree,
        t0.values,
        r0.values,
        np.round(s_y.values).astype("timedelta64[ns]"),
        s_x.values if s_x.values > 0 else np.array(1.0),
    )

    return X.assign_attrs({"type": "times"})


def optimize_curvature(esd, poly_degree: int):
    # extract valid indices
    ind = np.where(np.isfinite(esd["mean_phase"].values))[0]  # * esd.id.str.startswith(swath))[0]
    y = esd["mean_phase"].values[ind]
    w = esd["wrap_cycle"].values[ind]
    az = esd["azimuth_time"].values[ind]

    t = (az - az.min()).astype(float) / 1e9
    # Rescale azimuth time to have values from -1 to 1
    t = (t / np.ptp(t)) * 2 - 1.0

    # fit a polynomial function on the measured valid phases
    A = np.vstack([t**i for i in range(0, poly_degree + 1)]) * 2 * np.pi / w
    params = np.array([np.arange(-1, 1, np.mean(w) / 4) for i in range(poly_degree + 1)])

    opt_param = optimize_phase(params, y, A)
    opt_param = scipy.optimize.minimize(compute_phase, opt_param, args=(y, A), method="BFGS").x
    const = angle_phase(opt_param, y, A)
    y_model = opt_param @ A + const

    # unwrap (1D) the phase
    phase = wrap(y - y_model) + y_model

    # convert phase offset to shift in pixel
    esd_opt = esd.copy(deep=True)
    esd_opt["mean_phase"][ind] = phase
    esd_opt["mean_shift"] = esd_opt["mean_phase"] / (2 * np.pi) * esd_opt["wrap_cycle"]

    return esd_opt


@njit(cache=True)
def compute_phase(param, y, A):
    return -np.abs(np.sum(np.exp(1j * (y - param @ A))))


@njit(cache=True)
def angle_phase(param, y, A):
    return np.angle(np.sum(np.exp(1j * (y - param @ A))))


def optimize_phase(params, y, A, maxcorr: float = 3 * np.pi):
    n_rows = params.shape[0]
    n_cols = params.shape[1]
    total_combinations = n_cols**n_rows
    opt_res = 1e9
    opt_param = np.zeros(n_rows, dtype=params.dtype)
    for i in range(total_combinations):
        index = i
        param = np.zeros(n_rows, dtype=params.dtype)
        for row in range(n_rows - 1, -1, -1):
            col = index % n_cols
            param[row] = params[row, col]
            index = index // n_cols
        # try only parameters providing correction lower than a given threshold
        corr = np.max(np.abs(param @ A))
        if corr > maxcorr:
            continue
        res = compute_phase(param, y, A)
        if res < opt_res:
            opt_res = res
            opt_param = param
            print(f"check param {i} of {total_combinations} {param=} {corr=} {res=}")

    return opt_param


def scipy_optimize_phase(param, y, A) -> float:
    """
    Compute the objective function value for a given parameters x.
    This function is a wrapper to be used with Scipy optimizers.
    """
    return -compute_phase(param, y, A)
