import itertools

import numpy as np
import scipy.sparse
import xarray as xr
from numba import njit

from cop_ard_core.utils import compute_interf_norm, wrap


def compute_shift(
    burst0: xr.DataArray,
    burst1: xr.DataArray,
    reference0: xr.DataArray,
    reference1: xr.DataArray,
    acquisition_id: xr.DataArray,
    multilook: dict,
    coherence_threshold: float,
) -> tuple[xr.DataArray, xr.DataArray]:

    # output initialization #
    coords = acquisition_id.coords
    data = np.full(acquisition_id.size, np.nan)
    mean_phase = xr.DataArray(name="mean_phase", data=data, coords=coords)
    coherence = xr.DataArray(name="coherence", data=data, coords=coords)

    # interferogram multilooked and normalized #
    interf0 = compute_interf_norm(reference0.squeeze(), burst0, multilook)
    interf1 = compute_interf_norm(reference1.squeeze(), burst1, multilook)

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
        dims=["acquisition_id"],
        coords={"acquisition_id": interf_prod.acquisition_id},
        name="mean_phase",
    )
    mean_phase = mean_phase.where(coherence > 0)

    return (
        mean_phase,
        coherence.rename("coherence"),
    )


def compute_azimuth_shifts(
    burst0: xr.DataArray,
    burst1: xr.DataArray,
    reference0: xr.DataArray,
    reference1: xr.DataArray,
    multilook: dict = {"azimuth_pixel": 4, "slant_range_time": 16},
    coherence_threshold: float = 0.7,
):

    # Computation of doppler frequency at the burst overlap #
    annot0 = burst0.attrs[list(burst0.attrs)[0]]
    coeffs0 = annot0["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]
    dazm0 = annot0["Grid"]["Col"]["SS"]
    annot1 = burst1.attrs[list(burst1.attrs)[0]]
    coeffs1 = annot1["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"]
    dazm1 = annot0["Grid"]["Col"]["SS"]

    # average azimuth spacing in correspondence of burst overlap
    dazm = (dazm1 + dazm0) / 2.0

    # compute common area
    sm0, am0 = np.meshgrid(burst0.slant_range_m, burst0.azimuth_m)
    sm1, am1 = np.meshgrid(burst1.slant_range_m, burst1.azimuth_m)

    # doppler centroid in the area of interest
    f_d_0 = np.polynomial.polynomial.polyval2d(sm0, am0, coeffs0)
    f_d_1 = np.polynomial.polynomial.polyval2d(sm1, am1, coeffs1)

    with np.errstate(
        invalid="ignore",
        over="ignore",
    ):
        # shifts estimation in correspondence of the overlapping area
        mean_phase, coherence = compute_shift(
            burst0=burst0,
            burst1=burst1,
            reference0=reference0,
            reference1=reference1,
            acquisition_id=burst0.acquisition_id,
            multilook=multilook,
            coherence_threshold=coherence_threshold,
        )
        # compute scaling factor according to Doppler frequency difference
        wrap_cycle = xr.full_like(mean_phase, 1 / ((f_d_1 - f_d_0).mean() * dazm))

    # init mean_shift with all zeros
    # the estimated phase has to be unwrapped before rescaling it and provide the mean_shift
    mean_shift = xr.zeros_like(mean_phase)
    # obtained shifts returned as a xr.Dataset #
    esd = xr.Dataset(
        {
            "coherence": coherence,
            "mean_phase": mean_phase,
            "mean_shift": mean_shift,
            "wrap_cycle": wrap_cycle,
        }
    )

    return esd


def optimize_curvature(esd, poly_degree: int):
    # extract valid indices
    ind = np.where(np.isfinite(esd["mean_phase"].values))[0]

    # Check if there are valid points to process
    if len(ind) == 0:
        return esd.copy(deep=True)

    y = esd["mean_phase"].values[ind]
    w = esd["wrap_cycle"].values[ind]
    az = esd["azimuth_time"].values[ind]
    t = (az - az.min()).astype(float) / 1e9
    # fit a polynomial function on the measured valid phases
    if poly_degree == 0:
        opt_param = np.array([0.0])
        A = np.ones((1, t.size), dtype=np.float64)
    else:
        A = np.vstack([t**i for i in range(1, poly_degree + 1)]) * 2 * np.pi / w
        k = np.ptp(A, axis=1)
        k[k == 0] = 1.0
        A = A / k[:, np.newaxis]
        params = [np.arange(-k[i], k[i], 2 * np.pi) for i in range(poly_degree)]
        # Assuming optimize_phase, compute_phase, angle_phase, wrap are custom functions defined elsewhere
        opt_param = optimize_phase(params, y, A)
        opt_param = scipy.optimize.minimize(compute_phase, opt_param, args=(y, A), method="BFGS").x
    const = angle_phase(opt_param, y, A)
    y_model = opt_param @ A + const
    # unwrap (1D) the phase
    phase = wrap(y - y_model) + y_model
    # convert phase offset to shift in pixel
    esd_opt = esd.copy(deep=True)
    esd_opt["mean_phase"].data[ind] = phase
    esd_opt["mean_shift"] = esd_opt["mean_phase"] / (2 * np.pi) * esd_opt["wrap_cycle"]

    return esd_opt


@njit(cache=True)
def compute_phase(param, y, A):
    return -np.abs(np.sum(np.exp(1j * (y - param @ A))))


@njit(cache=True)
def angle_phase(param, y, A):
    return np.angle(np.sum(np.exp(1j * (y - param @ A))))


def optimize_phase(params, y, A, maxcorr: float = 3 * np.pi):
    opt_param = np.array([p[0] for p in params], dtype=float)
    opt_res = compute_phase(opt_param, y, A)
    for param_tuple in itertools.product(*params):
        param = np.array(param_tuple)
        corr = np.max(np.abs(param @ A))
        if corr > maxcorr:
            continue

        res = compute_phase(param, y, A)
        if res < opt_res:
            opt_res = res
            opt_param = param

    return opt_param


def scipy_optimize_phase(param, y, A) -> float:
    """
    Compute the objective function value for a given parameters x.
    This function is a wrapper to be used with Scipy optimizers.
    """
    return -compute_phase(param, y, A)
