import numpy as np
import xarray as xr

from cop_ard_core.logger import timeit
from cop_ard_core.warp import sinc_warp


def get_radar_spacing(azimuth_time: xr.DataArray, slant_range_time: xr.DataArray):

    az_diffs = np.diff(azimuth_time)
    sr_diffs = np.diff(slant_range_time)

    unique_az, counts_az = np.unique(az_diffs, return_counts=True)
    unique_sr, counts_sr = np.unique(sr_diffs, return_counts=True)

    az_step, rg_step = unique_az[np.argmax(counts_az)], unique_sr[np.argmax(counts_sr)]

    return az_step, rg_step


@timeit
def warp_cmlp(
    target: xr.DataArray,
    target_indexes: np.ndarray,
    source: xr.DataArray,
    x: np.ndarray,
    y: np.ndarray,
    kernel: dict,
    is_phase: bool,
    is_coherence: bool,
) -> None:

    # Create memory-mapped VIEWS
    spatial_size = target.shape[-2] * target.shape[-1]
    # target_view is now (N_leading, Y * X)
    target_view = target.reshape(-1, spatial_size)
    # source_view is (N_leading, Y, X)
    source_view = source.reshape(-1, source.shape[-2], source.shape[-1])
    # Loop and insert
    for i in range(target_view.shape[0]):
        inp = source_view[i]

        # 2. Apply the appropriate warping logic
        if is_phase:
            # Wrap phase to complex phasor to avoid interpolation artifacts
            complex_phasor = np.exp(1j * inp)
            out = np.angle(sinc_warp(complex_phasor, x, y, kernel))
        else:
            # Standard warp for amplitude/coherence/other
            out = sinc_warp(inp, x, y, kernel)

            if is_coherence:
                # np.clip instantly bounds the array between 0 and 1,
                out = np.clip(out, 0.0, 1.0)
        # Write back into the target view
        target_view[i, target_indexes] = out
