import numpy as np
import xarray as xr


def update_recursive_reference(
    slc: xr.DataArray,
    current_reference: xr.DataArray,
    normalized_interf: xr.DataArray,
    memory_param: float,
) -> xr.DataArray:
    """
    Updates the recursive reference using the current SLC and the
    upsampled phasor from the normalized interferogram.
    """
    # 1. Extract unit phasor
    normalized_interf = normalized_interf.data
    magnitude = np.abs(normalized_interf)
    safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)
    phasor_interf = normalized_interf / safe_magnitude

    # 2. Explicitly unpack dimensions for readability
    _, coarse_az, coarse_rg = phasor_interf.shape
    n_bands, full_az, full_rg = slc.shape

    # 3. Define 5D shapes for block-wise broadcasting
    blocked_slc_shape = (n_bands, coarse_az, full_az // coarse_az, coarse_rg, full_rg // coarse_rg)
    broadcast_phasor_shape = (n_bands, coarse_az, 1, coarse_rg, 1)

    # 4. Apply phase rotation using underlying Dask/NumPy arrays
    correction = slc.data.reshape(blocked_slc_shape) * phasor_interf.reshape(broadcast_phasor_shape).conj()
    correction = np.where(np.isnan(correction), 0, correction)
    # Apply fading memory filter
    new_reference = (memory_param * current_reference) + correction.reshape(slc.shape)

    return new_reference

