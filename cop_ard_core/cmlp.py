from typing import Union
import numpy as np
import xarray as xr

from cop_ard_core.utils import get_normalized_interf


def update_recursive_reference(
    slc: xr.DataArray,
    current_reference: xr.DataArray,
    normalized_interf: np.ndarray,
    memory_param: float,
) -> xr.DataArray:
    """
    Updates the recursive reference using the current SLC and the
    upsampled phasor from the normalized interferogram.
    """
    # 1. Extract unit phasor
    magnitude = np.abs(normalized_interf)
    safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)
    phasor_interf = normalized_interf / safe_magnitude

    # 2. Extract shapes
    _, coarse_az, coarse_rg = phasor_interf.shape
    n_bands, full_az, full_rg = slc.shape
    
    # Calculate posting (block size) dynamically
    block_az = full_az // coarse_az
    block_rg = full_rg // coarse_rg
    
    valid_az = coarse_az * block_az
    valid_rg = coarse_rg * block_rg
    
    # Get just the valid data slice as a numpy array for the calculation
    # Using [:, ...] handles the polarization dimension automatically
    slc_valid_data = slc.data[:, :valid_az, :valid_rg]

    # 3. Define shapes for block-wise broadcasting
    blocked_slc_shape = (n_bands, coarse_az, block_az, coarse_rg, block_rg)
    broadcast_phasor_shape = (n_bands, coarse_az, 1, coarse_rg, 1)

    # 4. Apply phase rotation
    correction = slc_valid_data.reshape(blocked_slc_shape) * phasor_interf.reshape(broadcast_phasor_shape).conj()
    correction = np.where(np.isnan(correction), 0, correction)
    
    # 5. Build the full-sized new reference
    # Step A: Apply fading memory to the ENTIRE original grid (including margins)
    new_ref_data = current_reference.data * memory_param
    
    # Step B: Add the computed correction ONLY to the valid spatial bounds
    correction_reshaped = correction.reshape((n_bands, valid_az, valid_rg))
    new_ref_data[:, :valid_az, :valid_rg] += correction_reshaped

    # Return as an xr.DataArray safely holding the full original dimensions
    return current_reference.copy(data=new_ref_data)


import numpy as np
import xarray as xr
from typing import Union

def calculate_stable_dates(
    acquisition_dates: np.ndarray,
    revisit_cycle: np.timedelta64,
    stable_reference_step: int,
    stable_reference: Union[xr.DataArray, xr.Dataset, None] = None,
    starting_date: Union[str, None] = None,
) -> np.ndarray:
    """
    Determines the dates that serve as stable reference milestones.

    Parameters
    ----------
    acquisition_dates : np.ndarray
        Array of all acquisition dates.
    revisit_cycle : np.timedelta64
        Time delta representing one revisit cycle.
    stable_reference_step : int
        Number of cycles between stable references.

    Returns
    -------
    stable_reference_dates : np.ndarray
        Subset of dates selected as stable references.
    """
    dates_ns = acquisition_dates.astype("datetime64[ns]")
    
    if stable_reference is not None and len(stable_reference.acquisition_date.values) > 0:
        ref_dates = stable_reference.acquisition_date.values
        ref_dates_ns = ref_dates.astype("datetime64[ns]")
        
        start_date = np.min(ref_dates_ns)
        stable_refs = list(ref_dates)

        max_ref_date = np.max(ref_dates_ns)
        max_ref_cycle = (max_ref_date - start_date) / revisit_cycle.astype("timedelta64[ns]")
        
        # Set threshold to the strictly NEXT expected milestone
        threshold = (int(max_ref_cycle // stable_reference_step) + 1) * stable_reference_step
    else:
        start_date = np.min(dates_ns) if starting_date is None else np.datetime64(starting_date, "ns")
        stable_refs = []
        threshold = stable_reference_step

    # Vectorized calculation of elapsed cycles
    elapsed_cycles = (dates_ns - start_date) / revisit_cycle.astype("timedelta64[ns]")
    
    for i, cycle in enumerate(elapsed_cycles):
        if cycle >= threshold:
            stable_refs.append(acquisition_dates[i])
            # Advance threshold relative to the *current* cycle
            threshold = (int(cycle // stable_reference_step) + 1) * stable_reference_step
            
    return np.array(stable_refs)

def get_current_reference(
    cstack: xr.DataArray,
    last_reference: xr.DataArray | None,
    stable_reference: xr.DataArray | None,
):
    date = np.min(cstack.acquisition_date.values.astype("datetime64"))
    date_l = np.datetime64(str(last_reference.acquisition_date.values)) if last_reference is not None else None
    date_s = (
        np.max(stable_reference.acquisition_date.values.astype("datetime64")) if stable_reference is not None else None
    )
    current_reference = cstack.sel(acquisition_date=date.astype(str))

    if (date_l is not None) and (date_s is not None):
        if date_l >= date_s:
            if date_l <= date and isinstance(last_reference, xr.DataArray):
                current_reference = last_reference
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        else:
            if date_s < date and isinstance(stable_reference, xr.DataArray):
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")
    else:
        if date_l is not None and isinstance(last_reference, xr.DataArray):
            if date_l < date:
                current_reference = last_reference
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        if date_s is not None and isinstance(stable_reference, xr.DataArray):
            if date_s < date:
                current_reference = stable_reference.sel(acquisition_date=date_s.astype(str))
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")

    dates = current_reference.acquisition_date.values

    if last_reference is None:
        current_reference.attrs["starting_date"] = str(date)
    return current_reference, dates


def update_drift(
    drift_interf: xr.DataArray,
    reference: xr.DataArray,
    stable_reference: xr.DataArray,
    posting: dict[str, int],
    coherence_threshold: float,
):
    # Calculate interference between history and current
    normalized_vals = get_normalized_interf(reference, stable_reference, posting).values
    coherence = np.abs(normalized_vals)
    is_empty = np.isnan(drift_interf)
    # Check if we are fully filled
    if is_empty.any():
        # Create Masks
        is_coherent = coherence >= coherence_threshold
        mask = is_coherent * is_empty
        # Fill holes
        drift_interf[mask] = normalized_vals[mask]
    return coherence
