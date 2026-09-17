"""
Coregistered Multi-Looked Phase (CMLP) processing
"""

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


def calculate_stable_dates(
    acquisition_dates: np.ndarray,
    stable_interval_days: int,
    stable_reference: Union[xr.DataArray, xr.Dataset, None] = None,
    starting_date: Union[str, None] = None,
) -> np.ndarray:
    """
    Determines the dates that serve as stable reference milestones based on a fixed time interval.

    Parameters
    ----------
    acquisition_dates : np.ndarray
        Array of all acquisition dates.
    stable_interval_days : int
        Number of days between each stable reference milestone (e.g., 24).
    stable_reference : Union[xr.DataArray, xr.Dataset, None], optional
        Previously calculated stable reference dataset to continue from. Default is None.
    starting_date : Union[str, None], optional
        Optional starting date to use as the baseline. If None, the minimum date
        in `acquisition_dates` is used. Default is None.

    Returns
    -------
    stable_reference_dates : np.ndarray
        Subset of dates selected as stable references.
    """
    # Convert acquisition dates to nanoseconds precision for accurate time delta math
    dates_ns = acquisition_dates.astype("datetime64[ns]")

    # Convert the interval in days directly to a nanosecond timedelta
    interval_ns = np.timedelta64(stable_interval_days, "D").astype("timedelta64[ns]")

    if stable_reference is not None and len(stable_reference.acquisition_date.values) > 0:
        # Extract past stable reference dates
        ref_dates = stable_reference.acquisition_date.values
        ref_dates_ns = ref_dates.astype("datetime64[ns]")

        # The start date is the earliest date in the stable reference history
        start_date = np.min(ref_dates_ns)
        stable_refs = list(ref_dates)

        # Find the most recent stable reference date to calculate elapsed time
        max_ref_date = np.max(ref_dates_ns)
        max_elapsed_time = max_ref_date - start_date

        # Set the threshold strictly to the NEXT expected milestone interval
        threshold = (max_elapsed_time // interval_ns + 1) * interval_ns
    else:
        # If no history exists, use starting_date or the first available date
        start_date = np.min(dates_ns) if starting_date is None else np.datetime64(starting_date, "ns")
        stable_refs = []

        # The first threshold is simply one interval ahead
        threshold = interval_ns

    # Vectorized calculation of elapsed time for all acquisition dates
    elapsed_time = dates_ns - start_date

    for i, elapsed in enumerate(elapsed_time):
        # When the elapsed time reaches or exceeds the threshold, we found a new stable date
        if elapsed >= threshold:
            stable_refs.append(acquisition_dates[i])

            # Advance the threshold relative to the *current* elapsed time
            threshold = (elapsed // interval_ns + 1) * interval_ns

    return np.array(stable_refs)


def get_current_reference(
    cstack: xr.DataArray,
    last_reference: xr.DataArray | None,
    stable_reference: xr.DataArray | None,
):
    """
    Select the most recent valid reference (last or stable) preceding the
    current stack, returning it with its acquisition date.
    """
    # 1. Parse cstack dates safely to datetime64[ns]
    dates_dt = cstack.acquisition_date.values.astype("datetime64[ns]")
    date = np.min(dates_dt)

    # 2. Find the index of the minimum date to extract the corresponding acquisition_id
    # This avoids string formatting issues when using .sel() later
    min_idx = np.argmin(dates_dt)
    min_acq_id = cstack.acquisition_id.values[min_idx]

    # 3. Extract the date for the last_reference if provided
    date_l = np.datetime64(str(last_reference.acquisition_date.values)) if last_reference is not None else None

    # 4. Extract the date and corresponding acquisition_id for stable_reference if provided
    date_s = None
    max_s_acq_id = None
    if stable_reference is not None:
        stable_dates_dt = stable_reference.acquisition_date.values.astype("datetime64[ns]")
        date_s = np.max(stable_dates_dt)

        # Get the ID of the most recent stable reference to avoid KeyError in .sel()
        max_s_idx = np.argmax(stable_dates_dt)
        max_s_acq_id = stable_reference.acquisition_id.values[max_s_idx]

    # 5. Initialize current_reference using the primary dimension (acquisition_id)
    current_reference = cstack.sel(acquisition_id=min_acq_id)

    # 6. Compare dates to determine the best reference to use
    if (date_l is not None) and (date_s is not None):
        if date_l >= date_s:
            if date_l <= date and isinstance(last_reference, xr.DataArray):
                current_reference = last_reference
            else:
                raise Exception(f"Last current reference {date_l} is more recent than cstack {date}")
        else:
            if date_s < date and isinstance(stable_reference, xr.DataArray):
                # Use acquisition_id instead of string date for safe slicing
                current_reference = stable_reference.sel(acquisition_id=max_s_acq_id)
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
                # Use acquisition_id instead of string date for safe slicing
                current_reference = stable_reference.sel(acquisition_id=max_s_acq_id)
            else:
                raise Exception(f"Last stable reference {date_s} is more recent than cstack {date}")

    # 7. Extract the date of the elected current_reference
    dates = current_reference.acquisition_date.values

    # 8. If this is the very first reference ever created, tag the starting date
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
    """
    Fill empty (NaN) values in the drift interferogram with coherent normalized
    interference values, returning the coherence array.
    """
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
