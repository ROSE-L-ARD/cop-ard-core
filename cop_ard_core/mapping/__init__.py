import json
from collections import defaultdict
import re
from datetime import datetime
from cop_ard_core.computing.orbit import read_orbit_file, overwrite_coeffs, update_slcs
import dateutil
from cop_ard_core.computing.simul import get_radar_bbox, get_geo_from_radar_bbox, extract_simul_on_radar_chunk, single_simulation_ml, init_simulation, update_elevation, compute_simulation, SPATIAL_DIMS_
from cop_ard_core.computing.etad import discover_etaddate_slcsdate, update_etad
from pyproj import CRS
import os
from cop_ard_core.computing.warp import warp_cslc, warp_dazdrg, sinc, warp_gslc
from cop_ard_core.computing.coreg import update_quality, update_conditions, update_LUT, initialize_LUT_coords
from typing import cast, Dict, Any, Mapping, Union, Sequence
import botocore.exceptions
import dask.array as da
import numpy as np
import xarray as xr
from scipy.constants import speed_of_light
from unittest import mock
import sarpy.io
import sarpy.io.general.base

from cop_ard_core.computing.area_projection import initialize_simulation, compute_chunks, sparse_to_dense, compute_rasterization_and_rtc_ap, compute_shadow, compute_geocoded_mask
from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.dem import GeoEnvPlugin, get_resolution_in_meters, get_default_epsg, check_available_transformation, extract_vertical_crs, compute_shape_and_off, interp_dem
from cop_ard_core.computing.griddata import init_interpolate
from cop_ard_core.computing.io import touch, get_storage_options, normpath, exists, delete, write, makedirs, save_empty_dataset, glob, read, Path, is_s3_path, path_to_dict, generate_empty_dataset
from cop_ard_core.computing.logger import logger, timeit
from cop_ard_core.computing.slc import get_margin, get_footprint, intersect, get_burst_id, get_burst_id_safe, get_relative_orbit_number, map_index_to_group, merge_dict, get_az_rg_meters, get_az_rg_times, get_coefficients, noise_removal, amplitude_calibration, image_deramp, apply_invalid_mask, extract_acquisition_dates, SentinelSarpyObj


def init_single_coreg_stack(
    slcs: xr.DataTree,
    ref_date: str,
    wdir_path: str,
    recompute: bool = False,
    **kwargs,
) -> xr.DataTree:
    """
    Initialize the empty 'coreg' group for a single burst or stripmap scene.

    This function creates the structure for the coregistered stack. It uses the
    dimensions and coordinates of the reference date image and creates empty
    Dask arrays (`cstack`, `phase_model`) that will be filled in during processing.

    Args:
        slcs: The input DataTree containing the 'orig' data.
        ref_date: The reference date for coregistration.
        store_path: The path where the 'coreg' group should be created.

    Returns:
        The newly created 'coreg' DataTree node, read back from disk.
    """
    if wdir_path is None:
        raise Exception("Output SLCs path is required.")
    if (ref_date in slcs.orig.keys()) is False:
        raise Exception("Reference image not available.")

    # --- Write the Empty Structure to Disk ---
    crg = xr.DataTree.from_dict({k: xr.Dataset() for k in ["conditions", "quality"]})
    acquisition_dates = list(slcs.orig.keys())
    if (recompute is False) and len(acquisition_dates) != 1:
        acquisition_dates = [d for d in acquisition_dates if d != ref_date]
    # 1. Gather polarizations directly as sets
    pol_sets = [set(str(p) for p in slcs.orig[d].measurements.polarization.values) for d in acquisition_dates]
    # 2. Find the total unique polarizations across all dates
    all_pols = set.union(*pol_sets) if pol_sets else set()

    # 3. Check that every date has the complete set of polarizations
    for d, p_set in zip(acquisition_dates, pol_sets):
        if p_set != all_pols:
            raise ValueError(f"For acquisition date {d} got polarizations {p_set}, expected {all_pols}")

    slc = slcs.orig[ref_date].measurements.slc
    slc_sec = slcs.orig[acquisition_dates[0]].measurements.slc
    shape = [len(all_pols), len(acquisition_dates), slc.azimuth_time.size, slc.slant_range_time.size]
    measurements_path = os.path.join(wdir_path, "coreg", "measurements")
    data_vars = ["cstack", "phase_model"]
    dims = {
        "cstack": ["polarization", "acquisition_date", "azimuth_time", "slant_range_time"],
        "phase_model": ["acquisition_date", "azimuth_time", "slant_range_time"],
    }
    shape_dict = {"cstack": shape, "phase_model": shape[1:]}
    dtype = {"cstack": "complex64", "phase_model": "float64"}
    chunks = {k: (slc.chunksizes[k][0] if k in slc.dims else v[0]) for k, v in slc_sec.chunksizes.items()}

    coords = {
        "polarization": ("polarization", list(all_pols)),
        "acquisition_date": ("acquisition_date", acquisition_dates),
        "azimuth_time": ("azimuth_time", slc.azimuth_time.values),
        "slant_range_time": ("slant_range_time", slc.slant_range_time.values),
        "azimuth_m": ("azimuth_time", slc.azimuth_m.values),
        "slant_range_m": ("slant_range_time", slc.slant_range_m.values),
    }

    measurements = save_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape_dict,
        chunks=chunks,
        dtype=dtype,
        data_path=measurements_path,
        coords=coords,
        attrs={"coregistration_reference_date": ref_date},
        attrs_data_vars={"cstack": slc.attrs},
        skip_read=False,
        compute=True,
        **kwargs,
    )
    crg["measurements"] = measurements

    data_vars = ["daz", "drg"]
    dims = {
        "daz": ["acquisition_date", "azimuth_time", "slant_range_time"],
        "drg": ["acquisition_date", "azimuth_time", "slant_range_time"],
    }
    shape_dict = {"daz": shape[1:], "drg": shape[1:]}
    dtype = {"daz": "timedelta64[ns]", "drg": "float64"}

    daz_drg = save_empty_dataset(
        data_vars=data_vars,
        dims=dims,
        shape=shape_dict,
        chunks=chunks,
        dtype=dtype,
        data_path=f"{wdir_path}/daz_drg",
        coords=coords,
        skip_read=False,
        compute=True,
        **kwargs,
    )

    recompute_daz_drg = True

    if daz_drg is not None:
        crg["measurements/daz"] = daz_drg["daz"]
        crg["measurements/drg"] = daz_drg["drg"]
        crg["measurements/daz"].attrs["recompute"] = recompute_daz_drg
        crg["measurements/daz"].attrs["store_path"] = f"{wdir_path}/daz_drg"

    if ref_date not in acquisition_dates:
        logger.info(f"Removing reference date for burst : {slcs.name}")
        del slcs.orig[ref_date]
    # Add the new 'coreg' node to the in-memory DataTree.
    slcs["coreg"] = crg

    if not isinstance(slcs, xr.DataTree):
        raise Exception(f"Expected xr.Datatree got {type(slcs)}")
    return slcs


def sarpy_open(in_path: str | xr.DataTree, allowed_partial_open: bool = False, storage_options: dict = {}):
    """
    A robust wrapper to open various SAR data formats using the Sarpy library.

    This function attempts to open SAR data, handling different input types like
    local file paths, S3 paths, and in-memory xarray.DataTree objects. It includes
    fallbacks for S3 access and a special handler for partially reading Sentinel-1
    metadata without downloading the full dataset.

    Args:
        in_path: The path or object to open.
        allowed_partial_open: If True, allows a special handling mode for
        Sentinel-1 on S3 to read only metadata.

    Returns:
        A Sarpy reader object.
    """
    ds = None
    try:
        # Handle in-memory DataTree by computing it first.
        if isinstance(in_path, xr.DataTree):
            with mock.patch("sarpy.io.complex.converter.is_file_like", lambda _: True):
                ds = sarpy.io.open(in_path)  # type: ignore
        # Handle standard file paths.
        else:
            ds = sarpy.io.open(in_path)
    except sarpy.io.general.base.SarpyIOError:
        # If standard open fails, it might be an S3 path that Sarpy can't handle directly.
        if not isinstance(in_path, str):
            raise ValueError(f"Expected str, got {type(in_path)}")
        try:
            # Try opening the path with fsspec first.
            ds = sarpy.io.open(open(in_path))  # type: ignore
        except Exception:
            # As a last resort, use a custom handler for partial reads of Sentinel-1 on S3.
            if allowed_partial_open:
                if is_s3_path(in_path) and (in_path.endswith(".SAFE")):
                    # This custom object downloads just the metadata files.
                    ds = SentinelSarpyObj(in_path, storage_options)
                else:
                    raise
            else:
                raise
    return ds


def init_zarr_groups(out_path, groups, multiburst, **kwargs):
    """
    Initialize the necessary Zarr groups on disk before writing data.
    """

    # Create and save an empty dataset in each group path to initialize it.
    def write_dt(dt_group):
        dt = xr.DataTree.from_dict({kk: xr.Dataset() for kk in ["measurements"]})
        write(
            dt,
            os.path.join(out_path, dt_group),
            mode="w",
            storage_options=get_storage_options(kwargs),
        )

    map(
        write_dt,
        args=list(groups),
    )


def read_slc_static(burst_id: str, acquisition_dates: list[str], working_directory: str, **kwargs):
    """
    Check the static directory for a reference SLC.

    This function attempts to read a pre-processed reference SLC for a specific
    burst from the static working directory and validates its presence in the
    current DataTree.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst.
    slc : xr.DataTree
        The DataTree for the specific burst which contains the reference
        SLC in its `orig` group.
    working_directory : str
        The root path of the static working directory.
    **kwargs : dict
        Additional keyword arguments for storage options.

    Returns
    -------
    burst_id : str
        The original identifier of the burst.
    reference_date : str or None
        The extracted reference date as a string, or None if the reference
        data is missing.
    ds : int or xr.Dataset
        Returns `0` if the reference SLC is missing. Returns `1` if it is present
        and already loaded in `slc.orig`. Otherwise, returns the loaded
        `xr.Datatree` itself.
    """
    storage_opts = get_storage_options(kwargs)
    slc_path = f"{working_directory}/processed_data/{burst_id}/slc.ok"
    reference_date = None

    if not exists(slc_path, storage_options=storage_opts):
        logger.info(f"Missing reference SLC marker for burst {burst_id}.")
        ds: Union[int, xr.DataTree] = 0
    else:
        ds = cast(
            xr.DataTree,
            read(
                f"{working_directory}/bursts/{burst_id}/reference",
                storage_options=storage_opts,
            ),
        )

        # Extract date (casting to string to ensure matching with slc.orig keys)
        reference_date = str(ds.measurements.acquisition_date.values[0])
        logger.info(f"Extracted reference_date: {reference_date} for burst {burst_id}.")

        if reference_date in acquisition_dates:
            logger.info(f"Reference date {reference_date} already in slc.orig.")
            ds = 1

    return (burst_id, reference_date, ds)


def save_slc_static(burst_id: str, slc: xr.DataTree, working_directory: str, **kwargs):
    """
    Save a single reference SLC DataTree to the static working directory.

    This function is intended to be used with a map operation (like the one
    in `finalize_import`) to save the reference SLC for a specific burst. It
    also writes an empty `slc.ok` marker file to indicate success.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst.
    slc : xr.DataTree
        The DataTree for the specific burst which contains the
        reference SLC in its `orig` group.
    working_directory : str
        The root path of the static working directory. The function will
        construct the full output path from this.
    **kwargs : dict
        Additional keyword arguments, primarily for passing
        `storage_options` to the `write` and `io` functions.

    Returns
    -------
    None
    """
    # Select the specific reference date DataArray from the burst's 'orig' group
    slc_path = f"{working_directory}/processed_data/{burst_id}/slc.ok"
    # Write the DataArray to the static working directory in overwrite mode ('w')
    write(
        slc,
        f"{working_directory}/bursts/{burst_id}/reference",
        mode="w",
        skip_read=True,  # Optimization: don't read data back after writing
        storage_options=get_storage_options(kwargs),
    )
    # No explicit return value is needed
    makedirs(
        normpath(f"{working_directory}/processed_data/{burst_id}"),
        exist_ok=True,
        **kwargs,
    )
    touch(slc_path, **kwargs)
    return


@timeit
def finalize_import(
    slcs: xr.DataTree,
    working_directory: str,
    recompute: bool = False,
    reference_date: str | None = None,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[xr.DataTree, str]:
    """
    Finalize the import process by ensuring the reference date SLC is
    available in the DataTree.

    This function handles different scenarios based on whether the data is
    organized by bursts and whether the user wants to recompute (overwrite)
    the static reference data in the `working_directory`.

    Scenarios handled:

    1.  **Bursts, No Recompute, Ref Date Missing:** Tries to load the reference
        date data from the `working_directory`.
    2.  **Bursts, Recompute, Ref Date Present:** The reference date data from
        the input `slcs` stack is saved (overwritten) to the `working_directory`.
    3.  **Bursts, Ref Date Partially Present:** Remove bursts with no reference
        date in an inconsistent state.
    4.  **No Bursts, Ref Date Missing:** Raises an error.

    Parameters
    ----------

    slcs : xr.DataTree
        The DataTree containing the stack of SLCs.
    working_directory : str
        The path to the static working directory where reference data
        is stored or will be saved.
    recompute : bool, optional
        If True, indicates that the static reference data in the
        `working_directory` should be overwritten if the reference date
        is present in the `slcs` input. Defaults to False.
    reference_date : str or None, optional
        The date string (e.g., 'YYYY-MM-DD') of the reference (master) SLC.
        Defaults to None.
    enable_dask : bool, optional
        Whether to enable parallel computation via Dask. Defaults to True.
    **kwargs : dict
        Additional keyword arguments, often including `storage_options`
        passed to read/write functions.

    Returns
    -------
    tuple[xr.DataTree, str]
        A tuple containing:

        - The `slcs` DataTree, potentially modified to include the reference
          date data loaded from the `working_directory`.
        - The reference date string.

    Raises
    ------
    ValueError
        If the reference image cannot be found/loaded, if there are multiple
        acquisition dates when one is expected, or if `reference_date` is missing.
    TypeError
        If the output of the static loading function or the reference date
        is of an unexpected type.
    """
    multiburst = "bursts" in slcs

    # Check if the DataTree is organized by bursts and needs a reference date
    if multiburst and reference_date is None:
        acquisition_dates = extract_acquisition_dates(slcs, multiburst=True)

        if recompute:
            if len(acquisition_dates) != 1:
                raise ValueError(f"Expected exactly one acquisition date, got {len(acquisition_dates)}")

            reference_date = acquisition_dates[0]
            logger.info("Saving reference SLC structure to static working directory.")

            # Unzip dictionary items into two lists for mapping: (burst_ids, slc_objects)
            args = [(burst_id, slc.orig[reference_date]) for burst_id, slc in slcs.bursts.items()]
            args = list(zip(*args))
            # Use a parallel map to save each burst's reference SLC
            map(
                save_slc_static,
                args=args,
                kwargs=dict(working_directory=working_directory, **kwargs),
                enable_dask=enable_dask,
            )
        else:
            args = [(burst_id, list(slc.orig)) for burst_id, slc in slcs.bursts.items()]
            args = list(zip(*args))
            static_slcs = map(
                read_slc_static,
                args=args,
                kwargs=dict(working_directory=working_directory, **kwargs),
                enable_dask=enable_dask,
            )
            if not isinstance(static_slcs, list):
                raise TypeError(f"Expected a list from computation.map, got {type(static_slcs).__name__}")

            for burst_id, ref_date, status_or_data in static_slcs:
                if ref_date is not None:
                    reference_date = ref_date
                if isinstance(status_or_data, int):
                    if status_or_data == 0:
                        logger.warning(f"No reference image found for burst {burst_id}. Removing from datatree.")
                        del slcs.bursts[burst_id]
                    elif status_or_data == 1:
                        logger.info(f"Reference image already in stack for burst {burst_id}.")
                else:
                    logger.info(f"Adding reference data for burst {burst_id}.")
                    slcs[f"bursts/{burst_id}/orig/{ref_date}"] = status_or_data

            if len(slcs.bursts) == 0:
                raise RuntimeError("No reference image found for any bursts. DataTree is empty.")

    # Handles cases where reference_date is provided, OR data is NOT organized by bursts
    else:
        if reference_date is None:
            raise ValueError("`reference_date` must be provided when data is not organized by bursts.")

        # Check if the reference date is present in the main 'orig' group
        if multiburst:
            if any(reference_date not in burst.orig for burst in slcs.bursts.values()):
                raise ValueError("Reference image not found in one or more bursts in the SLC stack.")
        else:
            if reference_date not in slcs.orig:
                raise ValueError("Reference image not found in the SLC stack.")

    if not isinstance(reference_date, str):
        raise TypeError(f"Expected `reference_date` to be a string, got {type(reference_date).__name__}")

    return slcs, reference_date


def import_slice(
    slc: xr.DataArray,
    in_path: str | xr.DataArray,
    index: int,
    processed_pol: str,
    calibration_flag: bool = False,
    calibration_polynomial_coeffs: np.ndarray | None = None,
    noise_removal_flag: str | None = None,
    noise_poly_coeffs: np.ndarray | None = None,
    apply_invalid_mask_flag: bool = False,
    valid_data_y_idxs: np.ndarray | None = None,
    valid_data_x_idxs: np.ndarray | None = None,
    deramp_flag: bool = False,
    dkcoa_coeffs: np.ndarray | None = None,
    chunk={},
):
    """
    Read, process, and save a single chunk of an SLC image.

    This function is the lowest-level worker in the pipeline. It reads a specific
    chunk of data from the source file, applies a series of processing steps
    (noise removal, calibration, deramping, masking), and writes the result
    to the corresponding region in the output Zarr store.

    Args:
        slc: The target dataset slice, used for coordinates and metadata.
        in_path: The path to the source data file or the xr.DataArray of the input data.
        index: The index of the image within the source file.
        calibration_flag: If True, apply radiometric calibration.
        noise_removal_flag: If True, apply thermal noise removal.
        apply_invalid_mask_flag: If True, apply an invalid data mask.
        deramp_flag: If True, apply phase deramping.
        chunk: A dictionary describing the specific slice (region) to process.
    """
    # --- Read Data Chunk ---
    # Skip if the polarization doesn't match the target.
    if processed_pol not in slc.polarization.values:
        return
    # Handle data that's already in memory (as an xarray.DataArray).
    if isinstance(in_path, xr.DataArray):
        # Assign the data from the in-memory array to the target slice.
        slc.data = in_path.data.reshape(slc.shape).astype("complex64")
    # Handle data that needs to be read from a file.
    else:
        ds = sarpy_open(in_path)
        # Read the specific binary data chunk from the file.
        arr = ds.read(slice(None, None), chunk["azimuth_time"], index=index).astype("complex64").T
        slc.data = arr.reshape(slc.shape)
    # --- Processing Steps ---
    # Noise must be removed before calibration as it's a function of raw DN.
    if noise_removal_flag:
        slc = noise_removal(slc, noise_poly_coeffs)
    # Apply amplitude calibration if specified.
    if calibration_flag:
        slc = amplitude_calibration(slc, calibration_polynomial_coeffs)
    # Perform deramping if specified.
    if deramp_flag:
        slc = image_deramp(slc, dkcoa_coeffs)
    # Apply a mask to zero out invalid data regions if specified.
    if apply_invalid_mask_flag:
        first_idx_y = chunk["azimuth_time"].start
        first_idx_x = chunk["slant_range_time"].start
        if valid_data_y_idxs is not None and valid_data_x_idxs is not None:
            slc = apply_invalid_mask(slc, valid_data_y_idxs, valid_data_x_idxs, first_idx_y, first_idx_x)
    slc = slc.astype("complex64")
    return slc.to_dataset(name="slc")


def import_image(
    group: str,
    outer_dict: dict,
    in_path: str | None,
    out_path: str,
    calib_type: str | None = None,
    noise_removal_flag: bool | None = None,
    deramp: bool = False,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks={"azimuth_time": 2048, "slant_range_time": 2048, "acquisition_date": 1, "polarization": 1},
    compute_on_host: bool = True,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Import, process, and save a single SLC image (or burst).

    This is the core worker function that handles a single image. It computes
    coordinates, initializes the output Zarr storage for the image, and then
    uses `map_blocks` to process the image data slice by slice.

    Args:
        group: The target group path within the Zarr store (e.g., 'bursts/iw1_123').
        outer_dict: A dictionary containing polarization and metadata information.
        in_path: The path to the input data file.
        out_path: The root path of the output Zarr store.
        calib_type: The radiometric calibration type.
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply phase deramping.
        apply_invalid_mask_flag: If True, apply an invalid data mask.
        chunks: Dask chunking configuration.
        reread: If True, reread the data from disk and return it.

    Returns:
        An xarray.DataTree of the processed image if `reread` is True, otherwise None.
    """

    # --- Metadata and Coordinate Extraction ---
    pols = []
    annot = {}
    data = {}
    indexes = {}
    if in_path is None:
        for e, (k, v) in enumerate(outer_dict.items()):
            pols.append(k)
            indexes[k] = v[0]
            data[k] = v[1]
            attrs = json.loads(v[2])
            annot[k] = attrs["annot"]
            data[k].attrs = attrs["annot"]
            if e == 0:
                stac_discovery = attrs["stac_discovery"]
                processing_history = attrs["processing_history"]
    else:
        # Extract polarization and metadata for each layer.
        for k, v in outer_dict.items():
            pols.append(k)
            indexes[k] = v[0]
            data[k] = v[1]
            annot[k] = sarpy_open(v[1]).sicd_meta[v[0]].to_dict()
            for el in ["Row", "Col"]:
                if "WgtFunct" in annot[k]["Grid"][el]:
                    del annot[k]["Grid"][el]["WgtFunct"]

    # Compute the coordinate arrays (azimuth/range in meters and time).
    azimuth_m, slant_range_m = get_az_rg_meters(annot[pols[0]])
    azimuth_time, slant_range_time = get_az_rg_times(annot[pols[0]])

    # Get the acquisition date from the metadata.
    try:
        acquisition_date = str(annot[pols[0]].Timeline.CollectStart)[:10]
    except AttributeError:
        acquisition_date = str(path_to_dict(annot[pols[0]], "Timeline.CollectStart"))[:10]

    # Define the final output path for this specific image group.
    data_path = os.path.join(out_path, group)
    # --- Initialize Zarr Storage ---
    dt = xr.DataTree.from_dict({k: xr.Dataset() for k in ["measurements", "conditions", "quality"]})
    if in_path is None:
        dt["conditions"].attrs["stac_discovery"] = stac_discovery
        dt["conditions"].attrs["processing_history"] = processing_history
    # --- Create and Save Empty SLC Dataset ---
    # This creates the structure on disk with the correct dimensions, coordinates, and metadata.
    coords = {
        "acquisition_date": ("acquisition_date", [acquisition_date]),
        "azimuth_time": ("azimuth_time", azimuth_time),
        "slant_range_time": ("slant_range_time", slant_range_time),
        "azimuth_m": ("azimuth_time", azimuth_m),
        "slant_range_m": ("slant_range_time", slant_range_m),
    }

    chunks_ = chunks.copy()
    if do_import:
        dims = ["polarization", "acquisition_date", "azimuth_time", "slant_range_time"]
        shape = [len(pols), 1, annot[pols[0]]["ImageData"]["NumCols"], annot[pols[0]]["ImageData"]["NumRows"]]
        coords["polarization"] = ("polarization", pols)
    else:
        dims = ["azimuth_time", "slant_range_time"]
        shape = [annot[pols[0]]["ImageData"]["NumCols"], annot[pols[0]]["ImageData"]["NumRows"]]
        chunks_.pop("polarization", None)
        chunks_.pop("acquisition_date", None)

    image = generate_empty_dataset(
        data_vars=["slc"],
        dims={"slc": dims},
        shape={"slc": shape},
        chunks=chunks_,
        dtype={"slc": "complex64"},
        coords=coords,
        attrs_data_vars={"slc": annot},
    )
    dt["measurements"] = image
    write(dt, data_path, mode="w", compute=False, storage_options=get_storage_options(kwargs))

    if do_import:
        # Define the path to the measurements group.
        measurements_path = os.path.join(data_path, "measurements")
        save_options = {
            "data_path": measurements_path,
            "save_options": {
                "skip_read": True,
                "consolidated": False,
            }
        }
        # --- Process Data Slice by Slice ---
        if in_path is not None:
            ds = sarpy_open(in_path)
        for pol in pols:
            index = indexes[pol]
            slc = data[pol]
            args = []
            obj = [image.slc]
            # Define the chunking strategy for `map_blocks`.
            if in_path is not None:
                chunksizes = {k: v for k, v in chunks.items() if k in image.dims}
                chunksizes["slant_range_time"] = slant_range_time.size
                chunksizes["azimuth_time"] = min([azimuth_time.size, chunksizes["azimuth_time"]])
                annot = ds.sicd_meta[index].to_dict()
                processed_pol = annot.get_processed_polarization().replace(":", "")
                args.append(slc)
            else:
                chunksizes = None
                slc = slc.chunk(
                    {k: v[0] for k, v in image.chunksizes.items() if k in ["slant_range_time", "azimuth_time"]}
                )
                annot = slc.attrs
                processed_pol = annot["ImageFormation"]["TxRcvPolarizationProc"].replace(":", "")
                obj.append(slc)
            calibration_flag = calib_type is not None
            noise_poly_coeffs = get_coefficients("Radiometric.NoiseLevel.NoisePoly.Coefs", annot)
            calibration_polynomial_coeffs = get_coefficients(f"Radiometric.{calib_type}SFPoly.Coefs", annot)
            dkcoa_polynomial_coeffs = get_coefficients("Grid.Col.DeltaKCOAPoly.Coefs", annot)
            valid_mask = annot.get("ImageData", {}).get("ValidData")
            valid_y, valid_x = None, None
            if valid_mask is not None:
                valid_y, valid_x = np.array([(e["Col"], e["Row"]) for e in valid_mask]).T
            args.extend(
                [
                    index,
                    processed_pol,
                    calibration_flag,
                    calibration_polynomial_coeffs,
                    noise_removal_flag,
                    noise_poly_coeffs,
                    apply_invalid_mask_flag,
                    valid_y,
                    valid_x,
                    deramp,
                    dkcoa_polynomial_coeffs,
                ]
            )
            map_blocks(
                import_slice,
                obj=obj,
                args=args,
                enable_dask=enable_dask,
                chunksizes=chunksizes,
                combine=False,
                compute_on_host=compute_on_host,
                store_mapping={0: save_options},
            )
    else:
        logger.info("Imported SLC image structure only (data import skipped).")

    image = read(data_path, storage_options=get_storage_options(kwargs))
    if not isinstance(image, xr.DataTree):
        raise ValueError(f"Expected xr.DataTree, got {type(image)}")
    return (group, image)


def group_discover(
    in_path: str | xr.DataTree,
    aoi_wkt: str | None = None,
    multiburst: bool = False,
    indices: Sequence[int] | None = None,
    polarizations: Sequence[str] | None = None,
    **kwargs,
):
    """
    Discover and group image metadata from a SAR data file.

    This function reads the metadata from a SAR product, filters it based on
    the area of interest (AOI), polarization, and other criteria, and then
    groups the images into a nested dictionary structure. For multiburst data,
    it organizes by burst ID; otherwise, it organizes by acquisition date.

    Args:
        in_path: Path to the input SAR data or a DataTree.
        aoi_wkt: Area of Interest in Well-Known Text format.
        multiburst: Flag to indicate multiburst mode.
        indices: Specific indices of the images to process.
        polarizations: List of polarizations of interest.

    Returns:
        A nested dictionary grouping image metadata by group ID and polarization.
    """
    storage_options = get_storage_options(kwargs, key="storage_options_slc")
    # Open the data with Sarpy, allowing partial reads for metadata discovery.
    ds = sarpy_open(in_path, allowed_partial_open=True, storage_options=storage_options)

    # If no specific indices are provided, process all images in the file.
    if indices is None:
        indices = range(len(ds.sicd_meta))

    # Use a defaultdict for convenient grouping.
    mydict: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for index in indices:
        annot = ds.sicd_meta[index]
        # Get acquisition date and polarization from metadata.
        acquisition_date = str(annot.Timeline.CollectStart)[:10]
        pol = annot.get_processed_polarization().replace(":", "")
        # Filter by polarization if specified.
        if polarizations:
            if pol not in polarizations:
                continue

        # --- Define the output group path ---
        if multiburst:
            # For multiburst, filter by geographic footprint intersection with AOI.
            footprint = get_footprint(ds, index)
            if (aoi_wkt is not None) and (not intersect(footprint, aoi_wkt)):
                continue

            # Generate a unique group ID based on swath and burst ID.
            if isinstance(in_path, xr.DataTree) or (not is_s3_path(in_path)):
                swath, burstid = get_burst_id(annot)
            else:
                storage_options = get_storage_options(kwargs)
                manifest_path = Path(f"{in_path}/manifest.safe", storage_options=storage_options)
                local_manifest_path = manifest_path.ensure_local()._path
                ns, root_node = sarpy.io.complex.sentinel._parse_xml(local_manifest_path)
                swath, burstid = get_burst_id_safe(
                    ds._sentinel_details._get_file_sets(),
                    annot,
                    get_relative_orbit_number(root_node, ns),
                )
            gg = os.path.join("bursts", f"{swath}_{burstid}", "orig", acquisition_date)
        else:
            # For single burst or stripmap, the group is just 'orig/date'.
            gg = os.path.join("orig", acquisition_date)
            swath, burstid = None, None
        # Append the relevant metadata/data to the corresponding group.
        if isinstance(in_path, xr.DataTree):
            datatree_group = map_index_to_group(in_path, swath, burstid, pol)
            annot = annot.to_dict()
            # Remove large, unnecessary fields from metadata.
            for e in ["Row", "Col"]:
                if "WgtFunct" in annot["Grid"][e]:
                    del annot["Grid"][e]["WgtFunct"]
            data = in_path[datatree_group]
            slc = data.measurements.slc
            attrs = {
                "annot": annot,
                "stac_discovery": data.attrs["stac_discovery"],
                "processing_history": data.attrs["processing_history"],
            }
            mydict[gg][pol].extend([index, slc, json.dumps(attrs)])
        else:
            mydict[gg][pol].extend([index, in_path])

    return mydict


def import_slc(
    in_path,
    out_path,
    aoi_wkt: str | None,
    multiburst: bool,
    calib_type: str | None = None,
    noise_removal_flag: bool = False,
    deramp: bool = False,
    indices: Sequence[int] | None = None,
    polarizations: list[str] | None = None,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks={"acquisition_date": 1, "azimuth_time": 2048, "slant_range_time": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> list[xr.DataTree]:
    """
    Import and process all the images contained within a single input file.

    This function acts as a dispatcher. It first discovers the relevant image groups
    within a single SAFE file (or other input format), then maps the `import_image`
    function to process each discovered group in parallel.

    Args:
        in_path: The path to the input data file (e.g., a .SAFE directory).
        out_path: The root path of the output Zarr store.
        aoi_wkt: Area of Interest in WKT format.
        multiburst: If True, handle data in multiburst mode.
        calib_type: The radiometric calibration type to apply.
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply phase deramping.
        indices: Specific image indices to process within the file.
        polarizations: Specific polarizations to process.
        reread: If True, reread the data from disk after processing and return it.
        apply_invalid_mask_flag: If True, apply a mask for invalid data.
        chunks: Dask chunking configuration.

    Returns:
        An xarray.DataTree of the processed data if `reread` is True, otherwise None.
    """
    storage_options = get_storage_options(kwargs, key="storage_options_slc")
    # Use a temporary directory for local processing, especially for S3 inputs.
    in_local_path = None
    if isinstance(in_path, str):
        path = Path(in_path, storage_options=storage_options).ensure_local()
        in_local_path = path._path
    # Discover all the image groups (e.g., bursts) within the local path.
    group_dict = group_discover(
        in_path if in_local_path is None else in_local_path, aoi_wkt, multiburst, indices, polarizations
    )
    # Prepare the arguments for the `import_image` map operation.
    args = [[index for index in group_dict.keys()], [group for group in group_dict.values()]]

    # Map the `import_image` function over the discovered groups.
    map_results = map(
        import_image,
        args,
        kwargs=dict(
            in_path=in_local_path,
            out_path=out_path,
            calib_type=calib_type,
            noise_removal_flag=noise_removal_flag,
            deramp=deramp,
            apply_invalid_mask_flag=apply_invalid_mask_flag,
            chunks=chunks,
            do_import=do_import,
            enable_dask=enable_dask,
            **kwargs,
        ),
        compute_on_host=True,  # Ensure data reading happens on the host.
        combine=False,
        enable_dask=enable_dask,
    )
    # If requested, open and return the DataTree from the output path.
    if not isinstance(map_results, list):
        raise ValueError(f"Expected list of xr.DataTree, got {type(map_results)}")
    return map_results


@timeit
def import_slc_stack(
    or_f: list[str | xr.DataTree],
    out_path,
    aoi_wkt: str | None = None,
    multiburst: bool = False,
    calib_type: str | None = None,
    noise_removal_flag: str | None = None,
    deramp: bool = True,
    indices: list[int] | None = None,
    polarizations: list[str] | None = None,
    apply_invalid_mask_flag: bool = False,
    do_import: bool = True,
    chunks={"polarization": 1, "acquisition_date": 1, "azimuth_time": 2048, "slant_range_time": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Import, process, and save a stack of SLC images into a unified DataTree structure.

    This function is the main entry point for the import pipeline. It first discovers all
    relevant image groups (e.g., bursts) from the input files, initializes the
    output Zarr storage, and then maps the `import_slc` function over each input
    file to perform the actual data processing and writing in parallel. Finally, it
    rereads the processed data from the output path and returns it as a DataTree.

    Args:
        or_f: A list of input file paths or pre-loaded DataTrees.
        out_path: The root path for the output Zarr store.
        aoi_wkt: An Area of Interest in WKT format to filter the data.
        multiburst: If True, handle the data as a collection of bursts that
        need to be organized by swath and burst ID.
        calib_type: The type of radiometric calibration to apply (e.g., "SigmaZero").
        noise_removal_flag: If True, apply thermal noise removal.
        deramp: If True, apply a phase deramping correction.
        indices: A list of specific image indices to process from the input files.
        polarizations: A string specifying the polarizations to process (e.g., "VV VH").
        apply_invalid_mask_flag: If True, apply a mask to zero out invalid data regions.
        chunks: A dictionary defining the chunk sizes for Dask processing.

    Returns:
        An xarray.DataTree containing the fully processed and consolidated SLC stack.
    """

    if not or_f:
        raise ValueError("The input list cannot be empty.")

    base_type = type(or_f[0])
    if base_type not in (str, xr.DataTree) or any(not isinstance(f, base_type) for f in or_f):
        raise TypeError(f"All elements must be uniformly xr.DataTree or str. Got: {base_type.__name__}")
    # --- Step 1: Discover Image Groups ---
    # In parallel, discover all the relevant image groups (bursts, polarizations, etc.) from the input files.
    res = map(
        group_discover,
        or_f,
        kwargs=dict(aoi_wkt=aoi_wkt, multiburst=multiburst, indices=indices, polarizations=polarizations, **kwargs),
        enable_dask=True if isinstance(base_type, str) else False,
        combine=False,
    )

    if not isinstance(res, list):
        raise Exception(f"Expected list, got {type(res)}")
    # Merge the list of dictionaries from all workers into a single dictionary of groups.
    groups = merge_dict(res)
    # --- Step 2: Initialize Output Storage ---
    logger.info(f"Initializing groups: {groups.keys()}")
    init_zarr_groups(out_path, groups.keys(), multiburst, **kwargs)
    # --- Step 3: Import and Process Data ---
    # Map the `import_slc` function over each input file. This will process the data
    # in parallel and write the results directly to the initialized Zarr store.
    slcs = xr.DataTree()
    if base_type is str:
        map_results = map(
            import_slc,
            or_f,
            kwargs=dict(
                out_path=out_path,
                aoi_wkt=aoi_wkt,
                multiburst=multiburst,
                calib_type=calib_type,
                noise_removal_flag=noise_removal_flag,
                deramp=deramp,
                indices=indices,
                polarizations=polarizations,
                apply_invalid_mask_flag=apply_invalid_mask_flag,
                chunks=chunks,
                do_import=do_import,
                enable_dask=enable_dask,
                **kwargs,
            ),
            map_equally=True,  # Distribute work evenly.
            combine=False,
            enable_dask=enable_dask,
        )
        # --- Step 4: Reread and Return Result ---
        # After all workers have finished writing, open the consolidated Zarr store.
        if not isinstance(map_results, list):
            raise ValueError(f"Expected list of xr.DataTree, got {type(map_results)}")
        for result in map_results:
            for group, slc in result:
                slcs[group] = slc
    else:
        # Prepare the arguments for the `import_image` map operation.
        args = [[index for index in groups.keys()], [group for group in groups.values()]]

        # Map the `import_image` function over the discovered groups.
        map_results = map(
            import_image,
            args,
            kwargs=dict(
                in_path=None,
                out_path=out_path,
                calib_type=calib_type,
                noise_removal_flag=noise_removal_flag,
                deramp=deramp,
                apply_invalid_mask_flag=apply_invalid_mask_flag,
                chunks=chunks,
                do_import=do_import,
                enable_dask=enable_dask,
                **kwargs,
            ),
            combine=False,
            enable_dask=enable_dask,
        )
        if not isinstance(map_results, list):
            raise ValueError(f"Expected list of xr.DataTree, got {type(map_results)}")
        for group, slc in map_results:
            slcs[group] = slc

    return slcs


def dem_to_radar(
    dem: xr.DataArray,
    orbits: xr.DataArray,
    working_dir: str,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_date": 1},
    skip_read: bool = True,
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset | None:
    """
    Compute the radar geometry (slant range, azimuth time) for each pixel of a DEM.

    This function orchestrates the DEM-to-radar-coordinate transformation. It first
    initializes the output data structure, populates it with elevation data, and then
    maps the `compute_simulation` function over blocks of the DEM to perform the
    geometric calculations in parallel.

    Parameters
    ----------
    dem : xr.DataArray
        The input DEM as an xarray.DataArray.
    orbits : xr.DataArray
        The satellite orbit data as an xarray.DataArray.
    working_dir : str
        A path to a directory for storing intermediate and final results.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    skip_read : bool
        If True, the function will not read and return the final dataset,
        useful for workflows that only need to ensure the data is written.
    enable_dask : bool
        If True, use Dask for parallel computation.

    Returns
    -------
    xr.Dataset or None
        The final simulation dataset read from disk if `skip_read` is False, otherwise None.
    """
    if (set(chunks.keys()) == set((*SPATIAL_DIMS_, "acquisition_date"))) is False:
        raise Exception("Chunks must contains x, y and acquisition_date keys")
    # Define the path for the simulation output.
    data_path = f"{working_dir}/simulation"
    acquisition_dates = orbits.acquisition_date.values
    xy_chunks = {k: v for k, v in chunks.items() if k in SPATIAL_DIMS_}
    x = dem.x.values
    y = dem.y.values
    # --- Step 1: Initialize Output Storage ---
    simulation = init_simulation(acquisition_dates, dem.rio.crs, x, y, xy_chunks, data_path, **kwargs)

    if not isinstance(simulation, xr.Dataset):
        raise Exception(f"Expected xr.Dataset, got {type(simulation)}")

    resampled_dem = dem.to_dataset(name="z")
    # --- Step 2: Populate with Elevation Data ---
    simulation = update_elevation(simulation, resampled_dem, xy_chunks, data_path, **kwargs)
    # --- Step 3: Compute Simulation in Parallel ---
    # Map the core computation function over blocks of the simulation and DEM datasets.
    save_options = {
        "data_path": normpath(data_path),
        "save_options": {
            "consolidated": False,
            "storage_options": get_storage_options(kwargs),
        }
    }
    map_blocks(
        compute_simulation,
        (simulation, resampled_dem.z),
        args=(orbits,),
        kwargs=kwargs,
        combine=False,
        enable_dask=enable_dask,
        store_mapping={0: save_options},
    )
    # Optionally, read and return the final result.
    return (
        read(normpath(data_path), as_type="xr.Dataset", storage_options=get_storage_options(kwargs))
        if not skip_read
        else None
    )


def single_simulation(
    burst_id: str,
    dem: xr.DataArray,
    multiburst: bool,
    orbits: xr.DataArray,
    working_directory: str,
    master_image: bool = False,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_date": 1},
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.Dataset]:
    """
    Run the DEM-to-radar simulation for a single burst or scene.

    This function acts as a wrapper around `dem_to_radar`. It handles the logic for
    path management and decides whether to recompute the simulation or use an
    existing result, which is particularly useful for the master image in a stack.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst or scene being processed.
    dem : xr.DataArray
        The DEM data for this specific burst.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    orbits : xr.DataArray
        The satellite orbit data.
    working_directory : str
        The root directory for all processing outputs.
    master_image : bool, optional
        If True, this is the master image of the stack. Its
        simulation can be cached to speed up subsequent runs. Defaults to False.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
        Defaults to {"x": 1024, "y": 1024, "acquisition_date": 1}.
    recompute : bool, optional
        If True, force recomputation even if a cached result exists.
        Defaults to False.
    enable_dask : bool, optional
        If True, use Dask for parallel computation. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing the burst_id (str) and the resulting simulation Dataset (xr.Dataset).
    """
    # Define the relative path for this burst's data.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory}/{path}")
    # Define a path for a "done" file to check for completion.
    simulation_path = f"{working_directory}/processed_data/{path}/simulation.ok"
    # --- Check for Existing Results ---
    # Decide whether to run the computation or use a cached result.
    if (
        (master_image is False)
        or (exists(simulation_path, storage_options=get_storage_options(kwargs)) is False)
        or recompute
    ):
        # If this is the master image and we're recomputing, delete the "done" file.
        if master_image:
            delete(
                simulation_path,
                allow_non_existent=True,
                storage_options=get_storage_options(kwargs),
            )
        # --- Run the Main Simulation ---
        dem_to_radar(
            dem=dem,
            orbits=orbits,
            working_dir=data_path,
            chunks=chunks,
            skip_read=True,  # We will read the data later.
            enable_dask=enable_dask,
            **kwargs,
        )
        # If this was the master image, create the "done" file to cache the result.
        if master_image:
            makedirs(
                normpath(f"{working_directory}/processed_data/{path}"),
                exist_ok=True,
                **kwargs,
            )
            touch(simulation_path, **kwargs)
    else:
        logger.info(f"Found existing master image simulation file for {burst_id}")
    # Read the final simulation data from disk.
    simulation = read(
        normpath(f"{data_path}/simulation"),
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs),
    )
    return (burst_id, simulation)


@timeit
def stack_simulation(
    dem_tree: xr.DataTree,
    orbits: xr.DataArray,
    multiburst: bool,
    working_directory: str,
    chunks: dict = {"x": 1024, "y": 1024, "acquisition_date": 1},
    master_image: bool = False,
    recompute: bool = False,
    **kwargs,
) -> xr.DataTree:
    """
    Run the DEM-to-radar simulation for an entire stack of bursts or scenes.

    This function maps the `single_simulation` function over all the items
    in a DataTree, processing each burst or scene in parallel. It then assembles
    the results back into a single output DataTree.

    Parameters
    ----------
    dem_tree : xr.DataTree
        An xarray.DataTree where each node contains the DEM for a burst.
    orbits : xr.DataArray
        The satellite orbit data for the entire stack.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    working_directory : str
        The root directory for all processing outputs.
    chunks : dict, optional
        A dictionary defining the chunk sizes for Dask processing.
        Defaults to {"x": 1024, "y": 1024, "acquisition_date": 1}.
    master_image : bool, optional
        If True, indicates that this is the master image's simulation.
        Defaults to False.
    recompute : bool, optional
        If True, force recomputation of all simulations. Defaults to False.

    Returns
    -------
    xr.DataTree
        An xarray.DataTree containing the simulation results for all bursts/scenes.
    """
    # Prepare the arguments for the parallel map operation.
    for k, d in (dem_tree.bursts if multiburst else {"data": dem_tree}).items():
        in_proj = d.z.rio.crs
        break
    args = [(k, d.z, multiburst) for k, d in (dem_tree.bursts if multiburst else {"data": dem_tree}).items()]
    args = list(zip(*args))
    # --- Parallel Execution ---
    results = cast(
        list,
        map(
            single_simulation,
            args,
            # Pass common arguments via kwargs.
            kwargs=dict(
                orbits=orbits,
                working_directory=working_directory,
                master_image=master_image,
                chunks=chunks,
                recompute=recompute,
                **kwargs,
            ),
            enable_dask=kwargs.get("enable_dask", True),
            Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
            combine=False,
        ),
    )
    # --- Assemble Results ---
    simulation_tree = xr.DataTree()
    for k, sim in results:
        # Reconstruct the DataTree structure from the list of results.
        if multiburst:
            simulation_tree[f"bursts/{k}"] = sim
        else:
            simulation_tree = xr.DataTree(sim)
    return simulation_tree


@timeit
def stack_simulation_ml(
    simulation_tree: xr.DataTree,
    reference_date: str,
    multiburst: str,
    working_directory: str,
    recompute: bool = False,
    **kwargs,
):
    """
    Create multilooked simulation data for an entire stack of bursts or scenes.

    This function maps `single_simulation_ml` over all items in a DataTree to
    generate the downsampled simulation overviews for each burst/scene in parallel.

    Parameters
    ----------
    simulation_tree : xr.Datatree
        The DataTree containing the full-resolution simulations.
    reference_date : str
        The reference acquisition date.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    working_directory :str
        The root directory for all outputs.
    recompute : bool
        If True, force recomputation.

    Returns
    -------
    dict
        A dictionary (or nested dictionary for multiburst) containing the
        multilooked simulation data for the entire stack.
    """
    # Prepare the arguments for the parallel map operation.
    args = [
        (k, s.to_dataset(), multiburst)
        for k, s in (simulation_tree.bursts if multiburst else {"data": simulation_tree}).items()
    ]
    args = list(zip(*args))
    # --- Parallel Execution ---
    results = cast(
        list,
        map(
            single_simulation_ml,
            args,
            kwargs=dict(
                reference_date=reference_date,
                working_directory=working_directory,
                recompute=recompute,
                **kwargs,
            ),
            combine=False,
            enable_dask=kwargs.get('enable_dask', True)
        ),
    )
    # --- Assemble Results ---
    simulation_ml_tree: dict = {"bursts": {}} if multiburst else {}
    for k, sim_ml in results:
        # Reconstruct the nested dictionary structure from the list of results.
        if multiburst:
            simulation_ml_tree["bursts"][k] = sim_ml
        else:
            simulation_ml_tree = sim_ml
    return simulation_ml_tree


def import_orbit(
    orbit_paths: list[Path] | None = None,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
    **kwargs,
) -> Union[xr.Dataset | None]:
    """
    Import and filter orbit data from specified files for a given time interval.

    This function reads orbit data from one or more files and then filters the
    resulting dataset to a specific time range defined by `start_time` and
    `end_time`. It is decorated with a retry mechanism to handle transient
    network or file access issues.

    Parameters
    ----------
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to the orbit files to import.
    start_time : np.datetime64 | None, optional
        The start of the desired time interval for the orbit data.
    end_time : np.datetime64 | None, optional
        The end of the desired time interval for the orbit data.

    Returns
    -------
    xr.Dataset | None
        An xarray.Dataset containing the filtered orbit data, or None if no
        `orbit_paths` are provided.
    """
    # Proceed only if orbit paths are provided.
    if (orbit_paths is not None) and all(e is not None for e in orbit_paths):
        logger.info(f"For the time interval {start_time} - {end_time} these orbit files will be used: {orbit_paths}")
        # Read the raw orbit data from the specified file(s).
        orbit = read_orbit_file(orbit_paths, **kwargs)
    else:
        # If no paths are given, log the information and return None.
        logger.info(f"For the time interval {start_time} - {end_time} no external orbits have been found")
        return None

    # If start_time is not provided, default to the first timestamp in the orbit data.
    if start_time is None:
        start_time = orbit.azimuth_time[0]
    # If end_time is not provided, default to the last timestamp.
    if end_time is None:
        end_time = orbit.azimuth_time[-1]

    # Filter the orbit dataset to the specified time interval and add the source filenames as an attribute.
    return orbit.where((orbit.azimuth_time >= start_time) & (orbit.azimuth_time <= end_time), drop=True).assign(
        orbit_file=(["acquisition_date"], np.array([",".join([e.name for e in orbit_paths])], dtype=object))
    )


def update_orbit(
    slcs: xr.DataTree,
    wdir_path: str,
    orbit_paths: list[Path] | None = None,
    multiburst: bool = True,
    poly_degree: int = 10,
    border: np.timedelta64 = np.timedelta64(100, "s"),
    enable_dask: bool = True,
    **kwargs,
):
    """
    Orchestrate the update of orbit polynomials for an entire SLC stack.

    This function identifies the required time intervals for each acquisition,
    imports the corresponding precise orbit data, and then maps the
    `overwrite_coeffs` worker function in parallel to refit the polynomials
    and update the metadata for every SLC in the stack.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data to be updated.
    orbit_paths : list[cop_ard_core.computing.io.Path] | None, optional
        A list of paths to precise orbit files. If None, it will try to
        find paths from ETAD metadata.
    multiburst : bool, optional
        A boolean flag for multiburst mode. Defaults to True.
    poly_degree : int, optional
        The degree for the new polynomial fit. Defaults to 10.
    border : np.timedelta64, optional
        A time border to add to the acquisition interval when fetching
        orbit data, to ensure sufficient data for fitting.
        Defaults to np.timedelta64(100, "s").

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree with new orbit information.
    """
    slcs_info: dict[str, Any] = {}
    # --- Step 1: Collect SLC Time Intervals and Metadata ---
    # Iterate through all original SLCs in the DataTree.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for k, v in i.orig.items():
            # Retrieve the orbit source file path if available (e.g., from ETAD metadata).
            if orbit_paths is None:
                try:
                    orb_file = v.conditions.ETAD.attrs.get("orbit_file", None)
                except Exception as e:
                    logger.warning(f"type is: {e.__class__.__name__}")
                    logger.warning("Orbit file not found")
                    orb_file = None
            else:
                orb_file = None
            # Extract metadata and the start/end azimuth times.
            annot = v.measurements.slc.attrs[list(v.measurements.slc.attrs.keys())[0]]
            info = [
                [
                    annot["CollectionInfo"]["CollectorName"],
                    orb_file,
                    v.measurements.slc.azimuth_time[0].values,
                    v.measurements.slc.azimuth_time[-1].values,
                ]
            ]
            # Group the information by acquisition date.
            if k in slcs_info:
                slcs_info[k] += info
            else:
                slcs_info[k] = info

    # Consolidate the information for each date to find the overall time range.
    slcs_info = {k: list(zip(*v)) for k, v in slcs_info.items()}
    slcs_info = {k: [v[0][:2], v[1], np.min(v[2]) - border, np.max(v[3]) + border] for k, v in slcs_info.items()}

    # --- Step 2: Match Orbit Files to Time Intervals ---
    if orbit_paths is not None:
        # Parse the validity time from the orbit filenames.
        orbit_intervals = np.array(
            [
                [dateutil.parser.parse(j).isoformat() for j in i.name.split("_V")[1][:-4].split("_")]
                for i in orbit_paths
            ],
            dtype="datetime64[ns]",
        )
        # For each acquisition, find the orbit files that cover its time interval.
        for k, v in slcs_info.items():
            (i_valid,) = np.where((v[2] <= orbit_intervals[:, 1]) & (v[3] >= orbit_intervals[:, 0]))
            if i_valid.size:
                slcs_info[k][1] = np.array(orbit_paths)[i_valid].tolist()

    # --- Step 3: Import Orbit Data in Parallel ---
    dates = list(slcs_info)
    args = [[slcs_info[e][i] for e in dates] for i in range(1, 4)]
    orbits = map(import_orbit, args=args, kwargs=kwargs, combine=False, enable_dask=False)
    if not isinstance(orbits, list):
        raise Exception(f"Expected list, got {type(orbits)}")
    orbits = {d: o for d, o in zip(dates, orbits)}

    # --- Step 4: Refit Polynomials and Update Metadata in Parallel ---
    # Prepare arguments for the `overwrite_coeffs` worker function.
    args = [
        (k, d, v.measurements.slc, orbits[d])
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
        for d, v in s.orig.items()
    ]
    args = list(zip(*args))
    results = cast(
        list,
        map(
            overwrite_coeffs,
            args,
            kwargs={"wdir_path": wdir_path, "poly_degree": poly_degree, "multiburst": multiburst, **kwargs},
            combine=False,
            enable_dask=enable_dask,
        ),
    )
    # --- Step 5: Integrate Results ---
    # Update the main DataTree with the new orbit condition datasets.
    slcs = update_slcs(slcs, multiburst, results)
    return slcs


@timeit
def import_ETAD(
    slcs: xr.DataTree,
    acquisition_dates: list[str],
    etad_files: list[Path],
    wdir_path: str,
    multiburst: bool = False,
    enable_dask: bool = True,
    **kwargs,
):
    """
    Find and import relevant ETAD and precise orbit files for a set of acquisitions.

    This function serves as a high-level wrapper. It filters a list of all available
    ETAD files to find the ones that match the acquisition dates of interest. It then
    calls `import_etad_stack` to perform the import and updates the precise orbit
    information in the `slcs` DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all the SLC data.
    acquisition_dates : list[str]
        A list of acquisition date strings to process.
    etad_files : list[str]
        A list of all available ETAD file paths to search through.
    wdir_path : str
        The root working directory.
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree, now containing the imported ETAD and precise
        orbit information.
    """
    is_valid = all(isinstance(x, Path) for x in etad_files)
    if not is_valid:
        raise TypeError("Invalid input: Expected a list of Path objects for ETAD files.")
    acquisition_date: Union[str, np.datetime64]
    etad_paths = []
    # --- Find Matching ETAD Files ---
    for acquisition_date in acquisition_dates:
        acquisition_date = np.datetime64(acquisition_date)
        # Format the date as YYYYMMDD to match typical ETAD filenames.
        date = acquisition_date.astype("M8[D]").astype(object).strftime("%Y%m%d")
        # Find all files in the `etad_files` list that contain this date string.
        files = [file for file in etad_files if date in file.name]
        if files:
            etad_paths.extend(files)
    used_etad: list[str] = []
    # --- Import ETAD and Orbits ---
    if etad_paths:
        # Call the main import function with the filtered list of ETAD paths.
        slcs, orbit_paths = import_etad_stack(
            etad_file_list=etad_paths,
            slcs=slcs,
            multiburst=multiburst,
            wdir_path=f"{wdir_path}/slcs",
            enable_dask=enable_dask,
            **kwargs,
        )
        # Check if orbit files were found for all necessary acquisitions.
        if any(e is None for e in orbit_paths):
            raise RuntimeError("The input slcs is not completely convered by the input ETAD file(s)")
        # Update the SLC DataTree with the new precise orbit information.
        logger.info(orbit_paths)
        update_orbit(
            slcs=slcs,
            multiburst=multiburst,
            orbit_paths=orbit_paths,
            enable_dask=enable_dask,
            wdir_path=wdir_path,
            **kwargs,
        )
        used_etad = [i.path.split(".SAFE")[0] + ".SAFE" for i in orbit_paths]
    return slcs, used_etad


@timeit
def import_ORBIT(
    slcs: xr.DataTree,
    acquisition_dates: list[np.datetime64],
    orbit_files: list[Path],
    wdir_path: str,
    multiburst: bool = False,
    enable_dask: bool = True,
    **kwargs,
):
    """
    High-level wrapper to find and import precise orbits for a stack.

    This function finds the relevant precise orbit files from a given folder that
    correspond to a list of acquisition dates, and then triggers the main
    `update_orbit` process to refit and update the orbit information in the `slcs`
    DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The main DataTree containing all SLC data.
    acquisition_dates : list[np.datetime64]
        A list of acquisition dates to find orbits for.
    orbit_folder : str
        The path to a folder containing orbit files (.EOF).
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
        Defaults to False.

    Returns
    -------
    xr.DataTree
        The updated `slcs` DataTree containing the new precise
        orbit information.
    """
    orbit_paths = []
    # --- Find Relevant Orbit Files ---
    # For each acquisition date, determine the total time span covered by all its bursts.
    for acquisition_date in acquisition_dates:
        for counter, (k, slc) in enumerate((slcs.bursts if multiburst else {"data": slcs}).items()):
            if acquisition_date in list(slc.orig.keys()):
                az = slc.orig[acquisition_date].measurements.azimuth_time
                s, e = az.min().values, az.max().values
                if counter == 0:
                    start_time = s
                    end_time = e
                else:
                    start_time = min(start_time, s)
                    end_time = max(end_time, e)
        # Find the orbit files whose validity period covers this time span.
        for o in orbit_files:
            # Parse the start and end validity times from the orbit filename.
            pattern = r"V(\d{8}T\d{6})_(\d{8}T\d{6})"
            match = re.search(pattern, o.name)
            if not match:
                raise ValueError(f"No validity dates found in filename: {o.name}")

            # Convert extracted strings directly to numpy datetime objects
            valid_start = np.datetime64(datetime.strptime(match.group(1), "%Y%m%dT%H%M%S"))
            valid_stop = np.datetime64(datetime.strptime(match.group(2), "%Y%m%dT%H%M%S"))
            # Add the orbit file if it completely covers the acquisition time window
            if valid_start <= start_time and end_time <= valid_stop:
                orbit_paths.append(o)

    # --- Trigger Orbit Update ---
    if orbit_paths:
        # Call the main update function with the filtered list of orbit paths.
        update_orbit(
            slcs, multiburst=multiburst, orbit_paths=orbit_paths, wdir_path=wdir_path, enable_dask=enable_dask, **kwargs
        )
    orbit_paths = [o.path for o in orbit_paths]
    return slcs, orbit_paths


def map_geocode_phase(
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path_static: str,
    data_path_tmp: str,
    kernel: dict,
    gamma_flattening: bool,
    spacing: dict,
    chunks: dict,
    do_reramp: bool = True,
    enable_dask: bool = True,
    skip_read: bool = True,
    **kwargs,
) -> xr.Dataset | None:
    """
    Orchestrate the block-wise geocoding of a single coregistered SLC.

    This function initializes an empty Zarr store for the final Geocoded SLC (GSLC)
    product and then uses `map_blocks` to apply the `warp_gslc` function to
    chunks of the simulation grid.

    Parameters
    ----------
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data.
    data_path_static : str
        Path for persistent static data.
    data_path_tmp : str
        Path for temporary processing data.
    coreg_path : str
        Path to the source coregistered data.
    kernel : dict
        A dictionary with parameters for the sinc interpolation kernel.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool, optional
        If True, re-apply the phase ramp. Defaults to True.
    enable_dask : bool, optional
        If True, use Dask for sub-tasks. Defaults to True.
    skip_read : bool, optional
        If True, do not read and return the final dataset. Defaults to True.

    Returns
    -------
    xr.Dataset or None
        The geocoded Dataset if `skip_read` is False, otherwise None.
    """
    # --- Prepare Simulation Subset ---
    annot = cslc.measurements.cstack.attrs
    annot = annot[list(annot.keys())[0]]
    # Calculate grid spacing for margin calculation.
    margin = get_margin(annot)
    # Extract the subset of the simulation grid relevant to this CSLC.
    azimuth_time = cslc.measurements.azimuth_time
    slant_range_time = cslc.measurements.slant_range_time
    bbox_radar = get_radar_bbox(azimuth_time, slant_range_time, margin)
    bbox_geo = get_geo_from_radar_bbox(simulation_ml, bbox_radar)
    
    if bbox_geo is None:
        raise ValueError("Could not determine geographic bounding box from radar bounding box.")
    selection = (
        (
            extract_simul_on_radar_chunk(
                simulation, simulation.acquisition_date.values[0], bbox_radar, bbox_geo,
            )
        )
        .squeeze()
        .drop_vars(["axis", "dem_distance", "acquisition_date", "h"])
    )

    # Define the output coordinates at pixel centers.
    shape, off = compute_shape_and_off(selection, spacing)
    # Create the coordinate arrays for the output grid.
    x, y = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output GSLC Dataset ---
    # Create the empty Zarr store for the Geocoded SLC.
    image = save_empty_dataset(
        data_vars=["gstack", "phase_model"],
        dims={
            "gstack": ["polarization", "acquisition_date", "y", "x"],
            "phase_model": ["acquisition_date", "y", "x"],
        },
        shape={
            "gstack": [
                cslc.measurements.polarization.size,
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
            "phase_model": [
                cslc.measurements.acquisition_date.size,
                y.size,
                x.size,
            ],
        },
        chunks=chunks,
        dtype={
            "gstack": cslc.measurements.cstack.dtype,
            "phase_model": cslc.measurements.phase_model.dtype,
        },
        coords={
            "polarization": cslc.measurements.polarization.values,
            "acquisition_date": cslc.measurements.acquisition_date.values,
            "y": y,
            "x": x,
        },
        crs=selection.rio.crs.to_string(),
        data_path=normpath(f"{data_path_tmp}/geo_sinc"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        skip_read=False,
    )

    # --- Map the Geocoding Worker Function ---
    # This call applies the `warp_gslc` function to each block of the simulation grid.
    chunksizes = {k: v[0] for k, v in image.chunksizes.items()}
    chunksizes["acquisition_date"] = 10
    chunksizes["polarization"] = image.polarization.size
    radar_path = normpath(f"{data_path_static}/radar_grid")
    radar_data = read(
        radar_path,
        consolidate_metadata=False,
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )
    # Define paths to the different components of the coregistered data.
    save_options = {
        "data_path": normpath(f"{data_path_tmp}/geo_sinc"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs, key="storage_options_dynamic"),
        },
    }

    def region_callback(region: Mapping[str, slice], kwargs: dict) -> Mapping[str, slice]:
        out = {}
        for k, v in region.items():
            if k in ["y", "x"]:
                out[k] = slice(v.start, v.stop - 1)
            elif k == "polarization":
                out[k] = slice(0, cslc.measurements.polarization.size)
            elif k == "acquisition_date":
                out[k] = v
            else:
                raise ValueError(f"Unexpected dimension {k}")
        return out

    # Extract metadata from the SLC attributes.
    annot = cslc.measurements.cstack.attrs
    annot = annot[list(annot)[0]]
    # Calculate the time interval per pixel in azimuth and range.
    dtaz = annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"] * 1e9
    dtrg = 2 * annot["Grid"]["Row"]["SS"] / speed_of_light
    scp_idx = annot["ImageData"]["SCPPixel"]["Col"], annot["ImageData"]["SCPPixel"]["Row"]


    map_blocks(
        warp_gslc,
        image,
        args=(
            selection,
            kernel,
            cslc.measurements.cstack,
            cslc.measurements.phase_model,
            cslc.conditions.phase_demod.phase_demod,
            radar_data.anf,
            radar_data.layover,
            radar_data.shadow,
            do_reramp,
            gamma_flattening,
            dtaz,
            dtrg,
            scp_idx,
        ),
        kwargs=kwargs,
        chunksizes=chunksizes,
        combine=False,
        enable_dask=enable_dask,
        region_callback={0: region_callback},
        store_mapping={0: save_options},
    )

    # Optionally read and return the final dataset.
    return (
        read(
            normpath(f"{data_path_tmp}/geo_sinc"),
            as_type="xr.Dataset",
            storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
        )
        if not skip_read
        else None
    )


def single_geocode_phase(
    burst_id: str,
    cslc: xr.DataTree,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    working_directory_static: str,
    working_directory_tmp: str,
    gamma_flattening: bool,
    spacing: dict,
    do_reramp: bool,
    chunks: dict,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Execute the end-to-end geocoding workflow for a single burst.

    This worker function prepares the interpolation kernel and calls `map_geocode_phase`
    to perform the core geocoding computation. It then merges the result with
    other geometric grid data (like incidence angles) to create the final
    geocoded product for the burst.

    Parameters
    ----------
    burst_id : str
        The identifier of the burst being processed.
    cslc : xr.DataTree
        The input DataTree for the coregistered burst.
    simulation : xr.Dataset
        The high-resolution simulation data for this burst.
    simulation_ml : dict
        The multilooked simulation data for this burst.
    multiburst : bool
        A boolean flag for multiburst mode.
    working_directory_static : str
        The path for storing persistent static data.
    working_directory_tmp : str
        The path for temporary processing data.
    gamma_flattening : bool
        If True, apply radiometric terrain correction.
    do_reramp : bool
        If True, re-apply the phase ramp.
    enable_dask : bool, optional
        If True, enable Dask for sub-tasks.
    **kwargs
        Additional keyword arguments for the warping function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the final geocoded xarray.Dataset for that burst.
    """
    # Define the output paths for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path_static = normpath(f"{working_directory_static}/{path}")
    data_path_tmp = normpath(f"{working_directory_tmp}/{path}")

    # --- Prepare Interpolation Kernel ---
    # Get warp parameters from kwargs or use default values.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = kwargs.get("warp_kernel_shape", np.array([11, 11]))
    kaiser_beta = kwargs.get("warp_kaiser_beta", 6)

    # Pre-compute the sinc kernels for x and y directions.
    sinc_x = sinc(kernel_shape[1], ov_fact, kaiser_beta)
    sinc_y = sinc(kernel_shape[0], ov_fact, kaiser_beta)

    kernel = {
        "kernel_shape": kernel_shape,
        "ov_fact": ov_fact,
        "kaiser_beta": kaiser_beta,
        "sinc_x": sinc_x,
        "sinc_y": sinc_y,
    }

    # --- Run Core Geocoding ---
    # This call orchestrates the block-wise warping from slant range to ground range.
    g = map_geocode_phase(
        cslc,
        simulation,
        simulation_ml,
        data_path_static,
        data_path_tmp,
        kernel,
        gamma_flattening,
        spacing,
        chunks,
        do_reramp,
        enable_dask,
        skip_read=False,  # We need the result back in memory.
        **kwargs,
    )

    # --- Merge with Geometric Data ---
    # Read the pre-computed geometric grid data (e.g., LIA, anf).
    geo_data = read(
        normpath(f"{data_path_static}/geo_grid"),
        mask_and_scale=False,
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )

    gslc = xr.DataTree.from_dict({"measurements": g, "conditions": geo_data})

    return (burst_id, gslc)


@timeit
def stack_geocoding(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict,
    working_directory_static: str,
    working_directory_tmp: str,
    spacing: dict,
    gamma_flattening: bool = False,
    do_reramp: bool = True,
    multiburst: bool = True,
    enable_dask: bool = True,
    chunks: dict = {"acquisition_date": 1, "polarization": 1, "y": 2048, "x": 2048},
    **kwargs,
):
    """
    Geocode a stack of coregistered SLCs in parallel.

    This function is the main entry point for the geocoding phase. It prepares
    the necessary arguments for each burst (or a single scene) and then maps the
    `single_geocode_phase` worker function across a Dask cluster to perform the
    geocoding in parallel. Finally, it assembles the geocoded results into a
    new DataTree.

    Parameters
    ----------
    slcs : xr.DataTree
        The input DataTree containing the coregistered SLC stack (`coreg` group).
    simulation_tree : xr.DataTree
        A DataTree containing the high-resolution simulation data
        (radar geometry on the DEM grid) for each burst.
    simulation_ml_tree : dict
        A dictionary with the downsampled ("multilooked") simulation
        data, used for quickly finding overlaps.
    working_directory_static : str
        The path to a directory for storing persistent static data.
    working_directory_tmp : str
        The path to a directory for temporary processing data.
    gamma_flattening : bool, optional
        If True, apply a radiometric terrain correction to produce
        gamma nought backscatter.
    do_reramp : bool, optional
        If True, re-apply the phase ramp to the geocoded data.
    multiburst : bool, optional
        A boolean flag indicating if the data is in multiburst format.
    **kwargs
        Additional keyword arguments passed to the processing functions,
        such as `warp_kernel_shape`.

    Returns
    -------
    xr.DataTree
        A new DataTree containing the geocoded SLC (GSLC) data for each burst.
    """
    args = []
    # --- Prepare Arguments for Parallel Mapping ---
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.coreg.drop_nodes("quality")
        unwanted_nodes = [child for child in v["conditions"].children if child != "phase_demod"]
        v["conditions"] = v["conditions"].drop_nodes(unwanted_nodes)
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # --- Execute Geocoding in Parallel ---
    # Map the `single_geocode_phase` worker function over the list of arguments.
    results = cast(
        list,
        map(
            single_geocode_phase,
            args,
            kwargs=dict(
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                gamma_flattening=gamma_flattening,
                do_reramp=do_reramp,
                enable_dask=enable_dask,
                chunks=chunks,
                **kwargs,
            ),
            enable_dask=enable_dask,
            combine=False,
        ),
    )

    # --- Assemble Results ---
    gslcs = xr.DataTree()
    # Reconstruct the DataTree from the list of results returned by the workers.
    for k, gslc in results:
        if multiburst:
            gslcs[f"bursts/{k}"] = gslc
        else:
            gslcs = gslc
    return gslcs


def import_etad_stack(
    etad_file_list: list,
    slcs: xr.DataTree,
    multiburst: bool,
    wdir_path: str,
    enable_dask: bool,
    **kwargs,
):

    slcs_az = xr.DataTree()
    for burst_id, slc in slcs.bursts.items():
        for date in slc.orig:
            measurements = slc.orig[date].measurements
            az_min, az_max = measurements.azimuth_time.min().values, measurements.azimuth_time.max().values
            slcs_az[f"bursts/{burst_id}/orig/{date}/measurements"] = xr.DataArray([az_min, az_max], dims=["point"])

    orbit_paths = []
    flat_burst_args = []
    # 1. Iterate over all files to collect bursts and metadata
    for etad_file in etad_file_list:
        nc_files = etad_file.glob("measurement/*.nc")
        if not nc_files:
            raise ValueError(f"No .nc file found at {etad_file.path}/measurement")
        etad_nc_path = nc_files[0].path
        open_kwargs: dict[str, Any] = {"chunks": {}, "cache": False}
        local_file = "://" not in etad_nc_path or etad_nc_path.startswith("file://")
        if not local_file:
            open_kwargs["storage_options"] = etad_file.storage_options
        etad_nc = xr.open_datatree(etad_nc_path, engine="h5netcdf", **open_kwargs)
        # Discover dates and times
        t0az = np.datetime64(etad_nc.attrs["azimuthTimeMin"], "ns")
        t0rg = etad_nc.attrs["rangeTimeMin"]
        etad_acquisition_date = discover_etaddate_slcsdate(slcs_az, etad_nc, multiburst).pop()

        if not etad_acquisition_date:
            logger.info(f"ETAD file without correspondence: {etad_nc_path}")
            continue

        # Save orbit path
        orbit_paths.append(etad_file.glob("annotation/*.EOF")[0])

        # Traverse the DataTree immediately to extract bursts
        for sw_name, etad_sw in etad_nc.items():
            for et_name, etad_burst in etad_sw.items():
                burst_key = f"{sw_name.lower()}_{etad_burst.attrs['burstId']}"

                # Check if burst exists in input SLC data
                if burst_key in (slcs.bursts if multiburst else {"data": slcs}):
                    flat_burst_args.append((etad_burst, sw_name, etad_acquisition_date, etad_nc_path, t0az, t0rg))

    if not flat_burst_args:
        return slcs, orbit_paths

    # Transpose the list of tuples
    mapped_args = list(zip(*flat_burst_args))
    flat_results = cast(
        list,
        map(
            update_etad,
            mapped_args,
            kwargs=dict(wdir_path=f"{wdir_path}/bursts" if multiburst else wdir_path, **kwargs),
            combine=False,
            enable_dask=enable_dask,
        ),
    )

    # 3. Integrate all results back into the DataTree
    for ad, bid, et in flat_results:
        if multiburst:
            slcs[f"bursts/{bid}/orig/{ad}/conditions/ETAD"] = et
        else:
            slcs[f"orig/{ad}/conditions/ETAD"] = et

    return slcs, orbit_paths


@timeit
def dem_reproject(
    dem: xr.DataArray,
    spacing: Dict[str, Any],
    outproj: str | int,
    data_path: str,
    chunks: dict = {"x": 2048, "y": 2048},
    enable_dask: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Reproject a Digital Elevation Model (DEM) to a specified CRS.

    This function orchestrates the reprojection of a DEM. It defines the output
    grid, initializes an empty Zarr store for the result, and then maps the
    `interp_dem` function over blocks to perform the interpolation in parallel.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    spacing
        The desired spatial resolution (spacing) in the output CRS units.
    outproj
        The target EPSG code for the output DEM.
    data_path
        The path where the reprojected DEM Zarr store will be created.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    enable_dask : bool
        If True, use Dask for parallel computation.
    **kwargs
        Additional keyword arguments, such as `method` for interpolation.

    Returns
    -------
    xarray.Dataset
        The reprojected DEM as an xarray.Dataset, read back from the Zarr store.
    """
    method = kwargs.pop("method", "cubic")
    # Ensure the input DEM has 'x' and 'y' coordinates.
    if (all([i in dem.coords for i in ["x", "y"]])) is False:
        raise Exception('DEM dataset coordinates must contain "x" and "y"')
    dem_ds = dem.to_dataset(name="z")

    # --- CRS Handling ---
    CRS_EPSG = f"EPSG:{outproj}" if not str(outproj).startswith("EPSG:") else outproj
    if not isinstance(CRS_EPSG, str):
        raise Exception(f"Expected str, got {type(CRS_EPSG)}")
    # Create 3D CRS objects for robust transformation.
    in_crs, out_crs = CRS(dem.rio.crs).to_3d(), CRS(CRS_EPSG).to_3d()
    dem_ds.rio.write_crs(in_crs, inplace=True)
    # Construct a composite CRS that preserves the vertical datum.
    vertical_crs = extract_vertical_crs(in_crs)
    crs_2d = out_crs.to_2d().to_string()
    MID_EPSG = f"{crs_2d}+{vertical_crs}" if vertical_crs != "" else crs_2d
    mid_crs = CRS(MID_EPSG).to_3d()

    # --- Define Output Grid ---
    shape, off = compute_shape_and_off(dem, spacing, CRS_EPSG)
    # Create the coordinate arrays for the output grid.
    x_lin, y_lin = [np.linspace(off[i], off[i] + (shape[i] - 1) * spacing[i], shape[i]) for i in ["x", "y"]]

    # --- Initialize Output ---
    # Create an empty Zarr store for the reprojected DEM.
    dem_out = save_empty_dataset(
        data_vars=["z"],
        dims={"z": ["y", "x"]},
        shape={"z": [y_lin.size, x_lin.size]},
        dtype={"z": dem_ds.z.dtype},
        chunks=chunks,
        crs=out_crs,
        coords={"y": y_lin, "x": x_lin},
        data_path=normpath(f"{data_path}/dem_resampled"),
        **kwargs,
    )

    # --- Run Reprojection in Parallel ---
    # Map the interpolation function over blocks of the output grid.
    save_options = {
        "data_path": normpath(f"{data_path}/dem_resampled"),
        "save_options": {
            "skip_read": True,
            "consolidated": False,
            "storage_options": get_storage_options(kwargs),
        },
    }
    map_blocks(
        interp_dem,
        dem_out.z,
        args=[dem_ds.z, mid_crs, method],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
        store_mapping={0: save_options},
    )
    # Read and return the final reprojected dataset.
    return read(
        normpath(f"{data_path}/dem_resampled"), as_type="xr.Dataset", storage_options=get_storage_options(kwargs)
    )


def single_dem(
    burst_id: str,
    slc_attrs: list[dict],
    multiburst: bool,
    dem: xr.Dataset,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    border: float = 10.0 / 3600.0,  # Approx. 300m border in degrees
    recompute=False,
    **kwargs,
):
    """
    Reproject a DEM for a single burst or scene.

    This worker function calculates the geographic bounding box of a single SAR
    burst, clips the input DEM to that area, and then calls `dem_reproject` to
    perform the reprojection.

    Parameters
    ----------
    burst_id
        The identifier for the burst or scene.
    slc
        The DataTree for the specific burst, used to find its footprint.
    multiburst : bool
        A boolean flag for multiburst mode.
    dem
        The full input Digital Elevation Model.
    spacing
        The desired spatial resolution of the output DEM.
    working_directory
        The root directory for outputs.
    chunks : dict
        A dictionary defining chunk sizes for processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    enable_dask : bool
        If True, use Dask for parallel computation.
    border
        A border in degrees to add to the burst's bounding box.
    **kwargs
        Additional keyword arguments for the reprojection function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the reprojected DEM Dataset.
    """
    # Define the output path for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{working_directory_tmp}/{path}")
    simulation_path = f"{working_directory_static}/processed_data/{path}/simulation.ok"
    storage_options_static = get_storage_options(kwargs, key="storage_options_static")
    # --- Check for Existing Results ---
    # Decide whether to run the computation or use a cached result.
    if exists(simulation_path, storage_options=storage_options_static) is False or recompute:
        # --- Calculate Bounding Box ---
        lat = []
        lon = []
        # Collect the corner coordinates from all original images in this burst.
        for attrs in slc_attrs:
            annot = attrs[list(attrs)[0]]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])
        # Determine the min/max lat/lon and add a border.
        lat_min, lat_max = min(lat) - border, max(lat) + border
        lon_min, lon_max = min(lon) - border, max(lon) + border

        # --- Clip and Reproject DEM ---
        # Clip the global DEM to the specific area of this burst.
        dem_aoi = dem.rio.clip_box(lon_min, lat_min, lon_max, lat_max, crs="EPSG:4326").chunk(chunks)
        # Determine the target CRS if not provided.
        if crs is None:
            crs = get_default_epsg(dem_aoi)
        # Call the main reprojection function.
        dem_resampled = dem_reproject(
            dem_aoi, spacing, crs, chunks=chunks, data_path=data_path, enable_dask=enable_dask, **kwargs
        )
    else:
        logger.info(f"Found existing DEM file for {burst_id}")
        dem_resampled = read(
            normpath(f"{working_directory_static}/{path}/simulation"),
            as_type="xr.Dataset",
            storage_options=storage_options_static,
        ).h.to_dataset(name="z")

    return (burst_id, dem_resampled)


@timeit
def stack_dem(
    dem: xr.Dataset,
    slcs: xr.DataTree,
    multiburst: bool,
    spacing: dict[str, float],
    working_directory_static: str,
    working_directory_tmp: str,
    chunks: dict = {"x": 2048, "y": 2048},
    crs: str | None = None,
    enable_dask: bool = True,
    recompute: bool = False,
    **kwargs,
):
    """
    Reproject a DEM for each burst or scene in a stack.

    This function orchestrates the reprojection of a DEM for an entire stack.
    It maps the `single_dem` function over each item in the `slcs` DataTree,
    processing each one in parallel, and then assembles the results into a new
    DataTree.

    Parameters
    ----------
    dem : xarray.DataArray
        The input Digital Elevation Model as an xarray.DataArray.
    slcs : xarray.DataTree
        An xarray.DataTree containing the SAR data, used to define the
        geographic extent for each burst.
    multiburst : bool
        A boolean flag indicating if multiburst mode is active.
    spacing
        The desired spatial resolution of the output DEMs.
    working_directory
        The root directory for all processing outputs.
    chunks : dict
        A dictionary defining the chunk sizes for Dask processing.
    crs
        The target EPSG code. If None, it will be automatically determined.
    **kwargs
        Additional keyword arguments passed to the reprojection function.

    Returns
    -------
    xarray.DataTree
        An xarray.DataTree containing the reprojected DEM for each burst or scene.
    """
    # --- Determine Target CRS ---
    if crs is None:
        # If no CRS is specified, determine the default UTM/UPS zone.
        epsg = get_default_epsg(dem)
    else:
        epsg = crs
    # Check if the required transformation is available.
    check_available_transformation(dem, epsg)

    # --- Prepare Arguments and Run in Parallel ---
    # Create a list of arguments for each burst/scene to be processed.
    args = [
        (k, [o.measurements.slc.attrs for o in s.orig.values()], multiburst)
        for k, s in (slcs.bursts if multiburst else {"data": slcs}).items()
    ]
    args = list(zip(*args))
    results = cast(
        list,
        map(
            single_dem,
            args,
            # Pass common arguments via kwargs.
            kwargs=dict(
                dem=dem,
                spacing=spacing,
                working_directory_static=working_directory_static,
                working_directory_tmp=working_directory_tmp,
                chunks=chunks,
                crs=crs,
                enable_dask=enable_dask,
                recompute=recompute,
                **kwargs,
            ),
            combine=False,
            enable_dask=enable_dask,
            Plugin=GeoEnvPlugin(in_proj=dem.rio.crs, out_proj=epsg),
        ),
    )

    # --- Assemble Results ---
    dem_tree = xr.DataTree()
    for k, d in results:
        # Reconstruct the DataTree structure from the list of results.
        if multiburst:
            dem_tree[f"bursts/{k}"] = d
        else:
            dem_tree = xr.DataTree(d)
    return dem_tree


def coreg_single_burst(
    burst_id: str,
    slc: xr.DataTree,
    simulation_reference: xr.Dataset,
    simulation_secondary: xr.Dataset | None,
    simulation_ml: dict,
    multiburst: bool,
    daz_drg: xr.Dataset | None,
    reference_date: str,
    working_directory: str,
    recompute: bool = False,
    save_daz_drg: bool = False,
    do_reramp: bool = True,
    do_flattening: bool = True,
    enable_dask: bool = True,
    **kwargs,
) -> tuple[str, xr.DataTree]:
    """
    Execute the end-to-end coregistration workflow for a single burst.

    This is the main worker function that is mapped across the Dask cluster.
    It initializes the output structure, runs the geometric warping process by
    calling `warp_dazdrg_cslc`, and then updates all the ancillary metadata
    (conditions and quality) for the newly coregistered burst.

    Parameters
    ----------
    burst_id
        The identifier for the burst being processed.
    slc
        The input DataTree for this burst, containing the `orig` data.
    simulation_reference
        The reference simulation data for this burst.
    simulation_secondary
        An optional secondary simulation dataset.
    simulation_ml
        The multilooked simulation data for this burst.
    multiburst
        A boolean flag for multiburst mode.
    reference_date
        The reference acquisition date.
    working_directory
        The root directory for all outputs.
    enable_dask
        If True, enable Dask for sub-tasks.
    **kwargs
        Additional keyword arguments for the warping function.

    Returns
    -------
    tuple
        A tuple containing the burst_id and the resulting `coreg` DataTree node.
    """
    # Define the output path for this burst.
    path = f"bursts/{burst_id}" if multiburst else "data"

    # --- Step 1: Initialize Output Structure ---
    # Create the empty 'coreg' group on disk.
    coreg_path = normpath(f"{working_directory}/{path}")
    slc = init_single_coreg_stack(
        slc,
        reference_date,
        coreg_path,
        recompute=recompute,
        **kwargs,
    )

    # --- Prepare Data Structures ---
    measurements = slc.coreg.measurements.to_dataset()
    conditions = slc.coreg.conditions
    quality = slc.coreg.quality

    # Initialize the coarse grid for the Lookup Tables.
    LUT_coords = initialize_LUT_coords(measurements)

    # --- Prepare Interpolation Kernel ---
    # Define the parameters for the sinc interpolation kernel.
    ov_fact = kwargs.get("warp_ov_fact", 100_000)
    kernel_shape = np.array(kwargs.get("warp_kernel_shape", [11, 11]))
    kaiser_beta = kwargs.get("warp_kaiser_beta", 6)

    # Pre-compute the sinc kernels for x and y directions.
    sinc_x = sinc(kernel_shape[1], ov_fact, kaiser_beta)
    sinc_y = sinc(kernel_shape[0], ov_fact, kaiser_beta)

    kernel = {
        "kernel_shape": kernel_shape,
        "ov_fact": ov_fact,
        "kaiser_beta": kaiser_beta,
        "sinc_x": sinc_x,
        "sinc_y": sinc_y,
    }

    # --- Step 2: Perform Warping in Parallel ---
    # Use `map_blocks` to apply the `warp_dazdrg_cslc` function to chunks of the data.
    chunkable_dazdrg = []
    placeholder_dazdrg = []
    if "daz" in measurements:
        compute_daz_drg = measurements.daz.attrs.get("recompute", True)
        daz_drg_store_path = measurements.daz.attrs["store_path"]
        chunkable_dazdrg = [measurements.daz, measurements.drg]
    else:
        placeholder_dazdrg = [None, None]
        compute_daz_drg = True

    store_mapping = {
        0 : {
            "data_path": os.path.join(coreg_path, "daz_drg"),
            "dims": ["acquisition_date", "azimuth_time", "slant_range_time"],
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs),
                "skip_read": True,
            }
        }
    }
    orig_slcs = {k: v.measurements.slc for k, v in slc.orig.children.items()}
    orig_etad = {}
    for k, v in slc["orig"].children.items():
        if "ETAD" in v.conditions:
            child_etad = v.conditions.ETAD.to_dataset()
            azcorr = child_etad.azcorr if "azcorr" in child_etad else None
            rgcorr = child_etad.rgcorr if "rgcorr" in child_etad else None
            orig_etad[k] = (azcorr, rgcorr)
        else:
            orig_etad[k] = (None, None)
    # --- Calculate Grid Spacing and Margins ---
    annotation = measurements.cstack.attrs
    single_pol_annotation = annotation[list(annotation.keys())[0]]
    # Calculate azimuth and range sampling intervals.
    margin = get_margin(single_pol_annotation)
    azimuth_grid_spacing = single_pol_annotation["Grid"]["Col"]["SS"] * single_pol_annotation["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1]
    range_grid_spacing = 2 * single_pol_annotation["Grid"]["Row"]["SS"] / speed_of_light
    map_blocks(
        warp_dazdrg,
        obj=[measurements.cstack, *chunkable_dazdrg],  # if available, chunk daz and drg; otherwise, pass placeholders
        args=(
            *placeholder_dazdrg,
            simulation_reference,
            simulation_secondary,
            simulation_ml,
            margin,
            measurements.coregistration_reference_date,
            azimuth_grid_spacing,
            range_grid_spacing,
        ),
        kwargs={
            "compute_daz_drg": compute_daz_drg,
        },
        combine=False,
        enable_dask=enable_dask,
        chunksizes={
            "azimuth_time": measurements.unify_chunks().chunksizes["azimuth_time"][0],
            "slant_range_time": measurements.unify_chunks().chunksizes["slant_range_time"][0],
            "acquisition_date": 10,
        },
        store_mapping=store_mapping,
    )

    daz_drg = read(
            daz_drg_store_path,
            consolidate_metadata=False,
            storage_options=get_storage_options(kwargs),
            as_type="xr.Dataset",
        )

    list_daz_drg_sel = cast(
        list,
        map_blocks(
            update_LUT,
            obj=daz_drg,
            kwargs={"LUT_coords": LUT_coords},
            combine=False,
            enable_dask=enable_dask,
        ),
    )
    
    store_mapping = {
        0: {
            "data_path": measurements.attrs["store_path"],
            "save_options": {
                "consolidated": False,
                "storage_options": get_storage_options(kwargs),
            },
        },
    }
    map_blocks(
        warp_cslc,
        obj=[
            measurements.cstack,
            measurements.phase_model,
            daz_drg.daz,
            daz_drg.drg,
            measurements.acquisition_date
        ],
        args=(
            measurements.coregistration_reference_date,
            orig_slcs,
            orig_etad,
            kernel,
        ),
        kwargs={
            "do_reramp": do_reramp,
            "do_flattening": do_flattening,
        },
        combine=False,
        enable_dask=enable_dask,
        chunksizes={
            "azimuth_time": measurements.unify_chunks().chunksizes["azimuth_time"][0],
            "slant_range_time": measurements.unify_chunks().chunksizes["slant_range_time"][0],
            "acquisition_date": 10,
        },
        store_mapping=store_mapping,
    )


    # --- Step 3: Update Metadata ---
    # Merge the list of daz/drg offsets computed by each chunk.
    LUT_daz_drg = sum([k for k in list_daz_drg_sel if k is not None])
    if not isinstance(LUT_daz_drg, xr.Dataset):
        raise Exception(f"Expected xr.Dataset, got {type(LUT_daz_drg)}")

    measurements = read(
        measurements.store_path,
        storage_options=get_storage_options(kwargs),
    )
    # Update the 'conditions' group (orbits, ETAD, LUTs) and write to disk.
    conditions = update_conditions(slc, LUT_daz_drg, conditions)
    conditions = write(
        conditions,
        normpath(f"{working_directory}/{path}/coreg/conditions"),
        mode="w",
        storage_options=get_storage_options(kwargs),
    )
    # Update the 'quality' group (accuracy, radiometric LUTs) and write to disk.
    quality = update_quality(slc, LUT_daz_drg, quality)
    quality = write(
        quality,
        normpath(f"{working_directory}/{path}/coreg/quality"),
        mode="w",
        storage_options=get_storage_options(kwargs),
    )

    if daz_drg is not None:
        measurements["daz"] = daz_drg["daz"]
        measurements["drg"] = daz_drg["drg"]

    coreg = xr.DataTree.from_dict({"measurements": measurements, "conditions": conditions, "quality": quality})
    return (burst_id, coreg)


@timeit
def stack_coregistration(
    slcs: xr.DataTree,
    simulation_reference_tree: xr.DataTree,
    simulation_ml_tree: dict,
    reference_date: str,
    working_directory: str,
    simulation_secondary_tree: xr.DataTree | None = None,
    save_daz_drg: bool = False,
    daz_drg: xr.DataTree | None = None,
    do_reramp: bool = True,
    do_flattening: bool = True,
    multiburst: bool = True,
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> xr.DataTree:
    """
    Coregister a stack of SAR images in parallel.

    This function orchestrates the coregistration of an entire stack of SLCs
    (either as multiple bursts or a single scene). It prepares the necessary
    arguments for each burst and then maps the `coreg_single_burst` function
    across a Dask cluster to process them in parallel.

    Parameters
    ----------
    slcs
        The input DataTree containing the stack of original SLC images.
    simulation_reference_tree
        A DataTree with the simulated radar geometry
        for the reference DEM.
    simulation_ml_tree
        A dictionary containing the downsampled ("multilooked")
        simulation data, used for quickly finding overlaps.
    reference_date
        The acquisition date of the reference image for the coregistration.
    working_directory
        The root directory for all processing outputs.
    simulation_secondary_tree
        An optional DataTree with simulation data for a
        secondary DEM, used for differential measurements.
    multiburst
        A boolean flag indicating if the data is in multiburst format.
    **kwargs
        Additional keyword arguments passed down to the processing functions,
        such as `warp_kernel_shape` and `warp_kaiser_beta`.

    Returns
    -------
    DataTree
        The updated `slcs` DataTree, with the newly created `coreg` nodes containing
        the coregistered stack and associated metadata.
    """
    args = []
    if daz_drg is not None and save_daz_drg is True:
        daz_drg = None
    # --- Prepare Arguments for Parallel Mapping ---
    # In multiburst mode, iterate over each burst in the DataTree.
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        s = (simulation_reference_tree.bursts[k] if multiburst else simulation_reference_tree).ds
        s2 = (
            (simulation_secondary_tree.bursts[k] if multiburst else simulation_secondary_tree).ds
            if simulation_secondary_tree is not None
            else None
        )
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        dz = (daz_drg.bursts[k] if multiburst else daz_drg).ds if daz_drg is not None else None
        a = (k, v, s, s2, s_ml, multiburst, dz)
        args.append(a)

    # Transpose the list of tuples for the map function.
    args = list(zip(*args))

    # JIT-compile the Numba interpolation function before starting the main computation.
    init_interpolate()

    # --- Execute Coregistration in Parallel ---
    # Map the `coreg_single_burst` function over the list of arguments.
    coregs = cast(
        list,
        map(
            coreg_single_burst,
            args,
            kwargs=dict(
                reference_date=reference_date,
                working_directory=working_directory,
                enable_dask=enable_dask,
                recompute=recompute,
                save_daz_drg=save_daz_drg,
                do_reramp=do_reramp,
                do_flattening=do_flattening,
                **kwargs,
            ),
            enable_dask=enable_dask,
        ),
    )

    # --- Assemble Results ---
    # Integrate the results from each worker back into the main DataTree.
    for k, coreg in coregs:
        if multiburst:
            slcs[f"bursts/{k}/coreg"] = coreg
        else:
            slcs["coreg"] = coreg

    return slcs


def map_shadow_projection(
    slc: xr.DataArray,
    data_path_static: str,
    threshold: float = 0.3,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Computes the geocoded Gamma Naught image by mapping radar data to the geogrid.

    This function orchestrates the geocoding process. It first reads global
    grid parameters, creates an empty Zarr dataset for the output geocoded
    image (`gamma0`), and then uses `map_blocks` to apply the `compute_geo_ap`
    function to each chunk in parallel (if Dask is enabled).

    Parameters
    ----------
    slc : xr.DataArray
        The input Single Look Complex (SLC) radar data,
        which will be read in chunks.
    data_path_static : str
        Path to the directory containing static data,
        such as global grid values and weights (w).
    data_path_tmp : str
        Path to the directory where temporary and final
        output Zarr stores (like 'geo_grid') are written.
    threshold : float, optional
        A threshold value passed to `compute_geo_ap`
        used for ANF normalization. Defaults to 1e-3.
    enable_dask : bool, optional
        Whether to enable Dask for parallel
        computation. Defaults to True.
    recompute : bool, optional
        A flag passed to `compute_geo_ap`, primarily
        to control mask (layover/shadow) re-computation.
        Defaults to True.
    skip_read : bool, optional
        If True, the function returns None after
        computation. If False, it reads the final
        'gamma0' dataset from disk and returns it.
        Defaults to True.
    chunksizes : dict[str, int] or None, optional
        Specific Dask chunk sizes
        to use for the computation.
        Defaults to None.

    Returns
    -------
    xr.DataArray or None
        The geocoded 'gamma0' values if `skip_read` is False,
        otherwise None.
    """
    # Load global metadata saved from the previous simulation step
    global_values = read(
        data_path=normpath(f"{data_path_static}/global_values"),
        as_type="xr.Dataset",
        storage_options=get_storage_options(kwargs),
    )
    dim_x_global = global_values.geo_xsize.values[0]
    dim_y_global = global_values.geo_ysize.values[0]
    # Get the chunk_length used during the weight computation
    chunk_length = global_values.chunk_length.values[0]

    # Define the dimensions, shape, and chunking for the output Zarr store
    dims = ["y", "x"]
    shape = [dim_y_global, dim_x_global]
    chunks = {"y": chunk_length, "x": chunk_length}

    # Create an empty Dask array with the correct shape and chunks to map over
    # Note: The +1 is because the grid coordinates define cell edges, not centers
    empty_data = xr.DataArray(da.empty(shape=(shape[0] + 1, shape[1] + 1), dtype="float32"), dims=dims).chunk(chunks)

    # Get the final computed chunk structure
    chunks = compute_chunks(dict(empty_data.chunksizes))

    # Define the arguments to be passed to the worker function `compute_geocoded_mask`
    args = [chunk_length, slc.shape[-1], data_path_static, threshold]

    def region_callback(region: Mapping[str, slice], kwargs: dict) -> Mapping[str, slice]:
        out = {}
        for k, v in region.items():
            if k in ["y", "x"]:
                out[k] = slice(v.start, v.stop - 1)
            else:
                raise ValueError(f"Unexpected dimension {k}")
        return out

    save_options = {
        "data_path": f"{data_path_static}/geo_grid",
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs),
        }
    }

    # Map the `compute_geo_ap` function over all chunks of the `empty_data`
    map_blocks(
        func=compute_geocoded_mask,
        obj=empty_data,
        args=args,
        kwargs=kwargs,
        chunks=chunks,
        enable_dask=enable_dask,
        combine=False,
        region_callback={0: region_callback},
        store_mapping={0: save_options}
    )

    return


def map_shadow(
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    data_path: str,
    eps: float,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Maps the shadow computation over the radar grid chunks.

    This function sets up an empty Dask array matching the radar grid and
    maps the `compute_shadow` function over its chunks.

    Parameters
    ----------
    slc : xr.DataArray
        The reference SLC DataArray (used for metadata
        and dimensions).
    simulation : xr.Dataset
        The radar simulation dataset.
    simulation_ml : dict
        The multi-looked simulation data.
    data_path : str
        Path to the directory where results will be saved
        and global values are read from.
    eps : float
        Epsilon value passed to `compute_shadow`.
    enable_dask : bool, optional
        Whether to enable Dask for parallel
        computation. Defaults to True.

    Returns
    -------
    None
        This function saves the shadow mask to disk.
    """

    # Load global values (chunk sizes, bounding box) from the static path
    # Extract annotation metadata from the SLC
    annot = slc.attrs[list(slc.attrs.keys())[0]]

    # Calculate time differentials for azimuth and range
    dt = {
        "azimuth_time": annot["Grid"]["Col"]["SS"] * annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1],
        "slant_range_time": 2 * annot["Grid"]["Row"]["SS"] / speed_of_light,
    }
    # Calculate time steps (used for margins)
    margin = get_margin(annot)
    # Compile Numba functions in griddata
    init_interpolate()

    # Create an empty Dask array with the shape and chunks of the radar grid
    if "polarization" in slc.dims:
        radar_data = slc.isel(acquisition_date=0, polarization=0)
    else:
        radar_data = slc
    # Define arguments for the worker function
    local_incidence_angle = read(
        f"{data_path}/geo_grid", consolidate_metadata=False, storage_options=get_storage_options(kwargs)
    ).lia
    simulation_h = simulation.h
    simulation_dem_distance = simulation.dem_distance
    simulation_azimuth_time = simulation.azimuth_time
    simulation_slant_range_time = simulation.slant_range_time

    # Map the `compute_shadow` function over all chunks
    save_options = {
        "data_path": normpath(f"{data_path}/radar_grid"),
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs, "storage_options"),
        }
    }
    map_blocks(
        func=compute_shadow,
        obj=radar_data,
        args = [
            simulation_h,
            simulation_dem_distance,
            simulation_azimuth_time,
            simulation_slant_range_time,
            simulation_ml,
            local_incidence_angle,
            eps,
            dt,
            margin,
        ],
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,  # Results are written to disk
        store_mapping={0: save_options},
    )
    return




def mask_shadow_single_burst(
    burst_id: str,
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    wdir_path_static: str,
    eps: float = 1e-4,
    threshold: float = 0.3,
    enable_dask: bool = True,
    recompute: bool = False,
    **kwargs,
) -> None:
    """
    Computes and saves the radar-grid shadow mask for a single burst.

    This function orchestrates the shadow computation by calling `map_shadow`.
    It checks for a completion file ('mask_radar.ok') to skip computation
    if `recompute` is False.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst (e.g., "IW1_001").
    slc : xr.DataArray
        The reference Single Look Complex (SLC) data array
        for this burst.
    simulation : xr.Dataset
        The radar simulation dataset corresponding
        to this burst.
    simulation_ml : dict
        The multi-looked simulation data.
    multiburst : bool
        Flag indicating if processing is per-burst.
    wdir_path_static : str
        Path to the static working directory.
    eps : float, optional
        Epsilon value used in shadow computation.
        Defaults to 1e-4.
    threshold : float, optional
        Threshold value used in geocoded mask computation.
        Defaults to 0.3.
    enable_dask : bool, optional
        Flag to enable Dask for sub-tasks.
        Defaults to True.
    recompute : bool, optional
        Flag to force re-computation.
        Defaults to False.

    Returns
    -------
    None
        This function saves the shadow mask to disk.
    """
    # Define the data path based on whether it's multiburst or single data mode
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path = normpath(f"{wdir_path_static}/{path}")

    # Define the path for the completion file
    mask_path = f"{wdir_path_static}/processed_data/{path}/mask_radar.ok"

    # Check if computation should be skipped
    if (exists(mask_path, **kwargs) is False) or recompute:
        # If recomputing, delete the old completion file
        delete(mask_path, allow_non_existent=True, **kwargs)

        # Call the mapping function to compute the shadow
        map_shadow(
            slc=slc,
            simulation=simulation,
            simulation_ml=simulation_ml,
            data_path=data_path,
            eps=eps,
            enable_dask=enable_dask,
            **kwargs,
        )
        map_shadow_projection(
            slc=slc,
            data_path_static=data_path,
            threshold=threshold,
            enable_dask=enable_dask,
            **kwargs,
        )
        # Create the 'processed_data' directory if it doesn't exist
        makedirs(normpath(f"{wdir_path_static}/processed_data/{path}"), exist_ok=True, **kwargs)
        # "Touch" the completion file to signify success
        touch(mask_path, **kwargs)
    else:
        # Skip computation if completion file exists and recompute is False
        logger.info(f"Found existing mask files for burst {burst_id}")
    return




@timeit
def stack_mask(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict[str, xr.Dataset],
    working_directory_static: str,
    reference_date: str,
    multiburst: bool = True,
    recompute: bool = False,
    **kwargs,
) -> None:
    """
    Orchestrates the shadow mask computation for a stack of SLCs.

    This function prepares arguments for each burst (or a single data block)
    and maps the `mask_shadow_single_burst` function across them.

    Parameters
    ----------
    slcs : xr.DataTree
        An xarray DataTree containing the SLC burst data.
    simulation_tree : xr.Dataset
        An xarray Dataset (or DataTree-like
        structure) containing the simulation results.
    simulation_ml_tree : dict[str, xr.Dataset]
        A dictionary containing the
        multi-looked simulation results.
    working_directory_static : str
        Path to the static working directory
        for saving persistent results.
    reference_date : str
        The acquisition date string used to select the
        reference SLC.
    multiburst : bool, optional
        Flag indicating whether to process data as
        individual bursts. Defaults to True.
    recompute : bool, optional
        Flag to force re-computation even if
        completion files exist. Defaults to False.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    args = []
    # Prepare arguments for each burst if in multiburst mode
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.orig[reference_date].measurements.slc
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        in_proj = s.rio.crs
        args.append(a)

    # Transpose the list of tuples for the map function
    args = list(zip(*args))

    # Map the `mask_shadow_single_burst` function across all argument sets
    map(
        mask_shadow_single_burst,
        args,
        kwargs=dict(wdir_path_static=working_directory_static, recompute=recompute, **kwargs),
        combine=False,  # Results are written to disk
        enable_dask=kwargs.get("enable_dask", True),
        Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
    )

    return


def map_rasterization_and_rtc_ap(
    simulation: xr.Dataset,
    global_values: xr.Dataset,
    data_path_static: str,
    data_path_tmp: str,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Orchestrate the rasterization and Radiometric Terrain Correction (RTC) using area projection.

    This function manages the parallel execution of the core rasterization and RTC
    algorithms. It reads global parameters, computes overlapping chunks, initializes
    output datasets, and then maps the `compute_rasterization_and_rtc_ap` worker
    function over blocks of the simulation data. Finally, it triggers the conversion
    of sparse results back to dense output grids.

    Parameters
    ----------
    simulation : xr.Dataset
        The simulation dataset containing radar geometry mapped
        onto the geographic grid.
    data_path_static : str
        Path to the directory for storing persistent static outputs
        (like the final `radar_grid` and `geo_grid`).
    data_path_tmp : str
        Path to the directory for storing temporary intermediate
        files (like sparse matrices).
    enable_dask : bool, optional
        If True, use Dask for parallel computation. Defaults to True.

    Returns
    -------
    None
        Results are written to Zarr stores specified by the paths.
    """
    # --- Clean Up Previous Results ---
    # Delete temporary sparse matrix directory and old weight matrix if they exist.
    for f, t in zip([f"{data_path_static}/w", f"{data_path_tmp}/sparse_dir"], ["static", "dynamic"]):
        try:
            delete(
                path=normpath(f),
                recursive=True,
                allow_non_existent=True,
                storage_options=get_storage_options(kwargs, key=f"storage_options_{t}"),
            )
        except botocore.exceptions.NoCredentialsError as e:
            raise PermissionError(
                f"S3 credentials not found or invalid. Cannot access {f}. "
                "Please provide 'storage_options' or set environment variables (AWS_ACCESS_KEY_ID, etc.)."
            ) from e
    # --- Load Global Parameters ---
    dim_x_global, dim_y_global = global_values.geo_xsize.values[0], global_values.geo_ysize.values[0]
    xmin_global, ymin_global = global_values.radar_xmin.values[0], global_values.radar_ymin.values[0]
    xsize_global, chunk_length = global_values.radar_xsize.values[0], global_values.chunk_length.values[0]
    radar_grid_pixel_area, bbox = global_values.radar_grid_pixel_area.values[0], global_values.bbox.values
    chunks_slc = global_values.chunks_slc.values
    az = global_values.az_coords.values
    rg = global_values.rg_coords.values
    crs = global_values.rio.crs

    # --- Prepare for Parallel Execution ---
    # Compute overlapping chunks for the simulation grid.
    chunks, x, y = compute_chunks(dict(simulation.chunksizes)), simulation.x.values, simulation.y.values
    # Determine winding order based on grid orientation.
    y_sign, x_sign = (1 if y[0] < y[1] else 0, 1 if x[0] < x[1] else 0)
    # Define output coordinates at pixel centers.
    normal_order, x, y = np.logical_xor(y_sign, x_sign), (x[0:-1] + x[1:]) / 2, (y[0:-1] + y[1:]) / 2
    # --- Initialize Output Datasets ---
    # Initialize the empty Zarr store for the final geographic grid outputs (lia, anf, nlooks, masks).
    data_vars = ["lia", "anf", "nlooks", "mask"]

    save_empty_dataset(
        data_vars=data_vars,
        dims={k: ["y", "x"] for k in data_vars},
        coords={"y": y, "x": x},
        crs=crs,
        shape={k: [dim_y_global, dim_x_global] for k in data_vars},
        chunks={"y": chunk_length, "x": chunk_length},
        dtype={k: "float32" if k not in ["mask"] else "int32" for k in data_vars},
        skip_read=True,
        data_path=f"{data_path_static}/geo_grid",
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )
    # Get the resolution in meters for RTC calculation.
    resolution = get_resolution_in_meters(simulation)

    # --- Map Rasterization and RTC Worker ---
    # Apply the core worker function `compute_rasterization_and_rtc_ap` to each chunk.
    # Write the geogrid dataset to the static path, placing it in the correct region
    store_mapping = {
        0: {
            "data_path": normpath(f"{data_path_static}/geo_grid"),
            "dims": ["y", "x"],
            "save_options": {
                "skip_read": True,
                "consolidated": False,
                "storage_options": get_storage_options(kwargs, "storage_options_static"),
            },
        },
        1: {
            "data_path": normpath(f"{data_path_static}/w"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": True,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_static"),
            }
        },
        2: {
            "data_path": normpath(f"{data_path_tmp}/sparse_dir"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": True,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_dynamic"),
            },
        },
        3: {
            "data_path": normpath(f"{data_path_tmp}/sparse_dir/A_block"),
            "drop_coords": False,
            "drop_attrs": False,
            "save_options": {
                "consolidated": True,
                "mode": "a",
                "skip_read": True,
                "storage_options": get_storage_options(kwargs, "storage_options_dynamic"),
            },
        }
    }
    def output_callback(region: dict[str, slice], kwargs: dict) -> dict[str, slice] | None:
        # Determine the chunk's row and column position in the global grid using the job_id
        row_number = kwargs['job_id'] // (np.ceil(dim_x_global / chunk_length).astype(int))
        column_number = kwargs['job_id'] % (np.ceil(dim_x_global / chunk_length).astype(int))

        # Define the geogrid region (slice) this chunk corresponds to, for writing output
        region["y"] = slice(row_number * chunk_length, row_number * chunk_length + region["y"].stop - region["y"].start - 1)
        region["x"] = slice(column_number * chunk_length, column_number * chunk_length + region["x"].stop - region["x"].start - 1)
        return region

    region_callback = {
        0: output_callback,
        1: lambda _, __ : None,
        2: lambda _, __ : None,
        3: lambda _, __ : None,
    }
    map_blocks(
        func=compute_rasterization_and_rtc_ap,
        obj=simulation,
        args=[
            xmin_global,
            ymin_global,
            xsize_global,
            dim_x_global,
            radar_grid_pixel_area,
            normal_order,
            bbox,
            chunk_length,
            resolution,
        ],
        chunks=chunks,
        enable_dask=enable_dask,
        combine=False,
        store_mapping=store_mapping,
        region_callback=region_callback,
    )

    # --- Convert Sparse Results to Dense ---
    # Find all the intermediate sparse matrix files generated by the workers.
    files_A = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/A_block/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_l1 = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/layover_block_0/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_l2 = glob(
        path=normpath(f"{data_path_tmp}/sparse_dir/layover_block_1/*"),
        storage_options=get_storage_options(kwargs, key="storage_options_dynamic"),
    )
    files_A = [file for file in files_A if not file.endswith(".json")]
    files_l1 = [file for file in files_l1 if not file.endswith(".json")]
    files_l2 = [file for file in files_l2 if not file.endswith(".json")]
    # Initialize the empty Zarr store for the final dense radar grid outputs (anf, masks).
    data_vars = ["anf", "layover", "shadow"]
    data_path = f"{data_path_static}/radar_grid"
    A = save_empty_dataset(
        data_vars=data_vars,
        dims={k: ["azimuth_time", "slant_range_time"] for k in data_vars},
        shape={k: [bbox[3], bbox[2]] for k in data_vars},  # Use full radar grid dimensions
        coords={"azimuth_time": az, "slant_range_time": rg},
        chunks={"azimuth_time": chunks_slc[0], "slant_range_time": chunks_slc[1]},
        dtype={k: "bool" if k != "anf" else "float32" for k in data_vars},
        skip_read=False,
        data_path=data_path,
        storage_options=get_storage_options(kwargs, key="storage_options_static"),
    )

    # Map the `sparse_to_dense` worker function over chunks of the empty output grid `A`.
    save_options = {
        "data_path": data_path,
        "save_options": {
            "consolidated": False,
            "skip_read": True,
            "storage_options": get_storage_options(kwargs, "storage_options_static"),
        },
    }
    map_blocks(
        sparse_to_dense,
        A,
        args=(files_A, files_l1, files_l2, data_path_static),
        kwargs=kwargs,
        enable_dask=enable_dask,
        combine=False,
        store_mapping={0: save_options},
    )
    return


def single_burst_geometry(
    burst_id: str,
    slc: xr.DataArray,
    simulation: xr.Dataset,
    simulation_ml: dict,
    multiburst: bool,
    chunk_length: int,
    wdir_path_static: str,
    wdir_path_tmp: str,
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Runs the geometry computation (rasterization and RTC-AP) for a single
    burst or data block.

    This function initializes the simulation, runs the
    `map_rasterization_and_rtc_ap` process, and writes a 'area_projection.ok'
    file to mark completion.

    Parameters
    ----------
    burst_id : str
        The identifier for the burst (e.g., "IW1_001").
    slc : xr.DataArray
        The reference Single Look Complex (SLC) data array
        for this burst.
    simulation : xr.Dataset
        The radar simulation dataset corresponding
        to this burst.
    simulation_ml : dict
        The multi-looked simulation data.
    multiburst : bool
        Flag indicating if processing is per-burst.
    chunk_length : int
        The edge size for geogrid chunks.
    wdir_path_static : str
        Path to the static working directory.
    wdir_path_tmp : str
        Path to the temporary working directory.
    recompute : bool, optional
        Flag to force re-computation.
        Defaults to False.
    enable_dask : bool, optional
        Flag to enable Dask for sub-tasks
        (passed to `map_rasterization_and_rtc_ap`).
        Defaults to True.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    # Type check for the SLC input
    if not isinstance(slc, xr.DataArray):
        raise Exception(f"Expected xr.DataArray, got {type(slc)}")

    # Define paths based on whether it's multiburst or single data mode
    path = f"bursts/{burst_id}" if multiburst else "data"
    data_path_static = normpath(f"{wdir_path_static}/{path}")
    data_path_tmp = normpath(f"{wdir_path_tmp}/{path}")

    # Define the path for the completion file
    area_projection_path = normpath(f"{wdir_path_static}/processed_data/{path}/area_projection.ok")

    # Check if computation should be skipped
    if (
        exists(area_projection_path, storage_options=get_storage_options(kwargs, key="storage_options_static")) is False
    ) or recompute:
        # If recomputing, delete the old completion file
        delete(
            area_projection_path,
            allow_non_existent=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )

        # Initialize the simulation to get the relevant chunk and global values
        selection, global_values = initialize_simulation(
            slc=slc,
            simulation=simulation,
            simulation_ml=simulation_ml,
            chunk_length=chunk_length,
        )

        # Write the global values (grid dims, CRS, etc.) to disk
        write(
            global_values,
            normpath(f"{data_path_static}/global_values"),
            mode="w",
            skip_read=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )

        # Run the main rasterization and RTC-AP mapping process
        map_rasterization_and_rtc_ap(
            selection, global_values, data_path_static, data_path_tmp, enable_dask=enable_dask, **kwargs
        )

        # Create the 'processed_data' directory
        makedirs(
            normpath(f"{wdir_path_static}/processed_data/{path}"),
            exist_ok=True,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )
        # "Touch" the completion file to signify success
        touch(
            area_projection_path,
            storage_options=get_storage_options(kwargs, key="storage_options_static"),
        )
    else:
        # Skip computation if completion file exists and recompute is False
        logger.info(f"Found existing area_projection files for burst {burst_id}")
    return


def stack_geometry(
    slcs: xr.DataTree,
    simulation_tree: xr.Dataset,
    simulation_ml_tree: dict[str, xr.Dataset],
    working_directory_static: str,
    working_directory_tmp: str,
    reference_date: str,
    chunk_length: int = 1024,
    multiburst: bool = True,
    recompute: bool = False,
    enable_dask: bool = True,
    **kwargs,
) -> None:
    """
    Orchestrates the geometry calculation (rasterization and RTC-AP) for a
    stack of SLCs, processing either individual bursts or a merged dataset.

    This function prepares and maps the `single_burst_geometry` function
    across all bursts (if `multiburst` is True) or a single data block.

    Parameters
    ----------
    slcs : xr.DataTree
        An xarray DataTree containing the SLC burst data.
    simulation_tree : xr.Dataset
        An xarray Dataset (or DataTree-like
        structure accessed within) containing
        the simulation results corresponding
        to the SLCs.
    simulation_ml_tree : dict[str, xr.Dataset]
        A dictionary containing the
        multi-looked simulation results,
        keyed similarly to `slcs`
        (e.g., by 'bursts').
    working_directory_static : str
        Path to the static working directory
        for saving persistent results
        (e.g., weights, final geo_grid).
    working_directory_tmp : str
        Path to the temporary working directory
        for intermediate files.
    reference_date : str
        The acquisition date string used to select the
        reference SLC for geometry calculation.
    chunk_length : int, optional
        The edge size for geogrid chunks.
        Defaults to 1024.
    multiburst : bool, optional
        Flag indicating whether to process data as
        individual bursts. Defaults to True.
    recompute : bool, optional
        Flag to force re-computation even if
        intermediate files exist. Defaults to False.

    Returns
    -------
    None
        This function executes the computation and saves results to disk.
    """
    args = []
    # Prepare arguments for each burst if in multiburst mode
    for k, v in (slcs.bursts if multiburst else {"data": slcs}).items():
        v = v.orig[reference_date].measurements.slc
        s = (simulation_tree.bursts[k] if multiburst else simulation_tree).to_dataset()
        s_ml = simulation_ml_tree["bursts"][k] if multiburst else simulation_ml_tree
        a = (k, v, s, s_ml, multiburst)
        args.append(a)
        in_proj = s.rio.crs

    # Transpose the list of tuples for the map function
    args = list(zip(*args))

    # Map the `single_burst_geometry` function across all argument sets
    map(
        single_burst_geometry,
        args,
        kwargs=dict(
            chunk_length=chunk_length,
            wdir_path_static=working_directory_static,
            wdir_path_tmp=working_directory_tmp,
            recompute=recompute,
            enable_dask=enable_dask,
            **kwargs,
        ),
        Plugin=GeoEnvPlugin(in_proj=in_proj, out_proj="EPSG:4978"),
        combine=False,  # Results are written to disk, not combined
        enable_dask=enable_dask,
    )

    return