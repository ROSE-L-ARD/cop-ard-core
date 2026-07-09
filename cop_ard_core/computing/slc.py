# Copyright 2024 European Union
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from collections import defaultdict
from typing import Any, Dict, Sequence, Union, cast
from unittest import mock

import dateutil.parser
import numpy as np
import sarpy.io
import xarray as xr
from scipy.constants import speed_of_light

import cop_ard_core.readers  # register sarpy readers
from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.io import (
    Path,
    exists,
    generate_empty_dataset,
    get_storage_options,
    is_s3_path,
    makedirs,
    normpath,
    open,
    path_to_dict,
    read,
    save_empty_dataset,
    touch,
    write,
)
from cop_ard_core.computing.logger import logger, timeit

cop_ard_core.readers.__file__


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


def get_coefficients(path: str, annot: dict) -> np.ndarray | None:
    """
    Utility function to extract polynomial coefficients from the annotation dictionary.

    Args:
        annot: The annotation dictionary containing the metadata.
        path: The dot-separated path to the desired coefficients (e.g., "Radiometric.SigmaZeroSFPoly.Coefs").
    Returns:
        A numpy array of coefficients if found, otherwise None.
    """
    try:
        coeffs = path_to_dict(annot, path)
        if coeffs is not None:
            return np.array(coeffs)
    except KeyError:
        return None
    return None


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
    chunk=None,
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


def merge_dict(res: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.
    """
    merged_dict = {}
    # Iterate through the list of dictionaries.
    for d in res:
        # Update the merged dictionary with the items from the current dictionary.
        for key, value in d.items():
            merged_dict[key] = value
    return merged_dict


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


class SentinelSarpyObj(object):
    """
    A wrapper class to handle Sentinel-1 metadata extraction using Sarpy,
    especially for data stored on S3 where partial file reads are needed.
    """

    def __init__(self, in_path, storage_options={}):
        # Check if the input is already a DataTree.
        self._is_dt = isinstance(in_path, xr.DataTree)
        if not self._is_dt:
            # For file paths, set up original path and a temporary working directory.
            self._input = Path(in_path, storage_options=storage_options)
        else:
            # For DataTree input, use it directly.
            self._input = in_path
        # Import the metadata upon initialization.
        self._import_metadata()

    def _import_metadata(self):
        """
        Import metadata from the Sentinel-1 product.
        """
        import sarpy.io.complex.sentinel

        if self._is_dt:
            # If input is a DataTree, Sarpy can read its metadata directly.
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(self._input)
        else:
            # For file paths (especially S3), copy essential metadata files locally.
            for j in ["manifest.safe", "annotation"]:
                self._input.append(j).ensure_local()
            # Open the local manifest file with Sarpy.
            manifest_path = self._input.append("manifest.safe").replace_dir(self._input._temp_dir.name)._path
            self._sentinel_details = sarpy.io.complex.sentinel.SentinelDetails(manifest_path)

        # Extract the collection of SICD (Sensor Independent Complex Data) metadata objects.
        try:
            self.sicd_meta = [j for k, v in self._sentinel_details.get_sicd_collection() for j in v]
        except Exception as e:
            logger.info(f"{e.__class__.__name__}: No bursts have been found.")
            self.sicd_meta = [v for k, v in self._sentinel_details.get_sicd_collection()]


def apply_invalid_mask(slc: xr.DataArray, valid_y: np.ndarray, valid_x: np.ndarray, first_idx_y: int = 0, first_idx_x: int = 0) -> xr.DataArray:
    """
    Apply a mask to the SLC data to zero out invalid regions.
    """
    # Adjust coordinates to be relative to the origin.
    valid_y = valid_y - first_idx_y
    valid_x = valid_x - first_idx_x

    # Fill the top and bottom invalid slices with zeros.
    slc[:, :, np.min((valid_y[1], valid_y[2])) :, :] = slc[:, :, : np.max((0, valid_y[0], valid_y[3])), :] = 0j

    # Fill the left and right invalid slices with zeros.
    slc[:, :, :, np.min((valid_x[2], valid_x[3])) :] = slc[:, :, :, : np.max((0, valid_x[0], valid_x[1]))] = 0j

    return slc


def amplitude_calibration(slc: xr.DataArray, calibration_polynomial_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Apply radiometric calibration to raw DN.

    This function uses a polynomial to convert the
    complex pixel values into a calibrated radiometric quantity, such as radar
    backscatter coefficient (Sigma Nought).

    Args:
        slc: The input SAR image as an xarray DataArray.
        calibration_polynomial_coeffs: The polynomial coefficients for the calibration.

    Returns:
        The calibrated SAR image.
    """
    # Check if radiometric information is available in the metadata.
    if calibration_polynomial_coeffs is None:
        logger.warning("No radiometric calibration information provided: image will be imported as is")
        return slc
    # Evaluate the 2D polynomial on the image grid to get the calibration factor.
    calib_factor = np.sqrt(
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, calibration_polynomial_coeffs)
    ).T

    # Apply the calibration factor to the complex data.
    slc *= xr.DataArray(
        calib_factor,
        dims=["azimuth_time", "slant_range_time"],
    )

    return slc


def noise_removal(slc: xr.DataArray, noise_polynomial_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Remove thermal noise from the SAR image.

    This function subtracts the estimated thermal noise power from the image
    intensity. The noise level is modeled as a 2D polynomial, with coefficients
    provided in the image metadata.

    Args:
        slc: The input SAR image as an xarray DataArray.
        noise_polynomial_coeffs: The polynomial coefficients for the noise level.

    Returns:
        The noise-corrected SAR image.
    """
    # Check if noise information is available.
    if noise_polynomial_coeffs is None:
        logger.warning("No radiometric noise level information provided: image will be imported as is")
        return slc
    # Extract the polynomial coefficients for the noise level.

    # Evaluate the polynomial to get the noise power across the image.
    noise_power = xr.DataArray(
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, noise_polynomial_coeffs).T,
        dims=["azimuth_time", "slant_range_time"],
    )

    # The calibration polynomial is typically in dB, so convert to linear power.
    noise_power = 10 ** (noise_power / 10)

    # Subtract noise power from the image intensity.
    intensity = cast(xr.DataArray, np.abs(slc) ** 2)
    intensity -= noise_power
    # Set any negative intensity values to a small positive number to avoid errors.
    intensity = intensity.where(intensity >= 0, 1e-9)
    # Reconstruct the complex image from the noise-corrected intensity and original phase.
    slc = np.sqrt(intensity) * np.exp(1j * np.angle(slc))
    return slc


def image_deramp(slc: xr.DataArray, polynomial_dkcoa_coeffs: np.ndarray | None) -> xr.DataArray:
    """
    Apply a phase deramping correction to the SAR image.

    This function removes a phase ramp caused by the Doppler frequency variation
    across the scene. The ramp is modeled as the integral of the Doppler Centroid
    frequency, which is itself a 2D polynomial provided in the metadata.

    Args:
        slc: The input SAR image as an xarray DataArray.
        polynomial_dkcoa_coeffs: The polynomial coefficients for the Doppler Centroid frequency.

    Returns:
        The deramped SAR image.
    """
    if polynomial_dkcoa_coeffs is None:
        logger.warning("No Doppler Centroid information provided: skipping deramping step")
        return slc

    # Extract the Doppler Centroid polynomial coefficients and integrate them to get the phase polynomial.
    coeff = (2 * np.pi) * np.polynomial.polynomial.polyint(polynomial_dkcoa_coeffs, axis=1)

    # Evaluate the phase polynomial on the image grid to get the phase ramp.
    phderamp = xr.DataArray(
        np.polynomial.polynomial.polygrid2d(slc.slant_range_m.values, slc.azimuth_m.values, coeff).T,
        dims=["azimuth_time", "slant_range_time"],
    )

    # Apply the deramping by multiplying with the complex conjugate of the phase ramp.
    return slc * np.exp(-1j * phderamp)


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
                ds = sarpy.io.open(in_path)
        # Handle standard file paths.
        else:
            ds = sarpy.io.open(in_path)
    except sarpy.io.general.base.SarpyIOError:
        # If standard open fails, it might be an S3 path that Sarpy can't handle directly.
        if not isinstance(in_path, str):
            raise ValueError(f"Expected str, got {type(in_path)}")
        try:
            # Try opening the path with fsspec first.
            ds = sarpy.io.open(open(in_path))
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
                swath, burstid = get_burst_id_safe(
                    ds._sentinel_details._get_file_sets(),
                    annot,
                    get_relative_orbit_number(in_path, storage_options=storage_options),
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


def map_index_to_group(dt, swath, burstid, pol):
    """
    Find the corresponding group in a DataTree based on swath, burst ID, and polarization.
    """
    # Iterate through the DataTree nodes.
    for k, v in dt.items():
        # Check if the node's metadata matches the search criteria.
        if swath is not None and burstid is not None:
            cond = (
                (
                    v.other_metadata["general_annotation"]["downlink_information_list"][0]["swath"].upper()
                    == swath.upper()
                )
                and (v.other_metadata["burst_id"] == str(burstid))
                and (v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol)
            )
        else:
            cond = v.attrs["stac_discovery"]["properties"]["sar:polarization"] == pol

        if cond:
            return k
    # Raise an error if no match is found.
    raise Exception(f"No correspondence found for swath={swath} burst_id={burstid} pol={pol}")


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


def extract_orbits(
    slcs: xr.DataTree,
    acquisition_date: list[str] | None = None,
    multiburst: bool = False,
    degrees: int = 11,
) -> xr.DataArray:
    """
    Extract and consolidate satellite orbit polynomial coefficients from metadata.
    """
    orbits_datatree = xr.DataTree()
    orbits, reference_times = {}, {}

    # --- Collect Orbit Data ---
    # Iterate through all scenes (bursts or single images) to collect orbit polynomials.
    for burst_id, i in (slcs.bursts if multiburst else {"data": slcs}).items():
        for data_id, j in i.orig.items():
            annot = j.measurements.slc.attrs[list(j.measurements.slc.attrs.keys())[0]]
            orbits.update({data_id: annot["Position"]["ARPPoly"]})
            reference_times.update(
                {data_id: annot["Timeline"]["CollectStart"]},
            )
            # Prioritize pre-computed orbit data if available.
            if "orbit" in j.conditions:
                orbits_datatree[data_id] = j.conditions.orbit.coeffs.sel(acquisition_date=data_id)

    # Filter by specific acquisition dates if provided.
    if acquisition_date is not None:
        orbits = {k: v for k, v in orbits.items() if k in acquisition_date}
        reference_times = {k: v for k, v in reference_times.items() if k in acquisition_date}

    # --- Consolidate Coefficients ---
    # Initialize an array to hold the coefficients.
    coeffs = np.zeros((len(orbits), degrees, 3))
    for counter, (k, v) in enumerate(orbits.items()):
        # Use pre-computed orbit data if available.
        if k in orbits_datatree:
            data = orbits_datatree[k].values
        # Otherwise, extract from the raw metadata.
        else:
            data = np.array([v[e]["Coefs"] for e in ["X", "Y", "Z"]]).T
        if not isinstance(data, np.ndarray):
            raise Exception(f"Expected np.ndarray, got {type(data)}")
        shape = data.shape
        # Place the coefficients into the consolidated array.
        coeffs[counter][: shape[0], : shape[1]] += data

    # --- Create Final DataArray ---
    orbits = xr.DataArray(
        data=coeffs,
        dims=["acquisition_date", "degree", "axis"],
        coords={
            "acquisition_date": list(orbits),
            "degree": np.arange(0, degrees, 1, dtype=int),
            "axis": np.arange(0, 3, 1, dtype=int),
        },
    ).assign_attrs({"reference_times": reference_times})

    return orbits


def extract_acquisition_dates(
    slcs: xr.DataTree,
    multiburst: bool = False,
) -> list[str]:
    """
    Extract a unique, sorted list of acquisition dates from the DataTree.
    """
    acquisition_dates = set()

    # Iterate through all original data nodes.
    for i in (slcs.bursts if multiburst else {"data": slcs}).values():
        for j in i.orig.values():
            # Get the acquisition date from the coordinates.
            k = j.measurements.acquisition_date.values[0]
            # Use a set to automatically handle uniqueness.
            acquisition_dates.add(k)

    return list(acquisition_dates)


def intersect(polygon1, polygon2):
    """
    Return True if the two input polygons intersect one another.
    Polygons can be passed either as a list of coordinates or as a WKT format string.

    Parameters
    ----------
    polygon1 : str or list
        The first polygon, either as a WKT string or a list of coordinates.
    polygon2 : str or list
        The second polygon, either as a WKT string or a list of coordinates.

    Returns
    -------
    bool
        True if the polygons intersect, False otherwise.
    """
    import shapely.geometry
    import shapely.wkt

    # Load the first polygon from WKT or a list of coordinates.
    try:
        poly1 = shapely.wkt.loads(polygon1)
    except TypeError:
        poly1 = shapely.geometry.Polygon(polygon1)

    # Load the second polygon.
    try:
        poly2 = shapely.wkt.loads(polygon2)
    except TypeError:
        poly2 = shapely.geometry.Polygon(polygon2)

    # Return the result of the intersection test.
    return poly2.intersects(poly1)


def get_footprint(ds, index: int):
    """
    Return the footprint coordinates of an image.

    The footprint of the indexed element (e.g., a single burst) will be returned.

    Parameters
    ----------
    ds : sarpy.io.general.base.SICDTypeReader
        The input data opened by sarpy.io.
    index : int
        Index of the image element whose footprint has to be returned.

    Returns
    -------
    list
        The footprint coordinates.
    """

    # Extract the corner coordinates for all images from the metadata.
    corners = {
        i: [
            (ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lon, ds.sicd_meta[i].GeoData.ImageCorners[j % 4].Lat)
            for j in range(5)
        ]
        for i in range(len(ds.sicd_meta))
    }

    #  return the footprint of that specific image.
    return corners[index]


def get_bbox(slcs, reference_date: str | None = None):
    """
    Get the bounding box of the SAR data.

    Parameters
    ----------
    slcs : xr.DataTree
        DataTree of SAR data.
    reference_date : str, optional
        Reference date for selecting specific data.

    Returns
    -------
    list
        Bounding box [min_lon, min_lat, max_lon, max_lat].
    """
    lat, lon = [], []

    # Iterate through all scenes.
    for slc in (slcs.bursts if "bursts" in slcs else {"data": slcs}).values():
        # Optionally filter by a reference date.
        for date in slc.orig if reference_date is None else [reference_date]:
            # Extract corner coordinates from the metadata.
            annot = slc.orig[date].measurements.slc.attrs
            annot = annot[list(annot.keys())[0]]
            for v in annot["GeoData"]["ImageCorners"]:
                lat.append(v["Lat"])
                lon.append(v["Lon"])

    # Calculate the min/max latitude and longitude, rounding to one decimal place.
    return [
        np.floor(np.min(lon) * 10) / 10,
        np.floor(np.min(lat) * 10) / 10,
        np.ceil(np.max(lon) * 10) / 10,
        np.ceil(np.max(lat) * 10) / 10,
    ]


def get_acquisition_date(or_f: Sequence[str]):
    """
    Get the acquisition dates of SAR data files.

    Parameters
    ----------
    or_f : Sequence[str]
        List of paths to SAR data files.

    Returns
    -------
    list
        List of acquisition dates.
    """
    dates = []
    for i in or_f:
        # Open the file with Sarpy to read metadata.
        ds = sarpy_open(i)
        # Parse the 'CollectStart' timestamp to get the date.
        dates.append(dateutil.parser.parse(ds.sicd_meta[0].to_dict()["Timeline"]["CollectStart"]).date())
    return dates


def get_burst_id(annot) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        annot: An annotation object containing metadata about the collection.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """
    # Get the burst number and swath identifier from the metadata parameters.
    burst_id = int(annot.CollectionInfo.Parameters.get("BURST"))
    swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
    return (swath, burst_id)


def get_burst_id_safe(file_sets: Sequence[dict], annot, relative_orbit_number: int) -> tuple[str, int]:
    """
    Computes the swath ID and burst absolute ID based on Sentinel-1 annotation and metadata.

    Parameters:
        file_sets (Sequence[dict]): A sequence of dictionaries representing file sets with metadata.
        annot: An annotation object containing metadata about the collection.
        relative_orbit_number (int): The relative orbit number for the current observation.

    Returns:
        tuple[str, int]: A tuple containing the swath ID (lowercase) and the burst absolute ID.
    """

    # Identify the corresponding product file
    for file_set in file_sets:
        burst = int(annot.CollectionInfo.Parameters.get("BURST")) - 1
        swath = annot.CollectionInfo.Parameters.get("SWATH").lower()
        polarization = annot.get_processed_polarization().lower().replace(":", "")

        product_file_name = file_set["product"]
        if f"-{swath}-" in os.path.basename(product_file_name) and f"-{polarization}-" in os.path.basename(
            product_file_name,
        ):
            break
    else:
        # If no matching product file is found
        product_file_name = None
    mode = swath[:-1]
    # Extract ANX (Ascending Node Crossing) time from the XML file
    root_node = sarpy.io.complex.sentinel._parse_xml(product_file_name, without_ns=True)
    anx_time = float(root_node.findall("./swathTiming/burstList/burst")[burst].find("./azimuthAnxTime").text)
    # Constants for timing calculations
    TBEAM = {"iw": 2.758273, "ew": 3.0388819516}  # Time per beam in seconds
    TORB = 12 * 86400 / 175  # Orbital period in seconds
    DELTA = {
        "iw1": -0.5469335,
        "iw2": 0.2852695,
        "iw3": 1.3633065,
        "ew1": -0.20158940051821037,
        "ew2": 0.4289565994970726,
        "ew3": 0.9952805994812225,
        "ew4": 1.5820385994932618,
        "ew5": 2.192150599480655,
    }  # Swath-specific time offsets
    t = (relative_orbit_number - 1) * TORB + anx_time - DELTA[swath]
    burst_id = int(np.round(t / TBEAM[mode]))
    if mode == "ew":
        burst_id += 11

    return (swath, burst_id)


def get_relative_orbit_number(in_path: str, **kwargs):
    """
    Extracts the relative orbit number from the Sentinel-1 manifest file.

    Parameters:
        in_path (str): The path to the directory containing the 'manifest.safe' file.

    Returns:
        int: The relative orbit number, or None if extraction fails.
    """
    storage_options = get_storage_options(kwargs)
    manifest_path = Path(f"{in_path}/manifest.safe", storage_options=storage_options)
    local_manifest_path = manifest_path.ensure_local()._path
    ns, root_node = sarpy.io.complex.sentinel._parse_xml(local_manifest_path)
    relative_orbit_number = int(root_node.findall(".//safe:relativeOrbitNumber", ns)[0].text)
    return relative_orbit_number


def get_margin(annot, border: int = 10):
    step = {
        "azimuth_time": np.timedelta64(
            round((annot["RMA"]["INCA"]["TimeCAPoly"]["Coefs"][1] * annot["Grid"]["Col"]["SS"]) * 1e9),
            "ns",
        ),
        "slant_range_time": 2 * annot["Grid"]["Row"]["SS"] / speed_of_light,
    }
    # Define a margin for data selection to avoid edge effects.
    margin = {e: step[e] * border for e in ["azimuth_time", "slant_range_time"]}
    return margin
