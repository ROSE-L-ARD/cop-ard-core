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
import copy
import datetime
import os
from typing import cast

import numpy as np
import xarray as xr
from scipy.constants import speed_of_light

import cop_ard_core
from cop_ard_core.computing.computation import map, map_blocks
from cop_ard_core.computing.griddata import init_interpolate
from cop_ard_core.computing.gslc import fix_attributes
from cop_ard_core.computing.io import get_id_product, get_storage_options, normpath, read, write
from cop_ard_core.computing.logger import timeit
from cop_ard_core.computing.slc import extract_orbits, init_single_coreg_stack, get_margin
from cop_ard_core.computing.warp import GRID, coreg_qlt, sinc, updateLut, warp_dazdrg, warp_cslc


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


def updateOrbits(slc: xr.DataTree) -> xr.Dataset:
    """
    Consolidate orbit information from all acquisitions into a single Dataset.

    This function extracts orbit polynomial coefficients from the metadata of
    each original SLC image and combines them with any externally provided
    precise orbit data. The result is a clean, unified Dataset of orbit
    information for the entire stack.

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing the `orig` nodes with
        source metadata.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset containing the consolidated orbit coefficients,
        reference times, and types for all acquisitions.
    """
    # --- Extract Orbits from Annotations ---
    # This provides a baseline set of orbit data from the original product files.
    orbits = extract_orbits(slc, multiburst=False)
    orbits = orbits.to_dataset(name="coeffs")
    num_dates = orbits.sizes["acquisition_date"]
    orbits["reference_time"] = (
        "acquisition_date",
        [orbits.coeffs.reference_times[date] for date in orbits.acquisition_date.values],
    )
    orbits["type"] = ("acquisition_date", ["AUX_ANNOTATED"] * num_dates)
    orbits["orbit_files"] = ("acquisition_date", [""] * num_dates)
    for date_key, v in slc.orig.items():
        if ("conditions" in v) and ("orbit" in v.conditions):
            precise_orbit = v.conditions.orbit
            # Safely locate the specific date across the DataArray
            loc_dict = {"acquisition_date": date_key}
            # Overwrite the coefficients
            orbits["coeffs"].loc[loc_dict] = precise_orbit.coeffs.loc[loc_dict]
            # Overwrite the metadata to reflect the precise orbit
            orbits["type"].loc[loc_dict] = precise_orbit.attrs.get("orbit_type", "AUX_POEORB")
            # Extract the orbit file name (accounting for lists or strings in the attributes)
            if "orbit_files" in precise_orbit.attrs:
                file_attr = precise_orbit.attrs["orbit_files"]
                file_name = file_attr[0] if isinstance(file_attr, list) else file_attr
                orbits["orbit_files"].loc[loc_dict] = file_name

            # Overwrite reference time if the precise orbit has a different one
            if "reference_time" in precise_orbit.attrs:
                orbits["reference_time"].loc[loc_dict] = precise_orbit.attrs["reference_time"]

    orbits = orbits.chunk(axis=-1, degree=-1, acquisition_date=1)

    return orbits


def updateETAD(slc: xr.DataTree) -> xr.Dataset | None:
    """
    Consolidate Extended Timing Annotation Data (ETAD) from all acquisitions.

    If ETAD correction files were used during processing, this function gathers
    the ETAD information from each acquisition's `conditions` group and
    combines them into a single Dataset for the stack.

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing the `orig` nodes.

    Returns
    -------
    xarray.Dataset or None
        An xarray.Dataset containing the consolidated ETAD corrections, or None
        if no ETAD data is found.
    """
    # Check if ETAD data is present for all acquisitions.
    if all(("conditions" in v) and ("ETAD" in v.conditions) for _, v in slc.orig.items()):
        # Create a list of ETAD datasets, one for each acquisition.
        etads = [
            v.conditions.ETAD.to_dataset().assign_coords(acquisition_date=k).expand_dims("acquisition_date")
            for k, v in slc.orig.items()
        ]
        # Add the original ETAD filename as a variable.
        etads = [
            e.assign({k: (["acquisition_date"], [os.path.basename(e.attrs.get(k, ""))]) for k in ["etad_file"]})
            for e in etads
        ]
        # Concatenate the list of datasets into a single dataset along the acquisition_date dimension.
        etads = xr.concat(etads, dim="acquisition_date")
        # Re-chunk the final dataset.
        etads = etads.chunk({k: 2048 for k in etads.dims})
    else:
        # Return None if ETAD data is not consistently available.
        etads = None

    return etads


def update_conditions(slc: xr.DataTree, daz_drg: xr.Dataset, conditions: xr.DataTree) -> xr.DataTree:
    """
    Update and populate the 'conditions' group of the coregistered stack.

    This function acts as a high-level manager for updating all ancillary data
    related to the processing conditions. It calls helper functions to update
    orbit data, ETAD corrections, and the phase demodulation (Doppler) LUT.

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing the `orig` data.
    daz_drg
        The dataset of computed azimuth and range shifts, used for LUT generation.
    conditions
        The target `conditions` DataTree node to be updated.

    Returns
    -------
    DataTree
        The updated `conditions` DataTree node.
    """
    # Consolidate and update the orbit information.
    conditions["orbit"] = updateOrbits(slc)
    # Consolidate and update ETAD information if available.
    if all(("conditions" in v) and ("ETAD" in v.conditions) for _, v in slc.orig.items()):
        conditions["ETAD"] = updateETAD(slc)
    # Generate the updated Doppler (phase_demod) LUT for the coregistered grid.
    xLUT = updateLut(slc, daz_drg, "Grid.Col.DeltaKCOAPoly.Coefs", "phase_demod")
    conditions["phase_demod"] = xLUT
    return conditions


def update_quality(slc: xr.DataTree, daz_drg: xr.Dataset, quality: xr.Dataset) -> xr.Dataset:
    """
    Update and populate the 'quality' group of the coregistered stack.

    This function computes and stores all data related to the quality of the
    coregistered product, including the estimated coregistration accuracy and
    Lookup Tables for radiometric quantities (noise, sigma0, gamma0, etc.).

    Parameters
    ----------
    slc
        The DataTree for a single burst, containing `orig` and `coreg` data.
    daz_drg
        The dataset of computed azimuth and range shifts, used for LUT generation.
    quality
        The target `quality` Dataset to be updated.

    Returns
    -------
    xarray.Dataset
        The updated `quality` Dataset.
    """
    from cop_ard_core.computing.io import path_to_dict

    # Mapping from official calibration names to variable names in the output.
    name_mapping = {
        "SigmaZero": "sigma0",
        "GammaZero": "gamma0",
        "BetaZero": "beta0",
        "RCS": "RCS",
        "NoiseLevel": "noise",
    }
    calib_ok = ["SigmaZero", "GammaZero", "BetaZero", "RCS"]
    NoiseLevel = "NoiseLevel"

    # --- Compute and Assign Quality Metrics ---
    # Compute the estimated coregistration accuracy.
    quality = quality.assign(coreg_qlt(slc, daz_drg))
    # Generate the updated Noise LUT for the coregistered grid.

    annot = slc.coreg.measurements.cstack.attrs
    annot = annot[list(annot)[0]]
    if path_to_dict(annot, "Radiometric.NoiseLevel.NoisePoly.Coefs") is not None:
        quality = quality.assign(
            updateLut(
                slc,
                daz_drg,
                "Radiometric.NoiseLevel.NoisePoly.Coefs",
                name_mapping[NoiseLevel],
            )
        )
    # Generate updated LUTs for all specified radiometric calibration types.
    for calib in calib_ok:
        if path_to_dict(annot, f"Radiometric.{calib}SFPoly.Coefs") is not None:
            quality = quality.assign(
                updateLut(
                    slc,
                    daz_drg,
                    f"Radiometric.{calib}SFPoly.Coefs",
                    name_mapping[calib],
                )
            )
    return quality


def initialize_LUT_coords(measurements: xr.Dataset) -> xr.Dataset:
    """
    Initialize a coarse coordinate grid for the Lookup Tables (LUTs).

    This function creates a downsampled version of the full-resolution measurement
    grid. This coarse grid is used as the coordinate system for all the LUTs
    (e.g., Doppler, noise) to save space and computation time. It also initializes
    the `daz` and `drg` variables that will be filled in later.

    Parameters
    ----------
    measurements
        The full-resolution coregistered measurements dataset.

    Returns
    -------
    xarray.Dataset
        An xarray.Dataset representing the coarse LUT grid with empty `daz`
        and `drg` data variables.
    """
    # Calculate the step size to achieve the target coarse grid size (e.g., 11x11).
    az_step, rg_step = (
        np.ceil(measurements.azimuth_time.size / GRID[0]).astype(int),
        np.ceil(measurements.slant_range_time.size / GRID[1]).astype(int),
    )
    # Create the coarse grid by decimating the full-resolution coordinates.
    LUT_coords = measurements.isel(
        azimuth_time=slice(None, None, az_step),
        slant_range_time=slice(None, None, rg_step),
    )
    # Keep only the essential coordinate variables.
    LUT_coords = LUT_coords.drop_vars(
        [e for e in LUT_coords.variables if e not in ["acquisition_date", "azimuth_time", "slant_range_time"]]
    )

    # --- Initialize daz and drg Data Variables ---
    # Assign empty arrays to hold the azimuth and range shift values on this coarse grid.
    LUT_coords = LUT_coords.assign(
        {
            "drg": (
                ["acquisition_date", "azimuth_time", "slant_range_time"],
                np.full(
                    (
                        LUT_coords.acquisition_date.size,
                        LUT_coords.azimuth_time.size,
                        LUT_coords.slant_range_time.size,
                    ),
                    0,
                    dtype=float,
                ),
            ),
            "daz": (
                ["acquisition_date", "azimuth_time", "slant_range_time"],
                np.full(
                    (
                        LUT_coords.acquisition_date.size,
                        LUT_coords.azimuth_time.size,
                        LUT_coords.slant_range_time.size,
                    ),
                    0,
                    dtype="timedelta64[ns]",
                ),
            ),
        }
    ).compute()

    return LUT_coords


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

    daz_drg = cast(
        xr.Dataset,
        read(
            daz_drg_store_path,
            consolidate_metadata=False,
            storage_options=get_storage_options(kwargs),
            as_type="xr.Dataset",
        ),
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


def update_LUT(daz_drg: xr.Dataset, LUT_coords: xr.Dataset):
    mask = (
        (LUT_coords.azimuth_time >= daz_drg.azimuth_time.min())
        & (LUT_coords.azimuth_time <= daz_drg.azimuth_time.max())
        & (LUT_coords.slant_range_time >= daz_drg.slant_range_time.min())
        & (LUT_coords.slant_range_time <= daz_drg.slant_range_time.max())
    ).values.ravel()
    (ind,) = np.where(mask)
    # If there are valid points, update the LUT with the computed daz and drg values.
    LUT_assign = xr.zeros_like(LUT_coords)
    if ind.size:
        az_ind, rg_ind = np.unravel_index(ind, (LUT_coords.azimuth_time.size, LUT_coords.slant_range_time.size))
        az_ind = xr.DataArray(az_ind, dims="point")
        rg_ind = xr.DataArray(rg_ind, dims="point")
        # Update daz values in the LUT for the current acquisition date chunk.
        LUT_assign.daz[..., az_ind, rg_ind] = daz_drg.daz.sel(
            azimuth_time=LUT_coords.azimuth_time[az_ind],
            slant_range_time=LUT_coords.slant_range_time[rg_ind],
        )
        # Update drg values in the LUT for the current acquisition date chunk.
        LUT_assign.drg[..., az_ind, rg_ind] = daz_drg.drg.sel(
            azimuth_time=LUT_coords.azimuth_time[az_ind],
            slant_range_time=LUT_coords.slant_range_time[rg_ind],
        )
    return LUT_assign


def update_coreg(slcs: xr.DataTree, external_metadata: dict = {}):
    """
    Finalize the coregistered DataTree by adding product-level metadata.

    This function iterates through the processed bursts, restructures the data
    (e.g., renaming 'cstack' to 'cslc'), and assigns a comprehensive set of
    attributes to conform to the final product specification.

    Parameters
    ----------
    slcs
        The DataTree containing all the processed coregistered bursts.
    external_metadata : dict
        A dictionary of external metadata to be included in
        the final product attributes.

    Returns
    -------
    DataTree
        A new DataTree, restructured and fully populated with final metadata.
    """
    processing_start_time = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    bursts = xr.DataTree()
    multiburst = True if "bursts" in slcs else False
    # --- Iterate Over Each Burst/Scene ---
    for k, slc in (slcs.bursts if multiburst else {"data": slcs}).items():
        name = k if multiburst else "data"

        stac_discovery = get_stac_cslc(
            slc,
            name,
            external_metadata=external_metadata,
            processing_start_time=processing_start_time,
        )
        processing_history = get_processing_history_cslc(
            slc,
            stac_discovery,
            external_metadata=external_metadata,
            processing_start_time=processing_start_time,
        )
        other_metadata = get_other_metadata_cslc(slc, external_metadata)
        bursts[stac_discovery["id"]] = clean_cslc(slc)
        # --- Assign Product-Level Metadata ---
        bursts[stac_discovery["id"]].attrs = {
            "stac_discovery": stac_discovery.copy(),
            "other_metadata": other_metadata.copy(),
            "processing_history": processing_history.copy(),
        }
        slc.attrs = {
            "stac_discovery": stac_discovery.copy(),
            "other_metadata": other_metadata.copy(),
            "processing_history": processing_history.copy(),
        }
    stac_group, history_group = update_cslc_container(bursts)

    # Apply any final attribute fixes.
    fix_attributes(bursts, ATTRS_RADAR)
    bursts.attrs = {"stac_discovery": stac_group, "processing_history": history_group}
    slcs.attrs = {"stac_discovery": stac_group, "processing_history": history_group}
    return bursts, slcs


def update_cslc_container(bursts: xr.DataTree):
    attrs = copy.deepcopy(next(iter(bursts.children.values())).attrs)
    stac = attrs["stac_discovery"]
    processing_history = attrs["processing_history"]
    stac["properties"].pop("sat:anx_datetime", None)
    stac["properties"].pop("sar:pixel_spacing_azimuth", None)
    stac["properties"].pop("sar:pixel_spacing_range", None)
    stac["properties"].pop("view:incidence_angle", None)
    stac.pop("id", None)
    start = np.datetime64(stac["properties"]["start_datetime"])
    end = np.datetime64(stac["properties"]["end_datetime"])
    bbox = stac["bbox"]
    coords = stac["geometry"]["coordinates"][0][:-1]
    assets = {}
    for k, v in bursts.items():
        stac_product = v.attrs["stac_discovery"]
        s = np.datetime64(stac_product["properties"]["start_datetime"])
        e = np.datetime64(stac_product["properties"]["end_datetime"])
        b = stac_product["bbox"]
        c = stac_product["geometry"]["coordinates"][0][:-1]
        bbox = [min(b[0], bbox[0]), min(b[1], bbox[1]), max(b[2], bbox[2]), max(b[3], bbox[3])]
        start = np.minimum(s, start)
        end = np.maximum(e, end)
        for i in range(4):
            i1, i2 = i // 2, i % 2
            lon_f = min if i1 == 0 else max
            lat_f = min if i2 == 0 else max
            coords[i] = [lon_f(coords[i][0], c[i][0]), lat_f(coords[i][1], c[i][1])]
        assets[k] = {"href": f"./{k}", "type": "application/vnd+zarr"}
    coords.append(coords[0])

    datetime = start + round((end.astype(int) - start.astype(int)) / 2)
    start = str(start) + "Z"
    end = str(end) + "Z"
    datetime = str(datetime) + "Z"
    stac["properties"]["datetime"] = datetime
    stac["properties"]["start_datetime"] = start
    stac["properties"]["end_datetime"] = end
    stac["bbox"] = bbox
    stac["geometry"]["coordinates"] = [coords]
    stac["properties"]["sar:beam_ids"] = beam_ids[stac["properties"]["sar:instrument_mode"]]
    stac["assets"] = assets
    id = get_id_product(stac)
    stac["id"] = id
    processing_history["Sentinel-1 ARD CSLC Product"][0]["outputs"] = id

    return stac, processing_history


def get_stac_cslc(slc: xr.DataTree, burst_id: str, external_metadata, processing_start_time):
    reference_date = slc.coreg.measurements.attrs["coregistration_reference_date"]
    acquisition_dates = slc.coreg.measurements.cstack.acquisition_date.values
    date = acquisition_dates[0] if acquisition_dates.size == 1 else reference_date
    orig = slc.orig[date].measurements.slc
    coreg = slc.coreg.measurements.cstack
    stac = slc.orig[date].conditions.attrs.get("stac_discovery", {})
    stac_discovery = stac_discovery_template(burst_id, orig, coreg, external_metadata, stac, processing_start_time)
    return stac_discovery


def get_processing_history_cslc(slc: xr.DataTree, stac_discovery, external_metadata, processing_start_time):
    reference_date = slc.coreg.measurements.attrs["coregistration_reference_date"]
    acquisition_dates = slc.coreg.measurements.cstack.acquisition_date.values
    date = acquisition_dates[0] if acquisition_dates.size == 1 else reference_date
    processing_history = slc.orig[date].conditions.attrs.get("processing_history", {})
    stac = slc.orig[date].conditions.attrs.get("stac_discovery", {})
    inputs = stac["id"] if "id" in stac else "n/a"
    if "Level-1 Product" in processing_history:
        if processing_history["Level-1 Product"][0]["outputs"]:
            inputs = processing_history["Level-1 Product"][0]["outputs"]

    adfs = [external_metadata["dem"]]
    for e in ["etad", "orbit"]:
        if e in external_metadata:
            adfs.extend(external_metadata[e])

    processing_history["Sentinel-1 ARD CSLC Product"] = [
        {
            "adfs": adfs,
            "inputs": inputs,
            "facility": external_metadata["facility"],
            "outputs": stac_discovery["id"],
            "processor": "Sentinel-1 ARD CORE",
            "time": processing_start_time,
            "version": f"{cop_ard_core.__version__}",
        }
    ]
    return processing_history


def get_other_metadata_cslc(slc: xr.DataTree, external_metadata):
    reference_date = slc.coreg.measurements.attrs["coregistration_reference_date"]
    acquisition_dates = slc.coreg.measurements.cstack.acquisition_date.values
    date = acquisition_dates[0] if acquisition_dates.size == 1 else reference_date

    other_metadata = {}
    other_metadata["SICD"] = slc.orig[date].measurements.slc.attrs
    other_metadata["product_geometry"] = "slant-range"
    other_metadata["product_pixel_coordinate_convention"] = external_metadata["product_pixel_coordinate_convention"]
    other_metadata["cslc_status"] = external_metadata["cslc_status"]
    other_metadata["product_family_specification"] = external_metadata["product_family_specification"]
    other_metadata["measurement_type"] = external_metadata["measurement_type"]
    other_metadata["noise_correction_flag"] = external_metadata["noise_removal"]

    return other_metadata


def clean_cslc(slc: xr.DataTree):
    # --- Restructure the DataTree ---
    # Rename 'cstack' to 'cslc' and drop unnecessary coordinates.
    cslc = xr.DataTree()

    measurements = (
        slc.coreg.measurements.to_dataset().rename({"cstack": "cslc"}).drop_vars(["slant_range_m", "azimuth_m"])
    ).drop_attrs()
    quality = slc.coreg.quality.to_dataset().rename({"RCS": "rcs", "crg_accuracy": "coregistration_accuracy"})
    orbit = slc.coreg.conditions.orbit.to_dataset()
    phase_demod = slc.coreg.conditions.phase_demod.to_dataset()
    # Ensure ETAD group exists, even if empty.
    if "ETAD" not in slc.coreg.conditions:
        etad = xr.Dataset(
            data_vars={
                "azcorr": (["slant_range_time", "azimuth_time"], [[np.nan]]),
                "rgcorr": (["slant_range_time", "azimuth_time"], [[np.nan]]),
            },
            coords={
                "slant_range_time": [0.0],
                "azimuth_time": [np.datetime64("0", "ns")],
            },
        )
    else:
        etad = slc.coreg.conditions.ETAD.to_dataset()

    if measurements.acquisition_date.size == 1:
        measurements = measurements.squeeze("acquisition_date")
        quality = quality.squeeze("acquisition_date")
        orbit = orbit.drop_vars(["orbit_files", "reference_time", "type"]).squeeze("acquisition_date")
        phase_demod = phase_demod.squeeze("acquisition_date")
        if "etad_file" in etad:
            etad = etad.drop_vars(["etad_file"]).squeeze("acquisition_date")

    cslc["measurements"] = measurements
    cslc["quality"] = quality
    cslc["conditions/orbit"] = orbit
    cslc["conditions/phase_demodulation"] = phase_demod
    cslc["conditions/ETAD"] = etad

    return cslc


def stac_discovery_template(burst_id, orig, coreg, external_metadata, stac, processing_start_time):
    sicd = orig.attrs[list(orig.attrs)[0]]
    sicd_coreg = coreg.attrs[list(coreg.attrs)[0]]
    coordinates = []
    for v in sicd["GeoData"]["ImageCorners"]:
        coordinates.append([round(v["Lon"], 6), round(v["Lat"], 6)])
    coordinates.append(coordinates[0])
    bbox = np.array(coordinates)
    bbox = [
        float(min(bbox[:, 0])),
        float(min(bbox[:, 1])),
        float(max(bbox[:, 0])),
        float(max(bbox[:, 1])),
    ]

    s = orig.azimuth_time.min().values.astype("datetime64[us]").item().isoformat() + "Z"
    e = orig.azimuth_time.max().values.astype("datetime64[us]").item().isoformat() + "Z"
    m = orig.azimuth_time[orig.azimuth_time.size // 2].values.astype("datetime64[us]").item().isoformat() + "Z"
    mode = sicd["CollectionInfo"]["RadarMode"]["ModeID"]
    pol = ["".join(i.split(":")) for i in orig.polarization.values]
    if "properties" in stac:
        datatake_id = stac["properties"]["eopf:datatake_id"]
        absolute_orbit = stac["properties"]["sat:absolute_orbit"]
        relative_orbit = stac["properties"]["sat:relative_orbit"]
        anx_time = stac["properties"]["sat:anx_datetime"]
        if not anx_time.endswith("Z"):
            anx_time += "Z"
        orbit_state = stac["properties"]["sat:orbit_state"]
        platform_international_designator = stac["properties"]["sat:platform_international_designator"]
    else:
        datatake_id = sicd["CollectionInfo"]["CoreName"][10:15]
        absolute_orbit = 1
        relative_orbit = 1
        anx_time = "1970-01-01T00:00:00.000000Z"
        orbit_state = "n/a"
        platform_international_designator = "n/a"

    stac_payload = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/eopf/v1.2.0/schema.json",
            "https://stac-extensions.github.io/product/v0.1.0/schema.json",
            "https://stac-extensions.github.io/sat/v1.1.0/schema.json",
            "https://stac-extensions.github.io/view/v1.0.0/schema.json",
            "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
            "https://stac-extensions.github.io/sar/v1.2.0/schema.json",
        ],
        "bbox": bbox,
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
        "properties": {
            "title": "Sentinel-1 ARD CSLC Product",
            "datetime": m,
            "start_datetime": s,
            "end_datetime": e,
            "created": processing_start_time,
            "constellation": "sentinel-1",
            "instruments": ["sar"],
            "platform": sicd["CollectionInfo"]["CollectorName"].replace("S", "sentinel-").lower(),
            "eopf:datatake_id": datatake_id,
            "eopf:instrument_mode": mode,
            "processing:lineage": "systematic",
            "processing:software": {"Sentinel-1 ARD CORE": f"{cop_ard_core.__version__}"},
            "processing:version": f"{cop_ard_core.__version__}",
            "processing:datetime": processing_start_time,
            "processing:facility": external_metadata["facility"],
            "product:timeliness": external_metadata["product_timeliness"],
            "product:timeliness_category": external_metadata["product_timeliness_category"],
            "product:type": f"S01S{mode}CSL",
            "providers": external_metadata["providers"],
            "sar:beam_ids": [sicd["CollectionInfo"]["Parameters"]["SWATH"].upper()],
            "sar:center_frequency": sicd["RMA"]["INCA"]["FreqZero"],
            "sar:frequency_band": "C",
            "sar:instrument_mode": mode,
            "sar:pixel_spacing_azimuth": round(sicd_coreg["Grid"]["Col"]["SS"], 5),
            "sar:pixel_spacing_range": round(sicd_coreg["Grid"]["Row"]["SS"], 5),
            "sar:polarizations": pol,
            "sat:absolute_orbit": absolute_orbit,
            "sat:anx_datetime": anx_time,
            "sat:orbit_state": orbit_state,
            "sat:platform_international_designator": platform_international_designator,
            "sat:relative_orbit": relative_orbit,
            "view:incidence_angle": sicd["SCPCOA"]["IncidenceAng"],
            "sci:citation": external_metadata["atbd_link"],
        },
        "links": [{"href": "./.zattrs.json", "rel": "self", "type": "application/json"}],
        "assets": {
            "measurements": {
                "href": "./measurements",
                "title": "CSLC Measurements",
                "description": "Zarr group containing the complex SAR image data",
                "type": "application/vnd+zarr",
                "roles": ["data"],
            },
            "conditions": {
                "href": "./conditions",
                "title": "CSLC Conditions",
                "description": "Zarr group containing measurement conditions",
                "type": "application/vnd+zarr",
                "roles": ["metadata"],
            },
            "quality": {
                "href": "./quality",
                "title": "CSLC Quality Metrics",
                "description": "Zarr group containing quality assessment and accuracy metrics",
                "type": "application/vnd+zarr",
                "roles": ["qa", "metadata"],
            },
        },
    }

    id = get_id_product(stac_payload, burst_id)
    stac_payload["id"] = id
    return stac_payload


ATTRS_RADAR = {
    "phase_model": {
        "description": "Topographic phase model, in rad, removed from the original interferometric phase."
        "The interferometric phase is evaluated wrt the reference orbit, used for co-registration process.",
        "long_name": "Interferometric phase model",
        "units": "rad",
    },
    "polarization": {
        "description": "acquired polarizations",
        "long_name": "Polarization",
        "units": "n/a",
    },
    "acquisition_date": {
        "description": "Image acquisition date, espressed as YYYY-MM-DD, data type is string",
        "long_name": "Acquisition date",
        "units": "n/a",
    },
    "slant_range_time": {
        "description": "Time across satellite track direction expressed as seconds",
        "long_name": "Slant range time",
        "units": "s",
    },
    "azimuth_time": {
        "description": "Time along satellite track direction expressed as a datetime64[ns]",
        "long_name": "Azimuth time",
    },
    "axis": {
        "description": "Index corresponding to the axis",
        "long_name": "Axis index",
        "units": "n/a",
    },
    "degree": {
        "description": "Polynomial coefficient degree",
        "long_name": "Polynomial coefficient degree",
        "units": "n/a",
    },
    "azcorr": {
        "description": "Extended Timining Annotation Data: azimuth time correction resampled on the common grid of\
                            the reference geometry",
        "long_name": "Azimuth timing correction",
        "units": "s",
    },
    "rgcorr": {
        "description": "Extended Timining Annotation Data: slant range time corection resampled on the common grid\
                            of the reference geometry",
        "long_name": "Slant range timing correction",
        "units": "s",
    },
    "etad_file": {
        "description": "ETAD file used for each of the images belonging to the cslc stack",
        "long_name": "Extended Timing Annotation Data file",
        "units": "n/a",
    },
    "orbit_files": {
        "description": "Orbit files used for each of the images belonging to the cslc stack",
        "long_name": "Orbit files",
        "units": "n/a",
    },
    "cslc": {
        "description": "Co-registred single look complex data",
        "long_name": "Co-registered SLC",
        "units": "n/a",
    },
    "beta0": {
        "description": "LUT of the beta0 coefficients obtained from input SLC annotations.",
        "long_name": "Beta nought",
        "units": "m^2 / m^2",
    },
    "gamma0": {
        "description": "LUT of the gamma0 coefficients obtained from input SLC annotations.",
        "long_name": "Gamma nought",
        "units": "m^2 / m^2",
    },
    "noise": {
        "description": "LUT of thermal noise correction obtained from input SLC annotations.",
        "long_name": "Thermal noise correction",
        "units": "n/a",
    },
    "RCS": {
        "description": "LUT of the Radar Cross Section (RCS) coefficients obtained from input SLC annotations.",
        "long_name": "Radar cross section",
        "units": "m^2 / m^2",
    },
    "sigma0": {
        "description": "LUT of the sigma0 coefficients obtained from input SLC annotations.",
        "long_name": "Sigma nought",
        "units": "m^2 / m^2",
    },
    "crg_accuracy": {
        "description": "Estimated co-registration accuracty in correspondence of each image of the cslc stack.",
        "long_name": "Co-registration accuracy",
        "units": "m",
    },
    "phase_demod": {
        "description": "Phase to be applied to demodulate each image belonging to the cslc stack.",
        "long_name": "Demodulation phase",
        "units": "rad",
    },
    "coeffs": {
        "description": "Polynomial coefficients describing the orbit associated to each input image belonging to\
                            the cslc stack.",
        "long_name": "Orbit coefficients",
        "units": "n/a",
    },
    "reference_time": {
        "description": "Reference time associated to the polynomial coefficients for each input image belonging to\
                            the cslc stack.",
        "long_name": "Reference polynomial coefficient time",
        "units": "s",
    },
    "type": {
        "description": "Type of orbital data used for each of the images belonging to the cslc stack:\
                            ETAD, POE, POD, RES.",
        "long_name": "Orbital data accuracy",
        "units": "n/a",
    },
}

beam_ids = {
    "IW": ["IW1", "IW2", "IW3"],  # Interferometric Wide (3 sub-swaths)
    "EW": ["EW1", "EW2", "EW3", "EW4", "EW5"],  # Extra Wide (5 sub-swaths)
    "SM": ["S1", "S2", "S3", "S4", "S5", "S6"],  # Stripmap (6 overlapping beams)
}
