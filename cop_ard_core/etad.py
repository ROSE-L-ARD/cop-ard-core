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
import numpy as np
import scipy.interpolate
import xarray as xr


def apply_etad_corrections(
    az: xr.DataArray,
    rg: xr.DataArray,
    etad_correction_grid: xr.Dataset,
    dtaz,
    dtrg,
    rg0,
    az0,
):
    """
    Apply ETAD corrections by interpolating the correction grids.

    This function takes azimuth and range time coordinates and applies the
    ETAD corrections by performing a 2D interpolation on the azimuth and range
    correction grids (`azcorr`, `rgcorr`).

    Parameters
    ----------
    az : xr.DataArray
        The input azimuth time coordinates to be corrected.
    rg : xr.DataArray
        The input slant range time coordinates to be corrected.
    orig : xr.DataTree
        The DataTree node for a single acquisition, containing the
        `conditions/ETAD` group with the correction grids.
    dtaz
        The azimuth time step (in ns) of the target grid.
    dtrg
        The slant range time step of the target grid.
    rg0
        The slant range time origin of the target grid.
    az0
        The azimuth time origin of the target grid.

    Returns
    -------
    tuple
        A tuple containing the interpolated azimuth (`daz`) and range (`drg`) corrections.
    """
    # initialize output
    daz = np.timedelta64(0, "ns")  # type: Union[np.timedelta64, np.ndarray]
    drg = np.zeros(1)  # type: np.ndarray

    if len(etad_correction_grid.data_vars) != 0:
        # --- Prepare Coordinates for Interpolation ---
        # `xi`: The target coordinates where we want to know the correction values, in pixel units.
        xi = np.vstack(((rg - rg0).values.ravel() / dtrg, (az - az0).astype(float).values.ravel() / dtaz)).T.reshape(-1, 2)
        # `point`: The source coordinates of the ETAD correction grid, also in pixel units.
        point = (
            (etad_correction_grid.slant_range_time - rg0).values / dtrg,
            (etad_correction_grid.azimuth_time - az0).astype(float).values / dtaz,
        )
        # Use cubic interpolation for smooth results.
        daz = np.round(
            scipy.interpolate.interpn(
                point, etad_correction_grid.azcorr.values.T, xi, method="cubic", fill_value=0.0, bounds_error=False
            )
            * 1e9
        ).astype("timedelta64[ns]")
        daz = daz.reshape(az.shape)

        # Interpolate the range correction grid.
        drg = scipy.interpolate.interpn(
            point, etad_correction_grid.rgcorr.values.T, xi, method="cubic", fill_value=0.0, bounds_error=False
        )
        drg = drg.reshape(rg.shape)

    # Return the computed correction values.
    return daz, drg
