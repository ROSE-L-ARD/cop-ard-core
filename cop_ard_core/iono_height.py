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

import os
from typing import Dict

import numpy as np
import pandas as pd
import PyIRI
import PyIRI.main_library as ml
from pyproj import CRS, Transformer

from cop_ard_core.logger import logger


def convert_ecef_to_llh(scp: np.ndarray) -> np.ndarray:
    """
    Converts Earth-Centered, Earth-Fixed (ECEF) coordinates to Longitude, Latitude, and Height (LLH).

    Parameters
    ----------
    scp : numpy.ndarray
        A numpy array containing the X, Y, Z coordinates in the ECEF frame.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the transformed Longitude, Latitude, and Height coordinates.
    """
    # Define the Source CRS (EPSG:4978 = WGS 84 ECEF) and Target CRS (EPSG:4326 = WGS 84 Geographic).
    # .to_3d() is used to ensure the CRS is treated as a 3D system (Lat/Lon/Height).
    source = CRS.from_epsg(4978)
    target = CRS.from_epsg(4326).to_3d()

    # Initialize the transformer.
    # always_xy=True ensures the order is (lon, lat) rather than (lat, lon).
    transformer = Transformer.from_crs(source, target, always_xy=True)

    # Perform the coordinate transformation on the input point.
    return np.array(transformer.transform(scp[0], scp[1], scp[2]))


def get_f107(df: pd.DataFrame, year: int, month: int, day: int) -> float:
    """
    Retrieve the observed F10.7 solar radio flux value for a specific date.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing space weather data, indexed by 'DATE'.
    year : int
        The year of the target date.
    month : int
        The month of the target date.
    day : int
        The day of the target date.

    Returns
    -------
    float
        The observed F10.7 index value.
    """
    date = pd.Timestamp(year=year, month=month, day=day)
    try:
        f107_val = float(df.loc[date, "F10.7_OBS"])
        return f107_val
    except KeyError as e:
        error_msg = f"F10.7 solar flux data missing for the requested date: {date.strftime('%Y-%m-%d')}."
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def init_iono_db(url: str = "https://celestrak.org/SpaceData/SW-All.csv", dbfile: str = "SW-ALL.pkl") -> None:
    """
    Download and initialize the space weather database.

    Fetches the latest space weather data (including F10.7 index) from Celestrak,
    formats the date column, and saves the database locally as a pickle file
    for faster subsequent retrieval.

    Parameters
    ----------
    url : str, optional
        The URL to the Celestrak space weather CSV file.
    dbfile : str, optional
        The local filepath where the database pickle will be saved.

    Returns
    -------
    None
    """
    logger.info(f"Downloading space weather database from {url}...")
    try:
        sw_all = pd.read_csv(url)
        sw_all["DATE"] = pd.to_datetime(sw_all["DATE"])
        sw_all = sw_all.set_index("DATE")
        sw_all.to_pickle(dbfile)
        logger.info(f"Space weather database successfully saved to {dbfile}.")
    except Exception as e:
        logger.critical(f"Type of error during DB initialization: {e.__class__.__name__}")
        logger.error(f"Failed to fetch or parse space weather data from {url}")
        raise RuntimeError(f"Could not initialize ionospheric database: {e}") from e


def read_iono_db(year: int, month: int, day: int, dbfile_tmp: str, dbfile: str | None = None) -> float:
    """
    Read the local space weather database and retrieve the F10.7 flux value.
    If the database file does not exist, it triggers a download automatically.

    Parameters
    ----------
    year : int
        The year of the target date.
    month : int
        The month of the target date.
    day : int
        The day of the target date.
    dbfile : str, optional
        The local filepath of the space weather pickle database.

    Returns
    -------
    float
        The observed F10.7 index value for the specified date.
    """
    if (dbfile is not None) or not os.path.exists(dbfile):
        logger.warning(f"Database file '{dbfile}' not found. Attempting to download it now.")
        init_iono_db(dbfile=dbfile_tmp)
        dbfile = dbfile_tmp
    sw_all = pd.read_pickle(dbfile)
    return get_f107(sw_all, year, month, day)


def get_iono_height(
    acquisition_date: str | pd.Timestamp, scp: Dict[str, float], wdir_path_tmp: str, dbfile: str | None = None
) -> float:
    """
    Calculate the peak height of the ionospheric F2 layer (hmF2).

    This function evaluates the ionospheric height for a given satellite acquisition
    timestamp and spatial position using the PyIRI empirical model. It retrieves necessary
    solar flux data (F10.7) and computes the ionospheric parameters.

    Parameters
    ----------
    acquisition_date : str | pandas.Timestamp
        The timestamp of the SAR acquisition.
    scp : dict
        A dictionary containing the spatial coordinates in ECEF, mapped to
        values representing X, Y, and Z.

    Returns
    -------
    float
        The estimated peak height of the F2 layer (hmF2) in meters.
    """
    if not isinstance(acquisition_date, pd.Timestamp):
        dt = pd.Timestamp(acquisition_date)
    else:
        dt = acquisition_date

    year, mth, day = dt.year, dt.month, dt.day

    # --- Time Setup ---
    # Define Universal Time (UTC) for the IRI model
    fractional_hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)
    aUT = np.array([fractional_hour])

    # --- Location Setup ---
    # Extract ECEF coordinates from dict and convert to Lon/Lat/Height
    ecef_coords = np.array([scp[i] for i in ["X", "Y", "Z"]])
    sat_pos_llh = convert_ecef_to_llh(ecef_coords)

    alon = np.array([sat_pos_llh[0]])  # Longitude
    alat = np.array([sat_pos_llh[1]])  # Latitude

    # Altitude grid (km) required for the density evaluation routine
    aalt = np.arange(90, 1001, 5)

    # --- Solar Flux Data ---
    # Retrieve the daily F10.7 value from the local database
    f107_val = read_iono_db(year, mth, day, f"{wdir_path_tmp}/SW-ALL.pkl", dbfile=dbfile)

    # --- PyIRI Model Execution ---
    logger.info(f"Evaluating PyIRI model for acquisition date {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # Evaluate the empirical International Reference Ionosphere (IRI) parameters
        F2, F1, E, Es, sun, mag, EDP = ml.IRI_density_1day(
            year, mth, day, aUT, alon, alat, aalt, f107_val, PyIRI.coeff_dir
        )
    except Exception as e:
        logger.critical(f"PyIRI model execution failed: {e.__class__.__name__}")
        raise RuntimeError(f"Error computing ionospheric density via PyIRI: {e}") from e

    # Return the peak height (hm) of the F2 layer
    peak_height = float(F2["hm"].ravel()[0]) * 1e3
    logger.info(f"Calculated hmF2 peak height: {peak_height:.2f} m")

    return peak_height
