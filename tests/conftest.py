"""Shared fixtures for cop_ard_core tests."""

from xml.etree.ElementTree import Element, ElementTree

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def complex_3x3():
    """3x3 complex DataArray with known values."""
    data = np.array(
        [
            [1 + 1j, 2 + 2j, 3 + 3j],
            [4 + 4j, 5 + 5j, 6 + 6j],
            [7 + 7j, 8 + 8j, 9 + 9j],
        ],
        dtype=np.complex128,
    )
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "azimuth_time": np.arange(3),
            "slant_range_time": np.arange(3),
        },
    )


@pytest.fixture
def complex_5x5():
    """5x5 complex DataArray with known values."""
    data = np.array(
        [[(i + 1) * (j + 1) * (1 + 0.5j) for j in range(5)] for i in range(5)],
        dtype=np.complex128,
    )
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "azimuth_time": np.arange(5),
            "slant_range_time": np.arange(5),
        },
    )


@pytest.fixture
def orbit_xml_element():
    """Create a minimal orbit XML ElementTree for testing build_orbit_dataset."""
    root = Element("ROOT")
    file_type = Element("File_Type")
    file_type.text = "POEORB"
    root.append(file_type)

    list_osvs = Element("List_of_OSVs")
    for i in range(3):
        osv = Element("OSV")
        for tag in ["UTC", "TAI", "UT1"]:
            t = Element(tag)
            t.text = f"2020-01-01T00:00:{i * 10:02d}={i}"
            osv.append(t)
        for coord in ["X", "Y", "Z"]:
            val = Element(coord)
            val.text = f"{6500000.0 + i * 1000.0}"
            osv.append(val)
        for coord in ["VX", "VY", "VZ"]:
            val = Element(coord)
            val.text = f"{7500.0 + i * 0.5}"
            osv.append(val)
        list_osvs.append(osv)
    root.append(list_osvs)
    tree = ElementTree(root)
    return tree


@pytest.fixture
def multilook_factor():
    """Default multilook factor dict."""
    return {"azimuth_time": 3, "slant_range_time": 3}


@pytest.fixture
def compute_shadow_radar_grid():

    data = np.zeros((2, 3), dtype=np.float64)
    az = np.array([0.0, 1.0], dtype=np.float64)
    sr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    grid = xr.DataArray(
        data, dims=["azimuth_time", "slant_range_time"], coords={"azimuth_time": az, "slant_range_time": sr}
    )
    grid.rio.write_crs("EPSG:4326", inplace=True)
    return grid


@pytest.fixture
def compute_shadow_h():

    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    h = xr.DataArray(np.full((len(y), len(x)), 100.0), dims=["y", "x"], coords={"y": y, "x": x})
    h = h.expand_dims(acquisition_id=["2020-01-01"])
    h.rio.write_crs("EPSG:4326", inplace=True)
    return h


@pytest.fixture
def compute_shadow_dem_distance():

    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    data = np.full((len(y), len(x), 3), 5000.0, dtype=np.float64)
    dem_dist = xr.DataArray(data, dims=["y", "x", "coord"], coords={"y": y, "x": x})
    dem_dist = dem_dist.expand_dims(acquisition_id=["2020-01-01"])
    dem_dist.rio.write_crs("EPSG:4326", inplace=True)
    return dem_dist


@pytest.fixture
def compute_shadow_azimuth_time():
    """Azimuth times must vary across y so pixels span the 2D grid.

    Azimuth times are in nanoseconds (datetime64[ns] values), so the
    pixel coordinate is:
        az_pixel = (az_time - az_ref) / (dt_az * 1e9)

    With dt_az=0.01s=10ms, each 10ms step gives 1 pixel step.
    """
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    data = np.zeros((len(y), len(x)), dtype=np.float64)
    for j in range(len(y)):
        data[j, :] = float(j) * 0.01  # 0.01s = 1 pixel step
    return xr.DataArray(data, dims=["y", "x"]).expand_dims(acquisition_id=["2020-01-01"])


@pytest.fixture
def compute_shadow_slant_range_time():
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    data = np.zeros((len(y), len(x)), dtype=np.float64)
    for i in range(len(x)):
        data[:, i] = float(i) * 0.5
    return xr.DataArray(data, dims=["y", "x"]).expand_dims(acquisition_id=["2020-01-01"])


@pytest.fixture
def compute_shadow_dt():
    return {"azimuth_time": 0.01, "slant_range_time": 0.5}


@pytest.fixture
def compute_shadow_local_incidence_angle():

    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.full((len(y), len(x)), 0.3, dtype=np.float64)
    arr = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
    arr.rio.write_crs("EPSG:4326", inplace=True)
    return arr


@pytest.fixture
def compute_shadow_simulation_ml():
    """Simulation_ml with coarse radar times matching radar_grid (az=[0,1], sr=[0,1,2])."""
    coarse_y = np.array([-0.5, 3.5])
    coarse_x = np.array([-0.5, 5.5])
    # Radar times must cover the radar_grid's extent so all coarse cells pass the mask
    az_min = np.array([[0.0, 0.0], [0.0, 0.0]])
    sr_min = np.array([[0.0, 0.0], [0.0, 0.0]])
    az_max = np.array([[1.0, 1.0], [1.0, 1.0]])
    sr_max = np.array([[2.0, 2.0], [2.0, 2.0]])
    min_ds = xr.Dataset(
        {
            "azimuth_time": xr.DataArray(az_min, dims=["y", "x"]),
            "slant_range_time": xr.DataArray(sr_min, dims=["y", "x"]),
        },
        coords={"y": coarse_y, "x": coarse_x, "acquisition_id": np.array(["2020-01-01"], dtype="datetime64[ns]")},
    )
    max_ds = xr.Dataset(
        {
            "azimuth_time": xr.DataArray(az_max, dims=["y", "x"]),
            "slant_range_time": xr.DataArray(sr_max, dims=["y", "x"]),
        },
        coords={"y": coarse_y, "x": coarse_x, "acquisition_id": np.array(["2020-01-01"], dtype="datetime64[ns]")},
    )
    return {"acquisition_id": np.array(["2020-01-01"], dtype="datetime64[ns]"), "min": min_ds, "max": max_ds}


@pytest.fixture
def compute_shadow_margin():
    return {"azimuth_time": 0.0, "slant_range_time": 0.0}


@pytest.fixture
def compute_shadow_lia_grazing():
    """LIA with very small cos_theta values (grazing angle, near 0)."""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.full((len(y), len(x)), 0.001, dtype=np.float64)  # cos_theta = 0.001, grazing angle
    arr = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
    arr.rio.write_crs("EPSG:4326", inplace=True)
    return arr


@pytest.fixture
def compute_shadow_lia_all_large():
    """LIA with large cos_theta values (> 0.05 everywhere, no grazing shadows)."""
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    data = np.full((len(y), len(x)), 0.5, dtype=np.float64)  # cos_theta = 0.5, normal incidence
    arr = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
    arr.rio.write_crs("EPSG:4326", inplace=True)
    return arr


# ===========================================================================
# area_projection fixtures
# ===========================================================================


@pytest.fixture
def area_proj_bbox():
    """Bounding box [min_rg, min_az, max_rg, max_az] for area projection tests."""
    return np.array([0.0, 0.0, 5.0, 4.0])


@pytest.fixture
def area_proj_bbox_int():
    """Bounding box with int64 dtype for functions that use bbox values as array sizes."""
    return np.array([0, 0, 5, 4], dtype=np.int64)


@pytest.fixture
def area_proj_simulation():
    """Simulation Dataset with all attributes for area_projection tests.

    Contains:
      - x, y: geogrid coordinates (4x5 grid of points = 3x4 cells)
      - az_px, rg_px: radar pixel indices (one per cell)
      - h: DEM elevation (constant, with CRS)
      - dem_distance: radar distance vectors (3D)
    """

    y = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dim_y, dim_x = len(y) - 1, len(x) - 1  # 3, 4

    # DEM elevation: (dim_y+1, dim_x+1) matching geogrid points
    h = xr.DataArray(
        np.full((dim_y + 1, dim_x + 1), 100.0, dtype=np.float64),
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    h.rio.write_crs("EPSG:4326", inplace=True)

    # DEM distance: (dim_y+1, dim_x+1, 3) matching geogrid points
    dem_distance = xr.DataArray(
        np.full((dim_y + 1, dim_x + 1, 3), 5000.0, dtype=np.float64),
        dims=["y", "x", "coord"],
        coords={"y": y, "x": x},
    )
    dem_distance.rio.write_crs("EPSG:4326", inplace=True)

    # Radar pixel indices: (dim_y, dim_x) - one per cell
    az_px = xr.DataArray(
        np.broadcast_to(np.arange(dim_y).reshape(dim_y, 1), (dim_y, dim_x)),
        dims=["y", "x"],
        coords={"y": y[:dim_y], "x": x[:dim_x]},
    )
    rg_px = xr.DataArray(
        np.broadcast_to(np.arange(dim_x).reshape(1, dim_x), (dim_y, dim_x)),
        dims=["y", "x"],
        coords={"y": y[:dim_y], "x": x[:dim_x]},
    )

    # Build simulation as a Dataset to hold all attributes with different dimensions
    simulation = xr.Dataset(
        data_vars={
            "h": h,
            "dem_distance": dem_distance,
            "az_px": az_px,
            "rg_px": rg_px,
        },
        coords={"y": y, "x": x},
    )
    return simulation


@pytest.fixture
def area_proj_csr_matrix():
    """Small CSR sparse matrix Dataset for compute_rtc_ap.

    Shape: (12, 10) - 12 triangles (rows), 10 radar pixels (columns)
    Has 5 non-zero elements.
    """
    data = np.array([1.0, 2.0, 1.5, 0.5, 3.0], dtype=np.float64)
    indices = np.array([0, 3, 5, 7, 9], dtype=np.int64)
    indptr = np.array([0, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int64)
    shape_arr = np.array([12, 10])

    ds = xr.Dataset(
        data_vars={
            "data": ("point", data),
            "indices": ("point", indices),
            "indptr": ("indptr", indptr),
            "nnz": ((), np.int64(data.shape[0])),
            "shape": ("dim", shape_arr),
        }
    )
    return ds


@pytest.fixture
def area_proj_geocoded_csr_matrix():
    """CSR matrix for compute_geocoded_mask tests.

    The CSR matrix has n_rows rows which gets reshaped to (dim_y, dim_x, 4).
    For output_shape=(4, 6): n_rows must equal 4*6*4 = 96.
    Shape: (96, 10) - 96 rows, 10 columns, 4 non-zero elements.
    """
    data = np.array([0.5, 0.3, 0.7, 0.2], dtype=np.float32)
    indices = np.array([1, 4, 6, 8], dtype=np.int64)
    # indptr: 97 entries for 96 rows
    indptr = np.array([0, 1, 2, 3, 4] + [4] * 92, dtype=np.int64)
    shape_arr = np.array([96, 10])

    ds = xr.Dataset(
        data_vars={
            "data": ("point", data),
            "indices": ("point", indices),
            "indptr": ("indptr", indptr),
            "nnz": ((), np.int64(4)),
            "shape": ("dim", shape_arr),
        }
    )
    return ds


@pytest.fixture
def get_w_simple_inputs():
    """Inputs for get_w with a simple small radar chunk.

    Uses a 2x3 geogrid (2 y-points, 3 x-points = 6 vertices).
    The triangulation indices must match the vertex count of x/y arrays.
    """
    from cop_ard_core.area_projection import triangulate_regular

    # Radar/geogrid points: 2 y-points, 3 x-points = 6 vertices
    x = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5], dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Triangulation for 2 rows, 3 columns of points => 1 cell in y, 2 cells in x
    triang = triangulate_regular(ysize=2, xsize=3).astype(np.int64)

    # bbox must have integer-compatible values for the numba kernel
    bbox = np.array([0.0, 0.0, 4.0, 2.0])  # [min_x, min_y, max_x, max_y]
    return {
        "x": x,
        "y": y,
        "triang": triang,
        "ymin": 0,
        "ymin_global": 0,
        "xmin": 0,
        "xmin_global": 0,
        "xsize_global": 10,
        "bbox": bbox,
        "expected_nlooks": 1,
    }


# ===========================================================================
# esd fixtures
# ===========================================================================


@pytest.fixture
def esd_acquisition_date():
    """1D acquisition_date DataArray for ESD tests."""
    return xr.DataArray(
        data=np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]"),
        dims=["acquisition_id"],
    )


@pytest.fixture
def esd_burst():
    """Minimal burst DataArray with required attrs for compute_azimuth_shifts.

    Required attrs keys:
      - burst.attrs["burst_meta"]["Grid"]["Col"]["DeltaKCOAPoly"]["Coefs"] : 1D array of poly coeffs
      - burst.attrs["burst_meta"]["Grid"]["Col"]["SS"] : sweep speed (float)
      - burst.slant_range_m : 1D coordinate array
      - burst.azimuth_m : 1D coordinate array
    """
    dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    data = np.array(
        [[[(i + 1) * (j + 1) * (k + 1) * (1 + 0.1j) for k in range(3)] for j in range(3)] for i in range(3)],
        dtype=np.complex128,
    )
    slant_range_m = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    azimuth_m = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    annot = {
        "Grid": {
            "Col": {
                "DeltaKCOAPoly": {"Coefs": np.array([1e3, 2e3, 0.0])},
                "SS": 0.01,
            }
        }
    }
    burst = xr.DataArray(
        data,
        dims=["acquisition_id", "azimuth_pixel", "slant_range_time"],
        coords={
            "acquisition_id": dates,
            "azimuth_pixel": np.arange(3),
            "slant_range_time": np.arange(3),
            "slant_range_m": ("slant_range_time", slant_range_m),
            "azimuth_m": ("azimuth_pixel", azimuth_m),
        },
    )
    burst = burst.assign_attrs({"meta": annot})
    burst.attrs = {"burst_meta": annot}
    return burst


@pytest.fixture
def esd_reference_pair(esd_burst):
    """Two reference DataArrays with acquisition_date coordinate.

    The acquisition_date dimension contains the same dates as the burst so
    that `all(reference_date in e.acquisition_date)` returns True.
    """
    dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    data0 = np.array(
        [[[(i + 1) * (j + 1) * (k + 1) * (1 + 0.1j) for k in range(3)] for j in range(3)] for i in range(3)],
        dtype=np.complex128,
    )
    data1 = np.array(
        [[[(i + 1) * (j + 1) * (k + 1) * (1 - 0.1j) for k in range(3)] for j in range(3)] for i in range(3)],
        dtype=np.complex128,
    )
    ref0 = xr.DataArray(
        data0,
        dims=["acquisition_id", "azimuth_pixel", "slant_range_time"],
        coords={
            "acquisition_id": dates,
            "azimuth_pixel": np.arange(3),
            "slant_range_time": np.arange(3),
        },
    )
    ref1 = xr.DataArray(
        data1,
        dims=["acquisition_id", "azimuth_pixel", "slant_range_time"],
        coords={
            "acquisition_id": dates,
            "azimuth_pixel": np.arange(3),
            "slant_range_time": np.arange(3),
        },
    )
    return ref0, ref1


@pytest.fixture
def esd_dataset():
    """Small ESD dataset for find_global_shift and optimize_curvature.

    Shape: acquisition_date=5, reference_date=5, stacked=(5,).
    mean_shift is NaN on the diagonal (self-interferogram) and valid elsewhere.
    """
    acquisition_dates = np.array(
        ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"],
        dtype="datetime64[ns]",
    )
    n = len(acquisition_dates)
    n_ids = 2
    stacked_ids = np.arange(n)
    stacked_dates = acquisition_dates.copy()

    # Use empty object array then assign tuples to keep 1D shape
    stacked = np.empty(n, dtype=object)
    for i in range(n):
        stacked[i] = (int(stacked_ids[i]), stacked_dates[i])

    burst_id = [str(i) for i in range(n_ids)]

    az_times = np.array(
        [[1_000_000_000 * i * j for i in range(n)] for j in range(1, n_ids + 1)], dtype="timedelta64[ns]"
    )
    sr_times = np.array([[500 * i * j for i in range(n)] for j in range(1, n_ids + 1)], dtype="timedelta64[ns]")

    # Build mean_shift: NaN on diagonal, valid off-diagonal
    mean_shift_data = np.full((n_ids, n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                mean_shift_data[:, i, j] = float(i - j) * 0.5

    mean_phase_data = np.full((n_ids, n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                mean_phase_data[:, i, j] = float(i - j) * np.pi / 4

    coherence_data = np.full((n_ids, n, n), 0.8)

    wrap_cycle_data = np.full((n_ids, n, n), 0.2)

    esd = xr.Dataset(
        data_vars={
            "mean_shift": (
                ["burst_id", "reference_date", "acquisition_date"],
                mean_shift_data,
            ),
            "mean_phase": (
                ["burst_id", "reference_date", "acquisition_date"],
                mean_phase_data,
            ),
            "coherence": (
                ["burst_id", "reference_date", "acquisition_date"],
                coherence_data,
            ),
            "wrap_cycle": (
                ["burst_id", "reference_date", "acquisition_date"],
                wrap_cycle_data,
            ),
            "azimuth_time": (["burst_id", "acquisition_date"], np.array(az_times, dtype="timedelta64[ns]")),
            "slant_range_time": (["burst_id", "acquisition_date"], np.array(sr_times, dtype="timedelta64[ns]")),
        },
        coords={
            "acquisition_date": acquisition_dates,
            "reference_date": acquisition_dates,
            "stacked": ("stacked", stacked),
            "burst_id": burst_id,
        },
        attrs={
            "coregistration_reference_date": np.datetime64("2020-01-01", "ns"),
        },
    )
    return esd


# ===========================================================================
# warp_gslc fixtures
# ===========================================================================


@pytest.fixture
def warp_gslc_gstack():
    """Geocoded stack (gstack) for warp_gslc.

    Dims: (polarization, acquisition_date, y, x)
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    pols = np.array(["VV", "VH"])
    ny, nx = 15, 15
    y = np.linspace(0.0, 900.0, ny)
    x = np.linspace(0.0, 900.0, nx)

    n_acq = len(acq_dates)
    n_pol = len(pols)

    # Create complex data with known values
    data = np.zeros((n_pol, n_acq, ny, nx), dtype=np.complex128)
    for i in range(n_acq):
        for j in range(n_pol):
            data[j, i] = (1.0 + 0.1j) * (i + 1) * (j + 1)

    gstack = xr.DataArray(
        data,
        dims=["polarization", "acquisition_date", "y", "x"],
        coords={
            "polarization": pols,
            "acquisition_date": acq_dates,
            "y": y,
            "x": x,
        },
    )
    gstack = gstack.rio.write_crs("EPSG:32632", inplace=True)
    return gstack


@pytest.fixture
def warp_gslc_gphase_model():
    """Geocoded phase model (gphase_model) for warp_gslc.

    Dims: (acquisition_date, y, x)
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    ny, nx = 15, 15
    y = np.linspace(0.0, 900.0, ny)
    x = np.linspace(0.0, 900.0, nx)

    n_acq = len(acq_dates)
    ny = len(y)
    nx = len(x)

    data = np.zeros((n_acq, ny, nx), dtype=np.float64)
    for i in range(n_acq):
        data[i] = float(i) * np.pi / 4.0

    gphase_model = xr.DataArray(
        data,
        dims=["acquisition_date", "y", "x"],
        coords={
            "acquisition_date": acq_dates,
            "y": y,
            "x": x,
        },
    )
    gphase_model = gphase_model.rio.write_crs("EPSG:32632", inplace=True)
    return gphase_model


@pytest.fixture
def warp_gslc_cstack():
    """Coregistered stack (cstack) for warp_gslc.

    Dims: (polarization, acquisition_date, azimuth_time, slant_range_time)
    azimuth_time uses datetime64 matching simulation.azimuth_time.
    Uses 10x10 grid to accommodate kernel margin.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    pols = np.array(["VV", "VH"])
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    # 10x10 grid to accommodate kernel margin of 2
    n_az = 15
    n_sr = 15
    az_times = np.array(
        [base_epoch + np.timedelta64(i * 1000000000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
    )
    sr_times = np.arange(0, n_sr) * 1.0  # 10 range pixels, 1s spacing

    n_acq = len(acq_dates)
    n_pol = len(pols)
    n_az = len(az_times)
    n_sr = len(sr_times)

    data = np.zeros((n_pol, n_acq, n_az, n_sr), dtype=np.complex128)
    for i in range(n_acq):
        for j in range(n_pol):
            data[j, i] = (1.0 + 0.1j) * (i + 1) * (j + 1)

    cstack = xr.DataArray(
        data,
        dims=["polarization", "acquisition_date", "azimuth_time", "slant_range_time"],
        coords={
            "polarization": pols,
            "acquisition_date": acq_dates,
            "azimuth_time": az_times,
            "slant_range_time": sr_times,
        },
    )
    # Add RMA metadata required by extract_grid (called from do_deramp)
    cstack = cstack.assign_attrs(
        {
            "burst_meta": {
                "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                "ImageData": {"SCPPixel": {"Row": 7, "Col": 7}},
                "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
            }
        }
    )
    return cstack


@pytest.fixture
def warp_gslc_phase_model():
    """Phase model for coregistered data (phase_model).

    Dims: (acquisition_date, azimuth_time, slant_range_time)
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    n_az = 15
    n_sr = 15
    az_times = np.array(
        [base_epoch + np.timedelta64(i * 1000000000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
    )
    sr_times = np.arange(0, n_sr) * 1.0

    n_acq = len(acq_dates)
    n_az = len(az_times)
    n_sr = len(sr_times)

    data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
    for i in range(n_acq):
        data[i] = float(i) * np.pi / 4.0

    phase_model = xr.DataArray(
        data,
        dims=["acquisition_date", "azimuth_time", "slant_range_time"],
        coords={
            "acquisition_date": acq_dates,
            "azimuth_time": az_times,
            "slant_range_time": sr_times,
        },
    )
    return phase_model


@pytest.fixture
def warp_gslc_phase_demod():
    """Phase demodulation LUT (phase_demod) for warp_gslc.

    Dims: (acquisition_date, azimuth_time, slant_range_time)
    Has attrs: reference, scale, deg
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    n_az = 15
    n_sr = 15
    az_times = np.array(
        [base_epoch + np.timedelta64(i * 1000000000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
    )
    sr_times = np.arange(0, n_sr) * 1.0

    n_acq = len(acq_dates)
    n_az = len(az_times)
    n_sr = len(sr_times)

    # Create a slowly varying phase surface
    data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
    for i in range(n_acq):
        for j in range(n_az):
            for k in range(n_sr):
                data[i, j, k] = float(j + k) * 0.1 + float(i) * 0.05

    phase_demod = xr.DataArray(
        data,
        dims=["acquisition_date", "azimuth_time", "slant_range_time"],
        coords={
            "acquisition_date": acq_dates,
            "azimuth_time": az_times,
            "slant_range_time": sr_times,
        },
    )
    phase_demod = phase_demod.assign_attrs(
        {
            "reference": {"slant_range_time": 0.0, "azimuth_time": "2020-01-01T00:00:00"},
            "scale": {"slant_range_time": 1.0, "azimuth_time": 1e-9},
            "deg": {"slant_range_time": 1, "azimuth_time": 1},
        }
    )
    return phase_demod


@pytest.fixture
def warp_gslc_anf():
    """Area normalizer field (anf) for warp_gslc.

    Dims: (acquisition_date, azimuth_time, slant_range_time)
    Values > eps for gamma flattening.
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    n_az = 15
    n_sr = 15
    az_times = np.array(
        [base_epoch + np.timedelta64(i * 1000000000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
    )
    sr_times = np.arange(0, n_sr) * 1.0

    n_acq = len(acq_dates)
    n_az = len(az_times)
    n_sr = len(sr_times)

    # Area normalizer values around 1.0 (always > eps=1e-3)
    data = np.ones((n_acq, n_az, n_sr), dtype=np.float64) * 1.0

    anf = xr.DataArray(
        data,
        dims=["acquisition_date", "azimuth_time", "slant_range_time"],
        coords={
            "acquisition_date": acq_dates,
            "azimuth_time": az_times,
            "slant_range_time": sr_times,
        },
    )
    return anf


@pytest.fixture
def warp_gslc_simulation():
    """Simulation dataset for warp_gslc.

    Has 1D y, x coordinates (for rio.resolution) and 2D data variables
    azimuth_time (datetime64), slant_range_time (float). After isel(acquisition_date=0).squeeze()
    the result is a 2D Dataset with x, y coords and azimuth_time, slant_range_time vars.
    Uses 10x10 grid.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
    ny, nx = 15, 15
    y = np.linspace(0.0, 900.0, ny)
    x = np.linspace(0.0, 900.0, nx)

    # Base epoch for azimuth_time
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    # azimuth_time as datetime64 matching cstack.azimuth_time (15 pixels)
    az_base = np.array([i * 1000000000 for i in range(15)], dtype="timedelta64[ns]")
    az_grid = np.array([[base_epoch + az_base[j] for j in range(nx)] for i in range(ny)], dtype="datetime64[ns]")

    # slant_range_time as float (seconds from reference)
    sr_grid = np.outer(np.ones(ny, dtype=np.float64), np.arange(nx, dtype=np.float64))

    sim = xr.Dataset(
        data_vars={
            "azimuth_time": (("y", "x"), az_grid),
            "slant_range_time": (("y", "x"), sr_grid),
        },
        coords={"y": y, "x": x, "acquisition_date": acq_dates},
    )
    sim = sim.rio.write_crs("EPSG:32632", inplace=True)
    return sim


@pytest.fixture
def warp_gslc_kernel():
    """Default sinc interpolation kernel for warp_gslc.

    Includes all keys required by sinc_warp: kernel_shape, ov_fact, sinc_x, sinc_y.
    """
    from cop_ard_core.warp import sinc

    kernel_shape = (3, 3)
    ov_fact = 100
    sinc_x = sinc(size=kernel_shape[1], ov_fact=ov_fact)
    sinc_y = sinc(size=kernel_shape[0], ov_fact=ov_fact)
    return {
        "kernel_shape": kernel_shape,
        "ov_fact": ov_fact,
        "sinc_x": sinc_x,
        "sinc_y": sinc_y,
    }


@pytest.fixture
def warp_gslc_all_inputs(
    warp_gslc_gstack,
    warp_gslc_gphase_model,
    warp_gslc_simulation,
    warp_gslc_kernel,
    warp_gslc_cstack,
    warp_gslc_phase_model,
    warp_gslc_phase_demod,
    warp_gslc_anf,
):
    """All inputs for warp_gslc combined as a dict."""
    return {
        "gstack": warp_gslc_gstack,
        "gphase_model": warp_gslc_gphase_model,
        "simulation": warp_gslc_simulation,
        "kernel": warp_gslc_kernel,
        "cstack": warp_gslc_cstack,
        "phase_model": warp_gslc_phase_model,
        "phase_demod": warp_gslc_phase_demod,
        "anf": warp_gslc_anf,
        "do_reramp": False,
        "gamma_flattening": False,
    }


# ===========================================================================
# warp_dazdrg fixtures
# ===========================================================================


@pytest.fixture
def warp_dazdrg_cstack():
    """Small cstack (coregistered stack) for warp_dazdrg tests.

    Dims: (acquisition_id, azimuth_time, slant_range_time)
    Uses 3 acquisition dates, 5 azimuth pixels, 5 range pixels.
    """
    acq_dates = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az_times = np.array([base_epoch + np.timedelta64(i * 1000000000, "ns") for i in range(5)], dtype="datetime64[ns]")
    sr_times = np.arange(5) * 0.5
    data = np.zeros((3, 5, 5), dtype=np.complex128)
    for i in range(3):
        data[i] = (1.0 + 0.1j) * (i + 1)

    return xr.DataArray(
        data,
        dims=["acquisition_id", "azimuth_time", "slant_range_time"],
        coords={
            "acquisition_id": acq_dates,
            "azimuth_time": az_times,
            "slant_range_time": sr_times,
        },
    )


@pytest.fixture
def warp_dazdrg_reference_simulation():
    """Reference simulation dataset for warp_dazdrg tests.

    Has x, y geographic coordinates and 2D azimuth_time / slant_range_time.
    The azimuth_time varies across both y and x to create non-degenerate
    triangles for interpolation.
    """
    # Geographic coordinates matching the radar bbox
    y = np.linspace(0.0, 400.0, 5)
    x = np.linspace(0.0, 400.0, 5)
    acq_dates = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")

    # Create azimuth_time that varies across both y and x (non-degenerate grid)
    az_base = np.array([i * 1000000000 for i in range(5)], dtype="timedelta64[ns]")
    sr_base = np.arange(5) * 0.5
    az_grid = np.array(
        [[base_epoch + az_base[j] + np.timedelta64(i * 500000000, "ns") for j in range(5)] for i in range(5)]
    )
    # slant_range_time varies across x only
    sr_grid = np.outer(np.ones(5, dtype=np.float64), sr_base)

    # add acquisition date dimension
    az_grid = np.tile(np.expand_dims(az_grid, 0), (3, 1, 1))
    sr_grid = np.tile(np.expand_dims(sr_grid, 0), (3, 1, 1))

    dem_distance = np.full((acq_dates.size, y.size, x.size, 3), 1)
    h = np.full((y.size, x.size), 1)

    simulation = xr.Dataset(
        coords={
            "acquisition_id": acq_dates,
            "y": y,
            "x": x,
            "axis": [0, 1, 2],
        },
        # Create empty Dask arrays for each data variable.
        data_vars={
            "azimuth_time": (["acquisition_id", "y", "x"], az_grid),
            "slant_range_time": (["acquisition_id", "y", "x"], sr_grid),
            "dem_distance": (["acquisition_id", "y", "x", "axis"], dem_distance),
            "h": (["y", "x"], h),
        },
    )
    return simulation


@pytest.fixture
def warp_dazdrg_secondary_simulation():
    """Secondary simulation dataset for warp_dazdrg tests.

    Similar structure to reference but with slightly shifted times.
    """
    y = np.linspace(0.0, 400.0, 5)
    x = np.linspace(0.0, 400.0, 5)
    acq_dates = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az_base = np.array([i * 1000000000 for i in range(5)], dtype="timedelta64[ns]")
    az_grid = np.array(
        [[base_epoch + az_base[j] + np.timedelta64(100000000, "ns") for j in range(5)] for i in range(5)]
    )
    sr_grid = np.outer(np.ones(5, dtype=np.float64), np.arange(5) * 0.5)

    sim = xr.Dataset(
        data_vars={
            "azimuth_time": (("y", "x"), az_grid),
            "slant_range_time": (("y", "x"), sr_grid),
        },
        coords={"y": y, "x": x, "acquisition_id": acq_dates},
    )
    return sim


@pytest.fixture
def warp_dazdrg_reference_ml(warp_dazdrg_reference_simulation):
    """Reference multilook (reference_ml) for warp_dazdrg tests.

    Contains 'min' and 'max' datasets from coarsened simulation,
    with y, x coordinates needed by get_geo_from_radar_bbox.
    """
    sim = warp_dazdrg_reference_simulation
    # Coarsen the simulation to create a downsampled version
    coarsen_config = {"y": 5, "x": 5}
    ref_sel = (
        sim.sel(acquisition_id="2020-01-01").drop_vars(["dem_distance", "h", "axis"], errors="ignore").astype(float)
    )

    min_ds = ref_sel.coarsen(coarsen_config, boundary="pad", coord_func="min").min()
    max_ds = ref_sel.coarsen(coarsen_config, boundary="pad", coord_func="max").max()

    # Ensure azimuth_time is datetime64[ns]
    min_ds["azimuth_time"] = min_ds.azimuth_time.astype("datetime64[ns]")
    max_ds["azimuth_time"] = max_ds.azimuth_time.astype("datetime64[ns]")

    return {"min": min_ds, "max": max_ds}


@pytest.fixture
def warp_dazdrg_margin():
    """Margin dict for warp_dazdrg tests.

    azimuth_time margin must be timedelta64 (to match datetime64 arithmetic in get_radar_bbox).
    slant_range_time margin is a float.
    """
    return {"azimuth_time": np.timedelta64(0, "ns"), "slant_range_time": 0.0}


@pytest.fixture
def warp_dazdrg_inputs(
    warp_dazdrg_cstack,
    warp_dazdrg_reference_simulation,
    warp_dazdrg_secondary_simulation,
    warp_dazdrg_reference_ml,
    warp_dazdrg_margin,
):
    """All inputs for warp_dazdrg combined as keyword arguments.

    daz/drg are pre-made 3D DataArrays with acquisition_date dimension
    so the source code skips its broken creation logic and uses these valid ones.
    """
    cstack = warp_dazdrg_cstack
    shape = cstack.shape  # (3, 5, 5)
    dims = cstack.dims  # ('acquisition_date', 'azimuth_time', 'slant_range_time')

    daz = xr.DataArray(
        np.zeros(shape, dtype="timedelta64[ns]"),
        dims=dims,
        coords={d: cstack[d] for d in dims},
        attrs={"recompute": False},
    )
    drg = xr.DataArray(
        np.zeros(shape, dtype=np.float64),
        dims=dims,
        coords={d: cstack[d] for d in dims},
        attrs={"recompute": False},
    )

    return {
        "cstack": warp_dazdrg_cstack,
        "daz": daz,
        "drg": drg,
        "simulation_reference": warp_dazdrg_reference_simulation,
        "simulation_secondary": None,
        "reference_ml": warp_dazdrg_reference_ml,
        "margin": warp_dazdrg_margin,
        "coregistration_reference_id": "2020-01-01",
        "azimuth_grid_spacing": 1.0,
        "range_grid_spacing": 0.5,
        "compute_daz_drg": True,
    }


# ===========================================================================
# warp test fixtures
# ===========================================================================


@pytest.fixture
def image_7x7():
    """7x7 real-valued image for boundary testing."""
    return np.arange(49, dtype=np.float64).reshape(7, 7)


@pytest.fixture
def csi_slc():
    """SLC DataArray used to extract az0/rg0 reference values.

    Dims: (azimuth_time, slant_range_time) with datetime and float coords.
    Shape: (5, 4).
    """
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
    rg = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    data = np.ones((5, 4), dtype=np.complex128)
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": az, "slant_range_time": rg},
    )


@pytest.fixture
def csi_coreg_az():
    """Coregistration azimuth time grid (5x4).

    Values are datetime64[ns] representing shifted acquisition times.
    Shape matches csi_slc: (5, 4).
    """
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az_vals = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
    az_grid = np.array([[az_vals[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]")
    return xr.DataArray(az_grid, dims=["azimuth_time", "slant_range_time"])


@pytest.fixture
def csi_coreg_rg():
    """Coregistration slant range time grid (5x4).

    Values are float64 representing shifted slant range times.
    Shape matches csi_slc: (5, 4).
    """
    base_rg = np.arange(4, dtype=np.float64) * 0.5
    rg_grid = np.array([[base_rg[j] for j in range(4)] for i in range(5)], dtype=np.float64)
    return xr.DataArray(rg_grid, dims=["azimuth_time", "slant_range_time"])


@pytest.fixture
def csi_daz():
    """Azimuth correction DataArray (5x4) with zero offsets.

    dtype is timedelta64[ns].
    """
    return xr.DataArray(
        np.zeros((5, 4), dtype="timedelta64[ns]"),
        dims=["azimuth_time", "slant_range_time"],
    )


@pytest.fixture
def csi_drg():
    """Range correction DataArray (5x4) with zero offsets.

    dtype is float64.
    """
    return xr.DataArray(
        np.zeros((5, 4), dtype=np.float64),
        dims=["azimuth_time", "slant_range_time"],
    )


@pytest.fixture
def csi_etad_empty():
    """Empty ETAD correction grid (no data variables).

    Must have azimuth_time and slant_range_time coords matching csi_slc grid.
    """
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
    rg = np.arange(4, dtype=np.float64) * 0.5
    # Empty dataset but with required coordinates
    return xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})


@pytest.fixture
def csi_etad_nonempty():
    """Non-empty ETAD correction grid with azcorr and rgcorr variables.

    Shape: (5, 4) — needs at least 4 points per dimension for cubic interpolation.
    Values create known correction patterns for testing.
    """
    base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
    az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
    rg = np.arange(4, dtype=np.float64) * 0.5

    # Correction grid — azimuth correction: 0 ns everywhere
    azcorr = np.zeros((5, 4), dtype=np.float64)
    # Range correction: small gradient
    rgcorr = np.outer(np.ones(5), np.arange(4, dtype=np.float64) * 0.01)

    return xr.Dataset(
        data_vars={
            "azcorr": (["azimuth_time", "slant_range_time"], azcorr),
            "rgcorr": (["azimuth_time", "slant_range_time"], rgcorr),
        },
        coords={"azimuth_time": az, "slant_range_time": rg},
    )


@pytest.fixture
def csi_y_poly():
    """y_poly DataArray with polynomial attributes for azimuth correction.

    Shape: (2, 2).
    attrs: offset, scale_fact, type ("pixels").
    """
    poly_values = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    y_poly = xr.DataArray(poly_values, dims=["i", "j"])
    y_poly = y_poly.assign_attrs(
        {
            "off_y": 0,
            "offset": [np.float64(0.0), np.float64(0.0)],
            "scale_fact": [np.float64(1.0), np.float64(1.0)],
            "type": "pixels",
        }
    )
    return y_poly


@pytest.fixture
def csi_x_poly():
    """x_poly DataArray with polynomial attributes for range correction.

    Shape: (2, 2).
    attrs: offset, scale_fact, type ("pixels").
    """
    poly_values = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    x_poly = xr.DataArray(poly_values, dims=["i", "j"])
    x_poly = x_poly.assign_attrs(
        {
            "off_y": 0,
            "offset": [np.float64(0.0), np.float64(0.0)],
            "scale_fact": [np.float64(1.0), np.float64(1.0)],
            "type": "pixels",
        }
    )
    return x_poly


@pytest.fixture
def csi_inputs(csi_slc, csi_coreg_az, csi_coreg_rg, csi_daz, csi_drg, csi_etad_empty):
    """All basic inputs for compute_secondary_image_coordinates combined.

    No polynomial corrections, empty ETAD grid.
    """
    return {
        "coreg_azimuth_time": csi_coreg_az,
        "coreg_slant_range_time": csi_coreg_rg,
        "slc": csi_slc,
        "etad_correction_grid": csi_etad_empty,
        "daz": csi_daz,
        "drg": csi_drg,
        "azimuth_grid_spacing": 1.0,
        "range_grid_spacing": 0.5,
        "y_poly": None,
        "x_poly": None,
        "offset": {"azimuth_time": 0.0, "slant_range_time": 0.0},
    }


def _with_1d_coreg(inputs):
    """Return a copy of ``inputs`` with 1-D coreg coordinate vectors.

    The polynomial branch of ``compute_secondary_image_coordinates`` evaluates
    ``polygrid2d(c1, c2, poly)`` where ``c1``/``c2`` are derived from the coreg
    coordinates. Real callers (``warp_cslc``) pass the 1-D ``.azimuth_time`` /
    ``.slant_range_time`` coordinate vectors; passing the full 2-D grid instead
    makes ``polygrid2d`` return a 4-D array. Selecting a single line of the 2-D
    grid yields the equivalent 1-D vectors.
    """
    inputs = dict(inputs)
    inputs["coreg_azimuth_time"] = inputs["coreg_azimuth_time"].isel(slant_range_time=0)
    inputs["coreg_slant_range_time"] = inputs["coreg_slant_range_time"].isel(azimuth_time=0)
    return inputs


@pytest.fixture
def csi_inputs_with_y_poly(csi_inputs, csi_y_poly):
    """Inputs with y_poly correction enabled."""
    inputs = _with_1d_coreg(csi_inputs)
    inputs["y_poly"] = csi_y_poly
    return inputs


@pytest.fixture
def csi_inputs_with_x_poly(csi_inputs, csi_x_poly):
    """Inputs with x_poly correction enabled."""
    inputs = _with_1d_coreg(csi_inputs)
    inputs["x_poly"] = csi_x_poly
    return inputs


@pytest.fixture
def csi_inputs_with_both_polys(csi_inputs, csi_y_poly, csi_x_poly):
    """Inputs with both y_poly and x_poly corrections enabled."""
    inputs = _with_1d_coreg(csi_inputs)
    inputs["y_poly"] = csi_y_poly
    inputs["x_poly"] = csi_x_poly
    return inputs


# ===========================================================================
# cmlp_gmlp fixtures (get_current_reference)
# ===========================================================================


@pytest.fixture
def cstack_3_dates():
    """3-date cstack with string acquisition_date coordinates."""
    return xr.DataArray(
        np.ones((3, 4, 4)),
        dims=["acquisition_date", "az", "rg"],
        coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"], "acquisition_id": xr.DataArray(np.atleast_1d(["2020-01-01", "2020-01-13", "2020-01-25"]), dims=["acquisition_date"])},
    )


@pytest.fixture
def last_reference():
    """A last_reference DataArray with a date before cstack's start.

    Uses scalar np.datetime64 for acquisition_date coordinate, which is
    required by the function's date_l parsing logic.
    """
    return xr.DataArray(
        np.ones((1, 4, 4)) * 99.0,
        dims=["acquisition_date", "az", "rg"],
        coords={"acquisition_date": np.datetime64("2019-12-15")},
    )
