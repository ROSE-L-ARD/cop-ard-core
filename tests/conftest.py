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
def complex_stack_3x3x3():
    """3D complex DataArray (acquisition x azimuth x range)."""
    data = np.array([[(i + 1) * (j + 1) * (k + 1) * (1 + 0.25j) for k in range(3)] for j in range(3)] for i in range(3))
    data = np.array(data, dtype=np.complex128)
    return xr.DataArray(
        data,
        dims=["acquisition_date", "azimuth_pixel", "slant_range_time"],
        coords={
            "acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"],
            "azimuth_pixel": np.arange(3),
            "slant_range_time": np.arange(3),
        },
    )


@pytest.fixture
def real_3x3():
    """3x3 real-valued DataArray."""
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={
            "azimuth_time": np.arange(3),
            "slant_range_time": np.arange(3),
        },
    )


@pytest.fixture
def dataset_with_coords():
    """Small xr.Dataset with azimuth_time and slant_range_time coordinates."""
    az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
    sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    data = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": az, "slant_range_time": sr},
    ).to_dataset(name="data")


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
def small_real_4x4():
    """4x4 real-valued DataArray for multilook operations."""
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
    )


@pytest.fixture
def small_complex_4x4():
    """4x4 complex-valued DataArray for multilook operations."""
    data = np.array(
        [[i * 4 + j + 1j * (i * 4 + j) for j in range(4)] for i in range(4)],
        dtype=np.complex128,
    )
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
    )


@pytest.fixture
def coherence_data_3x3():
    """3x3 real DataArray with coherence-like values (0 to 1)."""
    data = np.array([[0.9, 0.7, 0.5], [0.8, 0.6, 0.4], [0.3, 0.2, 0.1]])
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
    )


@pytest.fixture
def phase_data_3x3():
    """3x3 DataArray with wrapped phase values."""
    data = np.array(
        [[0.0, np.pi / 2, np.pi], [3 * np.pi / 2, np.pi / 4, -np.pi / 4], [-np.pi / 2, -np.pi / 3, np.pi / 6]]
    )
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
    )


@pytest.fixture
def nan_3x3():
    """3x3 DataArray with NaN values."""
    data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan], [7.0, 8.0, 9.0]], dtype=np.float64)
    return xr.DataArray(
        data,
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
    )


@pytest.fixture
def daz_drg_data():
    """Small daz/drg DataArrays for testing."""
    daz = xr.DataArray(
        np.array(
            [["2020-01-01T00:00:00", "2020-01-01T00:00:01"], ["2020-01-01T00:00:02", "2020-01-01T00:00:03"]],
            dtype="datetime64[ns]",
        ),
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
    )
    drg = xr.DataArray(
        np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float64),
        dims=["azimuth_time", "slant_range_time"],
        coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
    )
    return daz, drg


@pytest.fixture
def triang_4x4():
    """Triangle indices for a 4x4 grid (3x3 cells = 36 triangles)."""
    from cop_ard_core.area_projection import triangulate_regular
    return triangulate_regular(ysize=4, xsize=4)


@pytest.fixture
def compute_shadow_radar_grid():
    import rioxarray
    data = np.zeros((2, 3), dtype=np.float64)
    az = np.array([0.0, 1.0], dtype=np.float64)
    sr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    grid = xr.DataArray(data, dims=["azimuth_time", "slant_range_time"],
                        coords={"azimuth_time": az, "slant_range_time": sr})
    grid.rio.write_crs("EPSG:4326", inplace=True)
    return grid


@pytest.fixture
def compute_shadow_h():
    import rioxarray
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    h = xr.DataArray(np.full((len(y), len(x)), 100.0), dims=["y", "x"],
                     coords={"y": y, "x": x})
    h = h.expand_dims(acquisition_date=["2020-01-01"])
    h.rio.write_crs("EPSG:4326", inplace=True)
    return h


@pytest.fixture
def compute_shadow_dem_distance():
    import rioxarray
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    data = np.full((len(y), len(x), 3), 5000.0, dtype=np.float64)
    dem_dist = xr.DataArray(data, dims=["y", "x", "coord"], coords={"y": y, "x": x})
    dem_dist = dem_dist.expand_dims(acquisition_date=["2020-01-01"])
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
    return xr.DataArray(data, dims=["y", "x"]).expand_dims(acquisition_date=["2020-01-01"])


@pytest.fixture
def compute_shadow_slant_range_time():
    y = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
    x = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
    data = np.zeros((len(y), len(x)), dtype=np.float64)
    for i in range(len(x)):
        data[:, i] = float(i) * 0.5
    return xr.DataArray(data, dims=["y", "x"]).expand_dims(acquisition_date=["2020-01-01"])


@pytest.fixture
def compute_shadow_dt():
    return {"azimuth_time": 0.01, "slant_range_time": 0.5}


@pytest.fixture
def compute_shadow_local_incidence_angle():
    import rioxarray
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
    min_ds = xr.Dataset({
        "azimuth_time": xr.DataArray(az_min, dims=["y", "x"]),
        "slant_range_time": xr.DataArray(sr_min, dims=["y", "x"]),
    }, coords={"y": coarse_y, "x": coarse_x,
               "acquisition_date": np.array(["2020-01-01"], dtype="datetime64[ns]")})
    max_ds = xr.Dataset({
        "azimuth_time": xr.DataArray(az_max, dims=["y", "x"]),
        "slant_range_time": xr.DataArray(sr_max, dims=["y", "x"]),
    }, coords={"y": coarse_y, "x": coarse_x,
               "acquisition_date": np.array(["2020-01-01"], dtype="datetime64[ns]")})
    return {"acquisition_date": np.array(["2020-01-01"], dtype="datetime64[ns]"),
            "min": min_ds, "max": max_ds}


@pytest.fixture
def compute_shadow_margin():
    return {"azimuth_time": 0.0, "slant_range_time": 0.0}


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
    import rioxarray

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
def area_proj_csr_matrix_empty():
    """Empty CSR sparse matrix Dataset for edge case testing."""
    shape_arr = np.array([2, 2])

    ds = xr.Dataset(
        data_vars={
            "data": ("point", np.array([], dtype=np.float64)),
            "indices": ("point", np.array([], dtype=np.int64)),
            "indptr": ("indptr", np.array([0, 0], dtype=np.int64)),
            "nnz": ((), np.int64(0)),
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
def triang_3x4():
    """Triangle indices for a 4x5 grid (3x4 cells = 24 triangles)."""
    from cop_ard_core.area_projection import triangulate_regular
    return triangulate_regular(ysize=4, xsize=5)


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
