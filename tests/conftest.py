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
def compute_shadow_dt():
    return {"azimuth_time": 0.01, "slant_range_time": 0.5}


@pytest.fixture
def compute_shadow_margin():
    return {"azimuth_time": 0.0, "slant_range_time": 0.0}
