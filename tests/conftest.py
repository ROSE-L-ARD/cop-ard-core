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
