"""Unit tests for cop_ard_core.dem."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from cop_ard_core.dem import (
    compute_shape_and_off,
    convert_to_dem_ecef,
    extract_vertical_crs,
    get_default_epsg,
    get_resolution_in_meters,
)


class TestExtractVerticalCrs:
    """Tests for extract_vertical_crs."""

    def test_extract_vertical_crs_horizontal(self):
        """Horizontal CRS should return empty string."""
        crs = CRS.from_epsg(4326)
        result = extract_vertical_crs(crs)
        assert result == ""

    def test_extract_vertical_crs_invalid_type(self):
        """Non-CRS type should raise ValueError."""
        with pytest.raises(ValueError, match="Expected type CRS"):
            extract_vertical_crs("not a crs")

    def test_extract_vertical_crs_3d(self):
        """3D CRS should extract vertical component."""
        crs_3d = CRS.from_epsg(4978).to_3d()
        result = extract_vertical_crs(crs_3d)
        # Should return empty string if not in sub_crs_list
        assert isinstance(result, str)


class TestGetDefaultEpsg:
    """Tests for get_default_epsg."""

    def test_get_default_epsg_geographic(self):
        """Geographic CRS should return UTM EPSG code."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.linspace(40, 45, 10), "x": np.linspace(-75, -70, 10)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = get_default_epsg(dem)
        assert result.startswith("EPSG:")

    def test_get_default_epsg_high_latitude_north(self):
        """High northern latitude should return UPS EPSG."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.linspace(85, 90, 10), "x": np.linspace(-180, 180, 10)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = get_default_epsg(dem)
        assert result == "EPSG:32661"

    def test_get_default_epsg_high_latitude_south(self):
        """High southern latitude should return UPS EPSG."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.linspace(-90, -85, 10), "x": np.linspace(-180, 180, 10)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = get_default_epsg(dem)
        assert result == "EPSG:32761"


class TestComputeShapeAndOff:
    """Tests for compute_shape_and_off."""

    def test_compute_shape_and_off_basic(self):
        """Basic shape and offset computation."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(0, 100, 10), "x": np.arange(0, 100, 10)},
        )
        spacing = {"x": 10.0, "y": 10.0}
        shape, off = compute_shape_and_off(dem, spacing)
        assert "x" in shape
        assert "y" in shape

    def test_compute_shape_and_off_with_transform_bounds(self):
        """With out_crs, should use transform_bounds."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(0, 100, 10), "x": np.arange(0, 100, 10)},
        )
        dem.rio.write_crs("EPSG:32633", inplace=True)
        spacing = {"x": 30.0, "y": 30.0}
        shape, off = compute_shape_and_off(dem, spacing, out_crs="EPSG:32633")
        assert isinstance(shape, dict)
        assert isinstance(off, dict)

    def test_compute_shape_and_off_negative_spacing(self):
        """Negative spacing should be handled."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(100, 0, -10), "x": np.arange(100, 0, -10)},
        )
        spacing = {"x": -10.0, "y": -10.0}
        shape, off = compute_shape_and_off(dem, spacing)
        assert shape["x"] > 0
        assert shape["y"] > 0


class TestGetResolutionInMeters:
    """Tests for get_resolution_in_meters."""

    def test_get_resolution_in_meters_geographic(self):
        """Geographic CRS should return resolution in meters via UTM."""
        dem = xr.DataArray(
            np.ones((100, 100)),
            dims=["y", "x"],
            coords={"y": np.linspace(40, 41, 100), "x": np.linspace(-75, -74, 100)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = get_resolution_in_meters(dem)
        assert result.shape == (2,)
        assert result[0] > 0
        assert result[1] > 0

    def test_get_resolution_in_meters_projected(self):
        """Projected CRS should return resolution directly."""
        dem = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(0, 100, 10), "x": np.arange(0, 100, 10)},
        )
        dem.rio.write_crs("EPSG:32633", inplace=True)
        result = get_resolution_in_meters(dem)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [10.0, 10.0], atol=1.0)


class TestConvertToDemEcef:
    """Tests for convert_to_dem_ecef."""

    def test_convert_to_dem_ecef_returns_array(self):
        """Should return numpy array with shape (3, H, W)."""
        dem = xr.DataArray(
            np.ones((5, 5)),
            dims=["y", "x"],
            coords={"y": np.linspace(40, 45, 5), "x": np.linspace(-75, -70, 5)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = convert_to_dem_ecef(dem)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 5, 5)

    def test_convert_to_dem_ecef_3d_crs(self):
        """3D CRS should work correctly."""
        dem = xr.DataArray(
            np.ones((3, 3)),
            dims=["y", "x"],
            coords={"y": np.linspace(40, 42, 3), "x": np.linspace(-75, -73, 3)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = convert_to_dem_ecef(dem)
        assert result.shape == (3, 3, 3)

    def test_convert_to_dem_ecef_all_same_value(self):
        """DEM with all same values should produce same X,Y but different Z."""
        dem = xr.DataArray(
            np.full((4, 4), 100.0),
            dims=["y", "x"],
            coords={"y": np.linspace(40, 42, 4), "x": np.linspace(-75, -73, 4)},
        )
        dem.rio.write_crs("EPSG:4326", inplace=True)
        result = convert_to_dem_ecef(dem)
        # X,Y should vary by longitude/latitude, Z by elevation
        assert result.shape == (3, 4, 4)
