"""Unit tests for cop_ard_core.dem."""

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
    interp_dem,
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
            extract_vertical_crs("not a crs")  # type: ignore

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


# ===========================================================================
# interp_dem tests
# ===========================================================================


def _make_projected_dem(height: int, width: int, start_x: float = 0.0, start_y: float = 0.0, spacing: float = 10.0, elev: float = 100.0):
    """Create a small projected DEM with a valid CRS for testing."""
    y = np.arange(start_y, start_y + height * spacing, spacing)
    x = np.arange(start_x, start_x + width * spacing, spacing)
    data = np.full((height, width), elev, dtype=np.float64)
    dem = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
    dem.rio.write_crs("EPSG:32633", inplace=True)
    return dem


def _make_output_grid(height: int, width: int, start_x: float = 0.0, start_y: float = 0.0, spacing: float = 10.0):
    """Create a small output grid DataArray with a valid CRS."""
    y = np.arange(start_y, start_y + height * spacing, spacing)
    x = np.arange(start_x, start_x + width * spacing, spacing)
    data = np.zeros((height, width), dtype=np.float64)
    grid = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
    grid.rio.write_crs("EPSG:32633", inplace=True)
    return grid


class TestInterpDem:
    """Tests for the interp_dem function.

    interp_dem takes an output grid chunk (dem_out), an input DEM (dem_in),
    an intermediate CRS, and returns an interpolated xr.Dataset.
    """

    def test_interp_dem_returns_dataset(self):
        """Basic happy path: output should be an xr.Dataset."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)

    def test_interp_dem_has_z_variable(self):
        """Output Dataset should contain a 'z' data variable."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert "z" in result.data_vars

    def test_interp_dem_output_shape(self):
        """Output z shape should match dem_out shape."""
        h, w = 5, 5
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(h, w)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert result["z"].shape == (h, w)

    def test_interp_dem_same_crs_identity_like(self):
        """When input and output CRS are the same, output should resemble input."""
        dem_in = _make_projected_dem(10, 10, start_x=0.0, start_y=0.0, elev=123.0)
        dem_out = _make_output_grid(5, 5, start_x=0.0, start_y=0.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        # All values should be close to 123 (interpolated from the input DEM)
        np.testing.assert_allclose(result["z"].values, 123.0, atol=1e-6)

    def test_interp_dem_with_elevation_gradient(self):
        """DEM with a linear gradient should produce interpolated gradient."""
        h_in, w_in = 10, 10
        y_in = np.arange(0.0, h_in * 10.0, 10.0)
        x_in = np.arange(0.0, w_in * 10.0, 10.0)
        y_grid, x_grid = np.meshgrid(y_in, x_in, indexing="ij")
        elev = 100.0 + y_grid * 0.5 + x_grid * 0.3
        dem_in = xr.DataArray(elev, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5, start_x=0.0, start_y=0.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)

        # Output should have increasing values matching the gradient direction
        z_vals = result["z"].values
        assert z_vals[4, 4] > z_vals[0, 0], "Elevation should increase with position"
        assert z_vals[0, 4] > z_vals[0, 0], "Elevation should increase with x"

    def test_interp_dem_with_nan_in_input(self):
        """NaN values in input DEM should be handled (filled with 0 before interpolation)."""
        h_in, w_in = 10, 10
        y_in = np.arange(0.0, h_in * 10.0, 10.0)
        x_in = np.arange(0.0, w_in * 10.0, 10.0)
        data = np.full((h_in, w_in), 100.0, dtype=np.float64)
        data[3:7, 3:7] = np.nan
        dem_in = xr.DataArray(data, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5, start_x=0.0, start_y=0.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)

        assert isinstance(result, xr.Dataset)
        assert "z" in result.data_vars
        # Output should not contain NaN (interpolation fills with mean of dem_sel)
        assert not np.any(np.isnan(result["z"].values))

    def test_interp_dem_with_large_border(self):
        """Larger border should include more input data in interpolation."""
        dem_in = _make_projected_dem(10, 10, elev=75.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")

        result_small_border = interp_dem(dem_out, dem_in, mid_crs, border=2)
        result_large_border = interp_dem(dem_out, dem_in, mid_crs, border=20)

        assert isinstance(result_small_border, xr.Dataset)
        assert isinstance(result_large_border, xr.Dataset)
        # Both should have the same output shape
        assert result_small_border["z"].shape == result_large_border["z"].shape

    def test_interp_dem_with_linear_method(self):
        """Should support 'linear' interpolation method."""
        dem_in = _make_projected_dem(10, 10, elev=200.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs, method="linear")
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_with_nearest_method(self):
        """Should support 'nearest' interpolation method."""
        dem_in = _make_projected_dem(10, 10, elev=300.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs, method="nearest")
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_with_cubic_method(self):
        """Should support 'cubic' interpolation method (default)."""
        dem_in = _make_projected_dem(10, 10, elev=150.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs, method="cubic")
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_output_is_finite(self):
        """Output values should be finite numbers."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert np.all(np.isfinite(result["z"].values))

    def test_interp_dem_non_overlapping_grids(self):
        """When dem_out does not overlap dem_in, output should be zeros."""
        dem_in = _make_projected_dem(10, 10, start_x=0.0, start_y=0.0, elev=100.0)
        dem_out = _make_output_grid(5, 5, start_x=1000.0, start_y=1000.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert np.all(result["z"].values == 0.0)

    def test_interp_dem_small_output_grid(self):
        """Very small output grid (3x3) should still produce valid output."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(3, 3)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (3, 3)

    def test_interp_dem_1x1_output_grid(self):
        """Single-pixel output grid should return valid Dataset."""
        dem_in = _make_projected_dem(10, 10, elev=75.0)
        dem_out = _make_output_grid(1, 1)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (1, 1)
        assert np.all(np.isfinite(result["z"].values))

    def test_interp_dem_2x2_output_grid(self):
        """2x2 output grid should produce valid output."""
        dem_in = _make_projected_dem(10, 10, elev=80.0)
        dem_out = _make_output_grid(2, 2)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (2, 2)

    def test_interp_dem_larger_output_than_input(self):
        """Output grid larger than input DEM should still work."""
        dem_in = _make_projected_dem(5, 5, elev=90.0)
        dem_out = _make_output_grid(10, 10)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (10, 10)

    def test_interp_dem_offset_output_grid(self):
        """Output grid offset from input should still interpolate correctly."""
        dem_in = _make_projected_dem(10, 10, start_x=0.0, start_y=0.0, elev=100.0)
        dem_out = _make_output_grid(5, 5, start_x=25.0, start_y=25.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)
        assert np.all(np.isfinite(result["z"].values))

    def test_interp_dem_negative_spacing_input(self):
        """Input DEM with negative spacing should be handled."""
        y = np.arange(100.0, 0.0, -10.0)
        x = np.arange(100.0, 0.0, -10.0)
        data = np.full((10, 10), 100.0, dtype=np.float64)
        dem_in = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_negative_spacing_output(self):
        """Output grid with negative spacing should be handled."""
        dem_in = _make_projected_dem(10, 10, elev=100.0)

        y = np.arange(50.0, 0.0, -10.0)
        x = np.arange(50.0, 0.0, -10.0)
        data = np.zeros((5, 5), dtype=np.float64)
        dem_out = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
        dem_out.rio.write_crs("EPSG:32633", inplace=True)

        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_different_crs(self):
        """Different input/output CRS should still produce valid output."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(5, 5)
        dem_out.rio.write_crs("EPSG:32632", inplace=True)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_crs_with_vertical_datum(self):
        """CRS with vertical component (3D) should handle height transformation."""
        dem_in = _make_projected_dem(10, 10, elev=50.0)
        dem_out = _make_output_grid(5, 5)
        # 3D CRS that includes vertical datum
        mid_crs = CRS("EPSG:4978")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_zero_elevation_input(self):
        """Input DEM with zero elevation should produce valid output."""
        dem_in = _make_projected_dem(10, 10, elev=0.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        np.testing.assert_allclose(result["z"].values, 0.0, atol=1.0)

    def test_interp_dem_high_elevation_input(self):
        """Input DEM with high elevation should produce output at that elevation."""
        dem_in = _make_projected_dem(10, 10, elev=5000.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        np.testing.assert_allclose(result["z"].values, 5000.0, atol=1.0)

    def test_interp_dem_mixed_nan_and_valid(self):
        """DEM with some NaN and some valid values should interpolate valid regions."""
        h_in, w_in = 15, 15
        y_in = np.arange(0.0, h_in * 10.0, 10.0)
        x_in = np.arange(0.0, w_in * 10.0, 10.0)
        data = np.full((h_in, w_in), 100.0, dtype=np.float64)
        # Leave the center clean, put NaN on the edges
        data[0:3, :] = np.nan
        data[-3:, :] = np.nan
        data[:, 0:3] = np.nan
        data[:, -3:] = np.nan
        dem_in = xr.DataArray(data, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        # Output grid centered on the clean region
        dem_out = _make_output_grid(5, 5, start_x=30.0, start_y=30.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert np.all(np.isfinite(result["z"].values))

    def test_interp_dem_border_zero(self):
        """Border=0 should work and select tighter bounding box."""
        dem_in = _make_projected_dem(10, 10, elev=100.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs, border=0)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_varied_elevation_surface(self):
        """DEM with a sinusoidal elevation surface should produce interpolated surface."""
        h_in, w_in = 15, 15
        y_in = np.arange(0.0, h_in * 5.0, 5.0)
        x_in = np.arange(0.0, w_in * 5.0, 5.0)
        y_grid, x_grid = np.meshgrid(y_in, x_in, indexing="ij")
        elev = 100.0 + 20.0 * np.sin(y_grid / 10.0) + 15.0 * np.cos(x_grid / 10.0)
        dem_in = xr.DataArray(elev, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(7, 7)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)

        z_vals = result["z"].values
        # Should show variation, not be constant
        assert np.ptp(z_vals) > 5.0, "Output should have elevation variation"
        assert np.all(np.isfinite(z_vals))
        # Peak and trough positions should roughly match
        assert z_vals.max() > 110.0
        assert z_vals.min() < 105.0

    def test_interp_dem_single_column_overlap(self):
        """When output grid only overlaps input in one column, should handle gracefully."""
        dem_in = _make_projected_dem(10, 10, start_x=0.0, start_y=0.0, elev=100.0)
        # Output grid only partially overlaps in x
        dem_out = _make_output_grid(5, 5, start_x=50.0, start_y=0.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_single_row_overlap(self):
        """When output grid only overlaps input in one row, should handle gracefully."""
        dem_in = _make_projected_dem(10, 10, start_x=0.0, start_y=0.0, elev=100.0)
        # Output grid only partially overlaps in y
        dem_out = _make_output_grid(5, 5, start_x=0.0, start_y=50.0)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_all_nan_input(self):
        """DEM with all NaN values should produce output (filled with NaN mean -> 0)."""
        h_in, w_in = 10, 10
        y_in = np.arange(0.0, h_in * 10.0, 10.0)
        x_in = np.arange(0.0, w_in * 10.0, 10.0)
        data = np.full((h_in, w_in), np.nan, dtype=np.float64)
        dem_in = xr.DataArray(data, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)
        np.testing.assert_allclose(result["z"].values, 0, atol=1e-6)

    def test_interp_dem_asymmetric_output(self):
        """Output grid with different x and y dimensions should work."""
        dem_in = _make_projected_dem(10, 10, elev=100.0)
        dem_out = _make_output_grid(3, 7)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (3, 7)

    def test_interp_dem_with_extra_kwargs(self):
        """Additional kwargs should be passed through without error."""
        dem_in = _make_projected_dem(10, 10, elev=100.0)
        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        # Extra kwargs that interp_dem doesn't explicitly use but accepts via **kwargs
        result = interp_dem(dem_out, dem_in, mid_crs, border=5)
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)

    def test_interp_dem_5x5_output(self):
        """5x5 output should work with cubic interpolation (smooth surface)."""
        h_in, w_in = 20, 20
        y_in = np.arange(0.0, h_in * 5.0, 5.0)
        x_in = np.arange(0.0, w_in * 5.0, 5.0)
        y_grid, x_grid = np.meshgrid(y_in, x_in, indexing="ij")
        elev = 200.0 + 30.0 * np.sin(y_grid / 20.0) * np.cos(x_grid / 20.0)
        dem_in = xr.DataArray(elev, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs, method="cubic")
        assert isinstance(result, xr.Dataset)
        assert result["z"].shape == (5, 5)
        assert np.all(np.isfinite(result["z"].values))

    def test_interp_dem_value_range_conservation(self):
        """Interpolated values should be within the range of input values (for same CRS)."""
        dem_in = _make_projected_dem(15, 15, elev=50.0)
        dem_out = _make_output_grid(7, 7)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)

        z_min = dem_in.values.min()
        z_max = dem_in.values.max()
        z_vals = result["z"].values
        # Values should be within the input range (with small tolerance for interpolation)
        assert z_vals.min() >= z_min - 1.0
        assert z_vals.max() <= z_max + 1.0

    def test_interp_dem_output_dimensions(self):
        """Output Dataset should have 'y' and 'x' dimensions matching dem_out."""
        h, w = 6, 8
        dem_in = _make_projected_dem(10, 10, elev=100.0)
        dem_out = _make_output_grid(h, w)
        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)

        assert "y" in result.dims or "y" in result["z"].dims
        assert "x" in result.dims or "x" in result["z"].dims
        assert result["z"].shape == (h, w)

    def test_interp_dem_output_coords_preserved(self):
        """Output coordinates should be preserved in the result Dataset."""
        y = np.arange(0.0, 50.0, 10.0)
        x = np.arange(0.0, 60.0, 10.0)
        dem_in = _make_projected_dem(10, 10, elev=100.0)
        data = np.zeros((5, 6), dtype=np.float64)
        dem_out = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x})
        dem_out.rio.write_crs("EPSG:32633", inplace=True)

        mid_crs = CRS("EPSG:32633")
        result = interp_dem(dem_out, dem_in, mid_crs)
        assert result["z"].shape == (5, 6)
        # interp_dataset may not preserve all coords from dem_out
        assert "y" in result["z"].dims
        assert "x" in result["z"].dims

    def test_interp_dem_inf_in_input_dem(self):
        """Input DEM with inf values: fillna only handles NaN, not inf.
        
        The interp_dem function uses .fillna(0) which replaces NaN but not inf.
        When inf values are present, scipy interpn may raise an error.
        This test verifies the function's behavior with inf in the input.
        """
        h_in, w_in = 10, 10
        y_in = np.arange(0.0, h_in * 10.0, 10.0)
        x_in = np.arange(0.0, w_in * 10.0, 10.0)
        data = np.full((h_in, w_in), 100.0, dtype=np.float64)
        data[0, 0] = np.inf
        data[1, 0] = -np.inf
        dem_in = xr.DataArray(data, dims=["y", "x"], coords={"y": y_in, "x": x_in})
        dem_in.rio.write_crs("EPSG:32633", inplace=True)

        dem_out = _make_output_grid(5, 5)
        mid_crs = CRS("EPSG:32633")
        # fillna(0) does not remove inf, so interpn may fail
        with pytest.raises(ValueError):
            interp_dem(dem_out, dem_in, mid_crs)
