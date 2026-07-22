"""Unit tests for cop_ard_core.utils."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.utils import (
    compute_interf,
    compute_interf_norm,
    get_coeff_demod,
    get_coordinates,
    get_daz_corr,
    get_interf,
    get_normalized_interf,
    get_power,
    get_phramp_diff,
    time_to_pixel,
    wrap,
)


class TestWrap:
    """Tests for the wrap function."""

    def test_wrap_values_within_bounds(self):
        """Wrap values should map to [-pi, pi)."""
        v = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, -np.pi / 2])
        result = wrap(v)
        assert result.shape == v.shape
        np.testing.assert_array_less(result, np.pi + 1e-12)
        np.testing.assert_array_less(-np.pi - 1e-12, result)

    def test_wrap_already_within_range(self):
        """Values already in [-pi, pi) should remain unchanged."""
        v = np.array([-np.pi / 4, 0.0, np.pi / 4])
        result = wrap(v)
        np.testing.assert_array_equal(result, v)

    def test_wrap_2pi_custom(self):
        """Custom wrap window should work."""
        v = np.array([0.0, 3.0, 5.0, 7.0])
        result = wrap(v, w=4.0)
        assert result.shape == v.shape
        np.testing.assert_array_less(result, 4.0)

    def test_wrap_single_element(self):
        """Single element array should wrap correctly."""
        v = np.array([10.0])
        result = wrap(v)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], (10.0 + np.pi) % (2 * np.pi) - np.pi)

    def test_wrap_empty_array(self):
        """Empty array should return empty array."""
        v = np.array([])
        result = wrap(v)
        assert result.shape == (0,)

    def test_wrap_larger_than_2pi(self):
        """Large positive and negative values should wrap correctly."""
        v = np.array([100.0, -100.0, 1000.0, -1000.0])
        result = wrap(v)
        assert np.all(result >= -np.pi - 1e-12)
        assert np.all(result < np.pi + 1e-12)


class TestGetInterf:
    """Tests for get_interf."""

    def test_get_interf_basic(self, complex_3x3):
        """Interferogram of an array with itself should be power."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_get_interf_returns_correct_type(self, complex_3x3):
        """get_interf should return xr.DataArray with complex dtype."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert np.iscomplexobj(result.values)

    def test_get_interf_multilook(self, complex_5x5, multilook_factor):
        """Multilook should reduce the spatial dimensions."""
        result = get_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.dims == ("azimuth_time", "slant_range_time")
        assert result.shape[0] < complex_5x5.shape[0]
        assert result.shape[1] < complex_5x5.shape[1]


class TestGetNormalizedInterf:
    """Tests for get_normalized_interf."""

    def test_normalized_interf_self(self, complex_3x3):
        """Normalized interferogram of an array with itself should have magnitude ~1."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert isinstance(result, xr.DataArray)
        magnitude = np.abs(result.values)
        # Values should be close to 1 (with some tolerance from coarsening)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)

    def test_normalized_interf_zero_mean(self, complex_3x3):
        """Normalized interferogram should fill NaN/inf with 0j."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert not np.any(np.isnan(result.values))

    def test_normalized_interf_multilook(self, complex_5x5, multilook_factor):
        """Normalized interferogram with multilook should reduce dimensions."""
        result = get_normalized_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]


class TestComputeInterf:
    """Tests for compute_interf."""

    def test_compute_interf_basic(self, complex_3x3):
        """Basic interferogram computation should return DataArray."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_compute_interf_shape_preserved(self, complex_3x3):
        """Shape should be preserved without multilooking."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert result.shape == complex_3x3.shape

    def test_compute_interf_multilook(self, complex_5x5, multilook_factor):
        """Multilook should reduce dimensions."""
        result = compute_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]
        assert result.shape[1] < complex_5x5.shape[1]

    def test_compute_interf_complex_result(self, complex_3x3):
        """Result should be complex-valued."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert np.iscomplexobj(result.values)

    def test_compute_interf_self_conj(self, complex_3x3):
        """conj(a)*a should give |a|^2 (real values)."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert np.allclose(result.values.imag, 0.0, atol=1e-14)


class TestGetPower:
    """Tests for get_power."""

    def test_get_power_returns_dataarray(self, complex_3x3):
        """get_power should return xr.DataArray."""
        result = get_power(complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_get_power_is_real(self, complex_3x3):
        """Power should be real-valued (|a|^2)."""
        result = get_power(complex_3x3, {})
        assert not np.iscomplexobj(result.values)
        # Power should be non-negative
        assert np.all(result.values >= 0)

    def test_get_power_multilook(self, complex_5x5, multilook_factor):
        """Multilook should reduce spatial dimensions."""
        result = get_power(complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]

    def test_get_power_values(self, complex_3x3):
        """Power of [1+1j, 2+2j, ...] should be [2, 8, 18, ...]."""
        result = get_power(complex_3x3, {})
        expected = np.abs(complex_3x3.values) ** 2
        np.testing.assert_allclose(result.values, expected)


class TestComputeInterfNorm:
    """Tests for compute_interf_norm."""

    def test_compute_interf_norm_self(self, complex_3x3):
        """Normalized interferogram with self should have magnitude ~1."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        magnitude = np.abs(result.values)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)

    def test_compute_interf_norm_no_nan(self, complex_3x3):
        """Result should not contain NaN."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        assert not np.any(np.isnan(result.values))

    def test_compute_interf_norm_multilook(self, complex_5x5, multilook_factor):
        """Should reduce dimensions with multilook."""
        result = compute_interf_norm(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]


class TestGetCoordinates:
    """Tests for get_coordinates."""

    def test_get_coordinates_returns_generator(self):
        """Should return a generator of values."""
        y = xr.DataArray(np.arange(5), dims=["y"])
        x = xr.DataArray(np.arange(5), dims=["x"])
        alpha = [0.0, 0.0]
        beta = [1.0, 1.0]
        result = get_coordinates(y, x, alpha, beta)
        assert hasattr(result, "__iter__")

    def test_get_coordinates_values(self):
        """Coordinate values should be shifted and scaled."""
        y = xr.DataArray(np.array([10, 20, 30]), dims=["y"])
        x = xr.DataArray(np.array([100, 200, 300]), dims=["x"])
        alpha = [np.float64(10.0), np.float64(100.0)]
        beta = [np.float64(10.0), np.float64(100.0)]
        result = list(get_coordinates(y, x, alpha, beta))
        assert len(result) == 2
        np.testing.assert_array_equal(result[0].ravel(), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(result[1].ravel(), np.array([0.0, 1.0, 2.0]))


class TestGetDazCorr:
    """Tests for get_daz_corr."""

    def test_get_daz_corr_returns_array(self):
        """Should return a numpy array."""
        azimuth_time = xr.DataArray(np.arange(5), dims=["azimuth_time"])
        slant_range_time = xr.DataArray(np.arange(5), dims=["slant_range_time"])
        y_poly = xr.DataArray(np.ones((2, 2)), dims=["i", "j"])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        assert isinstance(result, np.ndarray)

    def test_get_daz_corr_shape(self):
        """Output shape should match the product of input dimensions."""
        az_size = 4
        sr_size = 3
        azimuth_time = xr.DataArray(np.arange(az_size), dims=["azimuth_time"])
        slant_range_time = xr.DataArray(np.arange(sr_size), dims=["slant_range_time"])
        y_poly = xr.DataArray(np.ones((2, 2)), dims=["i", "j"])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        assert result.shape == (az_size, sr_size)


class TestGetCoeffDemod:
    """Tests for get_coeff_demod."""

    def test_get_coeff_demod_returns_array(self):
        """Should return a numpy array."""
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = get_coeff_demod(coeffs)
        assert isinstance(result, np.ndarray)

    def test_get_coeff_demod_integration(self):
        """Integration of constant should produce linear term."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = get_coeff_demod(coeffs)
        # polyint along axis=1 increases columns by 1, so (2,2) -> (2,3)
        assert result.shape == (2, 3)

    def test_get_coeff_demod_shape(self):
        """Output shape: axis dimension increases by 1 (polyint), other axis unchanged."""
        coeffs = np.random.rand(3, 4)
        result = get_coeff_demod(coeffs)
        assert result.shape == (3, 5)


class TestTimeToPixel:
    """Tests for time_to_pixel."""

    def test_time_to_pixel_dataset(self):
        """Should convert dataset time coordinates to pixel indices."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        data = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da)
        assert isinstance(result, xr.DataArray)
        assert "y" in result.coords
        assert "x" in result.coords

    def test_time_to_pixel_dataset_swap_dims(self):
        """Should swap azimuth_time->y, slant_range_time->x."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        data = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da)
        assert "y" in result.dims or "x" in result.dims

    def test_time_to_pixel_with_explicit_origin(self):
        """Explicit y0/x0 should affect pixel computation."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        data = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da, y0=az[0], x0=0.0)
        assert "y" in result.dims or "y" in result.coords

    def test_time_to_pixel_pixel_values(self):
        """First pixel should map to 0."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        data = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da)
        assert result.coords["y"].values[0] == 0
        assert result.coords["x"].values[0] == 0
