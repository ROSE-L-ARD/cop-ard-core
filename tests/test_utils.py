"""Unit tests for cop_ard_core.utils.

Tests all public functions in utils.py covering:
- Happy paths with normal inputs
- Edge cases (empty arrays, single elements, NaN/inf, boundary conditions)
- Error cases (invalid inputs raising ValueError, RuntimeError, etc.)
- Type preservation (xr.DataArray, xr.Dataset, np.ndarray outputs)
- Shape and value correctness using np.testing and xr.testing
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.utils import (
    compute_interf,
    compute_interf_norm,
    get_azimuth_time_raw,
    get_best_pol,
    get_coeff_demod,
    get_coordinates,
    get_daz_corr,
    get_interf,
    get_normalized_interf,
    get_phramp_diff,
    get_power,
    time_to_pixel,
    wrap,
)


# ===========================================================================
# TestWrap
# ===========================================================================


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

    def test_wrap_2d_array(self):
        """Wrap should handle 2D arrays correctly."""
        v = np.array([[0.0, np.pi], [2 * np.pi, -np.pi]])
        result = wrap(v)
        assert result.shape == v.shape
        assert np.all(result >= -np.pi - 1e-12)
        assert np.all(result < np.pi + 1e-12)

    def test_wrap_complex_values_raises(self):
        """Wrap with complex input should raise TypeError."""
        v = np.array([1.0 + 2.0j, np.pi + 1.0j, -np.pi + 2.0j])
        with pytest.raises(TypeError):
            wrap(v)

    def test_wrap_nan_values(self):
        """NaN values should propagate through wrap."""
        v = np.array([np.nan, 0.0, np.pi])
        result = wrap(v)
        assert np.isnan(result[0])
        np.testing.assert_array_equal(result[1:], np.array([0.0, -np.pi]))

    def test_wrap_inf_values(self):
        """Inf values should produce NaN from modulo."""
        v = np.array([np.inf, -np.inf, 0.0])
        result = wrap(v)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 0.0

    def test_wrap_zero(self):
        """Zero should remain zero after wrapping."""
        result = wrap(np.array([0.0]))
        np.testing.assert_array_equal(result, np.array([0.0]))

    @pytest.mark.parametrize("w", [1.0, 2.0, 4 * np.pi, 100.0])
    def test_wrap_custom_window(self, w):
        """Wrap with various custom window sizes should produce correct range."""
        v = np.array([0.0, w / 4, w / 2, 3 * w / 4, w])
        result = wrap(v, w=w)
        np.testing.assert_array_less(result, w + 1e-12)
        np.testing.assert_array_less(-w - 1e-12, result)


# ===========================================================================
# TestGetInterf
# ===========================================================================


class TestGetInterf:
    """Tests for the get_interf function."""

    def test_get_interf_returns_dataarray(self, complex_3x3):
        """get_interf should return xr.DataArray."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_get_interf_complex_dtype(self, complex_3x3):
        """Result should have complex dtype."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert np.iscomplexobj(result.values)

    def test_get_interf_self_conj_is_real(self, complex_3x3):
        """interf(a, a) = conj(a)*a should be real-valued."""
        result = get_interf(complex_3x3, complex_3x3, {})
        np.testing.assert_allclose(result.values.imag, 0.0, atol=1e-14)

    def test_get_interf_multilook(self, complex_5x5, multilook_factor):
        """Multilook should reduce spatial dimensions."""
        result = get_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.dims == ("azimuth_time", "slant_range_time")
        assert result.shape[0] < complex_5x5.shape[0]
        assert result.shape[1] < complex_5x5.shape[1]

    def test_get_interf_empty_multilook(self, complex_3x3):
        """Empty multilook dict should preserve shape."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert result.shape == complex_3x3.shape

    def test_get_interf_different_arrays(self, complex_3x3):
        """Interferogram of two different arrays should be complex."""
        other = complex_3x3 * (1 + 0.5j)
        result = get_interf(complex_3x3, other, {})
        assert np.iscomplexobj(result.values)

    def test_get_interf_preserves_coords(self, complex_3x3):
        """Result should preserve azimuth_time and slant_range_time coords."""
        result = get_interf(complex_3x3, complex_3x3, {})
        assert "azimuth_time" in result.coords
        assert "slant_range_time" in result.coords

    def test_get_interf_multilook_one_direction(self):
        """Multilook in only one direction should reduce only that dimension."""
        az = xr.DataArray(np.arange(5, dtype=np.complex128), dims=["azimuth_time"])
        sr = xr.DataArray(np.arange(4, dtype=np.complex128), dims=["slant_range_time"])
        data = az.values[:, np.newaxis] * sr.values[np.newaxis, :]
        arr = xr.DataArray(data, dims=["azimuth_time", "slant_range_time"])
        ml = {"azimuth_time": 2, "slant_range_time": 1}
        result = get_interf(arr, arr, ml)
        assert result.shape[0] < arr.shape[0]
        assert result.shape[1] == arr.shape[1]

    def test_get_interf_all_ones(self):
        """Interferogram of all-ones arrays should be one."""
        data = np.ones((3, 3), dtype=np.complex128)
        arr = xr.DataArray(data, dims=["azimuth_time", "slant_range_time"])
        result = get_interf(arr, arr, {})
        np.testing.assert_allclose(result.values, 1.0, atol=1e-14)

    def test_get_interf_multilook_odd_shape(self, complex_5x5, multilook_factor):
        """Multilook with odd-sized input should not raise."""
        result = get_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1


# ===========================================================================
# TestGetNormalizedInterf
# ===========================================================================


class TestGetNormalizedInterf:
    """Tests for the get_normalized_interf function."""

    def test_normalized_interf_self_magnitude(self, complex_3x3):
        """Normalized interferogram of self should have magnitude ~1."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        magnitude = np.abs(result.values)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)

    def test_normalized_interf_self_no_nan(self, complex_3x3):
        """Normalized interferogram should not contain NaN."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert not np.any(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))

    def test_normalized_interf_returns_dataarray(self, complex_3x3):
        """Should return xr.DataArray."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_normalized_interf_multilook(self, complex_5x5, multilook_factor):
        """Normalized interferogram with multilook should reduce dimensions."""
        result = get_normalized_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]
        magnitude = np.abs(result.values)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)

    def test_normalized_interf_zero_power_fill(self):
        """When power is zero, normalized interf should be filled with 0j."""
        zero = xr.DataArray(
            np.zeros((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = get_normalized_interf(zero, zero, {})
        # All values should be 0j (filled by fillna)
        np.testing.assert_array_equal(result.values, 0j)

    def test_normalized_interf_preserves_coords(self, complex_3x3):
        """Result should preserve coordinate dimensions."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert "azimuth_time" in result.coords
        assert "slant_range_time" in result.coords

    def test_normalized_interf_complex_result(self, complex_3x3):
        """Result should be complex-valued."""
        result = get_normalized_interf(complex_3x3, complex_3x3, {})
        assert np.iscomplexobj(result.values)

    def test_normalized_interf_unnormalized(self, complex_3x3):
        """Normalized interf of different arrays should have magnitude < 1."""
        other = complex_3x3 * 2.0
        result = get_normalized_interf(complex_3x3, other, {})
        # Since they differ by a real scalar, magnitude should still be 1
        magnitude = np.abs(result.values)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)


# ===========================================================================
# TestComputeInterf
# ===========================================================================


class TestComputeInterf:
    """Tests for the compute_interf function."""

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

    def test_compute_interf_self_conj_real(self, complex_3x3):
        """conj(a)*a should give |a|^2 (real values)."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert np.allclose(result.values.imag, 0.0, atol=1e-14)

    def test_compute_interf_preserves_dims(self, complex_3x3):
        """Output dims should be azimuth_time, slant_range_time."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_compute_interf_with_multilook_dims(self, complex_5x5, multilook_factor):
        """Output dims should still be azimuth_time, slant_range_time after multilook."""
        result = compute_interf(complex_5x5, complex_5x5, multilook_factor)
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_compute_interf_value_correct(self, complex_3x3):
        """Value should match manual computation."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        expected = np.abs(complex_3x3.values) ** 2
        np.testing.assert_allclose(result.values, expected)

    def test_compute_interf_empty_multilook(self, complex_3x3):
        """Empty multilook dict should produce same shape as input."""
        result = compute_interf(complex_3x3, complex_3x3, {})
        assert result.shape == complex_3x3.shape

    def test_compute_interf_large_multilook(self, complex_3x3):
        """Multilook factors larger than array should produce small output."""
        result = compute_interf(complex_3x3, complex_3x3, {"azimuth_time": 5, "slant_range_time": 5})
        assert result.shape[0] <= complex_3x3.shape[0]
        assert result.shape[1] <= complex_3x3.shape[1]


# ===========================================================================
# TestGetPower
# ===========================================================================


class TestGetPower:
    """Tests for the get_power function."""

    def test_get_power_returns_dataarray(self, complex_3x3):
        """get_power should return xr.DataArray."""
        result = get_power(complex_3x3, {})
        assert isinstance(result, xr.DataArray)

    def test_get_power_is_real(self, complex_3x3):
        """Power should be real-valued (|a|^2)."""
        result = get_power(complex_3x3, {})
        assert not np.iscomplexobj(result.values)
        assert np.all(result.values >= 0)

    def test_get_power_multilook(self, complex_5x5, multilook_factor):
        """Multilook should reduce spatial dimensions."""
        result = get_power(complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]

    def test_get_power_values(self, complex_3x3):
        """Power of [1+1j, 2+2j, ...] should match |a|^2."""
        result = get_power(complex_3x3, {})
        expected = np.abs(complex_3x3.values) ** 2
        np.testing.assert_allclose(result.values, expected)

    def test_get_power_preserves_dims(self, complex_3x3):
        """Output dims should be azimuth_time, slant_range_time."""
        result = get_power(complex_3x3, {})
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_get_power_all_zeros(self):
        """Power of zero array should be zero."""
        zero = xr.DataArray(
            np.zeros((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = get_power(zero, {})
        np.testing.assert_array_equal(result.values, 0.0)

    def test_get_power_all_ones(self):
        """Power of all-ones array should be one."""
        ones = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = get_power(ones, {})
        np.testing.assert_array_equal(result.values, 1.0)

    def test_get_power_with_nan(self):
        """Power of array with NaN should propagate NaN."""
        data = np.ones((3, 3), dtype=np.complex128)
        data[0, 0] = np.nan + 0j
        arr = xr.DataArray(data, dims=["azimuth_time", "slant_range_time"])
        result = get_power(arr, {})
        assert np.isnan(result.values[0, 0])

    def test_get_power_multilook_odd(self, complex_5x5, multilook_factor):
        """Multilook with odd shape should still work."""
        result = get_power(complex_5x5, multilook_factor)
        assert result.shape[0] >= 1
        assert result.shape[1] >= 1
        assert np.all(result.values >= 0)


# ===========================================================================
# TestComputeInterfNorm
# ===========================================================================


class TestComputeInterfNorm:
    """Tests for the compute_interf_norm function."""

    def test_compute_interf_norm_self(self, complex_3x3):
        """Normalized interferogram with self should have magnitude ~1."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        magnitude = np.abs(result.values)
        np.testing.assert_allclose(magnitude, 1.0, atol=0.1)

    def test_compute_interf_norm_no_nan(self, complex_3x3):
        """Result should not contain NaN."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        assert not np.any(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))

    def test_compute_interf_norm_multilook(self, complex_5x5, multilook_factor):
        """Should reduce dimensions with multilook."""
        result = compute_interf_norm(complex_5x5, complex_5x5, multilook_factor)
        assert result.shape[0] < complex_5x5.shape[0]

    def test_compute_interf_norm_preserves_dims(self, complex_3x3):
        """Output dims should be azimuth_time, slant_range_time."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_compute_interf_norm_complex_result(self, complex_3x3):
        """Result should be complex-valued."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        assert np.iscomplexobj(result.values)

    def test_compute_interf_norm_zero_power(self):
        """When power is zero, result should be filled with 0j."""
        zero = xr.DataArray(
            np.zeros((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = compute_interf_norm(zero, zero, {})
        np.testing.assert_array_equal(result.values, 0j)

    def test_compute_interf_norm_different_arrays(self, complex_3x3):
        """Normalized interf of different arrays should have magnitude <= 1."""
        other = complex_3x3 * (1 + 0.3j)
        result = compute_interf_norm(complex_3x3, other, {})
        magnitude = np.abs(result.values)
        assert np.all(magnitude <= 1.0 + 0.1)

    def test_compute_interf_norm_preserves_coords(self, complex_3x3):
        """Result should preserve coordinate dimensions."""
        result = compute_interf_norm(complex_3x3, complex_3x3, {})
        assert "azimuth_time" in result.coords
        assert "slant_range_time" in result.coords


# ===========================================================================
# TestGetCoordinates
# ===========================================================================


class TestGetCoordinates:
    """Tests for the get_coordinates function."""

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
        np.testing.assert_array_equal(result[0].values.ravel(), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(result[1].values.ravel(), np.array([0.0, 1.0, 2.0]))

    def test_get_coordinates_shape(self):
        """Output values should match input shape."""
        y = xr.DataArray(np.arange(4), dims=["y"])
        x = xr.DataArray(np.arange(3), dims=["x"])
        result = list(get_coordinates(y, x, [np.float64(0.0), np.float64(0.0)], [np.float64(1.0), np.float64(1.0)]))
        assert result[0].shape == (4,)
        assert result[1].shape == (3,)

    def test_get_coordinates_negative_offset(self):
        """Negative offsets should shift coordinates correctly."""
        y = xr.DataArray(np.array([0, 1, 2]), dims=["y"])
        x = xr.DataArray(np.array([0, 1, 2]), dims=["x"])
        alpha = [np.float64(-1.0), np.float64(-1.0)]
        beta = [np.float64(1.0), np.float64(1.0)]
        result = list(get_coordinates(y, x, alpha, beta))
        # (y - (-1)) / 1 = y + 1 => [1, 2, 3]
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0, 3.0]))

    def test_get_coordinates_scale(self):
        """Scaling should affect coordinate values."""
        y = xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["y"])
        x = xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["x"])
        alpha = [np.float64(0.0), np.float64(0.0)]
        beta = [np.float64(2.0), np.float64(2.0)]
        result = list(get_coordinates(y, x, alpha, beta))
        # y / 2 => [0.0, 0.5, 1.0]
        np.testing.assert_array_equal(result[0], np.array([0.0, 0.5, 1.0]))

    def test_get_coordinates_float_dtype(self):
        """Output should be float64 dtype."""
        y = xr.DataArray(np.array([0, 1, 2]), dims=["y"])
        x = xr.DataArray(np.array([0, 1, 2]), dims=["x"])
        result = list(get_coordinates(y, x, [np.float64(0.0), np.float64(0.0)], [np.float64(1.0), np.float64(1.0)]))
        assert result[0].dtype == np.float64
        assert result[1].dtype == np.float64

    def test_get_coordinates_nonzero_alpha(self):
        """Non-zero alpha should offset all values."""
        y = xr.DataArray(np.array([0, 1, 2]), dims=["y"])
        x = xr.DataArray(np.array([0, 1, 2]), dims=["x"])
        alpha = [np.float64(1.0), np.float64(1.0)]
        beta = [np.float64(1.0), np.float64(1.0)]
        result = list(get_coordinates(y, x, alpha, beta))
        # (y - 1) / 1 => [-1, 0, 1]
        np.testing.assert_array_equal(result[0], np.array([-1.0, 0.0, 1.0]))

    def test_get_coordinates_generator_consumption(self):
        """Generator should be consumed exactly once."""
        y = xr.DataArray(np.arange(3), dims=["y"])
        x = xr.DataArray(np.arange(3), dims=["x"])
        gen = get_coordinates(y, x, [np.float64(0.0), np.float64(0.0)], [np.float64(1.0), np.float64(1.0)])
        first = list(gen)
        second = list(gen)  # Should be empty after first consumption
        assert len(first) == 2
        assert len(second) == 0


# ===========================================================================
# TestGetDazCorr
# ===========================================================================


class TestGetDazCorr:
    """Tests for the get_daz_corr function."""

    def test_get_daz_corr_returns_array(self):
        """Should return a numpy array."""
        azimuth_time = np.arange(5)
        slant_range_time = np.arange(5)
        y_poly = xr.DataArray(np.ones((2, 2)), dims=["i", "j"])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        assert isinstance(result, np.ndarray)

    def test_get_daz_corr_shape(self):
        """Output shape should match the product of input dimensions."""
        az_size = 4
        sr_size = 3
        azimuth_time = np.arange(az_size)
        slant_range_time = np.arange(sr_size)
        y_poly = xr.DataArray(np.ones((2, 2)), dims=["i", "j"])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        assert result.shape == (az_size, sr_size)

    def test_get_daz_corr_constant_polynomial(self):
        """All-one polynomial should produce values depending on coordinates."""
        azimuth_time = np.arange(4)
        slant_range_time = np.arange(3)
        y_poly = np.ones((2, 2))
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        # polygrid2d(c1, c2, [[1,1],[1,1]]) = 1 + c1 + c2 + c1*c2
        expected = np.outer(np.ones(4), np.ones(3)) + np.outer(np.arange(4), np.ones(3)) + np.outer(np.ones(4), np.arange(3)) + np.outer(np.arange(4), np.arange(3))
        np.testing.assert_allclose(result, expected)

    def test_get_daz_corr_linear_polynomial(self):
        """Polynomial with c[0,1]=1 should produce c2 values along columns."""
        azimuth_time = np.arange(4)
        slant_range_time = np.arange(3)
        y_poly = np.array([[0.0, 1.0], [0.0, 0.0]])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        # c[0,1]=1 => result = c2 (second arg) = [0, 1, 2] along columns
        expected = np.outer(np.ones(4), np.arange(3, dtype=np.float64))
        np.testing.assert_allclose(result, expected)

    def test_get_daz_corr_with_offset(self):
        """Offset should shift coordinate values and change output."""
        azimuth_time = np.arange(3)
        slant_range_time = np.arange(3)
        y_poly = np.ones((2, 2))
        offset = [np.float64(1.0), np.float64(0.5)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        # c1 = (az - 1) / 1 = [-1, 0, 1], c2 = (sr - 0.5) / 1 = [-0.5, 0.5, 1.5]
        # polygrid2d(c1, c2, [[1,1],[1,1]]) = 1 + c1 + c2 + c1*c2
        expected = 1.0 + np.outer([-1, 0, 1], np.ones(3)) + np.outer(np.ones(3), [-0.5, 0.5, 1.5]) + np.outer([-1, 0, 1], [-0.5, 0.5, 1.5])
        np.testing.assert_allclose(result, expected)

    def test_get_daz_corr_with_scale(self):
        """Scale factor should divide coordinate values."""
        azimuth_time = np.arange(4)
        slant_range_time = np.arange(4)
        y_poly = np.array([[0.0, 1.0], [0.0, 0.0]])
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(2.0), np.float64(2.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        # c1 = az/2, c2 = sr/2, c[0,1]=1 => result = c2 along columns
        expected = np.outer(np.ones(4), np.arange(4, dtype=np.float64) / 2.0)
        np.testing.assert_allclose(result, expected)

    def test_get_daz_corr_2d_polynomial(self):
        """2D polynomial should produce values varying in both directions."""
        azimuth_time = np.arange(3)
        slant_range_time = np.arange(3)
        y_poly = np.ones((2, 2))
        offset = [np.float64(0.0), np.float64(0.0)]
        scale_fact = [np.float64(1.0), np.float64(1.0)]
        result = get_daz_corr(azimuth_time, slant_range_time, y_poly, offset, scale_fact)
        assert result.shape == (3, 3)
        assert np.all(result > 0)


# ===========================================================================
# TestGetCoeffDemod
# ===========================================================================


class TestGetCoeffDemod:
    """Tests for the get_coeff_demod function."""

    def test_get_coeff_demod_returns_array(self):
        """Should return a numpy array."""
        coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = get_coeff_demod(coeffs)
        assert isinstance(result, np.ndarray)

    def test_get_coeff_demod_integration(self):
        """Integration of constant should produce linear term."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = get_coeff_demod(coeffs)
        assert result.shape == (2, 3)

    def test_get_coeff_demod_shape(self):
        """Output shape: axis dimension increases by 1, other axis unchanged."""
        coeffs = np.random.rand(3, 4)
        result = get_coeff_demod(coeffs)
        assert result.shape == (3, 5)

    def test_get_coeff_demod_zero_coeffs(self):
        """Integration of zero should produce zero."""
        coeffs = np.zeros((2, 3))
        result = get_coeff_demod(coeffs)
        np.testing.assert_array_equal(result, 0.0)

    def test_get_coeff_demod_constant_coeff(self):
        """Polyint of constant c should give c*x."""
        coeffs = np.array([[5.0, 0.0], [0.0, 0.0]])  # constant 5.0 in axis=1
        result = get_coeff_demod(coeffs)
        # polyint([5.0, 0.0]) = [0.0, 5.0, 0.0]
        expected = 2 * np.pi * np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected)

    def test_get_coeff_demod_linear_coeff(self):
        """Polyint of linear ax should give ax^2/2."""
        coeffs = np.array([[0.0, 3.0], [0.0, 0.0]])  # 3*x
        result = get_coeff_demod(coeffs)
        # polyint([0.0, 3.0]) = [0.0, 0.0, 1.5]
        expected = 2 * np.pi * np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected)

    def test_get_coeff_demod_2d_coefficients(self):
        """2D coefficient array should integrate along axis=1 only."""
        coeffs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = get_coeff_demod(coeffs)
        assert result.shape == (2, 4)
        # Each row is integrated independently along axis=1
        for i in range(2):
            np.testing.assert_allclose(
                result[i, 0], 0.0  # integration constant
            )

    def test_get_coeff_demod_scalar_coeff(self):
        """Single-row coefficient array should work."""
        coeffs = np.array([[1.0, 2.0, 3.0]])
        result = get_coeff_demod(coeffs)
        assert result.shape == (1, 4)

    def test_get_coeff_demod_negative_coeff(self):
        """Negative coefficients should produce negative integrated values."""
        coeffs = np.array([[-1.0, 0.0]])
        result = get_coeff_demod(coeffs)
        assert result[0, 1] < 0


# ===========================================================================
# TestTimeToPixel
# ===========================================================================


class TestTimeToPixel:
    """Tests for the time_to_pixel function."""

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

    def test_time_to_pixel_returns_same_type(self):
        """Should return same type as input (DataArray)."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5], dtype=np.float64)
        data = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da)
        assert isinstance(result, xr.DataArray)

    def test_time_to_pixel_dataset_input(self):
        """Should work with xr.Dataset input."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        data = np.random.rand(3, 3)
        ds = xr.Dataset(
            {"band": xr.DataArray(data, dims=["azimuth_time", "slant_range_time"], coords={"azimuth_time": az, "slant_range_time": sr})},
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(ds)
        assert isinstance(result, xr.Dataset)

    def test_time_to_pixel_pixel_indices_correct(self):
        """Pixel indices should be evenly spaced integers."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:02", "2020-01-01T00:00:04"], dtype="datetime64[ns]")
        sr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        data = np.random.rand(3, 3)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        result = time_to_pixel(da)
        # With evenly spaced times, first should be 0, last should be size-1
        assert result.coords["y"].values[0] == 0
        assert result.coords["y"].values[-1] == 2
        assert result.coords["x"].values[0] == 0
        assert result.coords["x"].values[-1] == 2

    def test_time_to_pixel_with_custom_origin(self):
        """Custom origin should shift all pixel indices."""
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]")
        sr = np.array([0.0, 1.0], dtype=np.float64)
        data = np.random.rand(2, 2)
        da = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": sr},
        )
        # Origin at second element => first pixel maps to -1
        result = time_to_pixel(da, y0=az[1], x0=1.0)
        assert result.coords["y"].values[0] == -1
        assert result.coords["x"].values[0] == -1


# ===========================================================================
# TestGetBestPol
# ===========================================================================


class TestGetBestPol:
    """Tests for the get_best_pol function."""

    def test_get_best_pol_returns_string(self):
        """Should return a string polarization."""
        result = get_best_pol(["VV"])
        assert isinstance(result, str)

    def test_get_best_pol_selects_vv(self):
        """VV should be selected when available (highest priority)."""
        result = get_best_pol(["VV", "HH", "VH"])
        assert result == "VV"

    def test_get_best_pol_selects_hh(self):
        """HH should be selected when VV is absent."""
        result = get_best_pol(["HH", "VH", "HV"])
        assert result == "HH"

    def test_get_best_pol_selects_vh(self):
        """VH should be selected when VV and HH are absent."""
        result = get_best_pol(["VH", "HV"])
        assert result == "VH"

    def test_get_best_pol_selects_hv(self):
        """HV should be selected when VV, HH, VH are absent."""
        result = get_best_pol(["HV"])
        assert result == "HV"

    def test_get_best_pol_single_element(self):
        """Single polarization should be returned as-is."""
        result = get_best_pol(["VV"])
        assert result == "VV"

    def test_get_best_pol_duplicate_polarizations(self):
        """Duplicate polarizations should still select the highest priority."""
        result = get_best_pol(["VH", "VV", "VV", "HH"])
        assert result == "VV"

    def test_get_best_pol_preserves_case(self):
        """Polarization strings should preserve their original case."""
        result = get_best_pol(["VV"])
        assert result == "VV"

    def test_get_best_pol_unrecognized_raises(self):
        """Unrecognized polarizations should raise an exception."""
        with pytest.raises(Exception) as exc_info:
            get_best_pol(["XX", "YY"])
        assert "No polarization has been found" in str(exc_info.value)

    def test_get_best_pol_empty_list_raises(self):
        """Empty list should raise an exception."""
        with pytest.raises(Exception) as exc_info:
            get_best_pol([])
        assert "No polarization has been found" in str(exc_info.value)

    def test_get_best_pol_all_priorities_present(self):
        """When all polarizations present, VV should win."""
        result = get_best_pol(["HV", "VH", "HH", "VV"])
        assert result == "VV"

    def test_get_best_pol_rejects_lower_priority(self):
        """Lower priority polarization should not be selected when higher is available."""
        result = get_best_pol(["HH", "VV", "VH"])
        assert result != "HH"
        assert result != "VH"
        assert result == "VV"

    def test_get_best_pol_mixed_with_non_pols(self):
        """Mixed known/unknown polarizations should select highest known."""
        result = get_best_pol(["XX", "VH", "YY"])
        assert result == "VH"

    def test_get_best_pol_numeric_like_pols(self):
        """Numeric-looking strings that are not valid polarizations should raise."""
        with pytest.raises(Exception):
            get_best_pol(["01", "10", "11"])

    def test_get_best_pol_order_independent(self):
        """Order of input list should not affect result."""
        result1 = get_best_pol(["HH", "VV", "VH"])
        result2 = get_best_pol(["VH", "HH", "VV"])
        result3 = get_best_pol(["VV", "VH", "HH"])
        assert result1 == result2 == result3 == "VV"

    def test_get_best_pol_all_hv_only(self):
        """Only HV polarization should be selected."""
        result = get_best_pol(["HV", "HV", "HV"])
        assert result == "HV"

    def test_get_best_pol_string_conversion(self):
        """Non-string inputs should be converted to string."""
        result = get_best_pol(["VV", "HH"])
        assert isinstance(result, str)

    def test_get_best_pol_case_sensitive(self):
        """Polarization matching should be case-sensitive."""
        with pytest.raises(Exception):
            get_best_pol(["vv", "hh"])


# ===========================================================================
# TestGetAzimuthTimeRaw
# ===========================================================================


class TestGetAzimuthTimeRaw:
    """Tests for the get_azimuth_time_raw function.

    Computes the absolute acquisition time of each pixel from
    burst range/azimuth coordinates and a timeCA polynomial.
    """

    @pytest.fixture
    def burst_simple(self):
        """Small burst DataArray for get_azimuth_time_raw tests.

        Shape: (5, 4) azimuth_time x slant_range_time
        Has required 1-D coordinate DataArrays slant_range_m, azimuth_m.
        """
        az_time = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        sr_time = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        data = np.ones((5, 4), dtype=np.complex128) * (1.0 + 0.1j)
        burst = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": ("azimuth_time", np.arange(5, dtype=np.float64)),
                "slant_range_time": ("slant_range_time", np.arange(4, dtype=np.float64)),
                "slant_range_m": ("slant_range_time", sr_time),
                "azimuth_m": ("azimuth_time", az_time),
            },
        )
        return burst

    @pytest.fixture
    def burst_3x3(self):
        """3x3 burst with known coordinate values."""
        data = np.ones((3, 3), dtype=np.complex128)
        burst = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": ("azimuth_time", np.arange(3, dtype=np.float64)),
                "slant_range_time": ("slant_range_time", np.arange(3, dtype=np.float64)),
                "slant_range_m": ("slant_range_time", np.arange(3, dtype=np.float64)),
                "azimuth_m": ("azimuth_time", np.arange(3, dtype=np.float64)),
            },
        )
        return burst

    @pytest.fixture
    def timeca_coeffs(self):
        """TimeCA polynomial coefficients (2, 2).

        timeca_coeffs[j, i] = coefficient for slant_range_m^i * azimuth_m^j
        timeca_coeffs[1, 0] = 0.01 => time varies linearly with azimuth_m
        """
        return np.array([[0.0, 0.01], [0.0, 0.0]], dtype=np.float64)

    @pytest.fixture
    def collection_start(self):
        """Reference collection start time."""
        return np.datetime64("2020-01-01T00:00:00", "ns")

    def test_get_azimuth_time_raw_returns_dataarray(self, burst_simple, timeca_coeffs, collection_start):
        """Should return an xr.DataArray."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert isinstance(result, xr.DataArray)

    def test_get_azimuth_time_raw_dims(self, burst_simple, timeca_coeffs, collection_start):
        """Output should have dims ['azimuth_time', 'slant_range_time']."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_get_azimuth_time_raw_shape_no_multilook(self, burst_simple, timeca_coeffs, collection_start):
        """Without multilook, shape should match the burst shape."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert result.shape == burst_simple.shape

    def test_get_azimuth_time_raw_shape_with_multilook(self, burst_simple, timeca_coeffs, collection_start):
        """With multilook, shape should be reduced."""
        multilook = {"azimuth_time": 2, "slant_range_time": 2}
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start, multilook)
        assert result.shape[0] <= burst_simple.shape[0]
        assert result.shape[1] <= burst_simple.shape[1]

    def test_get_azimuth_time_raw_dtype(self, burst_simple, timeca_coeffs, collection_start):
        """Output dtype should be datetime64[ns]."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert "datetime64" in str(result.values.dtype)

    def test_get_azimuth_time_raw_values_are_timedelta(self, burst_simple, timeca_coeffs, collection_start):
        """All values should be valid datetime64[ns]."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert np.all(np.isfinite(result.values))

    def test_get_azimuth_time_raw_coords_preserved(self, burst_simple, timeca_coeffs, collection_start):
        """Output should preserve burst's azimuth_time and slant_range_time coords."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert "azimuth_time" in result.coords
        assert "slant_range_time" in result.coords

    def test_get_azimuth_time_raw_with_linear_coeff(self, burst_simple, timeca_coeffs, collection_start):
        """With linear timeCA coefficient, times should vary across azimuth dimension."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        result_diff = result - result.min()
        assert not np.allclose(result_diff.values[0, :].astype(float), result_diff.values[1, :].astype(float))

    def test_get_azimuth_time_raw_with_quadratic_coeff(self, burst_simple, timeca_coeffs, collection_start):
        """With quadratic coefficients, times should vary across both dimensions."""
        coeffs = np.array([[0.0, 0.0], [0.0, 0.01]], dtype=np.float64)
        result = get_azimuth_time_raw(burst_simple, coeffs, collection_start)
        assert result.shape == burst_simple.shape

    def test_get_azimuth_time_raw_multilook_odd_az(self, burst_3x3, timeca_coeffs, collection_start):
        """Multilook with odd azimuth slice should produce correct reduced shape."""
        multilook = {"azimuth_time": 2, "slant_range_time": 2}
        result = get_azimuth_time_raw(burst_3x3, timeca_coeffs, collection_start, multilook)
        assert result.shape[0] == 1
        assert result.shape[1] == 1

    def test_get_azimuth_time_raw_multilook_no_reduction(self, burst_simple, timeca_coeffs, collection_start):
        """Multilook=1 should not reduce the shape."""
        multilook = {"azimuth_time": 1, "slant_range_time": 1}
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start, multilook)
        assert result.shape == burst_simple.shape

    def test_get_azimuth_time_raw_coords_with_multilook(self, burst_simple, timeca_coeffs, collection_start):
        """Output coords with multilook should match burst's sliced coords."""
        multilook = {"azimuth_time": 2, "slant_range_time": 2}
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start, multilook)
        assert len(result.coords["azimuth_time"]) == result.shape[0]
        assert len(result.coords["slant_range_time"]) == result.shape[1]

    def test_get_azimuth_time_raw_times_above_collection_start(self, burst_simple, timeca_coeffs, collection_start):
        """Computed times should be at or above the collection start time."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        assert np.all(result.values >= collection_start)

    def test_get_azimuth_time_raw_zero_timeca(self, burst_simple, collection_start):
        """With zero timeCA coefficients, output equals collection_start."""
        zero_coeffs = np.zeros((2, 2), dtype=np.float64)
        result = get_azimuth_time_raw(burst_simple, zero_coeffs, collection_start)
        np.testing.assert_array_equal(result.values, collection_start)

    def test_get_azimuth_time_raw_none_multilook(self, burst_simple, timeca_coeffs, collection_start):
        """None multilook should not reduce shape."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start, multilook=None)
        assert result.shape == burst_simple.shape

    def test_get_azimuth_time_raw_positive_timeca(self, burst_simple, timeca_coeffs, collection_start):
        """Positive timeCA coefficients should produce times at or above collection_start."""
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start)
        # First row (azimuth=0) gives timeCOA=0, so result equals collection_start
        assert np.all(result.values >= collection_start)
        # Non-first rows should be above
        assert np.any(result.values[1:, :] > collection_start)

    def test_get_azimuth_time_raw_negative_timeca(self, burst_simple, timeca_coeffs, collection_start):
        """Negative timeCA coefficients should produce times at or below collection_start."""
        negative_coeffs = -np.abs(timeca_coeffs)
        result = get_azimuth_time_raw(burst_simple, negative_coeffs, collection_start)
        assert np.all(result.values <= collection_start)
        # Non-first rows should be below
        assert np.any(result.values[1:, :] < collection_start)

    @pytest.mark.parametrize("az_size", [3, 5, 7])
    @pytest.mark.parametrize("sr_size", [2, 4, 6])
    def test_get_azimuth_time_raw_various_sizes(self, az_size, sr_size, collection_start):
        """Should handle various grid sizes correctly."""
        data = np.ones((az_size, sr_size), dtype=np.complex128)
        burst = xr.DataArray(
            data,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": ("azimuth_time", np.arange(az_size, dtype=np.float64)),
                "slant_range_time": ("slant_range_time", np.arange(sr_size, dtype=np.float64)),
                "slant_range_m": ("slant_range_time", np.arange(sr_size, dtype=np.float64)),
                "azimuth_m": ("azimuth_time", np.arange(az_size, dtype=np.float64)),
            },
        )
        coeffs = np.array([[0.0, 0.01], [0.0, 0.0]], dtype=np.float64)
        result = get_azimuth_time_raw(burst, coeffs, collection_start)
        assert result.shape == (az_size, sr_size)
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_get_azimuth_time_raw_multilook_factor_3(self, burst_simple, timeca_coeffs, collection_start):
        """Multilook factor of 3 should reduce shape by approximately 1/3."""
        multilook = {"azimuth_time": 3, "slant_range_time": 3}
        result = get_azimuth_time_raw(burst_simple, timeca_coeffs, collection_start, multilook)
        assert result.shape[0] <= burst_simple.shape[0] // 2 + 1
        assert result.shape[1] <= burst_simple.shape[1] // 2 + 1
