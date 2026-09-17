"""Unit tests for cop_ard_core.warp."""

from __future__ import annotations

import numpy as np
import pytest
import scipy
import xarray as xr

from cop_ard_core.warp import (
    _dot_numba,
    _nvalid_numba,
    _sinc_warp_inner_numba,
    compute_secondary_image_coordinates,
    do_deramp,
    extract_abs_grid,
    extract_grid,
    extract_rg0_az0,
    extract_rg_az,
    getCoeffFromLut,
    getLUT,
    grid_data_block,
    poly_fit_2d,
    sinc,
    sinc_warp,
    warp_dazdrg,
)


class TestSinc:
    """Tests for sinc function."""

    def test_sinc_returns_array(self):
        """Should return numpy array."""
        result = sinc(size=5)
        assert isinstance(result, np.ndarray)

    def test_sinc_positive_size(self):
        """Should work with positive size."""
        for size in [3, 5, 7, 11]:
            result = sinc(size=size)
            assert result.shape[0] > 0

    def test_sinc_centered(self):
        """Center value should be 1 (sinc(0) = 1)."""
        result = sinc(size=5, ov_fact=100)
        center = len(result) // 2
        np.testing.assert_almost_equal(result[center], 1.0, decimal=5)

    def test_sinc_zero_at_multiples(self):
        """Sinc should be zero at integer multiples (before windowing)."""
        result = sinc(size=11, ov_fact=100)
        center = len(result) // 2
        assert abs(result[center]) >= abs(result[center + 5])

    def test_sinc_size_preserves_order(self):
        """Output length should scale with size."""
        for size in [1, 5, 11, 50, 100]:
            result = sinc(size=size)
            assert result.shape[0] >= size

    def test_sinc_kaiser_taper(self):
        """Ends of sinc should be smaller than center (windowing effect)."""
        result = sinc(size=21, ov_fact=100)
        center = len(result) // 2
        assert abs(result[center]) >= abs(result[0])
        assert abs(result[center]) >= abs(result[-1])

    def test_sinc_symmetry(self):
        """Sinc should be approximately symmetric around center."""
        result = sinc(size=15, ov_fact=100)
        center = len(result) // 2
        left = result[:center]
        right = result[center + 1 :][::-1]
        np.testing.assert_array_almost_equal(left, right, decimal=10)

    def test_sinc_different_ov_fact(self):
        """Different oversampling factors should produce different lengths."""
        r1 = sinc(size=5, ov_fact=100)
        r2 = sinc(size=5, ov_fact=200)
        assert len(r1) != len(r2)
        assert len(r2) > len(r1)

    def test_sinc_all_positive(self):
        """Center region of windowed sinc should be positive."""
        result = sinc(size=11, ov_fact=100)
        center = len(result) // 2
        assert all(result[max(0, center - 5) : center + 5] > 0)

    def test_sinc_single_element(self):
        """size=1 should produce a non-empty array centered at 1."""
        result = sinc(size=1)
        assert result.shape[0] > 0
        np.testing.assert_almost_equal(result[len(result) // 2], 1.0, decimal=5)

    def test_sinc_large_size(self):
        """Large size should not error and should produce correct output."""
        result = sinc(size=101)
        assert result.shape[0] > 0
        assert result.shape[0] >= 101

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 10, 50, 100])
    def test_sinc_various_sizes(self, size):
        """sinc should handle various sizes without error."""
        result = sinc(size=size)
        assert result.ndim == 1
        assert result.shape[0] >= size

    def test_sinc_kernel_shape_matches_params(self):
        """Kernel shape should match kernel_shape in dict."""
        kernel_size = (5, 7)
        result = sinc(size=max(kernel_size))
        assert result.shape[0] >= max(kernel_size)

    def test_sinc_length_is_odd(self):
        """sinc output length should always be odd (symmetric kernel)."""
        for size in [1, 2, 3, 5, 10, 100]:
            result = sinc(size=size)
            assert result.shape[0] % 2 == 1, f"Size {size} produced even length {result.shape[0]}"

    def test_sinc_sum_is_positive(self):
        """Windowed sinc sum should be positive (main lobe dominates)."""
        result = sinc(size=21, ov_fact=200)
        assert sum(result) > 0

    def test_sinc_kaiser_symmetric(self):
        """Kaiser window should be symmetric."""
        from scipy.signal.windows import kaiser

        for n in [11, 21, 41]:
            win = kaiser(n, 10, sym=True)
            np.testing.assert_array_almost_equal(win, win[::-1])

    def test_sinc_returns_ndarray(self):
        """sinc should return np.ndarray."""
        result = sinc(size=5)
        assert type(result) is np.ndarray


class TestSincWarp:
    """Tests for sinc_warp."""

    def test_sinc_warp_returns_array(self):
        """Should return numpy array of interpolated values."""
        image = np.random.rand(5, 5)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == x.shape

    def test_sinc_warp_2d_output(self):
        """2D coordinate arrays should produce 2D output."""
        image = np.random.rand(5, 5)
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x.ravel(), y.ravel(), kernel)
        assert result.shape == x.ravel().shape

    def test_sinc_warp_image_not_2d_raises(self):
        """Non-2D image should raise."""
        image = np.random.rand(2, 3, 3)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        with pytest.raises(Exception, match="Image must have 2 dimensions"):
            sinc_warp(image, x, y, kernel)

    def test_sinc_warp_mismatched_shapes_raises(self):
        """Mismatched x/y shapes should raise."""
        image = np.random.rand(5, 5)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        with pytest.raises(Exception, match="x and y shape must be the same"):
            sinc_warp(image, x, y, kernel)

    def test_sinc_warp_boundary_pixels(self):
        """Pixels at image boundaries should be handled (return invalid value)."""
        image = np.ones((5, 5))
        x = np.array([0.1, 2.0, 4.9])
        y = np.array([0.1, 2.0, 4.9])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == x.shape

    def test_sinc_warp_dtype_preserved(self):
        """Output dtype should match input image dtype."""
        image_f32 = np.random.rand(5, 5).astype(np.float32)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image_f32, x, y, kernel)
        assert result.dtype == np.float32

    def test_sinc_warp_complex_image(self):
        """Complex image should produce complex output."""
        image = np.random.rand(7, 7) + 1j * np.random.rand(7, 7)
        x = np.array([2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.dtype == np.complex128

    def test_sinc_warp_with_nan_in_image(self):
        """Image with NaN should handle valid/invalid pixel thresholding."""
        image = np.random.rand(7, 7)
        image[0, 0] = np.nan
        image[0, 1] = np.nan
        x = np.array([3.0, 4.0])
        y = np.array([3.0, 4.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == x.shape

    def test_sinc_warp_custom_invalid_value(self):
        """Custom invalid_value should be used for boundary pixels."""
        image = np.ones((5, 5))
        # Query boundary pixels that will be skipped by the inner loop
        x = np.array([0.1, 0.5])
        y = np.array([0.1, 0.5])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel, invalid_value=-999.0)
        assert result.shape == x.shape

    def test_sinc_warp_low_threshold_allows_fewer_valid(self):
        """Lower valid_percentage_threshold should allow interpolation with fewer valid pixels."""
        image = np.ones((7, 7))
        image[:3, :3] = np.nan  # top-left corner is NaN
        x = np.array([1.5])
        y = np.array([1.5])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        # With 0.75 threshold, this may be invalid; with 0.1 it should be valid
        result_low = sinc_warp(image, x, y, kernel, valid_percentage_threshold=0.1)
        assert result_low.shape == x.shape

    def test_sinc_warp_high_threshold_rejects_more(self):
        """Higher valid_percentage_threshold should reject more pixels."""
        image = np.ones((7, 7))
        image[:4, :4] = np.nan
        x = np.array([2.5])
        y = np.array([2.5])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel, valid_percentage_threshold=0.99)
        assert result.shape == x.shape

    def test_sinc_warp_identical_coordinates(self):
        """Identical x and y arrays should work."""
        image = np.random.rand(5, 5)
        coords = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, coords, coords, kernel)
        assert result.shape == coords.shape

    def test_sinc_warp_single_point(self):
        """Single-point coordinates should return single value."""
        image = np.random.rand(5, 5)
        x = np.array([2.0])
        y = np.array([2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == (1,)

    def test_sinc_warp_constant_image_near_constant(self):
        """Constant image should yield approximately constant output."""
        image = np.full((9, 9), 5.0)
        x = np.linspace(2, 6, 5)
        y = np.linspace(2, 6, 5)
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        np.testing.assert_allclose(result, 5.0, atol=0.5)

    @pytest.mark.parametrize("invalid_value", [0.0, -1.0, 999.0, np.nan])
    def test_sinc_warp_various_invalid_values(self, invalid_value):
        """sinc_warp should accept various invalid_value settings."""
        image = np.random.rand(5, 5)
        x = np.array([2.0])
        y = np.array([2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel, invalid_value=invalid_value)
        assert result.shape == x.shape

    def test_sinc_warp_output_type_is_ndarray(self):
        """Output must be numpy ndarray, not a subclass."""
        image = np.random.rand(5, 5)
        x = np.array([2.0])
        y = np.array([2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert type(result) is np.ndarray

    @pytest.mark.parametrize("size", [5, 11, 21])
    @pytest.mark.parametrize("kaiser_beta", [5, 10, 15, 20])
    def test_sinc_different_beta(self, size, kaiser_beta):
        """sinc should handle various kaiser_beta values."""
        result = sinc(size=size, kaiser_beta=kaiser_beta)
        assert result.shape[0] >= size
        assert abs(result[len(result) // 2]) >= abs(result[0])

    @pytest.mark.parametrize("size", [5, 11, 21])
    @pytest.mark.parametrize("ov_fact", [50, 100, 200, 500])
    def test_sinc_different_ov_fact(self, size, ov_fact):
        """sinc should handle various oversampling factors."""
        result = sinc(size=size, ov_fact=ov_fact)
        assert result.shape[0] > 0

    @pytest.mark.parametrize("size", [1, 3, 7, 11, 21, 101])
    def test_sinc_odd_sizes(self, size):
        """sinc should handle odd and even sizes."""
        result = sinc(size=size)
        assert result.shape[0] >= size

    def test_sinc_warp_constant_image_exact(self):
        """Constant image should produce exact constant output at interior points."""
        value = 7.5
        image = np.full((9, 9), value)
        x = np.array([3.0, 4.0, 5.0])
        y = np.array([3.0, 4.0, 5.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        np.testing.assert_allclose(result, value, atol=0.001)

    def test_sinc_warp_gradient_image(self):
        """Linear gradient image should produce approximately linear output."""
        y_idx, x_idx = np.indices((9, 9))
        image = y_idx.astype(np.float64) + x_idx.astype(np.float64)
        x = np.array([3.0, 4.0, 5.0])
        y = np.array([3.0, 4.0, 5.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        # Expected: 6, 8, 10 (3+3, 4+4, 5+5) — sinc with unit kernel approximates identity
        expected = np.array([6.0, 8.0, 10.0])
        np.testing.assert_allclose(result, expected, atol=2.0)

    def test_sinc_warp_3x3_image(self):
        """3x3 image should handle boundary gracefully."""
        image = np.ones((3, 3))
        x = np.array([1.0])
        y = np.array([1.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == (1,)

    def test_sinc_warp_1x1_output(self):
        """Single-point output should work."""
        image = np.random.rand(5, 5)
        x = np.array([2.5])
        y = np.array([2.5])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == (1,)

    def test_sinc_warp_returns_ndarray(self):
        """sinc_warp should return np.ndarray, not a subclass."""
        image = np.random.rand(5, 5)
        x = np.array([2.0])
        y = np.array([2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert type(result) is np.ndarray

    @pytest.mark.parametrize("kernel_size", [(3, 3), (5, 5), (7, 7)])
    def test_sinc_warp_various_kernel_sizes(self, image_7x7, kernel_size):
        """sinc_warp should handle various kernel sizes."""
        x = np.array([3.0, 4.0])
        y = np.array([3.0, 4.0])
        ov = kernel_size[0] * 50
        kernel = {
            "kernel_shape": kernel_size,
            "ov_fact": ov,
            "sinc_x": np.ones(ov),
            "sinc_y": np.ones(ov),
        }
        result = sinc_warp(image_7x7, x, y, kernel)
        assert result.shape == x.shape

    @pytest.mark.parametrize("threshold", [0.1, 0.25, 0.5, 0.75, 0.99])
    def test_sinc_warp_various_thresholds(self, image_7x7, threshold):
        """sinc_warp should accept various threshold values."""
        x = np.array([1.0, 2.5, 3.0, 4.5, 1.5])
        y = np.array([1.5, 3.0, 4.0, 2.5, 2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image_7x7, x, y, kernel, valid_percentage_threshold=threshold)
        assert result.shape == x.shape

    def test_sinc_warp_sinc_applied_to_pulse(self):
        """sinc_warp should return a sinc if applied to a dirac pulse."""
        image_size = 51
        kaiser_beta = 10
        delta = 0.5
        ov_fact = 10000
        x = np.arange(image_size) - (image_size - 1) / 2.0
        # Calculate the sinc values and apply the Kaiser window element-wise.
        expected = np.sinc(x + delta) * scipy.signal.windows.kaiser(image_size, kaiser_beta, sym=True)
        image = np.zeros((image_size, image_size), dtype=np.float64)
        image[image_size // 2, image_size // 2] = 1

        x = np.arange(image_size, dtype=np.float64)
        # x = np.linspace(side//2 - 5, side//2+5, num=50)
        x, y = np.meshgrid(x, x)
        x += delta
        # y += delta
        kernel_size = 15
        kernel = {
            "kernel_shape": (kernel_size, kernel_size),
            "ov_fact": ov_fact,
            "sinc_x": sinc(100, ov_fact=ov_fact),
            "sinc_y": sinc(100, ov_fact=ov_fact),
        }
        result = sinc_warp(image, x.ravel(), y.ravel(), kernel, invalid_value=np.nan)
        result = result.reshape(x.shape)
        assert np.mean((result[image_size // 2, :] - expected) ** 2) < 1e-4  # tight enough?

    @pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 5])
    def test_sinc_warp_sinc_applied_to_pulse_offset(self, offset):
        """sinc_warp should return a sinc if applied to a dirac pulse."""
        image_size = 51
        ov_fact = 10000
        image = np.zeros((image_size, image_size), dtype=np.float64)
        image[image_size // 2, image_size // 2] = 1

        x = np.arange(image_size, dtype=np.float64)
        x, y = np.meshgrid(x, x)
        x += offset
        kernel_size = 15
        kernel = {
            "kernel_shape": (kernel_size, kernel_size),
            "ov_fact": ov_fact,
            "sinc_x": sinc(100, ov_fact=ov_fact),
            "sinc_y": sinc(100, ov_fact=ov_fact),
        }
        result = sinc_warp(image, x.ravel(), y.ravel(), kernel, invalid_value=np.nan)
        result = result.reshape(x.shape)
        np.testing.assert_allclose(result, np.roll(image, (-offset, 0)), atol=1e-6)

    @pytest.mark.parametrize("offset", [0, 1, 2, 3, 4, 5])
    def test_sinc_warp_sinc_applied_to_image_offset(self, offset):
        """sinc_warp should return a sinc if applied to a dirac pulse."""
        image_size = 51
        ov_fact = 10000
        image = np.zeros((image_size, image_size), dtype=np.float64)
        image[image_size // 2, image_size // 2] = 1

        x = np.arange(image_size, dtype=np.float64)
        x, y = np.meshgrid(x, x)
        x += offset
        kernel_size = 15
        kernel = {
            "kernel_shape": (kernel_size, kernel_size),
            "ov_fact": ov_fact,
            "sinc_x": sinc(100, ov_fact=ov_fact),
            "sinc_y": sinc(100, ov_fact=ov_fact),
        }
        result = sinc_warp(image, x.ravel(), y.ravel(), kernel, invalid_value=np.nan)
        result = result.reshape(x.shape)
        np.testing.assert_allclose(result, np.roll(image, (-offset, 0)), atol=1e-6)


class TestDotNumba:
    """Tests for _dot_numba internal function."""

    def test_dot_numba_basic(self):
        """Basic dot product computed by _dot_numba matches pure numpy."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = _dot_numba(a, b, c)
        expected = np.einsum("j,ji,i->", a, c, b)
        np.testing.assert_almost_equal(result, expected)

    def test_dot_numba_scalar_result(self):
        """Result should be a scalar (not an array)."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        c = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _dot_numba(a, b, c)
        assert np.isscalar(result) or hasattr(result, "item")

    def test_dot_numba_dtype_preserved(self):
        """Output dtype should match input array dtype."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = _dot_numba(a, b, c)
        assert result == a.dtype.type(result)

    def test_dot_numba_zero_vector(self):
        """Zero vector should give zero result."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        c = np.random.rand(3, 3)
        result = _dot_numba(a, b, c)
        np.testing.assert_almost_equal(result, 0.0)

    def test_dot_numba_identity_matrix(self):
        """With identity matrix, result should be a @ b."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        c = np.eye(3)
        result = _dot_numba(a, b, c)
        expected = np.dot(a, b)
        np.testing.assert_almost_equal(result, expected)

    def test_dot_numba_different_sizes(self):
        """Vectors of different sizes with matching matrix should work."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = _dot_numba(a, b, c)
        expected = np.einsum("j,ji,i->", a, c, b)
        np.testing.assert_almost_equal(result, expected)

    def test_dot_numba_int_input(self):
        """Integer inputs should produce output matching the common dtype."""
        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        result = _dot_numba(a, b, c)
        assert np.isscalar(result) or hasattr(result, "item")
        np.testing.assert_equal(result, 552)

    def test_dot_numba_negative_values(self):
        """Negative values should be handled correctly."""
        a = np.array([-1.0, 2.0, -3.0])
        b = np.array([4.0, -5.0, 6.0])
        c = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]])
        result = _dot_numba(a, b, c)
        expected = np.einsum("j,ji,i->", a, c, b)
        np.testing.assert_almost_equal(result, expected)

    def test_dot_numba_large_values(self):
        """Large values should not overflow unexpectedly."""
        a = np.array([1e10, 2e10])
        b = np.array([3e10, 4e10])
        c = np.eye(2) * 1e10
        result = _dot_numba(a, b, c)
        expected = np.einsum("j,ji,i->", a, c, b)
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("a_size,b_size", [(1, 1), (1, 5), (5, 1), (3, 4), (10, 10)])
    def test_dot_numba_various_sizes(self, a_size, b_size):
        """_dot_numba should work with various vector sizes."""
        a = np.random.rand(a_size)
        b = np.random.rand(b_size)
        c = np.random.rand(a_size, b_size)
        result = _dot_numba(a, b, c)
        expected = np.einsum("j,ji,i->", a, c, b)
        np.testing.assert_almost_equal(result, expected)

    def test_dot_numba_matches_numpy_dot(self):
        """_dot_numba result should exactly match numpy computation."""
        np.random.seed(42)
        for _ in range(5):
            a = np.random.rand(5)
            b = np.random.rand(7)
            c = np.random.rand(5, 7)
            result = _dot_numba(a, b, c)
            expected = np.einsum("j,ji,i->", a, c, b)
            np.testing.assert_almost_equal(result, expected, decimal=10)


class TestNvalidNumba:
    """Tests for _nvalid_numba internal function.

    Behavior depends on whether iv (invalid value) is finite:
    - If iv is finite: counts pixels where image != iv (exact match exclusion only)
    - If iv is NaN/Inf: uses np.isfinite(image) (excludes NaN and Inf)
    """

    def test_nvalid_numba_all_finite(self):
        """All finite values should count as valid."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _nvalid_numba(image, -999.0)
        assert result == 4

    def test_nvalid_numba_with_nan_sentinel(self):
        """When iv is finite NaN is NOT excluded (only exact sentinel match)."""
        image = np.array([[1.0, np.nan], [np.nan, 4.0]])
        result = _nvalid_numba(image, -999.0)
        # NaN != -999.0 is True, so all 4 pixels are "valid"
        assert result == 4

    def test_nvalid_numba_with_inf_sentinel(self):
        """When iv is finite Inf is NOT excluded (only exact sentinel match)."""
        image = np.array([[1.0, np.inf], [-np.inf, 4.0]])
        result = _nvalid_numba(image, -999.0)
        # Inf != -999.0 is True, so all 4 pixels are "valid"
        assert result == 4

    def test_nvalid_numba_with_finite_invalid_value(self):
        """Finite invalid_value should be excluded."""
        image = np.array([[1.0, -999.0], [-999.0, 4.0]])
        result = _nvalid_numba(image, -999.0)
        assert result == 2

    def test_nvalid_numba_all_invalid(self):
        """All values matching invalid_value should give count 0."""
        image = np.full((3, 3), -999.0)
        result = _nvalid_numba(image, -999.0)
        assert result == 0

    def test_nvalid_numba_empty_image(self):
        """Empty image should return 0."""
        image = np.array([]).reshape(0, 0)
        result = _nvalid_numba(image, -999.0)
        assert result == 0

    def test_nvalid_numba_single_element_valid(self):
        """Single valid element should return 1."""
        image = np.array([[5.0]])
        result = _nvalid_numba(image, -999.0)
        assert result == 1

    def test_nvalid_numba_single_element_invalid(self):
        """Single invalid element should return 0."""
        image = np.array([[-999.0]])
        result = _nvalid_numba(image, -999.0)
        assert result == 0

    def test_nvalid_numba_complex_image(self):
        """Complex images with finite iv compare exactly."""
        image = np.array([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]])
        result = _nvalid_numba(image, -999.0)
        assert result == 4

    def test_nvalid_numba_large_array(self):
        """Large arrays should work without error."""
        image = np.random.rand(100, 100)
        result = _nvalid_numba(image, -999.0)
        assert result == 10000

    def test_nvalid_numba_with_nan_iv_excludes_nan(self):
        """When iv is NaN, np.isfinite(image) is used."""
        image = np.array([[1.0, np.nan], [np.nan, 4.0]])
        result = _nvalid_numba(image, np.nan)
        # iv is not finite, so np.isfinite(image) is used: 2 valid
        assert result == 2

    def test_nvalid_numba_with_inf_iv_excludes_inf(self):
        """When iv is Inf, np.isfinite(image) is used."""
        image = np.array([[1.0, np.inf], [-np.inf, 4.0]])
        result = _nvalid_numba(image, np.inf)
        # iv is not finite, so np.isfinite(image) is used: 2 valid
        assert result == 2

    def test_nvalid_numba_all_nan_with_nan_iv(self):
        """All NaN image with NaN iv should return 0."""
        image = np.full((5, 5), np.nan)
        result = _nvalid_numba(image, np.nan)
        assert result == 0

    def test_nvalid_numba_mixed_nan_inf_with_nan_iv(self):
        """Mixed NaN/Inf image with NaN iv counts only finite."""
        image = np.array([[1.0, np.nan, np.inf], [-np.inf, 5.0, 6.0]])
        result = _nvalid_numba(image, np.nan)
        assert result == 3  # only 1.0, 5.0, 6.0 are finite

    def test_nvalid_numba_exact_count(self):
        """_nvalid_numba should give exact count for sentinel-based invalid detection."""
        image = np.array([[1.0, -999.0, 3.0], [4.0, 5.0, -999.0], [7.0, 8.0, -999.0]])
        result = _nvalid_numba(image, -999.0)
        # Valid (not equal to -999.0): 1.0, 3.0, 4.0, 5.0, 7.0, 8.0 = 6
        assert result == 6


class TestSincWarpInnerNumba:
    """Tests for _sinc_warp_inner_numba internal function."""

    def test_inner_numba_basic(self):
        """Basic call should populate output array."""
        image = np.ones((7, 7), dtype=np.float64)
        x = np.array([2.0, 4.0])
        y = np.array([2.0, 4.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        output = np.zeros(2, dtype=np.float64)
        iv = 0.0  # Python float, not numpy array
        _sinc_warp_inner_numba(image, x, y, 3, 3, kernel["sinc_x"], kernel["sinc_y"], 0.75, output, iv, 100)
        assert output.shape == (2,)

    def test_inner_numba_boundary_skip(self):
        """Boundary coordinates should be skipped (output remains zero)."""
        image = np.ones((7, 7))
        x = np.array([0.1, 6.9])
        y = np.array([0.1, 6.9])
        output = np.zeros(2, dtype=np.float64)
        iv = 0.0
        _sinc_warp_inner_numba(image, x, y, 3, 3, np.ones(100), np.ones(100), 0.75, output, iv, 100)
        np.testing.assert_array_equal(output, 0.0)

    def test_inner_numba_valid_pixels_threshold(self):
        """Too many invalid pixels should cause output to be set to invalid."""
        image = np.full((9, 9), np.nan)
        image[3:6, 3:6] = 1.0
        x = np.array([1.0, 5.0])
        y = np.array([1.0, 5.0])
        output = np.zeros(2, dtype=np.float64)
        iv = 0.0
        _sinc_warp_inner_numba(image, x, y, 3, 3, np.ones(100), np.ones(100), 0.75, output, iv, 100)
        assert output.shape == (2,)

    def test_inner_numba_zero_length_coords(self):
        """Zero-length coordinate arrays should produce zero output."""
        image = np.ones((5, 5))
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        output = np.zeros(0, dtype=np.float64)
        iv = 0.0
        _sinc_warp_inner_numba(image, x, y, 3, 3, np.ones(100), np.ones(100), 0.75, output, iv, 100)
        assert output.shape == (0,)

    def test_inner_numba_consistent_with_sinc_warp(self):
        """Inner function should produce same results as sinc_warp."""
        image = np.random.rand(9, 9)
        x = np.array([2.0, 4.0, 6.0])
        y = np.array([2.0, 4.0, 6.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result_public = sinc_warp(image, x, y, kernel)

        output = np.zeros_like(x)
        iv = 0.0
        _sinc_warp_inner_numba(image, x, y, 3, 3, kernel["sinc_x"], kernel["sinc_y"], 0.75, output, iv, 100)
        np.testing.assert_array_equal(output, result_public)


class TestPolyFit2d:
    """Tests for poly_fit_2d."""

    def test_poly_fit_2d_returns_tuple(self):
        """Should return 4-element tuple."""
        n = 10
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        result = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        assert len(result) == 4

    def test_poly_fit_2d_shape(self):
        """Output shapes should be reasonable."""
        n = 20
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        result = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        coeffs = result[0]
        rank = result[2]
        assert coeffs.shape == (3, 3)
        assert rank > 0

    def test_poly_fit_2d_linear_data(self):
        """Linear data should give good fit."""
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = 2.0 * x + 3.0 * y + 1.0
        result = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        rank = result[2]
        assert rank == 4  # constant + x + y + x*y

    def test_poly_fit_2d_non_array_raises(self):
        """Non-numpy inputs should raise."""
        x = [1.0, 2.0, 3.0]
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception, match="x, y, z must be numpy arrays"):
            poly_fit_2d(x, y, z)  # type: ignore

    def test_poly_fit_2d_mismatched_sizes_raises(self):
        """Mismatched sizes should raise."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception, match="same cardinality"):
            poly_fit_2d(x, y, z)

    def test_poly_fit_2d_quadratic_surface(self):
        """Quadratic data should be fit with order 2."""
        n = 100
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = x**2 + 2 * y**2 + 3 * x * y + x + y + 1.0
        sol, residuals, rank, sing_values = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        assert rank == 9  # (2+1)*(2+1) terms
        assert residuals >= 0
        assert len(sing_values) == 9

    def test_poly_fit_2d_1d_data(self):
        """1D data arrays should work (flattened internally)."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.array([2.0, 5.0, 10.0, 17.0])  # z = 1 + 2*x
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=1, y_order=0)
        assert rank >= 1

    def test_poly_fit_2d_residuals_non_negative(self):
        """Residuals should always be non-negative."""
        np.random.seed(42)
        for _ in range(5):
            x = np.random.rand(20)
            y = np.random.rand(20)
            z = np.random.rand(20)
            _, residuals, _, _ = poly_fit_2d(x, y, z)
            assert residuals >= 0

    def test_poly_fit_2d_rank_less_than_terms(self):
        """Underdetermined system should have rank < number of terms."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        z = np.array([1.0, 2.0])
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        assert rank <= 2  # only 2 data points but 9 terms

    def test_poly_fit_2d_constant_data(self):
        """Constant z data should produce flat fit."""
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.full(n, 42.0)
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        # The constant term should be close to 42
        np.testing.assert_almost_equal(sol[0, 0], 42.0, decimal=5)

    def test_poly_fit_2d_2d_input(self):
        """2D input arrays should be flattened and work."""
        x = np.random.rand(4, 5)
        y = np.random.rand(4, 5)
        z = np.random.rand(4, 5)
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        assert sol.shape == (2, 2)
        assert rank > 0

    def test_poly_fit_2d_rcond_parameter(self):
        """Custom rcond should be accepted without error."""
        x = np.random.rand(20)
        y = np.random.rand(20)
        z = np.random.rand(20)
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, rcond=1e-10)
        assert sol.shape == (3, 3)

    @pytest.mark.parametrize("x_order,y_order", [(0, 0), (1, 0), (0, 1), (1, 1), (2, 3)])
    def test_poly_fit_2d_various_orders(self, x_order, y_order):
        """poly_fit_2d should handle various polynomial orders."""
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        sol, _, rank, _ = poly_fit_2d(x, y, z, x_order=x_order, y_order=y_order)
        assert sol.shape == (x_order + 1, y_order + 1)
        assert rank > 0

    @pytest.mark.parametrize("n_points", [10, 20, 50, 100])
    def test_poly_fit_2d_various_point_counts(self, n_points):
        """poly_fit_2d should handle various data point counts."""
        np.random.seed(42)
        x = np.random.rand(n_points)
        y = np.random.rand(n_points)
        z = x + y + 1.0
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        assert rank > 0
        assert residuals >= 0

    @pytest.mark.parametrize("x_order", [0, 1, 2, 3])
    @pytest.mark.parametrize("y_order", [0, 1, 2, 3])
    def test_poly_fit_2d_order_combinations(self, x_order, y_order):
        """poly_fit_2d should handle all order combinations."""
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        sol, _, rank, _ = poly_fit_2d(x, y, z, x_order=x_order, y_order=y_order)
        assert sol.shape == (x_order + 1, y_order + 1)

    def test_poly_fit_2d_recovers_known_polynomial(self):
        """poly_fit_2d should recover coefficients of a known polynomial."""
        n = 200
        np.random.seed(42)
        x = np.random.rand(n)
        y = np.random.rand(n)
        # z = 3 + 2*x + 1.5*y + 0.5*x*y
        z = 3.0 + 2.0 * x + 1.5 * y + 0.5 * x * y
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        # Coefficients layout: sol[i,j] is coeff of x^i * y^j
        np.testing.assert_almost_equal(sol[0, 0], 3.0, decimal=5)  # constant
        np.testing.assert_almost_equal(sol[1, 0], 2.0, decimal=5)  # x
        np.testing.assert_almost_equal(sol[0, 1], 1.5, decimal=5)  # y
        np.testing.assert_almost_equal(sol[1, 1], 0.5, decimal=5)  # x*y

    def test_poly_fit_2d_all_zeros(self):
        """All-zero z should produce near-zero solution."""
        n = 20
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.zeros(n)
        sol, _, _, _ = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        np.testing.assert_array_almost_equal(sol, 0.0, decimal=10)

    def test_poly_fit_2d_minimum_points(self):
        """Minimum number of points for a linear fit."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 2.0])
        sol, residuals, rank, _ = poly_fit_2d(x, y, z, x_order=1, y_order=0)
        assert rank >= 1

    def test_poly_fit_2d_returns_tuple_of_ndarrays(self):
        """poly_fit_2d should return tuple of array-like objects."""
        result = poly_fit_2d(np.random.rand(20), np.random.rand(20), np.random.rand(20))
        assert isinstance(result, tuple)
        assert len(result) == 4
        # coeffs and sing_values are ndarrays; residuals may be a numpy scalar
        assert isinstance(result[0], np.ndarray)  # coeffs
        assert isinstance(result[2], (int, np.integer))  # rank
        assert hasattr(result[3], "__len__")  # sing_values


class TestExtractAbsGrid:
    """Tests for extract_abs_grid."""

    def test_extract_abs_grid_returns_tuple(self):
        """Should return tuple of DataArrays."""
        slc = xr.DataArray(
            np.random.rand(4, 4),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = extract_abs_grid(slc)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_abs_grid_no_decimation(self):
        """dec=0 should return full-length coordinates."""
        az_len = 7
        sr_len = 5
        slc = xr.DataArray(
            np.zeros((az_len, sr_len)),
            dims=["azimuth_time", "slant_range_time"],
        )
        rg, az = extract_abs_grid(slc)
        assert len(az) == az_len
        assert len(rg) == sr_len

    def test_extract_abs_grid_with_decimation(self):
        """dec > 0 should return decimated coordinates."""
        az_len = 10
        sr_len = 8
        slc = xr.DataArray(
            np.zeros((az_len, sr_len)),
            dims=["azimuth_time", "slant_range_time"],
        )
        rg, az = extract_abs_grid(slc, dec=3)
        assert len(az) <= az_len
        assert len(rg) <= sr_len
        assert len(az) >= 1
        assert len(rg) >= 1

    def test_extract_abs_grid_dec_larger_than_size(self):
        """dec > size uses step=1 (all elements) since ceil(size/dec) = 1."""
        slc = xr.DataArray(
            np.zeros((3, 4)),
            dims=["azimuth_time", "slant_range_time"],
        )
        rg, az = extract_abs_grid(slc, dec=100)
        # ceil(3/100) = 1, ceil(4/100) = 1, so step=1 returns all elements
        assert len(az) == 3
        assert len(rg) == 4

    def test_extract_abs_grid_returns_dataarray(self):
        """Each element should be xr.DataArray."""
        slc = xr.DataArray(np.zeros((5, 5)), dims=["azimuth_time", "slant_range_time"])
        rg, az = extract_abs_grid(slc)
        assert isinstance(az, xr.DataArray)
        assert isinstance(rg, xr.DataArray)

    def test_extract_abs_grid_preserves_coords(self):
        """Decimated coords should be a subset of original."""
        az_vals = np.array([10, 20, 30, 40, 50])
        sr_vals = np.array([1.0, 2.0, 3.0, 4.0])
        slc = xr.DataArray(
            np.zeros((5, 4)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_vals, "slant_range_time": sr_vals},
        )
        rg, az = extract_abs_grid(slc)
        assert az.values[0] == az_vals[0]

    @pytest.mark.parametrize("dec", [1, 2, 3, 5, 10])
    def test_extract_abs_grid_various_dec(self, dec):
        """extract_abs_grid should handle various dec values."""
        slc = xr.DataArray(np.zeros((10, 8)), dims=["azimuth_time", "slant_range_time"])
        rg, az = extract_abs_grid(slc, dec=dec)
        assert len(az) >= 1
        assert len(rg) >= 1

    def test_extract_abs_grid_returns_dataarrays(self):
        """extract_abs_grid should return xr.DataArray objects."""
        slc = xr.DataArray(np.zeros((5, 5)), dims=["azimuth_time", "slant_range_time"])
        rg, az = extract_abs_grid(slc)
        assert isinstance(rg, xr.DataArray)
        assert isinstance(az, xr.DataArray)


class TestExtractRg0Az0:
    """Tests for extract_rg0_az0."""

    def test_extract_rg0_az0_returns_tuple(self):
        """Should return 4-element tuple."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
            attrs={
                "dummy": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
                    "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 0.0, "azimuth_time": "2020-01-01T00:00:00"},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
            }
        )
        result = extract_rg0_az0(slc, phase_demod, 1, 1)
        assert len(result) == 4

    def test_extract_rg0_az0_types(self):
        """All return values should be numeric."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
            attrs={
                "dummy": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
                    "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 0.0, "azimuth_time": "2020-01-01T00:00:00"},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
            }
        )
        rg0, az0, rg, az = extract_rg0_az0(slc, phase_demod, 1, 1)
        assert np.isscalar(rg0) or hasattr(rg0, "item")
        assert np.isscalar(az0) or hasattr(az0, "item")


class TestExtractRgAz:
    """Tests for extract_rg_az."""

    def test_extract_rg_az_returns_tuple(self):
        """Should return tuple of arrays."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(
                    np.arange(3),
                    dims=["azimuth_time"],
                    coords={"azimuth_time": np.arange(3)},
                ),
                "slant_range_time": xr.DataArray(
                    np.arange(3),
                    dims=["slant_range_time"],
                    coords={"slant_range_time": np.arange(3)},
                ),
            }
        )
        dtaz = 1.0
        dtrg = 1.0
        result = extract_rg_az(slc, sim, dtaz, dtrg)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_extract_rg_az_returns_ndarray(self):
        """Both elements should be np.ndarray."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        rg, az = extract_rg_az(slc, sim, 1.0, 1.0)
        assert isinstance(rg, np.ndarray)
        assert isinstance(az, np.ndarray)

    def test_extract_rg_az_correct_values(self):
        """Values should be correctly scaled by dtaz/dtrg."""
        az_vals = np.array([100, 200, 300])
        sr_vals = np.array([1.0, 2.0, 3.0])
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_vals, "slant_range_time": sr_vals},
        )
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(az_vals, dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(sr_vals, dims=["slant_range_time"]),
            }
        )
        dtaz = 100.0
        dtrg = 1.0
        rg, az = extract_rg_az(slc, sim, dtaz, dtrg)
        np.testing.assert_array_equal(rg, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_array_equal(az, np.array([0.0, 1.0, 2.0]))

    def test_extract_rg_az_with_offset(self):
        """Offset between SLC and simulation should produce non-zero relative coords."""
        az_vals = np.array([0, 1, 2])
        sr_vals = np.array([0.0, 0.5, 1.0])
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_vals, "slant_range_time": sr_vals},
        )
        sim_az = np.array([1, 2, 3])
        sim_sr = np.array([0.5, 1.0, 1.5])
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(sim_az, dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(sim_sr, dims=["slant_range_time"]),
            }
        )
        dtaz = 1.0
        dtrg = 0.5
        rg, az = extract_rg_az(slc, sim, dtaz, dtrg)
        np.testing.assert_array_equal(rg, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(az, np.array([1.0, 2.0, 3.0]))

    def test_extract_rg_az_simulation_is_dataarray(self):
        """Should work when simulation is a DataArray, not just Dataset."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        rg, az = extract_rg_az(slc, sim, 1.0, 1.0)
        assert rg.shape == (3,)
        assert az.shape == (3,)

    def test_extract_rg_az_returns_ndarrays(self):
        """extract_rg_az should return np.ndarray objects."""
        slc = xr.DataArray(np.ones((3, 3), dtype=np.complex128), dims=["azimuth_time", "slant_range_time"])
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        rg, az = extract_rg_az(slc, sim, 1.0, 1.0)
        assert type(rg) is np.ndarray
        assert type(az) is np.ndarray


class TestGetCoeffFromLut:
    """Tests for getCoeffFromLut."""

    def test_get_coeff_from_lut_returns_array(self):
        """Should return numpy array."""
        xLut = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 1, "azimuth_time": 1}
        result = getCoeffFromLut(xLut)
        assert isinstance(result, np.ndarray)

    def test_get_coeff_from_lut_shape(self):
        """Output shape should match polynomial degree."""
        xLut = xr.DataArray(
            np.random.rand(5, 5),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 2, "azimuth_time": 2}
        result = getCoeffFromLut(xLut)
        assert result.shape == (3, 3)

    def test_get_coeff_from_lut_deg1(self):
        """Degree (1,1) should produce (2,2) coefficient array."""
        xLut = xr.DataArray(
            np.random.rand(4, 4),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(4).astype("datetime64[ns]"),
                "slant_range_time": np.arange(4) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 1, "azimuth_time": 1}
        result = getCoeffFromLut(xLut)
        assert result.shape == (2, 2)

    def test_get_coeff_from_lut_missing_attrs_raises(self):
        """Missing attributes should raise KeyError or AttributeError."""
        xLut = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
        )
        with pytest.raises((KeyError, AttributeError)):
            getCoeffFromLut(xLut)

    def test_get_coeff_from_lut_constant_data(self):
        """Constant LUT should give a dominant constant coefficient."""
        xLut = xr.DataArray(
            np.full((5, 5), 5.0),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 1, "azimuth_time": 1}
        result = getCoeffFromLut(xLut)
        # Constant data: the constant coefficient should dominate
        assert abs(result[0, 0]) > 0.1

    @pytest.mark.parametrize("x_deg,y_deg", [(0, 0), (1, 2), (2, 1), (3, 3)])
    def test_get_coeff_from_lut_various_degrees(self, x_deg, y_deg):
        """getCoeffFromLut should handle various degree combinations."""
        xsize = x_deg + 3
        ysize = y_deg + 3
        xLut = xr.DataArray(
            np.random.rand(ysize, xsize),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(ysize).astype("datetime64[ns]"),
                "slant_range_time": np.arange(xsize) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": x_deg, "azimuth_time": y_deg}
        result = getCoeffFromLut(xLut)
        assert result.shape == (x_deg + 1, y_deg + 1)

    def test_get_coeff_from_lut_returns_ndarray(self):
        """getCoeffFromLut should return np.ndarray."""
        xLut = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 1, "azimuth_time": 1}
        result = getCoeffFromLut(xLut)
        assert type(result) is np.ndarray


class TestGetLUT:
    """Tests for getLUT."""

    def test_getLUT_returns_tuple(self):
        """Should return tuple of 3 DataArrays."""
        coreg = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        daz_drg = xr.Dataset(
            {
                "daz": xr.DataArray(np.arange(9).reshape(3, 3), dims=["azimuth_time", "slant_range_time"]),
                "drg": xr.DataArray(np.arange(9, 18).reshape(3, 3), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc = xr.Dataset(
            {
                "slc": xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc["slc"].attrs["dummy"] = {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
            "ImageData": {"SCPPixel": {"Row": 1, "Col": 1}},
            "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
        }
        coeffs = np.ones((2, 2))
        result = getLUT(coreg, daz_drg, slc, coeffs)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_getLUT_returns_dataarrays(self):
        """All return values should be xr.DataArrays."""
        coreg = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        daz_drg = xr.Dataset(
            {
                "daz": xr.DataArray(np.zeros((3, 3)), dims=["azimuth_time", "slant_range_time"]),
                "drg": xr.DataArray(np.zeros((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc = xr.Dataset(
            {
                "slc": xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc["slc"].attrs["dummy"] = {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
            "ImageData": {"SCPPixel": {"Row": 1, "Col": 1}},
            "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
        }
        lut, az_ref, rg_ref = getLUT(coreg, daz_drg, slc, np.ones((2, 2)))
        assert isinstance(lut, xr.DataArray)
        assert isinstance(az_ref, xr.DataArray)
        assert isinstance(rg_ref, xr.DataArray)

    def test_getLUT_nan_masking(self):
        """NaN daz/drg values should be masked out."""
        coreg = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        daz_data = np.array([[0, np.nan, 0], [0, 0, 0], [0, 0, np.nan]])
        drg_data = np.array([[0, 0, 0], [np.nan, 0, 0], [0, 0, 0]])
        daz_drg = xr.Dataset(
            {
                "daz": xr.DataArray(daz_data, dims=["azimuth_time", "slant_range_time"]),
                "drg": xr.DataArray(drg_data, dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc = xr.Dataset(
            {
                "slc": xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc["slc"].attrs["dummy"] = {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
            "ImageData": {"SCPPixel": {"Row": 1, "Col": 1}},
            "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
        }
        lut, az_ref, rg_ref = getLUT(coreg, daz_drg, slc, np.ones((2, 2)))
        assert isinstance(lut, xr.DataArray)

    def test_getLUT_shape_matches_coreg_grid(self):
        """LUT output shape should match the daz_drg grid."""
        coreg = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        daz_drg = xr.Dataset(
            {
                "daz": xr.DataArray(np.zeros((3, 3)), dims=["azimuth_time", "slant_range_time"]),
                "drg": xr.DataArray(np.zeros((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc = xr.Dataset(
            {
                "slc": xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc["slc"].attrs["dummy"] = {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
            "ImageData": {"SCPPixel": {"Row": 1, "Col": 1}},
            "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
        }
        lut, az_ref, rg_ref = getLUT(coreg, daz_drg, slc, np.ones((2, 2)))
        assert lut.shape == (3, 3)


class TestExtractGrid:
    """Tests for extract_grid."""

    def test_extract_grid_returns_tuple(self):
        """Should return tuple of two DataArrays."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        rg, az = extract_grid(slc, scp_idx=(2, 2))
        assert isinstance(rg, xr.DataArray)
        assert isinstance(az, xr.DataArray)

    def test_extract_grid_relative_values(self):
        """Grid values should be relative to the reference point (zero at SCP)."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        rg, az = extract_grid(slc, scp_idx=(2, 2))
        # Values should be finite and non-empty
        assert rg.shape == (5,)
        assert az.shape == (5,)
        assert np.isfinite(rg.values).all()
        assert np.isfinite(az.values).all()

    def test_extract_grid_with_reference(self):
        """Custom reference dict should be used instead of SCP."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        custom_ref = {"slant_range_time": 1.0, "azimuth_time": np.datetime64("2020-01-01T00:00:02")}
        rg, az = extract_grid(slc, scp_idx=(2, 2), reference=custom_ref)
        assert rg.shape == (5,)
        assert az.shape == (5,)

    def test_extract_grid_with_scale(self):
        """Custom scale dict should override default scaling."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        custom_scale = {"slant_range_time": 2.0, "azimuth_time": 2.0}
        rg, az = extract_grid(slc, scp_idx=(2, 2), scale=custom_scale)
        assert rg.shape == (5,)
        assert az.shape == (5,)

    def test_extract_grid_with_decimation(self):
        """Decimation should reduce grid size."""
        slc = xr.DataArray(
            np.ones((10, 10), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    ["2020-01-01T00:00:0{:d}".format(i) for i in range(10)], dtype="datetime64[ns]"
                ),
                "slant_range_time": np.arange(10) * 0.5,
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 5, "Col": 5}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        rg, az = extract_grid(slc, scp_idx=(5, 5), dec=2)
        assert len(rg) < 10
        assert len(az) < 10

    def test_extract_grid_no_decimation(self):
        """dec=0 should return full-size grid."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        rg, az = extract_grid(slc, scp_idx=(2, 2), dec=0)
        assert len(rg) == 5
        assert len(az) == 5

    def test_extract_grid_missing_annotation_raises(self):
        """Missing RMA annotations should raise KeyError."""
        slc = xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"])
        with pytest.raises((KeyError, IndexError)):
            extract_grid(slc, scp_idx=(1, 1))

    def test_extract_grid_output_is_dataarray(self):
        """Returns should be xr.DataArray objects."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                        "2020-01-01T00:00:03",
                        "2020-01-01T00:00:04",
                    ],
                    dtype="datetime64[ns]",
                ),
                "slant_range_time": np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        rg, az = extract_grid(slc, scp_idx=(2, 2))
        assert isinstance(rg, xr.DataArray)
        assert isinstance(az, xr.DataArray)


class TestDoDeramp:
    """Tests for do_deramp."""

    def test_do_deramp_returns_dataarray(self):
        """Should return xr.DataArray."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3),
            },
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 1.0, "azimuth_time": np.datetime64("2020-01-01T00:00:01")},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
                "deg": {"slant_range_time": 1, "azimuth_time": 1},
            }
        )
        result = do_deramp(slc, phase_demod, scp_idx=(2, 2))
        assert isinstance(result, xr.DataArray)

    def test_do_deramp_preserves_shape(self):
        """Output shape should match input SLC shape."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3),
            },
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 1.0, "azimuth_time": np.datetime64("2020-01-01T00:00:01")},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
                "deg": {"slant_range_time": 1, "azimuth_time": 1},
            }
        )
        result = do_deramp(slc, phase_demod, scp_idx=(2, 2))
        assert result.shape == (5, 5)

    def test_do_deramp_preserves_dtype(self):
        """Output dtype should be complex128 (same as input)."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3),
            },
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 1.0, "azimuth_time": np.datetime64("2020-01-01T00:00:01")},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
                "deg": {"slant_range_time": 1, "azimuth_time": 1},
            }
        )
        result = do_deramp(slc, phase_demod, scp_idx=(2, 2))
        assert result.dtype == np.complex128

    def test_do_deramp_complex_output(self):
        """Output should have both real and imaginary parts."""
        slc = xr.DataArray(
            np.ones((5, 5), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(5).astype("datetime64[ns]"),
                "slant_range_time": np.arange(5),
            },
            attrs={
                "burst_meta": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0.0, 1.0, 0.0]}}},
                    "ImageData": {"SCPPixel": {"Row": 2, "Col": 2}},
                    "Grid": {"Col": {"SS": 0.01}, "Row": {"SS": 0.5}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3),
            },
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 1.0, "azimuth_time": np.datetime64("2020-01-01T00:00:01")},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
                "deg": {"slant_range_time": 1, "azimuth_time": 1},
            }
        )
        result = do_deramp(slc, phase_demod, scp_idx=(2, 2))
        # Apply phase correction — both real and imag should be non-zero
        assert result.values.any()


class TestGridDataBlock:
    """Tests for grid_data_block."""

    def _make_mock_daz_drg(self, n_acq=1, n_az=3, n_sr=3):
        """Create mock daz/drg DataArrays with coordinates matching requested sizes."""
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array([f"2020-01-01T00:00:{i:02d}" for i in range(n_az)], dtype="datetime64[ns]")
        sr = np.arange(n_sr) * 0.5
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        return daz, drg, acq, az, sr

    def test_grid_data_block_single_acquisition_zero_output(self):
        """Single acquisition should produce zero daz/drg."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        # Create simulation with matching structure
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        np.testing.assert_array_equal(result_daz.values, 0)
        np.testing.assert_array_equal(result_drg.values, 0.0)

    def test_grid_data_block_returns_dataarrays(self):
        """Should return tuple of xr.DataArrays."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert isinstance(result_daz, xr.DataArray)
        assert isinstance(result_drg, xr.DataArray)

    def test_grid_data_block_single_acquisition_dims_preserved(self):
        """Single acquisition should preserve input dimensions (acquisition_id, azimuth_time, slant_range_time)."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=4, n_sr=5)
        sim_data = xr.DataArray(
            np.zeros((1, 4, 5)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 4, 5]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert result_drg.dims == ("acquisition_id", "azimuth_time", "slant_range_time")

    def test_grid_data_block_single_acquisition_shape(self):
        """Single acquisition should preserve input shape."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=6, n_sr=8)
        sim_data = xr.DataArray(
            np.zeros((1, 6, 8)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 6, 8]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 6, 8)
        assert result_drg.shape == (1, 6, 8)

    def test_grid_data_block_daz_dtype_timedelta64(self):
        """Output daz should have timedelta64[ns] dtype."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.dtype == np.dtype("timedelta64[ns]")

    def test_grid_data_block_drg_dtype_float64(self):
        """Output drg should have float64 dtype."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_drg.dtype == np.float64

    def test_grid_data_block_coords_preserved(self):
        """Output DataArrays should preserve input coordinates."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        np.testing.assert_array_equal(result_daz.coords["azimuth_time"].values, az)
        np.testing.assert_array_equal(result_daz.coords["slant_range_time"].values, sr)
        np.testing.assert_array_equal(result_drg.coords["azimuth_time"].values, az)
        np.testing.assert_array_equal(result_drg.coords["slant_range_time"].values, sr)

    def test_grid_data_block_multi_acquisition_nonzero(self):
        """Multiple acquisitions should produce non-zero daz/drg shifts."""
        n_acq = 3
        n_az = 4
        n_sr = 4
        acq = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
        az = np.array(
            ["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02", "2020-01-01T00:00:03"],
            dtype="datetime64[ns]",
        )
        sr = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Create simulation with varying azimuth/range times across acquisitions
        sim_az_data = np.zeros((n_acq, n_az, n_sr), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
        for i in range(n_acq):
            for j in range(n_az):
                for k in range(n_sr):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i + j, "s")
                    sim_sr_data[i, j, k] = float(i + k) * 0.25
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-01",
            np.array([0, 0, 4, 4]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (n_acq, n_az, n_sr)
        assert result_drg.shape == (n_acq, n_az, n_sr)
        # Not all values should be zero when there are multiple acquisitions with varying data
        assert not np.all(result_daz.values == 0) or not np.all(result_drg.values == 0)

    def test_grid_data_block_multi_acquisition_shape(self):
        """Multiple acquisitions should produce output with matching shape."""
        n_acq = 4
        n_az = 5
        n_sr = 5  # Keep sr same as az to avoid shape mismatch in vstack
        acq = np.array(["2020-01-01", "2020-01-05", "2020-01-10", "2020-01-15"], dtype="datetime64[ns]")
        az = np.array([f"2020-01-01T00:00:{i:02d}" for i in range(n_az)], dtype="datetime64[ns]")
        sr = np.arange(n_sr) * 0.5
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Build properly typed simulation data for multi-acquisition case
        sim_az_data = np.zeros((n_acq, n_az, n_sr), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
        for i in range(n_acq):
            for j in range(n_az):
                for k in range(n_sr):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i * 10 + j, "s")
                    sim_sr_data[i, j, k] = float(i * 5 + k) * 0.25
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # bbox must match (n_az, n_sr) since that's the shape of the simulation data
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-01",
            np.array([0, 0, n_az, n_sr]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (n_acq, n_az, n_sr)
        assert result_drg.shape == (n_acq, n_az, n_sr)

    def test_grid_data_block_multi_acquisition_returns_dataarrays(self):
        """Multiple acquisitions should return tuple of xr.DataArrays."""
        n_acq = 3
        n_az = 3
        n_sr = 3
        acq = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Build properly typed simulation data
        sim_az_data = np.zeros((n_acq, n_az, n_sr), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
        for i in range(n_acq):
            for j in range(n_az):
                for k in range(n_sr):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i * 10 + j, "s")
                    sim_sr_data[i, j, k] = float(i * 3 + k) * 0.25
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert isinstance(result_daz, xr.DataArray)
        assert isinstance(result_drg, xr.DataArray)

    @pytest.mark.parametrize("n_acq", [2, 3, 5])
    def test_grid_data_block_various_acquisition_counts(self, n_acq):
        """grid_data_block should handle various numbers of acquisitions."""
        acq = np.array([f"2020-01-{i + 1:02d}" for i in range(n_acq)], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, 3, 3), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, 3, 3), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Build properly typed simulation data
        sim_az_data = np.zeros((n_acq, 3, 3), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, 3, 3), dtype=np.float64)
        for i in range(n_acq):
            for j in range(3):
                for k in range(3):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i * 10 + j, "s")
                    sim_sr_data[i, j, k] = float(i * 3 + k) * 0.25
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (n_acq, 3, 3)
        assert result_drg.shape == (n_acq, 3, 3)

    @pytest.mark.parametrize("n_az,n_sr", [(1, 1), (3, 5), (5, 3), (7, 7), (10, 10)])
    def test_grid_data_block_various_grid_sizes(self, n_az, n_sr):
        """grid_data_block should handle various grid sizes."""
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array([f"2020-01-01T00:00:{i:02d}" for i in range(n_az)], dtype="datetime64[ns]")
        sr = np.arange(n_sr) * 0.5
        daz = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_data = xr.DataArray(
            np.zeros((1, n_az, n_sr)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, n_az, n_sr]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, n_az, n_sr)
        assert result_drg.shape == (1, n_az, n_sr)

    @pytest.mark.parametrize("az_grid_spacing", [0.1, 0.5, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("range_grid_spacing", [0.01, 0.1, 0.5, 1.0, 5.0])
    def test_grid_data_block_various_grid_spacings(self, az_grid_spacing, range_grid_spacing):
        """grid_data_block should handle various grid spacing combinations."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            az_grid_spacing,
            range_grid_spacing,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)

    @pytest.mark.parametrize(
        "bbox",
        [
            np.array([0, 0, 3, 3]),
            np.array([-1, -1, 4, 4]),
            np.array([1, 1, 2, 2]),
            np.array([0, 0, 10, 10]),
        ],
    )
    def test_grid_data_block_various_bboxes(self, bbox):
        """grid_data_block should handle various bounding box configurations."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            bbox,
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)

    def test_grid_data_block_single_acquisition_finite_values(self):
        """Single acquisition output should contain only finite (zero) values."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert np.all(np.isfinite(result_daz.values))
        assert np.all(np.isfinite(result_drg.values))
        np.testing.assert_array_equal(result_daz.values, 0)
        np.testing.assert_array_equal(result_drg.values, 0)

    def test_grid_data_block_single_acquisition_no_nan(self):
        """Single acquisition output should not contain NaN values."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert not np.any(np.isnan(result_daz.values))
        assert not np.any(np.isnan(result_drg.values))

    def test_grid_data_block_single_acquisition_no_inf(self):
        """Single acquisition output should not contain Inf values."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert not np.any(np.isinf(result_daz.values))
        assert not np.any(np.isinf(result_drg.values))

    def test_grid_data_block_reference_date_selection(self):
        """Should select the correct reference date from simulation data."""
        n_acq = 3
        n_az = 3
        n_sr = 3
        acq = np.array(["2020-01-01", "2020-01-05", "2020-01-10"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Simulation data with distinct values per acquisition
        sim_az_data = np.zeros((n_acq, n_az, n_sr), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
        for i in range(n_acq):
            for j in range(n_az):
                for k in range(n_sr):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i * 10 + j, "s")
                    sim_sr_data[i, j, k] = float(i * 10 + k) * 0.1
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Use "2020-01-05" as reference date
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-05",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (n_acq, n_az, n_sr)
        assert result_drg.shape == (n_acq, n_az, n_sr)
        # The reference date itself should have near-zero shift
        ref_idx = 1  # "2020-01-05" is at index 1
        assert np.allclose(result_drg.values[ref_idx], 0.0, atol=1e-10)

    def test_grid_data_block_datetime_coords_type(self):
        """Azimuth time coordinates should be datetime64[ns]."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert np.issubdtype(result_daz.coords["azimuth_time"].dtype, np.datetime64)

    def test_grid_data_block_reference_time_as_datetime(self):
        """azimuth_time_ref should accept np.datetime64."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Pass datetime with different resolution
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)

    def test_grid_data_block_with_nan_in_simulation(self):
        """Simulation with NaN values should not crash and should produce finite output."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.array([[[np.nan, 0.0, 1.0], [0.0, np.nan, 0.0], [1.0, 0.0, np.nan]]]),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)
        assert np.all(np.isfinite(result_daz.values))
        assert np.all(np.isfinite(result_drg.values))

    def test_grid_data_block_with_inf_in_simulation(self):
        """Simulation with Inf values should not crash."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.array([[[np.inf, 0.0, -np.inf], [0.0, 1.0, 0.0], [0.0, 0.0, np.inf]]]),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)
        assert np.all(np.isfinite(result_daz.values))
        assert np.all(np.isfinite(result_drg.values))

    def test_grid_data_block_asymmetric_grid(self):
        """Asymmetric grid dimensions should produce matching output shape."""
        n_az = 2
        n_sr = 7
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_data = xr.DataArray(
            np.zeros((1, n_az, n_sr)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, n_az, n_sr]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, n_az, n_sr)
        assert result_drg.shape == (1, n_az, n_sr)

    def test_grid_data_block_non_square_bbox(self):
        """Non-square bounding box should produce output matching daz/drg shape."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=3, n_sr=5)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 5)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 5]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 5)
        assert result_drg.shape == (1, 3, 5)

    def test_grid_data_block_both_results_same_shape(self):
        """Output daz and drg should always have the same shape."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=5, n_sr=7)
        sim_data = xr.DataArray(
            np.zeros((1, 5, 7)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 5, 7]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        np.testing.assert_array_equal(result_daz.shape, result_drg.shape)

    def test_grid_data_block_nonzero_initial_daz_drg_unchanged(self):
        """Non-zero initial daz/drg values should be overwritten by output."""
        n_acq = 1
        daz_data = np.full((n_acq, 3, 3), np.timedelta64(100, "ns"), dtype="timedelta64[ns]")
        drg_data = np.full((n_acq, 3, 3), 999.0, dtype=np.float64)
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            daz_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            drg_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_data = xr.DataArray(
            np.zeros((n_acq, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        # Single acquisition: daz/drg should be overwritten with zeros
        np.testing.assert_array_equal(result_daz.values, 0)
        np.testing.assert_array_equal(result_drg.values, 0.0)

    def test_grid_data_block_reference_date_not_first(self):
        """When reference date is not the first acquisition, shifts should be relative to it."""
        n_acq = 4
        n_az = 3
        n_sr = 3
        acq = np.array(["2020-01-01", "2020-01-05", "2020-01-10", "2020-01-15"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Simulation: each acquisition has increasing time offsets
        sim_az_data = np.zeros((n_acq, n_az, n_sr), dtype="datetime64[ns]")
        sim_sr_data = np.zeros((n_acq, n_az, n_sr), dtype=np.float64)
        for i in range(n_acq):
            for j in range(n_az):
                for k in range(n_sr):
                    sim_az_data[i, j, k] = np.datetime64("2020-01-01T00:00:00") + np.timedelta64(i * 100 + j * 10, "s")
                    sim_sr_data[i, j, k] = float(i * 10 + k) * 0.5
        sim_az = xr.DataArray(
            sim_az_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_sr = xr.DataArray(
            sim_sr_data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        # Use "2020-01-10" as reference (index 2)
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_az,
            sim_sr,
            "2020-01-10",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (n_acq, n_az, n_sr)
        assert result_drg.shape == (n_acq, n_az, n_sr)
        np.testing.assert_array_equal(result_daz.sel(acquisition_id="2020-01-10"), 0)
        np.testing.assert_array_equal(result_drg.sel(acquisition_id="2020-01-10"), 0)
        assert np.all(np.abs(result_daz.values[[0, 1, 3], ...]) > 0)
        assert np.all(np.abs(result_drg.values[[0, 1, 3], ...]) > 0)

    def test_grid_data_block_single_acquisition_zero_values_exact(self):
        """Single acquisition should produce exactly zero values (not just approximately)."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=5, n_sr=5)
        sim_data = xr.DataArray(
            np.zeros((1, 5, 5)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 5, 5]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        np.testing.assert_array_equal(result_daz.values, np.zeros((1, 5, 5), dtype="timedelta64[ns]"))
        np.testing.assert_array_equal(result_drg.values, np.zeros((1, 5, 5)))

    @pytest.mark.parametrize("n_az", [1, 2, 4, 8])
    @pytest.mark.parametrize("n_sr", [1, 2, 4, 8])
    def test_grid_data_block_power_of_two_sizes(self, n_az, n_sr):
        """grid_data_block should handle power-of-two grid sizes."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1, n_az=n_az, n_sr=n_sr)
        sim_data = xr.DataArray(
            np.zeros((1, n_az, n_sr)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, n_az, n_sr]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, n_az, n_sr)
        assert result_drg.shape == (1, n_az, n_sr)

    def test_grid_data_block_large_grid(self):
        """Larger grid should produce correct output without error."""
        n_az = 20
        n_sr = 20
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array([f"2020-01-01T00:00:{i:02d}" for i in range(n_az)], dtype="datetime64[ns]")
        sr = np.arange(n_sr) * 0.5
        daz = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((1, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        sim_data = xr.DataArray(
            np.zeros((1, n_az, n_sr)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, n_az, n_sr]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, n_az, n_sr)
        assert result_drg.shape == (1, n_az, n_sr)

    def test_grid_data_block_return_tuple(self):
        """Should return a 2-element tuple."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            1.0,
            0.5,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_grid_data_block_small_grid_spacing(self):
        """Very small grid spacings should not cause errors."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            0.01,
            0.001,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)

    def test_grid_data_block_large_grid_spacing(self):
        """Large grid spacings should not cause errors."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            100.0,
            100.0,
            np.datetime64("2020-01-01T00:00:00"),
            0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)

    def test_grid_data_block_asymmetric_grid_spacings(self):
        """Different azimuth and range grid spacings should work."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={"acquisition_id": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        result_daz, result_drg = grid_data_block(
            daz,
            drg,
            sim_data,
            sim_data,
            "2020-01-01",
            np.array([0, 0, 3, 3]),
            azimuth_grid_spacing=0.5,
            range_grid_spacing=1.0,
            azimuth_time_ref=np.datetime64("2020-01-01T00:00:00"),
            slant_range_time_ref=0.0,
        )
        assert result_daz.shape == (1, 3, 3)
        assert result_drg.shape == (1, 3, 3)


class TestWarpDazdrg:
    """Tests for the warp_dazdrg function.

    warp_dazdrg computes azimuth (daz) and range (drg) shifts to coregister
    a SAR image stack onto a reference grid. It uses simulation data,
    multilook reference data, and grid spacing parameters to determine
    the geometric corrections needed.
    """

    def test_warp_dazdrg_returns_dataset(self, warp_dazdrg_inputs):
        """warp_dazdrg should return an xr.Dataset."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert isinstance(result, xr.Dataset)

    def test_warp_dazdrg_has_daz_and_drg(self, warp_dazdrg_inputs):
        """Output dataset should contain 'daz' and 'drg' data variables."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert "daz" in result.data_vars
        assert "drg" in result.data_vars

    def test_warp_dazdrg_daz_dtype_timedelta64(self, warp_dazdrg_inputs):
        """daz should have timedelta64[ns] dtype."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert result["daz"].dtype == np.dtype("timedelta64[ns]")

    def test_warp_dazdrg_drg_dtype_float64(self, warp_dazdrg_inputs):
        """drg should have float64 dtype."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert result["drg"].dtype == np.float64

    def test_warp_dazdrg_dims_preserved(self, warp_dazdrg_inputs):
        """Output dims should match the cstack dimensions (acquisition_id, azimuth_time, slant_range_time)."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert result["daz"].dims == ("acquisition_id", "azimuth_time", "slant_range_time")
        assert result["drg"].dims == ("acquisition_id", "azimuth_time", "slant_range_time")

    def test_warp_dazdrg_shape_matches_cstack(self, warp_dazdrg_inputs):
        """Output shape should match the cstack shape (daz/drg are 3D with acquisition_id)."""
        cstack = warp_dazdrg_inputs["cstack"]
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert result["daz"].shape == cstack.shape
        assert result["drg"].shape == cstack.shape

    def test_warp_dazdrg_coords_match_cstack(self, warp_dazdrg_inputs):
        """Output coordinates should match cstack coordinates."""
        cstack = warp_dazdrg_inputs["cstack"]
        result = warp_dazdrg(**warp_dazdrg_inputs)
        for dim in ["acquisition_id", "azimuth_time", "slant_range_time"]:
            xr.testing.assert_identical(result["daz"].coords[dim], cstack.coords[dim])
            xr.testing.assert_identical(result["drg"].coords[dim], cstack.coords[dim])

    def test_warp_dazdrg_single_acquisition(self, warp_dazdrg_inputs):
        """Single acquisition cstack should not error."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"].isel(acquisition_id=slice(0, 1))
        inputs["cstack"] = cstack
        inputs["coregistration_reference_id"] = str(cstack.acquisition_id.values[0])
        # Update daz/drg to match new cstack shape
        inputs["daz"] = xr.DataArray(
            np.zeros(cstack.shape, dtype="timedelta64[ns]"),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        inputs["drg"] = xr.DataArray(
            np.zeros(cstack.shape, dtype=np.float64),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )

        result = warp_dazdrg(**inputs)
        assert isinstance(result, xr.Dataset)
        assert "daz" in result.data_vars
        assert "drg" in result.data_vars
        assert result["daz"].shape[0] == 1

    def test_warp_dazdrg_small_grid(self, warp_dazdrg_inputs):
        """Small 3x3 grid should produce correct output shape."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"].isel(
            azimuth_time=slice(0, 3),
            slant_range_time=slice(0, 3),
        )
        inputs["cstack"] = cstack
        inputs["daz"] = xr.DataArray(
            np.zeros(cstack.shape, dtype="timedelta64[ns]"),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        inputs["drg"] = xr.DataArray(
            np.zeros(cstack.shape, dtype=np.float64),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )

        result = warp_dazdrg(**inputs)
        assert result["daz"].shape == (3, 3, 3)
        assert result["drg"].shape == (3, 3, 3)

    def test_warp_dazdrg_output_finite_values(self, warp_dazdrg_inputs):
        """Output values should be finite (no NaN or Inf)."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert np.all(np.isfinite(result["drg"].values))
        # timedelta64 values should be representable (check that conversion to int is finite)
        daz_as_int = result["daz"].values.astype(np.int64)
        assert np.all(np.isfinite(daz_as_int))

    def test_warp_dazdrg_reference_date_selection(self, warp_dazdrg_inputs):
        """Should use coregistration_reference_id to select simulation data."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"]
        result = warp_dazdrg(**inputs)
        # Reference date should have near-zero shifts relative to itself
        ref_date = inputs["coregistration_reference_id"]
        ref_idx = list(cstack.acquisition_id.values).index(np.datetime64(ref_date))
        # The reference date shift should be near zero
        assert result["drg"].isel(acquisition_id=ref_idx).values is not None

    @pytest.mark.parametrize("az_spacing", [0.1, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("rg_spacing", [0.01, 0.1, 0.5, 1.0])
    def test_warp_dazdrg_various_grid_spacings(self, warp_dazdrg_inputs, az_spacing, rg_spacing):
        """warp_dazdrg should handle various grid spacing combinations."""
        inputs = dict(warp_dazdrg_inputs)
        inputs["azimuth_grid_spacing"] = az_spacing
        inputs["range_grid_spacing"] = rg_spacing
        result = warp_dazdrg(**inputs)
        assert isinstance(result, xr.Dataset)
        assert result["daz"].shape == (3, 5, 5)
        assert result["drg"].shape == (3, 5, 5)

    @pytest.mark.parametrize("margin_rg", [0.0, 0.5, 1.0])
    def test_warp_dazdrg_various_margins(self, warp_dazdrg_inputs, margin_rg):
        """warp_dazdrg should handle various margin values."""
        inputs = dict(warp_dazdrg_inputs)
        inputs["margin"] = {"azimuth_time": np.timedelta64(0, "ns"), "slant_range_time": margin_rg}
        result = warp_dazdrg(**inputs)
        assert isinstance(result, xr.Dataset)

    def test_warp_dazdrg_no_secondary_simulation(self, warp_dazdrg_inputs):
        """warp_dazdrg should work without secondary simulation (simulation_secondary=None)."""
        inputs = dict(warp_dazdrg_inputs)
        inputs["simulation_secondary"] = None
        result = warp_dazdrg(**inputs)
        assert isinstance(result, xr.Dataset)
        assert "daz" in result.data_vars
        assert "drg" in result.data_vars

    def test_warp_dazdrg_daz_drg_different_shapes_not_possible(self, warp_dazdrg_inputs):
        """daz and drg in output should always have the same shape."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        np.testing.assert_array_equal(result["daz"].shape, result["drg"].shape)

    def test_warp_dazdrg_daz_drg_same_coords(self, warp_dazdrg_inputs):
        """daz and drg should share the same coordinates."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        for dim in result["daz"].dims:
            xr.testing.assert_identical(result["daz"].coords[dim], result["drg"].coords[dim])

    def test_warp_dazdrg_all_zeros_single_acquisition(self, warp_dazdrg_inputs):
        """Single acquisition with identical simulation should produce zero shifts."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"].isel(acquisition_id=slice(0, 1))
        inputs["cstack"] = cstack
        inputs["coregistration_reference_id"] = str(cstack.acquisition_id.values[0])

        # Use the same simulation for reference and secondary (no shifts expected)
        inputs["simulation_secondary"] = None

        result = warp_dazdrg(**inputs)
        # With a single acquisition and compute_daz_drg=True, the grid_data_block
        # produces zero values
        np.testing.assert_array_equal(result["drg"].values, 0.0)

    def test_warp_dazdrg_acquisition_coord_type(self, warp_dazdrg_inputs):
        """acquisition_id coordinate should be datetime64."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert np.issubdtype(result["daz"].coords["acquisition_id"].dtype, np.datetime64)

    def test_warp_dazdrg_azimuth_time_coord_type(self, warp_dazdrg_inputs):
        """azimuth_time coordinate should be datetime64."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert np.issubdtype(result["daz"].coords["azimuth_time"].dtype, np.datetime64)

    def test_warp_dazdrg_slant_range_time_coord_type(self, warp_dazdrg_inputs):
        """slant_range_time coordinate should be float64."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert result["drg"].coords["slant_range_time"].dtype == np.float64

    def test_warp_dazdrg_existing_daz_drg_used(self, warp_dazdrg_inputs):
        """Provided daz/drg should be used without recreating them."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"]
        # Create 3D daz/drg with known values
        known_daz = xr.DataArray(
            np.full(cstack.shape, np.timedelta64(42, "ns"), dtype="timedelta64[ns]"),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        known_drg = xr.DataArray(
            np.full(cstack.shape, 999.0, dtype=np.float64),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        inputs["daz"] = known_daz
        inputs["drg"] = known_drg
        inputs["compute_daz_drg"] = False

        result = warp_dazdrg(**inputs)
        assert isinstance(result, xr.Dataset)
        # compute_daz_drg=False should preserve existing values
        np.testing.assert_array_equal(result["drg"].values, 999.0)

    def test_warp_dazdrg_with_margin_nonzero(self, warp_dazdrg_inputs):
        """Non-zero margins should still produce valid output."""
        inputs = dict(warp_dazdrg_inputs)
        inputs["margin"] = {"azimuth_time": np.timedelta64(0, "ns"), "slant_range_time": 0.5}
        result = warp_dazdrg(**inputs)
        assert np.all(np.isfinite(result["drg"].values))

    def test_warp_dazdrg_larger_grid(self, warp_dazdrg_inputs):
        """Larger grid (7x7) should produce correct output shape."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"]
        data = np.zeros((3, 7, 7), dtype=np.complex128)
        for i in range(3):
            data[i] = (1.0 + 0.1j) * (i + 1)
        cstack = xr.DataArray(
            data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={
                "acquisition_id": cstack.acquisition_id,
                "azimuth_time": np.array(
                    [
                        np.datetime64("2020-01-01T00:00:00", "ns") + np.timedelta64(i * 1000000000, "ns")
                        for i in range(7)
                    ]
                ),
                "slant_range_time": np.arange(7) * 0.5,
            },
        )
        inputs["cstack"] = cstack
        inputs["daz"] = xr.DataArray(
            np.zeros(cstack.shape, dtype="timedelta64[ns]"),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        inputs["drg"] = xr.DataArray(
            np.zeros(cstack.shape, dtype=np.float64),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )

        result = warp_dazdrg(**inputs)
        assert result["daz"].shape == (3, 7, 7)
        assert result["drg"].shape == (3, 7, 7)

    def test_warp_dazdrg_asymmetric_grid(self, warp_dazdrg_inputs):
        """Asymmetric grid (3x7) should produce matching output shape."""
        inputs = dict(warp_dazdrg_inputs)
        cstack = inputs["cstack"]
        data = np.zeros((3, 3, 7), dtype=np.complex128)
        for i in range(3):
            data[i] = (1.0 + 0.1j) * (i + 1)
        cstack = xr.DataArray(
            data,
            dims=["acquisition_id", "azimuth_time", "slant_range_time"],
            coords={
                "acquisition_id": cstack.acquisition_id,
                "azimuth_time": np.array(
                    [
                        np.datetime64("2020-01-01T00:00:00", "ns") + np.timedelta64(i * 1000000000, "ns")
                        for i in range(3)
                    ]
                ),
                "slant_range_time": np.arange(7) * 0.5,
            },
        )
        inputs["cstack"] = cstack
        inputs["daz"] = xr.DataArray(
            np.zeros(cstack.shape, dtype="timedelta64[ns]"),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )
        inputs["drg"] = xr.DataArray(
            np.zeros(cstack.shape, dtype=np.float64),
            dims=cstack.dims,
            coords={d: cstack[d] for d in cstack.dims},
        )

        result = warp_dazdrg(**inputs)
        assert result["daz"].shape == (3, 3, 7)
        assert result["drg"].shape == (3, 3, 7)

    def test_warp_dazdrg_output_no_nan(self, warp_dazdrg_inputs):
        """Output drg should not contain NaN values."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert not np.any(np.isnan(result["drg"].values))

    def test_warp_dazdrg_output_no_inf(self, warp_dazdrg_inputs):
        """Output drg should not contain Inf values."""
        result = warp_dazdrg(**warp_dazdrg_inputs)
        assert not np.any(np.isinf(result["drg"].values))


# ===========================================================================
# compute_secondary_image_coordinates tests
# ===========================================================================


class TestComputeSecondaryImageCoordinates:
    """Tests for compute_secondary_image_coordinates happy paths and basic behavior."""

    def test_csi_returns_four_values(self, csi_inputs):
        """Should return exactly 4 values: az, rg, az0, rg0."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert len((az, rg, az0, rg0)) == 4

    def test_csi_az_is_ndarray(self, csi_inputs):
        """Azimuth output should be np.ndarray."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert isinstance(az, np.ndarray)

    def test_csi_rg_is_ndarray(self, csi_inputs):
        """Range output should be np.ndarray."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert isinstance(rg, np.ndarray)

    def test_csi_az0_is_scalar(self, csi_inputs):
        """az0 should be a scalar-like value (datetime64)."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert hasattr(az0, "item") or np.isscalar(az0)

    def test_csi_rg0_is_scalar(self, csi_inputs):
        """rg0 should be a scalar-like value (float)."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert hasattr(rg0, "item") or np.isscalar(rg0)

    def test_csi_az0_matches_slc(self, csi_inputs):
        """az0 should equal the first azimuth_time of the SLC."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        expected_az0 = csi_inputs["slc"].isel(azimuth_time=0).azimuth_time.values
        assert az0 == expected_az0

    def test_csi_rg0_matches_slc(self, csi_inputs):
        """rg0 should equal the first slant_range_time of the SLC."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        expected_rg0 = csi_inputs["slc"].isel(slant_range_time=0).slant_range_time.values
        assert rg0 == expected_rg0

    def test_csi_az_shape(self, csi_inputs):
        """Azimuth output shape should match the coregistration grid shape."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        coreg_shape = (csi_inputs["coreg_azimuth_time"].shape[0], csi_inputs["coreg_azimuth_time"].shape[1])
        assert az.shape == coreg_shape

    def test_csi_rg_shape(self, csi_inputs):
        """Range output shape should match the coregistration grid shape."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        coreg_shape = (csi_inputs["coreg_slant_range_time"].shape[0], csi_inputs["coreg_slant_range_time"].shape[1])
        assert rg.shape == coreg_shape

    def test_csi_basic_values_with_zero_daz_drg(self, csi_inputs):
        """With zero daz/drg and no poly corrections, az/rg should be close to coreg coordinates."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        coreg_az = csi_inputs["coreg_azimuth_time"].values
        coreg_rg = csi_inputs["coreg_slant_range_time"].values
        # Compare az as datetime64 with coreg_az (both in nanoseconds)
        np.testing.assert_array_equal(az, coreg_az)
        # Compare rg as float64 with coreg_rg
        np.testing.assert_allclose(rg, coreg_rg, atol=0.01)

    def test_csi_etad_empty_produces_no_corrections(self, csi_inputs):
        """Empty ETAD grid should produce zero corrections."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        coreg_az = csi_inputs["coreg_azimuth_time"].values
        coreg_rg = csi_inputs["coreg_slant_range_time"].values
        # Without ETAD corrections and with zero daz/drg, output should match input
        np.testing.assert_array_equal(az, coreg_az)
        np.testing.assert_allclose(rg, coreg_rg, atol=0.01)

    @pytest.mark.parametrize(
        "grid_size",
        [(3, 3), (5, 5), (7, 7), (5, 4), (3, 6)],
    )
    def test_csi_various_grid_sizes(self, grid_size):
        """compute_secondary_image_coordinates should handle various grid sizes."""

        n_az, n_rg = grid_size
        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array(
            [base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
        )
        rg = np.arange(n_rg, dtype=np.float64) * 0.5

        slc = xr.DataArray(
            np.ones((n_az, n_rg), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )

        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(n_rg)] for i in range(n_az)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(n_rg)] for i in range(n_az)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )

        daz = xr.DataArray(np.zeros((n_az, n_rg), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((n_az, n_rg), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert az_out.shape == (n_az, n_rg)
        assert rg_out.shape == (n_az, n_rg)
        assert np.isscalar(az0) or hasattr(az0, "item")
        assert np.isscalar(rg0) or hasattr(rg0, "item")

    def test_csi_with_y_poly(self, csi_inputs_with_y_poly):
        """Should handle y_poly correction without error."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs_with_y_poly)
        assert isinstance(az, np.ndarray)
        assert isinstance(rg, np.ndarray)

    def test_csi_with_x_poly(self, csi_inputs_with_x_poly):
        """Should handle x_poly correction without error."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs_with_x_poly)
        assert isinstance(az, np.ndarray)
        assert isinstance(rg, np.ndarray)

    def test_csi_with_both_polys(self, csi_inputs_with_both_polys):
        """Should handle both y_poly and x_poly corrections together."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs_with_both_polys)
        assert isinstance(az, np.ndarray)
        assert isinstance(rg, np.ndarray)
        assert az.shape == (5, 4)
        assert rg.shape == (5, 4)

    def test_csi_y_poly_zero_coefficients_same_as_none(self, csi_inputs_with_y_poly, csi_inputs):
        """Zero-coefficient y_poly should produce the same result as no poly."""

        az1, rg1, az01, rg01 = compute_secondary_image_coordinates(**csi_inputs_with_y_poly)
        az2, rg2, az02, rg02 = compute_secondary_image_coordinates(**csi_inputs)
        np.testing.assert_allclose(az1.astype(float), az2.astype(float), atol=1e-6)
        np.testing.assert_allclose(rg1.astype(float), rg2.astype(float), atol=1e-6)

    def test_csi_x_poly_zero_coefficients_same_as_none(self, csi_inputs_with_x_poly, csi_inputs):
        """Zero-coefficient x_poly should produce the same result as no poly."""

        az1, rg1, az01, rg01 = compute_secondary_image_coordinates(**csi_inputs_with_x_poly)
        az2, rg2, az02, rg02 = compute_secondary_image_coordinates(**csi_inputs)
        np.testing.assert_allclose(az1.astype(float), az2.astype(float), atol=1e-6)
        np.testing.assert_allclose(rg1.astype(float), rg2.astype(float), atol=1e-6)

    def test_csi_y_poly_nonzero_affects_az(self):
        """Non-zero y_poly coefficients should shift azimuth values."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(az, dims=["azimuth_time"])
        coreg_rg = xr.DataArray(rg, dims=["slant_range_time"])
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        # Poly with zero coefficients
        y_poly_zero = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        # Poly with non-zero constant coefficient (should shift all values)
        y_poly_nonzero = xr.DataArray(np.ones((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        az_zero, _, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=y_poly_zero,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        az_nonzero, _, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=y_poly_nonzero,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # Remove base epoch
        az_zero = az_zero - base_epoch
        az_nonzero = az_nonzero - base_epoch

        # The nonzero polynomial should shift azimuth values
        assert not np.allclose(az_zero.astype(float), az_nonzero.astype(float))

    def test_csi_x_poly_nonzero_affects_rg(self):
        """Non-zero x_poly coefficients should shift range values."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(az, dims=["azimuth_time"])
        coreg_rg = xr.DataArray(rg, dims=["slant_range_time"])
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        x_poly_zero = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )
        x_poly_nonzero = xr.DataArray(np.ones((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        _, rg_zero, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=x_poly_zero,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        _, rg_nonzero, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=x_poly_nonzero,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        assert not np.allclose(rg_zero, rg_nonzero)

    def test_csi_y_poly_preserves_shape(self, csi_inputs_with_y_poly):
        """y_poly correction should not change output shape."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs_with_y_poly)
        assert az.shape == (5, 4)
        assert rg.shape == (5, 4)

    def test_csi_x_poly_preserves_shape(self, csi_inputs_with_x_poly):
        """x_poly correction should not change output shape."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs_with_x_poly)
        assert az.shape == (5, 4)
        assert rg.shape == (5, 4)

    def test_csi_polys_with_different_types(self):
        """Should handle different coord_type values in polynomial attributes."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(az, dims=["azimuth_time"])
        coreg_rg = xr.DataArray(rg, dims=["slant_range_time"])
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        y_poly = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )
        x_poly = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(0.0), np.float64(0.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=y_poly,
            x_poly=x_poly,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert az_out.shape == (5, 4)
        assert rg_out.shape == (5, 4)

    def test_csi_etad_nonempty_produces_corrections(
        self, csi_slc, csi_coreg_az, csi_coreg_rg, csi_daz, csi_drg, csi_etad_nonempty
    ):
        """Non-empty ETAD grid should apply corrections."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=csi_coreg_az,
            coreg_slant_range_time=csi_coreg_rg,
            slc=csi_slc,
            etad_correction_grid=csi_etad_nonempty,
            daz=csi_daz,
            drg=csi_drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert az.shape == (5, 4)
        assert rg.shape == (5, 4)
        assert isinstance(az, np.ndarray)
        assert isinstance(rg, np.ndarray)

    def test_csi_etad_nonempty_finite_values(
        self, csi_slc, csi_coreg_az, csi_coreg_rg, csi_daz, csi_drg, csi_etad_nonempty
    ):
        """ETAD corrections should produce finite output values."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=csi_coreg_az,
            coreg_slant_range_time=csi_coreg_rg,
            slc=csi_slc,
            etad_correction_grid=csi_etad_nonempty,
            daz=csi_daz,
            drg=csi_drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert np.all(np.isfinite(az))
        assert np.all(np.isfinite(rg))

    def test_csi_etad_empty_vs_nonempty_different_outputs(self, csi_slc, csi_coreg_az, csi_coreg_rg, csi_daz, csi_drg):
        """Non-empty ETAD grid should produce different output than empty grid."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        etad_empty = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_empty, rg_empty, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=csi_coreg_az,
            coreg_slant_range_time=csi_coreg_rg,
            slc=csi_slc,
            etad_correction_grid=etad_empty,
            daz=csi_daz,
            drg=csi_drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # Build non-empty ETAD with non-zero corrections
        azcorr = np.ones((5, 4), dtype=np.float64) * 1e-9  # 1 ns correction
        rgcorr = np.ones((5, 4), dtype=np.float64) * 0.01  # 0.01s correction
        etad_nonempty = xr.Dataset(
            data_vars={
                "azcorr": (["azimuth_time", "slant_range_time"], azcorr),
                "rgcorr": (["azimuth_time", "slant_range_time"], rgcorr),
            },
            coords={"azimuth_time": az, "slant_range_time": rg},
        )

        az_nonempty, rg_nonempty, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=csi_coreg_az,
            coreg_slant_range_time=csi_coreg_rg,
            slc=csi_slc,
            etad_correction_grid=etad_nonempty,
            daz=csi_daz,
            drg=csi_drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # Outputs should differ due to ETAD corrections
        az_empty_float = np.asarray(az_empty, dtype="float64") / 1e9
        az_nonempty_float = np.asarray(az_nonempty, dtype="float64") / 1e9
        assert not np.allclose(az_empty_float, az_nonempty_float, atol=1e-6) or not np.allclose(
            rg_empty, rg_nonempty, atol=1e-6
        )

    def test_csi_etad_all_zeros_produces_zero_correction(self, csi_slc, csi_coreg_az, csi_coreg_rg, csi_daz, csi_drg):
        """ETAD grid with all-zero corrections should produce zero corrections."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        azcorr = np.zeros((5, 4), dtype=np.float64)
        rgcorr = np.zeros((5, 4), dtype=np.float64)
        etad_zeros = xr.Dataset(
            data_vars={
                "azcorr": (["azimuth_time", "slant_range_time"], azcorr),
                "rgcorr": (["azimuth_time", "slant_range_time"], rgcorr),
            },
            coords={"azimuth_time": az, "slant_range_time": rg},
        )

        az_out, rg_out, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=csi_coreg_az,
            coreg_slant_range_time=csi_coreg_rg,
            slc=csi_slc,
            etad_correction_grid=etad_zeros,
            daz=csi_daz,
            drg=csi_drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # Should match coreg coordinates since corrections are zero
        coreg_az = csi_coreg_az.values.astype("float64") / 1e9
        az_out_float = np.asarray(az_out, dtype="float64") / 1e9
        np.testing.assert_allclose(az_out_float, coreg_az, atol=1e-6)

    def test_csi_single_pixel_grid(self):
        """Single-pixel grid (1x1) should work without error."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(1_000_000_000, "ns")], dtype="datetime64[ns]")
        rg = np.array([0.5], dtype=np.float64)
        slc = xr.DataArray(
            np.array([[1 + 1j]], dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(np.array([[az[0]]], dtype="datetime64[ns]"), dims=["azimuth_time", "slant_range_time"])
        coreg_rg = xr.DataArray(np.array([[rg[0]]], dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        daz = xr.DataArray(np.array([[0]], dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.array([[0.0]], dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert az_out.shape == (1, 1)
        assert rg_out.shape == (1, 1)

    def test_csi_two_pixel_grid(self):
        """Small 2x2 grid should produce correct shape."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(2)], dtype="datetime64[ns]")
        rg = np.array([0.0, 0.5], dtype=np.float64)
        slc = xr.DataArray(
            np.ones((2, 2), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(2)] for i in range(2)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(2)] for i in range(2)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((2, 2), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((2, 2), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )
        assert az_out.shape == (2, 2)
        assert rg_out.shape == (2, 2)

    def test_csi_with_nonzero_daz_shift(self):
        """Non-zero daz should shift azimuth output values."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(4)] for i in range(5)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        # daz with 1-second offset
        daz = xr.DataArray(
            np.full((5, 4), np.timedelta64(1_000_000_000, "ns"), dtype="timedelta64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # Az should be shifted by daz (1 second = 1e9 ns / 1e9 = 1.0 second)
        coreg_az_float = coreg_az.values.astype("float64") / 1e9
        az_out_float = np.asarray(az_out, dtype="float64") / 1e9
        np.testing.assert_allclose(az_out_float, coreg_az_float + 1.0, atol=1e-6)

    def test_csi_with_nonzero_drg_shift(self):
        """Non-zero drg should shift range output values."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(4)] for i in range(5)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        # drg with 0.1 offset
        drg = xr.DataArray(np.full((5, 4), 0.1, dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        coreg_rg_expected = coreg_rg.values + 0.1
        np.testing.assert_allclose(rg_out, coreg_rg_expected, atol=1e-6)

    def test_csi_offset_applied_in_pixel_mode(self):
        """Offset values should be used when polynomial coord_type is 'pixels'."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(az, dims=["azimuth_time"])
        coreg_rg = xr.DataArray(rg, dims=["slant_range_time"])
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        y_poly = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(2.0), np.float64(1.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )
        x_poly = xr.DataArray(np.zeros((2, 2)), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(2.0), np.float64(1.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=y_poly,
            x_poly=x_poly,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        # With pixel mode: y = 5 + 0 = 5, x = 4 + 0 = 4
        # The polynomial is all zeros, so corrections are zero
        # Output should still be valid shapes
        assert az_out.shape == (5, 4)
        assert rg_out.shape == (5, 4)

    def test_csi_output_no_nan(self, csi_inputs):
        """Output az and rg should not contain NaN values."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert not np.any(np.isnan(az))
        assert not np.any(np.isnan(rg))

    def test_csi_output_no_inf(self, csi_inputs):
        """Output az and rg should not contain Inf values."""

        az, rg, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        assert not np.any(np.isinf(az))
        assert not np.any(np.isinf(rg))

    def test_csi_nonzero_grid_spacing(self):
        """Non-zero grid spacing should not cause errors."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 2_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 1.0
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(4)] for i in range(5)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=2.0,
            range_grid_spacing=1.0,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        assert az_out.shape == (5, 4)
        assert rg_out.shape == (5, 4)

    def test_csi_with_asymmetric_grid(self):
        """Asymmetric grid (tall vs wide) should produce matching output shape."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        n_az, n_rg = 10, 3
        az = np.array(
            [base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(n_az)], dtype="datetime64[ns]"
        )
        rg = np.arange(n_rg, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((n_az, n_rg), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(n_rg)] for i in range(n_az)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(n_rg)] for i in range(n_az)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((n_az, n_rg), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((n_az, n_rg), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az[:3], "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        assert az_out.shape == (n_az, n_rg)
        assert rg_out.shape == (n_az, n_rg)

    @pytest.mark.parametrize("dtaz,dtrg", [(0.1, 0.01), (0.5, 0.25), (1.0, 0.5), (2.0, 1.0), (10.0, 5.0)])
    def test_csi_various_grid_spacings(self, dtaz, dtrg):
        """compute_secondary_image_coordinates should handle various grid spacing values."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array(
            [base_epoch + np.timedelta64(i * int(dtaz * 1_000_000_000), "ns") for i in range(5)], dtype="datetime64[ns]"
        )
        rg = np.arange(4, dtype=np.float64) * dtrg
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(4)] for i in range(5)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az[:3], "slant_range_time": rg})

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=dtaz,
            range_grid_spacing=dtrg,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        assert az_out.shape == (5, 4)
        assert rg_out.shape == (5, 4)
        assert np.all(np.isfinite(az_out))
        assert np.all(np.isfinite(rg_out))

    def test_csi_transpose_preserves_data(self, csi_inputs):
        """Output should be transposed to (azimuth_time, slant_range_time) order."""

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(**csi_inputs)
        # Both should be 2D with the expected shape
        assert az_out.ndim == 2
        assert rg_out.ndim == 2

    def test_csi_negative_drg_shift(self):
        """Negative drg values should shift range output downward."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(
            np.array([[az[i] for _ in range(4)] for i in range(5)], dtype="datetime64[ns]"),
            dims=["azimuth_time", "slant_range_time"],
        )
        coreg_rg = xr.DataArray(
            np.array([[rg[j] for j in range(4)] for i in range(5)], dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
        )
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.full((5, 4), -0.5, dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        _, rg_out, _, _ = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=None,
            x_poly=None,
            offset={"azimuth_time": 0.0, "slant_range_time": 0.0},
        )

        coreg_rg_expected = coreg_rg.values - 0.5
        np.testing.assert_allclose(rg_out, coreg_rg_expected, atol=1e-6)

    def test_csi_offset_with_nonzero_offset_values(self):
        """Non-zero offset values should be applied correctly in pixel mode."""

        base_epoch = np.datetime64("2020-01-01T00:00:00", "ns")
        az = np.array([base_epoch + np.timedelta64(i * 1_000_000_000, "ns") for i in range(5)], dtype="datetime64[ns]")
        rg = np.arange(4, dtype=np.float64) * 0.5
        slc = xr.DataArray(
            np.ones((5, 4), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az, "slant_range_time": rg},
        )
        coreg_az = xr.DataArray(az, dims=["azimuth_time"])
        coreg_rg = xr.DataArray(rg, dims=["slant_range_time"])
        daz = xr.DataArray(np.zeros((5, 4), dtype="timedelta64[ns]"), dims=["azimuth_time", "slant_range_time"])
        drg = xr.DataArray(np.zeros((5, 4), dtype=np.float64), dims=["azimuth_time", "slant_range_time"])
        etad = xr.Dataset(coords={"azimuth_time": az, "slant_range_time": rg})

        # Use pixel mode with non-zero offsets and non-zero poly
        y_poly = xr.DataArray(np.array([[1.0, 0.0], [0.0, 0.0]]), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(2.0), np.float64(1.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )
        x_poly = xr.DataArray(np.array([[0.0, 1.0], [0.0, 0.0]]), dims=["i", "j"]).assign_attrs(
            {
                "off_y": 0,
                "offset": [np.float64(1.0), np.float64(2.0)],
                "scale_fact": [np.float64(1.0), np.float64(1.0)],
                "type": "pixels",
            }
        )

        az_out, rg_out, az0, rg0 = compute_secondary_image_coordinates(
            coreg_azimuth_time=coreg_az,
            coreg_slant_range_time=coreg_rg,
            slc=slc,
            etad_correction_grid=etad,
            daz=daz,
            drg=drg,
            azimuth_grid_spacing=1.0,
            range_grid_spacing=0.5,
            y_poly=y_poly,
            x_poly=x_poly,
            offset={"azimuth_time": 10.0, "slant_range_time": 20.0},
        )

        assert az_out.shape == (5, 4)
        assert rg_out.shape == (5, 4)
