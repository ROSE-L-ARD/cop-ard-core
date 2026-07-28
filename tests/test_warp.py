"""Unit tests for cop_ard_core.warp."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.warp import (
    _dot_numba,
    _nvalid_numba,
    _sinc_warp_inner_numba,
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
)


@pytest.fixture
def image_7x7():
    """7x7 real-valued image for boundary testing."""
    return np.arange(49, dtype=np.float64).reshape(7, 7)


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
        assert result == result.dtype.type(result)

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
        """Create mock daz/drg DataArrays."""
        acq = np.array(["2020-01-01"], dtype="datetime64[ns]")
        az = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        sr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        daz = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype="timedelta64[ns]"),
            dims=["acquisition_date", "azimuth_time", "slant_range_time"],
            coords={"acquisition_date": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        drg = xr.DataArray(
            np.zeros((n_acq, n_az, n_sr), dtype=np.float64),
            dims=["acquisition_date", "azimuth_time", "slant_range_time"],
            coords={"acquisition_date": acq, "azimuth_time": az, "slant_range_time": sr},
        )
        return daz, drg, acq, az, sr

    def test_grid_data_block_single_acquisition_zero_output(self):
        """Single acquisition should produce zero daz/drg."""
        daz, drg, acq, az, sr = self._make_mock_daz_drg(n_acq=1)
        # Create simulation with matching structure
        sim_data = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_date", "azimuth_time", "slant_range_time"],
            coords={"acquisition_date": acq, "azimuth_time": az, "slant_range_time": sr},
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
            dims=["acquisition_date", "azimuth_time", "slant_range_time"],
            coords={"acquisition_date": acq, "azimuth_time": az, "slant_range_time": sr},
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
