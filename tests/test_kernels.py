"""Unit tests for cop_ard_core.kernels."""

import numpy as np

from cop_ard_core.kernels import (
    gaussian_kernel,
    get_workspace,
    numba_dot,
    polyval,
)


class TestGaussianKernel:
    """Tests for gaussian_kernel."""

    def test_gaussian_kernel_size_3(self):
        """3x3 Gaussian kernel should sum to 1."""
        kernel = gaussian_kernel(size=3, sigma=1.0)
        assert kernel.shape == (3, 3)
        np.testing.assert_allclose(kernel.sum(), 1.0)

    def test_gaussian_kernel_size_5(self):
        """5x5 Gaussian kernel should sum to 1."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        assert kernel.shape == (5, 5)
        np.testing.assert_allclose(kernel.sum(), 1.0)

    def test_gaussian_kernel_symmetric(self):
        """Gaussian kernel should be symmetric."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-10)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-10)

    def test_gaussian_kernel_positive(self):
        """All kernel values should be positive."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        assert np.all(kernel > 0)

    def test_gaussian_kernel_center_peak(self):
        """Center value should be the maximum."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        center = kernel[2, 2]
        assert center == kernel.max()

    def test_gaussian_kernel_larger_sigma(self):
        """Larger sigma should produce wider kernel (more spread out)."""
        k1 = gaussian_kernel(size=7, sigma=0.5)
        k2 = gaussian_kernel(size=7, sigma=2.0)
        # Center should be lower with larger sigma (more spread)
        assert k2[3, 3] < k1[3, 3]

    def test_gaussian_kernel_odd_sizes(self):
        """Should work with various odd sizes."""
        for size in [1, 3, 5, 7, 11]:
            kernel = gaussian_kernel(size=size)
            assert kernel.shape == (size, size)
            np.testing.assert_allclose(kernel.sum(), 1.0)


class TestPolyval:
    """Tests for polyval (numba)."""

    def test_polyval_constant(self):
        """Constant polynomial (all coeffs 0 except c0) should return c0 values."""
        coeffs = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        result = polyval(0.0, coeffs)
        # Horner's method at t=0 returns the constant term coefficients
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_polyval_linear(self):
        """Linear polynomial p(x) = a + b*x."""
        coeffs = np.array([[1.0, 10.0], [2.0, 20.0]])  # p(x) = 1 + 2x for col 0
        result = polyval(3.0, coeffs)
        # Horner: result = coeffs[1]*3 + coeffs[0] = [2*3+1, 20*3+10] = [7, 70]
        np.testing.assert_array_equal(result, [7.0, 70.0])

    def test_polyval_quadratic(self):
        """Quadratic polynomial p(x) = a + b*x + c*x^2."""
        # p(x) = 1 + 2x + 3x^2, evaluate at x=2 => 1 + 4 + 12 = 17
        coeffs = np.array([[1.0], [2.0], [3.0]])
        result = polyval(2.0, coeffs)
        np.testing.assert_array_equal(result, [17.0])

    def test_polyval_at_zero(self):
        """Polynomial at x=0 should return constant term."""
        coeffs = np.array([[5.0], [3.0], [7.0]])
        result = polyval(0.0, coeffs)
        np.testing.assert_array_equal(result, [5.0])

    def test_polyval_single_coeff(self):
        """Single coefficient should work."""
        coeffs = np.array([[42.0]])
        result = polyval(100.0, coeffs)
        np.testing.assert_array_equal(result, [42.0])


class TestNumbaDot:
    """Tests for numba_dot."""

    def test_numba_dot_simple(self):
        """Dot product of [1,2,3] and [4,5,6] should be 32."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, 32.0)

    def test_numba_dot_orthogonal(self):
        """Orthogonal vectors should have dot product 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, 0.0)

    def test_numba_dot_self_dot(self):
        """Self dot product should be squared norm."""
        a = np.array([3.0, 4.0])
        result = numba_dot(a, a)
        np.testing.assert_almost_equal(result, 25.0)

    def test_numba_dot_single_element(self):
        """Single element dot product should be product."""
        result = numba_dot(np.array([5.0]), np.array([3.0]))
        np.testing.assert_almost_equal(result, 15.0)

    def test_numba_dot_matches_np_dot(self):
        """Should match numpy dot product."""
        a = np.random.rand(10)
        b = np.random.rand(10)
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, np.dot(a, b))


class TestGetWorkspace:
    """Tests for get_workspace."""

    def test_get_workspace_basic(self):
        """Basic workspace calculation."""
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        triangs = np.array([0, 1, 4])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 4, 4], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert isinstance(max_w, int)
        assert isinstance(max_h, int)
        assert isinstance(workspace, int)
        assert workspace == max_w * max_h

    def test_get_workspace_outside_bbox(self):
        """Polygon outside bounding box should not contribute."""
        y = np.array([10.0, 20.0, 30.0])
        x = np.array([10.0, 20.0, 30.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace == 0

    def test_get_workspace_intersecting_bbox(self):
        """Polygon intersecting bounding box should have positive workspace."""
        y = np.array([1.0, 3.0, 5.0])
        x = np.array([1.0, 3.0, 5.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 4, 4], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace > 0
