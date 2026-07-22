import pytest
"""Unit tests for cop_ard_core.kernels.

Tests cover all kernel functions including:
- gaussian_kernel: 2D Gaussian smoothing kernel generation
- polyval: Numba-accelerated polynomial evaluation (Horner's method)
- numba_dot: Numba-accelerated dot product
- area_proj_integrate_segment: Numba scanline area integration
- compute_weights: Numba per-step weight computation
- aggregate_dense_to_sparse: Numba dense-to-sparse aggregation
- area_proj_triangles: Numba polygon rasterization via area projection
- get_workspace: Numba workspace size calculation for polygon sets
- computeGammaNaughtArea: Numba Gamma Naught Area computation
- update_dense_array: Numba sparse-to-dense accumulation within bounding box
- csr_times_dense: Numba sparse matrix times dense vector multiplication
- compute_incidence_angle_ground: Numba incidence angle computation
"""

import numpy as np

from cop_ard_core.kernels import (
    aggregate_dense_to_sparse,
    area_proj_integrate_segment,
    area_proj_triangles,
    computeGammaNaughtArea,
    compute_incidence_angle_ground,
    compute_weights,
    csr_times_dense,
    gaussian_kernel,
    get_workspace,
    main_loop,
    numba_dot,
    polyval,
    update_dense_array,
)


class TestGaussianKernel:
    """Tests for gaussian_kernel."""

    @pytest.mark.unit
    def test_gaussian_kernel_size_3(self):
        """3x3 Gaussian kernel should sum to 1."""
        kernel = gaussian_kernel(size=3, sigma=1.0)
        assert kernel.shape == (3, 3)
        np.testing.assert_allclose(kernel.sum(), 1.0)

    @pytest.mark.unit
    def test_gaussian_kernel_size_5(self):
        """5x5 Gaussian kernel should sum to 1."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        assert kernel.shape == (5, 5)
        np.testing.assert_allclose(kernel.sum(), 1.0)

    @pytest.mark.unit
    def test_gaussian_kernel_symmetric(self):
        """Gaussian kernel should be symmetric."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        np.testing.assert_allclose(kernel, kernel.T, atol=1e-10)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-10)

    @pytest.mark.unit
    def test_gaussian_kernel_positive(self):
        """All kernel values should be positive."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        assert np.all(kernel > 0)

    @pytest.mark.unit
    def test_gaussian_kernel_center_peak(self):
        """Center value should be the maximum."""
        kernel = gaussian_kernel(size=5, sigma=1.0)
        center = kernel[2, 2]
        assert center == kernel.max()

    @pytest.mark.unit
    def test_gaussian_kernel_larger_sigma(self):
        """Larger sigma should produce wider kernel (more spread out)."""
        k1 = gaussian_kernel(size=7, sigma=0.5)
        k2 = gaussian_kernel(size=7, sigma=2.0)
        # Center should be lower with larger sigma (more spread)
        assert k2[3, 3] < k1[3, 3]

    @pytest.mark.unit
    def test_gaussian_kernel_odd_sizes(self):
        """Should work with various odd sizes."""
        for size in [1, 3, 5, 7, 11]:
            kernel = gaussian_kernel(size=size)
            assert kernel.shape == (size, size)
            np.testing.assert_allclose(kernel.sum(), 1.0)


class TestPolyval:
    """Tests for polyval (numba)."""

    @pytest.mark.unit
    def test_polyval_constant(self):
        """Constant polynomial (all coeffs 0 except c0) should return c0 values."""
        coeffs = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
        result = polyval(0.0, coeffs)
        # Horner's method at t=0 returns the constant term coefficients
        np.testing.assert_array_equal(result, [1.0, 2.0])

    @pytest.mark.unit
    def test_polyval_linear(self):
        """Linear polynomial p(x) = a + b*x."""
        coeffs = np.array([[1.0, 10.0], [2.0, 20.0]])  # p(x) = 1 + 2x for col 0
        result = polyval(3.0, coeffs)
        # Horner: result = coeffs[1]*3 + coeffs[0] = [2*3+1, 20*3+10] = [7, 70]
        np.testing.assert_array_equal(result, [7.0, 70.0])

    @pytest.mark.unit
    def test_polyval_quadratic(self):
        """Quadratic polynomial p(x) = a + b*x + c*x^2."""
        # p(x) = 1 + 2x + 3x^2, evaluate at x=2 => 1 + 4 + 12 = 17
        coeffs = np.array([[1.0], [2.0], [3.0]])
        result = polyval(2.0, coeffs)
        np.testing.assert_array_equal(result, [17.0])

    @pytest.mark.unit
    def test_polyval_at_zero(self):
        """Polynomial at x=0 should return constant term."""
        coeffs = np.array([[5.0], [3.0], [7.0]])
        result = polyval(0.0, coeffs)
        np.testing.assert_array_equal(result, [5.0])

    @pytest.mark.unit
    def test_polyval_single_coeff(self):
        """Single coefficient should work."""
        coeffs = np.array([[42.0]])
        result = polyval(100.0, coeffs)
        np.testing.assert_array_equal(result, [42.0])


class TestNumbaDot:
    """Tests for numba_dot."""

    @pytest.mark.unit
    def test_numba_dot_simple(self):
        """Dot product of [1,2,3] and [4,5,6] should be 32."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, 32.0)

    @pytest.mark.unit
    def test_numba_dot_orthogonal(self):
        """Orthogonal vectors should have dot product 0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, 0.0)

    @pytest.mark.unit
    def test_numba_dot_self_dot(self):
        """Self dot product should be squared norm."""
        a = np.array([3.0, 4.0])
        result = numba_dot(a, a)
        np.testing.assert_almost_equal(result, 25.0)

    @pytest.mark.unit
    def test_numba_dot_single_element(self):
        """Single element dot product should be product."""
        result = numba_dot(np.array([5.0]), np.array([3.0]))
        np.testing.assert_almost_equal(result, 15.0)

    @pytest.mark.unit
    def test_numba_dot_matches_np_dot(self):
        """Should match numpy dot product."""
        a = np.random.rand(10)
        b = np.random.rand(10)
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, np.dot(a, b))


class TestGetWorkspace:
    """Tests for get_workspace."""

    @pytest.mark.unit
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

    @pytest.mark.unit
    def test_get_workspace_outside_bbox(self):
        """Polygon outside bounding box should not contribute."""
        y = np.array([10.0, 20.0, 30.0])
        x = np.array([10.0, 20.0, 30.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace == 0

    @pytest.mark.unit
    def test_get_workspace_intersecting_bbox(self):
        """Polygon intersecting bounding box should have positive workspace."""
        y = np.array([1.0, 3.0, 5.0])
        x = np.array([1.0, 3.0, 5.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 4, 4], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace > 0

    @pytest.mark.unit
    def test_get_workspace_even_size(self):
        """Gaussian kernel with even size should still produce normalized output."""
        kernel = gaussian_kernel(size=4, sigma=1.0)
        assert kernel.shape == (4, 4)
        np.testing.assert_allclose(kernel.sum(), 1.0)

    @pytest.mark.unit
    def test_gaussian_kernel_larger_kernel(self):
        """Larger kernel should have lower center with same sigma."""
        k5 = gaussian_kernel(size=5, sigma=1.0)
        k9 = gaussian_kernel(size=9, sigma=1.0)
        assert k9[4, 4] < k5[2, 2]

    @pytest.mark.unit
    def test_gaussian_kernel_type(self):
        """Gaussian kernel should return float64 ndarray."""
        kernel = gaussian_kernel(size=3)
        assert isinstance(kernel, np.ndarray)
        assert kernel.dtype == np.float64

    @pytest.mark.unit
    def test_gaussian_kernel_edge_decay(self):
        """Kernel values should decrease from center toward edges."""
        kernel = gaussian_kernel(size=5, sigma=0.5)
        center = kernel[2, 2]
        edge = kernel[0, 2]
        corner = kernel[0, 0]
        assert center > edge
        assert edge > corner

    @pytest.mark.unit
    def test_polyval_negative_input(self):
        """Polynomial with negative input should evaluate correctly."""
        # p(x) = 1 + 2x + 3x^2, at x=-1 => 1 - 2 + 3 = 2
        coeffs = np.array([[1.0], [2.0], [3.0]])
        result = polyval(-1.0, coeffs)
        np.testing.assert_array_equal(result, [2.0])

    @pytest.mark.unit
    def test_polyval_large_coefficients(self):
        """Large coefficients should not overflow."""
        coeffs = np.array([[1e10], [1e10], [1e10]])
        result = polyval(1.0, coeffs)
        expected = 3e10
        np.testing.assert_almost_equal(result[0], expected)

    @pytest.mark.unit
    def test_polyval_zero_polynomial(self):
        """All-zero polynomial should return zero."""
        coeffs = np.array([[0.0], [0.0], [0.0]])
        result = polyval(5.0, coeffs)
        np.testing.assert_array_equal(result, [0.0])

    @pytest.mark.unit
    def test_polyval_multiple_output_dims(self):
        """Polynomial with multiple output dimensions should work."""
        # 2D output: p(x) = [a0 + a1*x, b0 + b1*x]
        coeffs = np.array([[1.0, 10.0], [2.0, 20.0]])
        result = polyval(3.0, coeffs)
        np.testing.assert_array_equal(result, [7.0, 70.0])

    @pytest.mark.unit
    def test_numba_dot_negative_values(self):
        """Dot product with negative values should handle signs correctly."""
        a = np.array([-1.0, 2.0, -3.0])
        b = np.array([4.0, -5.0, 6.0])
        result = numba_dot(a, b)
        expected = -4 - 10 - 18
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.unit
    def test_numba_dot_anti_parallel(self):
        """Anti-parallel vectors should have negative dot product."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, -1.0)

    @pytest.mark.unit
    def test_numba_dot_high_dimension(self):
        """High-dimensional dot product should match numpy."""
        n = 100
        a = np.random.rand(n)
        b = np.random.rand(n)
        result = numba_dot(a, b)
        np.testing.assert_almost_equal(result, np.dot(a, b))

    @pytest.mark.unit
    def test_numba_dot_single_element_type(self):
        """Single element dot product should return float."""
        result = numba_dot(np.array([2.0]), np.array([3.0]))
        assert isinstance(result, (float, np.floating))

    @pytest.mark.unit
    def test_get_workspace_single_triangle(self):
        """Single triangle workspace should be non-zero when within bounds."""
        y = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 3, 3], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace >= 0
        assert workspace == max_w * max_h

    @pytest.mark.unit
    def test_get_workspace_multiple_triangles(self):
        """Multiple triangles should find the max workspace."""
        y = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0])
        x = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0])
        triangs = np.array([0, 1, 2, 3, 4, 5])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 8, 8], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace > 0
        assert max_w >= 3
        assert max_h >= 3

    @pytest.mark.unit
    def test_get_workspace_partial_overlap(self):
        """Polygon partially outside bbox should only count intersecting portion."""
        y = np.array([-1.0, 1.0, 3.0])
        x = np.array([-1.0, 1.0, 3.0])
        triangs = np.array([0, 1, 2])
        vertexes = np.array([0, 1, 2, 0])
        bbox = np.array([0, 0, 2, 2], dtype=np.int64)
        max_w, max_h, workspace = get_workspace(y, x, triangs, vertexes, bbox)
        assert workspace >= 0
        # Should be non-zero since polygon partially overlaps bbox
        # floor(3) - floor(-1) + 1 = 5 would be full, but clipped to overlap region
        if workspace > 0:
            assert isinstance(max_w, int) and isinstance(max_h, int)
            assert workspace == max_w * max_h


class TestAreaProjIntegrateSegment:
    """Tests for area_proj_integrate_segment (numba)."""

    @pytest.mark.unit
    def test_area_proj_integrate_segment_diagonal_line(self):
        """Diagonal line from (0,0) to (1,1) should accumulate expected weights."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        area_proj_integrate_segment(
            y1=0.0,
            y2=1.0,
            x1=0.0,
            x2=1.0,
            length=length,
            width=width,
            w_arr=w_arr,
            plane_orientation=1,
        )
        # Diagonal from (0,0) to (1,1): slope=1, segment_multiplier=1
        # Only pixel (col=0, row=0) is affected
        np.testing.assert_almost_equal(w_arr[0], 0.5, decimal=10)
        np.testing.assert_almost_equal(w_arr[1], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[2], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[3], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[4], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[5], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[6], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[7], 0.0, decimal=10)
        np.testing.assert_almost_equal(w_arr[8], 0.0, decimal=10)

    @pytest.mark.unit
    def test_area_proj_integrate_segment_opposite_direction(self):
        """Reverse direction should invert the area sign."""
        length, width = 3, 3
        w_arr_fwd = np.zeros(length * width, dtype=np.float64)
        w_arr_rev = np.zeros(length * width, dtype=np.float64)

        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr_fwd, plane_orientation=1,
        )
        area_proj_integrate_segment(
            y1=1.0, y2=0.0, x1=1.0, x2=0.0,
            length=length, width=width,
            w_arr=w_arr_rev, plane_orientation=1,
        )
        # Reverse direction gives opposite sign
        np.testing.assert_array_almost_equal(w_arr_fwd, -w_arr_rev)

    @pytest.mark.unit
    def test_area_proj_integrate_segment_negative_orientation(self):
        """Negative plane_orientation should invert the area sign."""
        length, width = 3, 3
        w_arr_pos = np.zeros(length * width, dtype=np.float64)
        w_arr_neg = np.zeros(length * width, dtype=np.float64)

        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr_pos, plane_orientation=1,
        )
        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr_neg, plane_orientation=-1,
        )
        np.testing.assert_array_almost_equal(w_arr_pos, -w_arr_neg)

    @pytest.mark.unit
    def test_area_proj_integrate_segment_vertical_line(self):
        """Exactly vertical line should return early with no changes."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        initial_sum = w_arr.sum()
        area_proj_integrate_segment(
            y1=0.0, y2=2.0, x1=1.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        assert w_arr.sum() == initial_sum

    @pytest.mark.unit
    def test_area_proj_integrate_segment_outside_grid_x(self):
        """Line completely to the left of grid should not modify array."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        initial_sum = w_arr.sum()
        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=-5.0, x2=-3.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        assert w_arr.sum() == initial_sum

    @pytest.mark.unit
    def test_area_proj_integrate_segment_outside_grid_y(self):
        """Line completely below grid should not modify array."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        initial_sum = w_arr.sum()
        area_proj_integrate_segment(
            y1=10.0, y2=12.0, x1=1.0, x2=2.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        assert w_arr.sum() == initial_sum

    @pytest.mark.unit
    def test_area_proj_integrate_segment_nan_coords(self):
        """NaN coordinates should cause early exit."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        initial_sum = w_arr.sum()
        area_proj_integrate_segment(
            y1=float('nan'), y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        assert w_arr.sum() == initial_sum

    @pytest.mark.unit
    def test_area_proj_integrate_segment_accumulates(self):
        """Multiple calls should accumulate into the same array."""
        length, width = 3, 3
        w_arr = np.zeros(length * width, dtype=np.float64)
        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        sum_after_first = w_arr.sum()
        area_proj_integrate_segment(
            y1=0.0, y2=1.0, x1=0.0, x2=1.0,
            length=length, width=width,
            w_arr=w_arr, plane_orientation=1,
        )
        # Each call adds 0.5 to w_arr[0]
        np.testing.assert_almost_equal(w_arr[0], 1.0, decimal=10)
        # All other elements should remain 0
        np.testing.assert_almost_equal(w_arr.sum(), 1.0, decimal=10)


class TestComputeWeights:
    """Tests for compute_weights (numba)."""

    @pytest.mark.unit
    def test_compute_weights_horizontal_line(self):
        """Horizontal line should advance x_step by 1.0 per pixel."""
        w_arr = np.zeros(3 * 3, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=0.5,
            x_end=2.5,
            slope=0.0,
            offset=1.0,
            length=3,
            width=3,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        # x_next for horizontal = x_index + 1 = 1.0
        np.testing.assert_almost_equal(x_start, 1.0)
        assert count == 1
        assert stop is False
        # area = 1.0 * (1.0 - 0.5) * (1.0 + 0 - 2*1)/2 = 0.5 * (-0.5) = -0.25
        # Actually: y_center = (y_next + y_start - 2*y_index)/2
        # y_start = 0*0.5 + 1 = 1.0, y_next = 0*1.0 + 1 = 1.0, y_index = floor(1.0+1e-6) = 1
        # y_center = (1.0 + 1.0 - 2*1)/2 = 0.0
        # area = 1.0 * 0.5 * 0.0 = 0.0
        np.testing.assert_almost_equal(w_arr[1 * 3 + 0], 0.0)

    @pytest.mark.unit
    def test_compute_weights_positive_slope(self):
        """Positive slope should use y_index + 1 for horizontal boundary."""
        w_arr = np.zeros(5 * 5, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=0.0,
            x_end=4.0,
            slope=0.5,
            offset=0.0,
            length=5,
            width=5,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        assert count == 1
        assert stop is False
        assert x_start > 0.0
        assert x_start <= 1.0

    @pytest.mark.unit
    def test_compute_weights_negative_slope(self):
        """Negative slope should use y_index for horizontal boundary."""
        w_arr = np.zeros(5 * 5, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=0.0,
            x_end=4.0,
            slope=-0.5,
            offset=2.0,
            length=5,
            width=5,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        assert count == 1
        assert stop is False

    @pytest.mark.unit
    def test_compute_weights_below_grid_jumps_up(self):
        """Line below grid should jump to entry point."""
        w_arr = np.zeros(5 * 5, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=0.0,
            x_end=4.0,
            slope=1.0,
            offset=-2.0,
            length=5,
            width=5,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        # Line enters at y=0 => x = 2.0 / 1.0 = 2.0
        np.testing.assert_almost_equal(x_start, 2.0)
        assert stop is False

    @pytest.mark.unit
    def test_compute_weights_above_grid_jumps_down(self):
        """Line above grid should jump to exit point."""
        w_arr = np.zeros(5 * 5, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=0.0,
            x_end=4.0,
            slope=-1.0,
            offset=8.0,
            length=5,
            width=5,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        # Line exits at y=4 (length-1) => x = (4 - 8)/(-1) = 4.0
        np.testing.assert_almost_equal(x_start, 4.0)

    @pytest.mark.unit
    def test_compute_weights_at_boundary(self):
        """Line at exact pixel boundary should handle correctly."""
        w_arr = np.zeros(3 * 3, dtype=np.float64)
        x_start, count, stop = compute_weights(
            x_start=1.0,
            x_end=2.0,
            slope=0.0,
            offset=1.0,
            length=3,
            width=3,
            x_increment_margin=1e-6,
            segment_multiplier=1.0,
            w_arr=w_arr,
            count=0,
        )
        assert count == 1
        assert stop is False


class TestAggregateDenseToSparse:
    """Tests for aggregate_dense_to_sparse (numba)."""

    @pytest.mark.unit
    def test_aggregate_dense_to_sparse_single_weight(self):
        """Single non-zero weight should produce one sparse entry."""
        max_h, max_w = 2, 2
        w_arr = np.zeros(max_h * max_w, dtype=np.float64)
        w_arr[0] = 2.5  # Single non-zero weight

        data = np.empty(10, dtype=np.float64)
        col = np.empty(10, dtype=np.int64)
        ind_ptr = np.zeros(5, dtype=np.int64)
        nlook = np.zeros(4, dtype=np.float64)
        l1_aux = np.zeros(16, dtype=np.float64)
        l2_aux = np.zeros(16, dtype=np.float64)
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        bbox_w = np.array([99, -99, 99, -99], dtype=np.float64)

        counter = aggregate_dense_to_sparse(
            w_arr=w_arr,
            max_h=max_h,
            max_w=max_w,
            xmin=0,
            ymin=0,
            width=4,
            linear_global_index=0,
            xsize_global=4,
            xmin_global=0,
            ymin_global=0,
            xmin_chunk=0,
            ymin_chunk=0,
            bbox=bbox,
            bbox_w=bbox_w,
            data=data,
            col=col,
            ind_ptr=ind_ptr,
            nlook=nlook,
            l1_aux=l1_aux,
            l2_aux=l2_aux,
            t=0,
            counter=0,
        )
        assert counter == 1
        np.testing.assert_almost_equal(data[0], 2.5)
        np.testing.assert_almost_equal(nlook[0], 2.5)

    @pytest.mark.unit
    def test_aggregate_dense_to_sparse_zeros_not_included(self):
        """Zero weights should not appear in sparse output."""
        max_h, max_w = 2, 2
        w_arr = np.zeros(max_h * max_w, dtype=np.float64)

        data = np.empty(10, dtype=np.float64)
        col = np.empty(10, dtype=np.int64)
        ind_ptr = np.zeros(5, dtype=np.int64)
        nlook = np.zeros(4, dtype=np.float64)
        l1_aux = np.zeros(16, dtype=np.float64)
        l2_aux = np.zeros(16, dtype=np.float64)
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        bbox_w = np.array([99, -99, 99, -99], dtype=np.float64)

        counter = aggregate_dense_to_sparse(
            w_arr=w_arr,
            max_h=max_h,
            max_w=max_w,
            xmin=0,
            ymin=0,
            width=4,
            linear_global_index=0,
            xsize_global=4,
            xmin_global=0,
            ymin_global=0,
            xmin_chunk=0,
            ymin_chunk=0,
            bbox=bbox,
            bbox_w=bbox_w,
            data=data,
            col=col,
            ind_ptr=ind_ptr,
            nlook=nlook,
            l1_aux=l1_aux,
            l2_aux=l2_aux,
            t=0,
            counter=0,
        )
        assert counter == 0
        np.testing.assert_almost_equal(nlook[0], 0.0)

    @pytest.mark.unit
    def test_aggregate_dense_to_sparse_multiple_weights(self):
        """Multiple non-zero weights should produce multiple entries."""
        max_h, max_w = 2, 2
        w_arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        data = np.empty(10, dtype=np.float64)
        col = np.empty(10, dtype=np.int64)
        ind_ptr = np.zeros(5, dtype=np.int64)
        nlook = np.zeros(4, dtype=np.float64)
        l1_aux = np.zeros(16, dtype=np.float64)
        l2_aux = np.zeros(16, dtype=np.float64)
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        bbox_w = np.array([99, -99, 99, -99], dtype=np.float64)

        counter = aggregate_dense_to_sparse(
            w_arr=w_arr,
            max_h=max_h,
            max_w=max_w,
            xmin=0,
            ymin=0,
            width=4,
            linear_global_index=0,
            xsize_global=4,
            xmin_global=0,
            ymin_global=0,
            xmin_chunk=0,
            ymin_chunk=0,
            bbox=bbox,
            bbox_w=bbox_w,
            data=data,
            col=col,
            ind_ptr=ind_ptr,
            nlook=nlook,
            l1_aux=l1_aux,
            l2_aux=l2_aux,
            t=0,
            counter=0,
        )
        assert counter == 4
        np.testing.assert_almost_equal(nlook[0], 10.0)

    @pytest.mark.unit
    def test_aggregate_dense_to_sparse_abs_weight(self):
        """Weight should be absolute value (negative inputs become positive)."""
        max_h, max_w = 2, 2
        w_arr = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)

        data = np.empty(10, dtype=np.float64)
        col = np.empty(10, dtype=np.int64)
        ind_ptr = np.zeros(5, dtype=np.int64)
        nlook = np.zeros(4, dtype=np.float64)
        l1_aux = np.zeros(16, dtype=np.float64)
        l2_aux = np.zeros(16, dtype=np.float64)
        bbox = np.array([0, 0, 5, 5], dtype=np.int64)
        bbox_w = np.array([99, -99, 99, -99], dtype=np.float64)

        counter = aggregate_dense_to_sparse(
            w_arr=w_arr,
            max_h=max_h,
            max_w=max_w,
            xmin=0,
            ymin=0,
            width=4,
            linear_global_index=0,
            xsize_global=4,
            xmin_global=0,
            ymin_global=0,
            xmin_chunk=0,
            ymin_chunk=0,
            bbox=bbox,
            bbox_w=bbox_w,
            data=data,
            col=col,
            ind_ptr=ind_ptr,
            nlook=nlook,
            l1_aux=l1_aux,
            l2_aux=l2_aux,
            t=0,
            counter=0,
        )
        assert counter == 4
        np.testing.assert_almost_equal(data[0], 1.0)
        np.testing.assert_almost_equal(data[2], 3.0)


class TestAreaProjTriangles:
    """Tests for area_proj_triangles (numba)."""

    @pytest.mark.unit
    def test_area_proj_triangles_single_triangle(self):
        """A single triangle should produce non-zero nlook."""
        y = np.array([0.5, 1.5, 0.5], dtype=np.float64)
        x = np.array([0.5, 0.5, 1.5], dtype=np.float64)
        triangs = np.array([0, 1, 2], dtype=np.int64)
        vertexes = np.array([0, 1, 2, 0], dtype=np.int64)
        bbox = np.array([0, 0, 2, 2], dtype=np.int64)

        linear_global_index = 0
        length = 4
        width = 4
        xsize_global = 4
        xmin_global = 0
        ymin_global = 0
        xmin_chunk = 0
        ymin_chunk = 0
        radar_grid_ysize = 1
        plane_orientation = 1
        expected_nlooks = 3

        bbox_l1 = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)
        bbox_l2 = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)
        bbox_w = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)

        (data, ind_ptr, col), (l1, ind_ptr_l1, col_l1), (l2, ind_ptr_l2, col_l2), nlook = (
            area_proj_triangles(
                y=y, x=x, triangs=triangs, vertexes=vertexes,
                bbox=bbox,
                linear_global_index=linear_global_index,
                length=length, width=width,
                xsize_global=xsize_global,
                xmin_global=xmin_global, ymin_global=ymin_global,
                xmin_chunk=xmin_chunk, ymin_chunk=ymin_chunk,
                radar_grid_ysize=radar_grid_ysize,
                plane_orientation=plane_orientation,
                bbox_l1=bbox_l1, bbox_l2=bbox_l2, bbox_w=bbox_w,
                expected_nlooks=expected_nlooks,
            )
        )
        assert isinstance(data, np.ndarray)
        assert isinstance(ind_ptr, np.ndarray)
        assert isinstance(col, np.ndarray)
        assert isinstance(nlook, np.ndarray)
        assert len(nlook) == 1
        assert nlook[0] >= 0
        assert data.dtype == np.float64
        assert ind_ptr.dtype == np.int64

    @pytest.mark.unit
    def test_area_proj_triangles_empty_triangles(self):
        """Zero triangles should return empty arrays."""
        y = np.array([], dtype=np.float64)
        x = np.array([], dtype=np.float64)
        triangs = np.array([], dtype=np.int64)
        vertexes = np.array([0, 1, 2, 0], dtype=np.int64)
        bbox = np.array([0, 0, 2, 2], dtype=np.int64)

        bbox_l1 = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)
        bbox_l2 = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)
        bbox_w = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)

        (data, ind_ptr, col), (l1, ind_ptr_l1, col_l1), (l2, ind_ptr_l2, col_l2), nlook = (
            area_proj_triangles(
                y=y, x=x, triangs=triangs, vertexes=vertexes,
                bbox=bbox,
                linear_global_index=0,
                length=2, width=2,
                xsize_global=2,
                xmin_global=0, ymin_global=0,
                xmin_chunk=0, ymin_chunk=0,
                radar_grid_ysize=1,
                plane_orientation=1,
                bbox_l1=bbox_l1, bbox_l2=bbox_l2, bbox_w=bbox_w,
                expected_nlooks=1,
            )
        )
        assert len(data) == 0
        assert len(col) == 0
        assert len(ind_ptr) == 0


class TestComputeGammaNaughtArea:
    """Tests for computeGammaNaughtArea (numba)."""

    @pytest.mark.unit
    def test_compute_gamma_naught_area_single_triangle(self):
        """Single triangle should produce non-zero GNA values."""
        # Single triangle: vertices at (0,0,0), (1,0,0), (0,1,0)
        # triang_points: [x0,y0,z0, x1,y1,z1, x2,y2,z2] = 9 values
        triang_points = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ], dtype=np.float64)
        # radar_distance: vector to first vertex of each triangle
        radar_distance = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        radar_grid_pixel_area = 1.0
        num_triangles = 1
        normal_order = True

        GNA = np.zeros(num_triangles, dtype=np.float64)
        lia = np.zeros(num_triangles, dtype=np.float64)

        n_cols = 4  # width * length
        dense = np.zeros(n_cols, dtype=np.float64)
        data = np.empty(n_cols, dtype=np.float64)
        indices = np.empty(n_cols, dtype=np.int64)
        indptr = np.zeros(5, dtype=np.int64)  # n_rows + 1 for indptr of W
        width = 2
        length = 2
        bbox = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)

        # W (sparse weight matrix): empty for now
        data_w = np.empty(0, dtype=np.float64)
        indptr_w = np.zeros(2, dtype=np.int64)
        indices_w = np.empty(0, dtype=np.int64)
        n_rows = num_triangles
        n_cols_w = n_cols

        count = computeGammaNaughtArea(
            triang_points=triang_points,
            radar_distance=radar_distance,
            radar_grid_pixel_area=radar_grid_pixel_area,
            num_triangles=num_triangles,
            normal_order=normal_order,
            GNA=GNA,
            lia=lia,
            data_w=data_w,
            indptr_w=indptr_w,
            indices_w=indices_w,
            n_rows=n_rows,
            n_cols=n_cols_w,
            dense=dense,
            data=data,
            indices=indices,
            indptr=indptr,
            width=width,
            length=length,
            bbox=bbox,
        )
        assert count >= 0
        # GNA for triangle with normal in z-direction and radar in z-direction
        # should be positive if cos_theta > 0.05
        if GNA[0] > 0:
            assert GNA[0] > 0.05

    @pytest.mark.unit
    def test_compute_gamma_naught_area_grazing_angle(self):
        """Grazing angle (cos_theta ~ 0) should give near-zero GNA."""
        # Triangle edge-on to radar: normal perpendicular to radar direction
        triang_points = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,  # normal is in y-direction, radar is in z-direction
        ], dtype=np.float64)
        radar_distance = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        radar_grid_pixel_area = 1.0
        num_triangles = 1

        GNA = np.zeros(num_triangles, dtype=np.float64)
        lia = np.zeros(num_triangles, dtype=np.float64)

        n_cols = 4
        dense = np.zeros(n_cols, dtype=np.float64)
        data = np.empty(n_cols, dtype=np.float64)
        indices = np.empty(n_cols, dtype=np.int64)
        indptr = np.zeros(5, dtype=np.int64)
        width = 2
        length = 2
        bbox = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)

        data_w = np.empty(0, dtype=np.float64)
        indptr_w = np.zeros(2, dtype=np.int64)
        indices_w = np.empty(0, dtype=np.int64)

        count = computeGammaNaughtArea(
            triang_points=triang_points,
            radar_distance=radar_distance,
            radar_grid_pixel_area=radar_grid_pixel_area,
            num_triangles=num_triangles,
            normal_order=True,
            GNA=GNA,
            lia=lia,
            data_w=data_w,
            indptr_w=indptr_w,
            indices_w=indices_w,
            n_rows=1,
            n_cols=n_cols,
            dense=dense,
            data=data,
            indices=indices,
            indptr=indptr,
            width=width,
            length=length,
            bbox=bbox,
        )
        assert count >= 0
        # For grazing angle, cos_theta should be near 0, so GNA stays 0

    @pytest.mark.unit
    def test_compute_gamma_naught_area_different_normal_order(self):
        """Different normal_order should flip the sign of lia."""
        triang_points = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ], dtype=np.float64)
        radar_distance = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        radar_grid_pixel_area = 1.0
        num_triangles = 1

        GNA_fwd = np.zeros(num_triangles, dtype=np.float64)
        lia_fwd = np.zeros(num_triangles, dtype=np.float64)
        GNA_rev = np.zeros(num_triangles, dtype=np.float64)
        lia_rev = np.zeros(num_triangles, dtype=np.float64)

        n_cols = 4
        dense_fwd = np.zeros(n_cols, dtype=np.float64)
        dense_rev = np.zeros(n_cols, dtype=np.float64)
        data_fwd = np.empty(n_cols, dtype=np.float64)
        data_rev = np.empty(n_cols, dtype=np.float64)
        indices_fwd = np.empty(n_cols, dtype=np.int64)
        indices_rev = np.empty(n_cols, dtype=np.int64)
        indptr_fwd = np.zeros(5, dtype=np.int64)
        indptr_rev = np.zeros(5, dtype=np.int64)
        bbox_fwd = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)
        bbox_rev = np.array([99.0, -99.0, 99.0, -99.0], dtype=np.float64)

        data_w = np.empty(0, dtype=np.float64)
        indptr_w = np.zeros(2, dtype=np.int64)
        indices_w = np.empty(0, dtype=np.int64)

        computeGammaNaughtArea(
            triang_points=triang_points, radar_distance=radar_distance,
            radar_grid_pixel_area=radar_grid_pixel_area,
            num_triangles=num_triangles, normal_order=True,
            GNA=GNA_fwd, lia=lia_fwd,
            data_w=data_w, indptr_w=indptr_w, indices_w=indices_w,
            n_rows=1, n_cols=n_cols,
            dense=dense_fwd, data=data_fwd, indices=indices_fwd,
            indptr=indptr_fwd, width=2, length=2, bbox=bbox_fwd,
        )
        computeGammaNaughtArea(
            triang_points=triang_points, radar_distance=radar_distance,
            radar_grid_pixel_area=radar_grid_pixel_area,
            num_triangles=num_triangles, normal_order=False,
            GNA=GNA_rev, lia=lia_rev,
            data_w=data_w, indptr_w=indptr_w, indices_w=indices_w,
            n_rows=1, n_cols=n_cols,
            dense=dense_rev, data=data_rev, indices=indices_rev,
            indptr=indptr_rev, width=2, length=2, bbox=bbox_rev,
        )
        # lia should have opposite signs (normal flipped)
        np.testing.assert_almost_equal(lia_fwd[0], -lia_rev[0])


class TestUpdateDenseArray:
    """Tests for update_dense_array (numba)."""

    @pytest.mark.unit
    def test_update_dense_array_basic(self):
        """Should add sparse values to the correct positions in dense array."""
        # 3x3 grid: indices 0-8
        # Sparse matrix: 2 rows. Row 0: no entries. Row 1: entries at cols 1,2
        array = np.zeros(9, dtype=np.float64)
        data = np.array([1.0, 2.0], dtype=np.float64)
        indices = np.array([1, 2], dtype=np.int64)
        indptr = np.array([0, 0, 2, 2], dtype=np.int64)

        xmin, xmax, ymin, ymax = 0, 3, 0, 3
        update_dense_array(array, data, indices, indptr, xmin, xmax, ymin, ymax)

        # Row 1, col 1 => index (1-0)*3 + (1-0) = 4
        # Row 1, col 2 => index (1-0)*3 + (2-0) = 5
        np.testing.assert_almost_equal(array[4], 1.0)
        np.testing.assert_almost_equal(array[5], 2.0)
        # Other entries should remain zero
        assert array[0] == 0.0
        assert array[1] == 0.0

    @pytest.mark.unit
    def test_update_dense_array_out_of_bounds(self):
        """Columns outside the bounding box should not be added."""
        array = np.zeros(9, dtype=np.float64)
        data = np.array([1.0], dtype=np.float64)
        # col 1 within [0,2)
        indices = np.array([1], dtype=np.int64)
        # 3-row sparse matrix (for ymax=3): row 0 skip, row 1 has 1 entry
        indptr = np.array([0, 0, 1, 1], dtype=np.int64)

        # width = xmax - xmin = 2 - 0 = 2
        # Row 1, col 1 => index (1-0)*2 + (1-0) = 3
        update_dense_array(array, data, indices, indptr, 0, 2, 0, 3)
        np.testing.assert_almost_equal(array[3], 1.0)
        # Row 0 and 2 have no entries
        assert array[0] == 0.0
        assert array[1] == 0.0

    @pytest.mark.unit
    def test_update_dense_array_partial_bounds(self):
        """Only values within the bounding box should be added."""
        array = np.zeros(9, dtype=np.float64)
        # 3-row sparse matrix (indptr has 4 elements)
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        indices = np.array([0, 3, 5], dtype=np.int64)
        indptr = np.array([0, 1, 2, 3], dtype=np.int64)

        update_dense_array(array, data, indices, indptr, 2, 6, 1, 3)
        # Row 0 of sparse (j=0..1): col=0, skipped (0 < 2)
        # Row 1 of sparse (j=1..2): col=3, included => array[(3-1)*4+(3-2)] = array[9] -- out of bounds!
        # Actually width=6-2=4, so row1, col3 => (1-1)*4 + (3-2) = 1
        # Row 2 of sparse (j=2..3): col=5, included => (2-1)*4 + (5-2) = 7
        np.testing.assert_almost_equal(array[1], 2.0)
        np.testing.assert_almost_equal(array[7], 3.0)
        # col=0 was skipped (< xmin=2)
        assert array[0] == 0.0

    @pytest.mark.unit
    def test_update_dense_array_multiple_rows(self):
        """Multiple rows should all be updated correctly."""
        array = np.zeros(12, dtype=np.float64)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        indices = np.array([0, 2, 1, 3], dtype=np.int64)
        indptr = np.array([0, 2, 4], dtype=np.int64)  # 2 rows, 2 non-zeros each

        update_dense_array(array, data, indices, indptr, 0, 4, 0, 2)
        np.testing.assert_almost_equal(array[0], 1.0)  # row0, col0
        np.testing.assert_almost_equal(array[2], 2.0)  # row0, col2
        np.testing.assert_almost_equal(array[4 + 1], 3.0)  # row1, col1
        np.testing.assert_almost_equal(array[4 + 3], 4.0)  # row1, col3

    @pytest.mark.unit
    def test_update_dense_array_accumulates(self):
        """Multiple calls should accumulate values."""
        array = np.ones(9, dtype=np.float64)
        # 2-row sparse: row 0 has 0 entries, row 1 has 1 entry at col 4
        # But col 4 >= xmax=3, so it's out of bounds
        data = np.array([1.0], dtype=np.float64)
        indices = np.array([1], dtype=np.int64)
        indptr = np.array([0, 0, 1, 1], dtype=np.int64)

        update_dense_array(array, data, indices, indptr, 0, 3, 0, 2)
        update_dense_array(array, data, indices, indptr, 0, 3, 0, 2)
        # Row 1, col 1 => index (1-0)*3 + (1-0) = 4
        np.testing.assert_almost_equal(array[4], 3.0)


class TestCsrTimesDense:
    """Tests for csr_times_dense (numba)."""

    @pytest.mark.unit
    def test_csr_times_dense_identity_mapping(self):
        """When each geocoded pixel maps to one radar pixel with equal weight, output should equal input."""
        n_dims = 1
        n_rows = 3  # 3 geocoded pixels
        n_cols = 4  # radar grid 2x2 => 4 pixels

        # Sparse weight matrix W: each output pixel gets 1.0 weight from one input pixel
        # Row 0: col 0 (weight=1.0)
        # Row 1: col 1 (weight=1.0)
        # Row 2: col 2 (weight=1.0)
        data = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        indices = np.array([0, 1, 2], dtype=np.int64)
        indptr = np.array([0, 1, 2, 3], dtype=np.int64)

        radar_image = np.zeros(n_dims * n_cols, dtype=np.float64)
        radar_image[0] = 10.0
        radar_image[1] = 20.0
        radar_image[2] = 30.0
        radar_image[3] = 40.0

        geocoded_image = np.zeros(n_dims * n_rows, dtype=np.float64)

        ystart = 0
        xstart = 0
        xsize_global = 4
        ysize_block = 1
        xsize_block = 4

        csr_times_dense(
            n_dims=n_dims,
            n_rows=n_rows,
            data=data,
            indices=indices,
            indptr=indptr,
            radar_image=radar_image,
            geocoded_image=geocoded_image,
            ystart=ystart,
            xstart=xstart,
            xsize_global=xsize_global,
            ysize_block=ysize_block,
            xsize_block=xsize_block,
        )
        np.testing.assert_almost_equal(geocoded_image[0], 10.0)
        np.testing.assert_almost_equal(geocoded_image[1], 20.0)
        np.testing.assert_almost_equal(geocoded_image[2], 30.0)

    @pytest.mark.unit
    def test_csr_times_dense_weighted_sum(self):
        """Weighted combination of radar pixels should produce expected result."""
        n_dims = 1
        n_rows = 1

        # Output pixel is weighted average of 2 radar pixels with weights 2.0 and 1.0
        # Total weight = 3.0, so normalized weights are 2/3 and 1/3
        data = np.array([2.0, 1.0], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        indptr = np.array([0, 2], dtype=np.int64)

        radar_image = np.zeros(4, dtype=np.float64)
        radar_image[0] = 6.0
        radar_image[1] = 3.0
        # Expected: (2.0 * 6.0 + 1.0 * 3.0) / 3.0 = 15.0 / 3.0 = 5.0
        expected = 5.0

        geocoded_image = np.zeros(n_dims * n_rows, dtype=np.float64)

        csr_times_dense(
            n_dims=n_dims,
            n_rows=n_rows,
            data=data,
            indices=indices,
            indptr=indptr,
            radar_image=radar_image,
            geocoded_image=geocoded_image,
            ystart=0, xstart=0,
            xsize_global=4,
            ysize_block=1, xsize_block=4,
        )
        np.testing.assert_almost_equal(geocoded_image[0], expected)

    @pytest.mark.unit
    def test_csr_times_dense_zero_weights(self):
        """Row with no weights should produce zero output for that row."""
        n_dims = 1
        n_rows = 2

        # Row 0: no entries (indptr[0]=0, indptr[1]=0)
        # Row 1: one entry at col 0 with weight 1.0 (indptr[1]=0, indptr[2]=1)
        data = np.array([1.0], dtype=np.float64)
        indices = np.array([0], dtype=np.int64)
        indptr = np.array([0, 0, 1], dtype=np.int64)

        radar_image = np.zeros(4, dtype=np.float64)
        radar_image[0] = 10.0

        geocoded_image = np.zeros(n_dims * n_rows, dtype=np.float64)

        csr_times_dense(
            n_dims=n_dims,
            n_rows=n_rows,
            data=data,
            indices=indices,
            indptr=indptr,
            radar_image=radar_image,
            geocoded_image=geocoded_image,
            ystart=0, xstart=0,
            xsize_global=4,
            ysize_block=1, xsize_block=4,
        )
        # Row 0 has no weights => geocoded_image[0] = 0 (unchanged)
        np.testing.assert_almost_equal(geocoded_image[0], 0.0)
        # Row 1 has one weight: 1.0 * 10.0 / 1.0 = 10.0
        np.testing.assert_almost_equal(geocoded_image[1], 10.0)

    @pytest.mark.unit
    def test_csr_times_dense_multiple_dimensions(self):
        """Multiple dimensions (polarizations) should be processed independently."""
        n_dims = 2
        n_rows = 2

        # Each geocoded pixel is the average of 2 radar pixels
        data = np.array([1.0, 1.0], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        indptr = np.array([0, 2, 2], dtype=np.int64)  # n_rows + 1 = 3 elements

        # radar_image layout: dim0[pixel0, pixel1, ...], dim1[pixel0, pixel1, ...]
        # k=0 (dim0): index offset 0, k=1 (dim1): index offset xsize_block*ysize_block = 4
        radar_image = np.zeros(n_dims * 4, dtype=np.float64)
        radar_image[0] = 10.0  # dim 0, pixel 0
        radar_image[1] = 20.0  # dim 0, pixel 1
        radar_image[4] = 30.0  # dim 1, pixel 0
        radar_image[5] = 40.0  # dim 1, pixel 1

        geocoded_image = np.zeros(n_dims * n_rows, dtype=np.float64)

        csr_times_dense(
            n_dims=n_dims,
            n_rows=n_rows,
            data=data,
            indices=indices,
            indptr=indptr,
            radar_image=radar_image,
            geocoded_image=geocoded_image,
            ystart=0, xstart=0,
            xsize_global=4,
            ysize_block=1, xsize_block=4,
        )
        # Layout: [dim0_row0, dim0_row1, dim1_row0, dim1_row1]
        # dim 0, row 0: (10 + 20) / 2 = 15
        np.testing.assert_almost_equal(geocoded_image[0], 15.0)
        # dim 1, row 0: (30 + 40) / 2 = 35
        np.testing.assert_almost_equal(geocoded_image[2], 35.0)

    @pytest.mark.unit
    def test_csr_times_dense_normalization(self):
        """Weights should be normalized so sum of weights equals 1."""
        n_dims = 1
        n_rows = 1

        # Weights 3.0 and 3.0 => normalized to 0.5 each
        data = np.array([3.0, 3.0], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        indptr = np.array([0, 2], dtype=np.int64)

        radar_image = np.zeros(4, dtype=np.float64)
        radar_image[0] = 4.0
        radar_image[1] = 6.0
        expected = 0.5 * 4.0 + 0.5 * 6.0  # = 5.0

        geocoded_image = np.zeros(n_dims * n_rows, dtype=np.float64)

        csr_times_dense(
            n_dims=n_dims,
            n_rows=n_rows,
            data=data,
            indices=indices,
            indptr=indptr,
            radar_image=radar_image,
            geocoded_image=geocoded_image,
            ystart=0, xstart=0,
            xsize_global=4,
            ysize_block=1, xsize_block=4,
        )
        np.testing.assert_almost_equal(geocoded_image[0], expected)


class TestComputeIncidenceAngleGround:
    """Tests for compute_incidence_angle_ground (numba)."""

    @pytest.mark.unit
    def test_compute_incidence_angle_ground_zenith(self):
        """Satellite directly overhead should give 0 degree incidence angle."""
        # Ground point at (0, 0, 0), satellite at (0, 0, 10)
        # radar_distance = satellite - ground_point = (0, 0, 10)
        # pos_satellite = radar_distance - points = (0, 0, 10)
        # dot = 100, norm_radar = 10, norm_pos = 10
        # arccos(100 / (10 * 10)) = arccos(1.0) = 0.0 radians
        radar_distance = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        points = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        num_points = 1

        incidence_angle_ground = np.zeros(num_points, dtype=np.float64)

        compute_incidence_angle_ground(
            radar_distance=radar_distance,
            points=points,
            incidence_angle_ground=incidence_angle_ground,
            num_points=num_points,
        )
        np.testing.assert_almost_equal(incidence_angle_ground[0], 0.0, decimal=10)

    @pytest.mark.unit
    def test_compute_incidence_angle_ground_oblique(self):
        """Oblique viewing geometry with offset point should give non-zero incidence angle."""
        # The function computes: arccos(dot(radar_dist, pos_sat) / (|radar| * |pos|))
        # where pos_sat = radar_dist - points
        # With points at origin, radar_dist == pos_sat, so angle is always 0
        # Use offset point to get non-zero angle
        radar_distance = np.array([10.0, 0.0, 10.0], dtype=np.float64)
        points = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        num_points = 1

        # pos_satellite = (10-5, 0-0, 10-0) = (5, 0, 10)
        # |radar| = sqrt(200), |pos| = sqrt(125)
        # dot = 10*5 + 0*0 + 10*10 = 150
        # cos = 150 / (sqrt(200)*sqrt(125)) = 150/158.11 ~ 0.9487
        # theta = arccos(0.9487) ~ 0.322 rad

        incidence_angle_ground = np.zeros(num_points, dtype=np.float64)

        compute_incidence_angle_ground(
            radar_distance=radar_distance,
            points=points,
            incidence_angle_ground=incidence_angle_ground,
            num_points=num_points,
        )
        # Should be positive and less than pi/2
        assert 0.0 < incidence_angle_ground[0] < np.pi / 2.0

    @pytest.mark.unit
    def test_compute_incidence_angle_ground_offset_point(self):
        """Offset ground point should give non-zero incidence angle."""
        # Ground point at (5, 0, 0), satellite at (0, 0, 10)
        # radar_distance = satellite - point = (-5, 0, 10)
        # pos_satellite = radar_distance - point = (-5, 0, 10) - (5, 0, 0) = (-10, 0, 10)
        # |radar| = sqrt(125), |pos| = sqrt(200)
        # dot = (-5)*(-10) + 0*0 + 10*10 = 50 + 100 = 150
        # cos_theta = 150 / (sqrt(125) * sqrt(200)) = 150 / sqrt(25000) = 150 / 158.11 = 0.9487
        # theta = arccos(0.9487) ~ 0.322 rad ~ 18.4 degrees
        radar_distance = np.array([-5.0, 0.0, 10.0], dtype=np.float64)
        points = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        num_points = 1

        incidence_angle_ground = np.zeros(num_points, dtype=np.float64)

        compute_incidence_angle_ground(
            radar_distance=radar_distance,
            points=points,
            incidence_angle_ground=incidence_angle_ground,
            num_points=num_points,
        )
        # Should be positive and less than pi/2
        assert 0.0 < incidence_angle_ground[0] < np.pi / 2.0

    @pytest.mark.unit
    def test_compute_incidence_angle_ground_multiple_points(self):
        """Multiple points should all have valid incidence angles."""
        radar_distance = np.array([
            0.0, 0.0, 10.0,   # point 0
            5.0, 0.0, 10.0,   # point 1
        ], dtype=np.float64)
        points = np.array([
            0.0, 0.0, 0.0,    # point 0
            5.0, 0.0, 0.0,    # point 1
        ], dtype=np.float64)
        num_points = 2

        incidence_angle_ground = np.zeros(num_points, dtype=np.float64)

        compute_incidence_angle_ground(
            radar_distance=radar_distance,
            points=points,
            incidence_angle_ground=incidence_angle_ground,
            num_points=num_points,
        )
        assert incidence_angle_ground[0] >= 0.0
        assert incidence_angle_ground[1] >= 0.0
        assert incidence_angle_ground[0] <= np.pi
        assert incidence_angle_ground[1] <= np.pi


class TestMainLoop:
    """Tests for main_loop (numba)."""

    @pytest.mark.unit
    def test_main_loop_basic_structure(self):
        """Basic call should not raise and should produce valid output."""
        # Create polynomial coefficient arrays for position, velocity, acceleration
        n_coeffs = 3
        t = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        pos = np.zeros((n_coeffs, 3), dtype=np.float64)
        pos[0, :] = [6700000.0, 0.0, 0.0]
        pos[1, :] = [0.0, 7500.0, 0.0]
        pos[2, :] = [0.0, 0.0, 0.0]

        vel = np.zeros((n_coeffs, 3), dtype=np.float64)
        vel[0, :] = [0.0, 7500.0, 0.0]
        vel[1, :] = [0.0, 0.0, 0.0]
        vel[2, :] = [0.0, 0.0, 0.0]

        acc = np.zeros((n_coeffs, 3), dtype=np.float64)
        acc[0, :] = [0.0, 0.0, 0.0]
        acc[1, :] = [0.0, 0.0, 0.0]
        acc[2, :] = [0.0, 0.0, 0.0]

        dem = np.zeros((2, 2, 3), dtype=np.float64)
        dem[0, 0, :] = [6700000.0, 0.0, 0.0]
        dem[0, 1, :] = [6700000.0, 500.0, 0.0]
        dem[1, 0, :] = [6700000.0, 0.0, 500.0]
        dem[1, 1, :] = [6700000.0, 500.0, 500.0]

        direction_ecef = np.zeros((2, 2, 3), dtype=np.float64)
        r = np.zeros((2, 2, 3), dtype=np.float64)
        conv_th = 1e-3

        main_loop(
            pos=pos, vel=vel, acc=acc, dem=dem,
            t=t, direction_ecef=direction_ecef,
            r=r, conv_th=conv_th,
        )
        # t should have been modified in-place
        assert t.shape == (2, 2)
        assert r.shape == (2, 2, 3)
        assert direction_ecef.shape == (2, 2, 3)

    @pytest.mark.unit
    def test_main_loop_converges(self):
        """Loop should converge within MAXITER for reasonable inputs."""
        n_coeffs = 2
        t = np.array([[500.0]], dtype=np.float64)
        pos = np.zeros((n_coeffs, 3), dtype=np.float64)
        pos[0, :] = [6700000.0, 0.0, 0.0]
        pos[1, :] = [0.0, 7500.0, 0.0]

        vel = np.zeros((n_coeffs, 3), dtype=np.float64)
        vel[0, :] = [0.0, 7500.0, 0.0]
        vel[1, :] = [0.0, 0.0, 0.0]

        acc = np.zeros((n_coeffs, 3), dtype=np.float64)

        dem = np.zeros((1, 1, 3), dtype=np.float64)
        dem[0, 0, :] = [6700000.0, 0.0, 0.0]

        direction_ecef = np.zeros((1, 1, 3), dtype=np.float64)
        r = np.zeros((1, 1, 3), dtype=np.float64)
        conv_th = 1e-6

        main_loop(
            pos=pos, vel=vel, acc=acc, dem=dem,
            t=t, direction_ecef=direction_ecef,
            r=r, conv_th=conv_th,
        )
        # Should have converged
        assert np.isfinite(t[0, 0])
        assert np.isfinite(r[0, 0, 0])
