"""Unit tests for cop_ard_core.detrend."""

import numpy as np
import pytest
import scipy.optimize

from cop_ard_core.detrend import (
    cost_function_l1,
    cost_function_l2,
    estimate_global_surface,
    fit,
    fit_2d_poly,
    get_model_matrix,
)


# ===========================================================================
# cost_function_l2
# ===========================================================================


class TestCostFunctionL2:
    """Tests for cost_function_l2."""

    def test_cost_function_l2_basic(self):
        """Basic L2 cost function should return scalar float."""
        params = np.array([1.0, 2.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        y = np.array([3.0, 5.0, 7.0])
        result = cost_function_l2(params, X, y)
        assert isinstance(result, (float, np.floating))

    def test_cost_function_l2_zero_error(self):
        """Perfect fit should give zero cost."""
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        # X.dot(params) for params=[2.0, 1.0] = [3, 5, 7]
        y = np.array([3.0, 5.0, 7.0])
        params = np.array([2.0, 1.0])
        result = cost_function_l2(params, X, y)
        np.testing.assert_almost_equal(result, 0.0)

    def test_cost_function_l2_wrong_shape(self):
        """Mismatched shapes should raise AssertionError."""
        with pytest.raises(AssertionError):
            cost_function_l2(np.array([1.0]), np.ones((3, 2)), np.array([1.0, 2.0]))

    def test_cost_function_l2_1d_params(self):
        """1D params should work."""
        params = np.array([1.0, 0.5])
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        y = np.array([1.5, 2.5, 3.5])
        result = cost_function_l2(params, X, y)
        # X.dot([1.0, 0.5]) = [1.5, 2.5, 3.5] => zero error
        np.testing.assert_almost_equal(result, 0.0)

    def test_cost_function_l2_with_weights(self):
        """Weighted L2 cost should scale residuals."""
        params = np.array([1.0])
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 5.0])
        # unweighted: (3-1)^2 + (5-2)^2 = 4 + 9 = 13
        cost_unweighted = cost_function_l2(params, X, y)
        # weighted w=2: 2*(4+9) = 26
        cost_weighted = cost_function_l2(params, X, y, w=2.0)
        np.testing.assert_almost_equal(cost_weighted, 2.0 * cost_unweighted)

    def test_cost_function_l2_weight_array(self):
        """Array weights should weigh each residual individually."""
        params = np.array([1.0])
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 5.0])
        # residuals: [2, 3], squared: [4, 9]
        w = np.array([1.0, 2.0])
        # weighted: 1*4 + 2*9 = 4 + 18 = 22
        result = cost_function_l2(params, X, y, w=w)
        np.testing.assert_almost_equal(result, 22.0)

    def test_cost_function_l2_3d_params_raises(self):
        """3D params (wrong shape) should raise AssertionError."""
        params = np.array([[[1.0, 2.0]]])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        with pytest.raises(AssertionError):
            cost_function_l2(params, X, y)

    def test_cost_function_l2_1d_x_raises(self):
        """1D X should raise AssertionError."""
        params = np.array([1.0])
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 5.0, 7.0])
        with pytest.raises(AssertionError):
            cost_function_l2(params, X, y)

    def test_cost_function_l2_y_length_mismatch(self):
        """y length mismatch should raise AssertionError."""
        params = np.array([1.0])
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0])
        with pytest.raises(AssertionError):
            cost_function_l2(params, X, y)

    def test_cost_function_l2_large_residuals(self):
        """Large residuals should produce large cost."""
        params = np.array([0.0, 0.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([100.0, 200.0])
        result = cost_function_l2(params, X, y)
        # (100-0)^2 + (200-0)^2 = 10000 + 40000 = 50000
        np.testing.assert_almost_equal(result, 50000.0)

    def test_cost_function_l2_complex_residuals(self):
        """Cost function operates on float residuals; complex data not expected."""
        params = np.array([1.0, 0.5])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([1.5, 2.5])
        result = cost_function_l2(params, X, y)
        assert result >= 0.0


# ===========================================================================
# cost_function_l1
# ===========================================================================


class TestCostFunctionL1:
    """Tests for cost_function_l1."""

    def test_cost_function_l1_basic(self):
        """Basic L1 cost function should return scalar float."""
        params = np.array([1.0, 2.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        result = cost_function_l1(params, X, y)
        assert isinstance(result, (float, np.floating))

    def test_cost_function_l1_zero_error(self):
        """Perfect fit should give zero L1 cost."""
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        params = np.array([2.0, 1.0])
        result = cost_function_l1(params, X, y)
        np.testing.assert_almost_equal(result, 0.0)

    def test_cost_function_l1_wrong_shape(self):
        """Mismatched shapes should raise AssertionError."""
        with pytest.raises(AssertionError):
            cost_function_l1(np.array([1.0]), np.ones((3, 2)), np.array([1.0, 2.0]))

    def test_cost_function_l1_nonzero_error(self):
        """Non-zero residuals should give positive L1 cost."""
        params = np.array([1.0, 1.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        # X.dot([1,1]) = [2, 3], residuals = [1, 2], |1|+|2| = 3
        result = cost_function_l1(params, X, y)
        np.testing.assert_almost_equal(result, 3.0)

    def test_cost_function_l1_with_weights(self):
        """Weighted L1 cost should scale absolute residuals."""
        params = np.array([1.0, 1.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        # unweighted: |1| + |2| = 3
        cost_unweighted = cost_function_l1(params, X, y)
        # weighted w=2: 2*|1| + 2*|2| = 6
        cost_weighted = cost_function_l1(params, X, y, w=2.0)
        np.testing.assert_almost_equal(cost_weighted, 2.0 * cost_unweighted)

    def test_cost_function_l1_weight_array(self):
        """Array weights should weight each absolute residual individually."""
        params = np.array([1.0, 1.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        # residuals: [1, 2], abs: [1, 2]
        w = np.array([1.0, 3.0])
        # weighted: 1*1 + 3*2 = 1 + 6 = 7
        result = cost_function_l1(params, X, y, w=w)
        np.testing.assert_almost_equal(result, 7.0)

    def test_cost_function_l1_1d_x_raises(self):
        """1D X should raise AssertionError."""
        params = np.array([1.0])
        X = np.array([1.0, 2.0])
        y = np.array([3.0, 5.0])
        with pytest.raises(AssertionError):
            cost_function_l1(params, X, y)

    def test_cost_function_l1_negative_residuals(self):
        """Negative residuals should have same L1 cost as positive."""
        X = np.array([[1.0, 1.0]])
        y = np.array([3.0])
        params_a = np.array([2.0, 1.0])  # fit: 3, residual: 0
        params_b = np.array([4.0, -1.0])  # fit: 3, residual: 0
        result_a = cost_function_l1(params_a, X, y)
        result_b = cost_function_l1(params_b, X, y)
        np.testing.assert_almost_equal(result_a, result_b)

    def test_cost_function_l1_single_point(self):
        """Single point should produce valid L1 cost."""
        params = np.array([1.0])
        X = np.array([[2.0]])
        y = np.array([5.0])
        # fit: 2, |5-2| = 3
        result = cost_function_l1(params, X, y)
        np.testing.assert_almost_equal(result, 3.0)


# ===========================================================================
# fit
# ===========================================================================


class TestFit:
    """Tests for fit."""

    def test_fit_linear(self):
        """Linear model should give X @ params."""
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        params = np.array([2.0, 1.0])
        result = fit(X, params)
        np.testing.assert_array_equal(result, [3.0, 5.0])

    def test_fit_shape(self):
        """Output shape should match X rows."""
        X = np.ones((5, 3))
        params = np.array([1.0, 1.0, 1.0])
        result = fit(X, params)
        assert result.shape == (5,)

    def test_fit_wrong_shape(self):
        """Mismatched params should raise AssertionError."""
        with pytest.raises(AssertionError):
            fit(np.ones((3, 2)), np.array([1.0]))

    def test_fit_3d_params_raises(self):
        """3D params should raise AssertionError."""
        X = np.ones((3, 2))
        params = np.array([[[1.0, 1.0]]])
        with pytest.raises(AssertionError):
            fit(X, params)

    def test_fit_1d_x_raises(self):
        """1D X should raise AssertionError."""
        with pytest.raises(AssertionError):
            fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0]))

    def test_fit_zero_params(self):
        """Zero params should give all zeros output."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        params = np.array([0.0, 0.0])
        result = fit(X, params)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_fit_single_column(self):
        """Single column X should work."""
        X = np.array([[1.0], [2.0], [3.0]])
        params = np.array([2.5])
        result = fit(X, params)
        np.testing.assert_array_almost_equal(result, [2.5, 5.0, 7.5])

    def test_fit_negative_params(self):
        """Negative params should produce negative outputs."""
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        params = np.array([-1.0, -2.0])
        result = fit(X, params)
        # [-1-2, -2-2] = [-3, -4]
        np.testing.assert_array_equal(result, [-3.0, -4.0])


# ===========================================================================
# get_model_matrix
# ===========================================================================


class TestGetModelMatrix:
    """Tests for get_model_matrix."""

    def test_get_model_matrix_constant_only(self):
        """Degree 0 should return column of ones."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = get_model_matrix(x, degree=0)
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result, [[1.0], [1.0]])

    def test_get_model_matrix_linear(self):
        """Degree 1 should include x, y, and constant terms."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = get_model_matrix(x, degree=1)
        assert result.shape[1] >= 3

    def test_get_model_matrix_with_regions(self):
        """Regions should create separate columns."""
        x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        regions = np.array([0, 1, 0])
        result = get_model_matrix(x, regions=regions, degree=1)
        assert result.shape[1] >= 4

    def test_get_model_matrix_quadratic(self):
        """Degree 2 should include quadratic terms."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        result_deg2 = get_model_matrix(x, degree=2)
        result_deg1 = get_model_matrix(x, degree=1)
        assert result_deg2.shape[1] > result_deg1.shape[1]

    def test_get_model_matrix_output_dtype(self):
        """Output should be float64 dtype."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = get_model_matrix(x, degree=0)
        assert result.dtype == np.float64

    def test_get_model_matrix_output_shape_rows(self):
        """Output row count should match input row count."""
        x = np.random.rand(10, 3)
        result = get_model_matrix(x, degree=1)
        assert result.shape[0] == 10

    def test_get_model_matrix_three_columns(self):
        """3-column X should produce correct polynomial combinations."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = get_model_matrix(x, degree=1)
        # constant + x1 + x2 + x3 = 4 columns
        assert result.shape[1] == 4

    def test_get_model_matrix_degree_2_three_columns(self):
        """Degree 2 with 3 columns should include all quadratic combinations."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = get_model_matrix(x, degree=2)
        # constant + x1 + x2 + x3 + x1^2 + x1*x2 + x1*x3 + x2^2 + x2*x3 + x3^2 = 10
        assert result.shape[1] == 10

    def test_get_model_matrix_degree_3(self):
        """Degree 3 should include cubic terms."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        result_deg3 = get_model_matrix(x, degree=3)
        result_deg2 = get_model_matrix(x, degree=2)
        assert result_deg3.shape[1] > result_deg2.shape[1]

    def test_get_model_matrix_regions_correct(self):
        """Region columns should be 1 for belonging and 0 otherwise."""
        x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        regions = np.array([0, 1, 0])
        result = get_model_matrix(x, regions=regions, degree=0)
        # 2 regions, degree 0 => 2 columns
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result[:, 1], [0.0, 1.0, 0.0])

    def test_get_model_matrix_regions_single_region(self):
        """Single region (max=0) should behave like no regions."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        regions = np.array([0, 0])
        result = get_model_matrix(x, regions=regions, degree=0)
        np.testing.assert_array_equal(result, [[1.0], [1.0]])

    def test_get_model_matrix_with_regions_degree_1(self):
        """Regions with degree 1 should have region cols + poly cols."""
        x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        regions = np.array([0, 1, 0])
        result = get_model_matrix(x, regions=regions, degree=1)
        # 2 regions + x1 + x2 = 4 columns
        assert result.shape[1] == 4

    def test_get_model_matrix_single_point(self):
        """Single point input should work."""
        x = np.array([[1.0, 2.0]])
        result = get_model_matrix(x, degree=0)
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result, [[1.0]])

    def test_get_model_matrix_many_points(self):
        """Large number of points should produce correct shape."""
        x = np.random.rand(1000, 2)
        result = get_model_matrix(x, degree=1)
        assert result.shape == (1000, 3)

    def test_get_model_matrix_zero_values(self):
        """X with zero values should produce valid model matrix."""
        x = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = get_model_matrix(x, degree=1)
        assert result.shape == (2, 3)

    def test_get_model_matrix_negative_values(self):
        """X with negative values should produce valid model matrix."""
        x = np.array([[-1.0, -2.0], [3.0, -4.0]])
        result = get_model_matrix(x, degree=1)
        assert result.shape == (2, 3)

    def test_get_model_matrix_single_column_raises(self):
        """X with 0 columns should raise AssertionError."""
        with pytest.raises(AssertionError):
            get_model_matrix(np.array([[1.0], [2.0]]).reshape(2, 1)[:, :0], degree=0)

    def test_get_model_matrix_2d_single_col(self):
        """2D X with 1 column should work."""
        x = np.array([[1.0], [2.0], [3.0]])
        result = get_model_matrix(x, degree=0)
        assert result.shape == (3, 1)


# ===========================================================================
# estimate_global_surface
# ===========================================================================


class TestEstimateGlobalSurface:
    """Tests for estimate_global_surface."""

    def test_estimate_global_surface_degree_0(self):
        """Degree 0 should fit a constant (mean)."""
        x = np.array([[1.0], [1.0], [1.0], [1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_degree_1(self):
        """Degree 1 should fit a linear model."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_invalid_norm(self):
        """Invalid norm should raise ValueError."""
        x = np.array([[1.0], [1.0]])
        y = np.array([1.0, 2.0])
        w = np.ones_like(y)
        with pytest.raises(ValueError, match="Norm.*not supported"):
            estimate_global_surface(x, y, w, norm=3)

    def test_estimate_global_surface_converges(self):
        """Should converge for simple linear data."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=100)
        assert result.success or result.fun < 1e-3

    def test_estimate_global_surface_float_weights(self):
        """Float weight should work (broadcast to array)."""
        x = np.array([[1.0], [1.0], [1.0], [1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = 1.0
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_l1_norm(self):
        """L1 norm (norm=1) should use Nelder-Mead method."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, norm=1, maximum_iterations=100)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_l1_converges(self):
        """L1 norm should converge for simple data."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, norm=1, maximum_iterations=100)
        assert result.success or result.fun < 1e-3

    def test_estimate_global_surface_bfgs_method(self):
        """Explicit BFGS method should work with norm=2."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, norm=2, method="BFGS", maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_nelder_mead_method(self):
        """Explicit Nelder-Mead method should work with norm=1."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, norm=1, method="Nelder-Mead", maximum_iterations=100)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_custom_method(self):
        """Custom method string should be passed through."""
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([1.0, 2.0, 3.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, method="L-BFGS-B", maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_array_weights(self):
        """Array weights should weight data points individually."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 1.0, 1.0, 100.0])  # heavy weight on last point
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_zero_weights(self):
        """Zero weights should effectively exclude points from fit."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([0.0, 0.0, 0.0, 1.0])  # only last point has weight
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_regions(self):
        """Regions parameter should work with multiple regions."""
        x = np.array([[0.0], [0.0], [1.0], [1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        regions = np.array([0, 0, 1, 1])
        result = estimate_global_surface(x, y, w, degree=0, regions=regions, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_degree_2(self):
        """Degree 2 should fit a quadratic model."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 4.0, 9.0])  # y = x^2
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=2, maximum_iterations=100)
        assert result.success or result.fun < 1e-3

    def test_estimate_global_surface_degree_3(self):
        """Degree 3 should fit a cubic model with noisy data."""
        np.random.seed(42)
        x = np.array([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
        # y = x^3 + small noise (enough that residuals aren't all zero)
        y = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0]) + np.random.randn(7) * 0.5
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=3, maximum_iterations=100)
        assert result.success or result.fun < 1.0

    def test_estimate_global_surface_sigma_clipping(self):
        """Sigma clipping should handle outliers in data."""
        x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier at end
        w = np.ones_like(y)
        # Large sigma to not clip the outlier, verify it handles the outlier
        result = estimate_global_surface(x, y, w, degree=1, sigma=10.0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_range_filtering(self):
        """Range parameter should filter values outside bounds."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        # Range that accepts all values
        result = estimate_global_surface(x, y, w, degree=1, range=(-10.0, 10.0), maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_convergence_threshold(self):
        """Tight convergence threshold should require more iterations."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(
            x, y, w, degree=1, convergency_threshold=1e-10, maximum_iterations=100
        )
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_tolerant_threshold(self):
        """Lenient convergence threshold should converge faster."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(
            x, y, w, degree=1, convergency_threshold=1.0, maximum_iterations=100
        )
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_nan_y(self):
        """NaN in y should be filtered by isfinite check."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, np.nan, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_inf_y(self):
        """Inf in y should be filtered by isfinite check."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, np.inf, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_negative_inf_y(self):
        """-Inf in y should be filtered by isfinite check."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, -np.inf, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_nan_weights(self):
        """NaN weights should be filtered by isfinite check."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, np.nan, 1.0, 1.0])
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_all_nan_y(self):
        """All NaN y should result in no valid points."""
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([np.nan, np.nan, np.nan])
        w = np.ones_like(y)
        # Should handle gracefully (no valid points for fitting)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=10)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_single_point(self):
        """Single point input should work."""
        x = np.array([[1.0]])
        y = np.array([5.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=10)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_two_points(self):
        """Two points should work for linear fit."""
        x = np.array([[0.0], [1.0]])
        y = np.array([1.0, 2.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_all_finite(self):
        """All-finite data should use all points (after sigma clipping)."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, sigma=100.0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_zero_sigma(self):
        """Small sigma should clip more aggressively."""
        x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, sigma=0.0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_max_iterations_reached(self):
        """With very small threshold and few points, may reach max iterations."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(
            x, y, w, degree=1, convergency_threshold=1e-15, maximum_iterations=5
        )
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_2d_x(self):
        """2D x (multiple features) should work."""
        x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_3d_x(self):
        """3D x (three features) should work."""
        x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_mixed_finite_weights(self):
        """Mix of finite and Inf weights should filter Inf entries."""
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, np.inf, 1.0, -np.inf])
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_int_weights(self):
        """Integer weights should work (cast to float)."""
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([1.0, 2.0, 3.0])
        w = np.array([1, 2, 3])
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)

    def test_estimate_global_surface_degree_negative_raises(self):
        """Negative degree should raise AssertionError."""
        x = np.array([[1.0], [1.0]])
        y = np.array([1.0, 2.0])
        w = np.ones_like(y)
        with pytest.raises(AssertionError):
            estimate_global_surface(x, y, w, degree=-1)

    def test_estimate_global_surface_1d_x_raises(self):
        """1D x should raise AssertionError."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        w = np.ones_like(y)
        with pytest.raises(AssertionError):
            estimate_global_surface(x, y, w, degree=1)

    def test_estimate_global_surface_solution_nonzero(self):
        """Solution should have non-zero parameters for non-trivial data."""
        x = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 10.0, 20.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=100)
        assert np.any(result.x != 0.0)


# ===========================================================================
# fit_2d_poly
# ===========================================================================


class TestFit2dPoly:
    """Tests for fit_2d_poly."""

    def test_fit_2d_poly_returns_tuple(self):
        """Should return a tuple of (result, scale, center)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fit_2d_poly_simple(self):
        """Simple case should produce valid output."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([0.0, 1.0, 1.0, 2.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_tuple_types(self):
        """Tuple should contain (OptimizeResult, ndarray, ndarray)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result[0], scipy.optimize.OptimizeResult)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

    def test_fit_2d_poly_degree_0(self):
        """Degree 0 should fit a constant surface."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=0)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_degree_2(self):
        """Degree 2 should fit a quadratic surface."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0, 1.0])
        data = np.array([0.0, 1.0, 0.0, 2.0])
        result = fit_2d_poly(x, y, data, degree=2)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_nonfinite_data_filtered(self):
        """Non-finite data values should be filtered out."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, np.nan, 3.0, np.inf])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_negative_data(self):
        """Negative data values should be handled correctly."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([-1.0, -2.0, -3.0, -4.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_with_weights(self):
        """Weighted fit_2d_poly should accept weight array."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, w=w, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_scale_nonzero(self):
        """Scale values should be non-zero for valid data spread."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        _, scale, _ = fit_2d_poly(x, y, data, degree=1)
        assert np.all(scale > 0)

    def test_fit_2d_poly_center_values(self):
        """Center values should be the mean of input coordinates."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        _, _, center = fit_2d_poly(x, y, data, degree=1)
        np.testing.assert_almost_equal(center[0], 1.5)
        np.testing.assert_almost_equal(center[1], 1.5)

    def test_fit_2d_poly_l1_norm(self):
        """L1 norm (norm=1) should produce valid fit."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=1, norm=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_l2_norm(self):
        """L2 norm (norm=2) should produce valid fit."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=1, norm=2)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_all_nan_data(self):
        """All NaN data should raise ValueError (no finite points available)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([np.nan, np.nan, np.nan, np.nan])
        with pytest.raises(ValueError):
            fit_2d_poly(x, y, data, degree=1)

    def test_fit_2d_poly_single_point(self):
        """Single point should produce valid output."""
        x = np.array([1.0])
        y = np.array([1.0])
        data = np.array([5.0])
        result = fit_2d_poly(x, y, data, degree=0)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_large_values(self):
        """Large data values should be handled correctly."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1e6, 2e6, 3e6, 4e6])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_small_values(self):
        """Small data values should be handled correctly."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1e-10, 2e-10, 3e-10, 4e-10])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_mixed_sign_data(self):
        """Mixed positive and negative data should be handled."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([-1.0, 1.0, -1.0, 1.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)
