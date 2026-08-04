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


class TestCostFunctionL2:
    """Tests for cost_function_l2."""

    def test_cost_function_l2_basic(self):
        """Basic L2 cost function should return scalar."""
        params = np.array([1.0, 2.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        y = np.array([3.0, 5.0, 7.0])
        result = cost_function_l2(params, X, y)
        assert isinstance(result, (float, np.floating))

    def test_cost_function_l2_zero_error(self):
        """Perfect fit should give zero cost."""
        X = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        # X.dot(params) for params=[2.0, 1.0] = [1*2+1*1, 2*2+1*1, 3*2+1*1] = [3, 5, 7]
        y = np.array([3.0, 5.0, 7.0])
        params = np.array([2.0, 1.0])
        result = cost_function_l2(params, X, y)
        np.testing.assert_almost_equal(result, 0.0)

    def test_cost_function_l2_wrong_shape(self):
        """Mismatched shapes should raise assertion."""
        with pytest.raises(AssertionError):
            cost_function_l2(np.array([1.0]), np.ones((3, 2)), np.array([1.0, 2.0]))


class TestCostFunctionL1:
    """Tests for cost_function_l1."""

    def test_cost_function_l1_basic(self):
        """Basic L1 cost function should return scalar."""
        params = np.array([1.0, 2.0])
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        y = np.array([3.0, 5.0])
        result = cost_function_l1(params, X, y)
        assert isinstance(result, (float, np.floating))

    def test_cost_function_l1_zero_error(self):
        """Perfect fit should give zero cost."""
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        # X.dot(params) for params=[2.0, 1.0] = [1*2+1*1, 2*2+1*1] = [3, 5]
        y = np.array([3.0, 5.0])
        params = np.array([2.0, 1.0])
        result = cost_function_l1(params, X, y)
        np.testing.assert_almost_equal(result, 0.0)


class TestFit:
    """Tests for fit."""

    def test_fit_linear(self):
        """Linear model should give X @ params."""
        X = np.array([[1.0, 1.0], [2.0, 1.0]])
        # X.dot([2.0, 1.0]) = [1*2+1*1, 2*2+1*1] = [3, 5]
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
        """Mismatched params should raise assertion."""
        with pytest.raises(AssertionError):
            fit(np.ones((3, 2)), np.array([1.0]))


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
        # Columns: region_constant, x1, x2
        assert result.shape[1] >= 3

    def test_get_model_matrix_with_regions(self):
        """Regions should create separate columns."""
        x = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        regions = np.array([0, 1, 0])
        result = get_model_matrix(x, regions=regions, degree=1)
        assert result.shape[1] >= 4  # 2 regions + 2 polynomial terms

    def test_get_model_matrix_quadratic(self):
        """Degree 2 should include quadratic terms."""
        x = np.array([[1.0, 0.0], [0.0, 1.0]])
        result_deg2 = get_model_matrix(x, degree=2)
        result_deg1 = get_model_matrix(x, degree=1)
        # Degree 2 should have more columns than degree 1
        assert result_deg2.shape[1] > result_deg1.shape[1]


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
        y = np.array([1.0, 2.0, 3.0, 4.0])  # linear relationship
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
        """Should converge for simple data."""
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w = np.ones_like(y)
        result = estimate_global_surface(x, y, w, degree=1, maximum_iterations=100)
        assert result.success or result.fun < 1e-3

    def test_estimate_global_surface_float_weigths(self):
        """Degree 0 should fit a constant (mean)."""
        x = np.array([[1.0], [1.0], [1.0], [1.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = 1.0
        result = estimate_global_surface(x, y, w, degree=0, maximum_iterations=50)
        assert isinstance(result, scipy.optimize.OptimizeResult)


class TestFit2dPoly:
    """Tests for fit_2d_poly."""

    def test_fit_2d_poly_returns_tuple(self):
        """Should return a tuple of (result, scale, center)."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)

    def test_fit_2d_poly_simple(self):
        """Simple case should produce valid output."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        data = np.array([0.0, 1.0, 1.0, 2.0])
        result = fit_2d_poly(x, y, data, degree=1)
        assert isinstance(result, tuple)
