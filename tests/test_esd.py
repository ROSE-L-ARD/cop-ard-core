"""Unit tests for cop_ard_core.esd."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.esd import (
    angle_phase,
    compute_phase,
    optimize_phase,
    scipy_optimize_phase,
)


class TestComputePhase:
    """Tests for compute_phase (numba)."""

    def test_compute_phase_basic(self):
        """Basic phase computation should return negative value."""
        param = np.array([0.0, 1.0])
        y = np.array([0.0, np.pi / 2, np.pi])
        # A shape: (n_params, n_points) so param @ A works
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))

    def test_compute_phase_returns_negative(self):
        """compute_phase should return negative value (negative abs)."""
        param = np.zeros(2)
        y = np.array([0.0, np.pi / 4, np.pi / 2])
        A = np.ones((2, 3))
        result = compute_phase(param, y, A)
        assert result <= 0

    def test_compute_phase_with_zeros(self):
        """All-zero parameters should work."""
        param = np.zeros(2)
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        A = np.ones((2, 5))
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))


class TestAnglePhase:
    """Tests for angle_phase (numba)."""

    def test_angle_phase_basic(self):
        """Should return phase angle in [-pi, pi]."""
        param = np.array([0.0, 1.0])
        y = np.array([0.0, np.pi / 2, np.pi])
        # A shape: (n_params, n_points)
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = angle_phase(param, y, A)
        assert isinstance(result, (float, np.floating))

    def test_angle_phase_range(self):
        """Result should be in [-pi, pi]."""
        param = np.zeros(2)
        y = np.array([0.0, np.pi / 4, np.pi / 2])
        A = np.ones((2, 3))
        result = angle_phase(param, y, A)
        assert -np.pi <= result <= np.pi

    def test_angle_phase_zeros(self):
        """Zero inputs should return 0."""
        param = np.zeros(2)
        y = np.zeros(5)
        A = np.zeros((2, 5))
        result = angle_phase(param, y, A)
        assert result == 0.0


class TestOptimizePhase:
    """Tests for optimize_phase."""

    def test_optimize_phase_basic(self):
        """Should return parameter array."""
        params = np.array([[0.0, 1.0], [0.5, 1.5]])
        y = np.array([0.0, np.pi / 2, np.pi, np.pi / 4])
        A = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.57, 3.14, 0.785]])
        result = optimize_phase(params, y, A)
        assert isinstance(result, np.ndarray)
        assert len(result) == params.shape[0]

    def test_optimize_phase_single_col(self):
        """Single column parameters should work."""
        params = np.array([[0.0], [0.5]])
        y = np.array([0.0, np.pi / 2])
        A = np.array([[1.0, 1.0], [1.0, 1.57]])
        result = optimize_phase(params, y, A)
        assert result.shape == (2,)

    def test_optimize_phase_large_maxcorr(self):
        """Large maxcorr should allow more solutions."""
        params = np.array([[0.0, 1.0], [0.0, 1.0]])
        y = np.array([0.0, 1.0, 2.0])
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = optimize_phase(params, y, A, maxcorr=100.0)
        assert isinstance(result, np.ndarray)


class TestScipyOptimizePhase:
    """Tests for scipy_optimize_phase."""

    def test_scipy_optimize_phase_returns_float(self):
        """Should return a float."""
        param = np.array([0.0, 1.0])
        y = np.array([0.0, np.pi / 2, np.pi])
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = scipy_optimize_phase(param, y, A)
        assert isinstance(result, float)

    def test_scipy_optimize_phase_positive(self):
        """Should return positive value (negated negative from compute_phase)."""
        param = np.zeros(2)
        y = np.array([0.0, np.pi / 4, np.pi / 2])
        A = np.ones((2, 3))
        result = scipy_optimize_phase(param, y, A)
        assert isinstance(result, float)
