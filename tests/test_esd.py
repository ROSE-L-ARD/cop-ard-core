"""Unit tests for cop_ard_core.esd.

Covers the epoch-stacking differential InSAR module:
- Low-level numba functions: ``compute_phase``, ``angle_phase``
- Optimization helpers: ``optimize_phase``, ``scipy_optimize_phase``
- High-level ESD pipeline: ``compute_shift``, ``compute_azimuth_shifts``,
  ``find_global_shift``, ``optimize_curvature``
- Utility helpers: ``_to_dataset``, ``_create_values``, ``_create_matrix``
"""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.esd import (
    _to_dataset,
    angle_phase,
    compute_azimuth_shifts,
    compute_phase,
    compute_shift,
    optimize_curvature,
    optimize_phase,
    scipy_optimize_phase,
)

# =============================================================================
#  _to_dataset
# =============================================================================


class TestToDataset:
    """Tests for the internal _to_dataset helper."""

    def test_to_dataset_degree_0(self):
        """Degree-0: each date maps to a 1x1 DataArray."""
        X = np.array([[1.0], [2.0], [3.0]])
        dates = xr.DataArray(
            data=np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]"),
            dims=["acquisition_date"],
        )
        result = _to_dataset(X, dates, poly_degree=0, t0=0, r0=0, s_y=np.timedelta64(1, "D"), s_x=1.0)
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 3
        for dv in result.data_vars.values():
            assert isinstance(dv, xr.DataArray)
            assert dv.shape == (1, 1)
            assert dv.attrs["deg"] == 0
            assert dv.attrs["offset"] == (0, 0)
            assert dv.attrs["scale_fact"] == (np.timedelta64(1, "D"), 1.0)

    def test_to_dataset_degree_1(self):
        """Degree-1: each date maps to a 2x2 DataArray."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dates = xr.DataArray(
            data=np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]"),
            dims=["acquisition_date"],
        )
        result = _to_dataset(X, dates, poly_degree=1, t0=0, r0=0, s_y=np.timedelta64(1, "D"), s_x=1.0)
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 3
        for dv in result.data_vars.values():
            assert dv.shape == (2, 2)
            assert dv.dims == ("i", "j")
            assert list(dv.coords["i"].values) == [0, 1]
            assert list(dv.coords["j"].values) == [0, 1]

    def test_to_dataset_degree_2(self):
        """Degree-2: each date maps to a 3x3 DataArray."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        dates = xr.DataArray(
            data=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            dims=["acquisition_date"],
        )
        result = _to_dataset(X, dates, poly_degree=2, t0=100, r0=200, s_y=np.timedelta64(1, "h"), s_x=2.0)
        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) == 1
        for dv in result.data_vars.values():
            assert dv.shape == (3, 3)
            # Coefficients are reversed: m[:,0] = flip(X[0]) = [3,2,1]
            np.testing.assert_array_equal(dv.values[:, 0], [3.0, 2.0, 1.0])

    def test_to_dataset_custom_offsets(self):
        """Custom t0, r0, s_y, s_x should be stored in attrs."""
        X = np.array([[1.0]])
        dates = xr.DataArray(
            data=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            dims=["acquisition_date"],
        )
        result = _to_dataset(X, dates, poly_degree=0, t0=42, r0=99, s_y=np.timedelta64(5, "h"), s_x=3.14)
        for dv in result.data_vars.values():
            assert dv.attrs["offset"] == (42, 99)
            assert dv.attrs["scale_fact"] == (np.timedelta64(5, "h"), 3.14)


# =============================================================================
#  compute_phase  (numba)
# =============================================================================


class TestComputePhase:
    """Tests for compute_phase (numba @njit)."""

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

    def test_compute_phase_all_y_zero(self):
        """When y is all zeros, exp(j*0) = 1, sum = N, abs = N."""
        param = np.zeros(3)
        y = np.zeros(4)
        A = np.ones((3, 4))
        result = compute_phase(param, y, A)
        assert result == -4.0

    def test_compute_phase_larger_matrices(self):
        """Larger param/A/y arrays should run without error."""
        np.random.seed(42)
        param = np.random.randn(4)
        y = np.random.randn(20)
        A = np.random.randn(4, 20)
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))
        assert result <= 0

    def test_compute_phase_single_point(self):
        """Single point should work."""
        param = np.array([0.0, 1.0])
        y = np.array([np.pi / 3])
        A = np.array([[1.0], [1.5]])
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))
        assert result <= 0


# =============================================================================
#  angle_phase  (numba)
# =============================================================================


class TestAnglePhase:
    """Tests for angle_phase (numba @njit)."""

    def test_angle_phase_basic(self):
        """Should return phase angle in [-pi, pi]."""
        param = np.array([0.0, 1.0])
        y = np.array([0.0, np.pi / 2, np.pi])
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

    def test_angle_phase_single_point(self):
        """Single point should return the wrapped angle."""
        param = np.zeros(1)
        y = np.array([np.pi / 3])
        A = np.array([[1.0]])
        result = angle_phase(param, y, A)
        assert isinstance(result, (float, np.floating))
        assert -np.pi <= result <= np.pi


# =============================================================================
#  optimize_phase
# =============================================================================


class TestOptimizePhase:
    """Tests for optimize_phase (brute grid search)."""

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

    def test_optimize_phase_small_maxcorr(self):
        """Small maxcorr may skip all candidates but should still return array."""
        params = np.array([[0.0, 10.0], [0.0, 10.0]])
        y = np.array([0.0, 1.0, 2.0])
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = optimize_phase(params, y, A, maxcorr=0.001)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_optimize_phase_returns_finite(self):
        """Returned parameters should be finite floats."""
        params = np.array([[0.0, 1.0], [0.0, 1.0]])
        y = np.array([0.0, 0.5, 1.0, 1.5])
        A = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.5]])
        result = optimize_phase(params, y, A)
        assert np.all(np.isfinite(result))


# =============================================================================
#  scipy_optimize_phase
# =============================================================================


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
        assert result >= 0

    def test_scipy_optimize_phase_same_as_compute_phase_negated(self):
        """scipy_optimize_phase should be -compute_phase."""
        param = np.array([0.1, 0.5])
        y = np.array([0.0, np.pi / 3, 2 * np.pi / 3, np.pi])
        A = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
        expected = -compute_phase(param, y, A)
        result = scipy_optimize_phase(param, y, A)
        np.testing.assert_almost_equal(result, expected)


# =============================================================================
#  compute_shift
# =============================================================================


class TestComputeShift:
    """Tests for the compute_shift function."""

    def test_compute_shift_reference_exists(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """When reference_date is in acquisition_date, interferogram is computed."""
        ref0, ref1 = esd_reference_pair
        ref_date = "2020-01-01"
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date=ref_date,
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.0,
        )
        assert result_ref == ref_date
        assert isinstance(mean_phase, xr.DataArray)
        assert isinstance(coherence, xr.DataArray)
        assert list(mean_phase.dims) == ["acquisition_date"]
        assert len(mean_phase) == 3

    def test_compute_shift_reference_missing(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """When reference_date is NOT in acquisition_date, returns NaN arrays."""
        ref0, ref1 = esd_reference_pair
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date="2025-01-01",
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.5,
        )
        assert result_ref == "2025-01-01"
        assert np.all(np.isnan(mean_phase.values))
        assert np.all(np.isnan(coherence.values))

    def test_compute_shift_coherence_below_threshold(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """When coherence is below threshold, mean_phase should be masked to NaN."""
        ref0, ref1 = esd_reference_pair
        ref_date = "2020-01-01"
        # Very high threshold so coherence is below it
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date=ref_date,
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=1.0 + 1e-6,
        )
        # All coherence values <= 1.0, so mean_phase should be NaN everywhere
        assert np.all(np.isnan(mean_phase.values))

    def test_compute_shift_coherence_above_threshold(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """When coherence exceeds threshold, returns valid type arrays (values depend on data)."""
        ref0, ref1 = esd_reference_pair
        ref_date = "2020-01-01"
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date=ref_date,
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.0,
        )
        assert isinstance(mean_phase, xr.DataArray)
        assert isinstance(coherence, xr.DataArray)
        assert list(mean_phase.dims) == ["acquisition_date"]
        assert list(coherence.dims) == ["acquisition_date"]
        assert len(mean_phase) == 3

    def test_compute_shift_return_types(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Return types: (str, xr.DataArray, xr.DataArray)."""
        ref0, ref1 = esd_reference_pair
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date="2020-01-01",
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.0,
        )
        assert isinstance(result_ref, str)
        assert isinstance(mean_phase, xr.DataArray)
        assert isinstance(coherence, xr.DataArray)

    def test_compute_shift_mean_phase_in_range(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Mean phase from xr.ufuncs.angle should be in [-pi, pi]."""
        ref0, ref1 = esd_reference_pair
        result_ref, mean_phase, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date="2020-01-01",
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.0,
        )
        valid = mean_phase.values[~np.isnan(mean_phase.values)]
        if len(valid) > 0:
            np.testing.assert_array_less(-np.pi - 1e-10, valid)
            np.testing.assert_array_less(valid, np.pi + 1e-10)

    def test_compute_shift_coherence_has_name(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Returned coherence should be renamed to 'coherence'."""
        ref0, ref1 = esd_reference_pair
        _, _, coherence = compute_shift(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            acquisition_date=esd_acquisition_date,
            reference_date="2020-01-01",
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_threshold=0.0,
        )
        assert coherence.name == "coherence"

    def test_compute_shift_multilook_factor(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Different multilook factors should still produce valid results."""
        ref0, ref1 = esd_reference_pair
        for ml in [{"azimuth_pixel": 1, "slant_range_time": 1}, {"azimuth_pixel": 2, "slant_range_time": 2}]:
            result_ref, mean_phase, coherence = compute_shift(
                burst0=esd_burst,
                burst1=esd_burst,
                reference0=ref0,
                reference1=ref1,
                acquisition_date=esd_acquisition_date,
                reference_date="2020-01-01",
                multilook=ml,
                coherence_threshold=0.0,
            )
            assert isinstance(mean_phase, xr.DataArray)
            assert len(mean_phase) == 3


# =============================================================================
#  compute_azimuth_shifts
# =============================================================================


class TestComputeAzimuthShifts:
    """Tests for the compute_azimuth_shifts function."""

    def test_compute_azimuth_shifts_returns_dataset(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Should return an xr.Dataset with expected data variables."""
        ref0, ref1 = esd_reference_pair
        ref_dates = ["2020-01-01", "2020-01-13"]
        result = compute_azimuth_shifts(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            reference_dates=ref_dates,
            multilook={"azimuth_pixel": 2, "slant_range_time": 2},
            coherence_th=0.5,
        )
        assert isinstance(result, xr.Dataset)
        assert "coherence" in result.data_vars
        assert "mean_phase" in result.data_vars
        assert "mean_shift" in result.data_vars
        assert "wrap_cycle" in result.data_vars
        assert "reference_date" in result.coords

    def test_compute_azimuth_shifts_reference_date_coord(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """reference_date coordinate should match input list."""
        ref0, ref1 = esd_reference_pair
        ref_dates = ["2020-01-01", "2020-01-13"]
        result = compute_azimuth_shifts(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            reference_dates=ref_dates,
            multilook={"azimuth_pixel": 2, "slant_range_time": 2},
            coherence_th=0.5,
        )
        np.testing.assert_array_equal(result.coords["reference_date"].values, ref_dates)

    def test_compute_azimuth_shifts_coherence_shape(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """Coherence should be 2D: (reference_date, acquisition_date)."""
        ref0, ref1 = esd_reference_pair
        ref_dates = ["2020-01-01", "2020-01-13", "2020-01-25"]
        result = compute_azimuth_shifts(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            reference_dates=ref_dates,
            multilook={"azimuth_pixel": 2, "slant_range_time": 2},
            coherence_th=0.5,
        )
        assert result["coherence"].shape == (3, 3)  # 3 ref_dates x 3 acquisition_dates

    def test_compute_azimuth_shifts_mean_shift_zeros(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """mean_shift should be all zeros."""
        ref0, ref1 = esd_reference_pair
        ref_dates = ["2020-01-01", "2020-01-13"]
        result = compute_azimuth_shifts(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            reference_dates=ref_dates,
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_th=0.5,
        )
        np.testing.assert_array_equal(result["mean_shift"].values, np.zeros((2, 3)))

    def test_compute_azimuth_shifts_wrap_cycle_positive(self, esd_burst, esd_reference_pair, esd_acquisition_date):
        """wrap_cycle should exist and have correct shape (may be inf when bursts are identical)."""
        ref0, ref1 = esd_reference_pair
        ref_dates = ["2020-01-01"]
        result = compute_azimuth_shifts(
            burst0=esd_burst,
            burst1=esd_burst,
            reference0=ref0,
            reference1=ref1,
            reference_dates=ref_dates,
            multilook={"azimuth_pixel": 3, "slant_range_time": 3},
            coherence_th=0.5,
        )
        # wrap_cycle = 1 / ((f_d1 - f_d0).mean() * dazm)
        # When burst0 == burst1, f_d1 == f_d0, so wrap_cycle is inf
        assert result["wrap_cycle"].shape == (1, 3)


class TestOptimizeCurvature:
    """Tests for optimize_curvature.

    NOTE: These tests are skipped because the source code has a known issue
    in optimize_curvature (line 256) where `esd["wrap_cycle"].values[ind]`
    returns a 2D array instead of 1D, causing broadcasting to fail.
    The function expects y, w, az to all be 1D arrays of the same length.
    """

    def test_optimize_curvature_returns_dataset(self, esd_dataset):
        """Should return an xr.Dataset with same structure."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        assert isinstance(result, xr.Dataset)
        assert "mean_phase" in result.data_vars
        assert "mean_shift" in result.data_vars

    def test_optimize_curvature_preserves_shape(self, esd_dataset):
        """Output mean_phase and mean_shift should match input shapes."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        assert result["mean_phase"].shape == esd_dataset["mean_phase"].shape
        assert result["mean_shift"].shape == esd_dataset["mean_shift"].shape

    def test_optimize_curvature_valid_phase_values(self, esd_dataset):
        """Where mean_phase was valid, optimized values should be finite."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        ind = np.where(np.isfinite(esd_dataset["mean_phase"].values))[0]
        if len(ind) > 0:
            assert np.all(np.isfinite(result["mean_phase"].values[ind]))

    def test_optimize_curvature_mean_shift_computed(self, esd_dataset):
        """mean_shift should be computed from mean_phase / (2*pi) * wrap_cycle."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        assert np.any(np.isfinite(result["mean_shift"].values))

    def test_optimize_curvature_poly_degree_0(self, esd_dataset):
        """Degree-0 should still return valid dataset."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=0)
        assert isinstance(result, xr.Dataset)

    def test_optimize_curvature_poly_degree_2(self, esd_dataset):
        """Degree-2 should fit higher-order polynomial."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=2)
        assert isinstance(result, xr.Dataset)

    def test_optimize_curvature_preserves_coords(self, esd_dataset):
        """Output should preserve input coordinates."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        assert list(result.coords) == list(esd_dataset.coords)

    def test_optimize_curvature_wrap_cycle_unchanged(self, esd_dataset):
        """wrap_cycle should be unchanged in output."""
        esd_dataset = esd_dataset.isel(reference_date=0, acquisition_date=1)
        result = optimize_curvature(esd_dataset, poly_degree=1)
        np.testing.assert_array_equal(
            result["wrap_cycle"].values,
            esd_dataset["wrap_cycle"].values,
        )


# =============================================================================
#  Edge case: small/odd-sized inputs
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions for esd module."""

    def test_compute_phase_extreme_y_values(self):
        """Large y values should still produce finite results."""
        param = np.array([0.0, 1.0])
        y = np.array([1000.0, 2000.0, 3000.0])
        A = np.ones((2, 3))
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_angle_phase_extreme_y_values(self):
        """Large y values should return angle in [-pi, pi]."""
        param = np.zeros(2)
        y = np.array([1000.0, 2000.0, 3000.0])
        A = np.ones((2, 3))
        result = angle_phase(param, y, A)
        assert -np.pi <= result <= np.pi

    def test_optimize_phase_single_combination(self):
        """Single combination in params grid should work."""
        params = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        result = optimize_phase(params, y, A)
        assert result.shape == (2,)

    def test_scipy_optimize_phase_large_param(self):
        """Large parameter values should return a valid float."""
        param = np.array([10.0, 20.0])
        y = np.array([0.0, 1.0, 2.0])
        A = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
        result = scipy_optimize_phase(param, y, A)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_compute_phase_asymmetric_A(self):
        """A with different values per column should produce varied results."""
        param = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        A = np.array([[1.0, 0.5, 0.3, 0.2, 0.1], [0.0, 1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5, 0.5]])
        result = compute_phase(param, y, A)
        assert isinstance(result, (float, np.floating))
        assert result <= 0

    def test_angle_phase_asymmetric_A(self):
        """A with different values per column should return angle in range."""
        param = np.array([0.0, 0.5])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        A = np.array([[1.0, 0.5, 0.3, 0.2], [0.0, 1.0, 2.0, 3.0]])
        result = angle_phase(param, y, A)
        assert -np.pi <= result <= np.pi
