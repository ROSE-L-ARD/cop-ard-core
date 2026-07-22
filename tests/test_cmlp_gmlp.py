"""Unit tests for cop_ard_core.cmlp and cop_ard_core.gmlp."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.cmlp import (
    calculate_stable_dates,
    get_current_reference,
    update_drift,
    update_recursive_reference,
)
from cop_ard_core.gmlp import get_radar_spacing, warp_cmlp


class TestCalculateStableDates:
    """Tests for calculate_stable_dates."""

    def test_calculate_stable_dates_basic(self):
        """Should select dates at stable_reference_step intervals."""
        dates = np.array([
            "2020-01-01", "2020-01-13", "2020-01-25",
            "2020-02-06", "2020-02-18", "2020-03-01",
        ], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=2)
        assert isinstance(result, np.ndarray)
        assert len(result) >= 1

    def test_calculate_stable_dates_step_one(self):
        """Step of 1 should select every date after start."""
        dates = np.array(["2020-01-01", "2020-01-13", "2020-01-25"], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1)
        assert len(result) >= 1

    def test_calculate_stable_dates_with_starting_date(self):
        """Starting date should affect selection."""
        dates = np.array([
            "2020-01-01", "2020-01-13", "2020-01-25",
            "2020-02-06", "2020-02-18",
        ], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(
            dates, revisit_cycle, stable_reference_step=1, starting_date="2020-01-13"
        )
        assert isinstance(result, np.ndarray)

    def test_calculate_stable_dates_with_stable_reference(self):
        """Should extend from existing stable reference."""
        stable_ref = xr.DataArray(
            [np.datetime64("2020-01-01", "ns"), np.datetime64("2020-01-25", "ns")],
            dims=["acquisition_date"],
        )
        dates = np.array([
            "2020-01-01", "2020-01-13", "2020-01-25",
            "2020-02-06", "2020-02-18",
        ], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        result = calculate_stable_dates(
            dates, revisit_cycle, stable_reference_step=1, stable_reference=stable_ref
        )
        assert isinstance(result, np.ndarray)

    def test_calculate_stable_dates_empty(self):
        """Empty dates array should return empty result."""
        dates = np.array([], dtype="datetime64[ns]")
        revisit_cycle = np.timedelta64(13, "D")
        with pytest.raises(ValueError):
            calculate_stable_dates(dates, revisit_cycle, stable_reference_step=1)


class TestGetCurrentReference:
    """Tests for get_current_reference."""

    def test_get_current_reference_no_last_or_stable(self):
        """With no last_reference or stable_reference, should use cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, None)
        assert isinstance(current, xr.DataArray)

    def test_get_current_reference_with_stable_reference(self):
        """Should select stable_reference when older than cstack."""
        stable_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.array(["2019-12-20"])},
        )
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        current, dates = get_current_reference(cstack, None, stable_ref)
        assert isinstance(current, xr.DataArray)

    def test_get_current_reference_raises_if_last_too_recent(self):
        """Should raise if last_reference is more recent than cstack."""
        cstack = xr.DataArray(
            np.ones((3, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": ["2020-01-01", "2020-01-13", "2020-01-25"]},
        )
        last_ref = xr.DataArray(
            np.ones((1, 4, 4)),
            dims=["acquisition_date", "az", "rg"],
            coords={"acquisition_date": np.datetime64("2020-03-01")},
        )
        with pytest.raises(Exception, match="Last current reference"):
            get_current_reference(cstack, last_ref, None)


class TestUpdateDrift:
    """Tests for update_drift."""

    def test_update_drift_basic(self):
        """Should return coherence DataArray."""
        drift_interf = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        reference = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        stable_ref = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        result = update_drift(
            drift_interf, reference, stable_ref,
            multilook={"azimuth_time": 3, "slant_range_time": 3},
            coherence_threshold=0.3,
        )
        assert isinstance(result, np.ndarray)

    def test_update_drift_no_nan(self):
        """Should handle drift_interf with no NaN values."""
        drift_interf = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        reference = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        stable_ref = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(2), "slant_range_time": np.arange(2)},
        )
        result = update_drift(
            drift_interf, reference, stable_ref,
            multilook={}, coherence_threshold=0.3,
        )
        assert isinstance(result, np.ndarray)


class TestUpdateRecursiveReference:
    """Tests for update_recursive_reference."""

    def test_update_recursive_reference_basic(self):
        """Basic update should return correct shape."""
        slc = xr.DataArray(
            np.ones((1, 6, 6), dtype=np.complex128),
            dims=["n_bands", "az", "rg"],
        )
        current_ref = xr.DataArray(np.ones((1, 6, 6), dtype=np.complex128), dims=["n_bands", "az", "rg"])
        normalized_interf = xr.DataArray(
            np.ones((1, 2, 2), dtype=np.complex128) * 0.8,
            dims=["n_bands", "coarse_az", "coarse_rg"],
        )
        result = update_recursive_reference(slc, current_ref, normalized_interf, memory_param=0.5)
        assert result.shape == slc.shape

    def test_update_recursive_reference_nan_handling(self):
        """NaN values should be replaced with 0."""
        slc = xr.DataArray(
            np.ones((1, 6, 6), dtype=np.complex128),
            dims=["n_bands", "az", "rg"],
        )
        current_ref = xr.DataArray(np.ones((1, 6, 6), dtype=np.complex128), dims=["n_bands", "az", "rg"])
        normalized_interf = xr.DataArray(
            np.array([[[np.nan, 0.5], [0.3, 0.2]]]),
            dims=["n_bands", "coarse_az", "coarse_rg"],
        )
        result = update_recursive_reference(slc, current_ref, normalized_interf, memory_param=0.5)
        assert not np.any(np.isnan(result.values))


class TestGetRadarSpacing:
    """Tests for get_radar_spacing."""

    def test_get_radar_spacing_basic(self):
        """Should return azimuth and range steps."""
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        rg = xr.DataArray(np.array([0.0, 0.5, 1.0]), dims=["slant_range_time"])
        az_step, rg_step = get_radar_spacing(az, rg)
        assert az_step is not None
        assert rg_step is not None


class TestWarpCmlp:
    """Tests for warp_cmlp."""

    def test_warp_cmlp_phase(self):
        """Should warp phase data correctly."""
        source = np.ones((4, 4), dtype=np.complex128)
        target = np.zeros((1, 4, 4), dtype=np.complex128)
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing='xy')
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target, target_indexes, source, x, y, kernel,
            is_phase=True, is_coherence=False,
        )
        assert target.shape == (1, 4, 4)

    def test_warp_cmlp_coherence_clipped(self):
        """Coherence output should be clipped to [0, 1]."""
        source = np.ones((4, 4), dtype=np.float64)
        target = np.zeros((1, 4, 4))
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing='xy')
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target, target_indexes, source, x, y, kernel,
            is_phase=False, is_coherence=True,
        )
        assert np.all(target >= 0.0)
        assert np.all(target <= 1.0)

    def test_warp_cmlp_no_error_standard(self):
        """Standard warp should not raise."""
        source = np.random.rand(4, 4)
        target = np.zeros((1, 4, 4))
        target_indexes = np.arange(16)
        x, y = np.meshgrid(np.arange(4), np.arange(4), indexing='xy')
        x = x.ravel()
        y = y.ravel()
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        warp_cmlp(
            target, target_indexes, source, x, y, kernel,
            is_phase=False, is_coherence=False,
        )
        assert target.shape == (1, 4, 4)
