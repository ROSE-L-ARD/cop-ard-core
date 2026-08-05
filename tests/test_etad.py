"""Unit tests for cop_ard_core.etad."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.etad import apply_etad_corrections


class TestApplyEtadCorrections:
    """Tests for apply_etad_corrections."""

    # ------------------------------------------------------------------
    # Empty grid
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_empty_grid(self):
        """Empty correction grid should return zero timedelta and single-element zero array."""
        az = xr.DataArray(np.arange(4), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4), dims=["slant_range_time"])
        etad_grid = xr.Dataset()
        dtaz = np.timedelta64(1, "ns")
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert daz == np.timedelta64(0, "ns")
        assert drg.shape == (1,)
        np.testing.assert_array_equal(drg, np.zeros(1))

    def test_apply_etad_corrections_empty_grid_types(self):
        """Empty grid: daz should be scalar timedelta64, drg should be ndarray."""
        az = xr.DataArray(np.arange(3), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(3), dims=["slant_range_time"])
        etad_grid = xr.Dataset()
        dtaz = 1.0
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert isinstance(daz, np.timedelta64)
        assert isinstance(drg, np.ndarray)
        assert drg.dtype == np.float64

    # ------------------------------------------------------------------
    # Type preservation
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_type_daz_ndarray(self):
        """daz should be np.ndarray (not xr.DataArray) with timedelta64[ns] dtype."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz=1.0, dtrg=1.0, rg0=0.0, az0=0)
        assert isinstance(daz, np.ndarray)
        assert daz.dtype == np.dtype("timedelta64[ns]")

    def test_apply_etad_corrections_type_drg_ndarray(self):
        """drg should be np.ndarray with float64 dtype."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz=1.0, dtrg=1.0, rg0=0.0, az0=0)
        assert isinstance(drg, np.ndarray)
        assert drg.dtype == np.float64

    # ------------------------------------------------------------------
    # Shape preservation
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_shape_az_1d(self):
        """Azimuth correction output shape should match 1D input shape."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz=1.0, dtrg=1.0, rg0=0.0, az0=0)
        assert daz.shape == az.shape

    def test_apply_etad_corrections_shape_rang_1d(self):
        """Range correction output shape should match 1D input shape."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz=1.0, dtrg=1.0, rg0=0.0, az0=0)
        assert drg.shape == rg.shape

    def test_apply_etad_corrections_shape_2d_az(self):
        """daz should reshape correctly for 2D azimuth input."""
        n = 12
        az = xr.DataArray(
            np.arange(n, dtype=np.float64).reshape(3, 4),
            dims=["azimuth_time", "slant_range_time"],
        )
        rg = xr.DataArray(np.arange(n, dtype=np.float64), dims=["slant_range_time"])
        etad_grid = _build_etad_grid(n, n, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert daz.shape == az.shape

    def test_apply_etad_corrections_shape_2d_rang(self):
        """drg should reshape correctly for 2D range input."""
        n = 12
        az = xr.DataArray(np.arange(n, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(
            np.arange(n, dtype=np.float64).reshape(2, 6),
            dims=["azimuth_time", "slant_range_time"],
        )
        etad_grid = _build_etad_grid(n, n, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert drg.shape == rg.shape

    def test_apply_etad_corrections_shape_2d_both(self):
        """Both outputs should match their 2D inputs."""
        n = 12
        az = xr.DataArray(
            np.arange(n, dtype=np.float64).reshape(3, 4),
            dims=["azimuth_time", "slant_range_time"],
        )
        rg = xr.DataArray(
            np.arange(n, dtype=np.float64).reshape(3, 4),
            dims=["azimuth_time", "slant_range_time"],
        )
        etad_grid = _build_etad_grid(n, n, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert daz.shape == az.shape
        assert drg.shape == rg.shape

    # ------------------------------------------------------------------
    # Value correctness
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_values_constant_azcorr(self):
        """With uniform azcorr=0.001, output should be non-zero and uniform."""
        az, rg, etad_grid = _make_etad_inputs(n=4, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        # All values should be the same (uniform grid) and non-zero
        assert np.all(daz == daz[0])
        assert np.all(daz != np.timedelta64(0, "ns"))

    def test_apply_etad_corrections_values_zero_azcorr(self):
        """With zero azcorr, daz should be all zeros."""
        az, rg, etad_grid = _make_etad_inputs(n=4, azcorr_val=0.0, rgcorr_val=0.0)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        np.testing.assert_array_equal(daz, np.zeros(daz.shape, dtype="timedelta64[ns]"))

    def test_apply_etad_corrections_values_drg_proportional_to_rgcorr(self):
        """drg values should be proportional to rgcorr grid values."""
        rgcorr_val = 0.05
        az, rg, etad_grid = _make_etad_inputs(n=4, azcorr_val=0.0, rgcorr_val=rgcorr_val)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        # With uniform rgcorr=0.05, interpolation yields 0.05 for all interior points
        np.testing.assert_array_almost_equal(drg, rgcorr_val)

    def test_apply_etad_corrections_values_gradient_rangcorr(self):
        """Gradient rgcorr should produce gradient drg output."""
        az = xr.DataArray(np.arange(4, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4, dtype=np.float64), dims=["slant_range_time"])
        # Gradient rgcorr: increasing along slant_range dimension
        rgcorr_data = np.outer(np.ones(4), np.linspace(0.0, 0.1, 4))
        azcorr_data = np.zeros((4, 4))
        etad_grid = xr.Dataset(
            data_vars={
                "azcorr": (["azimuth_time", "slant_range_time"], azcorr_data),
                "rgcorr": (["azimuth_time", "slant_range_time"], rgcorr_data),
                "azimuth_time": (["azimuth_time"], np.arange(4, dtype=np.float64)),
                "slant_range_time": (["slant_range_time"], np.arange(4, dtype=np.float64)),
            }
        )
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        # drg should increase along the range dimension
        assert drg[0] < drg[-1]

    def test_apply_etad_corrections_values_match_interior_points(self):
        """Interior points should match grid values for constant grid."""
        rgcorr_val = 0.025
        az, rg, etad_grid = _make_etad_inputs(n=4, azcorr_val=0.0, rgcorr_val=rgcorr_val)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        np.testing.assert_array_almost_equal(drg, rgcorr_val)

    # ------------------------------------------------------------------
    # Edge cases: minimum valid grid size (cubic requires >= 4 points)
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_minimum_grid_size(self):
        """Minimum grid size of 4x4 for cubic interpolation should work."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert daz.shape == (4,)
        assert drg.shape == (4,)

    def test_apply_etad_corrections_mismatched_lengths_raises(self):
        """az and rg with different lengths should raise ValueError."""
        az = xr.DataArray(np.arange(3, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4, dtype=np.float64), dims=["slant_range_time"])
        etad_grid = _build_etad_grid(4, 4, azcorr_val=0.001, rgcorr_val=0.01)
        with pytest.raises(ValueError):
            apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)

    # ------------------------------------------------------------------
    # Different parameter combinations
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_different_dtaz(self):
        """Different dtaz values should affect the pixel coordinate mapping."""
        # Use large azcorr values so rounding doesn't eliminate differences
        az = xr.DataArray(np.arange(4, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4, dtype=np.float64), dims=["slant_range_time"])
        # Grid 1: azcorr = [0, 100, 200, 300] per row (large values)
        azcorr_1 = np.outer(np.array([0.0, 100.0, 200.0, 300.0]), np.ones(4))
        etad_grid_1 = xr.Dataset(
            data_vars={
                "azcorr": (["azimuth_time", "slant_range_time"], azcorr_1),
                "rgcorr": (["azimuth_time", "slant_range_time"], np.ones((4, 4)) * 0.01),
                "azimuth_time": (["azimuth_time"], np.arange(4, dtype=np.float64)),
                "slant_range_time": (["slant_range_time"], np.arange(4, dtype=np.float64)),
            }
        )
        # Grid 2: azcorr = [0, 50, 150, 250] per row, different coordinate spacing
        azcorr_2 = np.outer(np.array([0.0, 50.0, 150.0, 250.0]), np.ones(4))
        etad_grid_2 = xr.Dataset(
            data_vars={
                "azcorr": (["azimuth_time", "slant_range_time"], azcorr_2),
                "rgcorr": (["azimuth_time", "slant_range_time"], np.ones((4, 4)) * 0.01),
                "azimuth_time": (["azimuth_time"], np.arange(4, dtype=np.float64) * 2),
                "slant_range_time": (["slant_range_time"], np.arange(4, dtype=np.float64)),
            }
        )
        daz_1, _ = apply_etad_corrections(az, rg, etad_grid_1, 1.0, 1.0, 0.0, 0)
        daz_2, _ = apply_etad_corrections(az, rg, etad_grid_2, 2.0, 1.0, 0.0, 0)
        # Different grids produce different daz
        assert not np.all(daz_1 == daz_2)

    def test_apply_etad_corrections_different_dtrg(self):
        """Different dtrg values should affect the range pixel coordinate mapping."""
        az, rg, etad_grid_1 = _make_etad_inputs(n=4, dtrg=1.0)
        az, rg, etad_grid_2 = _make_etad_inputs(n=4, dtrg=2.0)
        _, drg_1 = apply_etad_corrections(az, rg, etad_grid_1, 1.0, 1.0, 0.0, 0)
        _, drg_2 = apply_etad_corrections(az, rg, etad_grid_2, 1.0, 2.0, 0.0, 0)
        assert not np.all(drg_1 == drg_2)

    def test_apply_etad_corrections_different_rg0(self):
        """Non-zero rg0 should shift range pixel coordinates."""
        az, rg, etad_grid_1 = _make_etad_inputs(n=4, dtrg=1.0, rg0=0.0)
        az, rg, etad_grid_2 = _make_etad_inputs(n=4, dtrg=1.0, rg0=5.0)
        _, drg_a = apply_etad_corrections(az, rg, etad_grid_1, 1.0, 1.0, 0.0, 0)
        _, drg_b = apply_etad_corrections(az, rg, etad_grid_2, 1.0, 1.0, 5.0, 0)
        assert not np.all(drg_a == drg_b)

    # ------------------------------------------------------------------
    # Out-of-bounds / fill_value behavior
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_out_of_bounds_fill_zero(self):
        """Points should produce finite output (fill_value=0 for out-of-bounds)."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert np.all(np.isfinite(daz))
        assert np.all(np.isfinite(drg))

    # ------------------------------------------------------------------
    # Return type tuple
    # ------------------------------------------------------------------

    def test_apply_etad_corrections_returns_tuple(self):
        """Should return a tuple of two elements."""
        az = xr.DataArray(np.arange(4), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4), dims=["slant_range_time"])
        etad_grid = xr.Dataset()
        result = apply_etad_corrections(az, rg, etad_grid, 1.0, 0.5, 0.0, np.timedelta64(0, "ns"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_apply_etad_corrections_returns_tuple_with_grid(self):
        """Tuple return with non-empty grid should have two elements."""
        az, rg, etad_grid = _make_etad_inputs(n=4)
        result = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # Parametrized: various grid sizes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("n", [4, 5, 10])
    def test_apply_etad_corrections_various_grid_sizes(self, n):
        """Should work with various grid sizes (minimum 4 for cubic)."""
        az = xr.DataArray(np.arange(n, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(n, dtype=np.float64), dims=["slant_range_time"])
        etad_grid = _build_etad_grid(n, n, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert daz.shape == (n,)
        assert drg.shape == (n,)

    # ------------------------------------------------------------------
    # Parametrized: various correction values
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("azcorr_val,rgcorr_val", [
        (0.0, 0.0),
        (0.001, 0.01),
        (-0.001, -0.01),
        (1.0, 10.0),
    ])
    def test_apply_etad_corrections_various_correction_values(self, azcorr_val, rgcorr_val):
        """Should handle various correction grid values including negative."""
        az = xr.DataArray(np.arange(4, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4, dtype=np.float64), dims=["slant_range_time"])
        etad_grid = _build_etad_grid(4, 4, azcorr_val=azcorr_val, rgcorr_val=rgcorr_val)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, 0.0, 0)
        assert daz.shape == (4,)
        assert drg.shape == (4,)

    # ------------------------------------------------------------------
    # Parametrized: non-zero origins
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("rg0", [0.0, 1.0, 5.0])
    def test_apply_etad_corrections_nonzero_rg0(self, rg0):
        """Non-zero rg0 should still produce valid output."""
        az = xr.DataArray(np.arange(4, dtype=np.float64), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4, dtype=np.float64) + rg0, dims=["slant_range_time"])
        etad_grid = _build_etad_grid(4, 4, azcorr_val=0.001, rgcorr_val=0.01)
        daz, drg = apply_etad_corrections(az, rg, etad_grid, 1.0, 1.0, rg0, 0)
        assert daz.shape == (4,)
        assert drg.shape == (4,)


# ===========================================================================
# Helper functions
# ===========================================================================


def _make_etad_inputs(n=4, azcorr_val=0.001, rgcorr_val=0.01, dtrg=1.0, dtaz=1.0, rg0=0.0, az0=0):
    """Build az, rg, and etad_grid inputs that align for proper interpolation.

    The grid coordinates are constructed to normalize to [0, 1, ..., n-1],
    matching the normalized query coordinates.

    Parameters
    ----------
    n : int
        Number of points.
    azcorr_val : float
        Constant azimuth correction value.
    rgcorr_val : float
        Constant range correction value.
    dtrg : float
        Range step for coordinate normalization.
    dtaz : float
        Azimuth step for coordinate normalization.
    rg0 : float
        Range origin for coordinate normalization.
    az0 : float
        Azimuth origin for coordinate normalization.

    Returns
    -------
    tuple
        (az, rg, etad_grid) — azimuth and range DataArrays and ETAD Dataset.
    """
    # Query coordinates
    az = xr.DataArray(np.arange(n, dtype=np.float64), dims=["azimuth_time"])
    rg = xr.DataArray(np.arange(n, dtype=np.float64), dims=["slant_range_time"])

    # Grid coordinates that normalize to [0, 1, ..., n-1]
    rg_coords = np.array([rg0 + i * dtrg for i in range(n)], dtype=np.float64)
    az_coords = np.array([az0 + i * dtaz for i in range(n)], dtype=np.float64)

    etad_grid = xr.Dataset(
        data_vars={
            "azcorr": (["azimuth_time", "slant_range_time"], np.ones((n, n)) * azcorr_val),
            "rgcorr": (["azimuth_time", "slant_range_time"], np.ones((n, n)) * rgcorr_val),
            "azimuth_time": (["azimuth_time"], az_coords),
            "slant_range_time": (["slant_range_time"], rg_coords),
        }
    )
    return az, rg, etad_grid


def _build_etad_grid(n_az, n_rg, azcorr_val=0.001, rgcorr_val=0.01):
    """Build a non-empty ETAD correction Dataset for testing.

    Grid coordinates are integers [0, 1, ..., n-1] which match the
    default query coordinates (also integers starting at 0).

    Parameters
    ----------
    n_az : int
        Number of azimuth grid points.
    n_rg : int
        Number of range grid points.
    azcorr_val : float
        Constant value for the azcorr grid.
    rgcorr_val : float
        Constant value for the rgcorr grid.

    Returns
    -------
    xr.Dataset
        ETAD correction grid with azcorr, rgcorr, azimuth_time, and
        slant_range_time data variables.
    """
    return xr.Dataset(
        data_vars={
            "azcorr": (["azimuth_time", "slant_range_time"], np.ones((n_az, n_rg)) * azcorr_val),
            "rgcorr": (["azimuth_time", "slant_range_time"], np.ones((n_az, n_rg)) * rgcorr_val),
            "azimuth_time": (["azimuth_time"], np.arange(n_az)),
            "slant_range_time": (["slant_range_time"], np.arange(n_rg)),
        }
    )
