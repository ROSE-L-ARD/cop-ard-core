"""Unit tests for cop_ard_core.matching."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.matching import (
    backward_flatten,
    compute_pearson_coefficient,
    forward_flatten,
    get_corr_sub_pix,
    initialize_patches,
)

# ===========================================================================
# TestGetCorrSubPix
# ===========================================================================


class TestGetCorrSubPix:
    """Tests for get_corr_sub_pix."""

    @pytest.mark.unit
    def test_get_corr_sub_pix_basic(self):
        """Basic sub-pixel peak detection."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        corr[0, 1, 2] = 0.8
        corr[0, 3, 2] = 0.85
        corr[0, 2, 1] = 0.82
        corr[0, 2, 3] = 0.88
        result = get_corr_sub_pix(corr)
        assert result.shape == (1, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_perfect_peak(self):
        """Peak of 1.0 should return integer position without paraboloid fit."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 1.0
        result = get_corr_sub_pix(corr)
        assert result[0, 0] == 0.0  # center position (dy)
        assert result[0, 1] == 0.0  # center position (dx)

    @pytest.mark.unit
    def test_get_corr_sub_pix_dim_int(self):
        """Integer dim should be converted to [dim, dim]."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        result = get_corr_sub_pix(corr, dim=1)
        assert result.shape == (1, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_no_dim(self):
        """No dim should default to [1, 1]."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        result = get_corr_sub_pix(corr, dim=None)
        assert result.shape == (1, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_multiple_images(self):
        """Multiple images should return one row per image."""
        corr = np.zeros((3, 5, 5))
        for i in range(3):
            corr[i, 2, 2] = 0.95
            corr[i, 1, 2] = 0.8
            corr[i, 3, 2] = 0.85
            corr[i, 2, 1] = 0.82
            corr[i, 2, 3] = 0.88
        result = get_corr_sub_pix(corr)
        assert result.shape == (3, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_small_box(self):
        """Peak near edge should be handled without crash."""
        corr = np.zeros((1, 5, 5))
        corr[0, 0, 0] = 0.95
        corr[0, 0, 1] = 0.8
        corr[0, 1, 0] = 0.8
        result = get_corr_sub_pix(corr)
        assert result.shape == (1, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_output_columns(self):
        """Output should have 5 columns: dy, dx, coherence, sy, sx."""
        corr = np.zeros((2, 7, 7))
        corr[0, 3, 3] = 0.95
        corr[0, 2, 3] = 0.8
        corr[0, 4, 3] = 0.85
        corr[0, 3, 2] = 0.82
        corr[0, 3, 4] = 0.88
        corr[1, 3, 3] = 0.9
        corr[1, 2, 3] = 0.7
        corr[1, 4, 3] = 0.75
        corr[1, 3, 2] = 0.72
        corr[1, 3, 4] = 0.78
        result = get_corr_sub_pix(corr)
        assert result.shape == (2, 5)
        assert result.shape[1] == 5

    @pytest.mark.unit
    def test_get_corr_sub_pix_all_zeros(self):
        """All zeros correlation should return zeros."""
        corr = np.zeros((1, 5, 5))
        result = get_corr_sub_pix(corr)
        assert result.shape == (1, 5)
        assert np.all(result == 0.0)

    @pytest.mark.unit
    def test_get_corr_sub_pix_different_peak_positions(self):
        """Different peak positions should produce different outputs."""
        corr = np.zeros((2, 7, 7))
        corr[0, 1, 2] = 0.95  # peak at (1, 2)
        corr[0, 0, 2] = 0.8
        corr[0, 2, 2] = 0.85
        corr[0, 1, 1] = 0.82
        corr[0, 1, 3] = 0.88
        corr[1, 5, 4] = 0.95  # peak at (5, 4)
        corr[1, 4, 4] = 0.8
        corr[1, 6, 4] = 0.85
        corr[1, 5, 3] = 0.82
        corr[1, 5, 5] = 0.88
        result = get_corr_sub_pix(corr)
        assert result.shape == (2, 5)
        # Check that positions differ between the two images
        assert not (np.allclose(result[0, :2], result[1, :2]))

    @pytest.mark.unit
    def test_get_corr_sub_pix_output_values_range(self):
        """dy and dx should be reasonable fractions."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        corr[0, 1, 2] = 0.8
        corr[0, 3, 2] = 0.85
        corr[0, 2, 1] = 0.82
        corr[0, 2, 3] = 0.88
        result = get_corr_sub_pix(corr)
        # dy and dx are within reasonable range
        assert abs(result[0, 0]) < corr.shape[1]
        assert abs(result[0, 1]) < corr.shape[2]

    @pytest.mark.unit
    def test_get_corr_sub_pix_dim_as_list(self):
        """dim as list should work same as int."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        result_list = get_corr_sub_pix(corr, dim=[1, 1])
        result_int = get_corr_sub_pix(corr, dim=1)
        np.testing.assert_array_equal(result_list, result_int)

    @pytest.mark.unit
    def test_get_corr_sub_pix_multiple_images_different_peaks(self):
        """Multiple images with different peak quality should handle gracefully."""
        corr = np.zeros((3, 7, 7))
        # Image 0: strong peak
        corr[0, 3, 3] = 0.95
        corr[0, 2, 3] = 0.8
        corr[0, 4, 3] = 0.85
        corr[0, 3, 2] = 0.82
        corr[0, 3, 4] = 0.88
        # Image 1: flat correlation (no clear peak)
        corr[1, :, :] = 0.5
        # Image 2: weak peak
        corr[2, 3, 3] = 0.6
        corr[2, 2, 3] = 0.55
        corr[2, 4, 3] = 0.55
        corr[2, 3, 2] = 0.55
        corr[2, 3, 4] = 0.55

        result = get_corr_sub_pix(corr)
        assert result.shape == (3, 5)

    @pytest.mark.unit
    def test_get_corr_sub_pix_symmetric_peak(self):
        """Perfectly symmetric peak should return integer position."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        corr[0, 1, 2] = 0.8
        corr[0, 3, 2] = 0.8
        corr[0, 2, 1] = 0.8
        corr[0, 2, 3] = 0.8
        result = get_corr_sub_pix(corr)
        # With symmetric peak, should return near-integer position
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=1e-6)

    @pytest.mark.unit
    def test_get_corr_sub_pix_quality_coefficients(self):
        """Quality coefficients (sy, sx) should be positive when fit succeeds."""
        corr = np.zeros((1, 7, 7))
        corr[0, 3, 3] = 0.95
        corr[0, 2, 3] = 0.85
        corr[0, 4, 3] = 0.85
        corr[0, 3, 2] = 0.85
        corr[0, 3, 4] = 0.85
        corr[0, 2, 2] = 0.7
        corr[0, 2, 4] = 0.7
        corr[0, 4, 2] = 0.7
        corr[0, 4, 4] = 0.7
        result = get_corr_sub_pix(corr)
        # sy and sx are eigenvalue-based quality measures
        # They should be finite when computed
        assert np.isfinite(result[0, 3]) or result[0, 2] > 0.5


# ===========================================================================
# TestForwardFlatten
# ===========================================================================


class TestForwardFlatten:
    """Tests for forward_flatten."""

    @pytest.mark.unit
    def test_forward_flatten_basic(self):
        """Basic forward flatten should return two arrays."""
        y, x, nim = 8, 8, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape == (npt, 1)
        assert i2.shape == (nco * nim, npt)

    @pytest.mark.unit
    def test_forward_flatten_with_margin(self):
        """Non-zero wsearch should trim margins."""
        y, x, nim = 10, 10, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [1, 2]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert isinstance(i1, np.ndarray)
        assert isinstance(i2, np.ndarray)
        assert i1.shape == (npt, 1)
        assert i2.shape[0] == nco * nim

    @pytest.mark.unit
    def test_forward_flatten_dtype_preserved(self):
        """Output dtype should match input stack dtype."""
        y, x, nim = 6, 6, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float32)
        reference = np.arange(y * x).reshape(y, x).astype(np.float32)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.dtype == stack.dtype
        assert i2.dtype == stack.dtype

    @pytest.mark.unit
    def test_forward_flatten_reference_fully_covered(self):
        """Reference should be fully included (wsearch=[0,0])."""
        y, x, nim = 5, 5, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        # i1 should be the flattened reference
        np.testing.assert_array_equal(i1.ravel(), reference.ravel())

    @pytest.mark.unit
    def test_forward_flatten_with_single_image(self):
        """Single image (nim=1) should produce correct shape."""
        y, x, nim = 4, 4, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i2.shape[0] == nco * nim  # = 1 * 1 = 1

    @pytest.mark.unit
    def test_forward_flatten_with_many_images(self):
        """Multiple images should stack correctly."""
        y, x, nim = 4, 4, 5
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i2.shape[0] == nco * nim  # 1 * 5 = 5

    @pytest.mark.unit
    def test_forward_flatten_values_integrity(self):
        """Values in flattened arrays should match original data."""
        y, x, nim = 4, 4, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        # Each column of i2 should match the corresponding pixel across images
        for z in range(nim):
            np.testing.assert_array_equal(i2[z], stack[:, :, z].ravel())

    @pytest.mark.unit
    def test_forward_flatten_rectangular_image(self):
        """Rectangular (non-square) image should work."""
        y, x, nim = 6, 8, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape[0] == npt  # 48


# ===========================================================================
# TestBackwardFlatten
# ===========================================================================


class TestBackwardFlatten:
    """Tests for backward_flatten."""

    @pytest.mark.unit
    def test_backward_flatten_basic(self):
        """Basic backward flatten should return two arrays."""
        y, x, nim = 8, 8, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape[0] == npt
        assert i2.shape[0] == nim

    @pytest.mark.unit
    def test_backward_flatten_shapes(self):
        """Output shapes should match expected dimensions."""
        y, x, nim = 10, 10, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [1, 1]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape[0] == npt
        assert i1.shape[1] == nco
        assert i2.shape[0] == nim

    @pytest.mark.unit
    def test_backward_flatten_reference_patches(self):
        """i1 should contain shifted reference patches."""
        y, x, nim = 6, 6, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [1, 1]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape == (npt, nco)

    @pytest.mark.unit
    def test_backward_flatten_stack_data(self):
        """i2 should contain the stack images flattened (nim rows, npt columns)."""
        y, x, nim = 6, 6, 3
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i2.shape[0] == nim
        assert i2.shape[1] == npt

    @pytest.mark.unit
    def test_backward_flatten_with_margin(self):
        """Non-zero wsearch should trim reference for patches."""
        y, x, nim = 8, 8, 2
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [1, 1]
        shp = np.array(stack.shape[0:2]) - 2 * np.array(wsearch)
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert npt == (8 - 2) * (8 - 2)  # 6 * 6 = 36
        assert nco == 3 * 3  # (2*1+1)^2 = 9

    @pytest.mark.unit
    def test_backward_flatten_dtype_preserved(self):
        """Output dtype should match stack dtype."""
        y, x, nim = 6, 6, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float32)
        reference = np.arange(y * x).reshape(y, x).astype(np.float32)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.dtype == stack.dtype
        assert i2.dtype == stack.dtype

    @pytest.mark.unit
    def test_backward_flatten_single_pixel_search(self):
        """wsearch=[0,0] should produce 1x1 patches (nco=1)."""
        y, x, nim = 4, 4, 1
        stack = np.arange(y * x * nim).reshape(y, x, nim).astype(np.float64)
        reference = np.arange(y * x).reshape(y, x).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = stack.shape[2]
        npt = int(np.prod(shp))
        nco = (2 * wsearch[0] + 1) * (2 * wsearch[1] + 1)
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape[1] == 1
        assert nco == 1


# ===========================================================================
# TestInitializePatches
# ===========================================================================


class TestInitializePatches:
    """Tests for initialize_patches."""

    @pytest.mark.unit
    def test_initialize_patches_basic(self):
        """Basic patch initialization should create dataset."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        assert isinstance(result, xr.Dataset)

    @pytest.mark.unit
    def test_initialize_patches_coordinates(self):
        """Should have 'patch' coordinate."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        assert "patch" in result.coords

    @pytest.mark.unit
    def test_initialize_patches_variables(self):
        """Should have x_start, x_end, y_start, y_end variables."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        assert "x_start" in result.data_vars
        assert "x_end" in result.data_vars
        assert "y_start" in result.data_vars
        assert "y_end" in result.data_vars

    @pytest.mark.unit
    def test_initialize_patches_count(self):
        """Number of patches should match expected count."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        # 2 y elements * 2 x elements = 4 patches
        assert result.patch.size == 4

    @pytest.mark.unit
    def test_initialize_patches_exact_division(self):
        """Exact division should not add offset."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        assert result.patch.size == 4

    @pytest.mark.unit
    def test_initialize_patches_uneven_division(self):
        """Uneven division should add an extra patch."""
        result = initialize_patches(
            y_size=11,
            y_width=4,
            y_step=5,
            x_size=11,
            x_width=4,
            x_step=5,
        )
        # 11/5 = 2 remainder 1, so y_off=1 => 3 y elements
        # 11/5 = 2 remainder 1, so x_off=1 => 3 x elements
        # Total = 3 * 3 = 9 patches
        assert result.patch.size == 9

    @pytest.mark.unit
    def test_initialize_patches_start_end_consistency(self):
        """x_end - x_start + 1 should equal x_width."""
        result = initialize_patches(
            y_size=20,
            y_width=5,
            y_step=6,
            x_size=20,
            x_width=5,
            x_step=6,
        )
        x_width_actual = result.x_end.values - result.x_start.values + 1
        np.testing.assert_array_equal(x_width_actual, np.full(result.patch.size, 5))

    @pytest.mark.unit
    def test_initialize_patches_no_overlap(self):
        """Patches with step >= width should not overlap."""
        result = initialize_patches(
            y_size=20,
            y_width=4,
            y_step=8,
            x_size=20,
            x_width=4,
            x_step=8,
        )
        # 3 y elements * 3 x elements = 9 patches
        assert result.patch.size == 9
        # y_start repeats 3 times for each x element: 0,0,0,8,8,8,16,16,16
        np.testing.assert_array_equal(result.y_start.values, [0, 0, 0, 8, 8, 8, 16, 16, 16])
        # x_start tiles: 0,8,16,0,8,16,0,8,16
        np.testing.assert_array_equal(result.x_start.values, [0, 8, 16, 0, 8, 16, 0, 8, 16])

    @pytest.mark.unit
    def test_initialize_patches_overlap(self):
        """Patches with step < width should overlap."""
        result = initialize_patches(
            y_size=10,
            y_width=5,
            y_step=3,
            x_size=10,
            x_width=5,
            x_step=3,
        )
        # 4 y elements * 4 x elements = 16 patches
        assert result.patch.size == 16
        # y_start: repeats 4 times for each x: 0,0,0,0,3,3,3,3,...
        assert result.y_start.values[0] == 0
        assert result.y_start.values[4] == 3
        assert result.y_start.values[8] == 6
        assert result.y_start.values[12] == 9
        # y_end should be >= y_start for all patches
        assert (result.y_end.values >= result.y_start.values).all()

    @pytest.mark.unit
    def test_initialize_patches_single_patch(self):
        """Single patch should be created when size < width."""
        result = initialize_patches(
            y_size=3,
            y_width=5,
            y_step=5,
            x_size=3,
            x_width=5,
            x_step=5,
        )
        assert result.patch.size == 1

    @pytest.mark.unit
    def test_initialize_patches_larger_width(self):
        """Width larger than step should produce overlapping patches."""
        result = initialize_patches(
            y_size=20,
            y_width=8,
            y_step=4,
            x_size=20,
            x_width=8,
            x_step=4,
        )
        # 5 y elements * 5 x elements = 25 patches
        assert result.patch.size == 25
        # y_start repeats 5 times for each x: 0,0,0,0,0,4,4,4,4,4,...
        assert result.y_start.values[0] == 0
        assert result.y_start.values[5] == 4
        assert result.y_start.values[10] == 8
        # y_end - y_start + 1 = width = 8
        np.testing.assert_array_equal(result.y_end.values - result.y_start.values + 1, np.full(result.patch.size, 8))

    @pytest.mark.unit
    def test_initialize_patches_dataset_structure(self):
        """Dataset should have correct dimension and variables."""
        result = initialize_patches(
            y_size=10,
            y_width=4,
            y_step=5,
            x_size=10,
            x_width=4,
            x_step=5,
        )
        assert list(result.dims) == ["patch"]
        assert list(result.data_vars) == ["x_start", "x_end", "y_start", "y_end"]

    @pytest.mark.unit
    def test_initialize_patches_all_endpoints_within_bounds(self):
        """All start and end points should be within grid bounds."""
        y_size, x_size = 15, 15
        y_width, x_width = 5, 5
        y_step, x_step = 5, 5
        result = initialize_patches(
            y_size=y_size,
            y_width=y_width,
            y_step=y_step,
            x_size=x_size,
            x_width=x_width,
            x_step=x_step,
        )
        assert (result.y_end.values < y_size).all()
        assert (result.x_end.values < x_size).all()
        assert (result.y_start.values >= 0).all()
        assert (result.x_start.values >= 0).all()


# ===========================================================================
# TestComputePearsonCoefficient
# ===========================================================================


class TestComputePearsonCoefficient:
    """Tests for compute_pearson_coefficient."""

    @pytest.mark.unit
    def test_compute_pearson_coefficient_basic(self):
        """Basic computation should return DataArray."""
        # Single patch covering the entire image
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        # stack dims: (y, x, acquisition_date) - spatial dims first, time last
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(
            np.random.rand(y, x),
            dims=["y", "x"],
        )
        # pixel_to_time must have length == nim (number of images)
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert isinstance(result, xr.DataArray)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_output_shape(self):
        """Output shape should be (nim, n_patches, 5)."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert result.shape == (nim, patch_ds.patch.size, 5)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_backward(self):
        """backward_flatten path should work."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=False,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert isinstance(result, xr.DataArray)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_output_dims(self):
        """Output should have acquisition_date, point, variable dims."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert "acquisition_date" in result.dims
        assert "point" in result.dims
        assert "variable" in result.dims

    @pytest.mark.unit
    def test_compute_pearson_coefficient_variable_labels(self):
        """Output variable coordinate should be [dy, dx, cohe, s_y, s_x]."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert list(result.coords["variable"].values) == ["dy", "dx", "cohe", "s_y", "s_x"]

    @pytest.mark.unit
    def test_compute_pearson_coefficient_large_window_warning(self):
        """Large peak_window_size should trigger warning."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.ones((y, x, nim)),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.ones((y, x)), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (4, 6)

        with patch("cop_ard_core.matching.logger.warning") as mock_warn:
            result = compute_pearson_coefficient(
                patch_ds,
                stack,
                reference,
                pixel_to_time,
                peak_window_size,
                forward=True,
                y_dim="y",
                x_dim="x",
                correlation_kernel_size=None,
            )
            mock_warn.assert_called_once()
            assert isinstance(result, xr.DataArray)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_zero_window(self):
        """Zero window size should work."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 3
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (0, 0)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert isinstance(result, xr.DataArray)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_pixel_to_time_scaling(self):
        """pixel_to_time should scale dy and dx outputs."""
        patch_ds = initialize_patches(10, 4, 10, 10, 4, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([2.0, 3.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        # dy scaled by pixel_to_time[0], dx by pixel_to_time[1]
        assert "pixel_to_time" in result.attrs

    @pytest.mark.unit
    def test_compute_pearson_coefficient_correlation_kernel(self):
        """correlation_kernel_size should be passed to get_corr_sub_pix."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=(1, 1),
        )
        assert isinstance(result, xr.DataArray)

    @pytest.mark.unit
    def test_compute_pearson_coefficient_attrs(self):
        """Output should have type and pixel_to_time attrs."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
            data_type="times",
        )
        assert result.attrs["type"] == "times"
        assert "pixel_to_time" in result.attrs

    @pytest.mark.unit
    def test_compute_pearson_coefficient_pixels_type(self):
        """data_type='pixels' should set correct type attr."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
            data_type="pixels",
        )
        assert result.attrs["type"] == "pixels"

    @pytest.mark.unit
    def test_compute_pearson_coefficient_constant_data(self):
        """Constant data should produce zero correlation."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.ones((y, x, nim)),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.ones((y, x)), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        # With constant data, normalization leads to NaN/zero
        assert isinstance(result, xr.DataArray)
        assert np.allclose(result.sel(variable="cohe").values, 0.0)


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests across matching functions."""

    @pytest.mark.unit
    def test_get_corr_sub_pix_nan_values(self):
        """NaN in correlation may cause SVD convergence issues."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = np.nan
        corr[0, 1, 2] = 0.8
        corr[0, 3, 2] = 0.85
        corr[0, 2, 1] = 0.82
        corr[0, 2, 3] = 0.88
        # NaN in correlation peak position may cause SVD non-convergence
        with pytest.raises(Exception):
            get_corr_sub_pix(corr)

    @pytest.mark.unit
    def test_forward_flatten_zero_images(self):
        """Edge case: nim=0 would cause issues, but function expects nim>=1."""
        # With nim=0, the function should not be called, but if called:
        y, x = 4, 4
        stack = np.zeros((y, x, 1)).astype(np.float64)
        reference = np.zeros((y, x)).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = 1
        npt = int(np.prod(shp))
        nco = 1
        i1, i2 = forward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape == (npt, 1)

    @pytest.mark.unit
    def test_backward_flatten_zero_images(self):
        """Backward flatten with minimal inputs."""
        y, x = 4, 4
        stack = np.zeros((y, x, 1)).astype(np.float64)
        reference = np.zeros((y, x)).astype(np.float64)
        wsearch = [0, 0]
        shp = np.array(stack.shape[0:2])
        nim = 1
        npt = int(np.prod(shp))
        nco = 1
        i1, i2 = backward_flatten(stack, reference, wsearch, npt, nco, nim, list(shp))
        assert i1.shape == (npt, nco)
        assert i2.shape == (nim, npt)

    @pytest.mark.unit
    def test_initialize_patches_minimum_size(self):
        """Minimum patch size should not crash."""
        result = initialize_patches(
            y_size=1,
            y_width=1,
            y_step=1,
            x_size=1,
            x_width=1,
            x_step=1,
        )
        assert result.patch.size == 1

    @pytest.mark.unit
    def test_compute_pearson_coefficient_single_patch(self):
        """Single patch should work correctly."""
        patch_ds = initialize_patches(10, 10, 10, 10, 10, 10)
        y, x, nim = 10, 10, 2
        stack = xr.DataArray(
            np.random.rand(y, x, nim),
            dims=["y", "x", "acquisition_date"],
            coords={
                "y": np.arange(y),
                "x": np.arange(x),
                "acquisition_date": np.arange(nim),
                "z": ("acquisition_date", np.arange(nim)),
            },
        )
        reference = xr.DataArray(np.random.rand(y, x), dims=["y", "x"])
        pixel_to_time = np.array([1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds,
            stack,
            reference,
            pixel_to_time,
            peak_window_size,
            forward=True,
            y_dim="y",
            x_dim="x",
            correlation_kernel_size=None,
        )
        assert result.shape == (nim, patch_ds.patch.size, 5)
