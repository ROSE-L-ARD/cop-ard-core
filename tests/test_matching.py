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


class TestGetCorrSubPix:
    """Tests for get_corr_sub_pix."""

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

    def test_get_corr_sub_pix_perfect_peak(self):
        """Peak of 1 should return integer position."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 1.0
        result = get_corr_sub_pix(corr)
        assert result[0, 0] == 0.0  # center position
        assert result[0, 1] == 0.0

    def test_get_corr_sub_pix_dim_int(self):
        """Integer dim should be converted to [dim, dim]."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        result = get_corr_sub_pix(corr, dim=1)
        assert result.shape == (1, 5)

    def test_get_corr_sub_pix_no_dim(self):
        """No dim should default to [1, 1]."""
        corr = np.zeros((1, 5, 5))
        corr[0, 2, 2] = 0.95
        result = get_corr_sub_pix(corr, dim=None)
        assert result.shape == (1, 5)

    def test_get_corr_sub_pix_multiple_images(self):
        """Multiple images should return one row per image."""
        corr = np.zeros((3, 5, 5))
        for i in range(3):
            corr[i, 2, 2] = 0.95
        result = get_corr_sub_pix(corr)
        assert result.shape == (3, 5)

    def test_get_corr_sub_pix_small_box(self):
        """Peak near edge should be handled."""
        corr = np.zeros((1, 5, 5))
        corr[0, 0, 0] = 0.95
        result = get_corr_sub_pix(corr)
        assert result.shape == (1, 5)


class TestForwardFlatten:
    """Tests for forward_flatten."""

    def test_forward_flatten_basic(self):
        """Basic forward flatten should return two arrays."""
        # stack shape: (y, x, nim) - spatial dims first, images last
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


class TestBackwardFlatten:
    """Tests for backward_flatten."""

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


class TestInitializePatches:
    """Tests for initialize_patches."""

    def test_initialize_patches_basic(self):
        """Basic patch initialization should create dataset."""
        result = initialize_patches(
            y_size=10, y_width=4, y_step=5,
            x_size=10, x_width=4, x_step=5,
        )
        assert isinstance(result, xr.Dataset)

    def test_initialize_patches_coordinates(self):
        """Should have 'patch' coordinate."""
        result = initialize_patches(
            y_size=10, y_width=4, y_step=5,
            x_size=10, x_width=4, x_step=5,
        )
        assert "patch" in result.coords

    def test_initialize_patches_variables(self):
        """Should have x_start, x_end, y_start, y_end variables."""
        result = initialize_patches(
            y_size=10, y_width=4, y_step=5,
            x_size=10, x_width=4, x_step=5,
        )
        assert "x_start" in result.data_vars
        assert "x_end" in result.data_vars
        assert "y_start" in result.data_vars
        assert "y_end" in result.data_vars

    def test_initialize_patches_count(self):
        """Number of patches should match expected count."""
        result = initialize_patches(
            y_size=10, y_width=4, y_step=5,
            x_size=10, x_width=4, x_step=5,
        )
        # 2 y elements * 2 x elements = 4 patches
        assert result.patch.size == 4

    def test_initialize_patches_exact_division(self):
        """Exact division should not add offset."""
        result = initialize_patches(
            y_size=10, y_width=4, y_step=5,
            x_size=10, x_width=4, x_step=5,
        )
        assert result.patch.size == 4


class TestComputePearsonCoefficient:
    """Tests for compute_pearson_coefficient."""

    def test_compute_pearson_coefficient_basic(self):
        """Basic computation should return DataArray."""
        patch_ds = initialize_patches(10, 4, 10, 10, 4, 10)
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
        pixel_to_time = np.array([1.0, 1.0, 1.0])
        peak_window_size = (1, 1)
        result = compute_pearson_coefficient(
            patch_ds, stack, reference, pixel_to_time,
            peak_window_size, forward=True,
            y_dim="y", x_dim="x",
            correlation_kernel_size=None,
        )
        assert isinstance(result, xr.DataArray)

    def test_compute_pearson_coefficient_large_window(self):
        """Large peak_window_size should trigger warning."""
        patch_ds = initialize_patches(10, 4, 10, 10, 4, 10)
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
                patch_ds, stack, reference, pixel_to_time,
                peak_window_size, forward=True,
                y_dim="y", x_dim="x",
                correlation_kernel_size=None,
            )
            mock_warn.assert_called_once()
            assert isinstance(result, xr.DataArray)
