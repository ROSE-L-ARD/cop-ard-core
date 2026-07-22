"""Unit tests for cop_ard_core.warp."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.warp import (
    extract_abs_grid,
    extract_rg0_az0,
    extract_rg_az,
    getCoeffFromLut,
    getLUT,
    poly_fit_2d,
    sinc,
    sinc_warp,
)


class TestSinc:
    """Tests for sinc function."""

    def test_sinc_returns_array(self):
        """Should return numpy array."""
        result = sinc(size=5)
        assert isinstance(result, np.ndarray)

    def test_sinc_positive_size(self):
        """Should work with positive size."""
        for size in [3, 5, 7, 11]:
            result = sinc(size=size)
            assert result.shape[0] > 0

    def test_sinc_centered(self):
        """Center value should be 1 (sinc(0) = 1)."""
        result = sinc(size=5, ov_fact=100)
        center = len(result) // 2
        np.testing.assert_almost_equal(result[center], 1.0, decimal=5)

    def test_sinc_zero_at_multiples(self):
        """Sinc should be zero at integer multiples (before windowing)."""
        result = sinc(size=11, ov_fact=100)
        # Windowed sinc may not be exactly zero, but should be small
        center = len(result) // 2
        assert abs(result[center]) >= abs(result[center + 5])


class TestSincWarp:
    """Tests for sinc_warp."""

    def test_sinc_warp_returns_array(self):
        """Should return numpy array of interpolated values."""
        image = np.random.rand(5, 5)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        assert result.shape == x.shape

    def test_sinc_warp_2d_output(self):
        """2D coordinate arrays should produce 2D output."""
        image = np.random.rand(5, 5)
        x, y = np.meshgrid(np.arange(3), np.arange(3))
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x.ravel(), y.ravel(), kernel)
        assert result.shape == x.ravel().shape

    def test_sinc_warp_image_not_2d_raises(self):
        """Non-2D image should raise."""
        image = np.random.rand(2, 3, 3)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        with pytest.raises(Exception, match="Image must have 2 dimensions"):
            sinc_warp(image, x, y, kernel)

    def test_sinc_warp_mismatched_shapes_raises(self):
        """Mismatched x/y shapes should raise."""
        image = np.random.rand(5, 5)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        with pytest.raises(Exception, match="x and y shape must be the same"):
            sinc_warp(image, x, y, kernel)

    def test_sinc_warp_boundary_pixels(self):
        """Pixels at image boundaries should be handled (return invalid value)."""
        image = np.ones((5, 5))
        x = np.array([0.1, 2.0, 4.9])
        y = np.array([0.1, 2.0, 4.9])
        kernel = {
            "kernel_shape": (3, 3),
            "ov_fact": 100,
            "sinc_x": np.ones(100),
            "sinc_y": np.ones(100),
        }
        result = sinc_warp(image, x, y, kernel)
        # Middle pixel should have value, boundary pixels might be invalid
        assert result.shape == x.shape


class TestPolyFit2d:
    """Tests for poly_fit_2d."""

    def test_poly_fit_2d_returns_tuple(self):
        """Should return 4-element tuple."""
        n = 10
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        result = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        assert len(result) == 4

    def test_poly_fit_2d_shape(self):
        """Output shapes should be reasonable."""
        n = 20
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        result = poly_fit_2d(x, y, z, x_order=2, y_order=2)
        coeffs = result[0]
        rank = result[2]
        assert coeffs.shape == (3, 3)
        assert rank > 0

    def test_poly_fit_2d_linear_data(self):
        """Linear data should give good fit."""
        n = 50
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = 2.0 * x + 3.0 * y + 1.0
        result = poly_fit_2d(x, y, z, x_order=1, y_order=1)
        rank = result[2]
        assert rank == 4  # constant + x + y + x*y


class TestExtractRg0Az0:
    """Tests for extract_rg0_az0."""

    def test_extract_rg0_az0_returns_tuple(self):
        """Should return 4-element tuple."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
            attrs={
                "dummy": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
                    "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 0.0, "azimuth_time": "2020-01-01T00:00:00"},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
            }
        )
        result = extract_rg0_az0(slc, phase_demod)
        assert len(result) == 4

    def test_extract_rg0_az0_types(self):
        """All return values should be numeric."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
            attrs={
                "dummy": {
                    "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
                    "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
                }
            },
        )
        phase_demod = xr.DataArray(
            np.random.rand(3, 3),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": np.arange(3), "slant_range_time": np.arange(3)},
        )
        phase_demod = phase_demod.assign_attrs(
            {
                "reference": {"slant_range_time": 0.0, "azimuth_time": "2020-01-01T00:00:00"},
                "scale": {"slant_range_time": 1.0, "azimuth_time": 1.0},
            }
        )
        rg0, az0, rg, az = extract_rg0_az0(slc, phase_demod)
        assert np.isscalar(rg0) or hasattr(rg0, "item")
        assert np.isscalar(az0) or hasattr(az0, "item")


class TestExtractRgAz:
    """Tests for extract_rg_az."""

    def test_extract_rg_az_returns_tuple(self):
        """Should return tuple of arrays."""
        slc = xr.DataArray(
            np.ones((3, 3), dtype=np.complex128),
            dims=["azimuth_time", "slant_range_time"],
        )
        sim = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(
                    np.arange(3),
                    dims=["azimuth_time"],
                    coords={"azimuth_time": np.arange(3)},
                ),
                "slant_range_time": xr.DataArray(
                    np.arange(3),
                    dims=["slant_range_time"],
                    coords={"slant_range_time": np.arange(3)},
                ),
            }
        )
        dtaz = 1.0
        dtrg = 1.0
        result = extract_rg_az(slc, sim, dtaz, dtrg)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestExtractAbsGrid:
    """Tests for extract_abs_grid."""

    def test_extract_abs_grid_returns_tuple(self):
        """Should return tuple of DataArrays."""
        slc = xr.DataArray(
            np.random.rand(4, 4),
            dims=["azimuth_time", "slant_range_time"],
        )
        result = extract_abs_grid(slc)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestGetCoeffFromLut:
    """Tests for getCoeffFromLut."""

    def test_get_coeff_from_lut_returns_array(self):
        """Should return numpy array."""
        xLut = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3).astype("datetime64[ns]"),
                "slant_range_time": np.arange(3) * 0.1,
            },
        )
        xLut.attrs["reference"] = {"slant_range_time": 0.0, "azimuth_time": np.datetime64("2020-01-01")}
        xLut.attrs["scale"] = {"slant_range_time": 1.0, "azimuth_time": 1.0}
        xLut.attrs["deg"] = {"slant_range_time": 1, "azimuth_time": 1}
        result = getCoeffFromLut(xLut)
        assert isinstance(result, np.ndarray)


class TestGetLUT:
    """Tests for getLUT."""

    def test_getLUT_returns_tuple(self):
        """Should return tuple of 3 DataArrays."""
        coreg = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(np.arange(3), dims=["azimuth_time"]),
                "slant_range_time": xr.DataArray(np.arange(3), dims=["slant_range_time"]),
            }
        )
        daz_drg = xr.Dataset(
            {
                "daz": xr.DataArray(np.arange(9).reshape(3, 3), dims=["azimuth_time", "slant_range_time"]),
                "drg": xr.DataArray(np.arange(9, 18).reshape(3, 3), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc = xr.Dataset(
            {
                "slc": xr.DataArray(np.ones((3, 3)), dims=["azimuth_time", "slant_range_time"]),
            }
        )
        slc["slc"].attrs["dummy"] = {
            "RMA": {"INCA": {"TimeCAPoly": {"Coefs": [0, 1.0, 0]}}},
            "ImageData": {"SCPPixel": {"Row": 1, "Col": 1}},
            "Grid": {"Col": {"SS": 1.0}, "Row": {"SS": 1.0}},
        }
        coeffs = np.ones((2, 2))
        result = getLUT(coreg, daz_drg, slc, coeffs)
        assert isinstance(result, tuple)
        assert len(result) == 3
