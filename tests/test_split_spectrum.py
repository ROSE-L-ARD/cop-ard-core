"""Unit tests for cop_ard_core.split_spectrum."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.split_spectrum import (
    fit_ionosphere,
    reconstruct_iono_surface,
    scale_azimuth_time,
    split_spectrum_core,
)


class TestSplitSpectrumCore:
    """Tests for split_spectrum_core."""

    def test_split_spectrum_core_returns_tuple(self):
        """Should return a tuple of two DataArrays."""
        ph_l = xr.DataArray(np.ones((3, 3)), dims=["az", "rg"])
        ph_h = xr.DataArray(np.ones((3, 3)), dims=["az", "rg"])
        phase_iono, phase_nd = split_spectrum_core(ph_l, ph_h, f0=5.4e9, f_l_tot=5.3e9, f_h_tot=5.5e9)
        assert isinstance(phase_iono, xr.DataArray)
        assert isinstance(phase_nd, xr.DataArray)

    def test_split_spectrum_core_shapes(self):
        """Output shapes should match input shapes."""
        ph_l = xr.DataArray(np.ones((4, 4)), dims=["az", "rg"])
        ph_h = xr.DataArray(np.ones((4, 4)), dims=["az", "rg"])
        phase_iono, phase_nd = split_spectrum_core(ph_l, ph_h, f0=5.4e9, f_l_tot=5.3e9, f_h_tot=5.5e9)
        assert phase_iono.shape == (4, 4)
        assert phase_nd.shape == (4, 4)

    def test_split_spectrum_core_zero_phases(self):
        """Zero input phases should give zero output."""
        ph_l = xr.DataArray(np.zeros((3, 3)), dims=["az", "rg"])
        ph_h = xr.DataArray(np.zeros((3, 3)), dims=["az", "rg"])
        phase_iono, phase_nd = split_spectrum_core(ph_l, ph_h, f0=5.4e9, f_l_tot=5.3e9, f_h_tot=5.5e9)
        np.testing.assert_allclose(phase_iono.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(phase_nd.values, 0.0, atol=1e-10)


class TestScaleAzimuthTime:
    """Tests for scale_azimuth_time."""

    def test_scale_azimuth_time_basic(self):
        """Basic scaling should return DataArray."""
        az_raw = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        result = scale_azimuth_time(az_raw, az, H_orb=800000.0, H_target=300000.0)
        assert isinstance(result, xr.DataArray)

    def test_scale_azimuth_time_same_height(self):
        """Same H_orb and H_target should give no change."""
        az_raw = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        result = scale_azimuth_time(az_raw, az, H_orb=800000.0, H_target=800000.0)
        # Difference should be zero since ratio = 1
        diff = (result - az_raw).astype(float)
        np.testing.assert_array_almost_equal(diff.values, 0.0)


class TestFitIonosphere:
    """Tests for fit_ionosphere."""

    def test_fit_ionosphere_returns_dataarray(self):
        """Should return xr.DataArray with coefficients."""
        az_size, sr_size = 3, 5
        phase_iono = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(az_size),
                "slant_range_time": np.arange(sr_size),
            },
        )
        # azimuth_time_raw must be 2D, matching phase_iono shape
        azimuth_time_raw = xr.DataArray(
            np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(az_size),
                "slant_range_time": np.arange(sr_size),
            },
        )
        phase_iono = phase_iono.assign_coords({
            "azimuth_time_raw": (["azimuth_time", "slant_range_time"],
                                 np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float)),
        })
        result = fit_ionosphere(
            phase_iono, azimuth_time_raw,
            H_iono=300000.0, H_orb=800000.0, degree=1,
        )
        assert isinstance(result, xr.DataArray)

    def test_fit_ionosphere_with_attrs(self):
        """Should have offset and scale_fact attributes."""
        az_size, sr_size = 3, 5
        phase_iono = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(az_size),
                "slant_range_time": np.arange(sr_size),
            },
        )
        azimuth_time_raw = xr.DataArray(
            np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(az_size),
                "slant_range_time": np.arange(sr_size),
            },
        )
        phase_iono = phase_iono.assign_coords({
            "azimuth_time_raw": (["azimuth_time", "slant_range_time"],
                                 np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float)),
        })
        result = fit_ionosphere(
            phase_iono, azimuth_time_raw,
            H_iono=300000.0, H_orb=800000.0, degree=1,
        )
        assert "offset" in result.attrs


class TestReconstructIonoSurface:
    """Tests for reconstruct_iono_surface."""

    def test_reconstruct_iono_surface_returns_dataarray(self):
        """Should return xr.DataArray."""
        interf = xr.DataArray(
            np.ones((3, 5)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(5),
            },
        )
        azimuth_time_raw = xr.DataArray(
            np.broadcast_to(np.arange(3)[:, None], (3, 5)).astype(float),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(5),
            },
        )
        # degree=1 gives 3 coefficients (const, x, y) for 2D linear fit
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5]),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0])
        result = reconstruct_iono_surface(
            interf, azimuth_time_raw, coeffs,
            H_iono=300000.0, H_orb=800000.0, degree=1,
        )
        assert isinstance(result, xr.DataArray)

    def test_reconstruct_iono_surface_shape(self):
        """Output shape should match azimuth_time_raw shape."""
        interf = xr.DataArray(
            np.ones((3, 5)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(5),
            },
        )
        azimuth_time_raw = xr.DataArray(
            np.broadcast_to(np.arange(3)[:, None], (3, 5)).astype(float),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(5),
            },
        )
        # degree=1 gives 3 coefficients (const, x, y) for 2D linear fit
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5]), dims=["degree"], coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0])
        result = reconstruct_iono_surface(
            interf, azimuth_time_raw, coeffs,
            H_iono=300000.0, H_orb=800000.0, degree=1,
        )
        assert result.shape == (3, 5)
