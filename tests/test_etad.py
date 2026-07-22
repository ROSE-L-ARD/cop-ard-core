"""Unit tests for cop_ard_core.etad."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.etad import apply_etad_corrections


class TestApplyEtadCorrections:
    """Tests for apply_etad_corrections."""

    def test_apply_etad_corrections_empty_grid(self):
        """Empty correction grid should return zeros."""
        az = xr.DataArray(np.arange(4), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(4), dims=["slant_range_time"])
        etad_grid = xr.Dataset()  # empty
        dtaz = np.timedelta64(1, "ns")
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert daz == np.timedelta64(0, "ns")
        assert drg.shape == (1,)

    def test_apply_etad_corrections_with_grid(self):
        """Should interpolate from correction grid."""
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02", "2020-01-01T00:00:03"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        rg = xr.DataArray(np.array([0.0, 0.5, 1.0, 1.5]), dims=["slant_range_time"])
        etad_grid = xr.Dataset({
            "azcorr": xr.DataArray(
                np.ones((4, 4)) * 0.001,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "rgcorr": xr.DataArray(
                np.ones((4, 4)) * 0.01,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "azimuth_time": (["azimuth_time"], np.arange(4)),
            "slant_range_time": (["slant_range_time"], np.arange(4)),
        })
        dtaz = 1.0  # azimuth time step as float (nanoseconds)
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert isinstance(daz, np.ndarray)
        assert isinstance(drg, np.ndarray)

    def test_apply_etad_corrections_output_shape_az(self):
        """Azimuth correction output should match input shape."""
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02", "2020-01-01T00:00:03"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        rg = xr.DataArray(np.array([0.0, 0.5, 1.0, 1.5]), dims=["slant_range_time"])
        etad_grid = xr.Dataset({
            "azcorr": xr.DataArray(
                np.ones((4, 4)) * 0.001,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "rgcorr": xr.DataArray(
                np.ones((4, 4)) * 0.01,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "azimuth_time": (["azimuth_time"], np.arange(4)),
            "slant_range_time": (["slant_range_time"], np.arange(4)),
        })
        dtaz = 1.0
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert daz.shape == az.shape

    def test_apply_etad_corrections_output_shape_rang(self):
        """Range correction output should match input shape."""
        az = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02", "2020-01-01T00:00:03"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        rg = xr.DataArray(np.array([0.0, 0.5, 1.0, 1.5]), dims=["slant_range_time"])
        etad_grid = xr.Dataset({
            "azcorr": xr.DataArray(
                np.ones((4, 4)) * 0.001,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "rgcorr": xr.DataArray(
                np.ones((4, 4)) * 0.01,
                dims=["azimuth_time", "slant_range_time"],
                coords={"azimuth_time": np.arange(4), "slant_range_time": np.arange(4)},
            ),
            "azimuth_time": (["azimuth_time"], np.arange(4)),
            "slant_range_time": (["slant_range_time"], np.arange(4)),
        })
        dtaz = 1.0
        dtrg = 0.5
        rg0 = 0.0
        az0 = np.timedelta64(0, "ns")
        daz, drg = apply_etad_corrections(az, rg, etad_grid, dtaz, dtrg, rg0, az0)
        assert drg.shape == rg.shape
