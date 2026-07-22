"""Unit tests for cop_ard_core.simul."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from cop_ard_core.simul import (
    backward_geocode,
    get_geo_from_radar_bbox,
    get_position,
    get_radar_bbox,
    get_velocity,
    init_simulation,
    simulate_acquisition,
)


class TestGetPosition:
    """Tests for get_position."""

    def test_get_position_returns_dataarray(self):
        """Should return xr.DataArray with ECEF positions."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {
            "X": np.array([6500000.0, 7500.0]),  # constant + linear
            "Y": np.array([2000000.0, 0.0]),
            "Z": np.array([4000000.0, 0.0]),
        }
        result = get_position(az_time, t0, arp_coeffs)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("axis", "azimuth_time")
        assert result.shape == (3, 2)

    def test_get_position_values(self):
        """Position values should match polynomial evaluation."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {
            "X": np.array([6500000.0, 7500.0]),
            "Y": np.array([2000000.0, 0.0]),
            "Z": np.array([4000000.0, 0.0]),
        }
        result = get_position(az_time, t0, arp_coeffs)
        # At t=0: X = 6500000.0
        # At t=1s: X = 6500000.0 + 7500.0*1 = 6507500.0
        assert result.values[0, 0] == pytest.approx(6500000.0)
        assert result.values[0, 1] == pytest.approx(6507500.0)

    def test_get_position_preserves_coords(self):
        """Should preserve original azimuth_time coordinates."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {"X": np.array([1.0, 1.0]), "Y": np.array([1.0, 1.0]), "Z": np.array([1.0, 1.0])}
        result = get_position(az_time, t0, arp_coeffs)
        assert list(result.coords["azimuth_time"].values) == list(az_time.values)


class TestGetVelocity:
    """Tests for get_velocity."""

    def test_get_velocity_returns_dataarray(self):
        """Should return xr.DataArray with velocity vectors."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {
            "X": np.array([6500000.0, 7500.0, -1.0]),  # constant + linear + quadratic
            "Y": np.array([2000000.0, 0.0, 0.0]),
            "Z": np.array([4000000.0, 0.0, 0.0]),
        }
        result = get_velocity(az_time, t0, arp_coeffs)
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("axis", "azimuth_time")

    def test_get_velocity_constant_position(self):
        """Constant position polynomial should give zero velocity."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {
            "X": np.array([6500000.0, 0.0]),
            "Y": np.array([2000000.0, 0.0]),
            "Z": np.array([4000000.0, 0.0]),
        }
        result = get_velocity(az_time, t0, arp_coeffs)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_get_velocity_linear_position(self):
        """Linear position polynomial should give constant velocity."""
        az_time = xr.DataArray(
            np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]"),
            dims=["azimuth_time"],
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        arp_coeffs = {
            "X": np.array([6500000.0, 7500.0]),
            "Y": np.array([2000000.0, 0.0]),
            "Z": np.array([4000000.0, 0.0]),
        }
        result = get_velocity(az_time, t0, arp_coeffs)
        # Velocity = derivative of position = 7500.0 m/s (constant)
        np.testing.assert_allclose(result.values[0, :], 7500.0, atol=0.1)


class TestInitSimulation:
    """Tests for init_simulation."""

    def test_init_simulation_returns_dataset(self):
        """Should return xr.Dataset."""
        dates = np.array(["2020-01-01", "2020-01-13"])
        crs = CRS.from_epsg(32633)
        x = np.arange(10)
        y = np.arange(10)
        result = init_simulation(dates, crs, x, y)
        assert isinstance(result, xr.Dataset)

    def test_init_simulation_has_variables(self):
        """Should have azimuth_time, slant_range_time, dem_distance, h."""
        dates = np.array(["2020-01-01"])
        crs = CRS.from_epsg(32633)
        x = np.arange(3)
        y = np.arange(3)
        result = init_simulation(dates, crs, x, y)
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" in result.data_vars
        assert "dem_distance" in result.data_vars
        assert "h" in result.data_vars

    def test_init_simulation_shape(self):
        """Shapes should match inputs."""
        n_dates = 2
        x_size = 5
        y_size = 4
        dates = np.arange(n_dates)
        crs = CRS.from_epsg(32633)
        x = np.arange(x_size)
        y = np.arange(y_size)
        result = init_simulation(dates, crs, x, y)
        assert result.azimuth_time.shape == (n_dates, y_size, x_size)
        assert result.h.shape == (y_size, x_size)

    def test_init_simulation_crs(self):
        """Should have CRS set."""
        dates = np.array(["2020-01-01"])
        crs = CRS.from_epsg(32633)
        x = np.arange(3)
        y = np.arange(3)
        result = init_simulation(dates, crs, x, y)
        assert result.rio.crs is not None


class TestGetGeoFromRadarBbox:
    """Tests for get_geo_from_radar_bbox."""

    def test_get_geo_from_radar_bbox_overlapping(self):
        """Overlapping radar bbox should return geographic bbox."""
        data_ml = {
            "min": xr.Dataset({
                "azimuth_time": xr.DataArray(
                    np.array([[np.datetime64("2020-01-01T00:00:00")]]),
                    dims=["y", "x"],
                ),
                "slant_range_time": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
                "y": xr.DataArray(np.array([[40.0]]), dims=["y", "x"]),
                "x": xr.DataArray(np.array([[-75.0]]), dims=["y", "x"]),
            }),
            "max": xr.Dataset({
                "azimuth_time": xr.DataArray(
                    np.array([[np.datetime64("2020-01-13T00:00:00")]]),
                    dims=["y", "x"],
                ),
                "slant_range_time": xr.DataArray(np.array([[10.0]]), dims=["y", "x"]),
                "y": xr.DataArray(np.array([[45.0]]), dims=["y", "x"]),
                "x": xr.DataArray(np.array([[-70.0]]), dims=["y", "x"]),
            }),
        }
        az_min = np.datetime64("2020-01-01T00:00:00")
        az_max = np.datetime64("2020-01-13T00:00:00")
        bbox_radar = (az_min, 0.0, az_max, 10.0)
        result = get_geo_from_radar_bbox(data_ml, bbox_radar)
        assert result is not None
        assert len(result) == 4

    def test_get_geo_from_radar_bbox_no_overlap(self):
        """Non-overlapping radar bbox should return None."""
        data_ml = {
            "min": xr.Dataset({
                "azimuth_time": xr.DataArray(
                    np.array([[np.datetime64("2020-01-01T00:00:00")]]),
                    dims=["y", "x"],
                ),
                "slant_range_time": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
                "y": xr.DataArray(np.array([[40.0]]), dims=["y", "x"]),
                "x": xr.DataArray(np.array([[-75.0]]), dims=["y", "x"]),
            }),
            "max": xr.Dataset({
                "azimuth_time": xr.DataArray(
                    np.array([[np.datetime64("2020-01-13T00:00:00")]]),
                    dims=["y", "x"],
                ),
                "slant_range_time": xr.DataArray(np.array([[10.0]]), dims=["y", "x"]),
                "y": xr.DataArray(np.array([[45.0]]), dims=["y", "x"]),
                "x": xr.DataArray(np.array([[-70.0]]), dims=["y", "x"]),
            }),
        }
        az_min = np.datetime64("2019-01-01T00:00:00")
        az_max = np.datetime64("2019-01-02T00:00:00")
        bbox_radar = (az_min, 100.0, az_max, 200.0)
        result = get_geo_from_radar_bbox(data_ml, bbox_radar)
        assert result is None


class TestGetRadarBbox:
    """Tests for get_radar_bbox."""

    def test_get_radar_bbox_basic(self):
        """Should return bounding box with margins."""
        az = xr.DataArray(np.arange(10), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(10), dims=["slant_range_time"])
        margin = {"azimuth_time": 1.0, "slant_range_time": 2.0}
        result = get_radar_bbox(az, rg, margin)
        assert len(result) == 4
        # max of arange(10) is 9, so 9+1=10 and 9+2=11
        assert result[2] == 10.0
        assert result[3] == 11.0

    def test_get_radar_bbox_no_margin(self):
        """Zero margin should return exact min/max."""
        az = xr.DataArray(np.arange(5), dims=["azimuth_time"])
        rg = xr.DataArray(np.arange(5), dims=["slant_range_time"])
        margin = {"azimuth_time": 0.0, "slant_range_time": 0.0}
        result = get_radar_bbox(az, rg, margin)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 4.0
        assert result[3] == 4.0


class TestBackwardGeocode:
    """Tests for backward_geocode."""

    def test_backward_geocode_missing_reference_time(self):
        """Missing referenceTime should raise."""
        dem_ecef = xr.DataArray(
            np.random.rand(3, 3, 3),
            dims=["axis", "y", "x"],
        )
        pos = xr.DataArray(
            np.ones((3, 3)),
            dims=["degree", "axis"],
        )
        # No referenceTime in attrs
        with pytest.raises(Exception, match="orbit reference time not defined"):
            backward_geocode(dem_ecef, pos)

    def test_backward_geocode_missing_scale_time(self):
        """Missing scaleTime should raise."""
        dem_ecef = xr.DataArray(
            np.random.rand(3, 3, 3),
            dims=["axis", "y", "x"],
        )
        pos = xr.DataArray(
            np.ones((3, 3)),
            dims=["degree", "axis"],
        )
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        # No scaleTime in attrs
        with pytest.raises(Exception, match="orbit scaling time not defined"):
            backward_geocode(dem_ecef, pos)

    def test_backward_geocode_t0_size_mismatch(self):
        """T0 size mismatch should raise."""
        dem_ecef = xr.DataArray(
            np.random.rand(3, 4, 4),
            dims=["axis", "y", "x"],
        )
        pos = xr.DataArray(
            np.ones((3, 3)),
            dims=["degree", "axis"],
        )
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        t0 = xr.DataArray(np.arange(5), dims=["guess"])  # size 5 != 4*4
        with pytest.raises(Exception, match="number of guess points"):
            backward_geocode(dem_ecef, pos, t0=t0)


class TestSimulateAcquisition:
    """Tests for simulate_acquisition."""

    def test_simulate_acquisition_basic(self):
        """Basic acquisition should return dataset."""
        dem_ecef = xr.DataArray(
            np.ones((3, 3, 3)),
            dims=["axis", "y", "x"],
            coords={"axis": [0, 1, 2], "y": [0, 1, 2], "x": [0, 1, 2]},
        )
        pos = xr.DataArray(
            np.ones((3, 3)),
            dims=["degree", "axis"],
        )
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = simulate_acquisition(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)
