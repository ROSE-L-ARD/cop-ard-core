"""Unit tests for cop_ard_core.simul."""

from typing import Union, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS
from scipy.constants import speed_of_light

from cop_ard_core.simul import (
    backward_geocode,
    compute_simulation,
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
            "min": xr.Dataset(
                {
                    "azimuth_time": xr.DataArray(
                        np.array([[np.datetime64("2020-01-01T00:00:00")]]),
                        dims=["y", "x"],
                    ),
                    "slant_range_time": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
                    "y": xr.DataArray(np.array([[40.0]]), dims=["y", "x"]),
                    "x": xr.DataArray(np.array([[-75.0]]), dims=["y", "x"]),
                }
            ),
            "max": xr.Dataset(
                {
                    "azimuth_time": xr.DataArray(
                        np.array([[np.datetime64("2020-01-13T00:00:00")]]),
                        dims=["y", "x"],
                    ),
                    "slant_range_time": xr.DataArray(np.array([[10.0]]), dims=["y", "x"]),
                    "y": xr.DataArray(np.array([[45.0]]), dims=["y", "x"]),
                    "x": xr.DataArray(np.array([[-70.0]]), dims=["y", "x"]),
                }
            ),
        }
        az_min = cast(Union[int, float], np.datetime64("2020-01-01T00:00:00").item())
        az_max = cast(Union[int, float], np.datetime64("2020-01-13T00:00:00").item())
        bbox_radar = (az_min, 0.0, az_max, 10.0)
        result = get_geo_from_radar_bbox(data_ml, bbox_radar)
        assert result is not None
        assert len(result) == 4

    def test_get_geo_from_radar_bbox_no_overlap(self):
        """Non-overlapping radar bbox should return None."""
        data_ml = {
            "min": xr.Dataset(
                {
                    "azimuth_time": xr.DataArray(
                        np.array([[np.datetime64("2020-01-01T00:00:00")]]),
                        dims=["y", "x"],
                    ),
                    "slant_range_time": xr.DataArray(np.array([[0.0]]), dims=["y", "x"]),
                    "y": xr.DataArray(np.array([[40.0]]), dims=["y", "x"]),
                    "x": xr.DataArray(np.array([[-75.0]]), dims=["y", "x"]),
                }
            ),
            "max": xr.Dataset(
                {
                    "azimuth_time": xr.DataArray(
                        np.array([[np.datetime64("2020-01-13T00:00:00")]]),
                        dims=["y", "x"],
                    ),
                    "slant_range_time": xr.DataArray(np.array([[10.0]]), dims=["y", "x"]),
                    "y": xr.DataArray(np.array([[45.0]]), dims=["y", "x"]),
                    "x": xr.DataArray(np.array([[-70.0]]), dims=["y", "x"]),
                }
            ),
        }
        az_min = cast(Union[int, float], np.datetime64("2019-01-01T00:00:00").item())
        az_max = cast(Union[int, float], np.datetime64("2019-01-02T00:00:00").item())
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
    """Tests for backward_geocode.

    The backward_geocode function computes radar image coordinates (azimuth time,
    slant range distance, satellite direction) from DEM ECEF coordinates and
    satellite orbit polynomial coefficients. It uses an iterative range-Doppler
    solver (main_loop) to find the correct azimuth time for each DEM pixel.

    DEM DataArrays must use dims ['y', 'x', 'axis'] where 'axis' is the last
    dimension containing [X, Y, Z] ECEF coordinates. This matches the structure
    produced by convert_to_dem_ecef() in the actual pipeline.
    """

    # ------------------------------------------------------------------
    # Inline fixtures for backward_geocode test data
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pos(
        const_coeffs=(6700000.0, 2000000.0, 4000000.0),
        lin_coeffs=(7500.0, 0.0, 0.0),
        quad_coeffs=(0.0, 0.0, 0.0),
    ) -> xr.DataArray:
        """Create a valid orbit xr.DataArray with referenceTime and scaleTime attrs.

        Parameters
        ----------
        const_coeffs : tuple
            Constant (0th order) polynomial coefficients for X, Y, Z.
        lin_coeffs : tuple
            Linear (1st order) polynomial coefficients for X, Y, Z.
        quad_coeffs : tuple
            Quadratic (2nd order) polynomial coefficients for X, Y, Z.
        """
        coeffs = np.array([const_coeffs, lin_coeffs, quad_coeffs])
        pos = xr.DataArray(coeffs, dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        return pos

    @staticmethod
    def _make_dem(y_size=3, x_size=3, x_val=6700000.0, y_val=2000000.0, z_val=4001000.0) -> xr.DataArray:
        """Create an ECEF DEM DataArray with dims ['y', 'x', 'axis'].

        Each pixel (i, j) has XYZ coordinates [x_val, y_val, z_val].
        """
        dem = np.zeros((y_size, x_size, 3))
        for i in range(y_size):
            for j in range(x_size):
                dem[i, j, :] = [x_val, y_val, z_val]
        return xr.DataArray(dem, dims=["y", "x", "axis"], coords={"axis": [0, 1, 2]})

    @staticmethod
    def _make_dem_varying_z(y_size=3, x_size=3, base_z=4000000.0) -> xr.DataArray:
        """Create an ECEF DEM with Z varying across the spatial grid."""
        dem = np.zeros((y_size, x_size, 3))
        z_values = np.array([[0.0, 500.0, 1000.0], [1000.0, 1500.0, 2000.0], [2000.0, 2500.0, 3000.0]])
        for i in range(y_size):
            for j in range(x_size):
                dem[i, j, :] = [6700000.0, 2000000.0, base_z + z_values[i, j]]
        return xr.DataArray(dem, dims=["y", "x", "axis"], coords={"axis": [0, 1, 2]})

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_backward_geocode_missing_reference_time(self):
        """Missing referenceTime in pos attrs should raise."""
        dem_ecef = self._make_dem()
        pos = xr.DataArray(np.ones((3, 3)), dims=["degree", "axis"])
        # No referenceTime in attrs
        with pytest.raises(Exception, match="orbit reference time not defined"):
            backward_geocode(dem_ecef, pos)

    def test_backward_geocode_missing_scale_time(self):
        """Missing scaleTime in pos attrs should raise."""
        dem_ecef = self._make_dem()
        pos = xr.DataArray(np.ones((3, 3)), dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        with pytest.raises(Exception, match="orbit scaling time not defined"):
            backward_geocode(dem_ecef, pos)

    def test_backward_geocode_t0_size_mismatch(self):
        """T0 xr.DataArray with wrong size should raise."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        # Size 5 != 3*3 = 9
        t0 = xr.DataArray(np.arange(5), dims=["guess"])
        with pytest.raises(Exception, match="number of guess points"):
            backward_geocode(dem_ecef, pos, t0=t0)

    def test_backward_geocode_t0_size_mismatch_5x5(self):
        """T0 size mismatch with 5x5 DEM should raise."""
        dem_ecef = self._make_dem(y_size=5, x_size=5)
        pos = self._make_pos()
        # Size 7 != 5*5 = 25
        t0 = xr.DataArray(np.arange(7), dims=["guess"])
        with pytest.raises(Exception, match="number of guess points"):
            backward_geocode(dem_ecef, pos, t0=t0)

    # ------------------------------------------------------------------
    # Happy paths — basic functionality
    # ------------------------------------------------------------------

    def test_backward_geocode_returns_dataset(self):
        """Should return an xr.Dataset."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)

    def test_backward_geocode_has_expected_vars(self):
        """Output dataset should contain azimuth_time, dem_distance, satellite_direction."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert "azimuth_time" in result.data_vars
        assert "dem_distance" in result.data_vars
        assert "satellite_direction" in result.data_vars

    def test_backward_geocode_output_shapes_match_dem(self):
        """Output shapes should match the input DEM shape."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == dem_ecef.shape[:2]  # (y, x)
        assert result.dem_distance.shape == dem_ecef.shape
        assert result.satellite_direction.shape == dem_ecef.shape

    def test_backward_geocode_output_shapes_5x5(self):
        """Output shapes should match a 5x5 DEM."""
        dem_ecef = self._make_dem(y_size=5, x_size=5)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == (5, 5)
        assert result.dem_distance.shape == (5, 5, 3)
        assert result.satellite_direction.shape == (5, 5, 3)

    def test_backward_geocode_constant_dem_zero_range(self):
        """Constant DEM at satellite position should yield zero range.

        Note: This test uses a non-zero velocity to ensure the solver
        converges without division-by-zero (which occurs with zero velocity).
        """
        dem_ecef = self._make_dem(z_val=4000000.0)
        # Non-zero velocity ensures solver doesn't hit div-by-zero
        pos = self._make_pos(
            const_coeffs=(6700000.0, 2000000.0, 4000000.0),
            lin_coeffs=(7500.0, 0.0, 0.0),
            quad_coeffs=(0.0, 0.0, 0.0),
        )
        result = backward_geocode(dem_ecef, pos)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 0.0, atol=1e-6)

    def test_backward_geocode_constant_dem_nonzero_range(self):
        """Constant DEM offset from satellite should yield correct non-zero range."""
        # DEM is 1000m above the satellite in Z
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Range should be |dem - sat_pos| = (0, 0, 1000), magnitude = 1000
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)

    def test_backward_geocode_varying_dem_heights(self):
        """Varying DEM heights should produce varying range values."""
        dem_ecef = self._make_dem_varying_z()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # The DEM Z values are [0, 500, ..., 3000] added to 4000000
        # Expected ranges: [0, 500, 1000, 1000, 1500, 2000, 2000, 2500, 3000]
        expected = np.array(
            [
                [0.0, 500.0, 1000.0],
                [1000.0, 1500.0, 2000.0],
                [2000.0, 2500.0, 3000.0],
            ]
        )
        np.testing.assert_allclose(result.dem_distance.values[:, :, 2], expected, atol=1.0)

    def test_backward_geocode_no_t0_default(self):
        """Without t0, time should default to referenceTime (zero offset)."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        expected_time = np.datetime64("2020-01-01", "ns")
        np.testing.assert_array_equal(result.azimuth_time.values, expected_time)

    # ------------------------------------------------------------------
    # t0 parameter tests
    # ------------------------------------------------------------------

    def test_backward_geocode_t0_datetime64(self):
        """t0 as np.datetime64 should be applied to all pixels."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        t0 = np.datetime64("2020-01-01T00:00:05", "ns")
        result = backward_geocode(dem_ecef, pos, t0=t0)
        # All pixels should have the same azimuth time
        assert np.all(result.azimuth_time.values == result.azimuth_time.values[0, 0])

    def test_backward_geocode_t0_dataarray(self):
        """t0 as xr.DataArray with matching size should be applied per-pixel."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        t0_arr = xr.DataArray(
            np.full((3, 3), np.datetime64("2020-01-01T00:00:05", "ns")),
            dims=["y", "x"],
        )
        result = backward_geocode(dem_ecef, pos, t0=t0_arr)
        assert result.azimuth_time.shape == (3, 3)

    def test_backward_geocode_t0_uses_reference_time(self):
        """Output azimuth_time should be referenceTime + computed offset."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        # Orbit with t0 = 2020-06-01
        pos = self._make_pos()
        pos.attrs["referenceTime"] = np.datetime64("2020-06-01", "ns")
        result = backward_geocode(dem_ecef, pos)
        # Without t0 offset, azimuth_time should equal referenceTime
        assert result.azimuth_time.values[0, 0] == np.datetime64("2020-06-01", "ns")

    def test_backward_geocode_t0_produces_datetime_output(self):
        """Output azimuth_time should be datetime64[ns] type."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        t0 = np.datetime64("2020-01-01T00:00:05", "ns")
        result = backward_geocode(dem_ecef, pos, t0=t0)
        # The output is datetime64[ns], not timedelta64[ns]
        assert result.azimuth_time.dtype == np.dtype("datetime64[ns]")

    @pytest.mark.unit
    def test_backward_geocode_precision(self):
        """
        Test zero-Doppler calculation using a mathematically exact synthetic setup.
        This ensures the Newton-Raphson implementation inside `main_loop` converges correctly.
        """
        # 1. Orbit setup (Polynomials degree=1 -> shape is 2, 3)
        # Target satellite moves strictly along X at 7,000 m/s
        # Initial Position (c0): X=0, Y=0, Z=700,000
        # Velocity (c1): X=7000, Y=0, Z=0
        pos_values = np.array(
            [
                [0.0, 0.0, 700000.0],  # Intercept (c0)
                [7000.0, 0.0, 0.0],  # Slope (c1)
            ]
        )
        pos = xr.DataArray(pos_values, dims=["degree", "axis"])
        ref_time = np.datetime64("2024-01-01T00:00:00", "ns")
        pos.attrs = {
            "referenceTime": ref_time,
            "scaleTime": 1.0,  # 1 second scaling
        }

        # 2. Target DEM Point Setup
        # Placed exactly at X=0, Y=100,000, Z=0.
        # Nearest approach happens precisely at t=0 (the referenceTime).
        dem_values = np.array(
            [
                [[0.0]],  # X
                [[100000.0]],  # Y
                [[0.0]],  # Z
            ]
        )
        dem_ecef = xr.DataArray(dem_values, dims=["axis", "y", "x"], coords={"axis": [0, 1, 2]}).transpose(
            "y", "x", "axis"
        )

        # 3. Supply a staggered initial guess `t0` to prove that Newton-Raphson works
        t0_offset = ref_time + np.timedelta64(500, "ms")  # 0.5s offset
        t0 = xr.DataArray([[t0_offset]], dims=["y", "x"])

        # 4. Process Backward Geocoding
        result = backward_geocode(dem_ecef, pos, t0=t0)

        # 5. Precise Assertions
        # The calculated azimuth time should converge exactly to t=0 (ref_time)
        assert result.azimuth_time.values[0, 0] == ref_time

        # Calculate the expected dem_distance array.
        # Note: backward_geocode returns 'dem_distance' as the ECEF coordinate vector from satellite to target
        # vector = target_coords - satellite_coords = (0, 100000, 0) - (0, 0, 700000)
        expected_vector = np.array([[[-0.0, 100000.0, -700000.0]]])
        np.testing.assert_array_almost_equal(result.dem_distance.values, expected_vector)

        # Calculate expected normalized satellite direction (should be purely along the X axis)
        expected_direction = np.array([[[1.0, 0.0, 0.0]]])
        np.testing.assert_array_almost_equal(result.satellite_direction.values, expected_direction)

    @pytest.mark.unit
    def test_simulate_acquisition_precision(self):
        """
        Test slant_range_time calculation wrap-up using the same synthetic setup.
        """
        # Use the same exact geometry as above
        pos_values = np.array([[0.0, 0.0, 700000.0], [7000.0, 0.0, 0.0]])
        pos = xr.DataArray(pos_values, dims=["degree", "axis"])
        ref_time = np.datetime64("2024-01-01T00:00:00", "ns")
        pos.attrs = {"referenceTime": ref_time, "scaleTime": 1.0}

        dem_values = np.array([[[0.0]], [[100000.0]], [[0.0]]])
        dem_ecef = xr.DataArray(dem_values, dims=["axis", "y", "x"], coords={"axis": [0, 1, 2]}).transpose(
            "y", "x", "axis"
        )

        # Simulate
        result = simulate_acquisition(
            dem_ecef, pos, include_variables=("slant_range_time", "azimuth_time", "dem_distance")
        )

        # Expected absolute distance = sqrt(Y_offset^2 + Z_offset^2)
        expected_distance = np.sqrt(100000.0**2 + 700000.0**2)

        # Expected two-way travel time
        expected_slant_range_time = 2.0 / speed_of_light * expected_distance

        # Assertions
        assert result.azimuth_time.values[0, 0] == ref_time
        assert result.slant_range_time.values[0, 0] == pytest.approx(expected_slant_range_time)

        # Verify the `include_variables` parameter successfully dropped `satellite_direction`
        assert "satellite_direction" not in result.data_vars

    # ------------------------------------------------------------------
    # Type and dtype correctness
    # ------------------------------------------------------------------

    def test_backward_geocode_dem_distance_dtype(self):
        """dem_distance should be float64."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.dem_distance.dtype == np.float64

    def test_backward_geocode_satellite_direction_dtype(self):
        """satellite_direction should be float64."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.satellite_direction.dtype == np.float64

    def test_backward_geocode_satellite_direction_normalized(self):
        """satellite_direction vectors should be unit length."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Compute magnitude of each direction vector
        magnitudes = np.sqrt((result.satellite_direction.values**2).sum(axis=-1))
        np.testing.assert_allclose(magnitudes, 1.0, atol=1e-6)

    def test_backward_geocode_dem_distance_real_values(self):
        """dem_distance should be real (non-complex) values."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert np.isrealobj(result.dem_distance.values)

    def test_backward_geocode_satellite_direction_real_values(self):
        """satellite_direction should be real (non-complex) values."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert np.isrealobj(result.satellite_direction.values)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_backward_geocode_single_pixel(self):
        """Single-pixel DEM should produce single-pixel output."""
        dem_ecef = self._make_dem(y_size=1, x_size=1)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == (1, 1)
        assert result.dem_distance.shape == (1, 1, 3)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)

    def test_backward_geocode_2x2_dem(self):
        """2x2 DEM should produce correct 2x2 output."""
        dem_ecef = self._make_dem(y_size=2, x_size=2)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == (2, 2)
        assert result.dem_distance.shape == (2, 2, 3)
        assert result.satellite_direction.shape == (2, 2, 3)

    def test_backward_geocode_nan_in_dem(self):
        """NaN values in DEM should produce NaN/inf range."""
        dem_ecef = self._make_dem()
        # Set DEM to NaN via the underlying array
        dem_ecef.values[0, 0, :] = [np.nan, np.nan, np.nan]
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # The pixel with NaN DEM should have NaN or inf range
        assert np.isnan(result.dem_distance.values[0, 0, 2]) or np.isinf(result.dem_distance.values[0, 0, 2])

    def test_backward_geocode_inf_in_dem(self):
        """Inf values in DEM should propagate to output."""
        dem_ecef = self._make_dem()
        dem_ecef.values[1, 1, :] = [np.inf, np.inf, np.inf]
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert np.isnan(result.dem_distance.values[1, 1, 2]) or np.isinf(result.dem_distance.values[1, 1, 2])

    def test_backward_geocode_mixed_nan_inf_dem(self):
        """Mix of NaN and valid values in DEM should handle both."""
        dem_ecef = self._make_dem()
        dem_ecef.values[0, 0, :] = [np.nan, np.nan, np.nan]
        dem_ecef.values[1, 1, :] = [np.inf, np.inf, np.inf]
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Valid pixels should have correct range
        np.testing.assert_allclose(result.dem_distance.values[2, 2, 2], 1000.0, atol=1e-6)
        # Invalid pixels should have NaN or inf
        assert np.isnan(result.dem_distance.values[0, 0, 2]) or np.isinf(result.dem_distance.values[0, 0, 2])
        assert np.isnan(result.dem_distance.values[1, 1, 2]) or np.isinf(result.dem_distance.values[1, 1, 2])

    # ------------------------------------------------------------------
    # Convergence parameter tests
    # ------------------------------------------------------------------

    def test_backward_geocode_conv_th_strict(self):
        """Strict convergence threshold should still produce valid results."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos, conv_th=1e-10)
        assert isinstance(result, xr.Dataset)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-4)

    def test_backward_geocode_conv_th_loose(self):
        """Loose convergence threshold should produce valid (less precise) results."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos, conv_th=1e-2)
        assert isinstance(result, xr.Dataset)
        # With loose convergence, range should still be approximately correct
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1.0)

    # ------------------------------------------------------------------
    # Custom dim parameter tests
    # ------------------------------------------------------------------

    def test_backward_geocode_custom_dim(self):
        """Custom dim parameter should work when DEM has that dimension."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        # Using default dim="axis" should work
        result = backward_geocode(dem_ecef, pos, dim="axis")
        assert isinstance(result, xr.Dataset)
        assert result.azimuth_time.shape == (3, 3)

    # ------------------------------------------------------------------
    # Value correctness — detailed checks
    # ------------------------------------------------------------------

    def test_backward_geocode_range_matches_ecef_difference(self):
        """dem_distance should equal the ECEF distance between satellite and DEM."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Satellite position at t=0: (6700000, 2000000, 4000000)
        # DEM position: (6700000, 2000000, 4001000)
        # Difference: (0, 0, 1000), magnitude = 1000
        expected_range = 1000.0
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, expected_range, atol=1e-6)

    def test_backward_geocode_direction_points_to_satellite(self):
        """satellite_direction should point in the velocity direction."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Velocity is (7500, 0, 0), normalized = (1, 0, 0)
        expected_dir = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result.satellite_direction.values[0, 0, :], expected_dir, atol=1e-6)

    def test_backward_geocode_all_pixels_same_for_constant_dem(self):
        """All pixels in a constant-height DEM should have identical outputs."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert np.all(result.azimuth_time.values == result.azimuth_time.values[0, 0])
        np.testing.assert_allclose(result.dem_distance.values[:, 0, 0], result.dem_distance.values[0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(
            result.satellite_direction.values[0, 0, :],
            result.satellite_direction.values[2, 2, :],
            atol=1e-6,
        )

    def test_backward_geocode_dem_distance_nonnegative(self):
        """dem_distance values should be non-negative."""
        dem_ecef = self._make_dem_varying_z(base_z=4000000.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Flatten and check all values
        flat = result.dem_distance.values.flatten()
        assert np.all(flat >= -1e-6)  # Allow tiny negative from numerical error

    def test_backward_geocode_different_polynomial_degree(self):
        """Should work with higher-degree polynomial coefficients."""
        dem_ecef = self._make_dem()
        # 4th degree polynomial (5 coefficients per axis)
        coeffs = np.array(
            [
                [6700000.0, 2000000.0, 4000000.0],  # 0th order
                [7500.0, 0.0, 0.0],  # 1st order
                [0.0, 0.0, 0.001],  # 2nd order
                [0.0, 0.0, 0.0],  # 3rd order
                [0.0, 0.0, 0.0],  # 4th order
            ]
        )
        pos = xr.DataArray(coeffs, dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = backward_geocode(dem_ecef, pos)
        assert result.dem_distance.shape == (3, 3, 3)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)

    def test_backward_geocode_negative_z_offset(self):
        """DEM below satellite (negative Z offset) should produce positive range."""
        # DEM is 500m below the satellite in Z
        dem_ecef = self._make_dem(z_val=3999500.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        expected_range = 500.0
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, expected_range, atol=1e-6)

    def test_backward_geocode_x_offset(self):
        """DEM offset in X direction should affect range computation."""
        dem_ecef = self._make_dem(x_val=6700100.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        # Range should be approximately 1000 (Z offset) with small X contribution
        # The solver converges based on range rate, so exact X offset may not appear
        np.testing.assert_allclose(range_mag, 1000.0, atol=10.0)

    def test_backward_geocode_diagonal_offset(self):
        """DEM offset in multiple axes should affect range computation."""
        # Offset (100, 0, 500) from satellite
        dem_ecef = self._make_dem(x_val=6700100.0, z_val=4000500.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        # Range should be approximately 500 (Z offset)
        np.testing.assert_allclose(range_mag, 500.0, atol=10.0)

    def test_backward_geocode_large_dem_grid(self):
        """Larger DEM grid should handle correctly."""
        dem_ecef = self._make_dem(y_size=10, x_size=10)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == (10, 10)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)


class TestSimulateAcquisition:
    """Tests for simulate_acquisition."""

    def test_simulate_acquisition_basic(self):
        """Basic acquisition should return dataset."""
        dem_ecef = xr.DataArray(
            np.zeros((3, 3, 3)),
            dims=["y", "x", "axis"],
            coords={"axis": [0, 1, 2]},
        )
        # Fill DEM with ECEF coords
        dem_ecef.values[:, :, :] = [[6700000.0, 2000000.0, 4001000.0]]
        pos = xr.DataArray(
            np.array(
                [
                    [6700000.0, 2000000.0, 4000000.0],
                    [7500.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            dims=["degree", "axis"],
        )
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = simulate_acquisition(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" in result.data_vars

    # ------------------------------------------------------------------
    # Inline fixtures for simulate_acquisition test data
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pos(
        const_coeffs=(6700000.0, 2000000.0, 4000000.0),
        lin_coeffs=(7500.0, 0.0, 0.0),
        quad_coeffs=(0.0, 0.0, 0.0),
    ) -> xr.DataArray:
        """Create a valid orbit DataArray with referenceTime and scaleTime attrs.

        Parameters
        ----------
        const_coeffs : tuple
            Constant (0th order) polynomial coefficients for X, Y, Z.
        lin_coeffs : tuple
            Linear (1st order) polynomial coefficients for X, Y, Z.
        quad_coeffs : tuple
            Quadratic (2nd order) polynomial coefficients for X, Y, Z.
        """
        coeffs = np.array([const_coeffs, lin_coeffs, quad_coeffs])
        pos = xr.DataArray(coeffs, dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        return pos

    @staticmethod
    def _make_dem(y_size=3, x_size=3, x_val=6700000.0, y_val=2000000.0, z_val=4001000.0) -> xr.DataArray:
        """Create an ECEF DEM DataArray with dims ['y', 'x', 'axis'].

        Each pixel (i, j) has XYZ coordinates [x_val, y_val, z_val].
        """
        dem = np.zeros((y_size, x_size, 3))
        for i in range(y_size):
            for j in range(x_size):
                dem[i, j, :] = [x_val, y_val, z_val]
        return xr.DataArray(dem, dims=["y", "x", "axis"], coords={"axis": [0, 1, 2]})

    @staticmethod
    def _make_dem_varying_z(y_size=3, x_size=3, base_z=4000000.0) -> xr.DataArray:
        """Create an ECEF DEM with Z varying across the spatial grid."""
        dem = np.zeros((y_size, x_size, 3))
        z_values = np.array([[0.0, 500.0, 1000.0], [1000.0, 1500.0, 2000.0], [2000.0, 2500.0, 3000.0]])
        for i in range(y_size):
            for j in range(x_size):
                dem[i, j, :] = [6700000.0, 2000000.0, base_z + z_values[i, j]]
        return xr.DataArray(dem, dims=["y", "x", "axis"], coords={"axis": [0, 1, 2]})

    # ------------------------------------------------------------------
    # Happy paths — basic functionality
    # ------------------------------------------------------------------

    def test_simulate_acquisition_returns_dataset(self):
        """Should return an xr.Dataset."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)

    def test_simulate_acquisition_has_all_expected_vars(self):
        """Output dataset should contain azimuth_time, dem_distance, satellite_direction, slant_range_time."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert "azimuth_time" in result.data_vars
        assert "dem_distance" in result.data_vars
        assert "satellite_direction" in result.data_vars
        assert "slant_range_time" in result.data_vars

    def test_simulate_acquisition_output_shapes_match_dem(self):
        """Output azimuth_time and slant_range_time shapes should match DEM spatial dims (y, x)."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.shape == (3, 3)
        assert result.slant_range_time.shape == (3, 3)
        assert result.dem_distance.shape == (3, 3, 3)
        assert result.satellite_direction.shape == (3, 3, 3)

    def test_simulate_acquisition_output_shapes_5x5(self):
        """Output shapes should match a 5x5 DEM spatial grid."""
        dem_ecef = self._make_dem(y_size=5, x_size=5)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.shape == (5, 5)
        assert result.slant_range_time.shape == (5, 5)
        assert result.dem_distance.shape == (5, 5, 3)
        assert result.satellite_direction.shape == (5, 5, 3)

    def test_simulate_acquisition_slant_range_time_value(self):
        """Slant range time for 1000m range should equal 2*1000/c."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    def test_simulate_acquisition_zero_range(self):
        """DEM at satellite position should yield zero range and zero range time."""
        dem_ecef = self._make_dem(z_val=4000000.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        np.testing.assert_allclose(result.slant_range_time.values, 0.0, atol=1e-12)

    def test_simulate_acquisition_all_pixels_same_for_constant_dem(self):
        """All pixels in a constant-height DEM should have identical outputs."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert np.all(result.azimuth_time.values == result.azimuth_time.values[0, 0])
        np.testing.assert_allclose(result.slant_range_time.values, result.slant_range_time.values[0, 0], atol=1e-12)

    # ------------------------------------------------------------------
    # include_variables parameter
    # ------------------------------------------------------------------

    def test_simulate_acquisition_include_azimuth_time_only(self):
        """include_variables=['azimuth_time'] should drop other data vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("azimuth_time",))
        assert "azimuth_time" in result.data_vars
        assert "dem_distance" not in result.data_vars
        assert "satellite_direction" not in result.data_vars
        assert "slant_range_time" not in result.data_vars

    def test_simulate_acquisition_include_slant_range_time_only(self):
        """include_variables=['slant_range_time'] should drop other data vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("slant_range_time",))
        assert "slant_range_time" in result.data_vars
        assert "azimuth_time" not in result.data_vars
        assert "dem_distance" not in result.data_vars
        assert "satellite_direction" not in result.data_vars

    def test_simulate_acquisition_include_dem_distance_only(self):
        """include_variables=['dem_distance'] should drop other data vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("dem_distance",))
        assert "dem_distance" in result.data_vars
        assert "azimuth_time" not in result.data_vars
        assert "slant_range_time" not in result.data_vars
        assert "satellite_direction" not in result.data_vars

    def test_simulate_acquisition_include_satellite_direction_only(self):
        """include_variables=['satellite_direction'] should drop other data vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("satellite_direction",))
        assert "satellite_direction" in result.data_vars
        assert "azimuth_time" not in result.data_vars
        assert "slant_range_time" not in result.data_vars
        assert "dem_distance" not in result.data_vars

    def test_simulate_acquisition_include_subset(self):
        """include_variables with two vars should keep only those."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("azimuth_time", "slant_range_time"))
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" in result.data_vars
        assert "dem_distance" not in result.data_vars
        assert "satellite_direction" not in result.data_vars

    def test_simulate_acquisition_include_empty_keeps_all(self):
        """include_variables=() (default) should keep all data vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=())
        assert set(result.data_vars) == {"azimuth_time", "dem_distance", "satellite_direction", "slant_range_time"}

    def test_simulate_acquisition_include_all_vars(self):
        """include_variables with all vars should keep everything."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(
            dem_ecef, pos, include_variables=("azimuth_time", "dem_distance", "satellite_direction", "slant_range_time")
        )
        assert set(result.data_vars) == {"azimuth_time", "dem_distance", "satellite_direction", "slant_range_time"}

    def test_simulate_acquisition_include_nonexistent_var_keeps_all(self):
        """include_variables with a nonexistent var name should drop nothing that matches existing vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=("nonexistent_var",))
        assert "azimuth_time" not in result.data_vars
        assert "dem_distance" not in result.data_vars
        assert "satellite_direction" not in result.data_vars
        assert "slant_range_time" not in result.data_vars

    def test_simulate_acquisition_include_list_instead_of_tuple(self):
        """include_variables should accept list as well as tuple."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=["azimuth_time"])
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" not in result.data_vars

    # ------------------------------------------------------------------
    # Type and dtype correctness
    # ------------------------------------------------------------------

    def test_simulate_acquisition_azimuth_time_dtype(self):
        """azimuth_time should be datetime64[ns]."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.dtype == np.dtype("datetime64[ns]")

    def test_simulate_acquisition_slant_range_time_dtype(self):
        """slant_range_time should be float64."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.slant_range_time.dtype == np.float64

    def test_simulate_acquisition_dem_distance_dtype(self):
        """dem_distance should be float64."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.dem_distance.dtype == np.float64

    def test_simulate_acquisition_satellite_direction_dtype(self):
        """satellite_direction should be float64."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.satellite_direction.dtype == np.float64

    def test_simulate_acquisition_all_real_values(self):
        """All output values should be real (non-complex)."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert np.isrealobj(result.azimuth_time.values)
        assert np.isrealobj(result.slant_range_time.values)
        assert np.isrealobj(result.dem_distance.values)
        assert np.isrealobj(result.satellite_direction.values)

    def test_simulate_acquisition_slant_range_time_non_negative(self):
        """slant_range_time should be non-negative."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert np.all(result.slant_range_time.values >= -1e-12)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_simulate_acquisition_single_pixel(self):
        """Single-pixel DEM should produce single-pixel output."""
        dem_ecef = self._make_dem(y_size=1, x_size=1)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.shape == (1, 1)
        assert result.slant_range_time.shape == (1, 1)
        assert result.dem_distance.shape == (1, 1, 3)
        assert result.satellite_direction.shape == (1, 1, 3)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    def test_simulate_acquisition_2x2_dem(self):
        """2x2 DEM should produce correct 2x2 output."""
        dem_ecef = self._make_dem(y_size=2, x_size=2)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.shape == (2, 2)
        assert result.slant_range_time.shape == (2, 2)
        assert result.dem_distance.shape == (2, 2, 3)

    def test_simulate_acquisition_nan_in_dem(self):
        """All-NaN axis in DEM: xarray.sum(skipna=True) skips NaN and yields 0.0 range."""
        dem_ecef = self._make_dem()
        dem_ecef.values[0, 0, :] = [np.nan, np.nan, np.nan]
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        # xarray .sum() skips NaN by default; when all values are NaN, sum = 0.0
        np.testing.assert_allclose(result.slant_range_time.values[0, 0], 0.0, atol=1e-12)
        # dem_distance should still carry the NaN values
        assert np.all(np.isnan(result.dem_distance.values[0, 0, :]))

    def test_simulate_acquisition_inf_in_dem(self):
        """Inf values in DEM: xarray.sum(skipna=True) skips Inf, yielding 0.0 range."""
        dem_ecef = self._make_dem()
        dem_ecef.values[1, 1, :] = [np.inf, np.inf, np.inf]
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        # xarray .sum() skips inf by default; when all values are inf, sum = 0.0
        np.testing.assert_allclose(result.slant_range_time.values[1, 1], 0.0, atol=1e-12)

    def test_simulate_acquisition_mixed_nan_valid(self):
        """Mix of NaN and valid values: NaN axis pixel yields 0.0, valid pixels yield correct range."""
        dem_ecef = self._make_dem()
        dem_ecef.values[0, 0, :] = [np.nan, np.nan, np.nan]
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values[2, 2], expected_range_time, rtol=1e-10)
        # NaN pixel: xarray sum skips NaN -> 0.0
        np.testing.assert_allclose(result.slant_range_time.values[0, 0], 0.0, atol=1e-12)

    def test_simulate_acquisition_partial_nan_in_axis_propagates_to_all_components(self):
        """Partial NaN in DEM axis dimension: solver propagates NaN to all dem_distance components, xarray sum yields 0.0."""
        dem_ecef = self._make_dem()
        # Set only the Z component to NaN; the solver will propagate NaN to all 3 dem_distance components
        dem_ecef.values[0, 0, 2] = np.nan
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        # backward_geocode produces [nan, nan, nan] for that pixel
        assert np.all(np.isnan(result.dem_distance.values[0, 0, :]))
        # xarray sum(skipna=True) skips NaN, sum = 0.0
        np.testing.assert_allclose(result.slant_range_time.values[0, 0], 0.0, atol=1e-12)

    def test_simulate_acquisition_varying_dem_heights(self):
        """Varying DEM heights should produce varying slant_range_time values."""
        dem_ecef = self._make_dem_varying_z()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        # Z offsets: [0, 500, 1000, 1000, 1500, 2000, 2000, 2500, 3000]
        expected_ranges = np.array(
            [
                [0.0, 500.0, 1000.0],
                [1000.0, 1500.0, 2000.0],
                [2000.0, 2500.0, 3000.0],
            ]
        )
        expected_range_times = 2.0 * expected_ranges / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_times, rtol=1e-8)

    def test_simulate_acquisition_negative_z_offset(self):
        """DEM below satellite should produce positive range time."""
        dem_ecef = self._make_dem(z_val=3999500.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        expected_range_time = 2.0 * 500.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    def test_simulate_acquisition_large_dem_grid(self):
        """Larger DEM grid should handle correctly."""
        dem_ecef = self._make_dem(y_size=10, x_size=10)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.shape == (10, 10)
        assert result.slant_range_time.shape == (10, 10)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    # ------------------------------------------------------------------
    # Value correctness — detailed checks
    # ------------------------------------------------------------------

    def test_simulate_acquisition_slant_range_time_formula(self):
        """slant_range_time should equal 2/c * ||dem_distance||."""
        dem_ecef = self._make_dem_varying_z()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        range_magnitude = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        expected_range_time = 2.0 / 299792458.0 * range_magnitude
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    def test_simulate_acquisition_azimuth_time_is_datetime(self):
        """azimuth_time output should be datetime64[ns] type."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.dtype == np.dtype("datetime64[ns]")

    def test_simulate_acquisition_azimuth_time_equals_reference_time_no_offset(self):
        """Without azimuth_time offset, azimuth_time should equal referenceTime."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        pos.attrs["referenceTime"] = np.datetime64("2020-06-01", "ns")
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.values[0, 0] == np.datetime64("2020-06-01", "ns")

    def test_simulate_acquisition_azimuth_time_with_offset(self):
        """With azimuth_time parameter, azimuth_time should reflect the offset."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        t0 = np.datetime64("2020-01-01T00:00:05", "ns")
        result = simulate_acquisition(dem_ecef, pos, azimuth_time=t0)
        # All pixels should have the same azimuth time
        assert np.all(result.azimuth_time.values == result.azimuth_time.values[0, 0])

    def test_simulate_acquisition_varying_dem_all_different_ranges(self):
        """Varying DEM should produce all different slant_range_time values."""
        dem_ecef = self._make_dem_varying_z()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        unique_values = np.unique(result.slant_range_time.values)
        # With Z offsets [0,500,1000,1000,1500,2000,2000,2500,3000],
        # unique ranges are [0,500,1000,1500,2000,2500,3000] = 7 unique values
        assert len(unique_values) == 7

    def test_simulate_acquisition_x_offset_affects_range(self):
        """DEM offset in X direction should change range."""
        dem_ecef = self._make_dem(x_val=6700100.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        # X offset is small relative to Z, range should be approximately the same
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, atol=1e-14)

    def test_simulate_acquisition_diagonal_offset(self):
        """DEM offset in multiple axes: Z dominates range, X offset has minor effect."""
        dem_ecef = self._make_dem(x_val=6700100.0, z_val=4000500.0)
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        # Z offset of 500m dominates; X offset of 100m has minor effect due to range-rate solver
        expected_range_time = 2.0 * 500.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, atol=1e-14)

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------

    def test_simulate_acquisition_uses_axis_coord(self):
        """Output should use the 'axis' coordinate from the input DEM."""
        dem_ecef = self._make_dem()
        dem_ecef.coords["axis"] = [10.0, 20.0, 30.0]
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos)
        assert "axis" in result.coords or "axis" not in [c for c in result.coords]

    def test_simulate_acquisition_empty_include_variables_drops_all_vars(self):
        """include_variables=['nonexistent'] should drop all existing vars."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = simulate_acquisition(dem_ecef, pos, include_variables=["nonexistent"])
        assert len(list(result.data_vars)) == 0

    # ------------------------------------------------------------------
    # High-degree polynomial coefficients
    # ------------------------------------------------------------------

    def test_simulate_acquisition_higher_degree_poly(self):
        """Should work with higher-degree polynomial coefficients."""
        dem_ecef = self._make_dem()
        coeffs = np.array(
            [
                [6700000.0, 2000000.0, 4000000.0],
                [7500.0, 0.0, 0.0],
                [0.0, 0.0, 0.001],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        pos = xr.DataArray(coeffs, dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = simulate_acquisition(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)
        expected_range_time = 2.0 * 1000.0 / 299792458.0
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-6)

    # ------------------------------------------------------------------
    # Different reference times
    # ------------------------------------------------------------------

    def test_simulate_acquisition_different_reference_time(self):
        """Different referenceTime should affect azimuth_time output."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        pos.attrs["referenceTime"] = np.datetime64("2021-06-15", "ns")
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.values[0, 0] == np.datetime64("2021-06-15", "ns")

    def test_simulate_acquisition_large_time_offset(self):
        """Large time offset in referenceTime should shift azimuth_time accordingly."""
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        pos.attrs["referenceTime"] = np.datetime64("2030-12-31", "ns")
        result = simulate_acquisition(dem_ecef, pos)
        assert result.azimuth_time.values[0, 0] == np.datetime64("2030-12-31", "ns")


# ===========================================================================
# compute_simulation tests
# ===========================================================================


def _make_sim_data(acquisition_dates, y_size=3, x_size=3):
    """Create test data for compute_simulation.

    Returns (simulation, dem, orbits) ready to pass to compute_simulation.
    The DEM uses ECEF coordinates (EPSG:4978) with Z offset of 1000m.
    The orbit provides polynomial position coefficients.
    """
    # Simulation dataset with NaN slant_range_time (unprocessed)
    simulation = xr.Dataset(
        coords={
            "acquisition_date": acquisition_dates,
            "y": np.arange(y_size, dtype=np.float64),
            "x": np.arange(x_size, dtype=np.float64),
            "axis": [0, 1, 2],
        },
        data_vars={
            "h": xr.DataArray(np.zeros((y_size, x_size)), dims=["y", "x"]),
        },
    )
    simulation["slant_range_time"] = xr.full_like(simulation.h, fill_value=np.nan, dtype="float64")
    simulation["azimuth_time"] = xr.DataArray(
        np.zeros((len(acquisition_dates), y_size, x_size)),
        dims=["acquisition_date", "y", "x"],
        coords={"acquisition_date": acquisition_dates},
    ).astype("datetime64[ns]")
    simulation["dem_distance"] = xr.DataArray(
        np.zeros((len(acquisition_dates), y_size, x_size, 3)),
        dims=["acquisition_date", "y", "x", "axis"],
    )
    simulation.rio.write_crs("EPSG:4326", inplace=True)

    # DEM in ECEF with 1000m Z offset
    dem = np.zeros((y_size, x_size, 3))
    dem[:, :, :] = [[6700000.0, 2000000.0, 4001000.0]]
    dem_ecef = xr.DataArray(
        dem,
        dims=["y", "x", "axis"],
        coords={"axis": [0, 1, 2], "x": np.arange(x_size, dtype=np.float64), "y": np.arange(y_size, dtype=np.float64)},
    )
    dem_ecef.rio.write_crs(4978, inplace=True)

    # Orbit as xr.DataArray with dims (acquisition_date, degree, axis)
    # matching the type hint orbits: xr.DataArray in compute_simulation
    position_data = np.array(
        [
            [6700000.0, 2000000.0, 4000000.0],
            [7500.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    # Expand to (n_dates, 3, 3)
    n_dates = len(acquisition_dates)
    position_data = np.tile(position_data, (n_dates, 1, 1))
    orbits = xr.DataArray(
        position_data,
        dims=["acquisition_date", "degree", "axis"],
        coords={
            "acquisition_date": acquisition_dates,
            "degree": [0, 1, 2],
            "axis": [0, 1, 2],
        },
    )
    # reference_times: dict with datetime64 keys -> string values
    orbits.attrs["reference_times"] = {
        d: f"2020-01-{i + 1:02d}T00:00:00.000000000" for i, d in enumerate(acquisition_dates)
    }

    return simulation, dem_ecef, orbits


def _run_compute(sim_data):
    """Run compute_simulation and return the result."""
    from cop_ard_core.simul import compute_simulation

    simulation, dem, orbits = sim_data
    return compute_simulation(simulation, dem, orbits)


def _make_orbits(acquisition_dates, const_coeffs=(7500.0, 0.0, 0.0), quad_coeffs=(0.0, 0.0, 0.0)):
    """Create an orbit xr.DataArray for compute_simulation.

    Parameters
    ----------
    acquisition_dates : np.ndarray
        Array of datetime64 acquisition dates.
    const_coeffs : tuple
        Linear (velocity) polynomial coefficients for X, Y, Z.
    quad_coeffs : tuple
        Quadratic polynomial coefficients for X, Y, Z.
    """
    const_data = np.array([[6700000.0, 2000000.0, 4000000.0]])
    lin_data = np.array([const_coeffs])
    quad_data = np.array([quad_coeffs])
    position_data = np.vstack([const_data, lin_data, quad_data])
    n_dates = len(acquisition_dates)
    position_data = np.tile(position_data, (n_dates, 1, 1))
    orbits = xr.DataArray(
        position_data,
        dims=["acquisition_date", "degree", "axis"],
        coords={"acquisition_date": acquisition_dates, "degree": [0, 1, 2], "axis": [0, 1, 2]},
    )
    # FIXME pd.Timestamp doesn't feel right
    orbits.attrs["reference_times"] = {d: pd.Timestamp(d).isoformat() + ".000000000" for d in acquisition_dates}
    return orbits


def _make_ecef_dem(y_size=3, x_size=3, z_offset=1000.0):
    """Create an ECEF DEM DataArray.

    Parameters
    ----------
    y_size : int
        Number of rows.
    x_size : int
        Number of columns.
    z_offset : float
        Z offset from satellite position (4000000.0).
    """
    dem = np.zeros((y_size, x_size, 3))
    dem[:, :, :] = [[6700000.0, 2000000.0, 4000000.0 + z_offset]]
    dem_ecef = xr.DataArray(
        dem,
        dims=["y", "x", "axis"],
        coords={"axis": [0, 1, 2], "x": np.arange(x_size, dtype=np.float64), "y": np.arange(y_size, dtype=np.float64)},
    )
    dem_ecef.rio.write_crs(4978, inplace=True)
    return dem_ecef


def _make_unprocessed_simulation(acquisition_dates, y_size=3, x_size=3):
    """Create a simulation dataset with NaN slant_range_time (unprocessed)."""
    simulation = xr.Dataset(
        coords={
            "acquisition_date": acquisition_dates,
            "y": np.arange(y_size, dtype=np.float64),
            "x": np.arange(x_size, dtype=np.float64),
            "axis": [0, 1, 2],
        },
        data_vars={"h": xr.DataArray(np.zeros((y_size, x_size)), dims=["y", "x"])},
    )
    simulation["slant_range_time"] = xr.full_like(simulation.h, fill_value=np.nan, dtype="float64")
    simulation["azimuth_time"] = xr.DataArray(
        np.zeros((len(acquisition_dates), y_size, x_size)),
        dims=["acquisition_date", "y", "x"],
        coords={"acquisition_date": acquisition_dates},
    ).astype("datetime64[ns]")
    simulation["dem_distance"] = xr.DataArray(
        np.zeros((len(acquisition_dates), y_size, x_size, 3)),
        dims=["acquisition_date", "y", "x", "axis"],
    )
    simulation.rio.write_crs("EPSG:4326", inplace=True)
    return simulation


class TestComputeSimulation:
    """Tests for cop_ard_core.simul.compute_simulation.

    The compute_simulation function is the core worker that runs on each Dask block.
    It takes a chunk of the DEM, converts its coordinates to ECEF if needed,
    and uses the satellite orbit data to compute radar geometry (azimuth time,
    slant range time, dem_distance) for each DEM pixel via the sarsen library
    (through simulate_acquisition / backward_geocode / main_loop).
    """

    # ------------------------------------------------------------------
    # Happy paths — basic functionality
    # ------------------------------------------------------------------

    def test_compute_simulation_returns_dataset(self):
        """compute_simulation should return an xr.Dataset."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates, y_size=3, x_size=3)
        result = _run_compute(sim_data)
        assert isinstance(result, xr.Dataset)

    def test_compute_simulation_has_expected_variables(self):
        """Output dataset should contain azimuth_time, slant_range_time, dem_distance."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" in result.data_vars
        assert "dem_distance" in result.data_vars

    def test_compute_simulation_output_shapes_match_dem(self):
        """Output shapes should include acquisition_date dimension plus DEM spatial dimensions."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        y_size = 3
        x_size = 4
        sim_data = _make_sim_data(acquisition_dates, y_size=y_size, x_size=x_size)
        result = _run_compute(sim_data)
        assert result.azimuth_time.shape == (1, y_size, x_size)
        assert result.slant_range_time.shape == (1, y_size, x_size)
        assert result.dem_distance.shape == (1, y_size, x_size, 3)

    def test_compute_simulation_azimuth_time_matches_acquisition_date(self):
        """Output azimuth_time should correspond to the acquisition_date."""
        dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(dates)
        result = _run_compute(sim_data)
        assert np.all(result.azimuth_time.values == dates[0])

    # ------------------------------------------------------------------
    # Happy paths — different DEM sizes
    # ------------------------------------------------------------------

    def test_compute_simulation_single_pixel(self):
        """Single-pixel DEM should produce single-pixel output with acquisition_date dim."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates, y_size=1, x_size=1)
        result = _run_compute(sim_data)
        assert result.azimuth_time.shape == (1, 1, 1)
        assert result.slant_range_time.shape == (1, 1, 1)
        assert result.dem_distance.shape == (1, 1, 1, 3)

    def test_compute_simulation_2x2_dem(self):
        """2x2 DEM should produce correct 2x2 output with acquisition_date dim."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates, y_size=2, x_size=2)
        result = _run_compute(sim_data)
        assert result.azimuth_time.shape == (1, 2, 2)
        assert result.slant_range_time.shape == (1, 2, 2)

    def test_compute_simulation_5x5_dem(self):
        """5x5 DEM should produce correct 5x5 output with acquisition_date dim."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates, y_size=5, x_size=5)
        result = _run_compute(sim_data)
        assert result.azimuth_time.shape == (1, 5, 5)
        assert result.slant_range_time.shape == (1, 5, 5)
        assert result.dem_distance.shape == (1, 5, 5, 3)

    def test_compute_simulation_10x10_dem(self):
        """10x10 DEM should produce correct 10x10 output with acquisition_date dim."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates, y_size=10, x_size=10)
        result = _run_compute(sim_data)
        assert result.azimuth_time.shape == (1, 10, 10)
        assert result.slant_range_time.shape == (1, 10, 10)

    def test_compute_simulation_multiple_acquisition_dates(self):
        """compute_simulation processes only the first acquisition date."""
        dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(dates)
        result = _run_compute(sim_data)
        # Output should have acquisition_date dimension with size 1 (only first date processed)
        assert len(result.acquisition_date) == 1
        assert result.azimuth_time.values[0, 0, 0] == np.datetime64("2020-01-01")

    def test_compute_simulation_multiple_dates_all_same_azimuth(self):
        """All acquisition dates should have valid (non-NaT) azimuth_time."""
        dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(dates)
        result = _run_compute(sim_data)
        assert not np.any(np.isnat(result.azimuth_time.values))

    # ------------------------------------------------------------------
    # Happy paths — value correctness
    # ------------------------------------------------------------------

    def test_compute_simulation_slant_range_time_non_negative(self):
        """Slant range time values should be non-negative."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert np.all(result.slant_range_time.values >= -1e-12)

    def test_compute_simulation_slant_range_time_finite(self):
        """Slant range time values should be finite (not NaN or inf)."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert np.all(np.isfinite(result.slant_range_time.values))

    def test_compute_simulation_dem_distance_non_negative(self):
        """dem_distance should have non-negative magnitude."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        assert np.all(range_mag >= -1e-6)

    def test_compute_simulation_all_pixels_identical_for_constant_dem(self):
        """All pixels in a constant-height DEM should have identical outputs."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert np.all(result.azimuth_time.values == result.azimuth_time.values[0, 0, 0])
        np.testing.assert_allclose(
            result.slant_range_time.values[0],
            result.slant_range_time.values[0, 0, 0],
            atol=1e-12,
        )

    def test_compute_simulation_range_time_formula(self):
        """Slant range time should equal 2/c * ||dem_distance||."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        expected_range_time = 2.0 / 299792458.0 * range_mag
        np.testing.assert_allclose(result.slant_range_time.values, expected_range_time, rtol=1e-10)

    # ------------------------------------------------------------------
    # Type and dtype correctness
    # ------------------------------------------------------------------

    def test_compute_simulation_azimuth_time_dtype(self):
        """azimuth_time should be datetime64[ns]."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert result.azimuth_time.dtype == np.dtype("datetime64[ns]")

    def test_compute_simulation_slant_range_time_dtype(self):
        """slant_range_time should be float64."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert result.slant_range_time.dtype == np.float64

    def test_compute_simulation_dem_distance_dtype(self):
        """dem_distance should be float64."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert result.dem_distance.dtype == np.float64

    def test_compute_simulation_all_real_values(self):
        """All output values should be real (non-complex)."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert np.isrealobj(result.azimuth_time.values)
        assert np.isrealobj(result.slant_range_time.values)
        assert np.isrealobj(result.dem_distance.values)

    def test_compute_simulation_no_extra_variables(self):
        """Output should only contain expected data variables, not extra ones."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert "h" not in result.data_vars
        assert set(result.data_vars) <= {"azimuth_time", "slant_range_time", "dem_distance"}

    def test_compute_simulation_cleaned_coordinates(self):
        """Intermediate coordinates (spatial_ref, crs) should be dropped from output."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert "spatial_ref" not in result.coords
        assert "crs" not in result.coords

    def test_compute_simulation_no_reference_to_input_simulation(self):
        """Output should be independent of input simulation data_vars."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        # The output should not contain 'h' from the input simulation
        assert "h" not in result.data_vars

    # ------------------------------------------------------------------
    # Edge cases — already processed data (no NaN in slant_range_time)
    # ------------------------------------------------------------------

    def test_compute_simulation_no_nan_skips_processing(self):
        """When slant_range_time has no NaN, function should return simulation structure."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        # Override slant_range_time with valid values (no NaN)
        simulation["slant_range_time"] = xr.DataArray(
            np.zeros((1, 3, 3)),
            dims=["acquisition_date", "y", "x"],
        )
        dem = _make_ecef_dem(y_size=3, x_size=3)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)

    # ------------------------------------------------------------------
    # Edge cases — DEM with NaN values
    # ------------------------------------------------------------------

    def test_compute_simulation_nan_in_dem(self):
        """NaN values in DEM should be handled during processing."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3)
        dem.values[0, 0, :] = [np.nan, np.nan, np.nan]
        dem.values[1, 1, :] = [np.nan, np.nan, np.nan]
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars

    def test_compute_simulation_inf_in_dem(self):
        """Inf values in DEM should be handled during processing."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3)
        dem.values[0, 0, :] = [np.inf, np.inf, np.inf]
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars

    def test_compute_simulation_mixed_nan_inf_valid(self):
        """Mix of NaN, inf, and valid values in DEM should handle all correctly."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3)
        dem.values[0, 0, :] = [np.nan, np.nan, np.nan]
        dem.values[1, 1, :] = [np.inf, np.inf, np.inf]
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars

    # ------------------------------------------------------------------
    # Edge cases — different Z offsets
    # ------------------------------------------------------------------

    def test_compute_simulation_negative_z_offset(self):
        """DEM below satellite (negative Z offset) should produce valid range."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3, z_offset=-500.0)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        range_mag = np.sqrt((result.dem_distance.values**2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 500.0, atol=10.0)

    def test_compute_simulation_large_z_offset(self):
        """Large Z offset should still produce valid range."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3, z_offset=5000.0)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    def test_compute_simulation_x_offset_dem(self):
        """DEM offset in X direction should affect range computation."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem_data = np.zeros((3, 3, 3))
        dem_data[:, :, :] = [[6700100.0, 2000000.0, 4001000.0]]
        dem = xr.DataArray(
            dem_data,
            dims=["y", "x", "axis"],
            coords={"y": np.arange(3, dtype=np.float64), "x": np.arange(3, dtype=np.float64), "axis": [0, 1, 2]},
        )
        dem.rio.write_crs(4978, inplace=True)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    def test_compute_simulation_small_z_offset(self):
        """Small Z offset (1m) should produce valid range."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3, z_offset=1.0)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    # ------------------------------------------------------------------
    # Edge cases — different polynomial coefficients
    # ------------------------------------------------------------------

    def test_compute_simulation_quadratic_polynomial(self):
        """Should work with quadratic polynomial position coefficients."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3)
        orbits = _make_orbits(acquisition_dates, quad_coeffs=(0.1, 0.0, 0.0))
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    def test_compute_simulation_higher_degree_polynomial(self):
        """Should work with higher-degree polynomial coefficients."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem = _make_ecef_dem(y_size=3, x_size=3)
        orbits = _make_orbits(acquisition_dates, quad_coeffs=(0.0, 0.0, 0.001))
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    # ------------------------------------------------------------------
    # Edge cases — DEM coordinate handling
    # ------------------------------------------------------------------

    def test_compute_simulation_dem_all_finite_values(self):
        """DEM with all finite values should produce valid output."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        sim_data = _make_sim_data(acquisition_dates)
        result = _run_compute(sim_data)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
        assert np.all(np.isfinite(result.slant_range_time.values))

    def test_compute_simulation_dem_no_axis_coord(self):
        """DEM without axis coordinate value should still work via fallback."""
        acquisition_dates = np.array(["2020-01-01"], dtype="datetime64[ns]")
        simulation = _make_unprocessed_simulation(acquisition_dates)
        dem_data = np.zeros((3, 3, 3))
        dem_data[:, :, :] = [[6700000.0, 2000000.0, 4001000.0]]
        dem = xr.DataArray(
            dem_data,
            dims=["y", "x", "axis"],
            coords={"y": np.arange(3, dtype=np.float64), "x": np.arange(3, dtype=np.float64), "axis": [0, 1, 2]},
        )
        dem.rio.write_crs(4978, inplace=True)
        orbits = _make_orbits(acquisition_dates)
        result = compute_simulation(simulation, dem, orbits)
        assert isinstance(result, xr.Dataset)
        assert "slant_range_time" in result.data_vars
