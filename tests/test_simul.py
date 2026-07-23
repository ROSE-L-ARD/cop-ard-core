"""Unit tests for cop_ard_core.simul."""

from typing import Union, cast

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
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 0.0, atol=1e-6)

    def test_backward_geocode_constant_dem_nonzero_range(self):
        """Constant DEM offset from satellite should yield correct non-zero range."""
        # DEM is 1000m above the satellite in Z
        dem_ecef = self._make_dem(z_val=4001000.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # Range should be |dem - sat_pos| = (0, 0, 1000), magnitude = 1000
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)

    def test_backward_geocode_varying_dem_heights(self):
        """Varying DEM heights should produce varying range values."""
        dem_ecef = self._make_dem_varying_z()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        # The DEM Z values are [0, 500, ..., 3000] added to 4000000
        # Expected ranges: [0, 500, 1000, 1000, 1500, 2000, 2000, 2500, 3000]
        expected = np.array([
            [0.0, 500.0, 1000.0],
            [1000.0, 1500.0, 2000.0],
            [2000.0, 2500.0, 3000.0],
        ])
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
        magnitudes = np.sqrt((result.satellite_direction.values ** 2).sum(axis=-1))
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
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
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
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-4)

    def test_backward_geocode_conv_th_loose(self):
        """Loose convergence threshold should produce valid (less precise) results."""
        dem_ecef = self._make_dem()
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos, conv_th=1e-2)
        assert isinstance(result, xr.Dataset)
        # With loose convergence, range should still be approximately correct
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
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
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
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
        coeffs = np.array([
            [6700000.0, 2000000.0, 4000000.0],  # 0th order
            [7500.0, 0.0, 0.0],                   # 1st order
            [0.0, 0.0, 0.001],                    # 2nd order
            [0.0, 0.0, 0.0],                      # 3rd order
            [0.0, 0.0, 0.0],                      # 4th order
        ])
        pos = xr.DataArray(coeffs, dims=["degree", "axis"])
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = backward_geocode(dem_ecef, pos)
        assert result.dem_distance.shape == (3, 3, 3)
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, 1000.0, atol=1e-6)

    def test_backward_geocode_negative_z_offset(self):
        """DEM below satellite (negative Z offset) should produce positive range."""
        # DEM is 500m below the satellite in Z
        dem_ecef = self._make_dem(z_val=3999500.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        expected_range = 500.0
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        np.testing.assert_allclose(range_mag, expected_range, atol=1e-6)

    def test_backward_geocode_x_offset(self):
        """DEM offset in X direction should affect range computation."""
        dem_ecef = self._make_dem(x_val=6700100.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        # Range should be approximately 1000 (Z offset) with small X contribution
        # The solver converges based on range rate, so exact X offset may not appear
        np.testing.assert_allclose(range_mag, 1000.0, atol=10.0)

    def test_backward_geocode_diagonal_offset(self):
        """DEM offset in multiple axes should affect range computation."""
        # Offset (100, 0, 500) from satellite
        dem_ecef = self._make_dem(x_val=6700100.0, z_val=4000500.0)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
        # Range should be approximately 500 (Z offset)
        np.testing.assert_allclose(range_mag, 500.0, atol=10.0)

    def test_backward_geocode_large_dem_grid(self):
        """Larger DEM grid should handle correctly."""
        dem_ecef = self._make_dem(y_size=10, x_size=10)
        pos = self._make_pos()
        result = backward_geocode(dem_ecef, pos)
        assert result.azimuth_time.shape == (10, 10)
        range_mag = np.sqrt((result.dem_distance.values ** 2).sum(axis=-1))
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
        dem_ecef.values[:, :, :] = [[6700000., 2000000., 4001000.]]
        pos = xr.DataArray(
            np.array([
                [6700000., 2000000., 4000000.],
                [7500., 0., 0.],
                [0., 0., 0.],
            ]),
            dims=["degree", "axis"],
        )
        pos.attrs["referenceTime"] = np.datetime64("2020-01-01", "ns")
        pos.attrs["scaleTime"] = 1e-9
        result = simulate_acquisition(dem_ecef, pos)
        assert isinstance(result, xr.Dataset)
        assert "azimuth_time" in result.data_vars
        assert "slant_range_time" in result.data_vars
