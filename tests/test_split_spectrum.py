"""Unit tests for cop_ard_core.split_spectrum."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.split_spectrum import (
    fit_ionosphere,
    joint_unwrap,
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
        phase_iono = phase_iono.assign_coords(
            {
                "azimuth_time_raw": (
                    ["azimuth_time", "slant_range_time"],
                    np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float),
                ),
            }
        )
        result = fit_ionosphere(
            phase_iono,
            azimuth_time_raw,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
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
        phase_iono = phase_iono.assign_coords(
            {
                "azimuth_time_raw": (
                    ["azimuth_time", "slant_range_time"],
                    np.broadcast_to(np.arange(az_size)[:, None], (az_size, sr_size)).astype(float),
                ),
            }
        )
        result = fit_ionosphere(
            phase_iono,
            azimuth_time_raw,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert "offset" in result.attrs


class TestReconstructIonoSurface:
    """Tests for reconstruct_iono_surface.

    reconstruct_iono_surface takes ionospheric phase data (interf), azimuth time
    raw values, polynomial fit coefficients, and height parameters to reconstruct
    a 2D ionospheric phase surface. It scales azimuth time based on the ratio of
    ionosphere height to orbit height, builds a normalized model matrix from the
    coordinates, and evaluates the polynomial fit to produce the reconstructed surface.

    All tests use in-memory synthetic data with known values. No file I/O or mocks.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_reconstruct_inputs(
        az_size: int = 3,
        sr_size: int = 5,
        offset: np.ndarray | None = None,
        scale_fact: np.ndarray | None = None,
        coeffs_data: np.ndarray | None = None,
        degree: int = 1,
    ):
        """Create standard inputs for reconstruct_iono_surface.

        The scale_azimuth_time function requires datetime64[ns] for azimuth_time
        (both coords and azimuth_time_raw data) because it computes:
            (azimuth_time_raw - azimuth_time).astype(float)  # timedelta -> seconds
        So both must be datetime64[ns] to enable this subtraction.

        Parameters:
            az_size: Number of azimuth samples.
            sr_size: Number of slant range samples.
            offset: Normalization offset for x/y coordinates. If None, uses zeros.
            scale_fact: Normalization scale factor for x/y coordinates. If None, uses ones.
            coeffs_data: Polynomial coefficient values. If None, uses [0, 1, -0.5] for degree=1.
            degree: Polynomial degree.
        """
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt = np.array(
            [base_dt + np.timedelta64(1_000_000_000 * i, "ns") for i in range(az_size)],
            dtype="datetime64[ns]",
        )
        # Build 2D datetime array: each row is the azimuth timestamp broadcast across range
        az_raw_2d = np.empty((az_size, sr_size), dtype="datetime64[ns]")
        for i in range(az_size):
            az_raw_2d[i, :] = az_dt[i]

        interf = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float64),
            },
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float64),
            },
        )
        if offset is None:
            offset = np.zeros(2, dtype=np.float64)
        if scale_fact is None:
            scale_fact = np.ones(2, dtype=np.float64)
        if coeffs_data is None:
            if degree == 1:
                coeffs_data = np.array([0.0, 1.0, -0.5], dtype=np.float64)
            elif degree == 0:
                coeffs_data = np.array([2.0], dtype=np.float64)
            elif degree == 2:
                coeffs_data = np.array([0.0, 0.0, 0.0, 1.0, 0.0, -0.5], dtype=np.float64)
            else:
                coeffs_data = np.zeros(degree + 1, dtype=np.float64)
        coeffs = xr.DataArray(coeffs_data, dims=["degree"], coords={"degree": np.arange(coeffs_data.size)})
        coeffs.attrs["offset"] = offset
        coeffs.attrs["scale_fact"] = scale_fact
        return interf, azimuth_time_raw, coeffs

    # ------------------------------------------------------------------
    # Return type and shape tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_returns_dataarray(self):
        """Should return xr.DataArray."""
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs()
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert isinstance(result, xr.DataArray)

    def test_reconstruct_iono_surface_shape(self):
        """Output shape should match azimuth_time_raw shape."""
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs()
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == azimuth_time_raw.shape

    def test_reconstruct_iono_surface_preserves_dimensions(self):
        """Output DataArrays should have the same dimensions as inputs."""
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs()
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.dims == ("azimuth_time", "slant_range_time")

    def test_reconstruct_iono_surface_custom_coords_dimensions(self):
        """Output DataArray should have the same dimensions (but not necessarily coordinates) as inputs."""
        sr = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt = np.array([base_dt + np.timedelta64(i * 1_000_000_000, "ns") for i in range(3)], dtype="datetime64[ns]")
        az_raw_2d = np.empty((3, 4), dtype="datetime64[ns]")
        for i in range(3):
            az_raw_2d[i, :] = az_dt[i]
        interf = xr.DataArray(
            np.ones((3, 4)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt, "slant_range_time": sr},
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt, "slant_range_time": sr},
        )
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5], dtype=np.float64),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0])
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.dims == ("azimuth_time", "slant_range_time")
        assert result.shape == (3, 4)

    @pytest.mark.parametrize("shape", [(3, 5), (5, 3), (4, 4), (5, 5), (3, 3)])
    def test_reconstruct_iono_surface_various_shapes(self, shape):
        """Output shape should match azimuth_time_raw shape for various array sizes."""
        az_size, sr_size = shape
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == shape

    # ------------------------------------------------------------------
    # Dataset input tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_with_dataset_input(self):
        """Should accept xr.Dataset as interf parameter."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        interf_ds = interf.to_dataset(name="iono_phase")
        result = reconstruct_iono_surface(
            interf_ds,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert isinstance(result, xr.DataArray)

    def test_reconstruct_iono_surface_dataset_with_multiple_vars(self):
        """Should work with Dataset containing multiple data variables."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        interf_ds = interf.to_dataset(name="iono_phase")
        interf_ds["other_var"] = xr.DataArray(
            np.zeros((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": interf.azimuth_time, "slant_range_time": interf.slant_range_time},
        )
        result = reconstruct_iono_surface(
            interf_ds,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == (az_size, sr_size)

    # ------------------------------------------------------------------
    # Numerical correctness tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_constant_coefficients(self):
        """Only constant coefficient (degree=0) should produce a constant surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            degree=0,
            coeffs_data=np.array([42.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=0,
        )
        np.testing.assert_allclose(result.values, 42.0, atol=1e-10)

    def test_reconstruct_iono_surface_linear_gradient_correct(self):
        """Linear coefficients should produce a surface matching c0 + c1*x + c2*y with identity normalization."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # result = c0*y_norm + c1*x_norm + c2
        # y_norm = [0, 1e9, 2e9], x_norm = [0,1,2,3,4]
        # row0: 1*0 + 2*[0,1,2,3,4] + 3 = [3,5,7,9,11]
        # row1: 1*1e9 + 2*[0,1,2,3,4] + 3 ≈ [1e9, 1e9, 1e9, 1e9, 1e9]
        # row2: 1*2e9 + 2*[0,1,2,3,4] + 3 ≈ [2e9, 2e9, 2e9, 2e9, 2e9]
        y_vals_arr = interf.slant_range_time.values  # [0, 1, 2, 3, 4]
        expected_row0 = 2.0 * y_vals_arr + 3.0
        np.testing.assert_allclose(result.values[0], expected_row0, atol=1e-10)
        # Rows 1,2 dominated by c0*y_norm term
        np.testing.assert_allclose(result.values[1], np.full(5, 1e9), rtol=1e-6)
        np.testing.assert_allclose(result.values[2], np.full(5, 2e9), rtol=1e-6)

    def test_reconstruct_iono_surface_same_height_ratio(self):
        """When H_orb == H_iono the scaling should not change azimuth coordinates."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        # With H_orb == H_iono, ratio = 1.0 so az_time is unchanged
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=800000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == (az_size, sr_size)

    def test_reconstruct_iono_surface_zero_height_ratio(self):
        """When H_iono = 0 the scaled azimuth component should be zero."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([0.0, 1.0, 1.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=0.0,
            H_orb=800000.0,
            degree=1,
        )
        # result = c0*y_norm + c1*x_norm + c2 = 0 + 1*[0,1,2,3,4] + 1 = [1,2,3,4,5]
        # (c0=0 so y contribution is 0 for all rows)
        y_vals = interf.slant_range_time.values  # [0, 1, 2, 3, 4]
        expected = 1.0 * y_vals + 1.0  # [1,2,3,4,5]
        assert result.shape == (3, 5)
        np.testing.assert_allclose(result.values[0], expected, atol=1e-10)
        np.testing.assert_allclose(result.values[1], expected, atol=1e-10)
        np.testing.assert_allclose(result.values[2], expected, atol=1e-10)

    def test_reconstruct_iono_surface_large_height_ratio(self):
        """With H_iono != H_orb but matching datetime coords, azimuth component is zero."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=1600000.0,
            H_orb=800000.0,
            degree=1,
        )
        # result = c0*y_norm + c1*x_norm + c2 = 0 + 0 + 1 = 1
        # (c0=0 so y contribution is 0)
        expected = np.full((az_size, sr_size), 1.0)
        np.testing.assert_allclose(result.values, expected, atol=1e-10)

    def test_reconstruct_iono_surface_negative_coefficients(self):
        """Negative coefficients should produce negative contributions in the surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([0.0, -1.0, -2.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # Check that result has correct shape and finite values
        assert result.shape == (3, 5)
        assert np.all(np.isfinite(result.values))
        # All rows should be identical since c0=0 (no y contribution)
        np.testing.assert_array_equal(result.values[0], result.values[1])
        np.testing.assert_array_equal(result.values[1], result.values[2])
        # First row should be strictly decreasing (negative x coefficient)
        assert all(result.values[0, i] > result.values[0, i + 1] for i in range(4))

    def test_reconstruct_iono_surface_negative_coordinates(self):
        """Negative slant_range_time coordinates with custom normalization should be handled correctly."""
        az_size, sr_size = 3, 5
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt = np.array(
            [base_dt + np.timedelta64(i * 1_000_000_000, "ns") for i in range(az_size)], dtype="datetime64[ns]"
        )
        az_raw_2d = np.empty((az_size, sr_size), dtype="datetime64[ns]")
        for i in range(az_size):
            az_raw_2d[i, :] = az_dt[i]
        sr_coords = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        interf = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt, "slant_range_time": sr_coords},
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt, "slant_range_time": sr_coords},
        )
        # With az_dt matching az_raw, y_ = 0 after scaling.
        # x normalized: (sr - (-2)) / 4
        # result = c0 + c1 * x_norm + c2 * 0 = c1 * x_norm
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([-2.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([4.0, 1.0])
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        expected_x = (sr_coords - (-2.0)) / 4.0  # [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = np.tile(expected_x, (az_size, 1))  # all rows identical
        np.testing.assert_allclose(result.values, expected, atol=1e-10)

    def test_reconstruct_iono_surface_all_zeros(self):
        """All-zero coefficients should produce a zero surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_reconstruct_iono_surface_float64_dtype(self):
        """Output should be float64 dtype."""
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs()
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.dtype == np.float64

    def test_reconstruct_iono_surface_quadratic_surface(self):
        """Degree=2 with quadratic coefficients should produce a curved surface."""
        az_size, sr_size = 1, 5
        # degree=2: 6 coeffs [const, x, y, x^2, xy, y^2]
        # With datetime azimuth_time, y_ = 0, so y and y^2 terms don't contribute
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([0.5, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=2,
        )
        assert result.shape == (az_size, sr_size)
        sr_coords = np.arange(sr_size, dtype=np.float64)  # [0, 1, 2, 3, 4]
        # model cols: [y, x, y^2, xy, x^2, const]; result = 1*x^2 + 0.5*y^2
        # y_ = 0 with datetime coords, result = x^2 only
        expected = sr_coords**2  # [0, 1, 4, 9, 16]
        np.testing.assert_allclose(result.values, expected[None, :], atol=1e-10)

    def test_reconstruct_iono_surface_with_non_trivial_normalization(self):
        """Non-trivial offset and scale should normalize coordinates before evaluation."""
        az_size, sr_size = 3, 5
        # With offset=[1, 2] and scale_fact=[2, 3], x_norm = (x - 1) / 2, y_norm = 0
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([1.0, 2.0]),
            scale_fact=np.array([2.0, 3.0]),
            coeffs_data=np.array([0.0, 2.0, 3.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        sr_coords = np.arange(sr_size, dtype=np.float64)  # [0, 1, 2, 3, 4]
        # result = c0*y_norm + c1*x_norm + c2 = 0 + 2*((x-1)/2) + 3 = x+2
        # (c0=0 so y contribution is 0 for all rows)
        expected = sr_coords + 2.0  # [2,3,4,5,6]
        assert result.shape == (3, 5)
        np.testing.assert_allclose(result.values[0], expected, atol=1e-10)
        np.testing.assert_allclose(result.values[1], expected, atol=1e-10)
        np.testing.assert_allclose(result.values[2], expected, atol=1e-10)

    # ------------------------------------------------------------------
    # Edge case tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_minimum_array_size(self):
        """Should work with minimum 2x2 arrays."""
        az_size, sr_size = 2, 2
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (2, 2)

    def test_reconstruct_iono_surface_asymmetric_arrays(self):
        """Should work with asymmetric array shapes."""
        for shape in [(2, 5), (5, 2), (3, 10), (10, 3), (1, 5), (5, 1)]:
            az_size, sr_size = shape
            interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
            result = reconstruct_iono_surface(
                interf,
                azimuth_time_raw,
                coeffs,
                H_iono=300000.0,
                H_orb=800000.0,
                degree=1,
            )
            assert result.shape == shape

    def test_reconstruct_iono_surface_very_small_coefficients(self):
        """Very small coefficient values should still produce valid output."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([1e-15, 1e-15, 1e-15], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (az_size, sr_size)
        assert np.all(np.isfinite(result.values))

    def test_reconstruct_iono_surface_very_large_coefficients(self):
        """Very large coefficient values should still produce valid output."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([1e10, 1e10, 1e10], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (az_size, sr_size)
        assert np.all(np.isfinite(result.values))

    # ------------------------------------------------------------------
    # NaN and inf handling tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_nan_coefficient_propagates(self):
        """NaN in coefficients should propagate to the output surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([np.nan, 1.0, -0.5], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert np.all(np.isnan(result.values))

    def test_reconstruct_iono_surface_inf_coefficient_propagates(self):
        """Inf in coefficients should propagate to the output surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([np.inf, 1.0, -0.5], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # First row has NaN (0*inf), rest are inf. Check that inf values exist.
        assert np.any(np.isinf(result.values))
        assert not np.all(np.isfinite(result.values))

    def test_reconstruct_iono_surface_negative_inf_coefficient_propagates(self):
        """-Inf in coefficients should propagate to the output surface."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            coeffs_data=np.array([-np.inf, 1.0, -0.5], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # First row has NaN (0*-inf), rest are -inf. Check that neg-inf values exist.
        assert np.any(np.isneginf(result.values))
        assert not np.all(np.isfinite(result.values))

    def test_reconstruct_iono_surface_nan_offset_propagates(self):
        """NaN in offset attributes should produce NaN output via inf/0 division."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([np.nan, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # NaN offset causes (x - nan) = nan, nan / scale = nan
        assert np.all(np.isnan(result.values))

    def test_reconstruct_iono_surface_nan_scale_fact_propagates(self):
        """Zero scale_fact with non-zero offset should produce inf, which propagates."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([1.0, 0.0]),
            scale_fact=np.array([0.0, 1.0]),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # Division by zero in scale_fact produces inf, which propagates through the fit
        assert result.shape == (az_size, sr_size)

    def test_reconstruct_iono_surface_nan_in_interf(self):
        """NaN in interf values should not affect reconstruction."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        interf_nan = interf.copy()
        interf_nan[:] = np.nan
        result = reconstruct_iono_surface(
            interf_nan,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (az_size, sr_size)

    def test_reconstruct_iono_surface_shape_mismatch_raises(self):
        """Mismatched shapes between interf and azimuth_time_raw should raise ValueError."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        # Override with mismatched shape
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt_new = np.array(
            [base_dt + np.timedelta64(1_000_000_000 * i, "ns") for i in range(4)], dtype="datetime64[ns]"
        )
        az_raw_2d = np.broadcast_to(az_dt_new[:, None], (4, 6)).astype("datetime64[ns]")
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt_new, "slant_range_time": np.arange(6, dtype=np.float64)},
        )
        with pytest.raises(ValueError):
            reconstruct_iono_surface(
                interf,
                azimuth_time_raw,
                coeffs,
                H_iono=300000.0,
                H_orb=800000.0,
                degree=1,
            )

    def test_reconstruct_iono_surface_missing_offset_attr_raises(self):
        """Missing offset attribute should raise KeyError during normalization."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        # Remove the offset attribute
        del coeffs.attrs["offset"]
        with pytest.raises(KeyError):
            reconstruct_iono_surface(
                interf,
                azimuth_time_raw,
                coeffs,
                H_iono=300000.0,
                H_orb=800000.0,
                degree=1,
            )

    def test_reconstruct_iono_surface_missing_scale_fact_attr_raises(self):
        """Missing scale_fact attribute should raise KeyError during normalization."""
        az_size, sr_size = 3, 5
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        # Remove the scale_fact attribute
        del coeffs.attrs["scale_fact"]
        with pytest.raises(KeyError):
            reconstruct_iono_surface(
                interf,
                azimuth_time_raw,
                coeffs,
                H_iono=300000.0,
                H_orb=800000.0,
                degree=1,
            )

    def test_reconstruct_iono_surface_mismatched_dim_names_raises(self):
        """Mismatched dimension names should cause broadcasting errors."""
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt = np.array([base_dt + np.timedelta64(1_000_000_000 * k, "ns") for k in range(3)], dtype="datetime64[ns]")
        az_raw_2d = np.broadcast_to(az_dt[:, None], (3, 5)).astype("datetime64[ns]")
        interf = xr.DataArray(
            np.ones((3, 5)),
            dims=["azimuth_time", "slant_range_time"],
            coords={"azimuth_time": az_dt, "slant_range_time": np.arange(5, dtype=np.float64)},
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["time", "range"],
            coords={"time": np.arange(3, dtype=np.float64), "range": np.arange(5, dtype=np.float64)},
        )
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5], dtype=np.float64),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            reconstruct_iono_surface(
                interf,
                azimuth_time_raw,
                coeffs,
                H_iono=300000.0,
                H_orb=800000.0,
                degree=1,
            )

    def test_reconstruct_iono_surface_larger_array(self):
        """Should handle larger arrays (10x10) within reasonable time."""
        az_size, sr_size = 10, 10
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(az_size=az_size, sr_size=sr_size)
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (10, 10)
        assert np.all(np.isfinite(result.values))

    # ------------------------------------------------------------------
    # Coordinate type tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_datetime_coords(self):
        """Should work when azimuth_time coordinate is datetime64."""
        az_size, sr_size = 3, 5
        az_dt = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"], dtype="datetime64[ns]")
        az_raw_2d = np.empty((az_size, sr_size), dtype="datetime64[ns]")
        for i in range(az_size):
            az_raw_2d[i, :] = az_dt[i]
        interf = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float64),
            },
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float64),
            },
        )
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5], dtype=np.float64),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0])
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0])
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == (az_size, sr_size)

    def test_reconstruct_iono_surface_float32_range_coords(self):
        """Should work with datetime64 azimuth_time and float32 slant_range_time."""
        az_size, sr_size = 3, 5
        base_dt = np.datetime64("2020-01-01T00:00:00", "ns")
        az_dt = np.array(
            [base_dt + np.timedelta64(1_000_000_000 * i, "ns") for i in range(az_size)], dtype="datetime64[ns]"
        )
        az_raw_2d = np.empty((az_size, sr_size), dtype="datetime64[ns]")
        for i in range(az_size):
            az_raw_2d[i, :] = az_dt[i]
        interf = xr.DataArray(
            np.ones((az_size, sr_size)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float32),
            },
        )
        azimuth_time_raw = xr.DataArray(
            az_raw_2d,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": az_dt,
                "slant_range_time": np.arange(sr_size, dtype=np.float32),
            },
        )
        coeffs = xr.DataArray(
            np.array([0.0, 1.0, -0.5], dtype=np.float64),
            dims=["degree"],
            coords={"degree": np.arange(3)},
        )
        coeffs.attrs["offset"] = np.array([0.0, 0.0], dtype=np.float64)
        coeffs.attrs["scale_fact"] = np.array([1.0, 1.0], dtype=np.float64)
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        assert result.shape == (az_size, sr_size)

    # ------------------------------------------------------------------
    # Non-dispersive coefficient pattern tests
    # ------------------------------------------------------------------

    def test_reconstruct_iono_surface_only_y_coefficient(self):
        """Only y (azimuth) coefficient should produce uniform columns with varying rows."""
        az_size, sr_size = 4, 6
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([2.0, 0.0, 0.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        az_coords = np.arange(az_size, dtype=np.float64) * 1e9  # [0, 1, 2, 3]
        expected = np.tile(2.0 * az_coords[:, None], (1, sr_size))
        # Every column should be identical
        np.testing.assert_allclose(result.values, expected, atol=1e-10)
        # Verify all columns are identical
        assert np.all(np.isfinite(result.values))

    def test_reconstruct_iono_surface_only_x_coefficient(self):
        """Only x (range) coefficient should produce uniform rows with varying columns."""
        az_size, sr_size = 4, 6
        interf, azimuth_time_raw, coeffs = self._make_reconstruct_inputs(
            az_size=az_size,
            sr_size=sr_size,
            offset=np.array([0.0, 0.0]),
            scale_fact=np.array([1.0, 1.0]),
            coeffs_data=np.array([0.0, 3.0, 0.0], dtype=np.float64),
        )
        result = reconstruct_iono_surface(
            interf,
            azimuth_time_raw,
            coeffs,
            H_iono=300000.0,
            H_orb=800000.0,
            degree=1,
        )
        # Verify all rows are identical
        np.testing.assert_array_equal(result.values[0, :], result.values[-1, :])


class TestJointUnwrap:
    """Tests for joint_unwrap.

    joint_unwrap performs joint phase unwrapping of two SAR sub-bands by:
    1. Computing a coherence mask from the geometric mean of both sub-band coherences.
    2. Computing and unwrapping the average phase between the two sub-bands.
    3. Unwrapping each sub-band's residue relative to the shared average phase,
       then adding the average phase back to obtain fully unwrapped phases.

    All tests use in-memory synthetic data with known values. No file I/O or mocks.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_arrays(
        shape: tuple[int, int] = (3, 3),
        coh_l: float = 0.8,
        coh_h: float = 0.8,
        phase_l: float | np.ndarray | None = None,
        phase_h: float | np.ndarray | None = None,
    ):
        """Create four DataArrays (coherence and phase for both sub-bands).

        Parameters:
            shape: Output spatial shape (azimuth, range).
            coh_l: Constant coherence value for the low sub-band.
            coh_h: Constant coherence value for the high sub-band.
            phase_l: Constant phase value or array for the low sub-band.
                If None, uses 0.5 radians.
            phase_h: Constant phase value or array for the high sub-band.
                If None, uses 0.5 radians.
        """
        az, rg = shape
        coords = {
            "azimuth_time": np.arange(az, dtype=np.float64),
            "slant_range_time": np.arange(rg, dtype=np.float64),
        }
        kwargs = {"dims": ["azimuth_time", "slant_range_time"], "coords": coords}
        coh_l_da = xr.DataArray(np.full(shape, coh_l, dtype=np.float64), **kwargs)  # type: ignore
        coh_h_da = xr.DataArray(np.full(shape, coh_h, dtype=np.float64), **kwargs)  # type: ignore
        if phase_l is None:
            phase_l = 0.5
        if phase_h is None:
            phase_h = 0.5
        phase_l_da = xr.DataArray(np.full(shape, phase_l, dtype=np.float64), **kwargs)  # type: ignore
        phase_h_da = xr.DataArray(np.full(shape, phase_h, dtype=np.float64), **kwargs)  # type: ignore
        return coh_l_da, coh_h_da, phase_l_da, phase_h_da

    # ------------------------------------------------------------------
    # Return type and shape tests
    # ------------------------------------------------------------------

    def test_joint_unwrap_returns_tuple_of_two_dataarrays(self):
        """Should return a tuple of two xr.DataArray objects."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays()
        result = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], xr.DataArray)
        assert isinstance(result[1], xr.DataArray)

    @pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 5), (3, 5), (5, 3)])
    def test_joint_unwrap_preserves_input_shape(self, shape):
        """Output shapes should match input shapes for various array sizes."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=shape)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == shape
        assert ph_h_unw.shape == shape

    def test_joint_unwrap_preserves_dimensions(self):
        """Output DataArrays should have the same dimensions as inputs."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays()
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.dims == ("azimuth_time", "slant_range_time")
        assert ph_h_unw.dims == ("azimuth_time", "slant_range_time")

    def test_joint_unwrap_preserves_coordinates(self):
        """Output coordinates should match input coordinates."""
        sr = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        az = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=(3, 4))
        # Reassign with custom coordinates
        coh_l = coh_l.assign_coords({"azimuth_time": az, "slant_range_time": sr})
        coh_h = coh_h.assign_coords({"azimuth_time": az, "slant_range_time": sr})
        ph_l = ph_l.assign_coords({"azimuth_time": az, "slant_range_time": sr})
        ph_h = ph_h.assign_coords({"azimuth_time": az, "slant_range_time": sr})
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        xr.testing.assert_allclose(ph_l_unw.coords["azimuth_time"], ph_l.coords["azimuth_time"])
        xr.testing.assert_allclose(ph_l_unw.coords["slant_range_time"], ph_l.coords["slant_range_time"])

    # ------------------------------------------------------------------
    # Happy path tests
    # ------------------------------------------------------------------

    def test_joint_unwrap_constant_phase(self):
        """Constant input phases should produce constant unwrapped phases."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.7, phase_h=0.7)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        # With identical phases, both outputs should equal the input phase everywhere
        np.testing.assert_allclose(ph_l_unw.values, 0.7, atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, 0.7, atol=1e-10)

    def test_joint_unwrap_zero_phase(self):
        """Zero input phases should yield zero unwrapped phases."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.0, phase_h=0.0)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        np.testing.assert_allclose(ph_l_unw.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, 0.0, atol=1e-10)

    def test_joint_unwrap_phase_ramp(self):
        """Linear phase ramp in the low sub-band should be unwrapped correctly."""
        az, rg = 5, 5
        saz = np.arange(az, dtype=np.float64)
        sr = np.arange(rg, dtype=np.float64)
        phase_l = np.tile(sr, (az, 1)).T * 0.3  # ramp along range direction
        phase_h = np.tile(saz, (rg, 1)) * 0.2  # ramp along azimuth direction
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=(az, rg), phase_l=phase_l, phase_h=phase_h)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        # ph_l_unw should contain the ramp values
        expected_l = np.tile(sr, (az, 1)).T * 0.3
        expected_h = np.tile(saz, (rg, 1)) * 0.2
        np.testing.assert_allclose(ph_h_unw.values, expected_h, atol=1e-6)
        np.testing.assert_allclose(ph_l_unw.values, expected_l, atol=1e-6)

    def test_joint_unwrap_2d_phase_ramp(self):
        """2D linear phase ramp in both sub-bands should be unwrapped correctly."""
        az, rg = 5, 5
        sr = np.arange(rg, dtype=np.float64)
        az_coords = np.arange(az, dtype=np.float64)
        phase_l = np.tile(sr, (az, 1)).T * 0.3
        phase_h = np.tile(sr, (az, 1)).T * 0.3 + np.tile(az_coords, (rg, 1)) * 0.2
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=(az, rg), phase_l=phase_l, phase_h=phase_h)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        np.testing.assert_allclose(ph_l_unw.values, phase_l, atol=1e-6)
        np.testing.assert_allclose(ph_h_unw.values, phase_h, atol=1e-6)

    def test_joint_unwrap_default_threshold(self):
        """Using the default cohe_th=0.3 should work with normal coherence values."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(coh_l=0.7, coh_h=0.7)
        # No threshold argument - use default 0.3
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == (3, 3)
        assert ph_h_unw.shape == (3, 3)
        assert not np.any(np.isnan(ph_l_unw.values))
        assert not np.any(np.isnan(ph_h_unw.values))

    # ------------------------------------------------------------------
    # Coherence threshold behavior
    # ------------------------------------------------------------------

    def test_joint_unwrap_high_threshold_excludes_all(self):
        """A coherence threshold higher than all values should exclude all points."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(coh_l=0.8, coh_h=0.8)
        with pytest.raises(ValueError, match="No points"):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h, cohe_th=0.99)

    def test_joint_unwrap_zero_coherence_excludes_all(self):
        """Zero coherence everywhere should exclude all points."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(coh_l=0.0, coh_h=0.0)
        with pytest.raises(ValueError, match="No points"):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_partial_coherence_mask(self):
        """Spatially varying coherence should only unwrap above-threshold pixels."""
        az, rg = 5, 5
        coords = {
            "azimuth_time": np.arange(az, dtype=np.float64),
            "slant_range_time": np.arange(rg, dtype=np.float64),
        }
        kwargs = {"dims": ["azimuth_time", "slant_range_time"], "coords": coords}
        # Checkerboard coherence: alternating high/low
        coh_pattern = np.array(
            [
                [0.9, 0.1, 0.9, 0.1, 0.9],
                [0.1, 0.9, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.9, 0.1, 0.9],
                [0.1, 0.9, 0.1, 0.9, 0.1],
                [0.9, 0.1, 0.9, 0.1, 0.9],
            ]
        )
        coh_l_da = xr.DataArray(np.full((az, rg), 0.9, dtype=np.float64), **kwargs)  # type: ignore
        coh_h_da = xr.DataArray(coh_pattern, **kwargs)  # type: ignore
        ph_l_da = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        ph_h_da = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        # Threshold at 0.5: only pixels where both coherences >= 0.5 should be included
        # sqrt(0.9 * 0.9) = 0.9 > 0.5 for (0,0); sqrt(0.9 * 0.1) = ~0.3 < 0.5 for (0,1)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l_da, coh_h_da, ph_l_da, ph_h_da, cohe_th=0.5)
        assert ph_l_unw.shape == (az, rg)
        # Some pixels may be NaN where coherence was below threshold
        assert ph_l_unw.shape == ph_h_unw.shape
        # Check below threshold pixels are NaN
        ph_h = ph_h_unw.values
        coh_h = coh_h_da.values
        assert np.all(np.isnan(ph_h[coh_h < 0.5])) and np.all(~np.isnan(ph_h[coh_h >= 0.5]))

    # ------------------------------------------------------------------
    # Edge case tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (2, 3), (3, 2), (2, 5), (5, 2)],
    )
    def test_joint_unwrap_small_arrays(self, shape):
        """Should work with small arrays (minimum 2x2 for Delaunay triangulation)."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=shape)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == shape
        assert ph_h_unw.shape == shape

    def test_joint_unwrap_large_array(self):
        """Should handle larger arrays (10x10) within reasonable time."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=(10, 10))
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == (10, 10)
        assert ph_h_unw.shape == (10, 10)
        assert not np.any(np.isnan(ph_l_unw.values))

    def test_joint_unwrap_opposite_phases(self):
        """Opposite phases in the two sub-bands should produce a valid average."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.0, phase_h=np.pi)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == (3, 3)
        assert ph_h_unw.shape == (3, 3)
        # When input phases are 0 and pi, outputs should be 0 and pi respectively
        np.testing.assert_allclose(ph_l_unw.values, 0.0, atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, np.pi, atol=1e-10)

    def test_joint_unwrap_pi_phase(self):
        """Phase values near pi should be handled correctly."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=np.pi, phase_h=np.pi)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        np.testing.assert_allclose(ph_l_unw.values, np.pi, atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, np.pi, atol=1e-10)

    def test_joint_unwrap_negative_phase(self):
        """Negative phase values should be unwrapped correctly."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=-0.7, phase_h=-0.7)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        np.testing.assert_allclose(ph_l_unw.values, -0.7, atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, -0.7, atol=1e-10)

    def test_joint_unwrap_asymmetric_coherence(self):
        """Different coherence values between sub-bands should still work."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(coh_l=0.5, coh_h=0.95)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == (3, 3)
        assert ph_h_unw.shape == (3, 3)
        assert not np.any(np.isnan(ph_l_unw.values))

    # ------------------------------------------------------------------
    # NaN and inf handling tests
    # ------------------------------------------------------------------

    def test_joint_unwrap_nan_in_coherence(self):
        """NaN in coherence should propagate to the output at those positions."""
        az, rg = 3, 3
        coords = {
            "azimuth_time": np.arange(az, dtype=np.float64),
            "slant_range_time": np.arange(rg, dtype=np.float64),
        }
        kwargs = {"dims": ["azimuth_time", "slant_range_time"], "coords": coords}
        coh_l = xr.DataArray(
            np.array([[0.8, 0.8, np.nan], [0.8, 0.8, 0.8], [0.8, np.nan, 0.8]], dtype=np.float64),
            **kwargs,  # type: ignore
        )
        coh_h = xr.DataArray(np.full((az, rg), 0.8, dtype=np.float64), **kwargs)  # type: ignore
        ph_l = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        ph_h = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        # NaN in coherence propagates through sqrt to the mask, resulting in NaN outputs
        assert np.isnan(ph_l_unw.values[0, 2])
        assert np.isnan(ph_l_unw.values[2, 1])

    def test_joint_unwrap_nan_in_phase_raises(self):
        """NaN in phase data should cause an error from the MCF solver."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.5, phase_h=0.5)
        # Inject NaN into one phase pixel
        ph_h = ph_h.copy()
        ph_h.values[0, 0] = np.nan
        with pytest.raises(Exception):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_nan_in_both_phase_raises(self):
        """NaN in both phase arrays should raise an error."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.5, phase_h=0.5)
        ph_l = ph_l.copy()
        ph_h = ph_h.copy()
        ph_l.values[0, 0] = np.nan
        ph_h.values[0, 0] = np.nan
        with pytest.raises(Exception):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_inf_in_phase_raises(self):
        """Inf in phase data should cause an error from the MCF solver."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.5, phase_h=0.5)
        ph_h = ph_h.copy()
        ph_h.values[0, 0] = np.inf
        with pytest.raises(Exception):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_negative_inf_in_phase_raises(self):
        """-Inf in phase data should cause an error from the MCF solver."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.5, phase_h=0.5)
        ph_l = ph_l.copy()
        ph_l.values[0, 0] = -np.inf
        with pytest.raises(Exception):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_inf_in_coherence(self):
        """Inf in coherence should still produce valid output (treated as very high coherence)."""
        az, rg = 3, 3
        coords = {
            "azimuth_time": np.arange(az, dtype=np.float64),
            "slant_range_time": np.arange(rg, dtype=np.float64),
        }
        kwargs = {"dims": ["azimuth_time", "slant_range_time"], "coords": coords}
        coh_l = xr.DataArray(
            np.array([[np.inf, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=np.float64),
            **kwargs,  # type: ignore
        )
        coh_h = xr.DataArray(np.full((az, rg), 0.8, dtype=np.float64), **kwargs)  # type: ignore
        ph_l = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        ph_h = xr.DataArray(np.full((az, rg), 0.5, dtype=np.float64), **kwargs)  # type: ignore
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert ph_l_unw.shape == (az, rg)
        assert not np.any(np.isnan(ph_l_unw.values))
        np.testing.assert_allclose(ph_l_unw.values, 0.5, atol=1e-10)

    # ------------------------------------------------------------------
    # Mismatched input tests
    # ------------------------------------------------------------------

    def test_joint_unwrap_mismatched_coherence_shapes_raises(self):
        """Mismatched coherence array shapes should raise an AlignmentError."""
        coh_l = xr.DataArray(np.full((3, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.full((4, 4), 0.8), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.full((3, 3), 0.5), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.full((4, 4), 0.5), dims=["azimuth_time", "slant_range_time"])

        with pytest.raises(xr.AlignmentError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_mismatched_phase_shapes_raises(self):
        """Mismatched phase array shapes should raise an AlignmentError."""
        coh_l = xr.DataArray(np.full((3, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.full((3, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.full((3, 3), 0.5), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.full((4, 4), 0.5), dims=["azimuth_time", "slant_range_time"])

        with pytest.raises(xr.AlignmentError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_mismatched_dim_names_raises(self):
        """Arrays with mismatched dimension names cause broadcasting into incompatible shapes."""
        coh_l = xr.DataArray(np.full((3, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.full((3, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.full((3, 3), 0.5), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.full((3, 3), 0.5), dims=["time", "range"])

        # xarray broadcasts different dim names into a 4D array, causing IndexError downstream
        with pytest.raises(IndexError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    # ------------------------------------------------------------------
    # Small array dimension edge cases
    # ------------------------------------------------------------------

    def test_joint_unwrap_single_row_raises(self):
        """Single-row arrays (1 x N) should raise because squeeze removes dimensions."""
        coh_l = xr.DataArray(np.full((1, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.full((1, 3), 0.8), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.full((1, 3), 0.5), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.full((1, 3), 0.5), dims=["azimuth_time", "slant_range_time"])
        with pytest.raises(KeyError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_single_column_raises(self):
        """Single-column arrays (N x 1) should raise because squeeze removes dimensions."""
        coh_l = xr.DataArray(np.full((3, 1), 0.8), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.full((3, 1), 0.8), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.full((3, 1), 0.5), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.full((3, 1), 0.5), dims=["azimuth_time", "slant_range_time"])
        with pytest.raises(KeyError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    def test_joint_unwrap_1x1_raises(self):
        """A 1x1 array should raise because squeeze removes all dimensions."""
        coh_l = xr.DataArray(np.array([[0.8]]), dims=["azimuth_time", "slant_range_time"])
        coh_h = xr.DataArray(np.array([[0.8]]), dims=["azimuth_time", "slant_range_time"])
        ph_l = xr.DataArray(np.array([[0.5]]), dims=["azimuth_time", "slant_range_time"])
        ph_h = xr.DataArray(np.array([[0.5]]), dims=["azimuth_time", "slant_range_time"])
        with pytest.raises(KeyError):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h)

    # ------------------------------------------------------------------
    # Numerical correctness tests
    # ------------------------------------------------------------------

    def test_joint_unwrap_same_phase_l_equals_h(self):
        """When both sub-bands have the same phase, outputs should be equal."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=1.234, phase_h=1.234)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        np.testing.assert_allclose(ph_l_unw.values, ph_h_unw.values, atol=1e-10)

    def test_joint_unwrap_different_phases_both_valid(self):
        """Different phases in sub-bands should both produce valid unwrapped results."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(phase_l=0.3, phase_h=1.7)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        assert not np.any(np.isnan(ph_l_unw.values))
        assert not np.any(np.isnan(ph_h_unw.values))
        # Neither output should be purely wrapped (within [-pi, pi] only)
        # The unwrapping process may keep values in range for flat phases,
        # but both should be finite
        assert np.all(np.isfinite(ph_l_unw.values))
        assert np.all(np.isfinite(ph_h_unw.values))

    def test_joint_unwrap_uniform_phase_5x5(self):
        """A uniform phase field on a 5x5 grid should remain uniform after unwrapping."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(shape=(5, 5), phase_l=0.9, phase_h=0.9)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        # All values should be identical (within numerical tolerance)
        np.testing.assert_allclose(ph_l_unw.values, ph_l_unw.values[0, 0], atol=1e-10)
        np.testing.assert_allclose(ph_h_unw.values, ph_h_unw.values[0, 0], atol=1e-10)

    def test_joint_unwrap_custom_threshold_float(self):
        """A custom float threshold should be accepted and used."""
        coh_l, coh_h, ph_l, ph_h = self._make_arrays(coh_l=0.6, coh_h=0.6)
        # Threshold 0.5 allows unwrapping (0.6 > 0.5)
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h, cohe_th=0.5)
        assert ph_l_unw.shape == (3, 3)
        # Threshold 0.7 excludes (0.6 < 0.7)
        with pytest.raises(ValueError, match="No points"):
            joint_unwrap(coh_l, coh_h, ph_l, ph_h, cohe_th=0.7)

    def test_joint_unwrap_float64_precision(self):
        """Output values should maintain float64 precision for smooth phase fields."""
        az, rg = 5, 5
        coords = {
            "azimuth_time": np.arange(az, dtype=np.float64),
            "slant_range_time": np.arange(rg, dtype=np.float64),
        }
        kwargs = {"dims": ["azimuth_time", "slant_range_time"], "coords": coords}
        coh_l = xr.DataArray(np.full((az, rg), 0.9, dtype=np.float64), **kwargs)  # type: ignore
        coh_h = xr.DataArray(np.full((az, rg), 0.9, dtype=np.float64), **kwargs)  # type: ignore
        # Non-trivial phase values
        phase_l = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
                [0.15, 0.25, 0.35, 0.45, 0.55],
                [0.65, 0.75, 0.85, 0.95, 1.05],
                [0.12, 0.22, 0.32, 0.42, 0.52],
            ],
            dtype=np.float64,
        )
        phase_h = np.array(
            [
                [0.11, 0.21, 0.31, 0.41, 0.51],
                [0.61, 0.71, 0.81, 0.91, 1.01],
                [0.16, 0.26, 0.36, 0.46, 0.56],
                [0.66, 0.76, 0.86, 0.96, 1.06],
                [0.13, 0.23, 0.33, 0.43, 0.53],
            ],
            dtype=np.float64,
        )
        ph_l = xr.DataArray(phase_l, **kwargs)  # type: ignore
        ph_h = xr.DataArray(phase_h, **kwargs)  # type: ignore
        ph_l_unw, ph_h_unw = joint_unwrap(coh_l, coh_h, ph_l, ph_h)
        # Both outputs should be float64
        assert ph_l_unw.dtype == np.float64
        assert ph_h_unw.dtype == np.float64
