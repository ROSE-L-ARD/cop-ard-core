"""Unit tests for cop_ard_core.slc."""

from xml.etree.ElementTree import Element

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.slc import (
    amplitude_calibration,
    apply_invalid_mask,
    deramp_numba,
    get_relative_orbit_number,
    image_deramp,
    intersect,
    merge_dict,
    noise_removal,
)


class TestMergeDict:
    """Tests for merge_dict."""

    def test_merge_dict_basic(self):
        """Basic merge should combine dictionaries."""
        res = [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        result = merge_dict(res)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_dict_overlapping_keys(self):
        """Later values should override earlier ones for same key."""
        res = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = merge_dict(res)
        assert result == {"a": 3}

    def test_merge_dict_empty_list(self):
        """Empty list should return empty dict."""
        result = merge_dict([])
        assert result == {}

    def test_merge_dict_single_dict(self):
        """Single dict should be returned as-is."""
        res = [{"a": 1}]
        result = merge_dict(res)
        assert result == {"a": 1}

    def test_merge_dict_nested_values(self):
        """Nested values should be copied by reference."""
        res = [{"a": [1, 2]}, {"b": {"c": 3}}]
        result = merge_dict(res)
        assert result == {"a": [1, 2], "b": {"c": 3}}


class TestApplyInvalidMask:
    """Tests for apply_invalid_mask."""

    def test_apply_invalid_mask_modifies_in_place(self):
        """Should zero out regions specified by valid_y and valid_x."""
        slc = xr.DataArray(
            np.ones((1, 4, 1, 4), dtype=np.complex128) * (1 + 1j),
            dims=["band", "az", "rg", "extra"],
        )
        valid_y = np.array([1, 0, 3, 2])
        valid_x = np.array([2, 1, 3, 0])
        result = apply_invalid_mask(slc, valid_y, valid_x)
        assert isinstance(result, xr.DataArray)

    def test_apply_invalid_mask_zeros_boundaries(self):
        """Top/bottom and left/right slices should be zeroed."""
        slc = xr.DataArray(
            np.ones((1, 5, 1, 5), dtype=np.complex128) * (1 + 1j),
            dims=["band", "az", "rg", "extra"],
        )
        valid_y = np.array([1, 0, 4, 3])
        valid_x = np.array([2, 1, 4, 1])
        result = apply_invalid_mask(slc, valid_y, valid_x)
        assert isinstance(result, xr.DataArray)

    def test_apply_invalid_mask_returns_same_type(self):
        """Return type should be xr.DataArray."""
        slc = xr.DataArray(
            np.ones((2, 4, 1, 4), dtype=np.complex128),
            dims=["band", "az", "rg", "extra"],
        )
        result = apply_invalid_mask(slc, np.array([1, 0, 3, 2]), np.array([1, 2, 3, 0]))
        assert isinstance(result, xr.DataArray)


class TestAmplitudeCalibration:
    """Tests for amplitude_calibration."""

    def test_amplitude_calibration_no_coeffs_returns_input(self, complex_3x3):
        """Without calibration coefficients, should return input unchanged."""
        result = amplitude_calibration(complex_3x3, None)
        xr.testing.assert_equal(result, complex_3x3)

    def test_amplitude_calibration_returns_dataarray(self, complex_3x3):
        """Should return xr.DataArray."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = amplitude_calibration(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_amplitude_calibration_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = amplitude_calibration(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_amplitude_calibration_with_coords(self, complex_3x3):
        """Calibration requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = amplitude_calibration(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)


class TestNoiseRemoval:
    """Tests for noise_removal."""

    def test_noise_removal_no_coeffs_returns_input(self, complex_3x3):
        """Without noise coefficients, should return input unchanged."""
        result = noise_removal(complex_3x3, None)
        xr.testing.assert_equal(result, complex_3x3)

    def test_noise_removal_returns_dataarray(self, complex_3x3):
        """Should return xr.DataArray."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_noise_removal_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_noise_removal_with_coords(self, complex_3x3):
        """Noise removal requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)


class TestImageDeramp:
    """Tests for image_deramp."""

    def test_image_deramp_no_coeffs_returns_input(self, complex_3x3):
        """Without dkcoa coefficients, should return input unchanged."""
        result = image_deramp(complex_3x3, None)
        xr.testing.assert_equal(result, complex_3x3)

    def _make_4d_slc(self, complex_3x3):
        """Build a 4D SLC (polarization, acquisition_date, azimuth, slant_range) with meter coords."""
        slc4 = complex_3x3.expand_dims("polarization", axis=0).expand_dims("acquisition_date", axis=0)
        return slc4.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)), "azimuth_m": (["azimuth_time"], np.arange(3))}
        )

    def test_image_deramp_returns_dataarray(self, complex_3x3):
        """Should return xr.DataArray."""
        coeffs = np.ones((2, 2))
        slc_with_coords = self._make_4d_slc(complex_3x3)
        result = image_deramp(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_image_deramp_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = self._make_4d_slc(complex_3x3)
        result = image_deramp(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_image_deramp_with_coords(self, complex_3x3):
        """Deramping requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = self._make_4d_slc(complex_3x3)
        result = image_deramp(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    @staticmethod
    def _build_4d_slc(
        data: np.ndarray,
        slant_range_m: np.ndarray,
        azimuth_m: np.ndarray,
    ) -> xr.DataArray:
        """Create a 4D SLC DataArray with required dimensions and meter coordinates."""
        npol, ndates, n_az, n_rg = data.shape
        return xr.DataArray(
            data,
            dims=["polarization", "acquisition_id", "azimuth_time", "slant_range_time"],
            coords={
                "polarization": [f"pol_{i}" for i in range(npol)],
                "acquisition_id": [f"2020-01-{i + 1:02d}" for i in range(ndates)],
                "azimuth_time": np.arange(n_az),
                "slant_range_time": np.arange(n_rg),
                "azimuth_m": ("azimuth_time", azimuth_m),
                "slant_range_m": ("slant_range_time", slant_range_m),
            },
        )

    @pytest.mark.unit
    def test_image_deramp_zero_coeffs_is_identity(self):
        """Zero Doppler centroid polynomial must leave amplitude and phase unchanged."""
        rng = np.random.default_rng(10)
        raw = (rng.standard_normal((2, 2, 8, 6)) + 1j * rng.standard_normal((2, 2, 8, 6))).astype(np.complex128)
        slc = self._build_4d_slc(raw.copy(), np.arange(6, dtype=np.float64), np.arange(8, dtype=np.float64))

        result = image_deramp(slc, np.zeros((3, 3), dtype=np.float64))
        np.testing.assert_allclose(result.values, raw, atol=1e-14)

    @pytest.mark.unit
    def test_image_deramp_removes_constant_doppler_linear_ramp(self):
        """Constant Doppler centroid f_dc = f0 produces linear azimuth phase 2*pi*f0*a that deramping removes."""
        f0 = 15.5  # cycles per unit azimuth coordinate
        dkcoa = np.array([[f0]], dtype=np.float64)

        n_az, n_rg = 20, 12
        azimuth_m = np.linspace(0.0, 2.0, n_az)
        slant_range_m = np.linspace(100.0, 200.0, n_rg)

        # True underlying scene has constant phase phi_0 = 0.45 rad and varying amplitude
        phi_0 = 0.45
        amplitude = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, n_az))[:, None] * np.ones((1, n_rg))
        ramp_phase = 2.0 * np.pi * f0 * azimuth_m[:, None]
        ramped_signal = (amplitude * np.exp(1j * (phi_0 + ramp_phase)))[None, None, :, :].astype(np.complex128)

        slc = self._build_4d_slc(ramped_signal, slant_range_m, azimuth_m)
        result = image_deramp(slc, dkcoa)

        # Phase ramp must be completely removed, leaving flat phi_0 across the scene
        residual_phase = np.angle(result.values[0, 0])
        np.testing.assert_allclose(residual_phase, phi_0, atol=1e-12)
        # Radiometric amplitude must be strictly conserved
        np.testing.assert_allclose(np.abs(result.values[0, 0]), amplitude, atol=1e-12)

    @pytest.mark.unit
    def test_image_deramp_removes_quadratic_topsar_azimuth_chirp(self):
        """Linear Doppler rate f_dc(a) = f0 + k_a * a integrates to quadratic phase 2*pi*(f0*a + 0.5*k_a*a^2)."""
        f0 = -8.0
        k_a = 12.0
        dkcoa = np.array([[f0, k_a]], dtype=np.float64)

        n_az, n_rg = 24, 10
        azimuth_m = np.linspace(-1.0, 1.0, n_az)
        slant_range_m = np.linspace(0.0, 50.0, n_rg)

        quad_phase = 2.0 * np.pi * (f0 * azimuth_m[:, None] + 0.5 * k_a * (azimuth_m[:, None] ** 2))
        signal = np.broadcast_to(np.exp(1j * quad_phase), (n_az, n_rg))[None, None, :, :].astype(np.complex128).copy()

        slc = self._build_4d_slc(signal, slant_range_m, azimuth_m)
        result = image_deramp(slc, dkcoa)

        np.testing.assert_allclose(result.values[0, 0], 1.0 + 0j, atol=1e-12)

    @pytest.mark.unit
    def test_image_deramp_2d_coupled_range_azimuth_polynomial(self):
        """2D range-azimuth coupled Doppler centroid polynomial must integrate along azimuth and deramp accurately."""
        # f_dc(r, a) = c00 + c01*a + c02*a^2 + c10*r + c11*r*a + c20*r^2
        dkcoa = np.array(
            [
                [2.5, -1.2, 0.3],
                [0.04, 0.02, -0.005],
                [-0.001, 0.0005, 0.0],
            ],
            dtype=np.float64,
        )

        npol, ndates, n_az, n_rg = 2, 3, 16, 12
        azimuth_m = np.linspace(-2.0, 2.0, n_az)
        slant_range_m = np.linspace(0.0, 30.0, n_rg)

        # Analytical integral along azimuth (axis=1) multiplied by 2*pi
        integrated_coeff = (2.0 * np.pi) * np.polynomial.polynomial.polyint(dkcoa, axis=1)
        expected_phase = np.polynomial.polynomial.polygrid2d(slant_range_m, azimuth_m, integrated_coeff).T

        rng = np.random.default_rng(99)
        base_scene = rng.uniform(0.5, 3.0, size=(npol, ndates, n_az, n_rg)) * np.exp(
            1j * rng.uniform(-np.pi, np.pi, size=(npol, ndates, n_az, n_rg))
        )
        ramped_data = (base_scene * np.exp(1j * expected_phase)[None, None, :, :]).astype(np.complex128)

        slc = self._build_4d_slc(ramped_data, slant_range_m, azimuth_m)
        result = image_deramp(slc, dkcoa)

        # Deramped 4D stack must recover the un-ramped complex scene across all pols and dates
        np.testing.assert_allclose(result.values, base_scene, rtol=1e-11, atol=1e-11)

    @pytest.mark.unit
    def test_image_deramp_preserves_complex64_dtype(self):
        """Native Sentinel-1 complex64 SLC arrays must retain complex64 dtype after deramping."""
        dkcoa = np.array([[5.0, -2.0]], dtype=np.float64)
        n_az, n_rg = 10, 8
        azimuth_m = np.linspace(0.0, 1.0, n_az)
        slant_range_m = np.linspace(0.0, 10.0, n_rg)

        phase = 2.0 * np.pi * (5.0 * azimuth_m[:, None] - azimuth_m[:, None] ** 2)
        data_c64 = np.broadcast_to(np.exp(1j * phase), (n_az, n_rg))[None, None, :, :].astype(np.complex64).copy()

        slc = self._build_4d_slc(data_c64, slant_range_m, azimuth_m)
        result = image_deramp(slc, dkcoa)

        assert result.dtype == np.complex64
        np.testing.assert_allclose(result.values[0, 0], 1.0 + 0j, atol=1e-5)


class TestIntersect:
    """Tests for intersect function."""

    def test_intersect_wkt_strings(self):
        """Two overlapping polygons should return True."""
        poly1 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        poly2 = "POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))"
        assert intersect(poly1, poly2) is True

    def test_intersect_non_overlapping(self):
        """Two non-overlapping polygons should return False."""
        poly1 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        poly2 = "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))"
        assert intersect(poly1, poly2) is False

    def test_intersect_list_coords(self):
        """Polygons as coordinate lists should work."""
        poly1 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        poly2 = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
        assert intersect(poly1, poly2) is True

    def test_intersect_one_inside_other(self):
        """A smaller polygon inside a larger one should intersect."""
        poly1 = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
        poly2 = "POLYGON ((2 2, 4 2, 4 4, 2 4, 2 2))"
        assert intersect(poly1, poly2) is True

    def test_intersect_tangent_edges(self):
        """Two polygons sharing an edge should intersect."""
        poly1 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        poly2 = "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))"
        assert intersect(poly1, poly2) is True

    def test_intersect_list_of_lists(self):
        """List of lists should work (not just list of tuples)."""
        poly1 = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        poly2 = "POLYGON ((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))"
        assert intersect(poly1, poly2) is True


class TestGetRelativeOrbitNumber:
    """Tests for get_relative_orbit_number."""

    def test_get_relative_orbit_number(self):
        """Should extract relative orbit number from manifest element."""
        ns = {"safe": "http://www.esa.int/safe/sentinel-1.0"}
        manifest = Element("{http://www.esa.int/safe/sentinel-1.0}manifest")
        orbit_num = Element("{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber")
        orbit_num.text = "67"
        manifest.append(orbit_num)
        result = get_relative_orbit_number(manifest, ns)
        assert result == 67

    def test_get_relative_orbit_number_zero(self):
        """Relative orbit number can be zero."""
        ns = {"safe": "http://www.esa.int/safe/sentinel-1.0"}
        manifest = Element("{http://www.esa.int/safe/sentinel-1.0}manifest")
        orbit_num = Element("{http://www.esa.int/safe/sentinel-1.0}relativeOrbitNumber")
        orbit_num.text = "0"
        manifest.append(orbit_num)
        result = get_relative_orbit_number(manifest, ns)
        assert result == 0


class TestDerampNumba:
    """Direct unit tests for the low-level deramp_numba kernel."""

    @pytest.mark.unit
    def test_deramp_numba_matches_polygrid2d_horner_evaluation(self):
        """2D Horner evaluation in deramp_numba must match polygrid2d phase rotation."""
        npol, ndates, n_az, n_rg = 2, 3, 14, 10
        slant_range = np.linspace(800_000.0, 805_000.0, n_rg, dtype=np.float64)
        azimuth = np.linspace(-1.5, 1.5, n_az, dtype=np.float64)

        coeff = np.array(
            [
                [0.2, 12.5, -3.1],
                [1e-4, -2e-4, 5e-5],
                [-1e-8, 3e-9, -1e-9],
            ],
            dtype=np.float64,
        )

        expected_phase = np.polynomial.polynomial.polygrid2d(slant_range, azimuth, coeff).T
        expected_rot = np.exp(-1j * expected_phase)

        rng = np.random.default_rng(7)
        raw = (
            rng.standard_normal((npol, ndates, n_az, n_rg)) + 1j * rng.standard_normal((npol, ndates, n_az, n_rg))
        ).astype(np.complex128)
        slc = raw.copy()

        deramp_numba(slc, slant_range, azimuth, coeff)

        expected = raw * expected_rot[None, None, :, :]
        np.testing.assert_allclose(slc, expected, rtol=1e-10, atol=1e-10)
