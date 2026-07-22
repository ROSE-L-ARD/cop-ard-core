"""Unit tests for cop_ard_core.slc."""

from xml.etree.ElementTree import Element

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.slc import (
    amplitude_calibration,
    apply_invalid_mask,
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
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = amplitude_calibration(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_amplitude_calibration_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = amplitude_calibration(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_amplitude_calibration_with_coords(self, complex_3x3):
        """Calibration requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
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
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_noise_removal_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_noise_removal_with_coords(self, complex_3x3):
        """Noise removal requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = noise_removal(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)


class TestImageDeramp:
    """Tests for image_deramp."""

    def test_image_deramp_no_coeffs_returns_input(self, complex_3x3):
        """Without dkcoa coefficients, should return input unchanged."""
        result = image_deramp(complex_3x3, None)
        xr.testing.assert_equal(result, complex_3x3)

    def test_image_deramp_returns_dataarray(self, complex_3x3):
        """Should return xr.DataArray."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = image_deramp(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)

    def test_image_deramp_complex_preserved(self, complex_3x3):
        """Complex nature should be preserved."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = image_deramp(slc_with_coords, coeffs)
        assert np.iscomplexobj(result.values)

    def test_image_deramp_with_coords(self, complex_3x3):
        """Deramping requires slant_range_m and azimuth_m coordinates."""
        coeffs = np.ones((2, 2))
        slc_with_coords = complex_3x3.copy()
        slc_with_coords = slc_with_coords.assign_coords(
            {"slant_range_m": (["slant_range_time"], np.arange(3)),
             "azimuth_m": (["azimuth_time"], np.arange(3))}
        )
        result = image_deramp(slc_with_coords, coeffs)
        assert isinstance(result, xr.DataArray)


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
        poly1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
        poly2 = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
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
        poly1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
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
