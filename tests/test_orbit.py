"""Unit tests for cop_ard_core.orbit."""

from xml.etree.ElementTree import Element, ElementTree

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.orbit import build_orbit_dataset, refit_orbit_coefficients


class TestBuildOrbitDataset:
    """Tests for build_orbit_dataset."""

    def test_build_orbit_dataset_returns_dataset(self, orbit_xml_element):
        """Should return an xr.Dataset."""
        result = build_orbit_dataset([orbit_xml_element])
        assert isinstance(result, xr.Dataset)

    def test_build_orbit_dataset_has_position_and_velocity(self, orbit_xml_element):
        """Dataset should contain 'position' and 'velocity' variables."""
        result = build_orbit_dataset([orbit_xml_element])
        assert "position" in result.data_vars
        assert "velocity" in result.data_vars

    def test_build_orbit_dataset_position_shape(self, orbit_xml_element):
        """Position should have shape (n_observations, 3)."""
        result = build_orbit_dataset([orbit_xml_element])
        assert result.position.shape == (3, 3)

    def test_build_orbit_dataset_velocity_shape(self, orbit_xml_element):
        """Velocity should have shape (n_observations, 3)."""
        result = build_orbit_dataset([orbit_xml_element])
        assert result.velocity.shape == (3, 3)

    def test_build_orbit_dataset_has_azimuth_time_coord(self, orbit_xml_element):
        """Dataset should have azimuth_time coordinate."""
        result = build_orbit_dataset([orbit_xml_element])
        assert "azimuth_time" in result.coords

    def test_build_orbit_dataset_has_ut1_and_tai(self, orbit_xml_element):
        """Dataset should have ut1 and tai coordinates."""
        result = build_orbit_dataset([orbit_xml_element])
        assert "ut1" in result.coords
        assert "tai" in result.coords

    def test_build_orbit_dataset_has_axis_coord(self, orbit_xml_element):
        """Dataset should have axis coordinate ['x', 'y', 'z']."""
        result = build_orbit_dataset([orbit_xml_element])
        assert list(result.axis.values) == ["x", "y", "z"]

    def test_build_orbit_dataset_has_type_attr(self, orbit_xml_element):
        """Dataset should have 'type' attribute set to orbit type."""
        result = build_orbit_dataset([orbit_xml_element])
        assert result.attrs.get("type") == "POEORB"

    def test_build_orbit_dataset_position_values(self, orbit_xml_element):
        """Position values should match the XML input."""
        result = build_orbit_dataset([orbit_xml_element])
        # X values from XML: 6500000.0, 6501000.0, 6502000.0
        np.testing.assert_array_equal(result.position.values[:, 0], np.array([6500000.0, 6501000.0, 6502000.0]))

    def test_build_orbit_dataset_velocity_values(self, orbit_xml_element):
        """Velocity values should match the XML input."""
        result = build_orbit_dataset([orbit_xml_element])
        # VX values from XML: 7500.0, 7500.5, 7501.0
        np.testing.assert_array_equal(result.velocity.values[:, 0], np.array([7500.0, 7500.5, 7501.0]))

    def test_build_orbit_dataset_mixed_orbit_types_raises(self):
        """Multiple orbit files of different types should raise RuntimeError."""
        tree1 = ElementTree(Element("ROOT"))
        file_type1 = Element("File_Type")
        file_type1.text = "POEORB"
        root1 = tree1.getroot()
        assert root1 is not None
        root1.append(file_type1)
        list_osvs1 = Element("List_of_OSVs")
        osv = Element("OSV")
        for tag in ["UTC", "TAI", "UT1"]:
            t = Element(tag)
            t.text = "2020-01-01T00:00:00=0"
            osv.append(t)
        for coord in ["X", "Y", "Z"]:
            val = Element(coord)
            val.text = "6500000.0"
            osv.append(val)
        for coord in ["VX", "VY", "VZ"]:
            val = Element(coord)
            val.text = "7500.0"
            osv.append(val)
        list_osvs1.append(osv)
        root1.append(list_osvs1)

        tree2 = ElementTree(Element("ROOT"))
        file_type2 = Element("File_Type")
        file_type2.text = "RESORB"
        root2 = tree2.getroot()
        assert root2 is not None
        root2.append(file_type2)
        list_osvs2 = Element("List_of_OSVs")
        osv2 = Element("OSV")
        for tag in ["UTC", "TAI", "UT1"]:
            t = Element(tag)
            t.text = "2020-01-01T00:00:00=1"
            osv2.append(t)
        for coord in ["X", "Y", "Z"]:
            val = Element(coord)
            val.text = "6500000.0"
            osv2.append(val)
        for coord in ["VX", "VY", "VZ"]:
            val = Element(coord)
            val.text = "7500.0"
            osv2.append(val)
        list_osvs2.append(osv2)
        root2.append(list_osvs2)

        with pytest.raises(RuntimeError, match="same orbit type"):
            build_orbit_dataset([tree1, tree2])  # type: ignore

    def test_build_orbit_dataset_missing_osv_raises(self):
        """Orbit file without List_of_OSVs should raise ValueError."""
        tree = ElementTree(Element("ROOT"))
        file_type = Element("File_Type")
        file_type.text = "POEORB"
        root = tree.getroot()
        assert root is not None
        root.append(file_type)

        with pytest.raises(ValueError, match="error reading orbit file"):
            build_orbit_dataset([tree])  # type: ignore

    def test_build_orbit_dataset_with_many_observations(self):
        """Should handle many orbit observations."""
        root = Element("ROOT")
        file_type = Element("File_Type")
        file_type.text = "POEORB"
        root.append(file_type)
        list_osvs = Element("List_of_OSVs")
        for i in range(100):
            osv = Element("OSV")
            for tag in ["UTC", "TAI", "UT1"]:
                t = Element(tag)
                t.text = f"2020-01-01T00:{i // 60:02d}:{i % 60:02d}={i}"
                osv.append(t)
            for coord in ["X", "Y", "Z"]:
                val = Element(coord)
                val.text = f"{6500000.0 + i}"
                osv.append(val)
            for coord in ["VX", "VY", "VZ"]:
                val = Element(coord)
                val.text = f"{7500.0}"
                osv.append(val)
            list_osvs.append(osv)
        root.append(list_osvs)
        tree = ElementTree(root)

        result = build_orbit_dataset([tree])  # type: ignore
        assert result.position.shape == (100, 3)


class TestRefitOrbitCoefficients:
    """Tests for refit_orbit_coefficients."""

    def test_refit_orbit_coefficients_none_orbit(self):
        """None orbit should return None."""
        result = refit_orbit_coefficients(
            burst_id="test_burst",
            acquisition_id="2020-01-01",
            orbit=None,
            t0=np.datetime64("2020-01-01", "ns"),
            reference_time=np.array(["2020-01-01"], dtype="datetime64[ns]"),
            poly_degree=4,
        )
        assert result is None

    def test_refit_orbit_coefficients_with_orbit(self):
        """With a valid orbit, should return an xr.Dataset."""
        utc = np.array(
            [
                "2020-01-01T00:00:00",
                "2020-01-01T00:00:10",
                "2020-01-01T00:00:20",
                "2020-01-01T00:00:30",
                "2020-01-01T00:00:40",
            ],
            dtype="datetime64[ns]",
        )
        x_vals = np.array([6500000.0, 6507500.0, 6515000.0, 6522500.0, 6530000.0])
        y_vals = np.array([2000000.0, 2000000.0, 2000000.0, 2000000.0, 2000000.0])
        z_vals = np.array([4000000.0, 4000000.0, 4000000.0, 4000000.0, 4000000.0])
        vx_vals = np.array([7500.0, 7500.0, 7500.0, 7500.0, 7500.0])
        vy_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        vz_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        position = xr.DataArray(
            np.column_stack([x_vals, y_vals, z_vals]),
            dims=["azimuth_time", "axis"],
            coords={"azimuth_time": utc, "axis": ["x", "y", "z"]},
        )
        velocity = xr.DataArray(
            np.column_stack([vx_vals, vy_vals, vz_vals]),
            dims=["azimuth_time", "axis"],
            coords={"azimuth_time": utc, "axis": ["x", "y", "z"]},
        )
        orbit = xr.Dataset(
            data_vars={"position": position, "velocity": velocity},
            coords={"orbit_file": ["test.orbit"]},
            attrs={"type": "POEORB"},
        )

        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        reference_time = np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")

        result = refit_orbit_coefficients(
            burst_id="test_burst",
            acquisition_id="2020-01-01",
            orbit=orbit,
            t0=t0,
            reference_time=reference_time,
            poly_degree=4,
        )

        assert isinstance(result, xr.Dataset)
        assert "coeffs" in result.data_vars

    def test_refit_orbit_coefficients_returns_correct_dims(self):
        """Output should have dims: acquisition_id, degree, axis."""
        utc = np.array(
            [
                "2020-01-01T00:00:00",
                "2020-01-01T00:00:10",
                "2020-01-01T00:00:20",
                "2020-01-01T00:00:30",
                "2020-01-01T00:00:40",
            ],
            dtype="datetime64[ns]",
        )
        x_vals = np.array([6500000.0, 6507500.0, 6515000.0, 6522500.0, 6530000.0])
        y_vals = np.array([2000000.0, 2000000.0, 2000000.0, 2000000.0, 2000000.0])
        z_vals = np.array([4000000.0, 4000000.0, 4000000.0, 4000000.0, 4000000.0])
        position = xr.DataArray(
            np.column_stack([x_vals, y_vals, z_vals]),
            dims=["azimuth_time", "axis"],
            coords={"azimuth_time": utc, "axis": ["x", "y", "z"]},
        )
        velocity = xr.DataArray(
            np.column_stack([np.zeros_like(x_vals), np.zeros_like(x_vals), np.zeros_like(x_vals)]),
            dims=["azimuth_time", "axis"],
            coords={"azimuth_time": utc, "axis": ["x", "y", "z"]},
        )
        orbit = xr.Dataset(
            data_vars={"position": position, "velocity": velocity},
            coords={"orbit_file": ["test.orbit"]},
            attrs={"type": "POEORB"},
        )
        t0 = np.datetime64("2020-01-01T00:00:00", "ns")
        reference_time = np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")

        result = refit_orbit_coefficients(
            burst_id="test",
            acquisition_id="2020-01-01",
            orbit=orbit,
            t0=t0,
            reference_time=reference_time,
            poly_degree=2,
        )

        assert result is not None
        assert "coeffs" in result
        assert result.coeffs.dims == ("acquisition_id", "degree", "axis")
        assert result.coeffs.shape[1] == 3  # degree 0, 1, 2
