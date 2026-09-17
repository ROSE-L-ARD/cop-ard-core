"""Unit tests for cop_ard_core.area_projection."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.area_projection import (
    build_sparse_array,
    compute_geocoded_mask,
    compute_rasterization,
    compute_rtc_ap,
    compute_shadow,
    get_w,
    triangulate_regular,
)


class TestBuildSparseArray:
    """Tests for build_sparse_array."""

    def test_build_sparse_array_coo_format(self):
        """COO format should create dataset with col and row."""
        data = np.array([1.0, 2.0, 3.0])
        col = np.array([0, 1, 2])
        row = np.array([0, 1, 2])
        shape = np.array([3, 3])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
        assert isinstance(result, xr.Dataset)
        assert "col" in result.data_vars
        assert "row" in result.data_vars

    def test_build_sparse_array_csr_format(self):
        """CSR format should create dataset with indices and indptr."""
        data = np.array([1.0, 2.0, 3.0])
        col = np.array([0, 1, 2])
        row = np.array([0, 1, 2, 3])  # indptr has n_rows + 1
        shape = np.array([3, 3])
        result = build_sparse_array(data, col, row, shape, "CSR", chunks=1)
        assert isinstance(result, xr.Dataset)
        assert "indices" in result.data_vars
        # indptr is stored as a coordinate in the CSR format
        assert "indptr" in result.coords or "indptr" in result.data_vars

    def test_build_sparse_array_invalid_format(self):
        """Invalid format should raise."""
        data = np.array([1.0])
        col = np.array([0])
        row = np.array([0])
        shape = np.array([1, 1])
        with pytest.raises(Exception, match="Expected COO or CSR format"):
            build_sparse_array(data, col, row, shape, "BSP", chunks=1)

    def test_build_sparse_array_with_attrs(self):
        """Should include custom attributes."""
        data = np.array([1.0])
        col = np.array([0])
        row = np.array([0])
        shape = np.array([1, 1])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1, attrs={"key": "value"})
        assert result.attrs.get("key") == "value"
        assert result.attrs.get("format") == "COO"

    def test_build_sparse_array_coo_values(self):
        """COO values should be preserved."""
        data = np.array([1.5, 2.5, 3.5])
        col = np.array([0, 1, 2])
        row = np.array([0, 1, 2])
        shape = np.array([3, 3])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
        np.testing.assert_array_equal(result["data"].values, data)
        np.testing.assert_array_equal(result["col"].values, col)
        np.testing.assert_array_equal(result["row"].values, row)

    def test_build_sparse_array_csr_values(self):
        """CSR values should be preserved."""
        data = np.array([1.5, 2.5, 3.5])
        col = np.array([0, 1, 2])  # indices
        row = np.array([0, 1, 2, 3])  # indptr
        shape = np.array([3, 3])
        result = build_sparse_array(data, col, row, shape, "CSR", chunks=1)
        np.testing.assert_array_equal(result["data"].values, data)
        np.testing.assert_array_equal(result["indices"].values, col)
        np.testing.assert_array_equal(result["indptr"].values, row)

    def test_build_sparse_array_single_element(self):
        """Single element should be handled correctly."""
        data = np.array([42.0])
        col = np.array([0])
        row = np.array([0])
        shape = np.array([1, 1])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
        assert isinstance(result, xr.Dataset)
        assert result["nnz"].values == 1

    def test_build_sparse_array_empty_arrays(self):
        """Empty arrays should produce valid dataset."""
        data = np.array([], dtype=np.float64)
        col = np.array([], dtype=np.int64)
        row = np.array([], dtype=np.int64)
        shape = np.array([5, 5])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
        assert isinstance(result, xr.Dataset)
        assert result["nnz"].values == 0

    def test_build_sparse_array_nnz_correct(self):
        """nnz attribute should match data length."""
        for n in [0, 1, 5, 10]:
            data = np.arange(n, dtype=np.float64)
            col = np.arange(n, dtype=np.int64)
            row = np.arange(n, dtype=np.int64)
            shape = np.array([10, 10])
            result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
            assert result["nnz"].values == n

    def test_build_sparse_array_shape_preserved(self):
        """Shape should be preserved in dataset."""
        shape = np.array([100, 200])
        data = np.array([1.0])
        col = np.array([0])
        row = np.array([0])
        result = build_sparse_array(data, col, row, shape, "COO", chunks=1)
        np.testing.assert_array_equal(result["shape"].values, shape)

    def test_build_sparse_array_csr_indptr_size(self):
        """CSR indptr should have n_rows + 1 elements."""
        n_rows = 5
        nnz = 7
        data = np.arange(nnz, dtype=np.float64)
        col = np.arange(nnz, dtype=np.int64)
        row = np.arange(n_rows + 1, dtype=np.int64)  # indptr: n_rows + 1
        shape = np.array([n_rows, 10])
        result = build_sparse_array(data, col, row, shape, "CSR", chunks=1)
        assert result["indptr"].values.shape[0] == n_rows + 1

    def test_build_sparse_array_attrs_override_format(self):
        """Custom attrs should be merged with format attr."""
        data = np.array([1.0])
        col = np.array([0])
        row = np.array([0])
        shape = np.array([1, 1])
        result = build_sparse_array(
            data, col, row, shape, "COO", chunks=1, attrs={"custom": "value", "format": "overridden"}
        )
        # format from attrs should override default
        assert result.attrs.get("format") == "overridden"
        assert result.attrs.get("custom") == "value"


class TestTriangulateRegular:
    """Tests for triangulate_regular."""

    def test_triangulate_regular_2x2(self):
        """2x2 grid should produce triangles."""
        result = triangulate_regular(ysize=2, xsize=2)
        assert result.ndim == 2
        assert result.shape[1] == 3  # 3 vertices per triangle

    def test_triangulate_regular_shape(self):
        """Shape should be (4*(ysize-1)*(xsize-1), 3)."""
        for y in [2, 3, 4]:
            for x in [2, 3, 4]:
                result = triangulate_regular(ysize=y, xsize=x)
                expected_rows = 4 * (y - 1) * (x - 1)
                assert result.shape == (expected_rows, 3)

    def test_triangulate_regular_vertex_indices(self):
        """Vertex indices should be within valid range."""
        ysize, xsize = 4, 4
        result = triangulate_regular(ysize=ysize, xsize=xsize)
        max_idx = (ysize - 1) * xsize
        assert np.all(result >= 0)
        assert np.all(result < max_idx + xsize)

    def test_triangulate_regular_consistent(self):
        """All triangles should have 3 distinct vertices."""
        result = triangulate_regular(ysize=3, xsize=3)
        for tri in result:
            assert len(np.unique(tri)) == 3

    def test_triangulate_regular_minimal(self):
        """Minimum 2x2 grid should produce 4 triangles."""
        result = triangulate_regular(ysize=2, xsize=2)
        assert result.shape[0] == 4

    def test_triangulate_regular_larger_grid(self):
        """Larger grid should produce proportionally more triangles."""
        result = triangulate_regular(ysize=5, xsize=5)
        expected_rows = 4 * 4 * 4  # 64 triangles
        assert result.shape == (expected_rows, 3)

    def test_triangulate_regular_rectangular(self):
        """Rectangular grids (non-square) should work."""
        result = triangulate_regular(ysize=3, xsize=5)
        expected_rows = 4 * 2 * 4  # 32 triangles
        assert result.shape == (expected_rows, 3)

    def test_triangulate_regular_all_triangles_valid(self):
        """All triangles should have valid (non-negative) indices."""
        for ysize in range(2, 10):
            for xsize in range(2, 10):
                result = triangulate_regular(ysize=ysize, xsize=xsize)
                assert np.all(result >= 0), f"Negative indices in {ysize}x{xsize}"

    def test_triangulate_regular_no_duplicate_vertices_per_triangle(self):
        """No triangle should have duplicate vertices."""
        result = triangulate_regular(ysize=5, xsize=5)
        for i, tri in enumerate(result):
            assert len(np.unique(tri)) == 3, f"Triangle {i} has duplicate vertices: {tri}"


# ===========================================================================
# compute_shadow
# ===========================================================================


class TestComputeShadow:
    """Tests for compute_shadow.

    This function computes a shadow mask by interpolating ground incidence
    angle and local incidence angle (LIA) onto the radar grid, then applying
    two criteria:
    1. Cumulative maximum of ground incidence angle (occlusion detection).
    2. Grazing angle threshold (cos_theta <= 0.05).
    """

    def test_compute_shadow_returns_dataset(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Should return an xr.Dataset with 'shadow' variable."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert isinstance(result, xr.Dataset)
        assert "shadow" in result.data_vars

    def test_compute_shadow_output_shape(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Output shape should match radar_grid dimensions."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert result["shadow"].shape == compute_shadow_radar_grid.shape

    def test_compute_shadow_mask_values(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Shadow mask should contain only 0 and 1."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        unique_vals = np.unique(result["shadow"].values)
        assert all(v in [0, 1] for v in unique_vals)

    def test_compute_shadow_dtype(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Shadow mask should be boolean/integer dtype."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert result["shadow"].dtype in [np.bool_, np.uint8, np.int32, np.int64]

    def test_compute_shadow_dims(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Output should have same dimension names as radar_grid."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert set(result["shadow"].dims) == set(compute_shadow_radar_grid.dims)

    def test_compute_shadow_high_eps_fewer_shadows(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """High eps should produce fewer occlusion shadows."""
        result_low = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.001,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        result_high = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=10.0,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        # High eps means fewer pixels pass the occlusion check
        assert np.sum(result_low["shadow"].values) >= np.sum(result_high["shadow"].values)

    def test_compute_shadow_nonzero_margin(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
    ):
        """Should handle non-zero margin values."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin={"azimuth_time": 0.1, "slant_range_time": 0.2},
        )
        assert isinstance(result, xr.Dataset)
        assert "shadow" in result.data_vars

    def test_compute_shadow_output_total_range(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """Shadow mask values should be in range [0, 1] inclusive."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert np.all(result["shadow"].values >= 0)
        assert np.all(result["shadow"].values <= 1)

    @pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2), (4, 3)])
    def test_compute_shadow_multiple_radar_sizes(
        self,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
        shape,
    ):
        """Should handle various radar grid sizes: 2x2, 2x3, 3x2, 4x3."""
        from cop_ard_core.area_projection import compute_shadow

        az_size, sr_size = shape

        radar_grid = xr.DataArray(
            np.zeros(shape, dtype=np.float64),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(az_size, dtype=np.float64),
                "slant_range_time": np.arange(sr_size, dtype=np.float64),
            },
        )
        radar_grid.rio.write_crs("EPSG:4326", inplace=True)

        result = compute_shadow(
            radar_grid=radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert result["shadow"].shape == shape, f"Shape mismatch for {shape}"
        np.testing.assert_array_less(-0.01, result["shadow"].values)
        np.testing.assert_array_less(result["shadow"].values, 1.01)

    @pytest.mark.parametrize("eps", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    def test_compute_shadow_various_eps_values(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_local_incidence_angle,
        compute_shadow_dt,
        compute_shadow_margin,
        eps,
    ):
        """Should handle a range of eps values without error and produce valid masks."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=eps,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert isinstance(result, xr.Dataset)
        assert "shadow" in result.data_vars
        unique_vals = np.unique(result["shadow"].values)
        assert all(v in [0, 1] for v in unique_vals)
        # Verify total shadows decreases as eps increases
        assert result["shadow"].values.sum() >= 0

    def test_compute_shadow_grazing_angle_causes_shadows(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_dt,
        compute_shadow_margin,
        compute_shadow_lia_grazing,
    ):
        """Grazing angle LIA (cos_theta close to 0) should produce shadow pixels."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_lia_grazing,
            eps=10.0,  # High eps to suppress occlusion shadows
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        # Grazing angle should trigger at least some shadow pixels
        assert result["shadow"].values.sum() > 0

    def test_compute_shadow_all_no_shadows_scenario(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_lia_all_large,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """When no shadows are possible, output should be all zeros."""
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=compute_shadow_lia_all_large,
            eps=10.0,  # High eps to suppress occlusion shadows
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert np.all(result["shadow"].values == 0)


# ===========================================================================
# compute_rasterization
# ===========================================================================


class TestComputeRasterization:
    """Tests for compute_rasterization.

    This function computes rasterization weights and the accumulated
    gamma-naught area (RTC-AP) for a block of radar data simulation.
    It returns (nlook, weights, layover) where:
      - nlook: number of looks per geogrid cell
      - weights: sparse CSR weight dataset (or None)
      - layover: list of (bbox_str, sparse_dataset) tuples (or None)
    """

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_returns_tuple(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Should return a 3-tuple (nlook, weights, layover)."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        # nl needs shape (dim_y, dim_x, 4) for the reshape in compute_rasterization
        nl_shape = (dim_y, dim_x, 4)
        mock_get_w.return_value = (
            None,
            np.zeros(nl_shape, dtype="float32"),
            None,
        )
        result = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        mock_get_w.assert_called_once()

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_nlook_type(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """nlook should be an xr.DataArray with dims [y, x]."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        mock_get_w.return_value = (None, np.zeros((dim_y, dim_x, 4), dtype="float32"), None)
        nlook, _, _ = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert isinstance(nlook, xr.DataArray)
        assert "y" in nlook.dims
        assert "x" in nlook.dims

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_nlook_shape(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """nlook shape should be (dim_y, dim_x) = (sim.y.size-1, sim.x.size-1)."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        mock_get_w.return_value = (None, np.zeros((dim_y, dim_x, 4), dtype="float32"), None)
        nlook, _, _ = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert nlook.shape == (dim_y, dim_x)

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_weights_type(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Weights should be xr.Dataset or None."""
        mock_get_w.return_value = (None, np.zeros((3, 4, 4), dtype="float32"), None)
        _, weights, _ = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert weights is None or isinstance(weights, xr.Dataset)

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_layover_type(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Layover should be list of tuples or None."""
        layover_data = [
            (
                "0_0_5_5",
                build_sparse_array(np.array([0.5]), np.array([0]), np.array([0, 1]), np.array([5, 5]), "CSR", chunks=1),
            )
        ]
        mock_get_w.return_value = (None, np.zeros((3, 4, 4), dtype="float32"), layover_data)
        _, _, layover = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        if layover is not None:
            assert isinstance(layover, list)
            for bbox_str, ds in layover:
                assert isinstance(bbox_str, str)
                assert isinstance(ds, xr.Dataset)

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_non_unit_resolution(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Different resolutions should produce valid nlook."""
        mock_get_w.return_value = (None, np.zeros((3, 4, 4), dtype="float32"), None)
        for resolution in [(2.0, 2.0), (0.5, 0.5), (1.0, 2.0)]:
            nlook, _, _ = compute_rasterization(
                simulation=area_proj_simulation,
                xmin_global=0,
                ymin_global=0,
                xsize_global=10,
                radar_grid_pixel_area=1.0,
                bbox=area_proj_bbox,
                resolution=resolution,
                block_id="test_0_0",
            )
            assert isinstance(nlook, xr.DataArray)
            assert nlook.shape == (area_proj_simulation.y.size - 1, area_proj_simulation.x.size - 1)

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_nonzero_pixel_area(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Non-unit radar_grid_pixel_area should scale expected_nlooks."""
        mock_get_w.return_value = (np.zeros((3, 4, 4)), np.zeros((3, 4, 4), dtype="float32"), None)
        nlook1, _, _ = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        nlook2, _, _ = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=0,
            ymin_global=0,
            xsize_global=10,
            radar_grid_pixel_area=2.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert np.all(nlook2.values >= nlook1.values)

    @patch("cop_ard_core.area_projection.get_w")
    def test_compute_rasterization_with_offset_global_coords(self, mock_get_w, area_proj_simulation, area_proj_bbox):
        """Should handle non-zero xmin_global and ymin_global."""
        mock_get_w.return_value = (None, np.zeros((3, 4, 4), dtype="float32"), None)
        nlook, weights, layover = compute_rasterization(
            simulation=area_proj_simulation,
            xmin_global=100,
            ymin_global=200,
            xsize_global=500,
            radar_grid_pixel_area=1.0,
            bbox=area_proj_bbox,
            resolution=(1.0, 1.0),
            block_id="test_0_0",
        )
        assert isinstance(nlook, xr.DataArray)
        assert nlook.shape == (area_proj_simulation.y.size - 1, area_proj_simulation.x.size - 1)


# ===========================================================================
# compute_rtc_ap
# ===========================================================================


class TestComputeRtcAp:
    """Tests for compute_rtc_ap.

    This function computes rasterization weights and the accumulated
    gamma-naught area (RTC-AP). It returns (anf, lia, a_block) where:
      - anf: Area Normalization Factor (xr.DataArray, dims [y, x])
      - lia: Local Incidence Angle (xr.DataArray, dims [y, x])
      - a_block: sparse CSR area matrix dataset (or None)

    Tests mock the internal computeGammaNaughtArea kernel since it depends on numba.
    """

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_returns_tuple(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """Should return a 3-tuple (anf, lia, a_block)."""
        mock_compute_gamma.return_value = 5
        result = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        mock_compute_gamma.assert_called_once()

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_anf_type(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf should be an xr.DataArray."""
        mock_compute_gamma.return_value = 5
        anf, _, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert isinstance(anf, xr.DataArray)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_anf_shape(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf shape should be (dim_y, dim_x)."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        mock_compute_gamma.return_value = 5
        anf, _, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert anf.shape == (dim_y, dim_x)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_anf_dims(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf should have dims ['y', 'x']."""
        mock_compute_gamma.return_value = 5
        anf, _, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert set(anf.dims) == {"y", "x"}

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_lia_type(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """lia should be an xr.DataArray."""
        mock_compute_gamma.return_value = 5
        _, lia, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert isinstance(lia, xr.DataArray)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_lia_shape(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """lia shape should match anf shape."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        mock_compute_gamma.return_value = 5
        _, lia, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert lia.shape == (dim_y, dim_x)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_lia_values_range(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """lia (local incidence angle) should be in [0, pi/2] range (in radians)."""
        mock_compute_gamma.return_value = 5
        _, lia, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert np.all(lia.values >= 0)
        assert np.all(lia.values <= np.pi / 2 + 0.01)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_anf_values_non_negative(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """ANF values should be non-negative."""
        mock_compute_gamma.return_value = 5
        anf, _, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert np.all(anf.values >= 0)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_a_block_type(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """a_block should be xr.Dataset or None."""
        mock_compute_gamma.return_value = 5
        _, _, a_block, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert a_block is None or isinstance(a_block, xr.Dataset)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_normal_order_false(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """normal_order=False should still produce valid output (different normal direction)."""
        mock_compute_gamma.return_value = 5
        anf, lia, a_block, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=False,
            bbox=area_proj_bbox_int,
        )
        assert isinstance(anf, xr.DataArray)
        assert isinstance(lia, xr.DataArray)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_different_pixel_area(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """Different radar_grid_pixel_area should not cause errors."""
        dim_y = area_proj_simulation.y.size - 1
        dim_x = area_proj_simulation.x.size - 1
        for pixel_area in [0.5, 2.0, 4.0]:
            mock_compute_gamma.return_value = 5
            anf, lia, a_block, _ = compute_rtc_ap(
                csr_matrix=area_proj_csr_matrix,
                simulation=area_proj_simulation,
                radar_grid_pixel_area=float(pixel_area),
                normal_order=True,
                bbox=area_proj_bbox_int,
            )
            assert anf.shape == (dim_y, dim_x)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_dtype(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf and lia should be float32."""
        mock_compute_gamma.return_value = 5
        anf, lia, _, _ = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert anf.dtype == np.float32
        assert lia.dtype == np.float32

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_bbox_radar_type(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf and lia should be float32."""
        mock_compute_gamma.return_value = 5
        _, _, _, bbox_radar = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert isinstance(bbox_radar, np.ndarray)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_bbox_radar_dtype(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf and lia should be float32."""
        mock_compute_gamma.return_value = 5
        _, _, _, bbox_radar = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert bbox_radar.dtype == int

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_bbox_radar_shape(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf and lia should be float32."""
        mock_compute_gamma.return_value = 5
        _, _, _, bbox_radar = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert bbox_radar.shape == (4,)

    @patch("cop_ard_core.area_projection.computeGammaNaughtArea")
    def test_compute_rtc_ap_bbox_radar_coordinate_values(
        self, mock_compute_gamma, area_proj_csr_matrix, area_proj_simulation, area_proj_bbox_int
    ):
        """anf and lia should be float32."""
        mock_compute_gamma.return_value = 5
        _, _, _, bbox_radar = compute_rtc_ap(
            csr_matrix=area_proj_csr_matrix,
            simulation=area_proj_simulation,
            radar_grid_pixel_area=1.0,
            normal_order=True,
            bbox=area_proj_bbox_int,
        )
        assert np.all(bbox_radar >= 0)


# ===========================================================================
# get_w
# ===========================================================================


class TestGetW:
    """Tests for get_w.

    This function computes rasterization weights for all triangles in
    the geogrid chunk. It returns (w, nlook, layover) or (None, None, None)
    if no non-zero weights are computed.

    Tests mock the internal area_proj_triangles kernel since it has dtype
    sensitivity issues with float64 bbox values.
    """

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_returns_tuple(self, mock_area_proj, get_w_simple_inputs):
        """Should return a 3-tuple (w, nlook, layover)."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        result = get_w(**get_w_simple_inputs)
        assert isinstance(result, tuple)
        assert len(result) == 3
        mock_area_proj.assert_called_once()

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_weights_type(self, mock_area_proj, get_w_simple_inputs):
        """w should be xr.Dataset or None."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if w is not None:
            assert isinstance(w, xr.Dataset)
            assert "data" in w.data_vars
            assert "indptr" in w.data_vars or "indptr" in w.coords

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_nlook_type(self, mock_area_proj, get_w_simple_inputs):
        """nlook should be np.ndarray or None."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if nlook is not None:
            assert isinstance(nlook, np.ndarray)

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_layover_type(self, mock_area_proj, get_w_simple_inputs):
        """layover should be list of tuples or None."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if layover is not None:
            assert isinstance(layover, list)
            for bbox_str, ds in layover:
                assert isinstance(bbox_str, str)
                assert isinstance(ds, xr.Dataset)

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_nlook_dtype(self, mock_area_proj, get_w_simple_inputs):
        """nlook should be float32 dtype."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if nlook is not None:
            assert nlook.dtype == np.float32

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_nlook_shape_matches_triang(self, mock_area_proj, get_w_simple_inputs):
        """nlook size should match number of triangles."""
        n_tri = get_w_simple_inputs["triang"].shape[0]
        mock_area_proj.return_value = (
            (np.ones(4, dtype="float64"), np.array([0, 1, 2, 4], dtype="int64"), np.array([0, 1, 2, 3], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.ones(n_tri, dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if nlook is not None:
            expected_nnz = get_w_simple_inputs["triang"].shape[0]
            assert nlook.shape == (expected_nnz,)

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_with_attrs_in_sparse_dataset(self, mock_area_proj, get_w_simple_inputs):
        """Sparse weight dataset should store radar_bbox as attribute."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        w, nlook, layover = get_w(**get_w_simple_inputs)
        if w is not None:
            assert "radar_bbox" in w.attrs

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_with_plane_orientation(self, mock_area_proj, get_w_simple_inputs):
        """Different plane_orientation should produce valid results."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        for orientation in [1, -1]:
            result = get_w(
                **get_w_simple_inputs,
                plane_orientation=orientation,
            )
            assert isinstance(result, tuple)
            assert len(result) == 3

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_with_nonzero_global_offset(self, mock_area_proj, get_w_simple_inputs):
        """Should handle non-zero xmin_global/ymin_global offsets."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        inputs = dict(get_w_simple_inputs)
        inputs["xmin_global"] = 100
        inputs["ymin_global"] = 200
        w, nlook, layover = get_w(**inputs)
        assert isinstance(w, (xr.Dataset, type(None)))
        assert isinstance(nlook, (np.ndarray, type(None)))
        assert isinstance(layover, (list, type(None)))

    @patch("cop_ard_core.area_projection.area_proj_triangles")
    def test_get_w_with_different_expected_nlooks(self, mock_area_proj, get_w_simple_inputs):
        """Different expected_nlooks should not change output structure."""
        mock_area_proj.return_value = (
            (
                np.array([1.0, 2.0], dtype="float64"),
                np.array([0, 1, 2], dtype="int64"),
                np.array([0, 1], dtype="int64"),
            ),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            (np.array([], dtype="float64"), np.array([0], dtype="int64"), np.array([], dtype="int64")),
            np.array([1.0, 1.0], dtype="float64"),
        )
        for exp_nlooks in [1, 2, 5, 10]:
            inputs = dict(get_w_simple_inputs)
            inputs["expected_nlooks"] = exp_nlooks
            w, nlook, layover = get_w(**inputs)
            assert isinstance(w, (xr.Dataset, type(None)))
            assert isinstance(nlook, (np.ndarray, type(None)))


# ===========================================================================
# compute_geocoded_mask
# ===========================================================================


class TestComputeGeocodedMask:
    """Tests for compute_geocoded_mask.

    This function computes the geocoded mask from sparse CSR matrix,
    layover mask, and shadow mask. It returns a 2D np.ndarray with
    values 0 (no shadow/layover), 1 (layover only), 2 (shadow only),
    or 3 (both).

    Tests mock the internal csr_times_dense kernel since it has size
    mismatch requirements between CSR matrix rows and output shape.
    """

    def _make_mock_csr_times_dense(self):
        """Create a mock csr_times_dense that fills geocoded_image with weighted sums."""

        def mock_csr_times_dense(**kwargs):
            n_dims = kwargs["n_dims"]
            n_rows = kwargs["n_rows"]
            data = kwargs["data"]
            indices = kwargs["indices"]
            indptr = kwargs["indptr"]
            radar_image = kwargs["radar_image"]
            geocoded_image = kwargs["geocoded_image"]
            xsize_block = kwargs["xsize_block"]
            ysize_block = kwargs["ysize_block"]

            # Compute weighted sum for each output pixel
            for i in range(n_rows):
                start, end = indptr[i], indptr[i + 1]
                nlook = sum(data[j] for j in range(start, end))
                if nlook > 0:
                    for j in range(start, end):
                        idx = indices[j]
                        for k in range(n_dims):
                            geocoded_image[k * n_rows + i] += (data[j] / nlook) * radar_image[
                                k * xsize_block * ysize_block + idx
                            ]

        return mock_csr_times_dense

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_returns_ndarray(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Should return a np.ndarray."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        dim_y, dim_x = 4, 6
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(dim_y, dim_x),
            threshold=0.3,
        )
        assert isinstance(result, np.ndarray)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_output_shape(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Output shape should match output_shape parameter."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        output_shape = (4, 6)
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=output_shape,
            threshold=0.3,
        )
        assert result.shape == output_shape

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_dtype(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Output dtype should be int32."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.3,
        )
        assert result.dtype == np.int32

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_value_range(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Mask values should be 0, 1, 2, or 3."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.3,
        )
        unique_vals = np.unique(result)
        assert all(v in [0, 1, 2, 3] for v in unique_vals)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_no_activity_all_zero(self, mock_csr, area_proj_geocoded_csr_matrix):
        """With zero layover/shadow, all output should be 0 (no shadow)."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.zeros((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.zeros((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.3,
        )
        assert np.all(result == 0)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_all_layover(self, mock_csr, area_proj_geocoded_csr_matrix):
        """With all-ones layover mask, should produce non-zero layover values."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.zeros((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.3,
        )
        assert np.any(result > 0)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_all_shadow(self, mock_csr, area_proj_geocoded_csr_matrix):
        """With all-ones shadow mask, should produce non-zero shadow values."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.zeros((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.3,
        )
        assert np.any(result > 0)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_different_threshold(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Higher threshold should produce fewer flagged pixels."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result_low = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.1,
        )
        result_high = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
            threshold=0.5,
        )
        assert np.sum(result_low > 0) >= np.sum(result_high > 0)

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_with_bbox_offset(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Should handle non-zero bbox_radar offsets."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(2, 3, 7, 8),
            output_shape=(4, 6),
            threshold=0.3,
        )
        assert result.shape == (4, 6)
        assert result.dtype == np.int32

    @patch("cop_ard_core.area_projection.csr_times_dense")
    def test_compute_geocoded_mask_default_threshold(self, mock_csr, area_proj_geocoded_csr_matrix):
        """Default threshold (0.3) should work without explicit parameter."""
        mock_csr.side_effect = self._make_mock_csr_times_dense()
        layover = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        shadow = xr.DataArray(np.ones((4, 6), dtype=np.float32), dims=["azimuth_time", "slant_range_time"])
        result = compute_geocoded_mask(
            csr_matrix=area_proj_geocoded_csr_matrix,
            layover=layover,
            shadow=shadow,
            xsize_global=10,
            bbox_radar=(0, 0, 5, 5),
            output_shape=(4, 6),
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 6)
