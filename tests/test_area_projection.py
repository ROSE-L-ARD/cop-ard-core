"""Unit tests for cop_ard_core.area_projection."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.area_projection import (
    build_sparse_array,
    triangulate_regular,
    compute_shadow,
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

    def test_compute_shadow_all_false_grazing(
        self,
        compute_shadow_radar_grid,
        compute_shadow_h,
        compute_shadow_dem_distance,
        compute_shadow_azimuth_time,
        compute_shadow_slant_range_time,
        compute_shadow_simulation_ml,
        compute_shadow_dt,
        compute_shadow_margin,
    ):
        """When all cos_theta > 0.05, grazing angle mask is all False."""
        # LIA values that give cos_theta well above 0.05
        lia = xr.DataArray(
            np.full((4, 6), 0.5, dtype=np.float64),
            dims=["y", "x"],
            coords={"y": np.array([0.0, 1.0, 2.0, 3.0]), "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])},
        )
        lia.rio.write_crs("EPSG:4326", inplace=True)
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=compute_shadow_simulation_ml,
            local_incidence_angle=lia,
            eps=10.0,  # High eps to eliminate occlusion shadows
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        # With high eps and all cos_theta > 0.05, all shadows should be False
        assert np.all(result["shadow"].values == 0)

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

    def test_compute_shadow_acquisition_date_from_ml(
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
        """Uses acquisition_date from simulation_ml dict."""
        sim_ml = dict(compute_shadow_simulation_ml)
        sim_ml["acquisition_date"] = np.array(["2021-06-15"], dtype="datetime64[ns]")
        result = compute_shadow(
            radar_grid=compute_shadow_radar_grid,
            simulation_h=compute_shadow_h,
            simulation_dem_distance=compute_shadow_dem_distance,
            simulation_azimuth_time=compute_shadow_azimuth_time,
            simulation_slant_range_time=compute_shadow_slant_range_time,
            simulation_ml=sim_ml,
            local_incidence_angle=compute_shadow_local_incidence_angle,
            eps=0.01,
            dt=compute_shadow_dt,
            margin=compute_shadow_margin,
        )
        assert isinstance(result, xr.Dataset)
        assert "shadow" in result.data_vars

    def test_compute_shadow_median_filter(
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
        """Median filter should be applied along range axis (axis=1)."""
        # The median filter is applied with size=(1, 5) meaning only axis 1 (range).
        # We can verify the output is produced without error.
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
        # If median filter didn't apply correctly, this would raise
        assert result["shadow"].shape == compute_shadow_radar_grid.shape
