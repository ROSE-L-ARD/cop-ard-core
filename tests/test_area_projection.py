"""Unit tests for cop_ard_core.area_projection."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.area_projection import (
    build_sparse_array,
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
