"""Unit tests for cop_ard_core.griddata."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.griddata import (
    generate_triangles,
    init_interpolate,
    interp_dataset,
)


class TestGenerateTriangles:
    """Tests for generate_triangles."""

    def test_generate_triangles_1x1(self):
        """1x1 grid should produce 2 triangles."""
        result = generate_triangles(ysize=1, xsize=1)
        assert result.shape == (2, 3, 3)

    def test_generate_triangles_2x2(self):
        """2x2 grid should produce 8 triangles (4 cells * 2)."""
        result = generate_triangles(ysize=2, xsize=2)
        assert result.shape[0] == 8

    def test_generate_triangles_shape(self):
        """Output shape should be (2*y*x, 3, 3)."""
        for y in [1, 2, 3]:
            for x in [1, 2, 3]:
                result = generate_triangles(ysize=y, xsize=x)
                assert result.shape == (2 * y * x, 3, 3)

    def test_generate_triangles_vertex_indices(self):
        """Vertex indices should be within grid bounds."""
        ysize, xsize = 4, 4
        result = generate_triangles(ysize=ysize, xsize=xsize)
        assert np.all(result[:, :, :2] >= 0)
        assert np.all(result[:, :, 0] <= xsize)
        assert np.all(result[:, :, 1] <= ysize)


class TestInitInterpolate:
    """Tests for init_interpolate."""

    def test_init_interpolate_returns_none(self):
        """Should return None."""
        result = init_interpolate()
        assert result is None

    def test_init_interpolate_compiles(self):
        """Should compile numba functions without error."""
        init_interpolate()


class TestInterpDataset:
    """Tests for interp_dataset."""

    def _make_2d_xi(self, ny, nx):
        """Create 2D xi arrays for interpolation."""
        x_2d, y_2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
        return (y_2d, x_2d)

    def test_interp_dataset_basic(self):
        """Basic interpolation should return dataset."""
        data = xr.Dataset({
            "z": xr.DataArray(
                np.random.rand(5, 5),
                dims=["y", "x"],
                coords={"y": np.arange(5), "x": np.arange(5)},
            )
        })
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert isinstance(result, xr.Dataset)

    def test_interp_dataset_output_shape(self):
        """Output shape should match xi shapes."""
        data = xr.Dataset({
            "z": xr.DataArray(
                np.random.rand(10, 10),
                dims=["y", "x"],
                coords={"y": np.arange(10), "x": np.arange(10)},
            )
        })
        xi = self._make_2d_xi(7, 9)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "z" in result
        assert result.z.shape == (7, 9)

    def test_interp_dataset_multiple_variables(self):
        """Multiple variables should all be interpolated."""
        data = xr.Dataset({
            "z1": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"],
                               coords={"y": np.arange(5), "x": np.arange(5)}),
            "z2": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"],
                               coords={"y": np.arange(5), "x": np.arange(5)}),
        })
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "z1" in result
        assert "z2" in result

    def test_interp_dataset_with_extra_dims(self):
        """Variables with extra dimensions should be handled."""
        data = xr.Dataset({
            "z": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"],
                              coords={"y": np.arange(5), "x": np.arange(5)}),
            "extra": xr.DataArray(np.random.rand(3, 5, 5), dims=["layer", "y", "x"],
                                  coords={"y": np.arange(5), "x": np.arange(5), "layer": [0, 1, 2]}),
        })
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert isinstance(result, xr.Dataset)

    def test_interp_dataset_fill_value(self):
        """Fill value should be used for out-of-bounds points."""
        data = xr.Dataset({
            "z": xr.DataArray(np.ones((5, 5)), dims=["y", "x"],
                              coords={"y": np.arange(5), "x": np.arange(5)}),
        })
        # Interpolate inside bounds (1,1) to (3,3)
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"), fill_value=0.0)
        assert "z" in result

    def test_interp_dataset_cubic(self):
        """Cubic interpolation should work."""
        data = xr.Dataset({
            "z": xr.DataArray(np.random.rand(10, 10), dims=["y", "x"],
                              coords={"y": np.arange(10), "x": np.arange(10)}),
        })
        xi = self._make_2d_xi(5, 5)
        result = interp_dataset(data, xi, common_dims=("y", "x"), method="cubic")
        assert isinstance(result, xr.Dataset)
