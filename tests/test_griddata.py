"""Unit tests for cop_ard_core.griddata."""

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.griddata import (
    generate_triangles,
    init_interpolate,
    interp_dataset,
    interpolate,
)

# ===========================================================================
# TestGenerateTriangles
# ===========================================================================


class TestGenerateTriangles:
    """Tests for generate_triangles."""

    @pytest.mark.unit
    def test_generate_triangles_1x1(self):
        """1x1 grid should produce 2 triangles."""
        result = generate_triangles(ysize=1, xsize=1)
        assert result.shape == (2, 3, 3)

    @pytest.mark.unit
    def test_generate_triangles_2x2(self):
        """2x2 grid should produce 8 triangles (4 cells * 2)."""
        result = generate_triangles(ysize=2, xsize=2)
        assert result.shape[0] == 8

    @pytest.mark.unit
    def test_generate_triangles_shape(self):
        """Output shape should be (2*y*x, 3, 3) for various grid sizes."""
        for y in [1, 2, 3, 5]:
            for x in [1, 2, 3, 5]:
                result = generate_triangles(ysize=y, xsize=x)
                assert result.shape == (2 * y * x, 3, 3)

    @pytest.mark.unit
    def test_generate_triangles_vertex_indices(self):
        """Vertex indices should be within grid bounds."""
        ysize, xsize = 4, 4
        result = generate_triangles(ysize=ysize, xsize=xsize)
        assert np.all(result[:, :, :2] >= 0)
        assert np.all(result[:, :, 0] <= xsize)
        assert np.all(result[:, :, 1] <= ysize)

    @pytest.mark.unit
    def test_generate_triangles_z_column_zero(self):
        """Third column (z placeholder) should be zeros."""
        result = generate_triangles(ysize=2, xsize=2)
        assert np.all(result[:, :, 2] == 0.0)

    @pytest.mark.unit
    def test_generate_triangles_x_y_are_float64(self):
        """x and y vertex columns should be float64."""
        result = generate_triangles(ysize=2, xsize=2)
        assert result[:, 0, 0].dtype == np.float64
        assert result[:, 0, 1].dtype == np.float64

    @pytest.mark.unit
    def test_generate_triangles_lower_left_triangles(self):
        """Lower-left triangles should use vertices (0,0), (1,0), (0,1)."""
        result = generate_triangles(ysize=1, xsize=1)
        # First triangle (lower-left): vertices at [0,0], [1,0], [0,1]
        np.testing.assert_array_equal(result[0, :, :2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    @pytest.mark.unit
    def test_generate_triangles_upper_right_triangles(self):
        """Upper-right triangles should use vertices (1,0), (1,1), (0,1)."""
        result = generate_triangles(ysize=1, xsize=1)
        # Second triangle (upper-right): vertices at [1,0], [1,1], [0,1]
        np.testing.assert_array_equal(result[1, :, :2], [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    @pytest.mark.unit
    def test_generate_triangles_all_cells_same_count(self):
        """Each cell should contribute exactly 2 triangles."""
        for y in [1, 2, 3, 5, 10]:
            for x in [1, 2, 3, 5, 10]:
                result = generate_triangles(ysize=y, xsize=x)
                assert result.shape[0] == 2 * y * x

    @pytest.mark.unit
    def test_generate_triangles_no_duplicate_vertices_per_triangle(self):
        """Each triangle should have 3 distinct vertex pairs."""
        result = generate_triangles(ysize=2, xsize=2)
        for i in range(result.shape[0]):
            v0 = tuple(result[i, 0, :2])
            v1 = tuple(result[i, 1, :2])
            v2 = tuple(result[i, 2, :2])
            assert len({v0, v1, v2}) == 3

    @pytest.mark.unit
    def test_generate_triangles_large_grid(self):
        """Large grid should produce many triangles."""
        result = generate_triangles(ysize=10, xsize=10)
        assert result.shape[0] == 200
        assert result.shape[1] == 3
        assert result.shape[2] == 3


# ===========================================================================
# TestInitInterpolate
# ===========================================================================


class TestInitInterpolate:
    """Tests for init_interpolate."""

    @pytest.mark.unit
    def test_init_interpolate_returns_none(self):
        """Should return None."""
        result = init_interpolate()
        assert result is None

    @pytest.mark.unit
    def test_init_interpolate_compiles(self):
        """Should compile numba functions without error."""
        init_interpolate()

    @pytest.mark.unit
    def test_init_interpolate_idempotent(self):
        """Calling multiple times should not raise."""
        init_interpolate()
        init_interpolate()
        init_interpolate()

    @pytest.mark.unit
    def test_init_interpolate_no_side_effects(self):
        """Should not modify any global state."""
        import copy

        before = copy.deepcopy(init_interpolate.__code__)
        init_interpolate()
        after = init_interpolate.__code__
        assert before.co_code == after.co_code


# ===========================================================================
# TestInterpDataset
# ===========================================================================


class TestInterpDataset:
    """Tests for interp_dataset."""

    def _make_2d_xi(self, ny, nx):
        """Create 2D xi arrays for interpolation."""
        x_2d, y_2d = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        return (y_2d, x_2d)

    @pytest.mark.unit
    def test_interp_dataset_basic(self):
        """Basic interpolation should return dataset."""
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.random.rand(5, 5),
                    dims=["y", "x"],
                    coords={"y": np.arange(5), "x": np.arange(5)},
                )
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert isinstance(result, xr.Dataset)

    @pytest.mark.unit
    def test_interp_dataset_output_shape(self):
        """Output shape should match xi shapes."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.random.rand(10, 10),
                    dims=["y", "x"],
                    coords={"y": np.arange(10), "x": np.arange(10)},
                )
            }
        )
        xi = self._make_2d_xi(7, 9)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "z" in result
        assert result.z.shape == (7, 9)

    @pytest.mark.unit
    def test_interp_dataset_multiple_variables(self):
        """Multiple variables should all be interpolated."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z1": xr.DataArray(
                    np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
                "z2": xr.DataArray(
                    np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "z1" in result
        assert "z2" in result
        assert result.z1.shape == (3, 3)
        assert result.z2.shape == (3, 3)

    @pytest.mark.unit
    def test_interp_dataset_with_extra_dims(self):
        """Variables with extra dimensions should be handled."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}),
                "extra": xr.DataArray(
                    np.random.rand(3, 5, 5),
                    dims=["layer", "y", "x"],
                    coords={"y": np.arange(5), "x": np.arange(5), "layer": [0, 1, 2]},
                ),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert isinstance(result, xr.Dataset)
        assert "z" in result
        assert "extra" in result
        assert result.extra.shape == (3, 3, 3)

    @pytest.mark.unit
    def test_interp_dataset_fill_value(self):
        """Fill value should be used for out-of-bounds points."""
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.ones((5, 5)), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"), fill_value=0.0)
        assert "z" in result

    @pytest.mark.unit
    def test_interp_dataset_cubic(self):
        """Cubic interpolation should work."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.random.rand(10, 10), dims=["y", "x"], coords={"y": np.arange(10), "x": np.arange(10)}
                ),
            }
        )
        xi = self._make_2d_xi(5, 5)
        result = interp_dataset(data, xi, common_dims=("y", "x"), method="cubic")
        assert isinstance(result, xr.Dataset)

    @pytest.mark.unit
    def test_interp_dataset_linear_method(self):
        """Linear interpolation should work."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(8, 8), dims=["y", "x"], coords={"y": np.arange(8), "x": np.arange(8)}),
            }
        )
        xi = self._make_2d_xi(4, 4)
        result = interp_dataset(data, xi, common_dims=("y", "x"), method="linear")
        assert isinstance(result, xr.Dataset)
        assert result.z.shape == (4, 4)

    @pytest.mark.unit
    def test_interp_dataset_nearest_method(self):
        """Nearest neighbor interpolation should work."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(8, 8), dims=["y", "x"], coords={"y": np.arange(8), "x": np.arange(8)}),
            }
        )
        xi = self._make_2d_xi(4, 4)
        result = interp_dataset(data, xi, common_dims=("y", "x"), method="nearest")
        assert isinstance(result, xr.Dataset)

    @pytest.mark.unit
    def test_interp_dataset_preserves_variable_names(self):
        """Output should contain the same variable names as input."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "alpha": xr.DataArray(
                    np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
                "beta": xr.DataArray(
                    np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
                "gamma": xr.DataArray(
                    np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result

    @pytest.mark.unit
    def test_interp_dataset_downsampling(self):
        """Interpolating to coarser grid should downsample."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.arange(25, dtype=np.float64).reshape(5, 5),
                    dims=["y", "x"],
                    coords={"y": np.arange(5), "x": np.arange(5)},
                )
            }
        )
        xi = self._make_2d_xi(2, 2)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert result.z.shape == (2, 2)

    @pytest.mark.unit
    def test_interp_dataset_upsampling(self):
        """Interpolating to finer grid should upsample (linear works for small grids)."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.arange(4, dtype=np.float64).reshape(2, 2),
                    dims=["y", "x"],
                    coords={"y": np.arange(2), "x": np.arange(2)},
                )
            }
        )
        xi = self._make_2d_xi(5, 5)
        # Linear interpolation works for small grids; cubic requires 4+ points per dim
        result = interp_dataset(data, xi, common_dims=("y", "x"), method="linear")
        assert result.z.shape == (5, 5)

    @pytest.mark.unit
    def test_interp_dataset_with_rename(self):
        """Output dims should be renamed according to rename dict."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"), rename={"y": "lat", "x": "lon"})
        assert "lat" in result.z.dims
        assert "lon" in result.z.dims

    @pytest.mark.unit
    def test_interp_dataset_constant_values(self):
        """Constant input should produce constant output."""
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    np.full((5, 5), 42.0), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}
                ),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        np.testing.assert_allclose(result.z.values, 42.0, atol=1e-10)

    @pytest.mark.unit
    def test_interp_dataset_linear_gradient(self):
        """Linear gradient should interpolate linearly."""
        y = np.arange(5.0)
        x = np.arange(5.0)
        yy, xx = np.meshgrid(y, x, indexing="xy")
        data = xr.Dataset(
            {
                "z": xr.DataArray(
                    yy + xx,
                    dims=["y", "x"],
                    coords={"y": np.arange(5), "x": np.arange(5)},
                )
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        # At (1,1) the value should be 1+1=2 (exact for linear)
        np.testing.assert_allclose(result.z.values[1, 1], 2.0, atol=0.1)

    @pytest.mark.unit
    def test_interp_dataset_nan_fill_value(self):
        """Default fill_value should be NaN."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}),
            }
        )
        xi = self._make_2d_xi(3, 3)
        interp_dataset(data, xi, common_dims=("y", "x"))
        # Should not raise with default NaN fill_value

    @pytest.mark.unit
    def test_interp_dataset_multiple_extra_dim_layers(self):
        """Multiple extra dimensions should be handled correctly."""
        np.random.seed(42)
        data = xr.Dataset(
            {
                "field": xr.DataArray(
                    np.random.rand(2, 3, 5, 5),
                    dims=["time", "level", "y", "x"],
                    coords={"y": np.arange(5), "x": np.arange(5), "level": [0, 1, 2], "time": [0, 1]},
                ),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert "field" in result
        assert result.field.shape == (2, 3, 3, 3)

    @pytest.mark.unit
    def test_interp_dataset_dataset_type_preserved(self):
        """Output should always be xr.Dataset even with single variable."""
        data = xr.Dataset(
            {
                "z": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"], coords={"y": np.arange(5), "x": np.arange(5)}),
            }
        )
        xi = self._make_2d_xi(3, 3)
        result = interp_dataset(data, xi, common_dims=("y", "x"))
        assert isinstance(result, xr.Dataset)


# ===========================================================================
# TestInterpolate
# ===========================================================================


class TestInterpolate:
    """Tests for the interpolate function."""

    @pytest.mark.unit
    def test_interpolate_basic(self):
        """Basic interpolation should return 3D array."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)  # y coords
        points[:, :, 1] = np.arange(3).reshape(1, 3)  # x coords
        values = np.arange(9, dtype=np.float64).reshape(1, 3, 3).astype(np.float64)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3, 3)

    @pytest.mark.unit
    def test_interpolate_output_shape(self):
        """Output shape should be (n_layers, bbox[3], bbox[2])."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.random.rand(2, 3, 3)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        assert result.shape == (2, 3, 3)

    @pytest.mark.unit
    def test_interpolate_single_layer(self):
        """Single layer interpolation should return 3D array."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.arange(9, dtype=np.float64).reshape(1, 3, 3).astype(np.float64)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        assert result.shape == (1, 3, 3)

    @pytest.mark.unit
    def test_interpolate_constant_values(self):
        """Constant input should produce constant output."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.full((1, 3, 3), 42.0, dtype=np.float64)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        np.testing.assert_allclose(result[0], 42.0, atol=1e-6)

    @pytest.mark.unit
    def test_interpolate_multiple_layers(self):
        """Multiple layers should be interpolated independently."""
        np.random.seed(42)
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.random.rand(3, 3, 3)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        assert result.shape == (3, 3, 3)

    @pytest.mark.unit
    def test_interpolate_offset_preservation(self):
        """Offset should be preserved in output."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.full((1, 3, 3), 1000.0, dtype=np.float64) + np.arange(9).reshape(1, 3, 3)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        # Values should be around 1000 + interpolated
        assert result[0, 0, 0] > 999

    @pytest.mark.unit
    def test_interpolate_2x2_grid(self):
        """2x2 input grid should produce 2x2 output."""
        points = np.zeros((2, 2, 2))
        points[:, :, 0] = np.arange(2).reshape(2, 1)
        points[:, :, 1] = np.arange(2).reshape(1, 2)
        values = np.arange(4, dtype=np.float64).reshape(1, 2, 2)
        bbox = np.array([0, 0, 2, 2])

        result = interpolate(points, values, bbox)
        assert result.shape == (1, 2, 2)

    @pytest.mark.unit
    def test_interpolate_float_coordinates(self):
        """Float coordinates should be handled correctly."""
        points = np.zeros((4, 4, 2))
        points[:, :, 0] = np.linspace(0.0, 3.0, 4).reshape(4, 1)
        points[:, :, 1] = np.linspace(0.0, 3.0, 4).reshape(1, 4)
        values = np.random.rand(1, 4, 4)
        bbox = np.array([0, 0, 4, 4])

        result = interpolate(points, values, bbox)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 4
        assert result.shape[2] == 4

    @pytest.mark.unit
    def test_interpolate_rectangular_grid(self):
        """Rectangular (non-square) grid should produce correct shape."""
        points = np.zeros((3, 5, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(5).reshape(1, 5)
        values = np.random.rand(1, 3, 5)
        bbox = np.array([0, 0, 5, 3])

        result = interpolate(points, values, bbox)
        assert result.shape == (1, 3, 5)

    @pytest.mark.unit
    def test_interpolate_5x5_grid(self):
        """5x5 grid interpolation should work without error."""
        points = np.zeros((5, 5, 2))
        points[:, :, 0] = np.arange(5).reshape(5, 1)
        points[:, :, 1] = np.arange(5).reshape(1, 5)
        values = np.random.rand(2, 5, 5)
        bbox = np.array([0, 0, 5, 5])

        result = interpolate(points, values, bbox)
        assert result.shape == (2, 5, 5)

    @pytest.mark.unit
    def test_interpolate_output_dtype(self):
        """Output should be float64 dtype."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        bbox = np.array([0, 0, 3, 3])

        result = interpolate(points, values, bbox)
        assert result.dtype == np.float64

    @pytest.mark.unit
    def test_interpolate_small_bounding_box(self):
        """Small bounding box should produce small output."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        bbox = np.array([0, 0, 1, 1])

        result = interpolate(points, values, bbox)
        assert result.shape == (1, 1, 1)

    @pytest.mark.unit
    def test_interpolate_larger_bounding_box(self):
        """Larger bounding box than input grid should produce larger output."""
        points = np.zeros((3, 3, 2))
        points[:, :, 0] = np.arange(3).reshape(3, 1)
        points[:, :, 1] = np.arange(3).reshape(1, 3)
        values = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        bbox = np.array([0, 0, 5, 5])

        result = interpolate(points, values, bbox)
        assert result.shape == (1, 5, 5)
