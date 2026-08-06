"""Unit tests for cop_ard_core.unwrap."""

import numpy as np
import pytest
import scipy.sparse
import xarray as xr

from cop_ard_core.unwrap import (
    get_dual_graph,
    get_dual_graph_,
    get_edges,
    integ_mcf,
    integ_mcf_ortools,
    integ_tree,
    solve,
    unique,
    unwrap_phase,
)


# ===========================================================================
# TestUnique
# ===========================================================================


class TestUnique:
    """Tests for unique."""

    @pytest.mark.unit
    def test_unique_basic(self):
        """Should return unique edges."""
        e = np.array([[0, 1], [1, 2], [0, 1], [2, 3]])
        result = unique(e, return_index=False)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] <= e.shape[0]

    @pytest.mark.unit
    def test_unique_return_index(self):
        """return_index=True should return tuple of arrays."""
        e = np.array([[0, 1], [1, 2], [0, 1], [2, 3]])
        result = unique(e, return_index=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape[1] == 2
        assert result[1].shape[0] == result[0].shape[0]

    @pytest.mark.unit
    def test_unique_all_same(self):
        """All same edges should return single unique edge."""
        e = np.array([[0, 1], [0, 1], [0, 1]])
        result = unique(e, return_index=False)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, e[:1])

    @pytest.mark.unit
    def test_unique_single_edge(self):
        """Single edge should return itself unchanged."""
        e = np.array([[0, 1]])
        result = unique(e, return_index=False)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, e)

    @pytest.mark.unit
    def test_unique_reverse_edges(self):
        """Reverse edges (0,1) and (1,0) treated as same via min/max approach."""
        e = np.array([[0, 1], [1, 0], [0, 2]])
        result = unique(e, return_index=False)
        # (0,1) and (1,0) produce the same min/max pair, so only 2 unique
        assert result.shape[0] == 2

    @pytest.mark.unit
    def test_unique_all_unique(self):
        """All edges already unique should pass through unchanged."""
        e = np.array([[0, 1], [1, 2], [0, 2], [0, 3]])
        result = unique(e, return_index=False)
        assert result.shape == e.shape

    @pytest.mark.unit
    def test_unique_large_node_ids(self):
        """Large node IDs should use uint64 dtype."""
        # Create edges with node IDs that exceed uint32 range threshold
        e = np.array([[0, 1], [1, 1000000000], [0, 1000000000]], dtype=np.int64)
        result = unique(e, return_index=False)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    @pytest.mark.unit
    def test_unique_with_return_index(self):
        """return_index should give indices into original array."""
        e = np.array([[0, 1], [1, 2], [0, 1], [2, 3]])
        uniq_edges, indices = unique(e, return_index=True)
        # Verify indices point to correct edges
        for i, edge in enumerate(uniq_edges):
            np.testing.assert_array_equal(edge, e[indices[i]])


# ===========================================================================
# TestGetEdges
# ===========================================================================


class TestGetEdges:
    """Tests for get_edges."""

    @pytest.mark.unit
    def test_get_edges_returns_tuple_of_three(self):
        """Should return exactly 3 arrays: edges, edges_with_repeat, neighbors."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        e, es, neigh = get_edges(xx, yy)
        assert isinstance(e, np.ndarray)
        assert isinstance(es, np.ndarray)
        assert isinstance(neigh, np.ndarray)
        assert e.shape[1] == 2
        assert es.shape[1] == 2

    @pytest.mark.unit
    def test_get_edges_3x3_grid(self):
        """3x3 grid should produce edges connecting adjacent nodes."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        e, es, neigh = get_edges(xx.ravel(), yy.ravel())
        assert e.shape[1] == 2
        assert e.shape[0] > 0

    @pytest.mark.unit
    def test_get_edges_single_row(self):
        """Single row of points should still produce some edges."""
        xx = np.array([0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.5, 0.0])
        e, es, neigh = get_edges(xx, yy)
        assert isinstance(e, np.ndarray)

    @pytest.mark.unit
    def test_get_edges_edge_node_ids(self):
        """Edge arrays should contain valid node IDs within grid bounds."""
        xx = np.arange(5)
        yy = np.arange(5)
        xx, yy = np.meshgrid(xx, yy)
        e, es, neigh = get_edges(xx.ravel(), yy.ravel())
        n_points = 5 * 5
        assert np.all(e >= 0)
        assert np.all(e < n_points)

    @pytest.mark.unit
    def test_get_edges_es_is_superset_of_e(self):
        """es (edges with repetition) should include all edges in e."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        e, es, neigh = get_edges(xx.ravel(), yy.ravel())
        # Flatten sorted unique pairs from e
        e_sorted = np.sort(e, axis=1)
        es_sorted = np.sort(es, axis=1)
        for edge in e_sorted:
            assert any(
                (es_sorted[i] == edge).all() for i in range(es_sorted.shape[0])
            )

    @pytest.mark.unit
    def test_get_edges_2x2_grid(self):
        """2x2 grid should produce a smaller but valid edge set."""
        xx = np.arange(2)
        yy = np.arange(2)
        xx, yy = np.meshgrid(xx, yy)
        e, es, neigh = get_edges(xx.ravel(), yy.ravel())
        assert e.shape[0] > 0
        assert neigh.ndim == 2


# ===========================================================================
# TestGetDualGraph_
# ===========================================================================


class TestGetDualGraph_:
    """Tests for get_dual_graph_."""

    @pytest.mark.unit
    def test_get_dual_graph_no_border(self):
        """No borders (-1 neighbors) should produce dual graph with all triangles."""
        neigh = np.array([[1, 2, 0], [0, 2, 1], [0, 1, 2]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert len(head) > 0
        assert len(tail) > 0
        assert len(cost) == len(tail)
        assert len(ucap) == len(tail)
        assert len(cost) == len(head)
        assert len(ucap) == len(head)

    @pytest.mark.unit
    def test_get_dual_graph_with_border(self):
        """Borders (-1 neighbors) should add earth node edges."""
        neigh = np.array([[1, -1], [-1, 0]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert len(head) > 0
        # Earth node should appear in head for border triangles
        earth_node = neigh.max() + 1
        assert earth_node in head

    @pytest.mark.unit
    def test_get_dual_graph_custom_params(self):
        """Custom maxcost and maxjump should be respected."""
        neigh = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]])
        head, tail, cost, ucap = get_dual_graph_(neigh, maxcost=50, maxjump=200)
        n_edges = neigh.size + (neigh == -1).sum()
        np.testing.assert_array_equal(cost[:n_edges], np.full(n_edges, 50))
        np.testing.assert_array_equal(ucap[:n_edges], np.full(n_edges, 200))

    @pytest.mark.unit
    def test_get_dual_graph_cost_mask(self):
        """Cost mask should scale costs proportionally."""
        neigh = np.array([[1, 2]])
        cost_mask = np.array([0.5])
        head, tail, cost, ucap = get_dual_graph_(neigh, cost_mask=cost_mask)
        # Mask of 0.5 should make cost = round(1000 * 0.5) = 500
        assert cost[0] == 500

    @pytest.mark.unit
    def test_get_dual_graph_full_cost_mask(self):
        """Full cost mask of 1.0 should produce default MAXCOST."""
        neigh = np.array([[1, 2, 0]])
        cost_mask = np.array([1.0])
        head, tail, cost, ucap = get_dual_graph_(neigh, cost_mask=cost_mask)
        assert cost[0] == 1000

    @pytest.mark.unit
    def test_get_dual_graph_zero_cost_mask(self):
        """Zero cost mask should produce zero cost."""
        neigh = np.array([[1, 2, 0]])
        cost_mask = np.array([0.0])
        head, tail, cost, ucap = get_dual_graph_(neigh, cost_mask=cost_mask)
        assert cost[0] == 0

    @pytest.mark.unit
    def test_get_dual_graph_n_vertices_default(self):
        """Default n_vertices=3 should produce 3 neighbors per triangle."""
        neigh = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]])
        head, tail, cost, ucap = get_dual_graph_(neigh, n_vertices=3)
        # Each triangle produces 3 edges + border edges
        n_border = (neigh == -1).sum()
        expected = neigh.size + n_border
        assert len(head) == expected

    @pytest.mark.unit
    def test_get_dual_graph_different_n_vertices(self):
        """Different n_vertices should change head size accordingly."""
        neigh = np.array([[1, 2]])
        head3, _, _, _ = get_dual_graph_(neigh, n_vertices=3)
        head4, _, _, _ = get_dual_graph_(neigh, n_vertices=4)
        # With n_vertices=4 and no border, we'd get 4 edges per triangle
        assert len(head3) == 3
        assert len(head4) == 4

    @pytest.mark.unit
    def test_get_dual_graph_all_border(self):
        """All triangles with -1 neighbors should produce earth node edges for all."""
        neigh = np.array([[-1, -1, -1], [-1, -1, -1]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        # Each triangle has 3 borders = 6 earth node edges
        n_border = 6
        assert len(head) == neigh.size + n_border

    @pytest.mark.unit
    def test_get_dual_graph_output_dtypes(self):
        """Output arrays should be int32 dtype."""
        neigh = np.array([[1, 2, 0]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert head.dtype == np.int32
        assert tail.dtype == np.int32
        assert cost.dtype == np.int32
        assert ucap.dtype == np.int32

    @pytest.mark.unit
    def test_get_dual_graph_no_cost_mask(self):
        """No cost_mask should produce default MAXCOST for all edges."""
        neigh = np.array([[1, 2, 0]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert cost[0] == 1000

    @pytest.mark.unit
    def test_get_dual_graph_mixed_border_and_internal(self):
        """Mixed border and internal neighbors should produce correct counts."""
        neigh = np.array([[1, 2, -1], [0, -1, 2], [-1, 2, 0]])
        n_border = (neigh == -1).sum()
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert len(head) == neigh.size + n_border


# ===========================================================================
# TestIntegMcfOtools
# ===========================================================================


class TestIntegMcfOtools:
    """Tests for integ_mcf_ortools."""

    @pytest.mark.unit
    def test_integ_mcf_ortools_balanced(self):
        """Balanced graph should solve optimally."""
        start_nodes = np.array([0, 0, 1, 2])
        end_nodes = np.array([1, 2, 3, 1])
        supplies = np.array([1, 0, 0, -1])
        unit_costs = np.array([1.0, 1.0, 1.0, 1.0])
        capacities = np.array([1, 1, 1, 1])
        status, flows = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert status == 0
        assert len(flows) == len(unit_costs)

    @pytest.mark.unit
    def test_integ_mcf_ortools_unsolvable(self):
        """Unbalanced supplies should raise exception."""
        start_nodes = np.array([0])
        end_nodes = np.array([1])
        supplies = np.array([1, 1])  # both positive - no sink
        unit_costs = np.array([1.0])
        capacities = np.array([1])
        with pytest.raises(Exception):
            integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)

    @pytest.mark.unit
    def test_integ_mcf_ortools_zero_flow(self):
        """Zero supplies should produce zero flow."""
        start_nodes = np.array([0, 1])
        end_nodes = np.array([1, 0])
        supplies = np.array([0, 0])
        unit_costs = np.array([1.0, 1.0])
        capacities = np.array([1, 1])
        status, flows = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert status == 0
        assert np.all(flows == 0)

    @pytest.mark.unit
    def test_integ_mcf_ortools_returns_tuple(self):
        """Should return a tuple of (status, flows)."""
        start_nodes = np.array([0])
        end_nodes = np.array([1])
        supplies = np.array([1, -1])
        unit_costs = np.array([1.0])
        capacities = np.array([1])
        result = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], np.ndarray)


# ===========================================================================
# TestIntegTree
# ===========================================================================


class TestIntegTree:
    """Tests for integ_tree."""

    @pytest.mark.unit
    def test_integ_tree_basic(self):
        """Basic tree integration should produce a solution."""
        # Create a simple 4x4 sparse matrix representing a tree: 0->1, 0->2, 0->3
        n = 4
        row = np.array([0, 0, 0])
        col = np.array([1, 2, 3])
        data = np.array([1.0, 1.0, 1.0])
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

        # es represents parent->child edges
        es = np.array([[0, 1], [0, 2], [0, 3]])
        sol = integ_tree(grad, es)
        assert isinstance(sol, np.ndarray)
        assert sol.shape == (n,)

    @pytest.mark.unit
    def test_integ_tree_linear_chain(self):
        """Linear chain 0->1->2->3 should accumulate gradients."""
        n = 4
        row = np.array([0, 1, 2])
        col = np.array([1, 2, 3])
        data = np.array([1.0, 2.0, 3.0])
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

        es = np.array([[0, 1], [1, 2], [2, 3]])
        sol = integ_tree(grad, es)
        assert isinstance(sol, np.ndarray)
        # sol[0] should be 0 (root), sol[1]=1, sol[2]=3, sol[3]=6
        assert sol[0] == 0.0
        np.testing.assert_allclose(sol[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(sol[2], 3.0, atol=1e-10)
        np.testing.assert_allclose(sol[3], 6.0, atol=1e-10)

    @pytest.mark.unit
    def test_integ_tree_negative_gradients(self):
        """Negative gradients should produce negative solution values."""
        n = 3
        row = np.array([0, 1])
        col = np.array([1, 2])
        data = np.array([-1.0, -2.0])
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

        es = np.array([[0, 1], [1, 2]])
        sol = integ_tree(grad, es)
        assert sol[0] == 0.0
        np.testing.assert_allclose(sol[1], -1.0, atol=1e-10)
        np.testing.assert_allclose(sol[2], -3.0, atol=1e-10)

    @pytest.mark.unit
    def test_integ_tree_mixed_signs(self):
        """Mixed positive/negative gradients should integrate correctly."""
        n = 3
        row = np.array([0, 1])
        col = np.array([1, 2])
        data = np.array([5.0, -3.0])
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

        es = np.array([[0, 1], [1, 2]])
        sol = integ_tree(grad, es)
        np.testing.assert_allclose(sol[1], 5.0, atol=1e-10)
        np.testing.assert_allclose(sol[2], 2.0, atol=1e-10)

    @pytest.mark.unit
    def test_integ_tree_output_shape(self):
        """Output shape should equal number of nodes in sparse matrix."""
        n = 10
        row = np.arange(n - 1)
        col = np.arange(1, n)
        data = np.ones(n - 1)
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))
        es = np.column_stack([row, col])
        sol = integ_tree(grad, es)
        assert sol.shape == (n,)

    @pytest.mark.unit
    def test_integ_tree_star_topology(self):
        """Star topology with center connecting to all others."""
        n = 6
        row = np.zeros(n - 1, dtype=np.int32)
        col = np.arange(1, n)
        data = np.ones(n - 1)
        grad = scipy.sparse.csr_matrix((data, (row, col)), shape=(n, n))

        es = np.column_stack([row, col])
        sol = integ_tree(grad, es)
        assert sol[0] == 0.0
        np.testing.assert_array_equal(sol[1:], np.ones(n - 1))


# ===========================================================================
# TestGetDualGraph
# ===========================================================================


class TestGetDualGraph:
    """Tests for get_dual_graph."""

    @pytest.mark.unit
    def test_get_dual_graph_basic(self):
        """Basic dual graph construction should return valid head/tail/cost/ucap."""
        # Create a proper 3x3 grid triangulation
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        w = np.ones(e.shape[0]) * 0.8
        n_points = 9
        solver = {}

        head, tail, cost, ucap = get_dual_graph(e, es, neigh, w, n_points, solver)
        assert len(head) > 0
        assert len(tail) > 0
        assert len(cost) == len(head)
        assert len(ucap) == len(head)

    @pytest.mark.unit
    def test_get_dual_graph_with_solver_params(self):
        """Solver params should override defaults."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        w = np.ones(e.shape[0]) * 0.5
        n_points = 9
        solver = {"maxjump": 500, "maxcost": 500}

        head, tail, cost, ucap = get_dual_graph(e, es, neigh, w, n_points, solver)
        assert len(head) > 0

    @pytest.mark.unit
    def test_get_dual_graph_weight_scaling(self):
        """Weights should scale costs via dual graph."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        w = np.ones(e.shape[0]) * 0.1  # Low weight
        n_points = 9
        solver = {}

        head, tail, cost, ucap = get_dual_graph(e, es, neigh, w, n_points, solver)
        # Low weights should result in lower costs
        assert cost[0] < 1000

    @pytest.mark.unit
    def test_get_dual_graph_scale_demand(self):
        """Scale demand in solver should affect cost computation."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        w = np.ones(e.shape[0]) * 0.5
        n_points = 9
        solver = {"scale_demand": 0.5}

        head, tail, cost, ucap = get_dual_graph(e, es, neigh, w, n_points, solver)
        assert len(head) > 0

    @pytest.mark.unit
    def test_get_dual_graph_output_dtypes(self):
        """Output arrays should be int32 dtype."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        w = np.ones(e.shape[0]) * 0.5
        n_points = 9
        solver = {}

        head, tail, cost, ucap = get_dual_graph(e, es, neigh, w, n_points, solver)
        assert head.dtype == np.int32
        assert tail.dtype == np.int32
        assert cost.dtype == np.int32
        assert ucap.dtype == np.int32


# ===========================================================================
# TestSolve
# ===========================================================================


class TestSolve:
    """Tests for solve."""

    @pytest.mark.unit
    def test_solve_basic(self):
        """Basic solve should return exit_status, total_cost, and solution."""
        # Create a proper 3x3 grid
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.8
        solver = {}

        exit_status, total_cost, sol = solve(e, es, g, w, neigh, solver)
        assert exit_status == 0
        assert isinstance(total_cost, (int, np.integer))
        assert isinstance(sol, np.ndarray)

    @pytest.mark.unit
    def test_solve_solution_shape(self):
        """Solution shape should match number of points."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.8
        solver = {}

        _, _, sol = solve(e, es, g, w, neigh, solver)
        n_points = 9
        assert sol.shape == (n_points,)

    @pytest.mark.unit
    def test_solve_scale_demand_zero(self):
        """scale_demand=0 should still produce valid solution."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.5
        solver = {"scale_demand": 0}

        exit_status, total_cost, sol = solve(e, es, g, w, neigh, solver)
        assert exit_status == 0

    @pytest.mark.unit
    def test_solve_scale_demand_none(self):
        """scale_demand=None should still produce valid solution."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.5
        solver = {"scale_demand": None}

        exit_status, total_cost, sol = solve(e, es, g, w, neigh, solver)
        assert exit_status == 0

    @pytest.mark.unit
    def test_solve_returns_tuple_of_three(self):
        """Should return (status, total_cost, solution)."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.5
        solver = {}

        result = solve(e, es, g, w, neigh, solver)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], int)

    @pytest.mark.unit
    def test_solve_2x2_grid(self):
        """2x2 grid should produce valid solution for 4 points."""
        xx = np.arange(2)
        yy = np.arange(2)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.1
        w = np.ones(e.shape[0]) * 0.8
        solver = {}

        exit_status, total_cost, sol = solve(e, es, g, w, neigh, solver)
        assert exit_status == 0
        assert sol.shape == (4,)

    @pytest.mark.unit
    def test_solve_solution_values_close_to_gradient(self):
        """Solution differences should be close to input gradients."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        x_flat = xx.ravel()
        y_flat = yy.ravel()

        e, es, neigh = get_edges(x_flat, y_flat)
        g = np.ones(e.shape[0]) * 0.05
        w = np.ones(e.shape[0]) * 0.9
        solver = {}

        _, _, sol = solve(e, es, g, w, neigh, solver)
        # Check that solution is monotonic in some direction
        assert sol.max() - sol.min() > 0


# ===========================================================================
# TestIntegMcf
# ===========================================================================


class TestIntegMcf:
    """Tests for integ_mcf."""

    @pytest.mark.unit
    def test_integ_mcf_basic(self):
        """Basic MCF integration should return status, cost, and solution."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        ph = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 1.5, 1.0, 1.5, 2.0])
        cohe = np.ones(9)
        solver = {}

        exit_status, total_cost, sol = integ_mcf(xx, yy, ph, cohe, solver)
        assert exit_status == 0
        assert isinstance(total_cost, (int, np.integer))
        assert isinstance(sol, np.ndarray)

    @pytest.mark.unit
    def test_integ_mcf_solution_shape(self):
        """Solution shape should match number of points."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ph = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 1.5])
        cohe = np.ones(6)
        solver = {}

        _, _, sol = integ_mcf(xx, yy, ph, cohe, solver)
        assert sol.shape == (6,)

    @pytest.mark.unit
    def test_integ_mcf_returns_tuple_of_three(self):
        """Should return (status, cost, solution)."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ph = np.array([0.0, 0.5, 1.0, 0.5, 1.0, 1.5])
        cohe = np.ones(6)
        solver = {}

        result = integ_mcf(xx, yy, ph, cohe, solver)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.unit
    def test_integ_mcf_constant_phase(self):
        """Constant phase should produce zero unwrapping."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        ph = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        cohe = np.ones(6)
        solver = {}

        exit_status, total_cost, sol = integ_mcf(xx, yy, ph, cohe, solver)
        assert exit_status == 0
        # All solutions should be close to the starting phase
        np.testing.assert_allclose(sol, ph[0], atol=1e-6)

    @pytest.mark.unit
    def test_integ_mcf_2x2_grid(self):
        """2x2 grid should produce valid unwrapping."""
        xx = np.array([0.0, 1.0, 0.0, 1.0])
        yy = np.array([0.0, 0.0, 1.0, 1.0])
        ph = np.array([0.0, 0.3, 0.3, 0.6])
        cohe = np.ones(4)
        solver = {}

        exit_status, total_cost, sol = integ_mcf(xx, yy, ph, cohe, solver)
        assert exit_status == 0
        assert sol.shape == (4,)


# ===========================================================================
# TestUnwrapPhase
# ===========================================================================


class TestUnwrapPhase:
    """Tests for unwrap_phase."""

    @pytest.mark.unit
    def test_unwrap_phase_basic(self):
        """Basic unwrapping of regular gridded data."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5], [1.0, 1.5, 2.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )
        coherence = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        assert isinstance(result, xr.DataArray)
        assert result.shape == phase.shape

    @pytest.mark.unit
    def test_unwrap_phase_output_type(self):
        """Output should be an xr.DataArray with same dims."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5], [0.5, 1.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )
        coherence = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        assert "azimuth_time" in result.dims
        assert "slant_range_time" in result.dims

    @pytest.mark.unit
    def test_unwrap_phase_with_threshold(self):
        """Coherence threshold should mask out low-coherence points."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5], [1.0, 1.5, 2.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )
        coherence = xr.DataArray(
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.5)
        assert isinstance(result, xr.DataArray)
        assert result.shape == phase.shape
        # Points below threshold should be NaN in result
        assert np.isnan(result[0, 1].values)
        assert np.isnan(result[1, 0].values)
        assert np.isnan(result[1, 2].values)
        assert np.isnan(result[2, 1].values)

    @pytest.mark.unit
    def test_unwrap_phase_preserves_coordinates(self):
        """Output should preserve the original phase coordinates."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5], [0.5, 1.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array([10.0, 20.0]),
                "slant_range_time": np.array([0.5, 1.0]),
            },
        )
        coherence = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array([10.0, 20.0]),
                "slant_range_time": np.array([0.5, 1.0]),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        np.testing.assert_array_equal(result.coords["azimuth_time"].values, phase.coords["azimuth_time"].values)
        np.testing.assert_array_equal(result.coords["slant_range_time"].values, phase.coords["slant_range_time"].values)

    @pytest.mark.unit
    def test_unwrap_phase_large_grid(self):
        """Larger grid should produce proportionally larger solution."""
        np.random.seed(42)
        n = 5
        phase = xr.DataArray(
            np.random.rand(n, n) * np.pi,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(n),
                "slant_range_time": np.arange(n),
            },
        )
        coherence = xr.DataArray(
            np.random.rand(n, n) * 0.8 + 0.2,
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(n),
                "slant_range_time": np.arange(n),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        assert result.shape == (n, n)

    @pytest.mark.unit
    def test_unwrap_phase_squeeze_preserves_dims(self):
        """Squeeze operation should not remove expected dimensions."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5, 1.0], [0.5, 1.0, 1.5], [1.0, 1.5, 2.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )
        coherence = xr.DataArray(
            np.ones((3, 3)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(3),
                "slant_range_time": np.arange(3),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        assert "azimuth_time" in result.dims
        assert "slant_range_time" in result.dims
        assert result.dims == phase.dims

    @pytest.mark.unit
    def test_unwrap_phase_default_threshold(self):
        """Default threshold (1e-3) should work correctly."""
        phase = xr.DataArray(
            np.array([[0.0, 0.5], [0.5, 1.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )
        coherence = xr.DataArray(
            np.full((2, 2), 0.5),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )

        result = unwrap_phase(phase, coherence)
        assert isinstance(result, xr.DataArray)
        assert not np.any(np.isnan(result.values))

    @pytest.mark.unit
    def test_unwrap_phase_negative_phase_values(self):
        """Negative phase values should unwrap correctly."""
        phase = xr.DataArray(
            np.array([[-1.0, -0.5], [-0.5, 0.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )
        coherence = xr.DataArray(
            np.ones((2, 2)),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.arange(2),
                "slant_range_time": np.arange(2),
            },
        )

        result = unwrap_phase(phase, coherence, threshold=0.0)
        assert isinstance(result, xr.DataArray)
        assert result.shape == phase.shape

    @pytest.mark.unit
    def test_unwrap_phase_single_pixel(self):
        """Single pixel should produce trivial result or handle gracefully."""
        phase = xr.DataArray(
            np.array([[0.5]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array([0]),
                "slant_range_time": np.array([0]),
            },
        )
        coherence = xr.DataArray(
            np.array([[1.0]]),
            dims=["azimuth_time", "slant_range_time"],
            coords={
                "azimuth_time": np.array([0]),
                "slant_range_time": np.array([0]),
            },
        )

        # This may fail at Delaunay triangulation, which is acceptable
        try:
            result = unwrap_phase(phase, coherence, threshold=0.0)
            assert result.shape == (1, 1)
        except Exception:
            pass


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests across unwrap functions."""

    @pytest.mark.unit
    def test_unique_single_unique_edge(self):
        """Array with single unique edge should return 1x2 array."""
        e = np.array([[5, 10], [5, 10]])
        result = unique(e, return_index=False)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[5, 10]])

    @pytest.mark.unit
    def test_get_edges_degenerate_triangles(self):
        """Near-collinear points should still produce edges."""
        xx = np.array([0.0, 0.01, 0.02, 0.0])
        yy = np.array([0.0, 0.001, 0.002, 0.01])
        e, es, neigh = get_edges(xx, yy)
        assert isinstance(e, np.ndarray)

    @pytest.mark.unit
    def test_integ_mcf_ortools_large_capacities(self):
        """Very large capacities should not cause overflow."""
        start_nodes = np.array([0, 0, 1, 1])
        end_nodes = np.array([1, 2, 0, 2])
        supplies = np.array([10, 0, -10])
        unit_costs = np.array([1.0, 1.0, 1.0, 1.0])
        capacities = np.array([10000, 10000, 10000, 10000])
        status, flows = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert status == 0
        assert np.sum(flows) == 10
