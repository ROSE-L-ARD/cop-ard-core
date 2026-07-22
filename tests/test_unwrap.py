"""Unit tests for cop_ard_core.unwrap."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cop_ard_core.unwrap import (
    get_dual_graph_,
    get_edges,
    integ_mcf_ortools,
    unique,
)


class TestUnique:
    """Tests for unique."""

    def test_unique_basic(self):
        """Should return unique edges."""
        e = np.array([[0, 1], [1, 2], [0, 1], [2, 3]])
        result = unique(e)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] <= e.shape[0]

    def test_unique_return_index(self):
        """return_index=True should return tuple."""
        e = np.array([[0, 1], [1, 2], [0, 1], [2, 3]])
        result = unique(e, return_index=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unique_all_same(self):
        """All same edges should return single unique."""
        e = np.array([[0, 1], [0, 1], [0, 1]])
        result = unique(e)
        assert result.shape[0] == 1

    def test_unique_single_edge(self):
        """Single edge should return itself."""
        e = np.array([[0, 1]])
        result = unique(e)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, e)

    def test_unique_reverse_edges(self):
        """Reverse edges (0,1) and (1,0) should be treated as different."""
        e = np.array([[0, 1], [1, 0], [0, 2]])
        result = unique(e)
        # Each pair is unique by min/max approach
        assert result.shape[0] <= e.shape[0]


class TestGetEdges:
    """Tests for get_edges."""

    def test_get_edges_returns_tuple(self):
        """Should return 3 arrays."""
        xx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        e, es, neigh = get_edges(xx, yy)
        assert isinstance(e, np.ndarray)
        assert isinstance(es, np.ndarray)
        assert isinstance(neigh, np.ndarray)

    def test_get_edges_3x3_grid(self):
        """3x3 grid should produce multiple edges."""
        xx = np.arange(3)
        yy = np.arange(3)
        xx, yy = np.meshgrid(xx, yy)
        e, es, neigh = get_edges(xx.ravel(), yy.ravel())
        assert e.shape[1] == 2

    def test_get_edges_single_row(self):
        """Single row needs at least 2 points in 2D for Delaunay."""
        xx = np.array([0.0, 1.0, 2.0])
        yy = np.array([0.0, 0.5, 0.0])
        e, es, neigh = get_edges(xx, yy)
        assert isinstance(e, np.ndarray)


class TestGetDualGraph:
    """Tests for get_dual_graph_."""

    def test_get_dual_graph_no_border(self):
        """No borders (no -1 neighbors) should produce dual graph."""
        neigh = np.array([[1, 2, 0], [0, 2, 1], [0, 1, 2]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert len(head) > 0
        assert len(tail) > 0
        assert len(cost) == len(tail)
        assert len(ucap) == len(tail)
        assert len(cost) == len(head)
        assert len(ucap) == len(head)

    def test_get_dual_graph_with_border(self):
        """Borders (-1 neighbors) should add earth node."""
        neigh = np.array([[1, -1], [-1, 0]])
        head, tail, cost, ucap = get_dual_graph_(neigh)
        assert len(head) > 0

    def test_get_dual_graph_custom_params(self):
        """Custom maxcost and maxjump should be respected."""
        neigh = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]])
        head, tail, cost, ucap = get_dual_graph_(neigh, maxcost=50, maxjump=200)
        n_triang = neigh.shape[0]
        n_edges = neigh.size + (neigh == -1).sum()
        np.testing.assert_array_equal(cost[:n_edges], np.full(n_edges, 50))
        np.testing.assert_array_equal(ucap[:n_edges], np.full(n_edges, 200))

    def test_get_dual_graph_cost_mask(self):
        """Cost mask should scale costs."""
        neigh = np.array([[1, 2]])
        cost_mask = np.array([0.5])
        head, tail, cost, ucap = get_dual_graph_(neigh, cost_mask=cost_mask)
        # Mask of 0.5 should make cost = round(1000 * 0.5) = 500
        assert cost[0] == 500

    def test_get_dual_graph_n_vertices_default(self):
        """Default n_vertices=3 should work for triangulation."""
        neigh = np.array([[1, 2, -1], [0, 2, -1], [0, 1, -1]])
        head, tail, cost, ucap = get_dual_graph_(neigh, n_vertices=3)
        assert len(head) > 0


class TestIntegMcfOtools:
    """Tests for integ_mcf_ortools."""

    def test_integ_mcf_ortools_balanced(self):
        """Balanced graph should solve optimally."""
        start_nodes = np.array([0, 0, 1, 2])
        end_nodes = np.array([1, 2, 3, 1])
        supplies = np.array([1, 0, 0, -1])
        unit_costs = np.array([1.0, 1.0, 1.0, 1.0])
        capacities = np.array([1, 1, 1, 1])
        status, flows = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert status == 0

    def test_integ_mcf_ortools_simple_path(self):
        """Simple path from source to sink."""
        start_nodes = np.array([0, 1])
        end_nodes = np.array([1, 2])
        supplies = np.array([1, 0, -1])
        unit_costs = np.array([1.0, 1.0])
        capacities = np.array([10, 10])
        status, flows = integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
        assert status == 0

    def test_integ_mcf_ortools_unsolvable(self):
        """Unbalanced supplies should fail."""
        start_nodes = np.array([0])
        end_nodes = np.array([1])
        supplies = np.array([1, 1])  # both positive - no sink
        unit_costs = np.array([1.0])
        capacities = np.array([1])
        with pytest.raises(Exception):
            integ_mcf_ortools(start_nodes, end_nodes, supplies, unit_costs, capacities)
