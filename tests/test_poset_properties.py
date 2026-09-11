"""Hypothesis property tests for `PartiallyOrderedSet` (P31).

Builds a random DAG on a handful of integer nodes, with edges only from a
smaller node to a larger one (acyclic by construction), added to the poset
in a drawn order. `networkx.has_path` over an equivalent `DiGraph` built
from the same edges is the oracle for `is_less_than`; `is_greater_than` and
the `add_order` cycle check are then checked against `is_less_than` itself.
"""

import pytest

pytest.importorskip("hypothesis")

import networkx as nx  # type: ignore[import-untyped]
from hypothesis import given
from hypothesis import strategies as st

from fhy_core.utils.poset import PartiallyOrderedSet

pytestmark = pytest.mark.property

_MIN_NODES = 2
_MAX_NODES = 6


@st.composite
def draw_dag_edges(draw: st.DrawFn) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Draw a random DAG: nodes `0..n-1`, edges only from a smaller node to
    a larger one, in a drawn add order.

    Restricting candidate edges to `(i, j)` with `i < j` makes the graph
    acyclic by construction, so `add_order` never sees a candidate edge
    that would close a cycle; the add order is still shuffled so the poset
    is not always built in a topologically sorted sequence.
    """
    node_count = draw(st.integers(min_value=_MIN_NODES, max_value=_MAX_NODES))
    candidate_edges = [
        (lower, upper)
        for lower in range(node_count)
        for upper in range(lower + 1, node_count)
    ]
    included_edges = [edge for edge in candidate_edges if draw(st.booleans())]
    ordered_edges = draw(st.permutations(included_edges))
    return node_count, tuple(ordered_edges)


def build_poset_and_graph(
    node_count: int, edges: tuple[tuple[int, int], ...]
) -> tuple[PartiallyOrderedSet[int], nx.DiGraph]:
    """Build a `PartiallyOrderedSet` and an equivalent `networkx.DiGraph`.

    Both are built from the same nodes and edges, added in the same order,
    so the poset's reachability and the graph's reachability agree by
    construction; the graph is the test-side oracle for `is_less_than`.
    """
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    graph: nx.DiGraph = nx.DiGraph()
    for node in range(node_count):
        poset.add_element(node)
        graph.add_node(node)
    for lower, upper in edges:
        poset.add_order(lower, upper)
        graph.add_edge(lower, upper)
    return poset, graph


@given(draw_dag_edges())
def test_is_less_than_matches_reachability_in_the_added_edges(
    dag: tuple[int, tuple[tuple[int, int], ...]],
) -> None:
    """Test `is_less_than(a, b)` equals `networkx.has_path(graph, a, b)` for
    every pair of nodes.

    Oracle: `networkx.has_path` over a `DiGraph` built from the same edges.
    """
    node_count, edges = dag
    poset, graph = build_poset_and_graph(node_count, edges)

    for lower in range(node_count):
        for upper in range(node_count):
            assert poset.is_less_than(lower, upper) == nx.has_path(graph, lower, upper)


@given(draw_dag_edges())
def test_is_greater_than_is_the_converse_of_is_less_than(
    dag: tuple[int, tuple[tuple[int, int], ...]],
) -> None:
    """Test `is_greater_than(a, b)` equals `is_less_than(b, a)` for every pair."""
    node_count, edges = dag
    poset, _ = build_poset_and_graph(node_count, edges)

    for lower in range(node_count):
        for upper in range(node_count):
            assert poset.is_greater_than(lower, upper) == poset.is_less_than(
                upper, lower
            )


@st.composite
def draw_dag_edges_with_pair(
    draw: st.DrawFn,
) -> tuple[int, tuple[tuple[int, int], ...], int, int]:
    """Draw a DAG together with a pair of its nodes to try reversing an order on."""
    node_count, edges = draw(draw_dag_edges())
    first = draw(st.integers(min_value=0, max_value=node_count - 1))
    second = draw(st.integers(min_value=0, max_value=node_count - 1))
    return node_count, edges, first, second


@given(draw_dag_edges_with_pair())
def test_add_order_raises_iff_the_reverse_order_already_holds(
    case: tuple[int, tuple[tuple[int, int], ...], int, int],
) -> None:
    """Test `add_order(b, a)` raises `RuntimeError` exactly when `is_less_than(a, b)`.

    Each example builds two fresh posets from the same edges: one to read
    `is_less_than(a, b)` from, and a second, untouched one to attempt
    `add_order(b, a)` on, so the attempt's mutation (or the cycle check
    that precedes it) never contaminates a poset used for a different
    assertion.
    """
    node_count, edges, first, second = case
    reference_poset, _ = build_poset_and_graph(node_count, edges)
    reverse_order_would_cycle = reference_poset.is_less_than(first, second)

    fresh_poset, _ = build_poset_and_graph(node_count, edges)
    if reverse_order_would_cycle:
        with pytest.raises(RuntimeError):
            fresh_poset.add_order(second, first)
    else:
        fresh_poset.add_order(second, first)
