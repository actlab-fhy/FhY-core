"""Hypothesis strategy for random directed acyclic graphs.

Shared by the order-theory property files: `test_poset_properties.py` draws
a `PartiallyOrderedSet` from the same edges, and `test_lattice_properties.py`
draws an arbitrary (not-necessarily-a-lattice) `Lattice` from them. Edges run
only from a smaller node to a larger one, which makes the graph acyclic by
construction, and the edge list is drawn in a shuffled add order so the
structure is not always built in a topologically sorted sequence.
"""

from typing import Final

from hypothesis import strategies as st

__all__ = ["draw_random_dag"]

_MIN_NODES: Final = 2
_MAX_NODES: Final = 6


@st.composite
def draw_random_dag(draw: st.DrawFn) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Draw a random DAG: nodes ``0..n-1``, edges only from a smaller node to
    a larger one, in a drawn add order.

    Restricting candidate edges to ``(i, j)`` with ``i < j`` makes the graph
    acyclic by construction, so no candidate edge can ever close a cycle;
    the add order is still shuffled so the structure is not always built in
    a topologically sorted sequence.
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
