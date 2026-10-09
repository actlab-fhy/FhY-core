"""Lattice (order theory) utility.

Backed by the Rust implementation: ``fhy_core._rs.Lattice`` keeps its
elements as the partially ordered set does, and computes meets, joins and
the missing bounds in the Rust core.
"""

__all__ = ["Lattice"]

from typing import Generic, TypeVar

from fhy_core import _rs
from fhy_core.traits.verifiable import VerifiableMixin

T = TypeVar("T")


class Lattice(_rs.Lattice, Generic[T]):
    """Lattice (order theory) over hashable elements.

    Any partial order can be held; it is a lattice when every pair of its
    elements has a unique greatest lower bound (meet) and a unique least
    upper bound (join).

    Methods:
        add_element(element), add_order(lower, upper): Build the order, as
            :class:`~fhy_core.utils.poset.PartiallyOrderedSet` does.
        get_meet(x, y), get_join(x, y): The meet or join, or ``None`` if
            there is none or several incomparable candidates. Raise
            ``ValueError`` if either argument is not a member.
        has_meet(x, y), has_join(x, y): Whether the meet or join exists.
        get_least_upper_bound(x, y): The join. Raises ``RuntimeError`` if
            there is none.
        is_lattice(): Whether every pair has a meet and a join.
        verify(): A :class:`~fhy_core.diagnostic.ValidationReport` with one
            ERROR diagnostic for every ordered pair of elements, in
            iteration order, that lacks a meet or a join, its meet first; the
            report is empty for a lattice.

    It is a :class:`~fhy_core.traits.verifiable.VerifiableMixin` by
    registration, and defines :meth:`verify` itself.
    """


VerifiableMixin.register(Lattice)
