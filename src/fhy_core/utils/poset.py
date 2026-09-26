"""Partially ordered set (poset) utility.

Backed by the Rust implementation (S11a of ``docs/design/python-switch.md``):
``fhy_core._rs.PartiallyOrderedSet`` keeps the elements in a ``dict`` from
element to position, so membership follows the elements' own ``__hash__``
and ``__eq__``, and runs the order over the positions in the Rust core,
where each element keeps its up-set and asking for an order is a bit test.
"""

__all__ = ["PartiallyOrderedSet"]

from typing import Generic, TypeVar

from fhy_core import _rs

T = TypeVar("T")


class PartiallyOrderedSet(_rs.PartiallyOrderedSet, Generic[T]):
    """A partially ordered set (poset) of hashable elements.

    The order is the reflexive and transitive closure of the orders added
    with :meth:`add_order`: every element is at most itself, and
    :meth:`is_less_than` reads "less than or equal to".

    Methods:
        add_element(element): Add an element. Raises ``ValueError`` if it is
            a member, and ``TypeError`` if it is not hashable.
        add_order(lower, upper): Order ``lower`` below ``upper``. Raises
            ``ValueError`` if either is not a member, and ``RuntimeError``
            if ``upper`` is already at most ``lower``, which includes
            ``lower == upper``. Adding an order that already holds is
            accepted and changes nothing.
        is_less_than(lower, upper): Whether ``lower`` is less than or equal
            to ``upper``. Raises ``ValueError`` if either is not a member.
        is_greater_than(lower, upper): Whether ``lower`` is greater than or
            equal to ``upper``. Raises ``ValueError`` if either is not a
            member.
        iter_stable(key=repr): Iterate in a topological order in which,
            among the elements that can come next, the one with the least
            ``key(element)`` comes first, and of equal keys the one added
            first. ``key`` is called once per element.

    Iterating the poset yields its elements in a topological order in which
    the element added first comes first among those that can come next, so
    the order is stable across runs.
    """
