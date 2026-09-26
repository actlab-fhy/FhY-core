"""Constraint-member representation: the member kinds and their collections.

A set constraint's members are stored, compared, ordered and serialized by
the Rust core (``fhy_core::constraint``), not with Python collection
semantics. ``ConstraintMember`` names the member kinds: the four primitive
Python types, ``Serializable`` leaves that are also ``Hashable``, and
``tuple``/``frozenset`` containers of the same. Validation rejects
everything else, and a float NaN, bare or nested, since NaN is unequal to
itself and could never be matched to a bound value. Members compare
type-strictly, so ``int``, ``float`` and ``bool`` never compare equal even
when they carry the same value, including at the leaves of a nested
``tuple``/``frozenset``. A number whose type subclasses ``int`` or
``float`` is stored as the exact number it denotes, and ``-0.0`` as
``0.0``. Members are kept in one canonical order, the same in every
process: by kind (``bool``, ``float``, ``frozenset``, ``int``, ``str``,
``tuple``, then ``Serializable`` members), then by value, numbers
numerically, strings by code point, containers element by element, and
``Serializable`` members by their type and payload.
"""

__all__ = [
    "ConstraintMember",
    "MemberCollection",
    "does_member_lift_to_expression",
]

from collections.abc import Iterator
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

from fhy_core import _rs
from fhy_core.serialization import Serializable

_ConstraintPrimitive: TypeAlias = str | int | float | bool

ConstraintMember: TypeAlias = (
    _ConstraintPrimitive
    | Serializable
    | tuple["ConstraintMember", ...]
    | frozenset["ConstraintMember"]
)
"""Allowed constraint member kinds.

A constraint member is one of: the four primitive Python types
(``str``, ``int``, ``float``, ``bool``); any ``Serializable`` instance
that is also ``Hashable``; or a tuple or frozenset of valid members.
Members are stored with type-strict equality: ``int``, ``float``, and
``bool`` are not interchangeable, even at the leaves of nested
containers. A number whose type subclasses ``int`` or ``float``, such as
an ``IntEnum`` member or a NumPy ``float64``, is stored as the exact
``int`` or ``float`` it denotes. A ``float`` equal to zero is stored as
positive-signed zero, so a member built from ``-0.0`` is the same member
as one built from ``0.0``.
"""

_MemberT_co = TypeVar("_MemberT_co", covariant=True)


@runtime_checkable
class MemberCollection(Protocol[_MemberT_co]):
    """Read-only collection input for a set constraint's members.

    A structural, immutable-by-contract collection: any ``set``, ``list``,
    ``tuple``, or ``frozenset`` of members satisfies it. Used as the
    constructor-input type for the set constraints so callers can pass any
    of those literals while the stored field is normalized to a
    deduplicated tuple of the members in canonical order. The collection is
    only iterated during construction; the constraint never mutates or
    retains the caller's collection.
    """

    def __iter__(self) -> Iterator[_MemberT_co]: ...

    def __len__(self) -> int: ...

    def __contains__(self, item: object) -> bool: ...


def does_member_lift_to_expression(value: ConstraintMember) -> bool:
    """Return whether a constraint member lifts to a ``LiteralExpression``.

    Answers the question a caller would otherwise have to ask by
    converting a set constraint and catching the failure. A member that
    does not lift cannot take part in the expression a set constraint
    converts to, so a caller lowering constraints to the solver uses this
    to partition the members it can represent from the ones it must drop.

    Args:
        value: Candidate constraint member.

    Returns:
        True for a ``bool``, an ``int``, a ``float``, or a ``Decimal`` a
        literal holds; False for a ``str``, whose literal would compare
        against numbers, and for any other value.

    """
    return _rs.does_member_lift_to_expression(value)
