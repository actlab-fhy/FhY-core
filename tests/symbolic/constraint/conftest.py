"""Shared helpers for the `tests/symbolic/constraint` sub-package."""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, SerializedDict, register_serializable
from fhy_core.symbolic.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.utils.override import override

from ..conftest import (  # re-exported below
    SerializableEqualHashable,
    mock_identifier,
)

__all__ = [
    "ALL_KINDS",
    "SET_KINDS",
    "HashCollidingMember",
    "HashableNotSerializable",
    "SerializableEqualHashable",
    "SerializableHashRaises",
    "UnhashableTuple",
    "build_equation_constraint",
    "build_in_set_constraint",
    "build_not_in_set_constraint",
    "make_serializable_dict",
    "mock_identifier",
]


def build_equation_constraint(variable: Identifier) -> Constraint:
    """Return a default `EquationConstraint` whose scope is ``variable``."""
    return EquationConstraint(IdentifierExpression(variable))


def build_in_set_constraint(variable: Identifier) -> Constraint:
    """Return a default `InSetConstraint` with values ``{1, 2}``."""
    return InSetConstraint(variable, {1, 2})


def build_not_in_set_constraint(variable: Identifier) -> Constraint:
    """Return a default `NotInSetConstraint` with values ``{1, 2}``."""
    return NotInSetConstraint(variable, {1, 2})


SET_KINDS = [
    pytest.param(InSetConstraint, id="in_set"),
    pytest.param(NotInSetConstraint, id="not_in_set"),
]
"""Parametrize list of the two set-constraint factories.

Use this in tests that only need the constraint class. Tests that also
need per-kind expected outcomes (member polarity, str marker, etc.)
should declare their own richer parametrize alongside this one.
"""

ALL_KINDS = [
    pytest.param(build_equation_constraint, id="equation"),
    pytest.param(build_in_set_constraint, id="in_set"),
    pytest.param(build_not_in_set_constraint, id="not_in_set"),
]
"""Parametrize list of single-argument factories for every constraint kind.

Each factory takes one identifier and returns a `Constraint` whose scope
(`get_free_identifiers()`) is exactly that identifier -- for the equation
factory, by wrapping it in a bare `IdentifierExpression`; for the set
factories, because a set constraint's variable is always its scope.
"""


class UnhashableTuple(tuple):  # type: ignore[type-arg]
    """A tuple subclass that explicitly opts out of hashability.

    Setting ``__hash__ = None`` makes instances fail the
    ``isinstance(value, Hashable)`` check in
    ``_validate_constraint_member`` even though the underlying type is
    a ``tuple`` and would otherwise pass the recursive validation pass.
    """

    __hash__ = None  # type: ignore[assignment]


@register_serializable(type_id="tests.constraint.serializable_hash_raises")
class SerializableHashRaises(Serializable):
    """Serializable + nominally-Hashable value whose ``__hash__`` raises.

    Member validation accepts this instance (the ``Hashable`` ABC check
    is structural - it just looks for a non-``None`` ``__hash__``
    attribute), but the explicit per-item ``hash`` call inside
    ``_normalize_constraint_member_collection`` then trips a ``TypeError``
    that gets re-wrapped as a ``ConstraintError``.
    """

    @override
    def __hash__(self) -> int:
        raise TypeError("intentionally unhashable at runtime")

    @override
    def __eq__(self, other: object) -> bool:
        return self is other

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableHashRaises":
        return cls()


class HashableNotSerializable:
    """Hashable value that is intentionally not a `Serializable` subclass.

    Drives the `_is_serializable_hashable` predicate: a value that is
    Hashable but not Serializable must be rejected.
    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HashableNotSerializable) and self._value == other._value
        )


@register_serializable(type_id="tests.constraint.hash_colliding_member")
class HashCollidingMember(Serializable):
    """Serializable member whose instances all share one hash value.

    Every instance hashes to the same number, so any two distinct
    instances collide inside the ``frozenset`` that normalizes a set
    constraint's members. Collisions are the only thing that makes a
    ``frozenset``'s iteration order depend on insertion order, so a pair
    of these members is what lets a test pin down that two constraints
    built from differently ordered inputs really do store their members
    in different orders -- and stay equivalent anyway. Members that do
    not collide land in insertion-independent slots, which canonicalizes
    the stored order and makes such a test unable to fail.

    Distinct ``value``s stay unequal, so both survive normalization.
    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        """Return the value distinguishing this member from its peers."""
        return self._value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, HashCollidingMember) and self._value == other._value

    @override
    def __hash__(self) -> int:
        return 0

    @override
    def __repr__(self) -> str:
        return f"HashCollidingMember({self._value})"

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "HashCollidingMember":
        return cls(value=int(data["value"]))


def make_serializable_dict(type_id: str, data: Any) -> SerializedDict:
    """Return a wrapped serialized dict matching the registry envelope."""
    return {"__type__": type_id, "__data__": data}
