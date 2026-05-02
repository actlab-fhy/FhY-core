"""Shared helpers for the `tests/constraint` sub-package."""

from typing import Any

from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import LiteralExpression
from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, SerializedDict, register_serializable

from ..conftest import (  # noqa: F401  # re-exported below
    SerializableEqualHashable,
    mock_identifier,
)

__all__ = [
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
    """Return a default `EquationConstraint` over ``variable``."""
    return EquationConstraint(variable, LiteralExpression(True))


def build_in_set_constraint(variable: Identifier) -> Constraint:
    """Return a default `InSetConstraint` with values ``{1, 2}``."""
    return InSetConstraint(variable, {1, 2})


def build_not_in_set_constraint(variable: Identifier) -> Constraint:
    """Return a default `NotInSetConstraint` with values ``{1, 2}``."""
    return NotInSetConstraint(variable, {1, 2})


class UnhashableTuple(tuple):  # type: ignore[type-arg]
    """A tuple subclass that explicitly opts out of hashability.

    Setting ``__hash__ = None`` makes instances fail the
    ``isinstance(value, Hashable)`` check on line 138 of constraint.py
    even though the underlying type is a ``tuple`` and would otherwise
    pass the recursive validation pass.
    """

    __hash__ = None  # type: ignore[assignment]


@register_serializable(type_id="tests.constraint.serializable_hash_raises")
class SerializableHashRaises(Serializable):
    """Serializable + nominally-Hashable value whose ``__hash__`` raises.

    Member validation accepts this instance (the ``Hashable`` ABC check
    is structural — it just looks for a non-``None`` ``__hash__``
    attribute), but the subsequent ``frozenset(values)`` call inside
    ``_normalize_constraint_member_collection`` invokes ``hash`` and
    trips the defensive ``except TypeError`` on lines 164-165.
    """

    def __hash__(self) -> int:
        raise TypeError("intentionally unhashable at runtime")

    def __eq__(self, other: object) -> bool:
        return self is other

    def serialize_to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableHashRaises":
        return cls()


class HashableNotSerializable:
    """Hashable value that is intentionally not a `Serializable` subclass.

    Drives the `_is_serializable_hashable` mutation that flips ``and`` to
    ``or``: the mutant accepts anything that is `Hashable` even when it is
    not `Serializable`, so a successful construction would indicate the
    relaxed validator is in effect.
    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, HashableNotSerializable) and self._value == other._value
        )


def make_serializable_dict(type_id: str, data: Any) -> SerializedDict:
    """Return a wrapped serialized dict matching the registry envelope."""
    return {"__type__": type_id, "__data__": data}
