"""Testing utilitiy functions."""

from importlib.util import find_spec
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.utils.override import override

__all__ = [
    "SerializableEqualHashable",
    "mock_identifier",
]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if find_spec("z3") is None:
        skip_z3 = pytest.mark.skip(reason="z3-solver not installed")
        for item in items:
            if "z3" in item.keywords:
                item.add_marker(skip_z3)


def mock_identifier(name_hint: str, identifier_id: int) -> Identifier:
    """Create a mock identifier.

    Args:
        name_hint: Variable name.
        identifier_id: Identifier ID.

    Returns:
        Mock identifier.

    """
    identifier = Mock(spec=Identifier)
    identifier._name_hint = name_hint
    identifier._id = identifier_id
    identifier.name_hint = name_hint
    identifier.id = identifier_id
    # Configure dunder methods via MagicMock's side_effect rather than direct
    # function assignment so mock-library internals own the dunder wiring.
    identifier.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: identifier.id == getattr(other, "id", object())
    )
    identifier.__hash__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda: hash(identifier.id)
    )
    identifier.serialize_to_dict = lambda: {
        "id": identifier.id,
        "name_hint": identifier.name_hint,
    }
    identifier.deserialize_from_dict = lambda data: mock_identifier(
        data["name_hint"], data["id"]
    )
    return identifier


@register_serializable(type_id="tests.serializable_equal_hashable")
class SerializableEqualHashable(Serializable):
    """Serializable value with overridden ``__eq__`` and a real ``__hash__``.

    Used as a generic value-semantics test helper: integer-valued, equal
    by ``_value``, hashable by ``_value``, and round-trippable through
    the serialization registry. Subpackages re-export this from their
    own ``conftest`` so leaf tests can keep importing locally.

    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SerializableEqualHashable) and (
            self._value == other._value
        )

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableEqualHashable":
        return cls(value=int(data["value"]))
