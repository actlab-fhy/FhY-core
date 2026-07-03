"""Shared helpers for the `tests/param` sub-package."""

from typing import Any

import pytest

from fhy_core.param import (
    Param,
    create_categorical_param,
    create_integer_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
)
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.traits import OrderableMixin
from fhy_core.utils.override import override

from ..conftest import (  # re-exported below
    SerializableEqualHashable,
    mock_identifier,
)

__all__ = [
    "SerializableEqualHashable",
    "SerializableEqualNoOrder",
    "SerializableHashOnly",
    "SerializableNonComparable",
    "SerializableOrderableInherited",
    "SerializableOrderableSelf",
    "SerializableOrderableTrait",
    "assert_all_satisfied",
    "assert_none_satisfied",
    "categorical_param_abc",
    "default_int_param",
    "default_real_param",
    "mock_identifier",
    "ordinal_param_123",
    "perm_param_nchw",
]


@register_serializable(type_id="tests.param.serializable_hash_only")
class SerializableHashOnly(Serializable):
    """Serializable value that is hashable but has identity equality semantics."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableHashOnly":
        return cls(value=int(data["value"]))


@register_serializable(type_id="tests.param.serializable_equal_no_order")
class SerializableEqualNoOrder(Serializable):
    """Serializable value with equality semantics but no ordering semantics."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SerializableEqualNoOrder) and (
            self._value == other._value
        )

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableEqualNoOrder":
        return cls(value=int(data["value"]))


class _OrderableBase(Serializable):
    """Base class that defines ``__lt__`` so subclasses inherit ordering."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    def __lt__(self, other: object) -> bool:
        return isinstance(other, _OrderableBase) and self._value < other._value

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OrderableBase) and self._value == other._value

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "_OrderableBase":
        return cls(value=int(data["value"]))


@register_serializable(type_id="tests.param.serializable_orderable_inherited")
class SerializableOrderableInherited(_OrderableBase):
    """Serializable value whose ``__lt__`` is inherited from a parent class.

    The leaf class itself does not define ``__lt__``; an MRO walk must look
    past ``__mro__[0]`` to discover ordering semantics.

    """


@register_serializable(type_id="tests.param.serializable_orderable_self")
class SerializableOrderableSelf(Serializable):
    """Serializable value whose ``__lt__`` is defined on the class itself."""

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    def __lt__(self, other: object) -> bool:
        return (
            isinstance(other, SerializableOrderableSelf) and self._value < other._value
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SerializableOrderableSelf) and self._value == other._value
        )

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableOrderableSelf":
        return cls(value=int(data["value"]))


@register_serializable(type_id="tests.param.serializable_orderable_trait")
class SerializableOrderableTrait(OrderableMixin, Serializable):
    """Serializable value that satisfies the `Orderable` runtime protocol.

    Used to exercise the `supports_orderable_value_semantics` early-return
    that consults ``value.supports_ordering`` for `Orderable`-trait values.
    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def __lt__(self, other: object) -> bool:
        return (
            isinstance(other, SerializableOrderableTrait) and self._value < other._value
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SerializableOrderableTrait)
            and self._value == other._value
        )

    @override
    def __hash__(self) -> int:
        return hash(self._value)

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(
        cls, data: dict[str, Any]
    ) -> "SerializableOrderableTrait":
        return cls(value=int(data["value"]))


@register_serializable(type_id="tests.param.serializable_non_comparable")
class SerializableNonComparable(Serializable):
    """Serializable value with neither equal nor orderable semantics.

    Inherits ``object.__eq__`` (identity), has no ``__lt__``, and uses the
    default identity hash. Useful for asserting the wrapped-leaf admissibility
    checks reject values that lack the required value semantics.

    """

    _value: int

    def __init__(self, value: int) -> None:
        self._value = value

    @override
    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self._value}

    @classmethod
    @override
    def deserialize_from_dict(cls, data: dict[str, Any]) -> "SerializableNonComparable":
        return cls(value=int(data["value"]))


def assert_all_satisfied(param: Param[Any], values: list[Any]) -> None:
    """Assert every value in ``values`` satisfies ``param``'s constraints."""
    for v in values:
        assert param.is_constraints_satisfied(v), (
            f"Value {v} should satisfy constraints of parameter {param}"
        )


def assert_none_satisfied(param: Param[Any], values: list[Any]) -> None:
    """Assert no value in ``values`` satisfies ``param``'s constraints."""
    for v in values:
        assert not param.is_constraints_satisfied(v), (
            f"Value {v} should not satisfy constraints of parameter {param}"
        )


@pytest.fixture
def default_real_param() -> Param[str | float]:
    return create_real_param()


@pytest.fixture
def default_int_param() -> Param[int]:
    return create_integer_param()


@pytest.fixture
def ordinal_param_123() -> Param[int]:
    return create_ordinal_param([1, 2, 3])


@pytest.fixture
def categorical_param_abc() -> Param[str]:
    return create_categorical_param({"a", "b", "c"})


@pytest.fixture
def perm_param_nchw() -> Param[tuple[str, ...]]:
    return create_permutation_param(["n", "c", "h", "w"])
