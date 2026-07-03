"""Value-type semantics for parameter domains.

Defines the leaf value types a parameter may range over, the structural
protocols those values must satisfy to be usable as ordinal, categorical, or
permutation members, and the predicate helpers that classify and compare them.

The classification rules are deliberately type-strict: ``bool``, ``int``, and
``float`` are mutually disjoint value kinds, and ``str`` is distinct from all of
them. Although ``True == 1`` and ``1 == 1.0`` in Python, a value set built from
booleans never matches one built from integers, and integers never match floats.
"""

from collections.abc import Callable, Collection, Sequence
from typing import Any, Protocol, TypeAlias, TypeGuard, TypeVar, cast

from fhy_core.error import register_error
from fhy_core.serialization import (
    DeserializationValueError,
    Serializable,
    SerializedDict,
    deserialize_registry_wrapped_value,
    is_serialized_dict,
    serialize_registry_wrapped_value,
)
from fhy_core.traits import Equal, Orderable

__all__ = [
    "CategoricalValue",
    "OrdinalValue",
    "ParamError",
    "PermutationMemberValue",
    "SerializableEqualValue",
    "SerializableOrderableValue",
]


@register_error
class ParamError(ValueError):
    """Domain error for parameter construction, validation, and assignment.

    Subclasses ``ValueError``, so call sites that catch ``ValueError`` also
    catch this error.
    """


class _SerializableValueLike(Protocol):
    """Structural instance-side serialization contract for param values."""

    @classmethod
    def get_serialization_class_type_id(cls) -> str: ...

    def serialize_to_dict(self) -> SerializedDict: ...


class SerializableEqualValue(Equal, _SerializableValueLike, Protocol):
    """Value with equality semantics and serializable instance behavior."""


class SerializableOrderableValue(Orderable, _SerializableValueLike, Protocol):
    """Value with ordering semantics and serializable instance behavior."""


# CategoricalValue omits `float`: floating-point equality is unreliable for
# category membership.
CategoricalValue: TypeAlias = bool | int | str | SerializableEqualValue
OrdinalValue: TypeAlias = bool | int | float | str | SerializableOrderableValue
PermutationMemberValue: TypeAlias = bool | int | float | str | SerializableEqualValue

_CategoricalValueT = TypeVar("_CategoricalValueT", bound=CategoricalValue)
_OrdinalValueT = TypeVar("_OrdinalValueT", bound=OrdinalValue)
_PermutationMemberValueT = TypeVar(
    "_PermutationMemberValueT", bound=PermutationMemberValue
)
_ParamValueT = TypeVar("_ParamValueT")


def is_sequence_unique_without_set(values: Sequence[Any]) -> bool:
    """Return whether ``values`` has no two strictly matching elements.

    Uses pairwise comparison through :func:`do_param_values_match` rather than a
    ``set`` so that values which are unhashable or only define ``__eq__`` are
    still handled and the numeric-kind distinction is honored (``1`` and ``1.0``
    are treated as distinct).
    """
    for i, value_1 in enumerate(values):
        for value_2 in values[i + 1 :]:
            if do_param_values_match(value_1, value_2):
                return False
    return True


def is_sorted_sequence_unique(values: Sequence[Any]) -> bool:
    """Return whether a pre-sorted ``values`` has no adjacent strictly matching pair.

    Adjacent values are compared through :func:`do_param_values_match`, so a pair
    such as ``1`` and ``1.0`` that sorts adjacently is treated as distinct.
    """
    return not any(
        do_param_values_match(values[i], values[i + 1]) for i in range(len(values) - 1)
    )


def supports_equal_value_semantics(value: Any) -> bool:
    """Return whether ``value`` defines usable equality (``__eq__`` + ``__hash__``)."""
    if isinstance(value, Equal):
        return value.supports_equality
    value_type = type(value)
    return (
        getattr(value_type, "__eq__", object.__eq__) is not object.__eq__
        and getattr(value_type, "__hash__", None) is not None
    )


def supports_orderable_value_semantics(value: Any) -> bool:
    """Return whether ``value`` defines a usable total order (``__lt__``)."""
    if isinstance(value, Orderable):
        return value.supports_ordering
    return any("__lt__" in cls.__dict__ for cls in type(value).__mro__[:-1])


def is_categorical_value(value: Any) -> TypeGuard[CategoricalValue]:
    """Return whether ``value`` is admissible as a categorical member."""
    return isinstance(value, (bool, int, str)) or (
        isinstance(value, Serializable) and supports_equal_value_semantics(value)
    )


def is_ordinal_value(value: Any) -> TypeGuard[OrdinalValue]:
    """Return whether ``value`` is admissible as an ordinal member."""
    return isinstance(value, (bool, int, float, str)) or (
        isinstance(value, Serializable) and supports_orderable_value_semantics(value)
    )


def is_permutation_member_value(value: Any) -> TypeGuard[PermutationMemberValue]:
    """Return whether ``value`` is admissible as a permutation member."""
    return isinstance(value, (bool, int, float, str)) or (
        isinstance(value, Serializable) and supports_equal_value_semantics(value)
    )


def _get_numeric_value_kind(value: Any) -> str | None:
    """Return ``"bool"``, ``"int"``, or ``"float"`` for a numeric ``value``.

    Returns ``None`` for any value that is not one of these three numeric kinds,
    so that non-numeric values (such as ``str`` or ``Serializable``) are compared
    by plain equality without a numeric-kind gate.
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return None


def _has_numeric_kind_mismatch(value_1: Any, value_2: Any) -> bool:
    kind_1 = _get_numeric_value_kind(value_1)
    kind_2 = _get_numeric_value_kind(value_2)
    if kind_1 is None and kind_2 is None:
        return False
    return kind_1 != kind_2


def do_param_values_match(candidate: Any, allowed_value: Any) -> bool:
    """Return whether ``candidate`` equals ``allowed_value`` with strict kinds.

    ``bool``, ``int``, and ``float`` are mutually disjoint value kinds: ``True``
    and ``1`` are reported as distinct despite ``True == 1``, and ``1`` and
    ``1.0`` are distinct despite ``1 == 1.0``, so a value set of one kind never
    silently matches one of another kind.
    """
    if _has_numeric_kind_mismatch(candidate, allowed_value):
        return False
    return cast(bool, candidate == allowed_value)


def does_collection_contain_param_value(
    allowed_values: Collection[Any] | Sequence[Any], candidate: Any
) -> bool:
    """Return whether ``candidate`` matches any element of ``allowed_values``."""
    return any(
        do_param_values_match(candidate, allowed_value)
        for allowed_value in allowed_values
    )


def serialize_wrapped_leaf_value(value: object) -> SerializedDict:
    """Serialize a validated leaf value through the wrapped registry.

    Args:
        value: The leaf value to serialize.

    Returns:
        The registry-wrapped serialized form.

    Raises:
        ParamError: If ``value`` is not a serializable leaf value.

    """
    # Param generics are stricter than the wrapped-registry alias, so validate
    # the runtime shape before handing the value to the shared serializer.
    if not isinstance(value, (bool, int, float, str, Serializable)):
        raise ParamError(
            "Parameter values must be serializable leaf values "
            "(bool/int/float/str/Serializable)."
        )
    return serialize_registry_wrapped_value(cast(Any, value))


def deserialize_wrapped_leaf_values(
    owner_cls: type[Any],
    serialized_values: Sequence[Any],
    value_type_guard: Callable[[Any], TypeGuard[_ParamValueT]],
    expected_description: str,
) -> list[_ParamValueT]:
    """Deserialize and validate a list of registry-wrapped leaf values.

    Args:
        owner_cls: The class used for error reporting context.
        serialized_values: The serialized wrapped values to deserialize.
        value_type_guard: Predicate that validates each deserialized value.
        expected_description: Human-readable description of the expected values
            for error messages.

    Returns:
        The validated, deserialized values.

    Raises:
        DeserializationValueError: If any entry is malformed or fails the guard.

    """
    if not all(is_serialized_dict(value) for value in serialized_values):
        raise DeserializationValueError(
            owner_cls,
            "possible_values",
            expected_description,
            serialized_values,
        )
    wrapped_values = [
        deserialize_registry_wrapped_value(value) for value in serialized_values
    ]
    typed_values: list[_ParamValueT] = []
    for value in wrapped_values:
        if not value_type_guard(value):
            raise DeserializationValueError(
                owner_cls,
                "possible_values",
                expected_description,
                value,
            )
        typed_values.append(value)
    return typed_values
