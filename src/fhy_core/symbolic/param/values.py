"""Value-type semantics for parameter domains.

Defines the leaf value types a parameter may range over, the structural
protocols those values must satisfy to be usable as ordinal, categorical, or
permutation members, and ``ParamError``.

The classification rules are deliberately type-strict: ``bool``, ``int``, and
``float`` are mutually disjoint value kinds, and ``str`` is distinct from all of
them. Although ``True == 1`` and ``1 == 1.0`` in Python, a value set built from
booleans never matches one built from integers, and integers never match floats.
The rules run on the Rust core (``fhy_core::param``), where a domain's values
are the constraint core's members and a ``Serializable`` value is compared,
and ordered, by its own ``==`` and ``<``.
"""

from typing import Protocol, TypeAlias, TypeVar

from fhy_core.error import register_error
from fhy_core.serialization import SerializedDict
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
