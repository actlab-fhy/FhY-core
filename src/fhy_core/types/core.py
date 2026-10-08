"""Core type system.

Backed by the Rust implementation: the four built-in classes are thin
subclasses of their ``fhy_core._rs`` classes, which hold the Rust type and
the objects each was built from, and compare and hash structurally.
Promotion and literal resolution run in the Rust core, over its promotion
orders. ``CoreDataType`` and ``TypeQualifier`` stay Python enums, converted
by value at the boundary.

``Type`` and ``DataType`` stay open to subclassing: a subclass Python
defines is frozen at the end of its outermost ``__init__``, as
``FrozenMixin`` freezes, and takes part in binding, substitution,
unification and structural equivalence through the handlers it registers on
the dispatchers of :mod:`fhy_core.types.dispatch`.
"""

__all__ = [
    "CoreDataType",
    "DataType",
    "FhYCoreTypeError",
    "IndexType",
    "NumericalType",
    "PrimitiveDataType",
    "TemplateDataType",
    "Type",
    "TypeQualifier",
    "get_core_data_type_bit_width",
    "is_weak_core_data_type",
    "promote_core_data_types",
    "promote_primitive_data_types",
    "promote_type_qualifiers",
    "resolve_literal_core_data_type",
]

from abc import ABC
from enum import StrEnum

from fhy_core import _rs
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.traits import FrozenMixin
from fhy_core.traits.frozen import _FrozenAfterInit

from ..error import register_error

# Imported for its effect: it registers the expression classes, among them
# the literal class the default stride of an `IndexType` is built through.
from ..symbolic.expression import core as _expression_core  # noqa: F401


@register_error
class FhYCoreTypeError(TypeError):
    """Core type error."""


class _DispatchedStructuralEquivalence(
    _FrozenAfterInit, WrappedFamilySerializable, ABC
):
    """Shared base of ``Type`` and ``DataType``.

    Forwards ``is_structurally_equivalent`` to the dispatcher in the sibling
    ``dispatch`` module through a deferred import; the built-in classes
    answer it in Rust.
    """

    __slots__ = ()

    def is_structurally_equivalent(self, other: object) -> bool:
        """Return whether ``other`` is structurally equivalent to this value."""
        from .dispatch import is_structurally_equivalent  # noqa: PLC0415

        return is_structurally_equivalent(self, other)


class Type(_rs.Type, _DispatchedStructuralEquivalence):
    """Abstract compiler type."""


class DataType(_rs.DataType, _DispatchedStructuralEquivalence):
    """Abstract data type."""


FrozenMixin.register(Type)
FrozenMixin.register(DataType)
Type._register_public_class()
DataType._register_public_class()


class CoreDataType(StrEnum):
    """Core data type primitives.

    ``BOOL`` is a fully-concrete one-bit boolean core data type. It does
    not participate in the integer or float/complex promotion lattices
    and does not promote with any other core data type.
    """

    UINT = "uint"
    INT = "int"
    FLOAT = "float"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX32 = "complex32"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    BOOL = "bool"


class TypeQualifier(StrEnum):
    """Type qualifier."""

    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAM = "param"
    TEMP = "temp"


def get_core_data_type_bit_width(core_data_type: CoreDataType) -> int | None:
    """Get the bit width of a core data type.

    Args:
        core_data_type: Core data type.

    Returns:
        Bit width of the core data type, or ``None`` for weak literal types
        that do not yet have a concrete width.

    Raises:
        TypeError: If ``core_data_type`` is not a ``CoreDataType``.

    """
    return _rs.get_core_data_type_bit_width(core_data_type)


def is_weak_core_data_type(core_data_type: CoreDataType) -> bool:
    """Return True when the core data type is a weak literal type."""
    return _rs.is_weak_core_data_type(core_data_type)


def promote_core_data_types(
    core_data_type1: CoreDataType, core_data_type2: CoreDataType
) -> CoreDataType:
    """Promote two core data types to a common type.

    Integers join in the integer promotion order (the unsigned and signed
    chains, the weak ``UINT`` below the weak ``INT``, and each sized
    unsigned type below the signed type twice its width), and floats and
    complex numbers in theirs.

    Args:
        core_data_type1: First core data type.
        core_data_type2: Second core data type.

    Returns:
        Common type to which both core data types can be promoted.

    Raises:
        FhYCoreTypeError: If the promotion is not supported. ``BOOL`` only
            promotes with itself; any other pairing involving ``BOOL`` is
            rejected, and so is a pair across the two families.

    """
    return _rs.promote_core_data_types(core_data_type1, core_data_type2)


def resolve_literal_core_data_type(
    literal: bool | int | float, core_data_type: CoreDataType
) -> CoreDataType:
    """Resolve a weak literal type to a concrete type compatible with the context.

    Integer literals resolve to the narrowest concrete integer type that
    represents the value (consistent with the integer lattice). Floating-
    point literals paired with the weak ``FLOAT`` context resolve to
    ``FLOAT64`` (matching Python's native ``float`` precision); narrower
    concrete float types must be requested explicitly via the context.
    Boolean literals resolve to ``BOOL`` only when paired with a ``BOOL``
    context; any other pairing involving a boolean literal or the ``BOOL``
    context is rejected.

    Args:
        literal: Literal value whose concrete type should be resolved.
        core_data_type: Target or contextual core data type.

    Returns:
        A concrete core data type compatible with both the literal value and
        the requested context.

    Raises:
        FhYCoreTypeError: If the literal cannot be represented in the requested
            type family.
        TypeError: If ``literal`` is not a ``bool``, ``int`` or ``float``.

    """
    return _rs.resolve_literal_core_data_type(literal, core_data_type)


@register_serializable(type_id="primitive_data_type")
class PrimitiveDataType(_rs.PrimitiveDataType, DataType):
    """Primitive data type: a ``CoreDataType`` as a data type.

    Compares and hashes by its core data type.
    """


PrimitiveDataType._register_public_class()


@register_serializable(type_id="template_data_type")
class TemplateDataType(_rs.TemplateDataType, DataType):
    """Template data type: a placeholder named by an ``Identifier``.

    ``widths``, when not ``None``, lists the bit widths of the data types
    the placeholder may be bound to; each must be a positive integer, and
    the list must not be empty. The widths are a set, kept sorted and
    without repeats. Compares and hashes by its identifier and widths.
    """


TemplateDataType._register_public_class()


def promote_primitive_data_types(
    primitive_data_type1: PrimitiveDataType, primitive_data_type2: PrimitiveDataType
) -> PrimitiveDataType:
    """Promote two primitive data types to a common type.

    Args:
        primitive_data_type1: First primitive data type.
        primitive_data_type2: Second primitive data type.

    Returns:
        Common type to which both primitive data types can be promoted.

    Raises:
        FhYCoreTypeError: If the promotion is not supported.
        TypeError: If an argument is not a ``PrimitiveDataType``.
    """
    promoted: PrimitiveDataType = _rs.promote_primitive_data_types(
        primitive_data_type1, primitive_data_type2
    )  # type: ignore[assignment]
    return promoted


@register_serializable(type_id="numerical_type")
class NumericalType(_rs.NumericalType, Type):
    """Numerical multi-dimensional array type; empty shapes indicate scalars.

    Shape elements are normally ``Expression`` values. The literal ``...``
    (``Ellipsis``) is also accepted in shapes used for template binding and
    substitution, with two distinct semantics:

    - A shape of *exactly* ``[...]`` (a single ``Ellipsis``) is a full-shape
      wildcard. When paired with a ``TemplateDataType`` it triggers the
      whole-type binding path in ``bind_template``; otherwise it accepts any
      shape on the actual without recording per-dimension bindings.
    - An ``Ellipsis`` at a specific position within an otherwise-concrete
      shape is a per-dimension wildcard: that single dimension matches any
      ``Expression`` on the actual without binding, while neighbouring
      dimensions are matched normally and the rank still has to agree.

    Numerical types whose shape contains ``Ellipsis`` round-trip through
    serialization. Each ``Ellipsis`` is encoded as a sentinel dictionary in
    place of an ``Expression``-serialized dict and is restored to
    ``Ellipsis`` on deserialization. ``==`` and ``hash`` are structural.
    """


NumericalType._register_public_class()


@register_serializable(type_id="index_type")
class IndexType(_rs.IndexType, Type):
    """Index type: a range of a lower bound, an upper bound and a stride.

    Notes:
        - Similar to a python slice or range(start, stop, step)
        - The stride defaults to the literal ``1``.
        - ``==`` and ``hash`` are structural.

    """


IndexType._register_public_class()


def promote_type_qualifiers(
    type_qualifier1: TypeQualifier, type_qualifier2: TypeQualifier
) -> TypeQualifier:
    """Promote two type qualifiers to a common type qualifier.

    Args:
        type_qualifier1: First type qualifier.
        type_qualifier2: Second type qualifier.

    Returns:
        ``PARAM`` when both are ``PARAM``, and ``TEMP`` otherwise.

    """
    return _rs.promote_type_qualifiers(type_qualifier1, type_qualifier2)
