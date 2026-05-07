"""Core type system."""

__all__ = [
    "CoreDataType",
    "DataType",
    "FhYCoreTypeError",
    "IndexType",
    "is_weak_core_data_type",
    "NumericalType",
    "PrimitiveDataType",
    "promote_core_data_types",
    "promote_primitive_data_types",
    "promote_type_qualifiers",
    "resolve_literal_core_data_type",
    "TemplateDataType",
    "Type",
    "TypeQualifier",
]

from abc import ABC
from collections.abc import Sequence
from types import EllipsisType
from typing import TypedDict, TypeGuard

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
    WrappedFamilySerializable,
    is_serialized_dict,
    register_serializable,
)
from fhy_core.trait import FrozenMixin, StructuralEquivalenceMixin

from ..error import register_error
from ..expression.core import Expression, LiteralExpression
from ..expression.pprint import pformat_expression
from ..identifier import Identifier
from ..lattice import Lattice
from ..utils import StrEnum, format_comma_separated_list


class Type(WrappedFamilySerializable, FrozenMixin, StructuralEquivalenceMixin, ABC):
    """Abstract compiler type."""

    def is_structurally_equivalent(self, other: object) -> bool:
        # Deferred import: ``dispatch`` registers handlers against
        # ``is_structurally_equivalent`` for the concrete ``Type`` subclasses
        # defined in this module, so importing it at the module top would
        # create a cycle.
        from .dispatch import is_structurally_equivalent  # noqa: PLC0415

        return is_structurally_equivalent(self, other)


@register_error
class FhYCoreTypeError(TypeError):
    """Core type error."""


class DataType(WrappedFamilySerializable, FrozenMixin, StructuralEquivalenceMixin, ABC):
    """Abstract data type."""

    def is_structurally_equivalent(self, other: object) -> bool:
        # Deferred import for the same reason as
        # ``Type.is_structurally_equivalent``.
        from .dispatch import is_structurally_equivalent  # noqa: PLC0415

        return is_structurally_equivalent(self, other)


class CoreDataType(StrEnum):
    """Core data type primitives."""

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


def get_core_data_type_bit_width(core_data_type: CoreDataType) -> int | None:
    """Get the bit width of a core data type.

    Args:
        core_data_type: Core data type.

    Returns:
        Bit width of the core data type, or ``None`` for weak literal types
        that do not yet have a concrete width.

    """
    match core_data_type:
        case CoreDataType.UINT | CoreDataType.INT | CoreDataType.FLOAT:
            return None
        case CoreDataType.UINT8 | CoreDataType.INT8:
            return 8
        case CoreDataType.UINT16 | CoreDataType.INT16 | CoreDataType.FLOAT16:
            return 16
        case (
            CoreDataType.UINT32
            | CoreDataType.INT32
            | CoreDataType.FLOAT32
            | CoreDataType.COMPLEX32
        ):
            return 32
        case CoreDataType.INT64 | CoreDataType.FLOAT64 | CoreDataType.COMPLEX64:
            return 64
        case CoreDataType.COMPLEX128:
            return 128


def is_weak_core_data_type(core_data_type: CoreDataType) -> bool:
    """Return True when the core data type is a weak literal type."""
    return core_data_type in {
        CoreDataType.UINT,
        CoreDataType.INT,
        CoreDataType.FLOAT,
    }


def _define_integer_data_type_lattice() -> Lattice[CoreDataType]:
    """Define the unified lattice over unsigned and signed integer types.

    The lattice links the unsigned chain (UINT < UINT8 < ... < UINT32) and the
    signed chain (INT < INT8 < ... < INT64) so that any UINT/INT pair has a
    well-defined promotion target:

      - UINT -> INT (weak promotion of weak unsigned to weak signed)
      - UINT_N -> INT_{2N} for N in {8, 16, 32} (concrete unsigned promotes to
        the next-wider signed type that can represent all of its values)

    """
    lattice = Lattice[CoreDataType]()
    for element in (
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.INT,
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    ):
        lattice.add_element(element)

    lattice.add_order(CoreDataType.UINT, CoreDataType.UINT8)
    lattice.add_order(CoreDataType.UINT8, CoreDataType.UINT16)
    lattice.add_order(CoreDataType.UINT16, CoreDataType.UINT32)

    lattice.add_order(CoreDataType.INT, CoreDataType.INT8)
    lattice.add_order(CoreDataType.INT8, CoreDataType.INT16)
    lattice.add_order(CoreDataType.INT16, CoreDataType.INT32)
    lattice.add_order(CoreDataType.INT32, CoreDataType.INT64)

    lattice.add_order(CoreDataType.UINT, CoreDataType.INT)
    lattice.add_order(CoreDataType.UINT8, CoreDataType.INT16)
    lattice.add_order(CoreDataType.UINT16, CoreDataType.INT32)
    lattice.add_order(CoreDataType.UINT32, CoreDataType.INT64)

    lattice.verify()
    return lattice


def _define_float_complex_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.FLOAT)
    lattice.add_element(CoreDataType.FLOAT16)
    lattice.add_element(CoreDataType.FLOAT32)
    lattice.add_element(CoreDataType.FLOAT64)
    lattice.add_element(CoreDataType.COMPLEX32)
    lattice.add_element(CoreDataType.COMPLEX64)
    lattice.add_element(CoreDataType.COMPLEX128)

    lattice.add_order(CoreDataType.FLOAT, CoreDataType.FLOAT16)
    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.FLOAT32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.FLOAT64)
    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.COMPLEX32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.FLOAT64, CoreDataType.COMPLEX128)
    lattice.add_order(CoreDataType.COMPLEX32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.COMPLEX64, CoreDataType.COMPLEX128)

    lattice.verify()
    return lattice


_INTEGER_DATA_TYPE_LATTICE = _define_integer_data_type_lattice()
_FLOAT_COMPLEX_DATA_TYPE_LATTICE = _define_float_complex_data_type_lattice()


def promote_core_data_types(
    core_data_type1: CoreDataType, core_data_type2: CoreDataType
) -> CoreDataType:
    """Promote two core data types to a common type.

    Args:
        core_data_type1: First core data type.
        core_data_type2: Second core data type.

    Returns:
        Common type to which both core data types can be promoted.

    Raises:
        FhYTypeError: If the promotion is not supported.

    """
    _INTEGER_DATA_TYPES = {
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.INT,
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
    _FLOAT_COMPLEX_DATA_TYPES = {
        CoreDataType.FLOAT,
        CoreDataType.FLOAT16,
        CoreDataType.FLOAT32,
        CoreDataType.FLOAT64,
        CoreDataType.COMPLEX32,
        CoreDataType.COMPLEX64,
        CoreDataType.COMPLEX128,
    }

    if (
        core_data_type1 in _INTEGER_DATA_TYPES
        and core_data_type2 in _INTEGER_DATA_TYPES
    ):
        return _INTEGER_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    elif (
        core_data_type1 in _FLOAT_COMPLEX_DATA_TYPES
        and core_data_type2 in _FLOAT_COMPLEX_DATA_TYPES
    ):
        return _FLOAT_COMPLEX_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    else:
        raise FhYCoreTypeError(
            "Unsupported primitive data type promotion: "
            f"{core_data_type1}, {core_data_type2}"
        )


def _get_smallest_uint_core_data_type(literal: int) -> CoreDataType:
    for core_data_type, upper_bound in (
        (CoreDataType.UINT8, 2**8 - 1),
        (CoreDataType.UINT16, 2**16 - 1),
        (CoreDataType.UINT32, 2**32 - 1),
    ):
        if literal <= upper_bound:
            return core_data_type
    raise FhYCoreTypeError(f"Literal {literal} does not fit in a supported uint type.")


def _get_smallest_int_core_data_type(literal: int) -> CoreDataType:
    for core_data_type, lower_bound, upper_bound in (
        (CoreDataType.INT8, -(2**7), 2**7 - 1),
        (CoreDataType.INT16, -(2**15), 2**15 - 1),
        (CoreDataType.INT32, -(2**31), 2**31 - 1),
        (CoreDataType.INT64, -(2**63), 2**63 - 1),
    ):
        if lower_bound <= literal <= upper_bound:
            return core_data_type
    raise FhYCoreTypeError(f"Literal {literal} does not fit in a supported int type.")


def resolve_literal_core_data_type(
    literal: int | float, core_data_type: CoreDataType
) -> CoreDataType:
    """Resolve a weak literal type to the narrowest compatible concrete type.

    Args:
        literal: Literal value whose concrete type should be resolved.
        core_data_type: Target or contextual core data type.

    Returns:
        A concrete core data type compatible with both the literal value and
        the requested context.

    Raises:
        FhYCoreTypeError: If the literal cannot be represented in the requested
            type family.

    """
    if isinstance(literal, bool):
        raise NotImplementedError("Boolean literals are not yet supported.")
    elif isinstance(literal, float):
        if core_data_type in {
            CoreDataType.FLOAT,
            CoreDataType.FLOAT16,
            CoreDataType.FLOAT32,
            CoreDataType.FLOAT64,
            CoreDataType.COMPLEX32,
            CoreDataType.COMPLEX64,
            CoreDataType.COMPLEX128,
        }:
            return (
                CoreDataType.FLOAT16
                if core_data_type == CoreDataType.FLOAT
                else core_data_type
            )
        else:
            raise FhYCoreTypeError(
                f"Float literal {literal} is incompatible with {core_data_type}."
            )
    elif literal >= 0:
        minimal_uint = _get_smallest_uint_core_data_type(literal)
        if core_data_type in {
            CoreDataType.UINT,
            CoreDataType.UINT8,
            CoreDataType.UINT16,
            CoreDataType.UINT32,
        }:
            return promote_core_data_types(
                minimal_uint,
                CoreDataType.UINT8
                if core_data_type == CoreDataType.UINT
                else core_data_type,
            )
        elif core_data_type in {
            CoreDataType.INT,
            CoreDataType.INT8,
            CoreDataType.INT16,
            CoreDataType.INT32,
            CoreDataType.INT64,
        }:
            minimal_int = _get_smallest_int_core_data_type(literal)
            return promote_core_data_types(
                minimal_int,
                CoreDataType.INT8
                if core_data_type == CoreDataType.INT
                else core_data_type,
            )
        elif core_data_type in {
            CoreDataType.FLOAT,
            CoreDataType.FLOAT16,
            CoreDataType.FLOAT32,
            CoreDataType.FLOAT64,
            CoreDataType.COMPLEX32,
            CoreDataType.COMPLEX64,
            CoreDataType.COMPLEX128,
        }:
            return (
                CoreDataType.FLOAT16
                if core_data_type == CoreDataType.FLOAT
                else core_data_type
            )
    else:
        minimal_int = _get_smallest_int_core_data_type(literal)
        if core_data_type in {
            CoreDataType.INT,
            CoreDataType.INT8,
            CoreDataType.INT16,
            CoreDataType.INT32,
            CoreDataType.INT64,
        }:
            return promote_core_data_types(
                minimal_int,
                CoreDataType.INT8
                if core_data_type == CoreDataType.INT
                else core_data_type,
            )
        elif core_data_type in {
            CoreDataType.FLOAT,
            CoreDataType.FLOAT16,
            CoreDataType.FLOAT32,
            CoreDataType.FLOAT64,
            CoreDataType.COMPLEX32,
            CoreDataType.COMPLEX64,
            CoreDataType.COMPLEX128,
        }:
            return (
                CoreDataType.FLOAT16
                if core_data_type == CoreDataType.FLOAT
                else core_data_type
            )

    raise FhYCoreTypeError(f"Literal {literal} is incompatible with {core_data_type}.")


class _PrimitiveDataTypeData(TypedDict):
    core_data_type: str


def _is_valid_primitive_data_type_data(
    data: SerializedDict,
) -> TypeGuard[_PrimitiveDataTypeData]:
    return "core_data_type" in data and isinstance(data["core_data_type"], str)


@register_serializable(type_id="primitive_data_type")
class PrimitiveDataType(DataType):
    """Primitive data type."""

    _core_data_type: CoreDataType

    def __init__(self, core_data_type: CoreDataType) -> None:
        self._core_data_type = core_data_type
        self.freeze(deep=True)

    @property
    def core_data_type(self) -> CoreDataType:
        return self._core_data_type

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"core_data_type": self._core_data_type.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "PrimitiveDataType":
        if not _is_valid_primitive_data_type_data(data):
            raise DeserializationDictStructureError(
                cls, _PrimitiveDataTypeData.__annotations__, data
            )
        if data["core_data_type"] not in CoreDataType._value2member_map_:
            raise DeserializationValueError(
                cls, "core_data_type", "a valid core data type", data["core_data_type"]
            )
        return cls(CoreDataType(data["core_data_type"]))

    def __str__(self) -> str:
        return str(self._core_data_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._core_data_type)})"


class _TemplateDataTypeData(TypedDict):
    data_type: SerializedDict
    widths: list[int] | None


def _is_valid_template_data_type_data(
    data: SerializedDict,
) -> TypeGuard[_TemplateDataTypeData]:
    return (
        "data_type" in data
        and is_serialized_dict(data["data_type"])
        and "widths" in data
        and (isinstance(data["widths"], list) or data["widths"] is None)
        and (
            data["widths"] is None
            or all(isinstance(width, int) for width in data["widths"])
        )
    )


@register_serializable(type_id="template_data_type")
class TemplateDataType(DataType):
    """Template data type."""

    _data_type: Identifier
    _widths: tuple[int, ...] | None

    def __init__(
        self, data_type: Identifier, widths: Sequence[int] | None = None
    ) -> None:
        self._data_type = data_type
        self._widths = tuple(widths) if widths is not None else None
        self.freeze(deep=True)

    @property
    def data_type(self) -> Identifier:
        return self._data_type

    @property
    def widths(self) -> list[int] | None:
        return list(self._widths) if self._widths is not None else None

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "data_type": self._data_type.serialize_to_dict(),
            "widths": list(self._widths) if self._widths is not None else None,
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "TemplateDataType":
        if not _is_valid_template_data_type_data(data):
            raise DeserializationDictStructureError(
                cls, _TemplateDataTypeData.__annotations__, data
            )
        if data["widths"] is not None and any(width <= 0 for width in data["widths"]):
            raise DeserializationValueError(
                cls, "widths", "a list of positive integers or None", data["widths"]
            )
        return cls(
            Identifier.deserialize_from_dict(data["data_type"]),
            data["widths"],
        )

    def __str__(self) -> str:
        return str(self._data_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._data_type)})"


def promote_primitive_data_types(
    primitive_data_type1: PrimitiveDataType, primitive_data_type2: PrimitiveDataType
) -> PrimitiveDataType:
    """Promote two primitive data types to a common type.

    Args:
        primitive_data_type1 (DataType): First primitive data type.
        primitive_data_type2 (DataType): Second primitive data type.

    Returns:
        DataType: Common type to which both primitive data types can be promoted.

    Raises:
        FhYTypeError: If the promotion is not supported.

    """
    return PrimitiveDataType(
        promote_core_data_types(
            primitive_data_type1.core_data_type, primitive_data_type2.core_data_type
        )
    )


class _NumericalTypeData(TypedDict):
    data_type: SerializedDict
    shape: list[SerializedDict]


def _is_valid_numerical_type_data(
    data: SerializedDict,
) -> TypeGuard[_NumericalTypeData]:
    return (
        "data_type" in data
        and is_serialized_dict(data["data_type"])
        and "shape" in data
        and isinstance(data["shape"], list)
        and all(is_serialized_dict(dimension_dict) for dimension_dict in data["shape"])
    )


def _format_numerical_shape_dimension(dimension: Expression | EllipsisType) -> str:
    if dimension is Ellipsis:
        return "..."
    else:
        return pformat_expression(dimension, show_id=True)


@register_serializable(type_id="numerical_type")
class NumericalType(Type):
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

    Numerical types whose shape contains ``Ellipsis`` cannot be serialized.
    """

    _data_type: DataType
    _shape: tuple[Expression | EllipsisType, ...]

    def __init__(
        self,
        data_type: DataType,
        shape: Sequence[Expression | EllipsisType] | None = None,
    ) -> None:
        super().__init__()
        self._data_type = data_type
        self._shape = tuple(shape) if shape is not None else ()
        self.freeze(deep=True)

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> list[Expression | EllipsisType]:
        return list(self._shape)

    def is_scalar(self) -> bool:
        """Return True when the numerical type is a scalar."""
        return not self._shape

    def serialize_data_to_dict(self) -> SerializedDict:
        serialized_shape: list[SerializedDict] = []
        for dimension in self._shape:
            if dimension is Ellipsis:
                raise TypeError(
                    "Cannot serialize a NumericalType whose shape contains "
                    "the wildcard `...`."
                )
            serialized_shape.append(dimension.serialize_to_dict())
        return {
            "data_type": self._data_type.serialize_to_dict(),
            "shape": serialized_shape,
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "NumericalType":
        if not _is_valid_numerical_type_data(data):
            raise DeserializationDictStructureError(
                cls, _NumericalTypeData.__annotations__, data
            )
        return cls(
            DataType.deserialize_from_dict(data["data_type"]),
            [
                Expression.deserialize_from_dict(dimension_dict)
                for dimension_dict in data["shape"]
            ],
        )

    def __str__(self) -> str:
        shape_string = format_comma_separated_list(
            self._shape, str_func=_format_numerical_shape_dimension
        )
        return f"{self._data_type}[{shape_string}]"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self._data_type)}, {repr(self._shape)})"
        )


class _IndexTypeData(TypedDict):
    lower_bound: SerializedDict
    upper_bound: SerializedDict
    stride: SerializedDict


def _is_valid_index_type_data(data: SerializedDict) -> TypeGuard[_IndexTypeData]:
    return (
        "lower_bound" in data
        and is_serialized_dict(data["lower_bound"])
        and "upper_bound" in data
        and is_serialized_dict(data["upper_bound"])
        and "stride" in data
        and is_serialized_dict(data["stride"])
    )


@register_serializable(type_id="index_type")
class IndexType(Type):
    """Index type.

    Notes:
        - Similar to a python slice or range(start, stop, step)

    """

    _lower_bound: Expression
    _upper_bound: Expression
    _stride: Expression

    def __init__(
        self,
        lower_bound: Expression,
        upper_bound: Expression,
        stride: Expression | None = None,
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._stride = stride if stride is not None else LiteralExpression(1)
        self.freeze(deep=True)

    @property
    def lower_bound(self) -> Expression:
        return self._lower_bound

    @property
    def upper_bound(self) -> Expression:
        return self._upper_bound

    @property
    def stride(self) -> Expression:
        return self._stride

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "lower_bound": self._lower_bound.serialize_to_dict(),
            "upper_bound": self._upper_bound.serialize_to_dict(),
            "stride": self._stride.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "IndexType":
        if not _is_valid_index_type_data(data):
            raise DeserializationDictStructureError(
                cls, _IndexTypeData.__annotations__, data
            )
        return cls(
            Expression.deserialize_from_dict(data["lower_bound"]),
            Expression.deserialize_from_dict(data["upper_bound"]),
            Expression.deserialize_from_dict(data["stride"]),
        )

    def __str__(self) -> str:
        lower_bound_str = pformat_expression(self._lower_bound, show_id=True)
        upper_bound_str = pformat_expression(self._upper_bound, show_id=True)
        stride_str = pformat_expression(self._stride, show_id=True)
        return f"index({lower_bound_str}:{upper_bound_str}:{stride_str})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self._lower_bound)}, "
            f"{repr(self._upper_bound)}, {repr(self._stride)})"
        )


class TypeQualifier(StrEnum):
    """Type qualifier."""

    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    PARAM = "param"
    TEMP = "temp"


def promote_type_qualifiers(
    type_qualifier1: TypeQualifier, type_qualifier2: TypeQualifier
) -> TypeQualifier:
    """Promote two type qualifiers to a common type qualifier.

    Args:
        type_qualifier1: First type qualifier.
        type_qualifier2: Second type qualifier.

    Returns:
        Common type qualifier to which both type qualifiers can be promoted.

    """
    if type_qualifier1 == type_qualifier2 == TypeQualifier.PARAM:
        return TypeQualifier.PARAM
    else:
        return TypeQualifier.TEMP
