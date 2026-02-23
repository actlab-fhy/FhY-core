"""Core type system."""

__all__ = [
    "CoreDataType",
    "DataType",
    "FhYCoreTypeError",
    "IndexType",
    "NumericalType",
    "PrimitiveDataType",
    "promote_core_data_types",
    "promote_primitive_data_types",
    "promote_type_qualifiers",
    "TemplateDataType",
    "TupleType",
    "Type",
    "TypeQualifier",
]

from abc import ABC
from collections.abc import Mapping
from functools import partial
from typing import Any

from fhy_core.serialization import (
    WrappedFamilySerializable,
    register_serializable,
)

from .error import register_error
from .expression import Expression, pformat_expression
from .identifier import Identifier
from .utils import Lattice, StrEnum, format_comma_separated_list


class Type(WrappedFamilySerializable, ABC):
    """Abstract compiler type."""


@register_error
class FhYCoreTypeError(TypeError):
    """Core type error."""


class DataType(WrappedFamilySerializable, ABC):
    """Abstract data type."""


class CoreDataType(StrEnum):
    """Core data type primitives."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
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


def get_core_data_type_bit_width(core_data_type: CoreDataType) -> int:
    """Get the bit width of a core data type.

    Args:
        core_data_type: Core data type.

    Returns:
        Bit width of the core data type

    """
    match core_data_type:
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
        case (
            CoreDataType.UINT64
            | CoreDataType.INT64
            | CoreDataType.FLOAT64
            | CoreDataType.COMPLEX64
        ):
            return 64
        case CoreDataType.COMPLEX128:
            return 128


def _define_uint_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.UINT8)
    lattice.add_element(CoreDataType.UINT16)
    lattice.add_element(CoreDataType.UINT32)
    lattice.add_element(CoreDataType.UINT64)

    lattice.add_order(CoreDataType.UINT8, CoreDataType.UINT16)
    lattice.add_order(CoreDataType.UINT16, CoreDataType.UINT32)
    lattice.add_order(CoreDataType.UINT32, CoreDataType.UINT64)

    if not lattice.is_lattice():
        raise RuntimeError("Unsigned integer lattice is not a lattice.")

    return lattice


def _define_int_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.INT8)
    lattice.add_element(CoreDataType.INT16)
    lattice.add_element(CoreDataType.INT32)
    lattice.add_element(CoreDataType.INT64)

    lattice.add_order(CoreDataType.INT8, CoreDataType.INT16)
    lattice.add_order(CoreDataType.INT16, CoreDataType.INT32)
    lattice.add_order(CoreDataType.INT32, CoreDataType.INT64)

    if not lattice.is_lattice():
        raise RuntimeError("Integer lattice is not a lattice.")

    return lattice


def _define_float_complex_data_type_lattice() -> Lattice[CoreDataType]:
    lattice = Lattice[CoreDataType]()
    lattice.add_element(CoreDataType.FLOAT16)
    lattice.add_element(CoreDataType.FLOAT32)
    lattice.add_element(CoreDataType.FLOAT64)
    lattice.add_element(CoreDataType.COMPLEX32)
    lattice.add_element(CoreDataType.COMPLEX64)
    lattice.add_element(CoreDataType.COMPLEX128)

    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.FLOAT32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.FLOAT64)
    lattice.add_order(CoreDataType.FLOAT16, CoreDataType.COMPLEX32)
    lattice.add_order(CoreDataType.FLOAT32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.FLOAT64, CoreDataType.COMPLEX128)
    lattice.add_order(CoreDataType.COMPLEX32, CoreDataType.COMPLEX64)
    lattice.add_order(CoreDataType.COMPLEX64, CoreDataType.COMPLEX128)

    if not lattice.is_lattice():
        raise RuntimeError("Floating point and complex lattice is not a lattice.")

    return lattice


_UINT_DATA_TYPE_LATTICE = _define_uint_data_type_lattice()
_INT_DATA_TYPE_LATTICE = _define_int_data_type_lattice()
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
    _UINT_DATA_TYPES = {
        CoreDataType.UINT8,
        CoreDataType.UINT16,
        CoreDataType.UINT32,
        CoreDataType.UINT64,
    }
    _INT_DATA_TYPES = {
        CoreDataType.INT8,
        CoreDataType.INT16,
        CoreDataType.INT32,
        CoreDataType.INT64,
    }
    _FLOAT_COMPLEX_DATA_TYPES = {
        CoreDataType.FLOAT16,
        CoreDataType.FLOAT32,
        CoreDataType.FLOAT64,
        CoreDataType.COMPLEX32,
        CoreDataType.COMPLEX64,
        CoreDataType.COMPLEX128,
    }

    if core_data_type1 in _UINT_DATA_TYPES and core_data_type2 in _UINT_DATA_TYPES:
        return _UINT_DATA_TYPE_LATTICE.get_least_upper_bound(
            core_data_type1, core_data_type2
        )
    elif core_data_type1 in _INT_DATA_TYPES and core_data_type2 in _INT_DATA_TYPES:
        return _INT_DATA_TYPE_LATTICE.get_least_upper_bound(
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


@register_serializable
class PrimitiveDataType(DataType):
    """Primitive data type."""

    _core_data_type: CoreDataType

    def __init__(self, core_data_type: CoreDataType) -> None:
        self._core_data_type = core_data_type

    @property
    def core_data_type(self) -> CoreDataType:
        return self._core_data_type

    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {"core_data_type": self._core_data_type.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "PrimitiveDataType":
        cls.raise_error_if_deserialization_data_invalid(data, {"core_data_type": str})
        core_data_type = CoreDataType(data["core_data_type"])
        return cls(core_data_type)

    def __str__(self) -> str:
        return str(self._core_data_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._core_data_type)})"


@register_serializable
class TemplateDataType(DataType):
    """Template data type."""

    # TODO: Fix this class!

    _data_type: Identifier
    widths: list[int] | None

    def __init__(self, data_type: Identifier, widths: list[int] | None = None) -> None:
        self._data_type = data_type
        self.widths = widths

    @property
    def template_type(self) -> Identifier:
        return self._data_type

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


@register_serializable
class NumericalType(Type):
    """Numerical multi-dimensional array type; empty shapes indicate scalars."""

    _data_type: DataType
    _shape: list[Expression]

    def __init__(
        self, data_type: DataType, shape: list[Expression] | None = None
    ) -> None:
        super().__init__()
        self._data_type = data_type
        self._shape = shape or []

    @property
    def data_type(self) -> DataType:
        return self._data_type

    @property
    def shape(self) -> list[Expression]:
        return self._shape

    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {
            "data_type": self._data_type.serialize_to_dict(),
            "shape": [dim.serialize_to_dict() for dim in self._shape],
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "NumericalType":
        cls.raise_error_if_deserialization_data_invalid(
            data, {"data_type": dict, "shape": list}
        )
        data_type_dict = data["data_type"]
        shape_list = data["shape"]
        data_type = DataType.deserialize_from_dict(data_type_dict)
        shape = [Expression.deserialize_from_dict(dim_dict) for dim_dict in shape_list]
        return cls(data_type, shape)

    def __str__(self) -> str:
        shape_str = format_comma_separated_list(
            self._shape, str_func=partial(pformat_expression, show_id=True)
        )
        return f"{self._data_type}[{shape_str}]"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self._data_type)}, {repr(self._shape)})"
        )


@register_serializable
class IndexType(Type):
    """Index type.

    Notes:
        - Similar to a python slice or range(start, stop, step)

    """

    _lower_bound: Expression
    _upper_bound: Expression
    _stride: Expression | None

    def __init__(
        self,
        lower_bound: Expression,
        upper_bound: Expression,
        stride: Expression | None,
    ) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._stride = stride

    @property
    def lower_bound(self) -> Expression:
        return self._lower_bound

    @property
    def upper_bound(self) -> Expression:
        return self._upper_bound

    @property
    def stride(self) -> Expression | None:
        return self._stride

    def serialize_data_to_dict(self) -> dict[str, Any]:
        data = {
            "lower_bound": self._lower_bound.serialize_to_dict(),
            "upper_bound": self._upper_bound.serialize_to_dict(),
        }
        if self._stride is not None:
            data["stride"] = self._stride.serialize_to_dict()
        return data

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "IndexType":
        cls.raise_error_if_deserialization_data_invalid(
            data,
            {
                "lower_bound": dict,
                "upper_bound": dict,
            },
            optional_fields={"stride": dict},
        )
        lower_bound_dict = data["lower_bound"]
        upper_bound_dict = data["upper_bound"]
        stride_dict = data.get("stride")
        lower_bound = Expression.deserialize_from_dict(lower_bound_dict)
        upper_bound = Expression.deserialize_from_dict(upper_bound_dict)
        stride = Expression.deserialize_from_dict(stride_dict) if stride_dict else None
        return cls(lower_bound, upper_bound, stride)

    def __str__(self) -> str:
        lower_bound_str = pformat_expression(self._lower_bound, show_id=True)
        upper_bound_str = pformat_expression(self._upper_bound, show_id=True)
        stride_str = (
            pformat_expression(self._stride, show_id=True) if self._stride else "1"
        )
        return f"index({lower_bound_str}:{upper_bound_str}:{stride_str})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({repr(self._lower_bound)}, "
            f"{repr(self._upper_bound)}, {repr(self._stride)})"
        )


@register_serializable
class TupleType(Type):
    """Tuple type."""

    _types: list[Type]

    def __init__(self, types: list[Type]) -> None:
        super().__init__()
        self._types = types

    @property
    def types(self) -> list[Type]:
        return self._types

    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {"types": [ty.serialize_to_dict() for ty in self._types]}

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "TupleType":
        cls.raise_error_if_deserialization_data_invalid(data, {"types": list})
        types_list = data["types"]
        types = [Type.deserialize_from_dict(ty_dict) for ty_dict in types_list]
        return cls(types)

    def __str__(self) -> str:
        return f"({format_comma_separated_list(self._types)})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._types)})"


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
