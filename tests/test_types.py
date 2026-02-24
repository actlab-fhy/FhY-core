"""Tests the core type system."""

import pytest
from fhy_core.expression import (
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TupleType,
    TypeQualifier,
    get_core_data_type_bit_width,
    promote_core_data_types,
    promote_type_qualifiers,
)

from .conftest import assert_exact_expression_equality, mock_identifier


@pytest.mark.parametrize(
    "core_data_type, expected_bit_width",
    [
        (CoreDataType.UINT8, 8),
        (CoreDataType.UINT16, 16),
        (CoreDataType.UINT32, 32),
        (CoreDataType.UINT64, 64),
        (CoreDataType.INT8, 8),
        (CoreDataType.INT16, 16),
        (CoreDataType.INT32, 32),
        (CoreDataType.INT64, 64),
        (CoreDataType.FLOAT16, 16),
        (CoreDataType.FLOAT32, 32),
        (CoreDataType.FLOAT64, 64),
        (CoreDataType.COMPLEX32, 32),
        (CoreDataType.COMPLEX64, 64),
        (CoreDataType.COMPLEX128, 128),
    ],
)
def test_get_core_data_type_bit_width(core_data_type, expected_bit_width):
    """Test get_core_data_type_bit_width function with various core data types."""
    assert get_core_data_type_bit_width(core_data_type) == expected_bit_width


@pytest.mark.parametrize(
    ("core_data_type1", "core_data_type2", "expected_core_data_type"),
    [
        (CoreDataType.UINT8, CoreDataType.UINT8, CoreDataType.UINT8),
        (CoreDataType.UINT8, CoreDataType.UINT16, CoreDataType.UINT16),
        (CoreDataType.UINT16, CoreDataType.UINT8, CoreDataType.UINT16),
        (CoreDataType.INT32, CoreDataType.INT64, CoreDataType.INT64),
        (
            CoreDataType.FLOAT16,
            CoreDataType.FLOAT32,
            CoreDataType.FLOAT32,
        ),
        (
            CoreDataType.FLOAT64,
            CoreDataType.FLOAT16,
            CoreDataType.FLOAT64,
        ),
        (
            CoreDataType.COMPLEX32,
            CoreDataType.COMPLEX64,
            CoreDataType.COMPLEX64,
        ),
        (
            CoreDataType.FLOAT32,
            CoreDataType.COMPLEX32,
            CoreDataType.COMPLEX64,
        ),
    ],
)
def test_promote_primitive_data_type(
    core_data_type1, core_data_type2, expected_core_data_type
):
    """Test primitive data types are correctly promoted."""
    assert (
        promote_core_data_types(core_data_type1, core_data_type2)
        == expected_core_data_type
    ), (
        f"Expected the promotion of {core_data_type1} and {core_data_type2} "
        f"to be {expected_core_data_type}."
    )


@pytest.mark.parametrize(
    ("type_qualifer1", "type_qualifer2", "expected_type_qualifer"),
    [
        (TypeQualifier.INPUT, TypeQualifier.INPUT, TypeQualifier.TEMP),
        (TypeQualifier.STATE, TypeQualifier.PARAM, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.TEMP, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.PARAM, TypeQualifier.PARAM),
    ],
)
def test_promote_type_qualifiers(
    type_qualifer1, type_qualifer2, expected_type_qualifer
):
    """Test type qualifiers are correctly promoted."""
    assert (
        promote_type_qualifiers(type_qualifer1, type_qualifer2)
        == expected_type_qualifer
    ), (
        f"Expected the promotion of {type_qualifer1} and {type_qualifer2} "
        f"to be {expected_type_qualifer}."
    )


def test_primitive_data_type_dict_serialization():
    """Test primitive data types can be serialized/deserialized via a dictionary."""
    for core_data_type in CoreDataType:
        primitive_data_type = PrimitiveDataType(core_data_type)
        expected_dict = {
            "__type__": "fhy_core.types.PrimitiveDataType",
            "__data__": {"core_data_type": core_data_type.value},
        }
        dictionary = primitive_data_type.serialize_to_dict()
        assert dictionary == expected_dict
        primitive_data_type_deserialized = PrimitiveDataType.deserialize_from_dict(
            dictionary
        )
        assert isinstance(primitive_data_type_deserialized, PrimitiveDataType)
        assert primitive_data_type_deserialized.core_data_type == core_data_type


def test_numerical_type_dict_serialization():
    """Test numerical types can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    shape = [
        IdentifierExpression(N),
        LiteralExpression(28),
    ]
    numerical_type = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape)
    expected_dict = {
        "__type__": "fhy_core.types.NumericalType",
        "__data__": {
            "data_type": {
                "__type__": "fhy_core.types.PrimitiveDataType",
                "__data__": {"core_data_type": CoreDataType.INT32.value},
            },
            "shape": [
                shape[0].serialize_to_dict(),
                shape[1].serialize_to_dict(),
            ],
        },
    }
    dictionary = numerical_type.serialize_to_dict()
    assert dictionary == expected_dict
    numerical_type_deserialized = NumericalType.deserialize_from_dict(dictionary)
    assert isinstance(numerical_type_deserialized, NumericalType)
    assert isinstance(numerical_type_deserialized.data_type, PrimitiveDataType)
    assert numerical_type_deserialized.data_type.core_data_type == CoreDataType.INT32
    assert len(numerical_type_deserialized.shape) == 2
    assert_exact_expression_equality(numerical_type_deserialized.shape[0], shape[0])
    assert_exact_expression_equality(numerical_type_deserialized.shape[1], shape[1])


def test_index_type_dict_serialization():
    """Test index types can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    lower_bound = LiteralExpression(1)
    upper_bound = IdentifierExpression(N)
    index_type = IndexType(lower_bound, upper_bound, None)
    expected_dict = {
        "__type__": "fhy_core.types.IndexType",
        "__data__": {
            "lower_bound": lower_bound.serialize_to_dict(),
            "upper_bound": upper_bound.serialize_to_dict(),
            "stride": None,
        },
    }
    dictionary = index_type.serialize_to_dict()
    assert dictionary == expected_dict
    index_type_deserialized = IndexType.deserialize_from_dict(dictionary)
    assert isinstance(index_type_deserialized, IndexType)
    assert_exact_expression_equality(index_type_deserialized.lower_bound, lower_bound)
    assert_exact_expression_equality(index_type_deserialized.upper_bound, upper_bound)


def test_index_type_with_stride_serialization():
    """Test index types with stride can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    lower_bound = LiteralExpression(1)
    upper_bound = IdentifierExpression(N)
    stride = LiteralExpression(2)
    index_type = IndexType(lower_bound, upper_bound, stride)
    expected_dict = {
        "__type__": "fhy_core.types.IndexType",
        "__data__": {
            "lower_bound": lower_bound.serialize_to_dict(),
            "upper_bound": upper_bound.serialize_to_dict(),
            "stride": stride.serialize_to_dict(),
        },
    }
    dictionary = index_type.serialize_to_dict()
    assert dictionary == expected_dict
    index_type_deserialized = IndexType.deserialize_from_dict(dictionary)
    assert isinstance(index_type_deserialized, IndexType)
    assert_exact_expression_equality(index_type_deserialized.lower_bound, lower_bound)
    assert_exact_expression_equality(index_type_deserialized.upper_bound, upper_bound)
    assert_exact_expression_equality(index_type_deserialized.stride, stride)


def test_tuple_type_dict_serialization():
    """Test tuple types can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    shape = [
        IdentifierExpression(N),
        LiteralExpression(28),
    ]
    numerical_type = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape)
    tuple_type = TupleType([numerical_type, numerical_type])
    expected_dict = {
        "__type__": "fhy_core.types.TupleType",
        "__data__": {
            "types": [
                numerical_type.serialize_to_dict(),
                numerical_type.serialize_to_dict(),
            ],
        },
    }
    dictionary = tuple_type.serialize_to_dict()
    assert dictionary == expected_dict
    tuple_type_deserialized = TupleType.deserialize_from_dict(dictionary)
    assert isinstance(tuple_type_deserialized, TupleType)
    assert len(tuple_type_deserialized.types) == 2
    for ty in tuple_type_deserialized.types:
        assert isinstance(ty, NumericalType)
        assert isinstance(ty.data_type, PrimitiveDataType)
        assert ty.data_type.core_data_type == CoreDataType.INT32
        assert len(ty.shape) == 2
        assert_exact_expression_equality(ty.shape[0], shape[0])
        assert_exact_expression_equality(ty.shape[1], shape[1])


# TODO: Check serialization structure errors and value errors for all types.
