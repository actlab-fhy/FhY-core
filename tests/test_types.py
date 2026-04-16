"""Tests the core type system."""

import pytest
from fhy_core.expression import (
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.trait import Frozen, FrozenMutationError, StructuralEquivalence
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    TypeQualifier,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)

from .conftest import mock_identifier


def test_type_structural_equivalence_runtime_protocol():
    """Test `Type` implementations satisfy `StructuralEquivalence` protocol."""
    ty = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    assert isinstance(ty, StructuralEquivalence)


def test_data_type_structural_equivalence_runtime_protocol():
    """Test `DataType` implementations satisfy `StructuralEquivalence` protocol."""
    data_type = PrimitiveDataType(CoreDataType.INT32)
    assert isinstance(data_type, StructuralEquivalence)


def test_type_family_is_frozen_on_construction():
    """Test all core type-family classes are frozen after construction."""
    N = mock_identifier("N", 1)
    shape = [IdentifierExpression(N), LiteralExpression(4)]
    data_type = PrimitiveDataType(CoreDataType.INT32)
    template_data_type = TemplateDataType(N, widths=[8, 16])
    numerical_type = NumericalType(data_type, shape)
    index_type = IndexType(LiteralExpression(0), LiteralExpression(10), None)
    tuple_type = TupleType([numerical_type, index_type])

    for value in (
        data_type,
        template_data_type,
        numerical_type,
        index_type,
        tuple_type,
    ):
        assert isinstance(value, Frozen)
        assert value.is_frozen
        with pytest.raises(FrozenMutationError):
            value._freeze_probe = "mutation"


def test_numerical_type_structural_equivalence_true():
    """Test structural equivalence is true for matching numerical types."""
    shape_1 = [LiteralExpression(4), LiteralExpression(8)]
    shape_2 = [LiteralExpression(4), LiteralExpression(8)]
    left = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape_1)
    right = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape_2)
    assert left.is_structurally_equivalent(right)


def test_numerical_type_structural_equivalence_false_for_data_type():
    """Test structural equivalence is false for differing numerical data types."""
    shape = [LiteralExpression(4)]
    left = NumericalType(PrimitiveDataType(CoreDataType.INT16), shape)
    right = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape)
    assert not left.is_structurally_equivalent(right)


def test_index_type_structural_equivalence_false_for_stride():
    """Test structural equivalence is false for differing index stride values."""
    lower_bound = LiteralExpression(0)
    upper_bound = LiteralExpression(10)
    left = IndexType(lower_bound, upper_bound, LiteralExpression(1))
    right = IndexType(lower_bound, upper_bound, LiteralExpression(2))
    assert not left.is_structurally_equivalent(right)


def test_tuple_type_structural_equivalence_false_for_element_order():
    """Test structural equivalence is false for differing tuple type order."""
    int_type = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    float_type = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))
    left = TupleType([int_type, float_type])
    right = TupleType([float_type, int_type])
    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize(
    "core_data_type, expected_bit_width",
    [
        (CoreDataType.UINT, None),
        (CoreDataType.INT, None),
        (CoreDataType.FLOAT, None),
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
        (CoreDataType.UINT, CoreDataType.UINT8, CoreDataType.UINT8),
        (CoreDataType.INT, CoreDataType.INT16, CoreDataType.INT16),
        (CoreDataType.FLOAT, CoreDataType.FLOAT16, CoreDataType.FLOAT16),
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
    ("core_data_type", "expected_is_weak"),
    [
        (CoreDataType.UINT, True),
        (CoreDataType.INT, True),
        (CoreDataType.FLOAT, True),
        (CoreDataType.UINT8, False),
        (CoreDataType.INT8, False),
        (CoreDataType.FLOAT16, False),
    ],
)
def test_is_weak_core_data_type(core_data_type, expected_is_weak):
    """Test detection of weak literal core data types."""
    assert is_weak_core_data_type(core_data_type) is expected_is_weak


@pytest.mark.parametrize(
    ("literal", "core_data_type", "expected_core_data_type"),
    [
        (0, CoreDataType.UINT, CoreDataType.UINT8),
        (255, CoreDataType.UINT, CoreDataType.UINT8),
        (256, CoreDataType.UINT, CoreDataType.UINT16),
        (1, CoreDataType.INT32, CoreDataType.INT32),
        (255, CoreDataType.INT8, CoreDataType.INT16),
        (-1, CoreDataType.INT, CoreDataType.INT8),
        (-129, CoreDataType.INT, CoreDataType.INT16),
        (1.5, CoreDataType.FLOAT, CoreDataType.FLOAT16),
    ],
)
def test_resolve_literal_core_data_type(
    literal, core_data_type, expected_core_data_type
):
    """Weak literal types should resolve to the narrowest compatible concrete type."""
    assert (
        resolve_literal_core_data_type(literal, core_data_type)
        == expected_core_data_type
    )


def test_resolve_large_positive_literal_to_uint64_without_signed_context():
    """Large positive literals should resolve in unsigned contexts lazily."""
    assert (
        resolve_literal_core_data_type(2**63, CoreDataType.UINT64)
        == CoreDataType.UINT64
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
            "__type__": "primitive_data_type",
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
        "__type__": "numerical_type",
        "__data__": {
            "data_type": {
                "__type__": "primitive_data_type",
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
    assert numerical_type_deserialized.shape[0].is_structurally_equivalent(shape[0])
    assert numerical_type_deserialized.shape[1].is_structurally_equivalent(shape[1])


def test_index_type_dict_serialization():
    """Test index types can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    lower_bound = LiteralExpression(1)
    upper_bound = IdentifierExpression(N)
    index_type = IndexType(lower_bound, upper_bound, None)
    expected_dict = {
        "__type__": "index_type",
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
    assert index_type_deserialized.lower_bound.is_structurally_equivalent(lower_bound)
    assert index_type_deserialized.upper_bound.is_structurally_equivalent(upper_bound)


def test_index_type_with_stride_serialization():
    """Test index types with stride can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    lower_bound = LiteralExpression(1)
    upper_bound = IdentifierExpression(N)
    stride = LiteralExpression(2)
    index_type = IndexType(lower_bound, upper_bound, stride)
    expected_dict = {
        "__type__": "index_type",
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
    assert index_type_deserialized.lower_bound.is_structurally_equivalent(lower_bound)
    assert index_type_deserialized.upper_bound.is_structurally_equivalent(upper_bound)
    assert index_type_deserialized.stride is not None
    assert index_type_deserialized.stride.is_structurally_equivalent(stride)


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
        "__type__": "tuple_type",
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
        assert ty.shape[0].is_structurally_equivalent(shape[0])
        assert ty.shape[1].is_structurally_equivalent(shape[1])


# TODO: Check serialization structure errors and value errors for all types.
