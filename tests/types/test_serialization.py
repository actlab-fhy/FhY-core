"""Serialization round-trip and error-path tests for `fhy_core.types`."""

from types import EllipsisType

import pytest

from fhy_core.expression import Expression, IdentifierExpression, LiteralExpression
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
)

from .conftest import mock_identifier

# The sentinel `__type__` for an ellipsis shape dimension. Pinned here so a
# rename is a one-line change that also forces the test author to think about
# whether the on-disk format is breaking.
ELLIPSIS_SHAPE_DIMENSION_TYPE_ID = "__numerical_type_shape_ellipsis__"


# =============================================================================
# Round-trip serialization
# =============================================================================


def test_primitive_data_type_dict_serialization() -> None:
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


def test_template_data_type_dict_serialization_with_widths() -> None:
    """Test template data types with widths round-trip through a dictionary."""
    name = mock_identifier("T", 1)
    template = TemplateDataType(name, widths=[8, 16])

    dictionary = template.serialize_to_dict()
    deserialized = TemplateDataType.deserialize_from_dict(dictionary)

    assert isinstance(deserialized, TemplateDataType)
    assert deserialized.widths == [8, 16]


def test_template_data_type_dict_serialization_without_widths() -> None:
    """Test template data types without widths round-trip through a dictionary."""
    name = mock_identifier("T", 2)
    template = TemplateDataType(name)

    dictionary = template.serialize_to_dict()
    deserialized = TemplateDataType.deserialize_from_dict(dictionary)

    assert isinstance(deserialized, TemplateDataType)
    assert deserialized.widths is None


def test_numerical_type_dict_serialization() -> None:
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
    deserialized_shape = numerical_type_deserialized.shape
    assert isinstance(deserialized_shape[0], Expression)
    assert isinstance(deserialized_shape[1], Expression)
    assert deserialized_shape[0].is_structurally_equivalent(shape[0])
    assert deserialized_shape[1].is_structurally_equivalent(shape[1])


def test_index_type_dict_serialization() -> None:
    """Test index types can be serialized/deserialized via a dictionary."""
    N = mock_identifier("N", 1)
    lower_bound = LiteralExpression(1)
    upper_bound = IdentifierExpression(N)
    index_type = IndexType(lower_bound, upper_bound)
    expected_dict = {
        "__type__": "index_type",
        "__data__": {
            "lower_bound": lower_bound.serialize_to_dict(),
            "upper_bound": upper_bound.serialize_to_dict(),
            "stride": LiteralExpression(1).serialize_to_dict(),
        },
    }
    dictionary = index_type.serialize_to_dict()
    assert dictionary == expected_dict
    index_type_deserialized = IndexType.deserialize_from_dict(dictionary)
    assert isinstance(index_type_deserialized, IndexType)
    assert index_type_deserialized.lower_bound.is_structurally_equivalent(lower_bound)
    assert index_type_deserialized.upper_bound.is_structurally_equivalent(upper_bound)
    assert index_type_deserialized.stride.is_structurally_equivalent(
        LiteralExpression(1)
    )


def test_index_type_with_stride_serialization() -> None:
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
    assert index_type_deserialized.stride.is_structurally_equivalent(stride)


def test_numerical_type_full_shape_wildcard_round_trips_through_serialization() -> None:
    """Test a numerical type with shape ``[...]`` round-trips structurally."""
    numerical_type = NumericalType(PrimitiveDataType(CoreDataType.INT32), [...])
    expected_dict = {
        "__type__": "numerical_type",
        "__data__": {
            "data_type": {
                "__type__": "primitive_data_type",
                "__data__": {"core_data_type": CoreDataType.INT32.value},
            },
            "shape": [
                {"__type__": ELLIPSIS_SHAPE_DIMENSION_TYPE_ID, "__data__": {}},
            ],
        },
    }
    dictionary = numerical_type.serialize_to_dict()
    assert dictionary == expected_dict

    deserialized = NumericalType.deserialize_from_dict(dictionary)
    assert isinstance(deserialized, NumericalType)
    assert len(deserialized.shape) == 1
    assert deserialized.shape[0] is Ellipsis
    assert deserialized.is_structurally_equivalent(numerical_type)


def test_numerical_type_per_dimension_wildcard_round_trips_through_serialization() -> (
    None
):
    """Test a numerical type with a mid-shape ``Ellipsis`` round-trips structurally."""
    N = mock_identifier("N", 1)
    shape: list[Expression | EllipsisType] = [
        IdentifierExpression(N),
        ...,
        LiteralExpression(4),
    ]
    numerical_type = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32), shape)

    dictionary = numerical_type.serialize_to_dict()
    assert isinstance(dictionary["__data__"], dict)
    serialized_shape = dictionary["__data__"]["shape"]
    assert isinstance(serialized_shape, list)
    assert serialized_shape[1] == {
        "__type__": ELLIPSIS_SHAPE_DIMENSION_TYPE_ID,
        "__data__": {},
    }

    deserialized = NumericalType.deserialize_from_dict(dictionary)
    assert isinstance(deserialized, NumericalType)
    deserialized_shape = deserialized.shape
    assert len(deserialized_shape) == 3
    assert isinstance(deserialized_shape[0], Expression)
    assert deserialized_shape[0].is_structurally_equivalent(IdentifierExpression(N))
    assert deserialized_shape[1] is Ellipsis
    assert isinstance(deserialized_shape[2], Expression)
    assert deserialized_shape[2].is_structurally_equivalent(LiteralExpression(4))
    assert deserialized.is_structurally_equivalent(numerical_type)


def test_numerical_type_empty_shape_round_trips() -> None:
    """Test a scalar numerical type round-trips through serialization."""
    scalar_type = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    dictionary = scalar_type.serialize_to_dict()
    deserialized = NumericalType.deserialize_from_dict(dictionary)

    assert isinstance(deserialized, NumericalType)
    assert deserialized.is_scalar()
    assert deserialized.is_structurally_equivalent(scalar_type)


# =============================================================================
# Deserialization error paths
# =============================================================================


def _wrap(type_id: str, inner: SerializedDict) -> SerializedDict:
    """Build the family-serialization wrapper expected by ``deserialize_from_dict``."""
    return {"__type__": type_id, "__data__": inner}


def test_primitive_data_type_deserialize_raises_on_missing_key() -> None:
    """Test deserialize raises a structure error when ``core_data_type`` is missing."""
    with pytest.raises(DeserializationDictStructureError):
        PrimitiveDataType.deserialize_from_dict(_wrap("primitive_data_type", {}))


def test_primitive_data_type_deserialize_raises_on_wrong_type_for_core_data_type() -> (
    None
):
    """Test deserialization raises a structure error on a non-string core data type."""
    with pytest.raises(DeserializationDictStructureError):
        PrimitiveDataType.deserialize_from_dict(
            _wrap("primitive_data_type", {"core_data_type": 32})
        )


def test_primitive_data_type_deserialize_raises_on_unknown_core_data_type_value() -> (
    None
):
    """Test deserialization raises a value error for an unknown core data type name."""
    with pytest.raises(DeserializationValueError):
        PrimitiveDataType.deserialize_from_dict(
            _wrap("primitive_data_type", {"core_data_type": "fakefloat"})
        )


def test_template_data_type_deserialize_raises_on_missing_data_type_field() -> None:
    """Test deserialization raises a structure error when ``data_type`` is missing."""
    with pytest.raises(DeserializationDictStructureError):
        TemplateDataType.deserialize_from_dict(
            _wrap("template_data_type", {"widths": [8]})
        )


def test_template_data_type_deserialize_raises_on_widths_wrong_type() -> None:
    """Test deserialization raises a structure error when ``widths`` is not a list."""
    name = mock_identifier("T", 1)
    serialized_name = name.serialize_to_dict()
    with pytest.raises(DeserializationDictStructureError):
        TemplateDataType.deserialize_from_dict(
            _wrap(
                "template_data_type",
                {"data_type": serialized_name, "widths": "not-a-list"},
            )
        )


def test_template_data_type_deserialize_raises_on_non_positive_width() -> None:
    """Test deserialize raises a value error when widths contain a non-positive."""
    name = mock_identifier("T", 1)
    serialized_name = name.serialize_to_dict()
    with pytest.raises(DeserializationValueError):
        TemplateDataType.deserialize_from_dict(
            _wrap(
                "template_data_type",
                {"data_type": serialized_name, "widths": [8, 0]},
            )
        )


def test_numerical_type_deserialize_raises_on_non_list_shape() -> None:
    """Test deserialization raises a structure error when ``shape`` is not a list."""
    primitive_dict = PrimitiveDataType(CoreDataType.INT32).serialize_to_dict()
    with pytest.raises(DeserializationDictStructureError):
        NumericalType.deserialize_from_dict(
            _wrap(
                "numerical_type",
                {"data_type": primitive_dict, "shape": "scalar"},
            )
        )


def test_numerical_type_deserialize_raises_on_non_dict_shape_element() -> None:
    """Test deserialize raises a structure error when a shape element isn't a dict."""
    primitive_dict = PrimitiveDataType(CoreDataType.INT32).serialize_to_dict()
    with pytest.raises(DeserializationDictStructureError):
        NumericalType.deserialize_from_dict(
            _wrap(
                "numerical_type",
                {"data_type": primitive_dict, "shape": [42]},
            )
        )


def test_index_type_deserialize_raises_on_missing_field() -> None:
    """Test deserialization raises a structure error when an index field is missing."""
    serialized = LiteralExpression(0).serialize_to_dict()
    with pytest.raises(DeserializationDictStructureError):
        IndexType.deserialize_from_dict(
            _wrap("index_type", {"lower_bound": serialized})
        )
