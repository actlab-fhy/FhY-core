"""Tests the core type system."""

from dataclasses import FrozenInstanceError

import pytest
from frozendict import frozendict

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import SerializedDict
from fhy_core.trait import (
    Frozen,
    FrozenMutationError,
    StructuralEquivalence,
    VerificationError,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    Type,
    TypeQualifier,
    TypeUnificationEnvironment,
    bind_data_template,
    bind_template,
    get_core_data_type_bit_width,
    is_structurally_equivalent,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
    substitute_data_template,
    substitute_template,
    unify,
)

from .conftest import mock_identifier

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def int32_data_type() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.INT32)


@pytest.fixture
def float32_data_type() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.FLOAT32)


@pytest.fixture
def empty_environment() -> TypeUnificationEnvironment:
    return TypeUnificationEnvironment.empty()


# =============================================================================
# `Type` and `DataType` runtime protocols
# =============================================================================


def test_type_structural_equivalence_runtime_protocol() -> None:
    """Test `Type` implementations satisfy `StructuralEquivalence` protocol."""
    ty = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    assert isinstance(ty, StructuralEquivalence)


def test_data_type_structural_equivalence_runtime_protocol() -> None:
    """Test `DataType` implementations satisfy `StructuralEquivalence` protocol."""
    data_type = PrimitiveDataType(CoreDataType.INT32)
    assert isinstance(data_type, StructuralEquivalence)


def test_type_family_is_frozen_on_construction() -> None:
    """Test all core type-family classes are frozen after construction."""
    N = mock_identifier("N", 1)
    shape = [IdentifierExpression(N), LiteralExpression(4)]
    data_type = PrimitiveDataType(CoreDataType.INT32)
    template_data_type = TemplateDataType(N, widths=[8, 16])
    numerical_type = NumericalType(data_type, shape)
    index_type = IndexType(LiteralExpression(0), LiteralExpression(10))

    for value in (
        data_type,
        template_data_type,
        numerical_type,
        index_type,
    ):
        assert isinstance(value, Frozen)
        assert value.is_frozen
        with pytest.raises(FrozenMutationError):
            value._freeze_probe = "mutation"


def test_numerical_type_structural_equivalence_true() -> None:
    """Test structural equivalence is true for matching numerical types."""
    shape_1 = [LiteralExpression(4), LiteralExpression(8)]
    shape_2 = [LiteralExpression(4), LiteralExpression(8)]
    left = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape_1)
    right = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape_2)
    assert left.is_structurally_equivalent(right)


def test_numerical_type_structural_equivalence_false_for_data_type() -> None:
    """Test structural equivalence is false for differing numerical data types."""
    shape = [LiteralExpression(4)]
    left = NumericalType(PrimitiveDataType(CoreDataType.INT16), shape)
    right = NumericalType(PrimitiveDataType(CoreDataType.INT32), shape)
    assert not left.is_structurally_equivalent(right)


def test_index_type_structural_equivalence_false_for_stride() -> None:
    """Test structural equivalence is false for differing index stride values."""
    lower_bound = LiteralExpression(0)
    upper_bound = LiteralExpression(10)
    left = IndexType(lower_bound, upper_bound, LiteralExpression(1))
    right = IndexType(lower_bound, upper_bound, LiteralExpression(2))
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
def test_get_core_data_type_bit_width(
    core_data_type: CoreDataType, expected_bit_width: int
) -> None:
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
        (CoreDataType.UINT, CoreDataType.INT, CoreDataType.INT),
        (CoreDataType.INT, CoreDataType.UINT, CoreDataType.INT),
        (CoreDataType.UINT8, CoreDataType.INT8, CoreDataType.INT16),
        (CoreDataType.UINT16, CoreDataType.INT16, CoreDataType.INT32),
        (CoreDataType.UINT32, CoreDataType.INT32, CoreDataType.INT64),
        (CoreDataType.UINT, CoreDataType.INT32, CoreDataType.INT32),
        (CoreDataType.UINT16, CoreDataType.INT8, CoreDataType.INT32),
        (CoreDataType.UINT32, CoreDataType.INT8, CoreDataType.INT64),
    ],
)
def test_promote_primitive_data_type(
    core_data_type1: CoreDataType,
    core_data_type2: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
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
def test_is_weak_core_data_type(
    core_data_type: CoreDataType, expected_is_weak: bool
) -> None:
    """Test detection of weak literal core data types."""
    assert is_weak_core_data_type(core_data_type) is expected_is_weak


@pytest.mark.parametrize(
    ("literal", "core_data_type", "expected_core_data_type"),
    [
        (0, CoreDataType.UINT, CoreDataType.UINT8),
        (255, CoreDataType.UINT, CoreDataType.UINT8),
        (256, CoreDataType.UINT, CoreDataType.UINT16),
        (1, CoreDataType.INT32, CoreDataType.INT32),
        (1, CoreDataType.FLOAT32, CoreDataType.FLOAT32),
        (1, CoreDataType.COMPLEX64, CoreDataType.COMPLEX64),
        (255, CoreDataType.INT8, CoreDataType.INT16),
        (-1, CoreDataType.INT, CoreDataType.INT8),
        (-129, CoreDataType.INT, CoreDataType.INT16),
        (-1, CoreDataType.FLOAT64, CoreDataType.FLOAT64),
        (1.5, CoreDataType.FLOAT, CoreDataType.FLOAT16),
    ],
)
def test_resolve_literal_core_data_type(
    literal: int | float,
    core_data_type: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Weak literal types should resolve to the narrowest compatible concrete type."""
    assert (
        resolve_literal_core_data_type(literal, core_data_type)
        == expected_core_data_type
    )


def test_resolve_large_positive_literal_to_uint64_without_signed_context() -> None:
    """Large positive literals should resolve in unsigned contexts lazily."""
    assert (
        resolve_literal_core_data_type(2**31, CoreDataType.UINT32)
        == CoreDataType.UINT32
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
    type_qualifer1: TypeQualifier,
    type_qualifer2: TypeQualifier,
    expected_type_qualifer: TypeQualifier,
) -> None:
    """Test type qualifiers are correctly promoted."""
    assert (
        promote_type_qualifiers(type_qualifer1, type_qualifer2)
        == expected_type_qualifer
    ), (
        f"Expected the promotion of {type_qualifer1} and {type_qualifer2} "
        f"to be {expected_type_qualifer}."
    )


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


# TODO: Check serialization structure errors and value errors for all types.


# =============================================================================
# `TypeUnificationEnvironment` construction and helpers
# =============================================================================


def test_empty_environment_has_no_bindings(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test ``empty()`` returns an environment whose binding tables are empty."""
    assert empty_environment.get_data_type_binding("T") is None
    assert empty_environment.get_type_binding("T") is None
    assert empty_environment.get_expression_binding(Identifier("N")) is None


def test_with_helpers_return_new_environments_without_mutating_original(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test ``with_*`` helpers return new environments and leave the original alone."""
    n_identifier = Identifier("N")

    environment_with_data_type_binding = empty_environment.with_data_type_binding(
        "T", int32_data_type
    )
    environment_with_type_binding = empty_environment.with_type_binding(
        "U", NumericalType(int32_data_type)
    )
    environment_with_expression_binding = empty_environment.with_expression_binding(
        n_identifier, LiteralExpression(4)
    )

    assert empty_environment.get_data_type_binding("T") is None
    assert empty_environment.get_type_binding("U") is None
    assert empty_environment.get_expression_binding(n_identifier) is None

    assert isinstance(
        environment_with_data_type_binding.get_data_type_binding("T"),
        PrimitiveDataType,
    )
    assert isinstance(
        environment_with_type_binding.get_type_binding("U"), NumericalType
    )
    assert (
        environment_with_expression_binding.get_expression_binding(n_identifier)
        is not None
    )

    assert not empty_environment.is_structurally_equivalent(
        environment_with_data_type_binding
    )
    assert not empty_environment.is_structurally_equivalent(
        environment_with_type_binding
    )
    assert not empty_environment.is_structurally_equivalent(
        environment_with_expression_binding
    )


def test_environment_is_a_frozen_dataclass(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the environment is frozen and rejects attribute assignment."""
    assert isinstance(empty_environment, Frozen)
    assert empty_environment.is_frozen
    with pytest.raises((FrozenInstanceError, FrozenMutationError)):
        empty_environment.data_type_bindings = frozendict()  # type: ignore[misc]


def test_environment_structural_equivalence_compares_bindings_by_value(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test environment structural equivalence compares bindings by structural value."""
    n_identifier = Identifier("N")
    environment_with_int = TypeUnificationEnvironment.empty().with_data_type_binding(
        "T", int32_data_type
    )
    environment_with_int_duplicate = (
        TypeUnificationEnvironment.empty().with_data_type_binding("T", int32_data_type)
    )
    environment_with_float = TypeUnificationEnvironment.empty().with_data_type_binding(
        "T", float32_data_type
    )
    environment_with_expression = (
        TypeUnificationEnvironment.empty().with_expression_binding(
            n_identifier, LiteralExpression(4)
        )
    )

    assert environment_with_int.is_structurally_equivalent(
        environment_with_int_duplicate
    )
    assert not environment_with_int.is_structurally_equivalent(environment_with_float)
    assert not environment_with_int.is_structurally_equivalent(
        environment_with_expression
    )
    assert not environment_with_int.is_structurally_equivalent("not an environment")


def test_chained_with_helpers_produce_pairwise_distinct_environments(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test successive ``with_*`` calls yield pairwise structurally distinct envs."""
    n_identifier = Identifier("N")
    environment_0 = TypeUnificationEnvironment.empty()
    environment_1 = environment_0.with_data_type_binding("T", int32_data_type)
    environment_2 = environment_1.with_expression_binding(
        n_identifier, LiteralExpression(7)
    )
    environment_3 = environment_2.with_type_binding(
        "U", NumericalType(float32_data_type)
    )

    environments = [environment_0, environment_1, environment_2, environment_3]
    for left_index, left_environment in enumerate(environments):
        for right_environment in environments[left_index + 1 :]:
            assert not left_environment.is_structurally_equivalent(right_environment)
    assert environment_0.get_data_type_binding("T") is None
    assert environment_0.get_expression_binding(n_identifier) is None
    assert environment_0.get_type_binding("U") is None


# =============================================================================
# `is_structurally_equivalent` dispatcher
# =============================================================================


def test_is_structurally_equivalent_dispatches_for_numerical_type(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test dispatcher and `NumericalType.is_structurally_equivalent` agree."""
    left = NumericalType(int32_data_type, [LiteralExpression(4), LiteralExpression(8)])
    right = NumericalType(int32_data_type, [LiteralExpression(4), LiteralExpression(8)])
    different = NumericalType(
        int32_data_type, [LiteralExpression(4), LiteralExpression(9)]
    )

    assert is_structurally_equivalent(left, right)
    assert left.is_structurally_equivalent(right)
    assert not is_structurally_equivalent(left, different)
    assert not left.is_structurally_equivalent(different)


def test_is_structurally_equivalent_dispatches_for_data_type(
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test dispatcher and `DataType.is_structurally_equivalent` agree on primitives."""
    int32_duplicate = PrimitiveDataType(CoreDataType.INT32)

    assert is_structurally_equivalent(int32_data_type, int32_duplicate)
    assert int32_data_type.is_structurally_equivalent(int32_duplicate)
    assert not is_structurally_equivalent(int32_data_type, float32_data_type)
    assert not int32_data_type.is_structurally_equivalent(float32_data_type)


def test_is_structurally_equivalent_returns_false_for_unrelated_concrete_classes(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the dispatcher returns ``False`` for unrelated concrete classes."""
    numerical_type = NumericalType(int32_data_type)
    index_type = IndexType(LiteralExpression(0), LiteralExpression(10))
    assert not is_structurally_equivalent(numerical_type, index_type)


# =============================================================================
# Template binding
# =============================================================================


def test_bind_template_then_substitute_round_trips_for_numerical_type(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test a `NumericalType` bind/substitute cycle reproduces the actual type."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    m_identifier = Identifier("M")
    pattern = NumericalType(
        template_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(m_identifier),
        ],
    )
    actual = NumericalType(
        float32_data_type, [LiteralExpression(10), LiteralExpression(20)]
    )

    environment = bind_template(pattern, actual, empty_environment)
    assert is_structurally_equivalent(
        environment.get_data_type_binding("T"), float32_data_type
    )
    n_binding = environment.get_expression_binding(n_identifier)
    m_binding = environment.get_expression_binding(m_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert m_binding is not None and m_binding.is_structurally_equivalent(
        LiteralExpression(20)
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_then_substitute_round_trips_for_index_type(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test an `IndexType` bind/substitute cycle reproduces the actual type."""
    n_identifier = Identifier("N")
    pattern = IndexType(
        LiteralExpression(0),
        IdentifierExpression(n_identifier),
        LiteralExpression(1),
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(64), LiteralExpression(1)
    )

    environment = bind_template(pattern, actual, empty_environment)
    binding = environment.get_expression_binding(n_identifier)
    assert binding is not None and binding.is_structurally_equivalent(
        LiteralExpression(64)
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_full_type_wildcard_records_entire_actual(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test `[T, ...]` against a concrete type binds the entire actual to ``T``."""
    template_data_type = TemplateDataType(Identifier("T"))
    pattern = NumericalType(template_data_type, [...])
    actual = NumericalType(
        int32_data_type,
        [LiteralExpression(4), LiteralExpression(5), LiteralExpression(6)],
    )

    environment = bind_template(pattern, actual, empty_environment)
    bound_full_type = environment.get_type_binding("T")
    assert bound_full_type is not None and is_structurally_equivalent(
        bound_full_type, actual
    )
    assert is_structurally_equivalent(
        environment.get_data_type_binding("T"), int32_data_type
    )

    substituted = substitute_template(pattern, environment)
    assert is_structurally_equivalent(substituted, actual)


def test_bind_template_wildcard_with_concrete_data_type_accepts_any_shape(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test `[concrete, ...]` accepts any actual shape without recording bindings."""
    pattern = NumericalType(float32_data_type, [...])
    actual = NumericalType(
        float32_data_type, [LiteralExpression(1), LiteralExpression(2)]
    )

    environment = bind_template(pattern, actual, empty_environment)
    assert environment.get_data_type_binding("anything") is None
    assert environment.get_type_binding("anything") is None


def test_bind_template_raises_on_shape_rank_mismatch(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises `VerificationError` when the pattern and actual ranks differ."""
    pattern = NumericalType(float32_data_type, [LiteralExpression(4)])
    actual = NumericalType(
        float32_data_type, [LiteralExpression(4), LiteralExpression(5)]
    )
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_template_raises_on_concrete_class_mismatch(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises `VerificationError` when pattern and actual classes differ."""
    pattern = NumericalType(float32_data_type)
    actual = IndexType(LiteralExpression(0), LiteralExpression(10))
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_template_raises_on_conflicting_shape_binding(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test bind raises when one shape variable binds to two different dimensions."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = NumericalType(
        template_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(n_identifier),
        ],
    )
    actual = NumericalType(
        int32_data_type, [LiteralExpression(4), LiteralExpression(5)]
    )
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, empty_environment)


def test_bind_data_template_raises_on_conflicting_binding(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test `bind_data_template` raises on a conflicting second binding."""
    template_data_type = TemplateDataType(Identifier("T"))
    environment = bind_data_template(
        template_data_type, int32_data_type, empty_environment
    )
    with pytest.raises(VerificationError):
        bind_data_template(template_data_type, float32_data_type, environment)


def test_bind_data_template_repeated_consistent_binding_is_idempotent(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test rebinding a template name to the same value leaves the env unchanged."""
    template_data_type = TemplateDataType(Identifier("T"))
    environment = bind_data_template(
        template_data_type, int32_data_type, empty_environment
    )
    same_environment = bind_data_template(
        template_data_type, int32_data_type, environment
    )
    assert environment.is_structurally_equivalent(same_environment)


# =============================================================================
# Template substitution
# =============================================================================


def test_substitute_template_leaves_unbound_placeholders_alone(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test substitution leaves unbound placeholders in the input unchanged."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = NumericalType(template_data_type, [IdentifierExpression(n_identifier)])
    substituted = substitute_template(pattern, empty_environment)
    assert is_structurally_equivalent(substituted, pattern)


def test_substitute_template_walks_compound_shape_expressions(
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test substitution recurses into binary shape expressions for placeholders."""
    n_identifier = Identifier("N")
    pattern = NumericalType(
        float32_data_type,
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_identifier),
                LiteralExpression(1),
            )
        ],
    )
    environment = TypeUnificationEnvironment.empty().with_expression_binding(
        n_identifier, LiteralExpression(8)
    )
    substituted = substitute_template(pattern, environment)
    assert isinstance(substituted, NumericalType)
    expected_dimension = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(8), LiteralExpression(1)
    )
    dimension = substituted.shape[0]
    assert isinstance(dimension, BinaryExpression)
    assert dimension.is_structurally_equivalent(expected_dimension)


def test_substitute_data_template_resolves_a_bound_template(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test data-type substitution returns the bound concrete type for a placeholder."""
    template_data_type = TemplateDataType(Identifier("T"))
    environment = TypeUnificationEnvironment.empty().with_data_type_binding(
        "T", int32_data_type
    )
    substituted = substitute_data_template(template_data_type, environment)
    assert is_structurally_equivalent(substituted, int32_data_type)


def test_substitute_data_template_leaves_an_unbound_template_alone(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test data-type substitution returns the placeholder unchanged when unbound."""
    template_data_type = TemplateDataType(Identifier("T"))
    substituted = substitute_data_template(template_data_type, empty_environment)
    assert isinstance(substituted, TemplateDataType)
    assert substituted.data_type.name_hint == "T"


# =============================================================================
# Unification
# =============================================================================


def test_unify_binds_a_placeholder_when_appearing_on_either_side(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification binds placeholders regardless of which side carries them."""
    n_identifier = Identifier("N")
    m_identifier = Identifier("M")
    expected = NumericalType(
        float32_data_type,
        [
            IdentifierExpression(n_identifier),
            IdentifierExpression(m_identifier),
        ],
    )
    actual = NumericalType(
        float32_data_type,
        [LiteralExpression(10), IdentifierExpression(m_identifier)],
    )

    unified, environment = unify(expected, actual, empty_environment)

    assert isinstance(unified, NumericalType)
    expected_unified = NumericalType(
        float32_data_type,
        [LiteralExpression(10), IdentifierExpression(m_identifier)],
    )
    assert is_structurally_equivalent(unified, expected_unified)

    n_binding = environment.get_expression_binding(n_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert environment.get_expression_binding(m_identifier) is None


def test_unify_raises_when_occurs_check_fails(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification raises when binding a placeholder would create a cycle."""
    n_identifier = Identifier("N")
    expected = NumericalType(float32_data_type, [IdentifierExpression(n_identifier)])
    actual = NumericalType(
        float32_data_type,
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_identifier),
                LiteralExpression(1),
            )
        ],
    )
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_returns_index_types_unchanged_when_already_structurally_equal(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two structurally equal `IndexType`s returns them unchanged."""
    expected = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, expected)
    assert environment.is_structurally_equivalent(empty_environment)


def test_unify_raises_on_mismatched_concrete_types(
    empty_environment: TypeUnificationEnvironment,
    float32_data_type: PrimitiveDataType,
) -> None:
    """Test unification raises when the two concrete type classes are incompatible."""
    expected = NumericalType(float32_data_type)
    actual = IndexType(LiteralExpression(0), LiteralExpression(10))
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_binds_data_type_template_appearing_on_either_side(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test unification binds a `TemplateDataType` when it appears on either side."""
    template_data_type = TemplateDataType(Identifier("T"))
    expected = NumericalType(template_data_type, [LiteralExpression(4)])
    actual = NumericalType(int32_data_type, [LiteralExpression(4)])
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, actual)
    assert is_structurally_equivalent(
        environment.get_data_type_binding("T"), int32_data_type
    )


def test_unify_raises_on_distinct_template_data_types(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying two distinct `TemplateDataType` placeholders raises."""
    left_template = TemplateDataType(Identifier("T"))
    right_template = TemplateDataType(Identifier("U"))
    expected = NumericalType(left_template, [LiteralExpression(4)])
    actual = NumericalType(right_template, [LiteralExpression(4)])
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)


def test_unify_accepts_two_identical_template_data_types(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test unifying a template with an equally-named template is a no-op."""
    template = TemplateDataType(Identifier("T"))
    duplicate = TemplateDataType(
        Identifier.deserialize_from_dict(template.data_type.serialize_to_dict())
    )
    expected = NumericalType(template, [LiteralExpression(4)])
    actual = NumericalType(duplicate, [LiteralExpression(4)])
    unified, environment = unify(expected, actual, empty_environment)
    assert is_structurally_equivalent(unified, expected)
    assert environment.is_structurally_equivalent(empty_environment)


# =============================================================================
# Out-of-tree extension
#
# Demonstrates that a downstream package can register a brand-new `Type`
# subclass against the dispatchers without modifying ``fhy_core``.
# =============================================================================


class SyntheticTaggedType(Type):
    """Test-only Type subclass that wraps a ``Type`` and a string tag."""

    _tag: str
    _inner: Type

    def __init__(self, tag: str, inner: Type) -> None:
        super().__init__()
        self._tag = tag
        self._inner = inner
        self.freeze(deep=True)

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def inner(self) -> Type:
        return self._inner

    def serialize_data_to_dict(self) -> SerializedDict:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def deserialize_data_from_dict(  # pragma: no cover
        cls, data: SerializedDict
    ) -> "SyntheticTaggedType":
        raise NotImplementedError


@is_structurally_equivalent.register
def _(left: SyntheticTaggedType, right: object) -> bool:
    return (
        isinstance(right, SyntheticTaggedType)
        and left.tag == right.tag
        and is_structurally_equivalent(left.inner, right.inner)
    )


@bind_template.register
def _(
    pattern: SyntheticTaggedType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> TypeUnificationEnvironment:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot bind SyntheticTaggedType against {type(actual).__name__}."
        )
    elif pattern.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {pattern.tag!r} vs {actual.tag!r}.")
    else:
        return bind_template(pattern.inner, actual.inner, environment)


@substitute_template.register
def _(type_: SyntheticTaggedType, environment: TypeUnificationEnvironment) -> Type:
    return SyntheticTaggedType(type_.tag, substitute_template(type_.inner, environment))


@unify.register
def _(
    expected: SyntheticTaggedType,
    actual: Type,
    environment: TypeUnificationEnvironment,
) -> tuple[Type, TypeUnificationEnvironment]:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot unify SyntheticTaggedType with {type(actual).__name__}."
        )
    elif expected.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {expected.tag!r} vs {actual.tag!r}.")
    else:
        unified_inner, next_environment = unify(
            expected.inner, actual.inner, environment
        )
        return (
            SyntheticTaggedType(expected.tag, unified_inner),
            next_environment,
        )


def test_out_of_tree_class_supports_structural_equivalence_dispatch(
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the registered out-of-tree class participates in structural equivalence."""
    inner_first = NumericalType(int32_data_type, [LiteralExpression(2)])
    inner_first_duplicate = NumericalType(int32_data_type, [LiteralExpression(2)])
    inner_second = NumericalType(int32_data_type, [LiteralExpression(3)])

    dense_first = SyntheticTaggedType("dense", inner_first)
    dense_first_duplicate = SyntheticTaggedType("dense", inner_first_duplicate)
    sparse_first = SyntheticTaggedType("sparse", inner_first)
    dense_second = SyntheticTaggedType("dense", inner_second)
    plain_numerical_type = NumericalType(int32_data_type, [LiteralExpression(2)])

    assert is_structurally_equivalent(dense_first, dense_first_duplicate)
    assert dense_first.is_structurally_equivalent(dense_first_duplicate)
    assert not is_structurally_equivalent(dense_first, sparse_first)
    assert not is_structurally_equivalent(dense_first, dense_second)
    assert not is_structurally_equivalent(dense_first, plain_numerical_type)


def test_out_of_tree_class_supports_bind_then_substitute_round_trip(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test the registered out-of-tree class participates in bind and substitute."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    pattern = SyntheticTaggedType(
        "dense",
        NumericalType(template_data_type, [IdentifierExpression(n_identifier)]),
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(int32_data_type, [LiteralExpression(8)])
    )

    environment = bind_template(pattern, actual, empty_environment)
    substituted = substitute_template(pattern, environment)
    assert isinstance(substituted, SyntheticTaggedType)
    assert is_structurally_equivalent(substituted, actual)


def test_out_of_tree_class_propagates_inner_bindings_during_unification(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test unification through an out-of-tree wrapper records bindings on its inner."""
    template_data_type = TemplateDataType(Identifier("T"))
    n_identifier = Identifier("N")
    expected = SyntheticTaggedType(
        "dense",
        NumericalType(template_data_type, [IdentifierExpression(n_identifier)]),
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(int32_data_type, [LiteralExpression(8)])
    )
    unified, environment = unify(expected, actual, empty_environment)
    assert isinstance(unified, SyntheticTaggedType)
    assert is_structurally_equivalent(unified, actual)
    assert is_structurally_equivalent(
        environment.get_data_type_binding("T"), int32_data_type
    )
    n_binding = environment.get_expression_binding(n_identifier)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(8)
    )


def test_out_of_tree_class_raises_on_tag_mismatch(
    empty_environment: TypeUnificationEnvironment,
    int32_data_type: PrimitiveDataType,
) -> None:
    """Test bind and unify both raise when out-of-tree wrapper tags differ."""
    inner = NumericalType(int32_data_type)
    expected = SyntheticTaggedType("dense", inner)
    actual = SyntheticTaggedType("sparse", inner)
    with pytest.raises(VerificationError):
        bind_template(expected, actual, empty_environment)
    with pytest.raises(VerificationError):
        unify(expected, actual, empty_environment)
