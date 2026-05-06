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
from fhy_core.type_dispatch import (
    TypeUnificationEnv,
    bind_data_template,
    bind_template,
    structural_eq,
    substitute_data_template,
    substitute_template,
    unify,
)
from fhy_core.types import (
    CoreDataType,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TupleType,
    Type,
    TypeQualifier,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)

from .conftest import mock_identifier


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


def test_tuple_type_structural_equivalence_false_for_element_order() -> None:
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


def test_tuple_type_dict_serialization() -> None:
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
        ty_shape = ty.shape
        assert isinstance(ty_shape[0], Expression)
        assert isinstance(ty_shape[1], Expression)
        assert ty_shape[0].is_structurally_equivalent(shape[0])
        assert ty_shape[1].is_structurally_equivalent(shape[1])


# TODO: Check serialization structure errors and value errors for all types.


# ---------------------------------------------------------------------------
# Type-system dispatchers: TypeUnificationEnv, bind/substitute/unify/eq.
# ---------------------------------------------------------------------------


def _float32() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.FLOAT32)


def _int32() -> PrimitiveDataType:
    return PrimitiveDataType(CoreDataType.INT32)


def test_type_unification_env_empty_has_no_bindings() -> None:
    env = TypeUnificationEnv.empty()
    assert env.get_data_type_binding("T") is None
    assert env.get_type_binding("T") is None
    assert env.get_expression_binding(Identifier("N")) is None


def test_type_unification_env_with_helpers_return_new_env() -> None:
    original = TypeUnificationEnv.empty()
    n_ident = Identifier("N")

    env_with_dt = original.with_data_type_binding("T", _int32())
    env_with_ty = original.with_type_binding("U", NumericalType(_int32()))
    env_with_expr = original.with_expression_binding(n_ident, LiteralExpression(4))

    assert original.get_data_type_binding("T") is None
    assert original.get_type_binding("U") is None
    assert original.get_expression_binding(n_ident) is None

    assert isinstance(env_with_dt.get_data_type_binding("T"), PrimitiveDataType)
    assert isinstance(env_with_ty.get_type_binding("U"), NumericalType)
    assert env_with_expr.get_expression_binding(n_ident) is not None

    assert not original.is_structurally_equivalent(env_with_dt)
    assert not original.is_structurally_equivalent(env_with_ty)
    assert not original.is_structurally_equivalent(env_with_expr)


def test_type_unification_env_is_frozen_dataclass() -> None:
    env = TypeUnificationEnv.empty()
    assert isinstance(env, Frozen)
    assert env.is_frozen
    with pytest.raises((FrozenInstanceError, FrozenMutationError)):
        env.data_type_bindings = frozendict()  # type: ignore[misc]


def test_type_unification_env_structural_equivalence_value_based() -> None:
    n_ident = Identifier("N")
    a = TypeUnificationEnv.empty().with_data_type_binding("T", _int32())
    b = TypeUnificationEnv.empty().with_data_type_binding("T", _int32())
    c = TypeUnificationEnv.empty().with_data_type_binding("T", _float32())
    d = TypeUnificationEnv.empty().with_expression_binding(
        n_ident, LiteralExpression(4)
    )

    assert a.is_structurally_equivalent(b)
    assert not a.is_structurally_equivalent(c)
    assert not a.is_structurally_equivalent(d)
    assert not a.is_structurally_equivalent("not an env")


def test_structural_eq_matches_legacy_for_numerical_type() -> None:
    a = NumericalType(_int32(), [LiteralExpression(4), LiteralExpression(8)])
    b = NumericalType(_int32(), [LiteralExpression(4), LiteralExpression(8)])
    c = NumericalType(_int32(), [LiteralExpression(4), LiteralExpression(9)])

    assert structural_eq(a, b)
    assert a.is_structurally_equivalent(b)
    assert not structural_eq(a, c)
    assert not a.is_structurally_equivalent(c)


def test_structural_eq_matches_legacy_for_data_type() -> None:
    a = _int32()
    b = _int32()
    c = _float32()

    assert structural_eq(a, b)
    assert a.is_structurally_equivalent(b)
    assert not structural_eq(a, c)
    assert not a.is_structurally_equivalent(c)


def test_structural_eq_returns_false_for_unrelated_classes() -> None:
    n = NumericalType(_int32())
    t = TupleType([n])
    assert not structural_eq(n, t)


def test_bind_template_then_substitute_round_trip_numerical() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    m_ident = Identifier("M")
    pattern = NumericalType(
        template_t, [IdentifierExpression(n_ident), IdentifierExpression(m_ident)]
    )
    actual = NumericalType(_float32(), [LiteralExpression(10), LiteralExpression(20)])

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    assert structural_eq(env.get_data_type_binding("T"), _float32())
    n_binding = env.get_expression_binding(n_ident)
    m_binding = env.get_expression_binding(m_ident)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert m_binding is not None and m_binding.is_structurally_equivalent(
        LiteralExpression(20)
    )

    substituted = substitute_template(pattern, env)
    assert structural_eq(substituted, actual)


def test_bind_template_then_substitute_round_trip_tuple() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    pattern_first = NumericalType(template_t, [IdentifierExpression(n_ident)])
    pattern_second = NumericalType(template_t, [IdentifierExpression(n_ident)])
    pattern = TupleType([pattern_first, pattern_second])

    actual_first = NumericalType(_int32(), [LiteralExpression(7)])
    actual_second = NumericalType(_int32(), [LiteralExpression(7)])
    actual = TupleType([actual_first, actual_second])

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    substituted = substitute_template(pattern, env)
    assert structural_eq(substituted, actual)


def test_bind_template_then_substitute_round_trip_index_type() -> None:
    n_ident = Identifier("N")
    pattern = IndexType(
        LiteralExpression(0), IdentifierExpression(n_ident), LiteralExpression(1)
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(64), LiteralExpression(1)
    )

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    binding = env.get_expression_binding(n_ident)
    assert binding is not None and binding.is_structurally_equivalent(
        LiteralExpression(64)
    )

    substituted = substitute_template(pattern, env)
    assert structural_eq(substituted, actual)


def test_bind_template_full_type_binding_records_entire_actual() -> None:
    template_t = TemplateDataType(Identifier("T"))
    pattern = NumericalType(template_t, [...])
    actual = NumericalType(
        _int32(),
        [LiteralExpression(4), LiteralExpression(5), LiteralExpression(6)],
    )

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    bound_full = env.get_type_binding("T")
    assert bound_full is not None and structural_eq(bound_full, actual)
    assert structural_eq(env.get_data_type_binding("T"), _int32())

    substituted = substitute_template(pattern, env)
    assert structural_eq(substituted, actual)


def test_bind_template_wildcard_with_concrete_data_type_accepts_any_shape() -> None:
    pattern = NumericalType(_float32(), [...])
    actual = NumericalType(_float32(), [LiteralExpression(1), LiteralExpression(2)])

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    assert env.get_data_type_binding("anything") is None
    assert env.get_type_binding("anything") is None


def test_bind_template_rank_mismatch_raises() -> None:
    pattern = NumericalType(_float32(), [LiteralExpression(4)])
    actual = NumericalType(_float32(), [LiteralExpression(4), LiteralExpression(5)])
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, TypeUnificationEnv.empty())


def test_bind_template_class_mismatch_raises() -> None:
    pattern = NumericalType(_float32())
    actual = TupleType([NumericalType(_float32())])
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, TypeUnificationEnv.empty())


def test_bind_template_conflicting_data_type_binding_raises() -> None:
    template_t = TemplateDataType(Identifier("T"))
    pattern = TupleType([NumericalType(template_t), NumericalType(template_t)])
    actual = TupleType([NumericalType(_float32()), NumericalType(_int32())])
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, TypeUnificationEnv.empty())


def test_bind_template_conflicting_shape_binding_raises() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    pattern = NumericalType(
        template_t,
        [IdentifierExpression(n_ident), IdentifierExpression(n_ident)],
    )
    actual = NumericalType(_int32(), [LiteralExpression(4), LiteralExpression(5)])
    with pytest.raises(VerificationError):
        bind_template(pattern, actual, TypeUnificationEnv.empty())


def test_substitute_template_leaves_unbound_placeholders_alone() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    pattern = NumericalType(template_t, [IdentifierExpression(n_ident)])
    substituted = substitute_template(pattern, TypeUnificationEnv.empty())
    assert structural_eq(substituted, pattern)


def test_substitute_template_handles_compound_shape_expressions() -> None:
    n_ident = Identifier("N")
    pattern = NumericalType(
        _float32(),
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_ident),
                LiteralExpression(1),
            )
        ],
    )
    env = TypeUnificationEnv.empty().with_expression_binding(
        n_ident, LiteralExpression(8)
    )
    substituted = substitute_template(pattern, env)
    assert isinstance(substituted, NumericalType)
    expected_dim = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(8), LiteralExpression(1)
    )
    dim = substituted.shape[0]
    assert isinstance(dim, BinaryExpression)
    assert dim.is_structurally_equivalent(expected_dim)


def test_substitute_data_template_resolves_bound_template() -> None:
    template_t = TemplateDataType(Identifier("T"))
    env = TypeUnificationEnv.empty().with_data_type_binding("T", _int32())
    substituted = substitute_data_template(template_t, env)
    assert structural_eq(substituted, _int32())


def test_substitute_data_template_leaves_unbound_template_alone() -> None:
    template_t = TemplateDataType(Identifier("T"))
    substituted = substitute_data_template(template_t, TypeUnificationEnv.empty())
    assert isinstance(substituted, TemplateDataType)
    assert substituted.data_type.name_hint == "T"


def test_unify_binds_placeholder_on_either_side() -> None:
    n_ident = Identifier("N")
    m_ident = Identifier("M")
    expected = NumericalType(
        _float32(), [IdentifierExpression(n_ident), IdentifierExpression(m_ident)]
    )
    actual = NumericalType(
        _float32(), [LiteralExpression(10), IdentifierExpression(m_ident)]
    )

    unified, env = unify(expected, actual, TypeUnificationEnv.empty())

    assert isinstance(unified, NumericalType)
    expected_unified = NumericalType(
        _float32(), [LiteralExpression(10), IdentifierExpression(m_ident)]
    )
    assert structural_eq(unified, expected_unified)

    n_binding = env.get_expression_binding(n_ident)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(10)
    )
    assert env.get_expression_binding(m_ident) is None


def test_unify_occurs_check_failure_raises() -> None:
    n_ident = Identifier("N")
    expected = NumericalType(_float32(), [IdentifierExpression(n_ident)])
    actual = NumericalType(
        _float32(),
        [
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(n_ident),
                LiteralExpression(1),
            )
        ],
    )
    with pytest.raises(VerificationError):
        unify(expected, actual, TypeUnificationEnv.empty())


def test_unify_structurally_equal_index_types_returns_unchanged() -> None:
    expected = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    actual = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(1)
    )
    unified, env = unify(expected, actual, TypeUnificationEnv.empty())
    assert structural_eq(unified, expected)
    assert env.is_structurally_equivalent(TypeUnificationEnv.empty())


def test_unify_mismatched_concrete_types_raises() -> None:
    expected = NumericalType(_float32())
    actual = TupleType([NumericalType(_float32())])
    with pytest.raises(VerificationError):
        unify(expected, actual, TypeUnificationEnv.empty())


def test_unify_data_type_template_on_either_side_binds() -> None:
    template_t = TemplateDataType(Identifier("T"))
    expected = NumericalType(template_t, [LiteralExpression(4)])
    actual = NumericalType(_int32(), [LiteralExpression(4)])
    unified, env = unify(expected, actual, TypeUnificationEnv.empty())
    assert structural_eq(unified, actual)
    assert structural_eq(env.get_data_type_binding("T"), _int32())


def test_bind_data_template_conflict_raises() -> None:
    template_t = TemplateDataType(Identifier("T"))
    env = TypeUnificationEnv.empty()
    env = bind_data_template(template_t, _int32(), env)
    with pytest.raises(VerificationError):
        bind_data_template(template_t, _float32(), env)


def test_bind_data_template_repeated_consistent_binding_is_idempotent() -> None:
    template_t = TemplateDataType(Identifier("T"))
    env = TypeUnificationEnv.empty()
    env = bind_data_template(template_t, _int32(), env)
    same = bind_data_template(template_t, _int32(), env)
    assert env.is_structurally_equivalent(same)


def test_bind_template_full_type_conflict_raises() -> None:
    template_t = TemplateDataType(Identifier("T"))
    pattern = NumericalType(template_t, [...])
    first_actual = NumericalType(_int32(), [LiteralExpression(2)])
    second_actual = NumericalType(_int32(), [LiteralExpression(3)])
    pair_pattern = TupleType([pattern, pattern])
    pair_actual = TupleType([first_actual, second_actual])
    with pytest.raises(VerificationError):
        bind_template(pair_pattern, pair_actual, TypeUnificationEnv.empty())


def test_frozen_env_chain_produces_distinct_envs_each_step() -> None:
    n_ident = Identifier("N")
    e0 = TypeUnificationEnv.empty()
    e1 = e0.with_data_type_binding("T", _int32())
    e2 = e1.with_expression_binding(n_ident, LiteralExpression(7))
    e3 = e2.with_type_binding("U", NumericalType(_float32()))

    envs = [e0, e1, e2, e3]
    for left_index, left in enumerate(envs):
        for right in envs[left_index + 1 :]:
            assert not left.is_structurally_equivalent(right)
    assert e0.get_data_type_binding("T") is None
    assert e0.get_expression_binding(n_ident) is None
    assert e0.get_type_binding("U") is None


# ---------------------------------------------------------------------------
# Out-of-tree extension test.
#
# Demonstrates that a downstream package can register a brand-new ``Type``
# subclass against the dispatchers without modifying ``fhy_core``.
# ---------------------------------------------------------------------------


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


@structural_eq.register
def _(a: SyntheticTaggedType, b: object) -> bool:
    return (
        isinstance(b, SyntheticTaggedType)
        and a.tag == b.tag
        and structural_eq(a.inner, b.inner)
    )


@bind_template.register
def _(
    pattern: SyntheticTaggedType, actual: Type, env: TypeUnificationEnv
) -> TypeUnificationEnv:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot bind SyntheticTaggedType against {type(actual).__name__}."
        )
    if pattern.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {pattern.tag!r} vs {actual.tag!r}.")
    return bind_template(pattern.inner, actual.inner, env)


@substitute_template.register
def _(t: SyntheticTaggedType, env: TypeUnificationEnv) -> Type:
    return SyntheticTaggedType(t.tag, substitute_template(t.inner, env))


@unify.register
def _(
    expected: SyntheticTaggedType, actual: Type, env: TypeUnificationEnv
) -> tuple[Type, TypeUnificationEnv]:
    if not isinstance(actual, SyntheticTaggedType):
        raise VerificationError(
            f"Cannot unify SyntheticTaggedType with {type(actual).__name__}."
        )
    if expected.tag != actual.tag:
        raise VerificationError(f"Tag mismatch: {expected.tag!r} vs {actual.tag!r}.")
    inner_unified, new_env = unify(expected.inner, actual.inner, env)
    return (
        SyntheticTaggedType(expected.tag, inner_unified),
        new_env,
    )


def test_out_of_tree_structural_eq() -> None:
    inner_a = NumericalType(_int32(), [LiteralExpression(2)])
    inner_b = NumericalType(_int32(), [LiteralExpression(2)])
    inner_c = NumericalType(_int32(), [LiteralExpression(3)])

    a = SyntheticTaggedType("dense", inner_a)
    b = SyntheticTaggedType("dense", inner_b)
    c = SyntheticTaggedType("sparse", inner_a)
    d = SyntheticTaggedType("dense", inner_c)
    plain = NumericalType(_int32(), [LiteralExpression(2)])

    assert structural_eq(a, b)
    assert a.is_structurally_equivalent(b)
    assert not structural_eq(a, c)
    assert not structural_eq(a, d)
    assert not structural_eq(a, plain)


def test_out_of_tree_bind_and_substitute() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    pattern = SyntheticTaggedType(
        "dense", NumericalType(template_t, [IdentifierExpression(n_ident)])
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(_int32(), [LiteralExpression(8)])
    )

    env = bind_template(pattern, actual, TypeUnificationEnv.empty())
    substituted = substitute_template(pattern, env)
    assert isinstance(substituted, SyntheticTaggedType)
    assert structural_eq(substituted, actual)


def test_out_of_tree_unify_propagates_inner_bindings() -> None:
    template_t = TemplateDataType(Identifier("T"))
    n_ident = Identifier("N")
    expected = SyntheticTaggedType(
        "dense", NumericalType(template_t, [IdentifierExpression(n_ident)])
    )
    actual = SyntheticTaggedType(
        "dense", NumericalType(_int32(), [LiteralExpression(8)])
    )
    unified, env = unify(expected, actual, TypeUnificationEnv.empty())
    assert isinstance(unified, SyntheticTaggedType)
    assert structural_eq(unified, actual)
    assert structural_eq(env.get_data_type_binding("T"), _int32())
    n_binding = env.get_expression_binding(n_ident)
    assert n_binding is not None and n_binding.is_structurally_equivalent(
        LiteralExpression(8)
    )


def test_out_of_tree_tag_mismatch_raises() -> None:
    inner = NumericalType(_int32())
    expected = SyntheticTaggedType("dense", inner)
    actual = SyntheticTaggedType("sparse", inner)
    with pytest.raises(VerificationError):
        bind_template(expected, actual, TypeUnificationEnv.empty())
    with pytest.raises(VerificationError):
        unify(expected, actual, TypeUnificationEnv.empty())
