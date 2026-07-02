"""Tests for `fhy_core.types.core`: concrete types, promotion, literals."""

import pytest
from immutabledict import immutabledict

from fhy_core.expression import IdentifierExpression, LiteralExpression
from fhy_core.identifier import Identifier
from fhy_core.traits import (
    Frozen,
    FrozenMutationError,
    StructuralEquivalence,
)
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
    TypeUnificationEnvironment,
    get_core_data_type_bit_width,
    is_weak_core_data_type,
    promote_core_data_types,
    promote_primitive_data_types,
    promote_type_qualifiers,
    resolve_literal_core_data_type,
)
from fhy_core.types.core import _FLOAT_COMPLEX_DATA_TYPES, _INTEGER_DATA_TYPES

from .conftest import mock_identifier

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


def test_environment_is_a_frozen_dataclass(
    empty_environment: TypeUnificationEnvironment,
) -> None:
    """Test the environment is frozen and rejects attribute assignment."""
    assert isinstance(empty_environment, Frozen)
    assert empty_environment.is_frozen
    with pytest.raises(FrozenMutationError):
        empty_environment.data_type_bindings = immutabledict()  # type: ignore[misc]


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


def test_numerical_type_is_scalar_for_empty_shape() -> None:
    """Test `is_scalar` returns True for an empty-shape numerical type."""
    scalar_type = NumericalType(PrimitiveDataType(CoreDataType.INT32))
    assert scalar_type.is_scalar()


def test_numerical_type_is_scalar_false_for_nonempty_shape() -> None:
    """Test `is_scalar` returns False for a non-empty-shape numerical type."""
    array_type = NumericalType(
        PrimitiveDataType(CoreDataType.INT32), [LiteralExpression(4)]
    )
    assert not array_type.is_scalar()


# =============================================================================
# `__str__` / `__repr__` smoke tests
# =============================================================================


def test_primitive_data_type_str_and_repr_render_core_data_type() -> None:
    """Test `PrimitiveDataType.__str__`/`__repr__` mention the core data type."""
    data_type = PrimitiveDataType(CoreDataType.INT32)
    rendered_str = str(data_type)
    rendered_repr = repr(data_type)

    assert "int32" in rendered_str
    assert "PrimitiveDataType" in rendered_repr
    assert "INT32" in rendered_repr


def test_template_data_type_str_and_repr_render_identifier() -> None:
    """Test `TemplateDataType.__str__`/`__repr__` mention the placeholder identifier."""
    template = TemplateDataType(Identifier("T"))
    rendered_str = str(template)
    rendered_repr = repr(template)

    assert "T" in rendered_str
    assert "TemplateDataType" in rendered_repr
    assert "T" in rendered_repr


def test_numerical_type_str_renders_data_type_and_shape() -> None:
    """Test `NumericalType.__str__` renders both the data type and the shape."""
    array_type = NumericalType(
        PrimitiveDataType(CoreDataType.FLOAT32),
        [LiteralExpression(4), LiteralExpression(8)],
    )
    rendered_str = str(array_type)

    assert "float32" in rendered_str
    assert "[" in rendered_str and "]" in rendered_str


def test_numerical_type_str_renders_ellipsis_in_shape() -> None:
    """Test `NumericalType.__str__` renders `...` for an ellipsis dimension."""
    array_type = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32), [...])
    rendered_str = str(array_type)

    assert "..." in rendered_str


def test_numerical_type_repr_renders_class_name() -> None:
    """Test `NumericalType.__repr__` renders the class name."""
    array_type = NumericalType(PrimitiveDataType(CoreDataType.FLOAT32))
    assert "NumericalType" in repr(array_type)


def test_index_type_str_and_repr_render_bounds() -> None:
    """Test `IndexType.__str__`/`__repr__` render the bounds and stride."""
    index_type = IndexType(
        LiteralExpression(0), LiteralExpression(10), LiteralExpression(2)
    )
    rendered_str = str(index_type)
    rendered_repr = repr(index_type)

    assert "index(" in rendered_str
    assert "IndexType" in rendered_repr


# =============================================================================
# `CoreDataType` bit width and weak-literal detection
# =============================================================================


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
    core_data_type: CoreDataType, expected_bit_width: int | None
) -> None:
    """Test `get_core_data_type_bit_width` returns the expected widths."""
    assert get_core_data_type_bit_width(core_data_type) == expected_bit_width


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


# =============================================================================
# Promotion
# =============================================================================


def test_core_data_type_partitions_cover_every_member() -> None:
    """Test the integer, float/complex, and BOOL partitions cover CoreDataType.

    ``promote_core_data_types`` treats each member as integer-partitioned,
    float/complex-partitioned, or BOOL (its own side branch). A new
    ``CoreDataType`` member added without slotting it into a partition would
    otherwise surface only as a downstream promotion gap; this makes the
    drift a deterministic failure.
    """
    covered = _INTEGER_DATA_TYPES | _FLOAT_COMPLEX_DATA_TYPES | {CoreDataType.BOOL}

    assert covered == frozenset(CoreDataType)


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
        (CoreDataType.FLOAT16, CoreDataType.FLOAT32, CoreDataType.FLOAT32),
        (CoreDataType.FLOAT64, CoreDataType.FLOAT16, CoreDataType.FLOAT64),
        (CoreDataType.COMPLEX32, CoreDataType.COMPLEX64, CoreDataType.COMPLEX64),
        (CoreDataType.FLOAT32, CoreDataType.COMPLEX32, CoreDataType.COMPLEX64),
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
def test_promote_core_data_types(
    core_data_type1: CoreDataType,
    core_data_type2: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Test core data types are correctly promoted within their families."""
    assert (
        promote_core_data_types(core_data_type1, core_data_type2)
        == expected_core_data_type
    )


@pytest.mark.parametrize(
    ("core_data_type1", "core_data_type2"),
    [
        (CoreDataType.INT32, CoreDataType.FLOAT32),
        (CoreDataType.UINT8, CoreDataType.COMPLEX64),
        (CoreDataType.FLOAT, CoreDataType.INT),
    ],
)
def test_promote_core_data_types_raises_for_cross_family_pairs(
    core_data_type1: CoreDataType, core_data_type2: CoreDataType
) -> None:
    """Test cross-family promotion raises `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError, match="Unsupported"):
        promote_core_data_types(core_data_type1, core_data_type2)


def test_promote_primitive_data_types_within_family() -> None:
    """Test `promote_primitive_data_types` returns the lattice LUB."""
    int8 = PrimitiveDataType(CoreDataType.INT8)
    int32 = PrimitiveDataType(CoreDataType.INT32)
    promoted = promote_primitive_data_types(int8, int32)

    assert isinstance(promoted, PrimitiveDataType)
    assert promoted.core_data_type == CoreDataType.INT32


def test_promote_primitive_data_types_raises_for_cross_family_pairs() -> None:
    """Test cross-family `promote_primitive_data_types` raises `FhYCoreTypeError`."""
    int32 = PrimitiveDataType(CoreDataType.INT32)
    float32 = PrimitiveDataType(CoreDataType.FLOAT32)

    with pytest.raises(FhYCoreTypeError, match="Unsupported"):
        promote_primitive_data_types(int32, float32)


# =============================================================================
# Literal resolution
# =============================================================================


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
        (1.5, CoreDataType.FLOAT, CoreDataType.FLOAT64),
        (3.141592653589793, CoreDataType.FLOAT, CoreDataType.FLOAT64),
    ],
)
def test_resolve_literal_core_data_type(
    literal: int | float,
    core_data_type: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Weak literal types should resolve to a concrete type compatible with context.

    Integer literals resolve to the narrowest concrete integer type that
    represents the value. Float literals paired with the weak ``FLOAT``
    context resolve to ``FLOAT64`` (Python's native float precision) so
    that decimal precision is not silently lost; narrower float widths
    must be requested explicitly via the context.
    """
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


def test_resolve_literal_core_data_type_resolves_bool_against_bool_context() -> None:
    """Test `bool` literals resolve to `BOOL` when checked against `BOOL`."""
    assert resolve_literal_core_data_type(True, CoreDataType.BOOL) == CoreDataType.BOOL
    assert resolve_literal_core_data_type(False, CoreDataType.BOOL) == CoreDataType.BOOL


@pytest.mark.parametrize(
    "core_data_type",
    [
        CoreDataType.INT32,
        CoreDataType.UINT8,
        CoreDataType.FLOAT64,
        CoreDataType.COMPLEX64,
    ],
)
def test_resolve_literal_core_data_type_rejects_bool_against_numeric_context(
    core_data_type: CoreDataType,
) -> None:
    """Test `bool` literals against numeric contexts raise `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError):
        resolve_literal_core_data_type(True, core_data_type)


@pytest.mark.parametrize("literal", [0, 1, -1, 3.14])
def test_resolve_literal_core_data_type_rejects_numeric_against_bool_context(
    literal: int | float,
) -> None:
    """Test numeric literals against the `BOOL` context raise `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError):
        resolve_literal_core_data_type(literal, CoreDataType.BOOL)


# =============================================================================
# BOOL core data type
# =============================================================================


def test_bool_core_data_type_enum_member_exists() -> None:
    """Test `CoreDataType.BOOL` is a member of the enum with string value `bool`."""
    assert CoreDataType.BOOL.value == "bool"


def test_bool_core_data_type_bit_width_is_one() -> None:
    """Test the bit width of `BOOL` is reported as `1` (one-bit value)."""
    assert get_core_data_type_bit_width(CoreDataType.BOOL) == 1


def test_bool_core_data_type_is_not_weak() -> None:
    """Test `BOOL` is not classified as a weak literal type."""
    assert is_weak_core_data_type(CoreDataType.BOOL) is False


def test_promote_core_data_types_bool_with_bool_returns_bool() -> None:
    """Test promoting `BOOL` with itself yields `BOOL`."""
    assert (
        promote_core_data_types(CoreDataType.BOOL, CoreDataType.BOOL)
        == CoreDataType.BOOL
    )


@pytest.mark.parametrize(
    "other",
    [
        CoreDataType.INT,
        CoreDataType.INT32,
        CoreDataType.UINT,
        CoreDataType.UINT8,
        CoreDataType.FLOAT,
        CoreDataType.FLOAT64,
        CoreDataType.COMPLEX64,
    ],
)
def test_promote_core_data_types_bool_with_non_bool_raises(
    other: CoreDataType,
) -> None:
    """Test promoting `BOOL` with any non-`BOOL` core data type raises."""
    with pytest.raises(FhYCoreTypeError):
        promote_core_data_types(CoreDataType.BOOL, other)
    with pytest.raises(FhYCoreTypeError):
        promote_core_data_types(other, CoreDataType.BOOL)


def test_promote_primitive_data_types_bool_with_bool_returns_bool_primitive() -> None:
    """Test promoting two `BOOL` primitives yields a `BOOL` primitive."""
    bool_data_type = PrimitiveDataType(CoreDataType.BOOL)
    promoted = promote_primitive_data_types(bool_data_type, bool_data_type)
    assert isinstance(promoted, PrimitiveDataType)
    assert promoted.core_data_type == CoreDataType.BOOL


def test_promote_primitive_data_types_bool_with_int_raises() -> None:
    """Test `BOOL` and any non-`BOOL` primitive do not promote together."""
    bool_data_type = PrimitiveDataType(CoreDataType.BOOL)
    int32 = PrimitiveDataType(CoreDataType.INT32)
    with pytest.raises(FhYCoreTypeError):
        promote_primitive_data_types(bool_data_type, int32)


def test_resolve_literal_core_data_type_rejects_float_against_integer_target() -> None:
    """Test float literals against integer targets raise `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError, match="incompatible"):
        resolve_literal_core_data_type(1.5, CoreDataType.INT32)


def test_resolve_literal_core_data_type_rejects_positive_overflow_in_uint() -> None:
    """Test a positive literal exceeding UINT32 raises `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError, match="uint"):
        resolve_literal_core_data_type(2**33, CoreDataType.UINT)


def test_resolve_literal_core_data_type_rejects_negative_overflow_in_int() -> None:
    """Test a negative literal exceeding INT64 range raises `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError, match="int"):
        resolve_literal_core_data_type(-(2**70), CoreDataType.INT)


def test_resolve_literal_core_data_type_rejects_positive_overflow_in_int() -> None:
    """Test a positive literal exceeding INT64 range raises `FhYCoreTypeError`."""
    with pytest.raises(FhYCoreTypeError, match="int"):
        resolve_literal_core_data_type(2**70, CoreDataType.INT)


# =============================================================================
# Type qualifiers
# =============================================================================


@pytest.mark.parametrize(
    ("type_qualifier1", "type_qualifier2", "expected_type_qualifier"),
    [
        (TypeQualifier.INPUT, TypeQualifier.INPUT, TypeQualifier.TEMP),
        (TypeQualifier.STATE, TypeQualifier.PARAM, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.TEMP, TypeQualifier.TEMP),
        (TypeQualifier.PARAM, TypeQualifier.PARAM, TypeQualifier.PARAM),
        (TypeQualifier.OUTPUT, TypeQualifier.PARAM, TypeQualifier.TEMP),
        (TypeQualifier.OUTPUT, TypeQualifier.OUTPUT, TypeQualifier.TEMP),
    ],
)
def test_promote_type_qualifiers(
    type_qualifier1: TypeQualifier,
    type_qualifier2: TypeQualifier,
    expected_type_qualifier: TypeQualifier,
) -> None:
    """Test type qualifiers promote to PARAM only when both inputs are PARAM."""
    assert (
        promote_type_qualifiers(type_qualifier1, type_qualifier2)
        == expected_type_qualifier
    )
