"""Tests for `fhy_core.types.checking.type_checker`.

Every assertion on a returned ``Type`` goes through
``Type.is_structurally_equivalent`` against a built-up expected
``NumericalType(PrimitiveDataType(core_data_type))`` or ``IndexType(...)`` so
the full ``(data_type, shape)`` / ``(lower_bound, upper_bound, stride)``
contract is compared on every case.
"""

from unittest.mock import Mock

import pytest

from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    pformat_expression,
)
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TemplateDataType,
    TypeQualifier,
)
from fhy_core.types.checking.type_checker import (
    ExpressionTypeChecker,
    _get_numeric_literal_value,
    _get_primitive_data_type,
    _get_real_float_core_data_type_for_bit_width,
    _TypeCheckContext,
    check_expression_type,
    get_core_data_type_from_literal_type,
    synthesize_expression_type,
)

from .conftest import (
    make_identifier_checker,
    make_single_type_checker,
    mock_identifier,
)


def _make_scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _make_tensor(core_data_type: CoreDataType, *shape: int) -> NumericalType:
    return NumericalType(
        PrimitiveDataType(core_data_type), [LiteralExpression(dim) for dim in shape]
    )


_ALL_TYPE_QUALIFIERS = tuple(TypeQualifier)
_NON_OUTPUT_TYPE_QUALIFIERS = tuple(
    qualifier for qualifier in TypeQualifier if qualifier is not TypeQualifier.OUTPUT
)


# =============================================================================
# get_core_data_type_from_literal_type
# =============================================================================


@pytest.mark.parametrize(
    "literal, expected_core_data_type",
    [
        (1, CoreDataType.UINT),
        (0, CoreDataType.UINT),
        (-1, CoreDataType.INT),
        (1.5, CoreDataType.FLOAT),
    ],
)
def test_get_core_data_type_from_literal_type_returns_weak_types(
    literal: int | float, expected_core_data_type: CoreDataType
) -> None:
    """Test `get_core_data_type_from_literal_type` maps numerics to weak core types."""
    assert get_core_data_type_from_literal_type(literal) is expected_core_data_type


def test_get_core_data_type_from_literal_type_rejects_string_literal() -> None:
    """Test string literals are rejected with `NotImplementedError`."""
    with pytest.raises(NotImplementedError):
        get_core_data_type_from_literal_type("1")


# =============================================================================
# Literal synthesis - weak typing & bidirectional check
# =============================================================================


def test_synthesize_string_literal_expression_is_rejected() -> None:
    """Test a string-form `LiteralExpression` is rejected during type synthesis."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))
    with pytest.raises(NotImplementedError):
        checker.visit(LiteralExpression("1.0"))


def test_unary_negation_of_positive_integer_literal_becomes_weak_signed_int() -> None:
    """Test negating a positive weak integer literal produces a weak signed int."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, result_qualifier = checker.visit(
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5))
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT))
    assert result_qualifier is TypeQualifier.PARAM


def test_unary_negate_of_zero_literal_stays_weak_unsigned_int() -> None:
    """Test negating a zero integer literal stays a weak unsigned int (not signed)."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(0))
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT))


def test_positive_integer_literal_upgrades_to_signed_context() -> None:
    """Test a positive integer literal adopts the surrounding signed strong type."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))
    assert result_qualifier is TypeQualifier.PARAM


def test_synthesize_large_positive_integer_literal_stays_weak_unsigned() -> None:
    """Test a positive integer above 32-bit range still synthesizes as weak ``UINT``.

    Without a surrounding context, no automatic bit-width selection happens;
    the literal stays as the weak ``UINT`` core type and only narrows when
    placed in a ``check`` flow with a concrete expected type.
    """
    result_type, _ = synthesize_expression_type(
        LiteralExpression(2**40),
        lambda _: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT))


def test_integer_literals_remain_weak_without_context() -> None:
    """Test two integer literals without context stay weakly typed."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT))
    assert result_qualifier is TypeQualifier.PARAM


def test_integer_literal_can_upgrade_to_float_context() -> None:
    """Test a positive integer literal adopts the surrounding float context."""
    identifier = mock_identifier("y", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.FLOAT32), TypeQualifier.PARAM)}
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT32))
    assert result_qualifier is TypeQualifier.PARAM


def test_check_literal_against_expected_type_resolves_weak_literal() -> None:
    """Test `check` against a concrete integer type adopts that type for the literal."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, result_qualifier = checker.check(
        LiteralExpression(256), _make_scalar(CoreDataType.UINT16)
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT16))
    assert result_qualifier is TypeQualifier.PARAM


def test_check_integer_literal_against_float_context_uses_context_type() -> None:
    """Test `check` preserves a concrete float context when given an integer literal."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, _ = checker.check(
        LiteralExpression(1), _make_scalar(CoreDataType.FLOAT32)
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT32))


def test_check_propagates_expected_type_to_two_weak_literal_operands() -> None:
    """Test `check` propagates the expected type into both literal operands.

    With both sides starting as weak literals, the late-stage recheck logic
    inside ``_infer_binary_expression`` cannot rescue them - one side has to
    already be non-weak for the rescue to fire. This exercises the early
    propagation that hands the expected type to each literal up front.
    """
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, _ = checker.check(
        BinaryExpression(
            BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(3)
        ),
        _make_scalar(CoreDataType.INT8),
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT8))


def test_check_positive_literal_overflowing_narrow_unsigned_expected_raises() -> None:
    """Test `check` rejects a positive literal that exceeds the expected width."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(
        FhYCoreTypeError,
        match=r"is wider than the expected type uint8",
    ):
        checker.check(LiteralExpression(256), _make_scalar(CoreDataType.UINT8))


def test_check_negative_literal_against_unsigned_expected_raises() -> None:
    """Test `check` rejects a negative literal against an unsigned expected type."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(
        FhYCoreTypeError,
        match=r"Literal -1 is incompatible with uint16",
    ):
        checker.check(LiteralExpression(-1), _make_scalar(CoreDataType.UINT16))


def test_check_binary_expression_uses_expected_type_bidirectionally() -> None:
    """Test an arithmetic expression adopts the expected type bidirectionally."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, _ = checker.check(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        ),
        _make_scalar(CoreDataType.UINT16),
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT16))


def test_check_binary_hint_reaches_left_literal_operand() -> None:
    """Test `check` propagates the expected type into a literal left operand."""
    identifier = mock_identifier("right", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.UINT16), TypeQualifier.PARAM)}
    )

    result_type, _ = checker.check(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(5),
            IdentifierExpression(identifier),
        ),
        _make_scalar(CoreDataType.UINT16),
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT16))


def test_check_binary_hint_reaches_right_literal_operand() -> None:
    """Test `check` propagates the expected type into a literal right operand."""
    identifier = mock_identifier("left", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.UINT16), TypeQualifier.PARAM)}
    )

    result_type, _ = checker.check(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(5),
        ),
        _make_scalar(CoreDataType.UINT16),
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT16))


# =============================================================================
# check: expected-type validation
# =============================================================================


def test_check_accepts_expression_exactly_matching_expected() -> None:
    """Test `check` returns normally when the synthesized type equals the expected."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.UINT16), TypeQualifier.PARAM)}
    )

    result_type, _ = checker.check(
        IdentifierExpression(identifier), _make_scalar(CoreDataType.UINT16)
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.UINT16))


@pytest.mark.parametrize(
    "actual_core_data_type, expected_core_data_type",
    [
        (CoreDataType.INT16, CoreDataType.UINT8),
        (CoreDataType.UINT16, CoreDataType.UINT8),
        (CoreDataType.INT32, CoreDataType.INT8),
        (CoreDataType.FLOAT32, CoreDataType.FLOAT16),
    ],
)
def test_check_rejects_expression_wider_than_expected(
    actual_core_data_type: CoreDataType, expected_core_data_type: CoreDataType
) -> None:
    """Test `check` rejects an expression wider than its expected narrower type."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(actual_core_data_type), TypeQualifier.PARAM)}
    )

    with pytest.raises(FhYCoreTypeError, match=r"is wider than the expected type"):
        checker.check(
            IdentifierExpression(identifier), _make_scalar(expected_core_data_type)
        )


def test_check_index_against_equivalent_index_succeeds() -> None:
    """Test `check` passes when actual and expected are equal index types."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.check(
        IdentifierExpression(identifier),
        IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1)),
    )

    assert result_type.is_structurally_equivalent(index)


def test_check_index_against_nonequivalent_index_raises() -> None:
    """Test `check` raises when actual and expected are different index types."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"is not structurally equivalent to the expected index type",
    ):
        checker.check(
            IdentifierExpression(identifier),
            IndexType(
                LiteralExpression(2), LiteralExpression(10), LiteralExpression(1)
            ),
        )


def test_check_index_against_numerical_raises() -> None:
    """Test `check` rejects an index-typed expression against a numerical type."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"one is an index type and the other is a numerical type",
    ):
        checker.check(
            IdentifierExpression(identifier), _make_scalar(CoreDataType.INT32)
        )


def test_check_numerical_against_index_raises() -> None:
    """Test `check` rejects a numerical expression against an index expected type."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(
        FhYCoreTypeError,
        match=r"a literal value cannot be checked against index type",
    ):
        checker.check(
            LiteralExpression(5),
            IndexType(
                LiteralExpression(1), LiteralExpression(10), LiteralExpression(1)
            ),
        )


# =============================================================================
# Numerical addition promotion - bit-width and sign edges
# =============================================================================


def test_addition_of_mixed_bit_width_signed_integers_promotes_to_wider_width() -> None:
    """Test ``INT8 + INT64`` synthesizes as ``INT64``."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT64), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT64))


def test_addition_of_unsigned_and_signed_same_width_promotes_to_wider_signed() -> None:
    """Test ``UINT32 + INT32`` widens to ``INT64`` to fit both operand ranges."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.UINT32), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT64))


# =============================================================================
# Division & floor-division core-data-type promotion
# =============================================================================


@pytest.mark.parametrize(
    "operand_core_data_type, expected_float_core_data_type",
    [
        (CoreDataType.UINT8, CoreDataType.FLOAT16),
        (CoreDataType.INT8, CoreDataType.FLOAT16),
        (CoreDataType.UINT16, CoreDataType.FLOAT16),
        (CoreDataType.INT16, CoreDataType.FLOAT16),
        (CoreDataType.UINT32, CoreDataType.FLOAT32),
        (CoreDataType.INT32, CoreDataType.FLOAT32),
        (CoreDataType.INT64, CoreDataType.FLOAT64),
    ],
)
def test_integer_division_promotes_to_smallest_sufficient_float_width(
    operand_core_data_type: CoreDataType,
    expected_float_core_data_type: CoreDataType,
) -> None:
    """Test same-type integer division promotes to the smallest float >= that width."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(operand_core_data_type), TypeQualifier.PARAM),
            right: (_make_scalar(operand_core_data_type), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(
        _make_scalar(expected_float_core_data_type)
    )


def test_division_of_strong_integers_produces_concrete_float() -> None:
    """Test division of two `INT32` identifiers yields a concrete `FLOAT32` type."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT32))
    assert result_qualifier is TypeQualifier.PARAM


def test_weak_integer_division_produces_weak_float() -> None:
    """Test division of two weak integer literals yields a weak `FLOAT` type."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(4),
            LiteralExpression(2),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT))


@pytest.mark.parametrize(
    "left_core_data_type, right_core_data_type, expected_core_data_type",
    [
        (CoreDataType.INT16, CoreDataType.FLOAT32, CoreDataType.FLOAT32),
        (CoreDataType.FLOAT32, CoreDataType.INT16, CoreDataType.FLOAT32),
    ],
)
def test_division_mixed_int_and_float_promotes_to_float(
    left_core_data_type: CoreDataType,
    right_core_data_type: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Test mixed-integer-and-float division promotes to the dominant float type."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(left_core_data_type), TypeQualifier.PARAM),
            right: (_make_scalar(right_core_data_type), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(expected_core_data_type))


@pytest.mark.parametrize(
    "left_core_data_type, right_core_data_type, expected_core_data_type",
    [
        (CoreDataType.INT16, CoreDataType.COMPLEX64, CoreDataType.COMPLEX64),
        (CoreDataType.COMPLEX64, CoreDataType.INT16, CoreDataType.COMPLEX64),
    ],
)
def test_division_mixed_int_and_complex_promotes_to_complex(
    left_core_data_type: CoreDataType,
    right_core_data_type: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Test mixed-integer-and-complex division promotes to the dominant complex type."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(left_core_data_type), TypeQualifier.PARAM),
            right: (_make_scalar(right_core_data_type), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(expected_core_data_type))


def test_unsigned_and_signed_integers_share_integral_promotion_under_division() -> None:
    """Test `UINT8 / INT8` promotes integral-to-integral, then picks `FLOAT16`."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.UINT8), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT8), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT16))


def test_floor_division_of_integers_produces_integer_type() -> None:
    """Test floor division of two `INT32` identifiers yields `INT32`."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


@pytest.mark.parametrize(
    "left_core_data_type, right_core_data_type, expected_core_data_type",
    [
        (CoreDataType.FLOAT32, CoreDataType.INT32, CoreDataType.FLOAT32),
        (CoreDataType.INT32, CoreDataType.FLOAT32, CoreDataType.FLOAT32),
    ],
)
def test_floor_division_mixed_int_and_float_promotes_to_float(
    left_core_data_type: CoreDataType,
    right_core_data_type: CoreDataType,
    expected_core_data_type: CoreDataType,
) -> None:
    """Test floor division with one float operand promotes to a float result type."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(left_core_data_type), TypeQualifier.PARAM),
            right: (_make_scalar(right_core_data_type), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(expected_core_data_type))


def test_floor_division_int64_by_float16_widens_to_max_bit_width_float() -> None:
    """Test ``INT64 // FLOAT16`` promotes to ``FLOAT64`` to honor the wider operand."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.INT64), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.FLOAT16), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT64))


@pytest.mark.parametrize(
    "left_core_data_type, right_core_data_type",
    [
        (CoreDataType.INT16, CoreDataType.COMPLEX64),
        (CoreDataType.COMPLEX64, CoreDataType.INT16),
    ],
)
def test_floor_division_involving_complex_is_rejected(
    left_core_data_type: CoreDataType, right_core_data_type: CoreDataType
) -> None:
    """Test floor division with any complex operand is rejected."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(left_core_data_type), TypeQualifier.PARAM),
            right: (_make_scalar(right_core_data_type), TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"floor division is not defined for complex numerical types",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


# =============================================================================
# Tensor & tuple type rejection
# =============================================================================


def test_tensor_type_is_rejected_in_expression_typing() -> None:
    """Test a tensor-typed operand in a compound expression is rejected."""
    identifier = mock_identifier("tensor", 0)
    checker = make_identifier_checker(
        {identifier: (_make_tensor(CoreDataType.FLOAT32, 4), TypeQualifier.PARAM)}
    )

    with pytest.raises(
        NotImplementedError,
        match=r"resolves to tensor type .* only scalar numerical and index types",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(identifier),
                LiteralExpression(1),
            )
        )


def test_tensor_identifier_is_rejected_immediately() -> None:
    """Test a tensor-typed identifier is rejected outside a compound expression."""
    identifier = mock_identifier("tensor", 0)
    checker = make_identifier_checker(
        {identifier: (_make_tensor(CoreDataType.FLOAT32, 4), TypeQualifier.PARAM)}
    )

    with pytest.raises(NotImplementedError, match=r"resolves to tensor type"):
        checker.visit(IdentifierExpression(identifier))


# =============================================================================
# TypeQualifier gating
# =============================================================================


def test_output_qualifier_identifier_is_rejected_on_read() -> None:
    """Test reading an identifier with an ``output`` qualifier is rejected."""
    identifier = mock_identifier("out", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.OUTPUT)}
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r'type qualifier "output" and cannot be read from',
    ):
        checker.visit(IdentifierExpression(identifier))


@pytest.mark.parametrize("qualifier", _NON_OUTPUT_TYPE_QUALIFIERS)
def test_non_output_qualifier_identifier_is_accepted_on_read(
    qualifier: TypeQualifier,
) -> None:
    """Test an identifier with any non-``output`` qualifier is accepted on read."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), qualifier)}
    )

    result_type, result_qualifier = checker.visit(IdentifierExpression(identifier))

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))
    assert result_qualifier is qualifier


def test_aliased_input_identifier_promotes_qualifier_through_addition() -> None:
    """Test ``x + x`` for a single ``INPUT`` identifier yields the promoted qualifier.

    Reusing the same identifier on both sides of a binary expression must not
    confuse the context tracker; the synthesized qualifier should follow the
    documented promotion rule for ``INPUT + INPUT``.
    """
    x = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {x: (_make_scalar(CoreDataType.INT32), TypeQualifier.INPUT)}
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(x),
            IdentifierExpression(x),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))
    assert result_qualifier is TypeQualifier.TEMP


@pytest.mark.parametrize("qualifier", _NON_OUTPUT_TYPE_QUALIFIERS)
def test_non_output_qualifier_operand_is_accepted_in_unary_expression(
    qualifier: TypeQualifier,
) -> None:
    """Test a non-``output`` operand is accepted inside a unary expression."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), qualifier)}
    )

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


# =============================================================================
# Index arithmetic - shift by scalar
# =============================================================================


def test_index_plus_scalar_produces_shifted_index_type() -> None:
    """Test adding an integer scalar to an index shifts both bounds."""
    identifier = mock_identifier("idx", 0)
    upper = mock_identifier("N", 1)
    index = IndexType(
        LiteralExpression(1),
        IdentifierExpression(upper),
        LiteralExpression(2),
    )
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(3),
        IdentifierExpression(upper) + LiteralExpression(3),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)
    assert result_qualifier is TypeQualifier.PARAM


def test_scalar_plus_index_produces_shifted_index_type() -> None:
    """Test adding a scalar on the left also shifts both bounds."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(2),
            IdentifierExpression(identifier),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(2),
        LiteralExpression(10) + LiteralExpression(2),
        LiteralExpression(1),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_minus_scalar_produces_shifted_index_type() -> None:
    """Test subtracting a scalar from an index shifts both bounds downward."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(
        LiteralExpression(2),
        LiteralExpression(12),
        LiteralExpression(3),
    )
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.SUBTRACT,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    expected = IndexType(
        LiteralExpression(2) - LiteralExpression(1),
        LiteralExpression(12) - LiteralExpression(1),
        LiteralExpression(3),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_with_negative_stride_shifts_bounds_and_preserves_stride() -> None:
    """Test shifting a negative-stride index leaves the stride untouched."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(
        LiteralExpression(10), LiteralExpression(0), LiteralExpression(-1)
    )
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected = IndexType(
        LiteralExpression(10) + LiteralExpression(3),
        LiteralExpression(0) + LiteralExpression(3),
        LiteralExpression(-1),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_with_negative_stride_scales_bounds_and_stride() -> None:
    """Test scaling a negative-stride index multiplies bounds and stride together."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(
        LiteralExpression(10), LiteralExpression(0), LiteralExpression(-1)
    )
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(identifier),
            LiteralExpression(2),
        )
    )

    expected = IndexType(
        LiteralExpression(2) * LiteralExpression(10),
        LiteralExpression(2) * LiteralExpression(0),
        LiteralExpression(-2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_float_is_rejected() -> None:
    """Test adding a float scalar to an index is rejected."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index shift requires an integral scalar offset, "
        r"but the right operand has type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(identifier),
                LiteralExpression(1.5),
            )
        )


# =============================================================================
# Unary operations on index types
# =============================================================================


def test_unary_positive_index_preserves_index_type() -> None:
    """Test unary positive preserves the exact index type."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    )

    assert result_type.is_structurally_equivalent(index)


def test_unary_negation_of_index_is_rejected() -> None:
    """Test unary negation of an index type is rejected."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"unary negation is not defined for index types",
    ):
        checker.visit(
            UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(identifier))
        )


# =============================================================================
# Index scaling
# =============================================================================


def test_index_times_positive_integer_literal_scales_bounds_and_stride() -> None:
    """Test scaling an index with an explicit stride multiplies bounds and stride."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(2))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected = IndexType(
        LiteralExpression(3) * LiteralExpression(1),
        LiteralExpression(3) * LiteralExpression(8),
        LiteralExpression(6),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_positive_integer_literal_times_index_scales_symmetrically() -> None:
    """Test scalar-times-index scales the same way as index-times-scalar."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(3),
            IdentifierExpression(identifier),
        )
    )

    expected = IndexType(
        LiteralExpression(3) * LiteralExpression(1),
        LiteralExpression(3) * LiteralExpression(8),
        LiteralExpression(3),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_scaling_by_one_preserves_unit_stride() -> None:
    """Test scaling an index with unit stride by ``1`` yields a unit-stride index."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(1),
            IdentifierExpression(identifier),
        )
    )

    expected = IndexType(
        LiteralExpression(1) * LiteralExpression(1),
        LiteralExpression(1) * LiteralExpression(8),
        LiteralExpression(1),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_scaling_with_non_literal_stride_emits_scalar_times_stride() -> None:
    """Test scaling an index with a non-literal stride emits ``scalar * stride``."""
    identifier = mock_identifier("idx", 0)
    stride_identifier = mock_identifier("s", 1)
    index = IndexType(
        LiteralExpression(1),
        LiteralExpression(8),
        IdentifierExpression(stride_identifier),
    )
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected = IndexType(
        LiteralExpression(3) * LiteralExpression(1),
        LiteralExpression(3) * LiteralExpression(8),
        LiteralExpression(3) * IdentifierExpression(stride_identifier),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_times_zero_literal_is_rejected() -> None:
    """Test scaling an index by `0` is rejected (positive-integer requirement)."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index scaling requires a positive integer literal scalar, "
        r"but got scalar value 0",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(0),
            )
        )


def test_index_times_negative_literal_is_rejected() -> None:
    """Test scaling an index by a negative integer literal is rejected."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index scaling requires a positive integer literal scalar, "
        r"but got scalar value -3",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(-3),
            )
        )


def test_index_times_non_literal_scalar_is_rejected() -> None:
    """Test scaling an index by a non-literal integer identifier is rejected."""
    identifier = mock_identifier("idx", 0)
    scalar = mock_identifier("k", 1)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker(
        {
            identifier: (index, TypeQualifier.PARAM),
            scalar: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index scaling requires a positive integer literal scalar, "
        r"but the left operand .* is not a literal expression",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(scalar),
                IdentifierExpression(identifier),
            )
        )


def test_index_times_float_literal_is_rejected() -> None:
    """Test scaling an index by a float literal is rejected."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index scaling requires a positive integer literal scalar, "
        r"but the right operand has non-integral type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(2.0),
            )
        )


# =============================================================================
# Division and other non-additive operations on index types
# =============================================================================


def test_index_division_is_rejected_with_specific_message() -> None:
    """Test dividing an index by a scalar raises a division-specific message."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"division is not defined for operands of index type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(identifier),
                LiteralExpression(2),
            )
        )


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.DIVIDE, BinaryOperation.FLOOR_DIVIDE],
)
@pytest.mark.parametrize("index_on", ["left", "right"])
def test_division_with_one_index_operand_uses_division_specific_message(
    operation: BinaryOperation, index_on: str
) -> None:
    """Test mixed index/numerical division raises a division-specific error message."""
    index_identifier = mock_identifier("idx", 0)
    scalar_identifier = mock_identifier("k", 1)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker(
        {
            index_identifier: (index, TypeQualifier.PARAM),
            scalar_identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )
    index_expression = IdentifierExpression(index_identifier)
    scalar_expression = IdentifierExpression(scalar_identifier)
    if index_on == "left":
        expression = BinaryExpression(operation, index_expression, scalar_expression)
    else:
        expression = BinaryExpression(operation, scalar_expression, index_expression)

    with pytest.raises(
        FhYCoreTypeError,
        match=r"division is not defined for operands of index type",
    ):
        checker.visit(expression)


# =============================================================================
# Index op index (rejected)
# =============================================================================
#
# The type checker rejects every binary operation between two index types:
# arithmetic on two strided ranges does not in general produce a result
# whose bounds and stride can be inferred safely as another ``IndexType``.
# Shift (``index +/- int``) and scale (``index * int literal``) remain
# supported and are exercised in adjacent sections of this file.


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.ADD,
        BinaryOperation.SUBTRACT,
        BinaryOperation.MULTIPLY,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    ],
)
def test_index_index_binary_operation_is_rejected(
    operation: BinaryOperation,
) -> None:
    """Test every binary operation between two ``IndexType`` operands is rejected.

    ``DIVIDE`` and ``FLOOR_DIVIDE`` hit an earlier, more specific
    division-not-defined branch and are tested separately.
    """
    left = mock_identifier("i", 0)
    right = mock_identifier("j", 1)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker(
        {
            left: (index, TypeQualifier.PARAM),
            right: (index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError, match="between two index types"):
        checker.visit(
            BinaryExpression(
                operation,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.DIVIDE, BinaryOperation.FLOOR_DIVIDE],
)
def test_index_index_division_uses_specific_division_message(
    operation: BinaryOperation,
) -> None:
    """Test ``DIVIDE``/``FLOOR_DIVIDE`` between indices use the division-error path."""
    left = mock_identifier("i", 0)
    right = mock_identifier("j", 1)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker(
        {
            left: (index, TypeQualifier.PARAM),
            right: (index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError, match="division is not defined for operands of index type"
    ):
        checker.visit(
            BinaryExpression(
                operation,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


# =============================================================================
# Error-message context tracking
#
# Every type error the checker raises begins with "Type error while inferring
# type of `<root>`" where `<root>` is the top-level expression passed to
# synthesize / check, and -- when the failing sub-expression is not the root
# itself -- includes " at sub-expression `<sub>`". These tests pin that
# framing so future edits to the error-reporting machinery cannot regress
# context information without being caught.
# =============================================================================


def test_type_error_includes_root_expression_for_top_level_failure() -> None:
    """Test the error message frames the failure with the root expression."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})
    expression = UnaryExpression(
        UnaryOperation.NEGATE, IdentifierExpression(identifier)
    )

    with pytest.raises(FhYCoreTypeError) as exc_info:
        checker.visit(expression)

    message = str(exc_info.value)
    assert message.startswith("Type error while inferring type of `")
    assert pformat_expression(expression, show_id=True) in message
    # The failure surfaces at the root, so no "at sub-expression" framing.
    assert "at sub-expression" not in message


def test_type_error_includes_sub_expression_for_deeply_nested_failure() -> None:
    """Test the error message names the sub-expression where the failure surfaced."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})
    inner = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(identifier))
    root = UnaryExpression(UnaryOperation.POSITIVE, inner)

    with pytest.raises(FhYCoreTypeError) as exc_info:
        checker.visit(root)

    message = str(exc_info.value)
    assert message.startswith("Type error while inferring type of `")
    assert pformat_expression(root, show_id=True) in message
    assert "at sub-expression" in message
    assert pformat_expression(inner, show_id=True) in message


def test_type_error_from_check_includes_root_expression() -> None:
    """Test `check` failures are framed by the root expression, not just `_infer`."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})
    expression = IdentifierExpression(identifier)
    expected = IndexType(
        LiteralExpression(2), LiteralExpression(10), LiteralExpression(1)
    )

    with pytest.raises(FhYCoreTypeError) as exc_info:
        checker.check(expression, expected)

    message = str(exc_info.value)
    assert message.startswith("Type error while inferring type of `")
    assert pformat_expression(expression, show_id=True) in message


# =============================================================================
# Behavioural fine-tuning - pinning specific code paths
# =============================================================================


def test_check_negate_of_weak_literal_against_strong_type_returns_strong() -> None:
    """Test ``check(NEGATE(Lit(5)), INT32)`` keeps the strong context type."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    actual_type, _ = checker.check(
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
        _make_scalar(CoreDataType.INT32),
    )

    assert actual_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


def test_check_arithmetic_of_literals_against_index_uses_general_message() -> None:
    """Test ``check(ADD(Lit, Lit), IndexType)`` reports the index/numerical mismatch."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(
        FhYCoreTypeError,
        match=r"one is an index type and the other is a numerical type",
    ):
        checker.check(
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(1),
                LiteralExpression(2),
            ),
            IndexType(
                LiteralExpression(1), LiteralExpression(10), LiteralExpression(1)
            ),
        )


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.MODULO, BinaryOperation.POWER],
)
def test_synthesize_non_additive_index_and_scalar_uses_operands_of_types_message(
    operation: BinaryOperation,
) -> None:
    """Test ``MODULO``/``POWER`` of index and numerical reports an operand-types err."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(FhYCoreTypeError, match=r"is not defined for operands of types"):
        checker.visit(
            BinaryExpression(
                operation,
                IdentifierExpression(identifier),
                LiteralExpression(5),
            )
        )


def test_division_complex_left_real_float_right_promotes_to_complex() -> None:
    """Test ``DIVIDE(complex64, float64)`` promotes to ``complex128``."""
    left = mock_identifier("c", 0)
    right = mock_identifier("f", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.COMPLEX64), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.FLOAT64), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.COMPLEX128))


def test_division_real_float_left_complex_right_promotes_to_complex() -> None:
    """Test ``DIVIDE(float64, complex64)`` promotes to ``complex128``."""
    left = mock_identifier("f", 0)
    right = mock_identifier("c", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.FLOAT64), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.COMPLEX64), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.COMPLEX128))


# =============================================================================
# Top-level convenience functions
# =============================================================================


def test_synthesize_expression_type_top_level_function() -> None:
    """Test `synthesize_expression_type` returns the synthesized type from the class."""
    identifier = mock_identifier("x", 0)
    lookup_type = _make_scalar(CoreDataType.INT32)

    result_type, result_qualifier = synthesize_expression_type(
        IdentifierExpression(identifier),
        lambda i: (lookup_type, TypeQualifier.PARAM),
    )

    assert result_type.is_structurally_equivalent(lookup_type)
    assert result_qualifier is TypeQualifier.PARAM


def test_check_expression_type_top_level_function() -> None:
    """Test `check_expression_type` returns the checked type from the class."""
    expected = _make_scalar(CoreDataType.UINT16)

    result_type, result_qualifier = check_expression_type(
        LiteralExpression(5),
        expected,
        lambda i: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
    )

    assert result_type.is_structurally_equivalent(expected)
    assert result_qualifier is TypeQualifier.PARAM


def test_checker_synthesize_method_routes_through_infer() -> None:
    """Test `ExpressionTypeChecker.synthesize` yields the same result as `visit`."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )

    synthesized_type, _ = checker.synthesize(IdentifierExpression(identifier))

    assert synthesized_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


# =============================================================================
# Bidirectional weak-literal hint on the LEFT operand
# =============================================================================


def test_weak_left_literal_upgrades_against_strong_right_identifier() -> None:
    """Test a weak left-literal adopts the right identifier's strong type."""
    right_identifier = mock_identifier("right", 0)
    checker = make_identifier_checker(
        {right_identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(5),
            IdentifierExpression(right_identifier),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))


# =============================================================================
# Index arithmetic - remaining non-integral / non-literal operand paths
# =============================================================================


def test_numerical_float_plus_index_is_rejected() -> None:
    """Test ``ADD(Lit(1.5), idx)`` rejects because the left operand isn't integral."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index shift requires an integral scalar offset, "
        r"but the left operand has type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(1.5),
                IdentifierExpression(identifier),
            )
        )


def test_index_minus_float_literal_is_rejected() -> None:
    """Test ``SUBTRACT(idx, Lit(1.5))`` rejects for a non-integral right operand."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index shift requires an integral scalar offset, "
        r"but the right operand has type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(identifier),
                LiteralExpression(1.5),
            )
        )


def test_multiply_index_by_non_literal_identifier_on_right_is_rejected() -> None:
    """Test ``MULTIPLY(idx, scalar_id)`` rejects because right is a non-literal."""
    index_identifier = mock_identifier("idx", 0)
    scalar_identifier = mock_identifier("k", 1)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker(
        {
            index_identifier: (index, TypeQualifier.PARAM),
            scalar_identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"but the right operand .* is not a literal expression",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(index_identifier),
                IdentifierExpression(scalar_identifier),
            )
        )


def test_multiply_float_literal_times_index_rejected_on_left() -> None:
    """Test ``MULTIPLY(Lit(2.0), idx)`` rejects because the left is non-integral."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(1))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"but the left operand has non-integral type",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(2.0),
                IdentifierExpression(identifier),
            )
        )


# =============================================================================
# Floor division on real-float operands skipping both int-side elif branches
# =============================================================================


def test_floor_division_of_two_real_floats_promotes_via_lattice() -> None:
    """Test floor-div of two real-float identifiers skips both int-side branches."""
    left = mock_identifier("left", 0)
    right = mock_identifier("right", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.FLOAT32), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.FLOAT64), TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.FLOAT64))


# =============================================================================
# Non-arithmetic binary operations and LOGICAL_NOT
# =============================================================================


def test_synthesize_logical_and_on_non_bool_identifiers_raises_type_error() -> None:
    """Test `LOGICAL_AND` rejects non-boolean operands with `FhYCoreTypeError`."""
    left = mock_identifier("a", 0)
    right = mock_identifier("b", 1)
    checker = make_identifier_checker(
        {
            left: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
            right: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


def test_synthesize_logical_not_on_non_bool_identifier_raises_type_error() -> None:
    """Test `LOGICAL_NOT` rejects a non-boolean operand with `FhYCoreTypeError`."""
    identifier = mock_identifier("p", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT, IdentifierExpression(identifier)
            )
        )


# =============================================================================
# Value-type resolution and literal checking corner cases
# =============================================================================


def test_synthesize_rejects_identifier_with_template_numerical_data_type() -> None:
    """Test an identifier bound to a template (non-primitive) data type is rejected."""
    identifier = mock_identifier("t", 0)
    template_type = NumericalType(TemplateDataType(mock_identifier("T", 1)))
    checker = make_identifier_checker(
        {identifier: (template_type, TypeQualifier.PARAM)}
    )

    with pytest.raises(
        FhYCoreTypeError, match=r"must resolve to a primitive numerical type"
    ):
        checker.visit(IdentifierExpression(identifier))


def test_check_bool_literal_against_numerical_type_rejects() -> None:
    """Test ``check(Lit(True), NumericalType)`` rejects the bool literal.

    Boolean does not promote with numerical core data types; checking a
    boolean literal against an integer type therefore fails.
    """
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(FhYCoreTypeError):
        checker.check(LiteralExpression(True), _make_scalar(CoreDataType.INT32))


def test_check_string_literal_against_numerical_type_rejects_via_resolver() -> None:
    """Test ``check(Lit("1.0"), NumericalType)`` rejects the string-form literal."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(FhYCoreTypeError, match=r"expected a numeric literal value"):
        checker.check(LiteralExpression("1.0"), _make_scalar(CoreDataType.INT32))


# =============================================================================
# Direct private-helper exercises for defensive guards
# =============================================================================


def test_get_core_data_type_from_literal_type_rejects_unsupported_type() -> None:
    """Test `get_core_data_type_from_literal_type` rejects container values."""
    with pytest.raises(ValueError, match=r"Unsupported literal type"):
        get_core_data_type_from_literal_type([1, 2, 3])  # type: ignore[arg-type]


def test_get_primitive_data_type_rejects_non_primitive_data_type() -> None:
    """Test `_get_primitive_data_type` rejects a numerical with a template data type."""
    numerical = NumericalType(TemplateDataType(mock_identifier("T", 0)))

    with pytest.raises(FhYCoreTypeError, match=r"expected a primitive data type"):
        _get_primitive_data_type(numerical)


def test_get_numeric_literal_value_rejects_bool_literal() -> None:
    """Test `_get_numeric_literal_value` rejects a boolean literal."""
    with pytest.raises(FhYCoreTypeError, match=r"expected a numeric literal value"):
        _get_numeric_literal_value(LiteralExpression(True))


def test_get_real_float_for_bit_width_raises_when_no_match() -> None:
    """Test `_get_real_float_core_data_type_for_bit_width` rejects oversized widths."""
    with pytest.raises(
        FhYCoreTypeError,
        match=r"no real float core data type found for bit width 256",
    ):
        _get_real_float_core_data_type_for_bit_width(256)


def test_type_check_context_type_error_with_empty_stack() -> None:
    """Test `_TypeCheckContext.type_error` builds a bare message when stack empty."""
    context = _TypeCheckContext()

    error = context.type_error("the reason")

    assert isinstance(error, FhYCoreTypeError)
    assert str(error) == "Type error: the reason"


def test_checker_get_noop_output_raises() -> None:
    """Test `ExpressionTypeChecker.get_noop_output` raises `PassExecutionError`."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))

    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        checker.get_noop_output(LiteralExpression(0))


def test_infer_rejects_unsupported_expression_subclass() -> None:
    """Test `_infer` raises `NotImplementedError` on unknown Expression subclass."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))
    unknown = Mock(spec=Expression)

    with pytest.raises(NotImplementedError, match=r"Unsupported expression type"):
        checker.synthesize(unknown)


# =============================================================================
# Index types with literal-zero stride are rejected at first contact
# =============================================================================


def test_synthesize_rejects_index_with_literal_zero_stride() -> None:
    """Test an identifier whose `IndexType` has stride literal `0` is rejected."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(0))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index type with stride `0` is not allowed",
    ):
        checker.visit(IdentifierExpression(identifier))


def test_check_rejects_index_with_literal_zero_stride() -> None:
    """Test `check` also rejects an `IndexType` with stride literal `0`."""
    identifier = mock_identifier("idx", 0)
    index = IndexType(LiteralExpression(1), LiteralExpression(8), LiteralExpression(0))
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})
    expected = IndexType(
        LiteralExpression(1), LiteralExpression(8), LiteralExpression(1)
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index type with stride `0` is not allowed",
    ):
        checker.check(IdentifierExpression(identifier), expected)


def test_index_arithmetic_with_zero_stride_operand_is_rejected() -> None:
    """Test arithmetic involving a stride-`0` index is rejected at the operand."""
    left = mock_identifier("i", 0)
    right = mock_identifier("j", 1)
    zero_stride = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(0)
    )
    nonzero_stride = IndexType(
        LiteralExpression(1), LiteralExpression(9), LiteralExpression(5)
    )
    checker = make_identifier_checker(
        {
            left: (zero_stride, TypeQualifier.PARAM),
            right: (nonzero_stride, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index type with stride `0` is not allowed",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


def test_unary_positive_on_zero_stride_index_is_rejected() -> None:
    """Test unary `+` of a stride-`0` index also surfaces the rejection."""
    identifier = mock_identifier("idx", 0)
    zero_stride = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(0)
    )
    checker = make_identifier_checker({identifier: (zero_stride, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"index type with stride `0` is not allowed",
    ):
        checker.visit(
            UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
        )


# =============================================================================
# Unknown UnaryOperation falls through to NotImplementedError
# =============================================================================


def test_unary_expression_with_unknown_operation_raises_not_implemented() -> None:
    """Test an `UnaryExpression` carrying an unknown operation is rejected."""
    identifier = mock_identifier("x", 0)
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), TypeQualifier.PARAM)}
    )
    unary = UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    object.__setattr__(unary, "operation", Mock(spec=UnaryOperation))

    with pytest.raises(NotImplementedError, match=r"unary operation"):
        checker.synthesize(unary)


# =============================================================================
# Pass registration name
# =============================================================================


def test_pass_registered_under_renamed_name() -> None:
    """Test the pass is registered under the `type_checker` name (was `type_check`)."""
    assert (
        ExpressionTypeChecker.get_pass_name() == "fhy_core.types.checking.type_checker"
    )
