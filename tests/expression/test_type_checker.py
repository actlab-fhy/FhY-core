"""Tests for `fhy_core.expression.passes.type_checker`.

Every assertion on a returned ``Type`` goes through
``Type.is_structurally_equivalent`` against a built-up expected
``NumericalType(PrimitiveDataType(core_data_type))`` or ``IndexType(...)`` so
the full ``(data_type, shape)`` / ``(lower_bound, upper_bound, stride)``
contract is compared on every case.
"""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    pformat_expression,
)
from fhy_core.expression.passes.type_checker import (
    get_core_data_type_from_literal_type,
)
from fhy_core.identifier import Identifier
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TupleType,
    TypeQualifier,
)

from .conftest import make_identifier_checker, make_single_type_checker


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


@pytest.mark.parametrize("literal", [True, False])
def test_get_core_data_type_from_literal_type_rejects_bool_literal(
    literal: bool,
) -> None:
    """Test boolean literals are rejected with `NotImplementedError`."""
    with pytest.raises(NotImplementedError):
        get_core_data_type_from_literal_type(literal)


def test_get_core_data_type_from_literal_type_rejects_string_literal() -> None:
    """Test string literals are rejected with `NotImplementedError`."""
    with pytest.raises(NotImplementedError):
        get_core_data_type_from_literal_type("1")


# =============================================================================
# Literal synthesis - weak typing & bidirectional check
# =============================================================================


def test_synthesize_bool_literal_expression_is_rejected() -> None:
    """Test a `LiteralExpression(True)` is rejected during type synthesis."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))
    with pytest.raises(NotImplementedError):
        checker.visit(LiteralExpression(True))


def test_synthesize_string_literal_expression_is_rejected() -> None:
    """Test a numeric-string `LiteralExpression` is rejected during type synthesis."""
    checker = make_single_type_checker(_make_scalar(CoreDataType.INT32))
    with pytest.raises(NotImplementedError):
        checker.visit(LiteralExpression("1"))


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
    identifier = Identifier("x")
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
    identifier = Identifier("y")
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
    identifier = Identifier("right")
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
    identifier = Identifier("left")
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
    identifier = Identifier("x")
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
    identifier = Identifier("x")
    checker = make_identifier_checker(
        {identifier: (_make_scalar(actual_core_data_type), TypeQualifier.PARAM)}
    )

    with pytest.raises(FhYCoreTypeError, match=r"is wider than the expected type"):
        checker.check(
            IdentifierExpression(identifier), _make_scalar(expected_core_data_type)
        )


def test_check_index_against_equivalent_index_succeeds() -> None:
    """Test `check` passes when actual and expected are equal index types."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.check(
        IdentifierExpression(identifier),
        IndexType(LiteralExpression(1), LiteralExpression(10), None),
    )

    assert result_type.is_structurally_equivalent(index)


def test_check_index_against_nonequivalent_index_raises() -> None:
    """Test `check` raises when actual and expected are different index types."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    with pytest.raises(
        FhYCoreTypeError,
        match=r"is not structurally equivalent to the expected index type",
    ):
        checker.check(
            IdentifierExpression(identifier),
            IndexType(LiteralExpression(2), LiteralExpression(10), None),
        )


def test_check_index_against_numerical_raises() -> None:
    """Test `check` rejects an index-typed expression against a numerical type."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
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
            IndexType(LiteralExpression(1), LiteralExpression(10), None),
        )


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
    """Test same-type integer division promotes to the smallest float ≥ that width."""
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    left = Identifier("left")
    right = Identifier("right")
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
    identifier = Identifier("tensor")
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
    identifier = Identifier("tensor")
    checker = make_identifier_checker(
        {identifier: (_make_tensor(CoreDataType.FLOAT32, 4), TypeQualifier.PARAM)}
    )

    with pytest.raises(NotImplementedError, match=r"resolves to tensor type"):
        checker.visit(IdentifierExpression(identifier))


def test_tuple_identifier_is_rejected_immediately() -> None:
    """Test a tuple-typed identifier is rejected in expressions."""
    identifier = Identifier("pair")
    tuple_type = TupleType(
        [
            _make_scalar(CoreDataType.INT32),
            _make_scalar(CoreDataType.INT32),
        ]
    )
    checker = make_identifier_checker({identifier: (tuple_type, TypeQualifier.PARAM)})

    with pytest.raises(NotImplementedError, match=r"resolves to tuple type"):
        checker.visit(IdentifierExpression(identifier))


# =============================================================================
# TypeQualifier gating
# =============================================================================


def test_output_qualifier_identifier_is_rejected_on_read() -> None:
    """Test reading an identifier with an ``output`` qualifier is rejected."""
    identifier = Identifier("out")
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
    identifier = Identifier("x")
    checker = make_identifier_checker(
        {identifier: (_make_scalar(CoreDataType.INT32), qualifier)}
    )

    result_type, result_qualifier = checker.visit(IdentifierExpression(identifier))

    assert result_type.is_structurally_equivalent(_make_scalar(CoreDataType.INT32))
    assert result_qualifier is qualifier


@pytest.mark.parametrize("qualifier", _NON_OUTPUT_TYPE_QUALIFIERS)
def test_non_output_qualifier_operand_is_accepted_in_unary_expression(
    qualifier: TypeQualifier,
) -> None:
    """Test a non-``output`` operand is accepted inside a unary expression."""
    identifier = Identifier("x")
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
    identifier = Identifier("idx")
    upper = Identifier("N")
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
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
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_minus_scalar_produces_shifted_index_type() -> None:
    """Test subtracting a scalar from an index shifts both bounds downward."""
    identifier = Identifier("idx")
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


def test_index_plus_float_is_rejected() -> None:
    """Test adding a float scalar to an index is rejected."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    )

    assert result_type.is_structurally_equivalent(index)


def test_unary_negation_of_index_is_rejected() -> None:
    """Test unary negation of an index type is rejected."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
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
        LiteralExpression(3) * LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_positive_integer_literal_times_index_scales_symmetrically() -> None:
    """Test scalar-times-index scales the same way as index-times-scalar."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    """Test scaling an index with stride ``None`` by ``1`` keeps stride ``None``."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_times_zero_literal_is_rejected() -> None:
    """Test scaling an index by `0` is rejected (positive-integer requirement)."""
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    scalar = Identifier("k")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    index_identifier = Identifier("idx")
    scalar_identifier = Identifier("k")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
# Index + index
# =============================================================================


def test_index_plus_index_combines_bounds_when_strides_are_none() -> None:
    """Test index addition sums bounds and keeps stride `None` when both are `None`."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    right_index = IndexType(LiteralExpression(1), LiteralExpression(5), None)
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(5),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)
    assert result_qualifier is TypeQualifier.PARAM


def test_index_plus_index_preserves_matching_stride() -> None:
    """Test index addition with matching integer strides keeps that stride."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(2)
    )
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(6), LiteralExpression(2)
    )
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_treats_literal_one_stride_as_none() -> None:
    """Test `LiteralExpression(1)` stride combined with `None` yields `None`."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(1)
    )
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), None)
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_canonicalizes_matching_unit_literal_stride() -> None:
    """Test two `LiteralExpression(1)` strides canonicalize to `None`."""
    left = Identifier("i")
    right = Identifier("j")
    unit = LiteralExpression(1)
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), unit)
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), unit)
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_unit_stride_dominates_non_unit_stride() -> None:
    """Test combining stride `None` with stride `2` yields stride `None`."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(6), LiteralExpression(2)
    )
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(6),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_uses_min_of_literal_strides() -> None:
    """Test mismatched literal strides combine via `min`: `min(2, 3) == 2`."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(2)
    )
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(9), LiteralExpression(3)
    )
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(9),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_preserves_zero_stride_via_min() -> None:
    """Test combining a `0` stride with a `5` stride yields a `0` stride via `min`."""
    left = Identifier("i")
    right = Identifier("j")
    left_index = IndexType(
        LiteralExpression(1), LiteralExpression(10), LiteralExpression(0)
    )
    right_index = IndexType(
        LiteralExpression(1), LiteralExpression(9), LiteralExpression(5)
    )
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        LiteralExpression(10) + LiteralExpression(9),
        LiteralExpression(0),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_index_plus_index_rejects_non_literal_stride() -> None:
    """Test combining indices where at least one stride is non-literal is rejected."""
    left = Identifier("i")
    right = Identifier("j")
    stride_identifier = Identifier("s")
    left_index = IndexType(
        LiteralExpression(1),
        LiteralExpression(10),
        IdentifierExpression(stride_identifier),
    )
    right_index = IndexType(LiteralExpression(1), LiteralExpression(6), None)
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(
        FhYCoreTypeError,
        match=r"combining two index types requires both strides to be integer literals",
    ):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


def test_index_plus_index_with_symbolic_bounds() -> None:
    """Test two indices with symbolic upper bounds combine via bound addition."""
    left = Identifier("i")
    right = Identifier("j")
    upper_n = Identifier("N")
    upper_m = Identifier("M")
    left_index = IndexType(LiteralExpression(1), IdentifierExpression(upper_n), None)
    right_index = IndexType(LiteralExpression(1), IdentifierExpression(upper_m), None)
    checker = make_identifier_checker(
        {
            left: (left_index, TypeQualifier.PARAM),
            right: (right_index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(left),
            IdentifierExpression(right),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + LiteralExpression(1),
        IdentifierExpression(upper_n) + IdentifierExpression(upper_m),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_nested_index_plus_index_synthesizes_bottom_up_on_left() -> None:
    """Test `(a + b) + c` synthesizes the left subtree first, chaining strides."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    idx_c = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(6))
    checker = make_identifier_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
            c: (idx_c, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(a),
                IdentifierExpression(b),
            ),
            IdentifierExpression(c),
        )
    )

    expected = IndexType(
        (LiteralExpression(1) + LiteralExpression(1)) + LiteralExpression(1),
        (LiteralExpression(10) + LiteralExpression(10)) + LiteralExpression(10),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_nested_index_plus_index_synthesizes_bottom_up_on_right() -> None:
    """Test `a + (b + c)` synthesizes the right subtree first, chaining strides."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(6))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_c = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    checker = make_identifier_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
            c: (idx_c, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(a),
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(b),
                IdentifierExpression(c),
            ),
        )
    )

    expected = IndexType(
        LiteralExpression(1) + (LiteralExpression(1) + LiteralExpression(1)),
        LiteralExpression(10) + (LiteralExpression(10) + LiteralExpression(10)),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


def test_scaled_shifted_index_plus_index_synthesizes_bottom_up() -> None:
    """Test ``2 * (i1 - 1) + i2`` synthesizes scale, shift, and combine in order."""
    i1 = Identifier("i1")
    i2 = Identifier("i2")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = make_identifier_checker(
        {
            i1: (index, TypeQualifier.PARAM),
            i2: (index, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(2),
                BinaryExpression(
                    BinaryOperation.SUBTRACT,
                    IdentifierExpression(i1),
                    LiteralExpression(1),
                ),
            ),
            IdentifierExpression(i2),
        )
    )

    expected = IndexType(
        (LiteralExpression(2) * (LiteralExpression(1) - LiteralExpression(1)))
        + LiteralExpression(1),
        (LiteralExpression(2) * (LiteralExpression(10) - LiteralExpression(1)))
        + LiteralExpression(10),
        None,
    )
    assert result_type.is_structurally_equivalent(expected)


def test_shifted_index_plus_index_propagates_stride_bottom_up() -> None:
    """Test ``(a + 1) + b`` keeps ``a``'s stride through the scalar shift."""
    a = Identifier("a")
    b = Identifier("b")
    idx_a = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(2))
    idx_b = IndexType(LiteralExpression(1), LiteralExpression(10), LiteralExpression(3))
    checker = make_identifier_checker(
        {
            a: (idx_a, TypeQualifier.PARAM),
            b: (idx_b, TypeQualifier.PARAM),
        }
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(a),
                LiteralExpression(1),
            ),
            IdentifierExpression(b),
        )
    )

    expected = IndexType(
        (LiteralExpression(1) + LiteralExpression(1)) + LiteralExpression(1),
        (LiteralExpression(10) + LiteralExpression(1)) + LiteralExpression(10),
        LiteralExpression(2),
    )
    assert result_type.is_structurally_equivalent(expected)


# =============================================================================
# Index - index and other non-additive index arithmetic
# =============================================================================


def test_index_minus_index_uses_stride_semantics_error_message() -> None:
    """Test subtracting two indices raises with the stride-semantics reason."""
    left = Identifier("i")
    right = Identifier("j")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = make_identifier_checker(
        {
            left: (index, TypeQualifier.PARAM),
            right: (index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError, match="stride semantics"):
        checker.visit(
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )


@pytest.mark.parametrize(
    "operation",
    [
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    ],
)
def test_non_additive_index_index_operation_does_not_use_stride_semantics_message(
    operation: BinaryOperation,
) -> None:
    """Test non-additive index-index operations use a generic (non-stride) message."""
    left = Identifier("i")
    right = Identifier("j")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
    checker = make_identifier_checker(
        {
            left: (index, TypeQualifier.PARAM),
            right: (index, TypeQualifier.PARAM),
        }
    )

    with pytest.raises(FhYCoreTypeError) as exc_info:
        checker.visit(
            BinaryExpression(
                operation,
                IdentifierExpression(left),
                IdentifierExpression(right),
            )
        )
    assert "stride semantics" not in str(exc_info.value)


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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(8), None)
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
    identifier = Identifier("idx")
    index = IndexType(LiteralExpression(1), LiteralExpression(10), None)
    checker = make_identifier_checker({identifier: (index, TypeQualifier.PARAM)})
    expression = IdentifierExpression(identifier)
    expected = IndexType(LiteralExpression(2), LiteralExpression(10), None)

    with pytest.raises(FhYCoreTypeError) as exc_info:
        checker.check(expression, expected)

    message = str(exc_info.value)
    assert message.startswith("Type error while inferring type of `")
    assert pformat_expression(expression, show_id=True) in message
