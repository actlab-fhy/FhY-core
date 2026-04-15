"""Tests for expression type checking and literal promotion."""

import pytest
from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.expression.passes.type_checker import (
    ExpressionTypeChecker,
    get_core_data_type_from_literal_type,
)
from fhy_core.identifier import Identifier
from fhy_core.types import (
    CoreDataType,
    FhYCoreTypeError,
    IndexType,
    NumericalType,
    PrimitiveDataType,
    TypeQualifier,
)


def _assert_core_data_type(type_, expected_core_data_type: CoreDataType) -> None:
    assert isinstance(type_, NumericalType)
    assert isinstance(type_.data_type, PrimitiveDataType)
    assert type_.data_type.core_data_type == expected_core_data_type


def _assert_index_type(type_: object, expected_type: IndexType) -> None:
    assert isinstance(type_, IndexType)
    assert type_.is_structurally_equivalent(expected_type)


def test_get_core_data_type_from_literal_type_returns_weak_types():
    """Test get_core_data_type_from_literal_type function with various literals."""
    assert get_core_data_type_from_literal_type(1) == CoreDataType.UINT
    assert get_core_data_type_from_literal_type(-1) == CoreDataType.INT
    assert get_core_data_type_from_literal_type(1.5) == CoreDataType.FLOAT


def test_unary_negation_of_positive_integer_literal_becomes_weak_int():
    """Test unary negation of positive integer literal."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.visit(
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5))
    )

    _assert_core_data_type(result_type, CoreDataType.INT)
    assert result_qualifier == TypeQualifier.PARAM


def test_positive_integer_literal_upgrades_to_signed_context():
    """Test positive integer literal upgrades to signed context."""
    identifier = Identifier("x")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.INT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_integer_literals_remain_weak_without_context():
    """Test that integer literals remain weak without context."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.UINT)
    assert result_qualifier == TypeQualifier.PARAM


def test_integer_literal_can_upgrade_to_float_context():
    """Test that integer literals can upgrade to float context."""
    identifier = Identifier("y")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(PrimitiveDataType(CoreDataType.FLOAT32)),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    _assert_core_data_type(result_type, CoreDataType.FLOAT32)
    assert result_qualifier == TypeQualifier.PARAM


def test_check_literal_against_expected_type_resolves_weak_literal():
    """Test that checking a literal against an expected type resolves weak literal."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.check(
        LiteralExpression(256),
        NumericalType(PrimitiveDataType(CoreDataType.UINT16)),
    )

    _assert_core_data_type(result_type, CoreDataType.UINT16)
    assert result_qualifier == TypeQualifier.PARAM


def test_check_binary_expression_uses_expected_type_bidirectionally():
    """Test checking an arithmetic expression uses expected type bidirectionally."""
    checker = ExpressionTypeChecker(
        lambda _: (
            NumericalType(PrimitiveDataType(CoreDataType.INT32)),
            TypeQualifier.PARAM,
        )
    )

    result_type, result_qualifier = checker.check(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(255),
            LiteralExpression(256),
        ),
        NumericalType(PrimitiveDataType(CoreDataType.UINT16)),
    )

    _assert_core_data_type(result_type, CoreDataType.UINT16)
    assert result_qualifier == TypeQualifier.PARAM


def test_tensor_type_is_rejected_in_expression_typing():
    """Test that tensor-valued expressions are rejected in expression typing."""
    identifier = Identifier("tensor")
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (
            NumericalType(
                PrimitiveDataType(CoreDataType.FLOAT32),
                [LiteralExpression(4)],
            ),
            TypeQualifier.PARAM,
        )
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(identifier),
                LiteralExpression(1),
            )
        )


def test_index_plus_scalar_produces_shifted_index_type():
    """Test that adding a scalar to an index produces shifted index type."""
    identifier = Identifier("idx")
    upper_bound_identifier = Identifier("N")
    index_type = IndexType(
        LiteralExpression(0),
        IdentifierExpression(upper_bound_identifier),
        LiteralExpression(2),
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, result_qualifier = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(identifier),
            LiteralExpression(3),
        )
    )

    expected_type = IndexType(
        LiteralExpression(0) + LiteralExpression(3),
        IdentifierExpression(upper_bound_identifier) + LiteralExpression(3),
        LiteralExpression(2),
    )
    _assert_index_type(result_type, expected_type)
    assert result_qualifier == TypeQualifier.PARAM


def test_scalar_plus_index_produces_shifted_index_type():
    """Test that adding an index to a scalar produces shifted index type."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(1),
        LiteralExpression(10),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(2),
            IdentifierExpression(identifier),
        )
    )

    expected_type = IndexType(
        LiteralExpression(1) + LiteralExpression(2),
        LiteralExpression(10) + LiteralExpression(2),
        None,
    )
    _assert_index_type(result_type, expected_type)


def test_index_minus_scalar_produces_shifted_index_type():
    """Test that subtracting a scalar from an index produces shifted index type."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(2),
        LiteralExpression(12),
        LiteralExpression(3),
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        BinaryExpression(
            BinaryOperation.SUBTRACT,
            IdentifierExpression(identifier),
            LiteralExpression(1),
        )
    )

    expected_type = IndexType(
        LiteralExpression(2) - LiteralExpression(1),
        LiteralExpression(12) - LiteralExpression(1),
        LiteralExpression(3),
    )
    _assert_index_type(result_type, expected_type)


def test_unary_positive_index_preserves_index_type():
    """Test that unary positive preserves index types."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(0),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    result_type, _ = checker.visit(
        UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(identifier))
    )

    _assert_index_type(result_type, index_type)


def test_unary_negation_of_index_is_rejected():
    """Test that unary negation of index types is rejected."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(0),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(identifier))
        )


def test_index_multiplication_is_rejected():
    """Test that index multiplication is rejected."""
    identifier = Identifier("idx")
    index_type = IndexType(
        LiteralExpression(0),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier == identifier
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(identifier),
                LiteralExpression(2),
            )
        )


def test_index_plus_index_is_rejected():
    """Test that index plus index is rejected."""
    left_identifier = Identifier("left_idx")
    right_identifier = Identifier("right_idx")
    index_type = IndexType(
        LiteralExpression(0),
        LiteralExpression(8),
        None,
    )
    checker = ExpressionTypeChecker(
        lambda seen_identifier: (index_type, TypeQualifier.PARAM)
        if seen_identifier in {left_identifier, right_identifier}
        else (_ for _ in ()).throw(AssertionError("Unexpected identifier lookup"))
    )

    with pytest.raises(FhYCoreTypeError):
        checker.visit(
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(left_identifier),
                IdentifierExpression(right_identifier),
            )
        )
