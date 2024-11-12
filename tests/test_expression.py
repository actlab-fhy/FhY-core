"""Tests the expression utility."""

import operator

import pytest
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier

from .utils import assert_exact_expression_equality, mock_identifier


def test_unary_expression():
    """Test that the unary expression is correctly initialized."""
    operand = LiteralExpression(5)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    assert expr.operation == UnaryOperation.NEGATE
    assert expr.operand is operand


def test_binary_expression():
    """Test that the binary expression is correctly initialized."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    assert expr.operation == BinaryOperation.ADD
    assert expr.left is left
    assert expr.right is right


def test_identifier_expression():
    """Test that the identifier expression is correctly initialized."""
    identifier = Identifier("test_identifier")
    expr = IdentifierExpression(identifier)
    assert expr.identifier == identifier


@pytest.mark.parametrize("value", [5, 3.14, True])
def test_literal_expression_valid_values(value):
    """Test that the literal expression is correctly initialized with valid values."""
    expr = LiteralExpression(value)
    assert expr._value == value if not isinstance(value, str) else complex(value)


def test_literal_expression_invalid_string():
    """Test that the literal expression raises an exception for invalid string
    values.
    """
    with pytest.raises(ValueError, match="Invalid literal expression value:"):
        LiteralExpression("invalid_literal")


@pytest.mark.parametrize(
    "unary_operator, expected_operation",
    [
        (operator.neg, UnaryOperation.NEGATE),
        (operator.pos, UnaryOperation.POSITIVE),
        (lambda x: x.logical_not(), UnaryOperation.LOGICAL_NOT),
    ],
)
def test_unary_operator_dunder_methods(
    unary_operator, expected_operation: UnaryOperation
):
    """Test that the unary operation dunder methods correctly create unary
    expressions.
    """
    operand = LiteralExpression(5)
    expected_expr = UnaryExpression(expected_operation, operand)
    assert_exact_expression_equality(unary_operator(operand), expected_expr)


_binary_operator_operations_pairs = pytest.mark.parametrize(
    "binary_operator, expected_operation",
    [
        (operator.add, BinaryOperation.ADD),
        (operator.sub, BinaryOperation.SUBTRACT),
        (operator.mul, BinaryOperation.MULTIPLY),
        (operator.truediv, BinaryOperation.DIVIDE),
        (operator.mod, BinaryOperation.MODULO),
        (operator.pow, BinaryOperation.POWER),
        (lambda x, y: x.equals(y), BinaryOperation.EQUAL),
        (lambda x, y: x.not_equals(y), BinaryOperation.NOT_EQUAL),
        (operator.lt, BinaryOperation.LESS),
        (operator.le, BinaryOperation.LESS_EQUAL),
        (operator.gt, BinaryOperation.GREATER),
        (operator.ge, BinaryOperation.GREATER_EQUAL),
    ],
)


@_binary_operator_operations_pairs
def test_binary_operation_dunder_methods(
    binary_operator, expected_operation: BinaryOperation
):
    """Test that the binary operation dunder methods correctly create binary"""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expected_expr = BinaryExpression(expected_operation, left, right)
    assert_exact_expression_equality(binary_operator(left, right), expected_expr)


@_binary_operator_operations_pairs
@pytest.mark.parametrize(
    "left, right, expected_right_type",
    [
        (LiteralExpression(5), 10, LiteralExpression),
        (IdentifierExpression(mock_identifier("x", 0)), 10.23, LiteralExpression),
        (
            UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(10)),
            False,
            LiteralExpression,
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            "2.264",
            LiteralExpression,
        ),
        (LiteralExpression(5), mock_identifier("x", 2), IdentifierExpression),
    ],
)
def test_binary_operation_left_dunder_methods_for_literals(
    binary_operator,
    expected_operation: BinaryOperation,
    left: Expression,
    right: Identifier | str | float | int | bool,
    expected_right_type: type[Expression],
):
    """Test that the binary operation left dunder methods correctly create
    literal expressions.
    """
    expected_expr = BinaryExpression(
        expected_operation, left, expected_right_type(right)
    )
    assert_exact_expression_equality(binary_operator(left, right), expected_expr)


@pytest.mark.parametrize(
    "binary_operator, expected_operation",
    [
        (operator.add, BinaryOperation.ADD),
        (operator.sub, BinaryOperation.SUBTRACT),
        (operator.mul, BinaryOperation.MULTIPLY),
        (operator.truediv, BinaryOperation.DIVIDE),
        (operator.mod, BinaryOperation.MODULO),
        (operator.pow, BinaryOperation.POWER),
    ],
)
@pytest.mark.parametrize(
    "left, expected_left_type, right",
    [
        (6, LiteralExpression, LiteralExpression(10)),
        (10.3, LiteralExpression, IdentifierExpression(mock_identifier("y", 19))),
        (
            True,
            LiteralExpression,
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(15)),
        ),
        (
            "2.4",
            LiteralExpression,
            BinaryExpression(
                BinaryOperation.SUBTRACT, LiteralExpression(2), LiteralExpression(3)
            ),
        ),
        (mock_identifier("x", 1), IdentifierExpression, LiteralExpression(5)),
    ],
)
def test_binary_operation_right_dunder_methods_for_literals(
    binary_operator,
    expected_operation: BinaryOperation,
    left: Identifier | str | float | int | bool,
    expected_left_type: type[Expression],
    right: Expression,
):
    """Test that the binary operation right dunder methods correctly create
    literal expressions.
    """
    if binary_operator == operator.mod and isinstance(left, str):
        pytest.skip(
            "Modulo operation with string on the left is reserved for formatting."
        )
    expected_expr = BinaryExpression(
        expected_operation, expected_left_type(left), right
    )
    assert_exact_expression_equality(binary_operator(left, right), expected_expr)


def test_binary_operation_dunder_method_fails_to_create_expression_with_unknown_type():
    """Test that the binary operation dunder methods fail to create an expression
    with an unknown type.
    """
    with pytest.raises(ValueError):
        operator.add(LiteralExpression(5), [])


def test_logical_and():
    """Test that the logical and static method creates an AND tree."""
    expression_1 = LiteralExpression(True)
    expression_2 = LiteralExpression(False)
    expression_3 = True

    result = Expression.logical_and(expression_1, expression_2, expression_3)

    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        expression_1,
        BinaryExpression(
            BinaryOperation.LOGICAL_AND, expression_2, LiteralExpression(expression_3)
        ),
    )
    assert_exact_expression_equality(result, expected_expression)


def test_logical_or():
    """Test that the logical or static method creates an OR tree."""
    expression_1 = LiteralExpression(True)
    expression_2 = False
    expression_3 = LiteralExpression(True)

    result = Expression.logical_or(expression_1, expression_2, expression_3)

    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_OR,
        expression_1,
        BinaryExpression(
            BinaryOperation.LOGICAL_OR, LiteralExpression(expression_2), expression_3
        ),
    )
    assert_exact_expression_equality(result, expected_expression)
