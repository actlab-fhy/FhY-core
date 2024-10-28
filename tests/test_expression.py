"""Tests the expression utility."""

import pytest
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier


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
