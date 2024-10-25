"""Tests the expression utility."""

from unittest.mock import MagicMock

import pytest
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    copy_expression,
    evaluate_expression,
    pformat_expression,
)
from fhy_core.expression.core import LiteralType
from fhy_core.expression.transformer import ExpressionTransformer
from fhy_core.identifier import Identifier


def test_unary_operations():
    """Tests that unary operations are correctly defined."""
    assert UnaryOperation.NEGATE == "negate"
    assert UnaryOperation.BITWISE_NOT == "bitwise_not"
    assert UnaryOperation.LOGICAL_NOT == "logical_not"


def test_binary_operations():
    """Tests that binary operations are correctly defined."""
    assert BinaryOperation.ADD == "add"
    assert BinaryOperation.SUBTRACT == "subtract"
    assert BinaryOperation.MULTIPLY == "multiply"
    assert BinaryOperation.DIVIDE == "divide"
    assert BinaryOperation.MODULO == "modulo"
    assert BinaryOperation.POWER == "power"
    assert BinaryOperation.BITWISE_AND == "bitwise_and"
    assert BinaryOperation.BITWISE_OR == "bitwise_or"
    assert BinaryOperation.BITWISE_XOR == "bitwise_xor"
    assert BinaryOperation.LEFT_SHIFT == "left_shift"
    assert BinaryOperation.RIGHT_SHIFT == "right_shift"
    assert BinaryOperation.LOGICAL_AND == "logical_and"
    assert BinaryOperation.LOGICAL_OR == "logical_or"
    assert BinaryOperation.EQUAL == "equal"
    assert BinaryOperation.NOT_EQUAL == "not_equal"
    assert BinaryOperation.LESS == "less"
    assert BinaryOperation.LESS_EQUAL == "less_equal"
    assert BinaryOperation.GREATER == "greater"
    assert BinaryOperation.GREATER_EQUAL == "greater_equal"


def test_unary_expression():
    """Tests that the unary expression is correctly initialized."""
    operand = LiteralExpression(5)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    assert expr.operation == UnaryOperation.NEGATE
    assert expr.operand is operand


def test_binary_expression():
    """Tests that the binary expression is correctly initialized."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    assert expr.operation == BinaryOperation.ADD
    assert expr.left is left
    assert expr.right is right


def test_identifier_expression():
    """Tests that the identifier expression is correctly initialized."""
    identifier = Identifier("test_identifier")
    expr = IdentifierExpression(identifier=identifier)
    assert expr.identifier == identifier


@pytest.mark.parametrize("value", [5, 3.14, True, complex(2, 3), "2+3j"])
def test_literal_expression_valid_values(value):
    """Tests that the literal expression is correctly initialized with valid values."""
    expr = LiteralExpression(value)
    assert expr._value == value if not isinstance(value, str) else complex(value)


def test_literal_expression_invalid_string():
    """Tests that the literal expression raises an exception for invalid string
    values.
    """
    with pytest.raises(ValueError, match="Invalid literal expression value:"):
        LiteralExpression("invalid_literal")


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (
            IdentifierExpression(identifier=Identifier("test_identifier")),
            "test_identifier",
        ),
        (
            UnaryExpression(
                operation=UnaryOperation.NEGATE, operand=LiteralExpression(5)
            ),
            "(negate 5)",
        ),
        (
            BinaryExpression(
                operation=BinaryOperation.ADD,
                left=LiteralExpression(5),
                right=LiteralExpression(10),
            ),
            "(add 5 10)",
        ),
        (
            BinaryExpression(
                operation=BinaryOperation.DIVIDE,
                left=UnaryExpression(
                    operation=UnaryOperation.NEGATE, operand=LiteralExpression(5)
                ),
                right=LiteralExpression(10),
            ),
            "(divide (negate 5) 10)",
        ),
    ],
)
def test_pformat_expression(expression: Expression, expected_str: str):
    """Tests that the expression is correctly pretty-formatted."""
    assert pformat_expression(expression) == expected_str


# TODO: Revisit the use of MagicMock here and in the following tests.
@pytest.fixture
def transformer():
    class ConcreteExpressionTransformer(ExpressionTransformer):
        """Concrete expression transformer for testing."""

    transformer = ConcreteExpressionTransformer()
    transformer.visit_unary_expression = MagicMock()
    transformer.visit_binary_expression = MagicMock()
    transformer.visit_identifier_expression = MagicMock()
    transformer.visit_literal_expression = MagicMock()
    return transformer


def test_transformer_calls_unary_expression_visitor(transformer: ExpressionTransformer):
    """Tests that the visit method calls the correct visit method for
    UnaryExpression.
    """
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=MagicMock())

    transformer.visit(expr)

    transformer.visit_unary_expression.assert_called_once_with(expr)


def test_transformer_calls_binary_expression_visitor(
    transformer: ExpressionTransformer,
):
    """Tests that the visit method calls the correct visit method for
    BinaryExpression.
    """
    expr = BinaryExpression(
        operation=BinaryOperation.ADD, left=MagicMock(), right=MagicMock()
    )

    transformer.visit(expr)

    transformer.visit_binary_expression.assert_called_once_with(expr)


def test_transformer_calls_identifier_expression_visitor(
    transformer: ExpressionTransformer,
):
    """Tests that the visit method calls the correct visit method for
    IdentifierExpression.
    """
    expr = IdentifierExpression(identifier=Identifier("x"))

    transformer.visit(expr)

    transformer.visit_identifier_expression.assert_called_once_with(expr)


def test_transformer_calls_literal_visitor(transformer: ExpressionTransformer):
    """Tests that the visit method calls the correct visit method for
    LiteralExpression.
    """
    expr = LiteralExpression(value=42)

    transformer.visit(expr)

    transformer.visit_literal_expression.assert_called_once_with(expr)


def test_visit_unsupported_expression(transformer: ExpressionTransformer):
    """Tests that NotImplementedError is raised for unsupported expression types."""
    expr = MagicMock(spec=Expression)

    with pytest.raises(NotImplementedError):
        transformer.visit(expr)


def test_copy_literal_expression():
    """Tests that the literal expression is correctly copied."""
    expr = LiteralExpression(value=42)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.value == expr.value


def test_copy_identifier_expression():
    """Tests that the identifier expression is correctly copied."""
    expr = IdentifierExpression(identifier=Identifier("x"))
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.identifier == expr.identifier


def test_copy_unary_expression():
    """Tests that the unary expression is correctly copied."""
    operand = LiteralExpression(value=42)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.operation == expr.operation
    assert copy.operand is not expr.operand
    assert copy.operand.value == expr.operand.value


def test_copy_binary_expression():
    """Tests that the binary expression is correctly copied."""
    left = LiteralExpression(value=42)
    right = LiteralExpression(value=24)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.operation == expr.operation
    assert copy.left is not expr.left
    assert copy.left.value == expr.left.value
    assert copy.right is not expr.right
    assert copy.right.value == expr.right.value


@pytest.mark.parametrize(
    "expression, expected_value",
    [
        (LiteralExpression(5), 5),
        (UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)), -5),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            15,
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            -0.5,
        ),
    ],
)
def test_evaluate_constant_expression(
    expression: Expression, expected_value: LiteralType
):
    """Tests that the expression is correctly evaluated."""
    result = evaluate_expression(expression)
    assert isinstance(result, LiteralExpression)
    assert result.value == expected_value


def test_evaluate_expression_with_environment():
    """Tests that the expression is correctly evaluated with an environment."""
    x = Identifier("x")
    environment = {x: 5}
    expr = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(10)
    )
    result = evaluate_expression(expr, environment)
    assert isinstance(result, LiteralExpression)
    assert result.value == 15
