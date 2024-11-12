"""Tests the expression tree parser and lexer."""

from unittest.mock import patch

import pytest
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    parse_expression,
    tokenize_expression,
)

from .utils import assert_exact_expression_equality, mock_identifier


# TODO: More tests for tokenization!
@pytest.mark.parametrize(
    "expression_str, expected_tokens",
    [
        ("7", ["7"]),
        ("342.5", ["342.5"]),
        ("True", ["True"]),
        ("False", ["False"]),
        ("y", ["y"]),
        ("-5", ["-", "5"]),
        ("10.5 + _x_", ["10.5", "+", "_x_"]),
        (
            "((10+2) >= 2) > 5",
            ["(", "(", "10", "+", "2", ")", ">=", "2", ")", ">", "5"],
        ),
        (
            "x // 5 + 3",
            ["x", "//", "5", "+", "3"],
        ),
    ],
)
def test_tokenize_expression(expression_str: str, expected_tokens: list[str]):
    """Test that the expression is correctly tokenized."""
    assert tokenize_expression(expression_str) == expected_tokens


# TODO: More tests for parsing!
@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        ("5", LiteralExpression("5")),
        ("3.2", LiteralExpression("3.2")),
        ("True", LiteralExpression(True)),
        ("False", LiteralExpression(False)),
        ("-5", UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("5"))),
        (
            "34 / 5.7",
            BinaryExpression(
                BinaryOperation.DIVIDE,
                LiteralExpression("34"),
                LiteralExpression("5.7"),
            ),
        ),
        (
            "10 + -2 * 5",
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression("10"),
                BinaryExpression(
                    BinaryOperation.MULTIPLY,
                    UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("2")),
                    LiteralExpression("5"),
                ),
            ),
        ),
        (
            "(2 + (5+6)) * -0",
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                BinaryExpression(
                    BinaryOperation.ADD,
                    LiteralExpression("2"),
                    BinaryExpression(
                        BinaryOperation.ADD,
                        LiteralExpression("5"),
                        LiteralExpression("6"),
                    ),
                ),
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("0")),
            ),
        ),
        (
            "x + y",
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
        ),
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_expression(expression_str: str, expected_tree: Expression):
    """Test that the expression is correctly parsed."""
    result = parse_expression(expression_str)
    assert_exact_expression_equality(result, expected_tree)
