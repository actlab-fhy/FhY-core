"""Tests for `fhy_core.expression.pprint`."""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    pformat_expression,
)
from fhy_core.expression.pprint import ExpressionPrettyFormatter
from fhy_core.identifier import Identifier

# =============================================================================
# Symbolic format (default)
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(4.5), "4.5"),
        (IdentifierExpression(Identifier("baz")), "baz"),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            "(!True)",
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(5 + 6j),  # type: ignore[arg-type]  # test: complex literal
                LiteralExpression(10.5),
            ),
            "((5+6j) * 10.5)",
        ),
    ],
)
def test_pformat_expression_renders_symbolic_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression` emits the expected symbolic-notation string."""
    assert pformat_expression(expression) == expected_str


# =============================================================================
# Functional format
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (IdentifierExpression(Identifier("test_identifier")), "test_identifier"),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
            "(negate 5)",
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(5),
                LiteralExpression(10),
            ),
            "(add 5 10)",
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            "(divide (negate 5) 10)",
        ),
    ],
)
def test_pformat_expression_renders_functional_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression(..., functional=True)` emits functional-notation."""
    assert pformat_expression(expression, functional=True) == expected_str


# =============================================================================
# Identifier-id visibility
# =============================================================================


def test_pformat_expression_with_show_id_includes_name_hint_and_id() -> None:
    """Test `pformat_expression(..., show_id=True)` renders both name hint and id."""
    identifier = Identifier("foo")
    result = pformat_expression(IdentifierExpression(identifier), show_id=True)
    assert identifier.name_hint in result
    assert str(identifier.id) in result


# =============================================================================
# ExpressionPrettyFormatter defaults
# =============================================================================


def test_pretty_formatter_default_does_not_show_identifier_id() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render identifiers without an id."""
    identifier = Identifier("name_only")
    result = ExpressionPrettyFormatter()(IdentifierExpression(identifier))
    assert result == identifier.name_hint


def test_pretty_formatter_default_uses_symbolic_notation() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render binary ops symbolically."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    assert ExpressionPrettyFormatter()(expression) == "(1 + 2)"
