"""Testing utilitiy functions."""

from unittest.mock import Mock

from fhy_core.expression import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    pformat_expression,
)
from fhy_core.identifier import Identifier


def mock_identifier(name_hint: str, identifier_id: int) -> Identifier:
    """Create a mock identifier.

    Args:
        name_hint: Variable name.
        identifier_id: Identifier ID.

    Returns:
        Mock identifier.

    """
    identifier = Mock(spec=Identifier)
    identifier._name_hint = name_hint
    identifier._id = identifier_id
    identifier.name_hint = name_hint
    identifier.id = identifier_id
    return identifier


def assert_exact_expression_equality(
    expression1: Expression, expression2: Expression
) -> None:
    """Assert that two expressions are exactly equal.

    Args:
        expression1: First expression.
        expression2: Second expression.

    """
    if isinstance(expression1, LiteralExpression) and isinstance(
        expression2, LiteralExpression
    ):
        assert expression1.value == expression2.value
    elif isinstance(expression1, IdentifierExpression) and isinstance(
        expression2, IdentifierExpression
    ):
        assert expression1.identifier == expression2.identifier
    elif isinstance(expression1, UnaryExpression) and isinstance(
        expression2, UnaryExpression
    ):
        assert expression1.operation == expression2.operation
        assert_exact_expression_equality(expression1.operand, expression2.operand)
    elif isinstance(expression1, BinaryExpression) and isinstance(
        expression2, BinaryExpression
    ):
        assert expression1.operation == expression2.operation
        assert_exact_expression_equality(expression1.left, expression2.left)
        assert_exact_expression_equality(expression1.right, expression2.right)
    else:
        assert False, (
            "Expression trees did not have the same structure: "
            f"{pformat_expression(expression1, show_id=True)} ,"
            f"{pformat_expression(expression2, show_id=True)}."
        )
