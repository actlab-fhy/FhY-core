"""Basic expression passes."""

from fhy_core.expression.core import (
    Expression,
    IdentifierExpression,
)
from fhy_core.expression.visitor import ExpressionTransformer, ExpressionVisitor
from fhy_core.identifier import Identifier


class IdentifierCollector(ExpressionVisitor):
    """Collect all identifiers in an expression tree."""

    _identifiers: set[Identifier]

    def __init__(self):
        self._identifiers = set()

    @property
    def identifiers(self) -> set[Identifier]:
        return self._identifiers

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> None:
        self._identifiers.add(identifier_expression.identifier)


def collect_identifiers(expression: Expression) -> set[Identifier]:
    """Collect all identifiers in an expression tree.

    Args:
        expression: Expression to collect identifiers from.

    Returns:
        Set of identifiers in the expression.

    """
    collector = IdentifierCollector()
    collector(expression)
    return collector.identifiers


class ExpressionCopier(ExpressionTransformer):
    """Shallow copier for an expression tree."""


def copy_expression(expression: Expression) -> Expression:
    """Shallow-copy an expression.

    Args:
        expression: Expression to copy.

    Returns:
        Copied expression.

    """
    return ExpressionCopier()(expression)
