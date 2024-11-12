"""Basic expression passes."""

__all__ = [
    "collect_identifiers",
    "copy_expression",
    "substitute_identifiers",
]

from fhy_core.expression.core import (
    Expression,
    IdentifierExpression,
)
from fhy_core.expression.visitor import ExpressionTransformer, ExpressionVisitor
from fhy_core.identifier import Identifier


class IdentifierCollector(ExpressionVisitor):
    """Collect all identifiers in an expression tree."""

    _identifiers: set[Identifier]

    def __init__(self) -> None:
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


class IdentifierSubstituter(ExpressionTransformer):
    """Substitute identifiers in an expression tree."""

    _substitutions: dict[Identifier, Expression]

    def __init__(self, substitutions: dict[Identifier, Expression]) -> None:
        self._substitutions = substitutions

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        identifier = identifier_expression.identifier
        if identifier in self._substitutions:
            return self._substitutions[identifier]
        return identifier_expression


def substitute_identifiers(
    expression: Expression, substitutions: dict[Identifier, Expression]
) -> Expression:
    """Substitute identifiers in an expression tree.

    Args:
        expression: Expression to substitute identifiers in.
        substitutions: Substitutions to make.

    Returns:
        Expression with identifiers substituted.

    """
    return IdentifierSubstituter(substitutions)(expression)


def replace_identifiers(
    expression: Expression, replacements: dict[Identifier, Identifier]
) -> Expression:
    """Replace identifiers in an expression tree.

    Args:
        expression: Expression to replace identifiers in.
        replacements: Replacements to make.

    Returns:
        Expression with identifiers replaced.

    """
    substitutions: dict[Identifier, Expression] = {
        old_identifier: IdentifierExpression(new_identifier)
        for old_identifier, new_identifier in replacements.items()
    }
    return substitute_identifiers(expression, substitutions)
