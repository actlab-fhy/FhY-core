"""Expression tree passes and manipulations."""

import sympy  # type: ignore

from fhy_core.identifier import Identifier

from .core import (
    Expression,
    IdentifierExpression,
    LiteralType,
)
from .parser import parse_expression
from .pprint import pformat_expression, pformat_identifier
from .visitor import ExpressionTransformer, ExpressionVisitor


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


def simplify_expression(
    expression: Expression, environment: dict[Identifier, LiteralType] | None = None
) -> Expression:
    """Simplify an expression.

    Args:
        expression: Expression to simplify.
        environment: Environment to simplify the expression in. Defaults to None.

    Returns:
        Result of the expression.

    """
    sympy_expression = sympy.parsing.sympy_parser.parse_expr(
        pformat_expression(expression, show_id=True, functional=False)
    )
    if environment is not None:
        sympy_expression = sympy_expression.subs(
            {pformat_identifier(k, show_id=True): v for k, v in environment.items()}
        )
    result = sympy.simplify(sympy_expression)
    return parse_expression(str(result))
