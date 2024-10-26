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


class _ExpressionIdentifierRepairer(ExpressionTransformer):
    """Repairs the identifiers in an expression tree after manipulation in sympy.

    Expressions are all parsed using the same parser which assumes that all symbols
    have not been assigned unique IDs yet. As such, the symbols are assigned new
    unique IDs after parsing. This transformer repairs the identifiers in an expression
    to match the old IDs using the appended ID to the symbol name in the expression.

    """

    def visit_identifier_expression(
        self, identifier_expression
    ) -> IdentifierExpression:
        identifier = identifier_expression.identifier
        last_underscore_index = identifier_expression.identifier.name_hint.rfind("_")
        if last_underscore_index == -1:
            raise RuntimeError(
                "After parsing an expression emitted from sympy, "
                "an identifier without an underscore was found."
            )
        identifier_id = int(identifier.name_hint[last_underscore_index + 1 :])
        identifier._name_hint = identifier._name_hint[:last_underscore_index]
        identifier._id = identifier_id
        return IdentifierExpression(identifier)


def _repair_sympy_expression_identifiers(expression: sympy.Expr) -> sympy.Expr:
    """Repairs the identifiers in a sympy expression after manipulation in sympy."""
    repairer = _ExpressionIdentifierRepairer()
    return repairer(expression)


def convert_expression_to_sympy_expression(expression: Expression) -> sympy.Expr:
    """Convert an expression to a SymPy expression.

    Args:
        expression: Expression to convert.

    Returns:
        SymPy expression.

    """
    return sympy.parse_expr(
        pformat_expression(expression, show_id=True, functional=False),
        evaluate=False,
    )


def substitute_sympy_expression_variables(
    sympy_expression: sympy.Expr, environment: dict[Identifier, LiteralType]
) -> sympy.Expr:
    """Substitute variables in a SymPy expression.

    Args:
        sympy_expression: SymPy expression to substitute variables in.
        environment: Environment to substitute variables from.

    Returns:
        SymPy expression with substituted variables.

    """
    return sympy_expression.subs(
        {pformat_identifier(k, show_id=True): v for k, v in environment.items()}
    )


def convert_sympy_expression_to_expression(
    sympy_expression: sympy.Expr,
) -> Expression:
    """Convert a SymPy expression to an expression.

    Args:
        sympy_expression: SymPy expression to convert.

    Returns:
        Expression.

    """
    expression = parse_expression(str(sympy_expression))
    return _repair_sympy_expression_identifiers(expression)


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
    sympy_expression = convert_expression_to_sympy_expression(expression)
    if environment is not None:
        sympy_expression = substitute_sympy_expression_variables(
            sympy_expression, environment
        )
    result = sympy.simplify(sympy_expression)
    return parse_expression(str(result))
