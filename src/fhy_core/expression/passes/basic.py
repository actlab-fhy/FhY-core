"""Basic expression passes."""

__all__ = [
    "collect_identifiers",
    "substitute_identifiers",
]

from fhy_core.expression.core import (
    BinaryExpression,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    AnalysisVisitablePass,
    VisitablePass,
    register_pass,
)


@register_pass(
    "fhy_core.expression.collect_identifiers",
    "Collect identifier references from an expression tree.",
)
class IdentifierCollector(AnalysisVisitablePass[Expression]):
    """Collect all identifiers in an expression tree."""

    _identifiers: set[Identifier]

    def __init__(self) -> None:
        super().__init__()
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


@register_pass(
    "fhy_core.expression.substitute_identifiers",
    "Rewrite identifier references using a substitution map.",
)
class IdentifierSubstituter(VisitablePass[Expression, Expression]):
    """Substitute identifiers in an expression tree."""

    _substitutions: dict[Identifier, Expression]

    def __init__(self, substitutions: dict[Identifier, Expression]) -> None:
        super().__init__()
        self._substitutions = substitutions

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        identifier = identifier_expression.identifier
        if identifier in self._substitutions:
            return self._substitutions[identifier]
        return identifier_expression

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Expression:
        new_expression = self.visit(unary_expression.operand)
        return UnaryExpression(unary_expression.operation, new_expression)

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> Expression:
        new_left = self.visit(binary_expression.left)
        new_right = self.visit(binary_expression.right)
        return BinaryExpression(binary_expression.operation, new_left, new_right)

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> Expression:
        return literal_expression

    def visit_ternary_expression(
        self, ternary_expression: TernaryExpression
    ) -> Expression:
        return TernaryExpression(
            self.visit(ternary_expression.condition),
            self.visit(ternary_expression.true_value),
            self.visit(ternary_expression.false_value),
        )

    def visit_call_expression(self, call_expression: CallExpression) -> Expression:
        new_arguments = tuple(
            self.visit(argument) for argument in call_expression.arguments
        )
        return CallExpression(call_expression.function_name, new_arguments)

    def get_noop_output(self, ir: Expression) -> Expression:
        return ir


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
