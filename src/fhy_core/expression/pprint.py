"""Pretty-printer for expressions."""

__all__ = ["pformat_expression"]

from .core import (
    BINARY_OPERATION_FUNCTION_NAMES,
    BINARY_OPERATION_SYMBOLS,
    UNARY_OPERATION_FUNCTION_NAMES,
    UNARY_OPERATION_SYMBOLS,
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
)
from .visitor import ExpressionBasePass


class ExpressionPrettyFormatter(ExpressionBasePass):
    """Pretty-formatter for expressions."""

    _is_id_shown: bool
    _is_printed_functional: bool

    def __init__(
        self, is_id_shown: bool = False, is_printed_functional: bool = False
    ) -> None:
        super().__init__()
        self._is_id_shown = is_id_shown
        self._is_printed_functional = is_printed_functional

    def __call__(self, expression: Expression) -> str:
        formatted_expression = super().__call__(expression)
        if not isinstance(formatted_expression, str):
            raise TypeError(
                f"Invalid formatted expression type: {type(formatted_expression)}"
            )
        return formatted_expression

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> str:
        if self._is_printed_functional:
            return (
                f"({UNARY_OPERATION_FUNCTION_NAMES[unary_expression.operation]} "
                f"{self.visit(unary_expression.operand)})"
            )
        else:
            return (
                f"({UNARY_OPERATION_SYMBOLS[unary_expression.operation]}"
                f"{self.visit(unary_expression.operand)})"
            )

    def visit_binary_expression(self, binary_expression: BinaryExpression) -> str:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        if self._is_printed_functional:
            return (
                f"({BINARY_OPERATION_FUNCTION_NAMES[binary_expression.operation]} "
                f"{left} {right})"
            )
        else:
            return (
                f"({left} "
                f"{BINARY_OPERATION_SYMBOLS[binary_expression.operation]} "
                f"{right})"
            )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> str:
        identifier = identifier_expression.identifier
        if not self._is_id_shown:
            return identifier.name_hint
        else:
            return repr(identifier)

    def visit_literal_expression(self, literal_expression: LiteralExpression) -> str:
        return str(literal_expression.value)


def pformat_expression(
    expression: Expression, show_id: bool = False, functional: bool = False
) -> str:
    """Pretty-format an expression.

    Note:
        There is no guarantee that the pretty-formatted expression can be parsed back.

    Args:
        expression: Expression to pretty-format.
        show_id: Whether to show the identifier ID.
        functional: Whether to use functional notation.

    Returns:
        Pretty-formatted expression.

    """
    return ExpressionPrettyFormatter(
        is_id_shown=show_id, is_printed_functional=functional
    )(expression)
