"""Pretty-printer for expressions."""

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


class ExpressionPrettyFormatter:
    """Pretty-formatter for expressions."""

    _is_printed_functional: bool

    def __init__(self, is_printed_functional: bool = False) -> None:
        self._is_printed_functional = is_printed_functional

    def __call__(self, expression: Expression) -> str:
        return self.pformat(expression)

    def pformat(self, expression: Expression) -> str:
        """Pretty-format an expression.

        Args:
            expression: Expression to visit.

        Returns:
            Pretty-formatted expression.

        Raises:
            NotImplementedError: If the expression type is not supported.

        """
        if isinstance(expression, UnaryExpression):
            return self.pformat_unary_expression(expression)
        elif isinstance(expression, BinaryExpression):
            return self.pformat_binary_expression(expression)
        elif isinstance(expression, IdentifierExpression):
            return self.pformat_identifier_expression(expression)
        elif isinstance(expression, LiteralExpression):
            return self.pformat_literal_expression(expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: {type(expression)}"
            )

    def pformat_unary_expression(self, unary_expression: UnaryExpression) -> str:
        """Pretty-format a unary expression.

        Args:
            unary_expression: Unary expression to visit.

        Returns:
            Pretty-formatted unary expression.

        """
        if self._is_printed_functional:
            return (
                f"({UNARY_OPERATION_FUNCTION_NAMES[unary_expression.operation]} "
                f"{self.pformat(unary_expression.operand)})"
            )
        else:
            return (
                f"({UNARY_OPERATION_SYMBOLS[unary_expression.operation]}"
                f"{self.pformat(unary_expression.operand)})"
            )

    def pformat_binary_expression(self, binary_expression: BinaryExpression) -> str:
        """Pretty-format a binary expression.

        Args:
            binary_expression: Binary expression to visit.

        Returns:
            Pretty-formatted binary expression.

        """
        left = self.pformat(binary_expression.left)
        right = self.pformat(binary_expression.right)
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

    def pformat_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> str:
        """Pretty-format an identifier.

        Args:
            identifier_expression: Identifier expression to visit.

        Returns:
            Pretty-formatted identifier expression.

        """
        return str(identifier_expression.identifier)

    def pformat_literal_expression(self, literal_expression: LiteralExpression) -> str:
        """Pretty-format a literal.

        Args:
            literal_expression: Literal expression to visit.

        Returns:
            Pretty-formatted literal expression.

        """
        return str(literal_expression.value)


def pformat_expression(expression: Expression, functional: bool = False) -> str:
    """Pretty-format an expression.

    Args:
        expression: Expression to pretty-format.
        functional: Whether to use functional notation.

    Returns:
        Pretty-formatted expression.

    """
    return ExpressionPrettyFormatter(is_printed_functional=functional)(expression)
