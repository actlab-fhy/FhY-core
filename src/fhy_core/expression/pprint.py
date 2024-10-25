"""Pretty-printer for expressions."""

from .core import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
)


class ExpressionPrettyFormatter:
    """Pretty-formatter for expressions."""

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
        return (
            f"({unary_expression.operation} {self.pformat(unary_expression.operand)})"
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
        return f"({binary_expression.operation} {left} {right})"

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


def pformat_expression(expression: Expression) -> str:
    """Pretty-format an expression.

    Args:
        expression: Expression to pretty-format.

    Returns:
        Pretty-formatted expression.

    """
    return ExpressionPrettyFormatter()(expression)
