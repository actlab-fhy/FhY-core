"""Expression tree transformer."""

from abc import ABC

from .core import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
)


class ExpressionTransformer(ABC):
    """Transformer for expression trees."""

    def __call__(self, expression: Expression) -> Expression:
        return self.visit(expression)

    def visit(self, expression: Expression) -> Expression:
        """Visit an expression.

        Args:
            expression: Expression to visit.

        Returns:
            New expression.

        Raises:
            NotImplementedError: If the expression type is not supported.

        """
        if isinstance(expression, UnaryExpression):
            return self.visit_unary_expression(expression)
        elif isinstance(expression, BinaryExpression):
            return self.visit_binary_expression(expression)
        elif isinstance(expression, IdentifierExpression):
            return self.visit_identifier_expression(expression)
        elif isinstance(expression, LiteralExpression):
            return self.visit_literal_expression(expression)
        else:
            raise NotImplementedError(
                f"Unsupported expression type: {type(expression)}"
            )

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Expression:
        """Visit a unary expression.

        Args:
            unary_expression: Unary expression to visit.

        Returns:
            New unary expression.

        """
        new_expression = self.visit(unary_expression.operand)
        return UnaryExpression(
            operation=unary_expression.operation, operand=new_expression
        )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> Expression:
        """Visit a binary expression.

        Args:
            binary_expression: Binary expression to visit.

        Returns:
            New binary expression.

        """
        new_left = self.visit(binary_expression.left)
        new_right = self.visit(binary_expression.right)
        return BinaryExpression(
            operation=binary_expression.operation, left=new_left, right=new_right
        )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        """Visit an identifier.

        Args:
            identifier_expression: Identifier expression to visit.

        Returns:
            New identifier expression.

        """
        return IdentifierExpression(identifier=identifier_expression.identifier)

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> Expression:
        """Visit a literal.

        Args:
            literal_expression: Literal expression to visit.

        Returns:
            New literal expression.

        """
        return LiteralExpression(value=literal_expression.value)
