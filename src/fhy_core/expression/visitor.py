"""Expression tree visitor and transformer."""

__all__ = [
    "ExpressionBasePass",
    "ExpressionVisitor",
    "ExpressionTransformer",
]

from abc import ABC
from typing import Any

from .core import (
    BinaryExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
)


class ExpressionBasePass(ABC):
    """Base class for expression tree passes."""

    def __call__(self, expression: Expression) -> Any:
        return self.visit(expression)

    def visit(self, expression: Expression) -> Any:
        """Visit an expression.

        Args:
            expression: Expression to visit.

        Returns:
            Result of the visit.

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

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Any:
        """Visit a unary expression.

        Args:
            unary_expression: Unary expression to visit.

        Returns:
            Result of the visit.

        """

    def visit_binary_expression(self, binary_expression: BinaryExpression) -> Any:
        """Visit a binary expression.

        Args:
            binary_expression: Binary expression to visit.

        Returns:
            Result of the visit.

        """

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Any:
        """Visit an identifier.

        Args:
            identifier_expression: Identifier expression to visit.

        Returns:
            Result of the visit.

        """

    def visit_literal_expression(self, literal_expression: LiteralExpression) -> Any:
        """Visit a literal.

        Args:
            literal_expression: Literal expression to visit.

        Returns:
            Result of the visit.

        """


class ExpressionVisitor(ExpressionBasePass, ABC):
    """Visitor for expression trees."""

    def __call__(self, expression: Expression) -> None:
        super().__call__(expression)

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> None:
        self.visit(unary_expression.operand)

    def visit_binary_expression(self, binary_expression: BinaryExpression) -> None:
        self.visit(binary_expression.left)
        self.visit(binary_expression.right)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> None: ...

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> None: ...


class ExpressionTransformer(ExpressionBasePass, ABC):
    """Transformer for expression trees."""

    def __call__(self, expression: Expression) -> Expression:
        transformed_expression = super().__call__(expression)
        if not isinstance(transformed_expression, Expression):
            raise TypeError(
                f"Invalid transformed expression type: {type(transformed_expression)}"
            )
        return transformed_expression

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Expression:
        new_expression = self.visit(unary_expression.operand)
        return UnaryExpression(
            operation=unary_expression.operation, operand=new_expression
        )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> Expression:
        new_left = self.visit(binary_expression.left)
        new_right = self.visit(binary_expression.right)
        return BinaryExpression(
            operation=binary_expression.operation, left=new_left, right=new_right
        )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        return IdentifierExpression(identifier=identifier_expression.identifier)

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> Expression:
        return LiteralExpression(value=literal_expression.value)
