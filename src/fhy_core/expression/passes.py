"""Expression tree passes."""

from operator import (
    add,
    and_,
    eq,
    ge,
    gt,
    invert,
    le,
    lshift,
    lt,
    mod,
    mul,
    ne,
    neg,
    not_,
    or_,
    pow,
    rshift,
    sub,
    truediv,
    xor,
)
from typing import Callable, ClassVar

from fhy_core.identifier import Identifier

from .core import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
    UnaryOperation,
)
from .transformer import ExpressionTransformer


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


# TODO: Resolve how different data types are handled here by the operations
#       (i.e., string representations of decimal numbers, etc.)
class EvaluateExpression(ExpressionTransformer):
    """Evaluate an expression tree.

    Args:
        environment: Environment to evaluate the expression in.

    """

    _BINARY_OPERATIONS: ClassVar[dict[BinaryOperation, Callable]] = {
        BinaryOperation.ADD: add,
        BinaryOperation.SUBTRACT: sub,
        BinaryOperation.MULTIPLY: mul,
        BinaryOperation.DIVIDE: truediv,
        BinaryOperation.MODULO: mod,
        BinaryOperation.POWER: pow,
        BinaryOperation.BITWISE_AND: and_,
        BinaryOperation.BITWISE_OR: or_,
        BinaryOperation.BITWISE_XOR: xor,
        BinaryOperation.LEFT_SHIFT: lshift,
        BinaryOperation.RIGHT_SHIFT: rshift,
        BinaryOperation.LOGICAL_AND: and_,
        BinaryOperation.LOGICAL_OR: or_,
        BinaryOperation.EQUAL: eq,
        BinaryOperation.NOT_EQUAL: ne,
        BinaryOperation.LESS: lt,
        BinaryOperation.LESS_EQUAL: le,
        BinaryOperation.GREATER: gt,
        BinaryOperation.GREATER_EQUAL: ge,
    }
    _UNARY_OPERATIONS: ClassVar[dict[UnaryOperation, Callable]] = {
        UnaryOperation.NEGATE: neg,
        UnaryOperation.BITWISE_NOT: invert,
        UnaryOperation.LOGICAL_NOT: not_,
    }

    _environment: dict[Identifier, LiteralType]

    def __init__(self, environment: dict[Identifier, LiteralType]):
        self._environment = environment

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> Expression:
        new_operand = self.visit(unary_expression.operand)
        if isinstance(new_operand, LiteralExpression):
            operation = self._UNARY_OPERATIONS[unary_expression.operation]
            return LiteralExpression(operation(new_operand.value))
        else:
            return UnaryExpression(
                operation=unary_expression.operation, operand=new_operand
            )

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> Expression:
        new_left = self.visit(binary_expression.left)
        new_right = self.visit(binary_expression.right)
        if isinstance(new_left, LiteralExpression) and isinstance(
            new_right, LiteralExpression
        ):
            operation = self._BINARY_OPERATIONS[binary_expression.operation]
            return LiteralExpression(operation(new_left.value, new_right.value))
        else:
            return BinaryExpression(
                operation=binary_expression.operation, left=new_left, right=new_right
            )

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> Expression:
        if identifier_expression.identifier in self._environment:
            return LiteralExpression(
                self._environment[identifier_expression.identifier]
            )
        else:
            return IdentifierExpression(identifier=identifier_expression.identifier)


def evaluate_expression(
    expression: Expression, environment: dict[Identifier, LiteralType] | None = None
) -> LiteralType:
    """Evaluate an expression.

    Args:
        expression: Expression to evaluate.
        environment: Environment to evaluate the expression in. Defaults to None.

    Returns:
        Result of the expression.

    """
    return EvaluateExpression(environment or {})(expression)
