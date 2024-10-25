"""General expression tree."""

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

from fhy_core.identifier import Identifier


class Expression(ABC):
    """Abstract base class for expressions."""


class UnaryOperation(StrEnum):
    """Unary operation."""

    NEGATE = "negate"
    BITWISE_NOT = "bitwise_not"
    LOGICAL_NOT = "logical_not"


@dataclass(frozen=True, eq=False)
class UnaryExpression(Expression):
    """Unary expression."""

    operation: UnaryOperation
    operand: Expression


class BinaryOperation(StrEnum):
    """Binary operation."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MODULO = "modulo"
    POWER = "power"
    BITWISE_AND = "bitwise_and"
    BITWISE_OR = "bitwise_or"
    BITWISE_XOR = "bitwise_xor"
    LEFT_SHIFT = "left_shift"
    RIGHT_SHIFT = "right_shift"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"


@dataclass(frozen=True, eq=False)
class BinaryExpression(Expression):
    """Binary expression."""

    operation: BinaryOperation
    left: Expression
    right: Expression


@dataclass(frozen=True, eq=False)
class IdentifierExpression(Expression):
    """Identifier expression."""

    identifier: Identifier


LiteralType = str | complex | float | int | bool


class LiteralExpression(Expression):
    """Literal expression."""

    _value: LiteralType

    def __init__(self, value: LiteralType) -> None:
        if isinstance(value, str):
            try:
                value = complex(value)
            except ValueError:
                raise ValueError(
                    f"Invalid literal expression value: "
                    f"{value} with type {type(value)}."
                )
        self._value = value

    @property
    def value(self) -> LiteralType:
        return self._value
