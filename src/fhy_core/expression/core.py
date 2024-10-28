"""General expression tree."""

import operator
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.utils import invert_frozen_dict


class Expression(ABC):
    """Abstract base class for expressions."""


class UnaryOperation(Enum):
    """Unary operation."""

    NEGATE = auto()
    POSITIVE = auto()
    LOGICAL_NOT = auto()


UNARY_OPERATION_FUNCTION_NAMES: frozendict[UnaryOperation, str] = frozendict(
    {
        UnaryOperation.NEGATE: "negate",
        UnaryOperation.POSITIVE: "positive",
        UnaryOperation.LOGICAL_NOT: "logical_not",
    }
)
UNARY_FUNCTION_NAME_OPERATIONS: frozendict[str, UnaryOperation] = invert_frozen_dict(
    UNARY_OPERATION_FUNCTION_NAMES
)
UNARY_OPERATION_SYMBOLS: frozendict[UnaryOperation, str] = frozendict(
    {
        UnaryOperation.NEGATE: "-",
        UnaryOperation.POSITIVE: "+",
        UnaryOperation.LOGICAL_NOT: "!",
    }
)
UNARY_SYMBOL_OPERATIONS: frozendict[str, UnaryOperation] = invert_frozen_dict(
    UNARY_OPERATION_SYMBOLS
)
UNARY_OPERATION_OPERATORS: frozendict[UnaryOperation, Callable[[Any], Any]] = (
    frozendict(
        {
            UnaryOperation.NEGATE: operator.neg,
            UnaryOperation.POSITIVE: operator.pos,
            UnaryOperation.LOGICAL_NOT: operator.not_,
        }
    )
)


@dataclass(frozen=True, eq=False)
class UnaryExpression(Expression):
    """Unary expression."""

    operation: UnaryOperation
    operand: Expression


class BinaryOperation(Enum):
    """Binary operation."""

    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()


BINARY_OPERATION_FUNCTION_NAMES: frozendict[BinaryOperation, str] = frozendict(
    {
        BinaryOperation.ADD: "add",
        BinaryOperation.SUBTRACT: "subtract",
        BinaryOperation.MULTIPLY: "multiply",
        BinaryOperation.DIVIDE: "divide",
        BinaryOperation.MODULO: "modulo",
        BinaryOperation.POWER: "power",
        BinaryOperation.LOGICAL_AND: "logical_and",
        BinaryOperation.LOGICAL_OR: "logical_or",
        BinaryOperation.EQUAL: "equal",
        BinaryOperation.NOT_EQUAL: "not_equal",
        BinaryOperation.LESS: "less",
        BinaryOperation.LESS_EQUAL: "less_equal",
        BinaryOperation.GREATER: "greater",
        BinaryOperation.GREATER_EQUAL: "greater_equal",
    }
)
BINARY_FUNCTION_NAME_OPERATIONS: frozendict[str, BinaryOperation] = invert_frozen_dict(
    BINARY_OPERATION_FUNCTION_NAMES
)
BINARY_OPERATION_SYMBOLS: frozendict[BinaryOperation, str] = frozendict(
    {
        BinaryOperation.ADD: "+",
        BinaryOperation.SUBTRACT: "-",
        BinaryOperation.MULTIPLY: "*",
        BinaryOperation.DIVIDE: "/",
        BinaryOperation.MODULO: "%",
        BinaryOperation.POWER: "**",
        BinaryOperation.LOGICAL_AND: "&&",
        BinaryOperation.LOGICAL_OR: "||",
        BinaryOperation.EQUAL: "==",
        BinaryOperation.NOT_EQUAL: "!=",
        BinaryOperation.LESS: "<",
        BinaryOperation.LESS_EQUAL: "<=",
        BinaryOperation.GREATER: ">",
        BinaryOperation.GREATER_EQUAL: ">=",
    }
)
BINARY_SYMBOL_OPERATIONS: frozendict[str, BinaryOperation] = invert_frozen_dict(
    BINARY_OPERATION_SYMBOLS
)
BINARY_OPERATION_OPERATORS: frozendict[BinaryOperation, Callable[[Any, Any], Any]] = (
    frozendict(
        {
            BinaryOperation.ADD: operator.add,
            BinaryOperation.SUBTRACT: operator.sub,
            BinaryOperation.MULTIPLY: operator.mul,
            BinaryOperation.DIVIDE: operator.truediv,
            BinaryOperation.MODULO: operator.mod,
            BinaryOperation.POWER: operator.pow,
            BinaryOperation.LOGICAL_AND: operator.and_,
            BinaryOperation.LOGICAL_OR: operator.or_,
            BinaryOperation.EQUAL: operator.eq,
            BinaryOperation.NOT_EQUAL: operator.ne,
            BinaryOperation.LESS: operator.lt,
            BinaryOperation.LESS_EQUAL: operator.le,
            BinaryOperation.GREATER: operator.gt,
            BinaryOperation.GREATER_EQUAL: operator.ge,
        }
    )
)


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


LiteralType = str | float | int | bool


class LiteralExpression(Expression):
    """Literal expression."""

    _value: LiteralType

    def __init__(self, value: LiteralType) -> None:
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                raise ValueError(
                    f"Invalid literal expression value: "
                    f"{value} with type {type(value)}."
                )
        self._value = value

    @property
    def value(self) -> LiteralType:
        return self._value
