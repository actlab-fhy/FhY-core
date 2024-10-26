"""General expression tree."""

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.utils import invert_frozen_dict


class Expression(ABC):
    """Abstract base class for expressions."""


class UnaryOperation(Enum):
    """Unary operation."""

    NEGATE = auto()
    POSITIVE = auto()
    BITWISE_NOT = auto()
    LOGICAL_NOT = auto()


UNARY_OPERATION_FUNCTION_NAMES: frozendict[UnaryOperation, str] = frozendict(
    {
        UnaryOperation.NEGATE: "negate",
        UnaryOperation.POSITIVE: "positive",
        UnaryOperation.BITWISE_NOT: "bitwise_not",
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
        UnaryOperation.BITWISE_NOT: "~",
    }
)
UNARY_SYMBOL_OPERATIONS: frozendict[str, UnaryOperation] = invert_frozen_dict(
    UNARY_OPERATION_SYMBOLS
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
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
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
        BinaryOperation.BITWISE_AND: "bitwise_and",
        BinaryOperation.BITWISE_OR: "bitwise_or",
        BinaryOperation.BITWISE_XOR: "bitwise_xor",
        BinaryOperation.LEFT_SHIFT: "left_shift",
        BinaryOperation.RIGHT_SHIFT: "right_shift",
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
        BinaryOperation.BITWISE_AND: "&",
        BinaryOperation.BITWISE_OR: "|",
        BinaryOperation.BITWISE_XOR: "^",
        BinaryOperation.LEFT_SHIFT: "<<",
        BinaryOperation.RIGHT_SHIFT: ">>",
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
                complex(value)
            except ValueError:
                raise ValueError(
                    f"Invalid literal expression value: "
                    f"{value} with type {type(value)}."
                )
        self._value = value

    @property
    def value(self) -> LiteralType:
        return self._value
