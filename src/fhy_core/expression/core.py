"""General expression tree."""

__all__ = [
    "Expression",
    "UnaryOperation",
    "UNARY_OPERATION_FUNCTION_NAMES",
    "UNARY_FUNCTION_NAME_OPERATIONS",
    "UNARY_OPERATION_SYMBOLS",
    "UNARY_SYMBOL_OPERATIONS",
    "UnaryExpression",
    "BinaryOperation",
    "BINARY_OPERATION_FUNCTION_NAMES",
    "BINARY_FUNCTION_NAME_OPERATIONS",
    "BINARY_OPERATION_SYMBOLS",
    "BINARY_SYMBOL_OPERATIONS",
    "BinaryExpression",
    "IdentifierExpression",
    "LiteralExpression",
]

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.utils import invert_frozen_dict


class SymbolType(Enum):
    """Symbol type."""

    REAL = auto()
    INT = auto()
    BOOL = auto()


class Expression(ABC):
    """Abstract base class for expressions."""

    def __neg__(self) -> "UnaryExpression":
        return UnaryExpression(UnaryOperation.NEGATE, self)

    def __pos__(self) -> "UnaryExpression":
        return UnaryExpression(UnaryOperation.POSITIVE, self)

    def logical_not(self) -> "UnaryExpression":
        """Create a logical NOT expression.

        Returns:
            Logical NOT expression.

        """
        return UnaryExpression(UnaryOperation.LOGICAL_NOT, self)

    def __add__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.ADD, self, self._get_expression_from_other(other)
        )

    def __radd__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.ADD, self._get_expression_from_other(other), self
        )

    def __sub__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.SUBTRACT, self, self._get_expression_from_other(other)
        )

    def __rsub__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.SUBTRACT, self._get_expression_from_other(other), self
        )

    def __mul__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.MULTIPLY, self, self._get_expression_from_other(other)
        )

    def __rmul__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.MULTIPLY, self._get_expression_from_other(other), self
        )

    def __truediv__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.DIVIDE, self, self._get_expression_from_other(other)
        )

    def __rtruediv__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.DIVIDE, self._get_expression_from_other(other), self
        )

    def __floordiv__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE, self, self._get_expression_from_other(other)
        )

    def __rfloordiv__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.FLOOR_DIVIDE, self._get_expression_from_other(other), self
        )

    def __mod__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.MODULO, self, self._get_expression_from_other(other)
        )

    def __rmod__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.MODULO, self._get_expression_from_other(other), self
        )

    def __pow__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.POWER, self, self._get_expression_from_other(other)
        )

    def __rpow__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.POWER, self._get_expression_from_other(other), self
        )

    def equals(self, other: Any) -> "BinaryExpression":
        """Create an equality expression.

        Args:
            other: Other expression.

        Returns:
            Equality expression.

        """
        return BinaryExpression(
            BinaryOperation.EQUAL, self, self._get_expression_from_other(other)
        )

    def not_equals(self, other: Any) -> "BinaryExpression":
        """Create an inequality expression.

        Args:
            other: Other expression.

        Returns:
            Inequality expression.

        """
        return BinaryExpression(
            BinaryOperation.NOT_EQUAL, self, self._get_expression_from_other(other)
        )

    def __lt__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.LESS, self, self._get_expression_from_other(other)
        )

    def __le__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.LESS_EQUAL, self, self._get_expression_from_other(other)
        )

    def __gt__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.GREATER, self, self._get_expression_from_other(other)
        )

    def __ge__(self, other: Any) -> "BinaryExpression":
        return BinaryExpression(
            BinaryOperation.GREATER_EQUAL, self, self._get_expression_from_other(other)
        )

    @staticmethod
    def logical_and(*expressions: "Expression") -> "BinaryExpression":
        """Create a logical AND expression.

        Args:
            expressions: Expressions to AND together.

        Returns:
            Logical AND expression.

        """
        return Expression._generate_commutative_associative_operation_tree(
            BinaryOperation.LOGICAL_AND, *expressions
        )

    @staticmethod
    def logical_or(*expressions: "Expression") -> "BinaryExpression":
        """Create a logical OR expression.

        Args:
            expressions: Expressions to OR together.

        Returns:
            Logical OR expression.

        """
        return Expression._generate_commutative_associative_operation_tree(
            BinaryOperation.LOGICAL_OR, *expressions
        )

    @staticmethod
    def _generate_commutative_associative_operation_tree(
        operation: "BinaryOperation", *expressions: "Expression"
    ) -> "BinaryExpression":
        if len(expressions) < 2:  # noqa: PLR2004
            raise ValueError("At least two expressions are required.")
        reversed_expressions = list(reversed(expressions))
        result = BinaryExpression(
            operation,
            Expression._get_expression_from_other(reversed_expressions[1]),
            Expression._get_expression_from_other(reversed_expressions[0]),
        )
        for next_expression in reversed_expressions[2:]:
            result = BinaryExpression(
                operation,
                Expression._get_expression_from_other(next_expression),
                result,
            )
        return result

    @staticmethod
    def _get_expression_from_other(other: Any) -> "Expression":
        if isinstance(other, Expression):
            return other
        elif isinstance(other, Identifier):
            return IdentifierExpression(other)
        elif isinstance(other, (int, float, bool, str)):
            return LiteralExpression(other)
        raise ValueError(
            f"Unsupported type for creating literal expression: {type(other)}."
        )


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
    FLOOR_DIVIDE = auto()
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
        BinaryOperation.FLOOR_DIVIDE: "floor_divide",
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
        BinaryOperation.FLOOR_DIVIDE: "//",
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
