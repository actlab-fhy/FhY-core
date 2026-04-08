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
from functools import singledispatch
from typing import Any, TypedDict, TypeGuard

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
    WrappedFamilySerializable,
    is_serialized_dict,
    register_serializable,
)
from fhy_core.trait import (
    FrozenMixin,
    HasOperandsMixin,
    StructuralEquivalenceMixin,
    VisitableMixin,
)
from fhy_core.utils import invert_frozen_dict


class SymbolType(Enum):
    """Symbol type."""

    REAL = auto()
    INT = auto()
    BOOL = auto()


class Expression(
    WrappedFamilySerializable,
    FrozenMixin,
    StructuralEquivalenceMixin,
    VisitableMixin,
    ABC,
):
    """Abstract base class for expressions."""

    def is_structurally_equivalent(self, other: object) -> bool:
        return _is_expression_structurally_equivalent(self, other)

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


class _UnaryExpressionData(TypedDict):
    operation: str
    operand: SerializedDict


def _is_valid_unary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_UnaryExpressionData]:
    return (
        "operation" in data
        and isinstance(data["operation"], str)
        and "operand" in data
        and is_serialized_dict(data["operand"])
    )


@register_serializable(type_id="unary_expression")
@dataclass(frozen=True, eq=False)
class UnaryExpression(Expression, HasOperandsMixin[Expression]):
    """Unary expression."""

    operation: UnaryOperation
    operand: Expression

    @property
    def operands(self) -> tuple[Expression]:
        return (self.operand,)

    def get_visit_children(self) -> tuple["Expression", ...]:
        return (self.operand,)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "operation": UNARY_OPERATION_FUNCTION_NAMES[self.operation],
            "operand": self.operand.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "UnaryExpression":
        if not _is_valid_unary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _UnaryExpressionData.__annotations__, data
            )
        operation_name = data["operation"]
        if operation_name not in UNARY_FUNCTION_NAME_OPERATIONS:
            raise DeserializationValueError(
                cls, "operation", "a valid unary operation name", operation_name
            )
        operand = Expression.deserialize_from_dict(data["operand"])
        return cls(
            UNARY_FUNCTION_NAME_OPERATIONS[operation_name],
            operand,
        )


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


class _BinaryExpressionData(TypedDict):
    operation: str
    left: SerializedDict
    right: SerializedDict


def _is_valid_binary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_BinaryExpressionData]:
    return (
        "operation" in data
        and isinstance(data["operation"], str)
        and "left" in data
        and is_serialized_dict(data["left"])
        and "right" in data
        and is_serialized_dict(data["right"])
    )


@register_serializable(type_id="binary_expression")
@dataclass(frozen=True, eq=False)
class BinaryExpression(Expression, HasOperandsMixin[Expression]):
    """Binary expression."""

    operation: BinaryOperation
    left: Expression
    right: Expression

    @property
    def operands(self) -> tuple[Expression, Expression]:
        return (self.left, self.right)

    def get_visit_children(self) -> tuple["Expression", ...]:
        return (self.left, self.right)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "operation": BINARY_OPERATION_FUNCTION_NAMES[self.operation],
            "left": self.left.serialize_to_dict(),
            "right": self.right.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "BinaryExpression":
        if not _is_valid_binary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _BinaryExpressionData.__annotations__, data
            )
        operation_name = data["operation"]
        if operation_name not in BINARY_FUNCTION_NAME_OPERATIONS:
            raise DeserializationValueError(
                cls, "operation", "a valid binary operation name", operation_name
            )
        left = Expression.deserialize_from_dict(data["left"])
        right = Expression.deserialize_from_dict(data["right"])
        return cls(
            BINARY_FUNCTION_NAME_OPERATIONS[operation_name],
            left,
            right,
        )


class _IdentifierExpressionData(TypedDict):
    identifier: SerializedDict


def _is_valid_identifier_expression_data(
    data: SerializedDict,
) -> TypeGuard[_IdentifierExpressionData]:
    return "identifier" in data and is_serialized_dict(data["identifier"])


@register_serializable(type_id="identifier_expression")
@dataclass(frozen=True, eq=False)
class IdentifierExpression(Expression):
    """Identifier expression."""

    identifier: Identifier

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"identifier": self.identifier.serialize_to_dict()}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "IdentifierExpression":
        if not _is_valid_identifier_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _IdentifierExpressionData.__annotations__, data
            )
        return cls(Identifier.deserialize_from_dict(data["identifier"]))


LiteralType = str | float | int | bool


class _LiteralExpressionData(TypedDict):
    value: LiteralType


def _is_valid_literal_expression_data(
    data: SerializedDict,
) -> TypeGuard[_LiteralExpressionData]:
    return "value" in data and isinstance(data["value"], (str, float, int, bool))


@register_serializable(type_id="literal_expression")
@dataclass(frozen=True, eq=False)
class LiteralExpression(Expression):
    """Literal expression."""

    value: LiteralType

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            return
        try:
            float(self.value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid literal expression value: "
                f'{self.value} with type "{type(self.value)}".'
            ) from exc

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"value": self.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "LiteralExpression":
        if not _is_valid_literal_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _LiteralExpressionData.__annotations__, data
            )
        return cls(data["value"])


@singledispatch
def _is_expression_structurally_equivalent(
    expression: Expression, other: object
) -> bool:
    return False


@_is_expression_structurally_equivalent.register
def _(expression: UnaryExpression, other: object) -> bool:
    return (
        isinstance(other, UnaryExpression)
        and expression.operation == other.operation
        and expression.operand.is_structurally_equivalent(other.operand)
    )


@_is_expression_structurally_equivalent.register
def _(expression: BinaryExpression, other: object) -> bool:
    return (
        isinstance(other, BinaryExpression)
        and expression.operation == other.operation
        and expression.left.is_structurally_equivalent(other.left)
        and expression.right.is_structurally_equivalent(other.right)
    )


@_is_expression_structurally_equivalent.register
def _(expression: IdentifierExpression, other: object) -> bool:
    return (
        isinstance(other, IdentifierExpression)
        and expression.identifier == other.identifier
    )


@_is_expression_structurally_equivalent.register
def _(expression: LiteralExpression, other: object) -> bool:
    return isinstance(other, LiteralExpression) and expression.value == other.value
