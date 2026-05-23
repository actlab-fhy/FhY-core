"""General expression tree."""

__all__ = [
    "Expression",
    "LiteralType",
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
    "CallExpression",
    "IdentifierExpression",
    "LiteralExpression",
    "TernaryExpression",
    "call",
    "logical_and",
    "logical_or",
    "ternary",
]

import re
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from functools import singledispatch
from typing import Any, TypeAlias, TypedDict, TypeGuard

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

LiteralType: TypeAlias = str | float | int | bool


def _build_right_folded_binary_tree(
    operation: "BinaryOperation",
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    if len(expressions) < 2:  # noqa: PLR2004
        operation_name = BINARY_OPERATION_FUNCTION_NAMES[operation]
        raise ValueError(
            f"{operation_name} requires at least two expressions, but got "
            f"{len(expressions)}."
        )
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


def logical_and(
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a right-folded ``LOGICAL_AND`` tree from two or more operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        expressions: Operands to AND together. Must be at least two.

    Returns:
        Right-folded binary AND expression.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand has an unsupported type.

    """
    return _build_right_folded_binary_tree(BinaryOperation.LOGICAL_AND, *expressions)


def logical_or(
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a right-folded ``LOGICAL_OR`` tree from two or more operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        expressions: Operands to OR together. Must be at least two.

    Returns:
        Right-folded binary OR expression.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand has an unsupported type.

    """
    return _build_right_folded_binary_tree(BinaryOperation.LOGICAL_OR, *expressions)


def ternary(
    condition: "Expression | Identifier | LiteralType",
    true_value: "Expression | Identifier | LiteralType",
    false_value: "Expression | Identifier | LiteralType",
) -> "TernaryExpression":
    """Build a ``TernaryExpression`` from three operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        condition: Scalar boolean expression operand.
        true_value: Expression chosen when ``condition`` is true.
        false_value: Expression chosen when ``condition`` is false.

    Returns:
        A ``TernaryExpression`` over the three coerced operands.

    Raises:
        ValueError: If an operand has an unsupported type.

    """
    return TernaryExpression(
        Expression._get_expression_from_other(condition),
        Expression._get_expression_from_other(true_value),
        Expression._get_expression_from_other(false_value),
    )


def call(
    function_name: str,
    *arguments: "Expression | Identifier | LiteralType",
) -> "CallExpression":
    """Build a ``CallExpression`` from a name and positional arguments.

    Each argument may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        function_name: Registry key of the function being called.
        arguments: Positional argument operands.

    Returns:
        A ``CallExpression`` over the coerced arguments.

    Raises:
        ValueError: If an argument has an unsupported type.

    """
    coerced = tuple(
        Expression._get_expression_from_other(argument) for argument in arguments
    )
    return CallExpression(function_name, coerced)


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

    def logical_and(
        self, *others: "Expression | Identifier | LiteralType"
    ) -> "BinaryExpression":
        """Create a logical AND expression over ``self`` and ``others``.

        Each item in ``others`` may be an ``Expression``, an ``Identifier``,
        or a value of ``LiteralType``; the same coercion rules as the
        operator dunders apply.

        Args:
            others: Additional operands to AND with ``self``. At least one
                is required (``self`` is always included).

        Returns:
            Right-folded logical AND expression.

        Raises:
            ValueError: If no additional operands are supplied, or if an
                operand has an unsupported type.

        """
        return logical_and(self, *others)

    def logical_or(
        self, *others: "Expression | Identifier | LiteralType"
    ) -> "BinaryExpression":
        """Create a logical OR expression over ``self`` and ``others``.

        Each item in ``others`` may be an ``Expression``, an ``Identifier``,
        or a value of ``LiteralType``; the same coercion rules as the
        operator dunders apply.

        Args:
            others: Additional operands to OR with ``self``. At least one
                is required (``self`` is always included).

        Returns:
            Right-folded logical OR expression.

        Raises:
            ValueError: If no additional operands are supplied, or if an
                operand has an unsupported type.

        """
        return logical_or(self, *others)

    @staticmethod
    def ternary(
        condition: "Expression | Identifier | LiteralType",
        true_value: "Expression | Identifier | LiteralType",
        false_value: "Expression | Identifier | LiteralType",
    ) -> "TernaryExpression":
        """Build a ``TernaryExpression`` from three operands.

        Each operand may be an ``Expression``, an ``Identifier``, or a
        value of ``LiteralType``; the same coercion rules as the operator
        dunders apply.

        Args:
            condition: Scalar boolean expression operand.
            true_value: Expression chosen when ``condition`` is true.
            false_value: Expression chosen when ``condition`` is false.

        Returns:
            A ``TernaryExpression`` over the three coerced operands.

        Raises:
            ValueError: If an operand has an unsupported type.

        """
        return ternary(condition, true_value, false_value)

    @staticmethod
    def call(
        function_name: str,
        *arguments: "Expression | Identifier | LiteralType",
    ) -> "CallExpression":
        """Build a ``CallExpression`` from a name and positional arguments.

        Each argument may be an ``Expression``, an ``Identifier``, or a
        value of ``LiteralType``; the same coercion rules as the operator
        dunders apply.

        Args:
            function_name: Registry key of the function being called.
            arguments: Positional argument operands.

        Returns:
            A ``CallExpression`` over the coerced arguments.

        Raises:
            ValueError: If an argument has an unsupported type.

        """
        return call(function_name, *arguments)

    @staticmethod
    def _get_expression_from_other(other: Any) -> "Expression":
        if isinstance(other, Expression):
            return other
        elif isinstance(other, Identifier):
            return IdentifierExpression(other)
        elif type(other) is bool or type(other) in (int, float, str):
            return LiteralExpression(other)
        else:
            raise ValueError(
                f"Unable to cast {other!r} with type {type(other)} to an expression."
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

    def get_operands(self) -> tuple[Expression]:
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

    def get_operands(self) -> tuple[Expression, Expression]:
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


_INTEGER_LITERAL_PATTERN = re.compile(r"\d+")
_FLOAT_LITERAL_PATTERN = re.compile(r"\d+\.\d*|\.\d+")


class _LiteralExpressionData(TypedDict):
    value: LiteralType


def _is_valid_literal_expression_data(
    data: SerializedDict,
) -> TypeGuard[_LiteralExpressionData]:
    return "value" in data and isinstance(data["value"], (str, float, int, bool))


@register_serializable(type_id="literal_expression")
@dataclass(frozen=True, eq=False)
class LiteralExpression(Expression):
    r"""Literal expression.

    Stored value follows these canonicalization rules:

    - ``bool`` / ``int`` / ``float`` values are stored unchanged. ``bool`` is
      checked before ``int`` to keep the two distinct.
    - ``str`` values matching the integer grammar (``\d+``) are canonicalized
      to ``int`` via ``int(value)`` so ``LiteralExpression("5")`` and
      ``LiteralExpression(5)`` are indistinguishable.
    - ``str`` values matching the float grammar (``\d+\.\d*`` or
      ``\.\d+``) are kept as ``str`` to preserve exact decimal text. Native
      ``float`` would impose IEEE-754 rounding, which the design avoids.
    - Any other ``str`` raises ``ValueError``; any other Python type raises
      ``TypeError``.
    """

    value: LiteralType

    def __post_init__(self) -> None:
        value = self.value
        if type(value) is bool:
            return
        if type(value) is int:
            return
        if type(value) is float:
            return
        if not isinstance(value, str):
            raise TypeError(
                f"Unsupported type for literal expression value: "
                f"{type(value).__name__}; expected str, float, int, or bool."
            )
        if _INTEGER_LITERAL_PATTERN.fullmatch(value):
            object.__setattr__(self, "value", int(value))
            return
        if _FLOAT_LITERAL_PATTERN.fullmatch(value):
            return
        raise ValueError(
            f"Invalid string-form literal expression value: {value!r} "
            f"does not match the integer or float grammar."
        )

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"value": self.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "LiteralExpression":
        if not _is_valid_literal_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _LiteralExpressionData.__annotations__, data
            )
        return cls(data["value"])


class _TernaryExpressionData(TypedDict):
    condition: SerializedDict
    true_value: SerializedDict
    false_value: SerializedDict


def _is_valid_ternary_expression_data(
    data: SerializedDict,
) -> TypeGuard[_TernaryExpressionData]:
    return (
        "condition" in data
        and is_serialized_dict(data["condition"])
        and "true_value" in data
        and is_serialized_dict(data["true_value"])
        and "false_value" in data
        and is_serialized_dict(data["false_value"])
    )


@register_serializable(type_id="ternary_expression")
@dataclass(frozen=True, eq=False)
class TernaryExpression(Expression, HasOperandsMixin[Expression]):
    """Ternary conditional expression: ``condition ? true_value : false_value``.

    A pure 3-arg form: when ``condition`` evaluates to true the
    expression takes the value of ``true_value``; otherwise it takes the
    value of ``false_value``. Both branches are part of the expression
    tree; the form does not imply lazy evaluation at the IR level.

    Attributes:
        condition: Scalar boolean expression.
        true_value: Result when ``condition`` is true.
        false_value: Result when ``condition`` is false.

    """

    condition: Expression
    true_value: Expression
    false_value: Expression

    def get_operands(self) -> tuple[Expression, Expression, Expression]:
        return (self.condition, self.true_value, self.false_value)

    def get_visit_children(self) -> tuple["Expression", ...]:
        return (self.condition, self.true_value, self.false_value)

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "condition": self.condition.serialize_to_dict(),
            "true_value": self.true_value.serialize_to_dict(),
            "false_value": self.false_value.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "TernaryExpression":
        if not _is_valid_ternary_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _TernaryExpressionData.__annotations__, data
            )
        return cls(
            Expression.deserialize_from_dict(data["condition"]),
            Expression.deserialize_from_dict(data["true_value"]),
            Expression.deserialize_from_dict(data["false_value"]),
        )


class _CallExpressionData(TypedDict):
    function_name: str
    arguments: list[SerializedDict]


def _is_valid_call_expression_data(
    data: SerializedDict,
) -> TypeGuard[_CallExpressionData]:
    return (
        "function_name" in data
        and isinstance(data["function_name"], str)
        and "arguments" in data
        and isinstance(data["arguments"], list)
        and all(is_serialized_dict(argument) for argument in data["arguments"])
    )


@register_serializable(type_id="call_expression")
@dataclass(frozen=True, eq=False)
class CallExpression(Expression, HasOperandsMixin[Expression]):
    """Reference to a registered function applied to argument expressions.

    The node stores the function's registry key and the argument
    expressions. Arity is not validated at construction time; resolution
    and arity checking happen in the type-checker and the inliner so the
    AST itself can be built without the registry having to be loaded.

    Attributes:
        function_name: Key under which the called function is registered.
        arguments: Positional argument expressions, in declared order.

    """

    function_name: str
    arguments: tuple[Expression, ...]

    def get_operands(self) -> tuple[Expression, ...]:
        return self.arguments

    def get_visit_children(self) -> tuple["Expression", ...]:
        return self.arguments

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "function_name": self.function_name,
            "arguments": [argument.serialize_to_dict() for argument in self.arguments],
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "CallExpression":
        if not _is_valid_call_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _CallExpressionData.__annotations__, data
            )
        arguments = tuple(
            Expression.deserialize_from_dict(argument) for argument in data["arguments"]
        )
        return cls(data["function_name"], arguments)


@singledispatch
def _is_expression_structurally_equivalent(
    expression: Expression, other: object
) -> bool:
    raise NotImplementedError(
        f"is_structurally_equivalent is not registered for {type(expression).__name__}."
    )


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
    return (
        isinstance(other, LiteralExpression)
        and type(expression.value) is type(other.value)
        and expression.value == other.value
    )


@_is_expression_structurally_equivalent.register
def _(expression: TernaryExpression, other: object) -> bool:
    return (
        isinstance(other, TernaryExpression)
        and expression.condition.is_structurally_equivalent(other.condition)
        and expression.true_value.is_structurally_equivalent(other.true_value)
        and expression.false_value.is_structurally_equivalent(other.false_value)
    )


@_is_expression_structurally_equivalent.register
def _(expression: CallExpression, other: object) -> bool:
    return (
        isinstance(other, CallExpression)
        and expression.function_name == other.function_name
        and len(expression.arguments) == len(other.arguments)
        and all(
            left.is_structurally_equivalent(right)
            for left, right in zip(expression.arguments, other.arguments)
        )
    )
