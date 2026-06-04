"""General expression tree."""

__all__ = [
    "Expression",
    "LiteralType",
    "UnaryOperation",
    "UNARY_OPERATION_SYMBOLS",
    "UNARY_SYMBOL_OPERATIONS",
    "UnaryExpression",
    "BinaryOperation",
    "BINARY_OPERATION_SYMBOLS",
    "BINARY_SYMBOL_OPERATIONS",
    "BinaryExpression",
    "CallExpression",
    "IdentifierExpression",
    "LiteralExpression",
    "TernaryExpression",
    "call",
    "logical_and",
    "logical_not",
    "logical_or",
    "make_binary_expression",
    "make_unary_expression",
    "ternary",
]

import re
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, TypeAlias, TypedDict, TypeGuard

from frozendict import frozendict

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    SerializedDict,
    WrappedFamilySerializable,
    register_serializable,
)
from fhy_core.traits import (
    DerivedEquivalenceMixin,
    FrozenMixin,
    HasOperandsMixin,
    RewritableMixin,
    VisitableMixin,
    compared_as_reference,
    compared_as_value,
)
from fhy_core.utils import StrEnum, invert_frozen_dict

LiteralType: TypeAlias = str | float | int | bool


def make_binary_expression(
    operation: "BinaryOperation",
    left: "Expression | Identifier | LiteralType",
    right: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a ``BinaryExpression`` from two coercible operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        operation: Binary operation to apply.
        left: Left operand.
        right: Right operand.

    Returns:
        A ``BinaryExpression`` over the two coerced operands.

    Raises:
        ValueError: If an operand has an unsupported type.

    """
    return BinaryExpression(
        operation,
        Expression._get_expression_from_other(left),
        Expression._get_expression_from_other(right),
    )


def make_unary_expression(
    operation: "UnaryOperation",
    operand: "Expression | Identifier | LiteralType",
) -> "UnaryExpression":
    """Build a ``UnaryExpression`` from one coercible operand.

    The operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        operation: Unary operation to apply.
        operand: Operand to wrap.

    Returns:
        A ``UnaryExpression`` over the coerced operand.

    Raises:
        ValueError: If the operand has an unsupported type.

    """
    return UnaryExpression(operation, Expression._get_expression_from_other(operand))


def _build_right_folded_binary_tree(
    operation: "BinaryOperation",
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    if len(expressions) < 2:  # noqa: PLR2004
        operation_name = operation.value
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


def logical_not(
    expression: "Expression | Identifier | LiteralType",
) -> "UnaryExpression":
    """Wrap ``expression`` in a ``LOGICAL_NOT`` unary expression.

    The operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    (wrapped in ``LiteralExpression``); the same coercion rules as the
    operator dunders apply.

    Args:
        expression: Operand to negate.

    Returns:
        ``LOGICAL_NOT`` unary expression over the coerced operand.

    Raises:
        ValueError: If the operand has an unsupported type.

    """
    return make_unary_expression(UnaryOperation.LOGICAL_NOT, expression)


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
    DerivedEquivalenceMixin,
    VisitableMixin,
    RewritableMixin["Expression"],
    ABC,
):
    """Abstract base class for expressions.

    Expression subclasses are ``@dataclass(frozen=True, eq=False)`` so
    that the comparison dunders (``__lt__``, ``__eq__``, etc.) can
    return :class:`BinaryExpression` IR nodes instead of ``bool``. As a
    consequence, ``__eq__`` and ``__hash__`` fall back to object
    identity. Two structurally equivalent expressions are therefore
    **distinct dict keys** and **distinct set members**: use
    :meth:`is_structurally_equivalent` for value-equality semantics,
    and avoid using :class:`Expression` instances as dict keys when you
    expect value-based lookups.
    """

    def get_visit_children(self) -> tuple["Expression", ...]:
        return ()

    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "Expression":
        if not new_children:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} has children but does not implement "
            "`rebuild_with_visit_children`."
        )

    def __neg__(self) -> "UnaryExpression":
        return make_unary_expression(UnaryOperation.NEGATE, self)

    def __pos__(self) -> "UnaryExpression":
        return make_unary_expression(UnaryOperation.POSITIVE, self)

    def __add__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.ADD, self, other)

    def __radd__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.ADD, other, self)

    def __sub__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.SUBTRACT, self, other)

    def __rsub__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.SUBTRACT, other, self)

    def __mul__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.MULTIPLY, self, other)

    def __rmul__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.MULTIPLY, other, self)

    def __truediv__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.DIVIDE, self, other)

    def __rtruediv__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.DIVIDE, other, self)

    def __floordiv__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.FLOOR_DIVIDE, self, other)

    def __rfloordiv__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.FLOOR_DIVIDE, other, self)

    def __mod__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.MODULO, self, other)

    def __rmod__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.MODULO, other, self)

    def __pow__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.POWER, self, other)

    def __rpow__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.POWER, other, self)

    def equals(self, other: Any) -> "BinaryExpression":
        """Create an equality expression.

        Args:
            other: Other expression.

        Returns:
            Equality expression.

        """
        return make_binary_expression(BinaryOperation.EQUAL, self, other)

    def not_equals(self, other: Any) -> "BinaryExpression":
        """Create an inequality expression.

        Args:
            other: Other expression.

        Returns:
            Inequality expression.

        """
        return make_binary_expression(BinaryOperation.NOT_EQUAL, self, other)

    def __lt__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.LESS, self, other)

    def __le__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.LESS_EQUAL, self, other)

    def __gt__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.GREATER, self, other)

    def __ge__(self, other: Any) -> "BinaryExpression":
        return make_binary_expression(BinaryOperation.GREATER_EQUAL, self, other)

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


class UnaryOperation(StrEnum):
    """Unary operation.

    Each member's value is its canonical function name (``NEGATE`` ->
    ``"negate"``), which is also its serialized form.
    """

    NEGATE = "negate"
    POSITIVE = "positive"
    LOGICAL_NOT = "logical_not"


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

    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "UnaryExpression":
        (operand,) = new_children
        return UnaryExpression(self.operation, operand)


class BinaryOperation(StrEnum):
    """Binary operation.

    Each member's value is its canonical function name (``ADD`` -> ``"add"``),
    which is also its serialized form.
    """

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    FLOOR_DIVIDE = "floor_divide"
    MODULO = "modulo"
    POWER = "power"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"


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

    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "BinaryExpression":
        left, right = new_children
        return BinaryExpression(self.operation, left, right)


@register_serializable(type_id="identifier_expression")
@dataclass(frozen=True, eq=False)
class IdentifierExpression(Expression):
    """Identifier expression."""

    identifier: Identifier = field(metadata=compared_as_reference())


_INTEGER_LITERAL_PATTERN = re.compile(r"\d+")
_FLOAT_LITERAL_PATTERN = re.compile(r"\d+\.\d*|\.\d+")


_LiteralBucket: TypeAlias = tuple[str, "bool | int | float | Decimal"]


def _classify_literal_value(value: LiteralType) -> _LiteralBucket:
    """Return the (bucket, canonical-form) pair used for literal equivalence."""
    if isinstance(value, bool):
        return ("bool", value)
    elif isinstance(value, int):
        return ("int", value)
    elif isinstance(value, float):
        return ("float-binary", value)
    elif _INTEGER_LITERAL_PATTERN.fullmatch(value):
        return ("int", int(value))
    else:
        return ("float-decimal", Decimal(value))


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

    Stored value follows these rules:

    - ``bool`` / ``int`` / ``float`` values are stored unchanged. ``bool`` is
      checked before ``int`` to keep the two distinct.
    - ``str`` values matching the integer grammar (``\d+``) or the float
      grammar (``\d+\.\d*`` or ``\.\d+``) are stored as ``str`` to preserve
      the caller's exact text. Native ``float`` would impose IEEE-754
      rounding; native ``int`` would erase the string representation
      choice. The design preserves both so downstream passes can perform
      exact-decimal arithmetic before any conversion to ``float``.
    - Any other ``str`` raises ``ValueError``; any other Python type raises
      ``TypeError``.

    Structural and alpha equivalence compare literals by *bucket* and
    *canonical form*, not by stored Python type:

    - ``bool``: by value.
    - integer (Python ``int`` or integer-grammar ``str``): canonicalized
      to the underlying integer, so ``LiteralExpression("5")``,
      ``LiteralExpression("05")``, and ``LiteralExpression(5)`` are all
      equivalent.
    - float-binary (Python ``float``): by value.
    - float-decimal (float-grammar ``str``): canonicalized to a
      ``decimal.Decimal`` so ``"1.5"`` and ``"1.50"`` are equivalent;
      ``str`` form and ``float`` form are *not* cross-equivalent, since
      the exact-decimal text and the IEEE-754 binary value carry
      different precision contracts.
    """

    value: LiteralType = field(metadata=compared_as_value(key=_classify_literal_value))

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

    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "TernaryExpression":
        condition, true_value, false_value = new_children
        return TernaryExpression(condition, true_value, false_value)


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

    def __post_init__(self) -> None:
        if not self.function_name:
            raise ValueError("CallExpression.function_name must be non-empty.")

    def get_operands(self) -> tuple[Expression, ...]:
        return self.arguments

    def get_visit_children(self) -> tuple["Expression", ...]:
        return self.arguments

    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "CallExpression":
        return CallExpression(self.function_name, tuple(new_children))
