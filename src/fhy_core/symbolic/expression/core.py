"""General expression tree."""

from fhy_core.utils.override import override

__all__ = [
    "BINARY_OPERATION_SYMBOLS",
    "BINARY_SYMBOL_OPERATIONS",
    "UNARY_OPERATION_SYMBOLS",
    "UNARY_SYMBOL_OPERATIONS",
    "BinaryExpression",
    "BinaryOperation",
    "CallExpression",
    "Expression",
    "IdentifierExpression",
    "LiteralExpression",
    "LiteralType",
    "PiecewiseExpression",
    "UnaryExpression",
    "UnaryOperation",
    "build_literal_equivalence_key",
    "call",
    "is_integer_valued_literal",
    "logical_and",
    "logical_not",
    "logical_or",
    "make_binary_expression",
    "make_unary_expression",
    "piecewise",
    "validate_logical_operands",
]

import math
import re
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal
from typing import Any, TypeAlias, TypedDict, TypeGuard

from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    SerializedDict,
    WrappedFamilySerializable,
    register_serializable,
)
from fhy_core.term import (
    DerivedEquivalenceMixin,
    Term,
    compared_as_reference,
    compared_as_value,
)
from fhy_core.traits import (
    FrozenMixin,
    HasOperands,
    RewritableMixin,
    VisitableMixin,
)
from fhy_core.utils import StrEnum, invert_frozen_dict

from .errors import NonBooleanLogicalOperandError
from .sort import FunctionSort

LiteralType: TypeAlias = str | float | int | bool


def _make_bare_bool_coercion_error(position: str, value: bool) -> ValueError:
    """Return the error refusing to coerce a bare Python ``bool`` to an expression.

    ``Expression.__eq__`` is object identity, so ``expr == k`` evaluates to
    a Python ``bool`` rather than to an IR equality node. Lifting that
    ``bool`` would plant a constant where the caller meant a comparison,
    so no builder coerces one; a Boolean constant is written
    ``LiteralExpression(True)`` or ``LiteralExpression(False)``.

    Args:
        position: Where the ``bool`` was supplied; leads the message.
        value: The refused ``bool``.

    Returns:
        ``ValueError`` naming the position and both intended spellings.

    """
    return ValueError(
        f"{position} is a bare Python bool ({value!r}), not an expression. "
        "This is almost always the accidental result of `expr == k`, which "
        "is Expression identity comparison, not IR equality; use `.equals()` "
        "or `.not_equals()` to build an equality, or write "
        f"`LiteralExpression({value!r})` for a Boolean constant."
    )


def make_binary_expression(
    operation: "BinaryOperation",
    left: "Expression | Identifier | LiteralType",
    right: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a ``BinaryExpression`` from two coercible operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        operation: Binary operation to apply.
        left: Left operand.
        right: Right operand.

    Returns:
        A ``BinaryExpression`` over the two coerced operands.

    Raises:
        ValueError: If an operand is a bare Python ``bool`` or has an
            unsupported type.

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
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        operation: Unary operation to apply.
        operand: Operand to wrap.

    Returns:
        A ``UnaryExpression`` over the coerced operand.

    Raises:
        ValueError: If the operand is a bare Python ``bool`` or has an
            unsupported type.

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
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        expression: Operand to negate.

    Returns:
        ``LOGICAL_NOT`` unary expression over the coerced operand.

    Raises:
        ValueError: If the operand is a bare Python ``bool`` or has an
            unsupported type.

    """
    return make_unary_expression(UnaryOperation.LOGICAL_NOT, expression)


def logical_and(
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a right-folded ``LOGICAL_AND`` tree from two or more operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        expressions: Operands to AND together. Must be at least two.

    Returns:
        Right-folded binary AND expression.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand is a bare Python ``bool`` or has an unsupported type.

    """
    return _build_right_folded_binary_tree(BinaryOperation.LOGICAL_AND, *expressions)


def logical_or(
    *expressions: "Expression | Identifier | LiteralType",
) -> "BinaryExpression":
    """Build a right-folded ``LOGICAL_OR`` tree from two or more operands.

    Each operand may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        expressions: Operands to OR together. Must be at least two.

    Returns:
        Right-folded binary OR expression.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand is a bare Python ``bool`` or has an unsupported type.

    """
    return _build_right_folded_binary_tree(BinaryOperation.LOGICAL_OR, *expressions)


def piecewise(
    *cases: tuple[
        "Expression | Identifier | LiteralType",
        "Expression | Identifier | LiteralType",
    ],
    otherwise: "Expression | Identifier | LiteralType",
) -> "PiecewiseExpression":
    """Build a ``PiecewiseExpression`` from ``(condition, value)`` case pairs.

    Each element of each pair, and ``otherwise``, may be an ``Expression``
    (used as-is), an ``Identifier`` (wrapped in ``IdentifierExpression``),
    or a value of ``LiteralType`` other than ``bool`` (wrapped in
    ``LiteralExpression``); the same coercion rules as the operator
    dunders apply.

    Args:
        cases: One or more ``(condition, value)`` pairs, in evaluation
            order.
        otherwise: Result when no case's condition holds.

    Returns:
        A ``PiecewiseExpression`` over the coerced cases and ``otherwise``.

    Raises:
        ValueError: If no case is supplied, if a case is not a 2-tuple,
            or if an operand is a bare Python ``bool`` or has an
            unsupported type.

    """
    if not cases:
        raise ValueError("piecewise requires at least one (condition, value) case.")
    conditions: list[Expression] = []
    values: list[Expression] = []
    for index, case in enumerate(cases):
        if not (isinstance(case, tuple) and len(case) == 2):  # noqa: PLR2004
            raise ValueError(
                f"piecewise case {index} must be a 2-tuple of (condition, value), "
                f"got {case!r}."
            )
        condition, value = case
        if type(condition) is bool:
            raise _make_bare_bool_coercion_error(
                f"piecewise case {index} condition", condition
            )
        conditions.append(Expression._get_expression_from_other(condition))
        values.append(Expression._get_expression_from_other(value))
    return PiecewiseExpression(
        tuple(conditions),
        tuple(values),
        Expression._get_expression_from_other(otherwise),
    )


def call(
    function_name: str,
    *arguments: "Expression | Identifier | LiteralType",
) -> "CallExpression":
    """Build a ``CallExpression`` from a name and positional arguments.

    Each argument may be an ``Expression`` (used as-is), an ``Identifier``
    (wrapped in ``IdentifierExpression``), or a value of ``LiteralType``
    other than ``bool`` (wrapped in ``LiteralExpression``); the same
    coercion rules as the operator dunders apply.

    Args:
        function_name: Registry key of the function being called.
        arguments: Positional argument operands.

    Returns:
        A ``CallExpression`` over the coerced arguments.

    Raises:
        ValueError: If an argument is a bare Python ``bool`` or has an
            unsupported type.

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
    that the comparison and arithmetic dunders (``__lt__``, ``__le__``,
    ``__gt__``, ``__ge__``, ``__add__``, etc.) can return
    :class:`BinaryExpression` IR nodes instead of ``bool``. As a
    consequence, ``__eq__`` and ``__hash__`` fall back to object
    identity. Two structurally equivalent expressions are therefore
    **distinct dict keys** and **distinct set members**: use
    :meth:`is_structurally_equivalent` for value-equality semantics,
    and avoid using :class:`Expression` instances as dict keys when you
    expect value-based lookups.

    For the same reason, no operator dunder or builder coerces a bare
    Python ``bool`` operand: ``expr == k`` is a ``bool``, and lifting it
    would plant a constant where a comparison was meant. Build an
    equality with :meth:`equals` and a Boolean constant with
    ``LiteralExpression(True)`` or ``LiteralExpression(False)``.

    Expressions are :class:`~fhy_core.term.Term` instances: they compare
    by alpha-equivalence (derived from the field schema), report their free
    identifiers, and support substitution. The IR has no binders, so every
    referenced identifier is free and substitution is always capture-free.
    Both trait methods derive generically from
    :meth:`get_visit_children` / :meth:`rebuild_with_visit_children`;
    :class:`IdentifierExpression` overrides them as the recursion base case.
    """

    @override
    def get_visit_children(self) -> tuple["Expression", ...]:
        return ()

    @override
    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "Expression":
        if not new_children:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} has children but does not implement "
            "`rebuild_with_visit_children`."
        )

    def get_free_identifiers(self) -> frozenset[Identifier]:
        """Return the identifiers referenced free in this expression.

        Defaults to the union of the children's free identifiers;
        :class:`IdentifierExpression` overrides this to report itself.
        """
        free: frozenset[Identifier] = frozenset()
        for child in self.get_visit_children():
            free |= child.get_free_identifiers()
        return free

    def substitute(self, replacements: Mapping[Identifier, Term]) -> "Expression":
        """Return this expression with mapped identifiers replaced.

        Substitution is capture-free (the IR has no binders). Defaults to
        rebuilding from substituted children;
        :class:`IdentifierExpression` overrides this to perform the
        replacement.

        Args:
            replacements: Identifier-to-expression substitutions. Values
                must be :class:`Expression` instances.

        Returns:
            The substituted expression.

        Raises:
            TypeError: If a replacement value is not an
                :class:`Expression`.
            ValueError: If a replacement puts a non-``bool`` literal in
                a piecewise case condition, which
                :class:`PiecewiseExpression` refuses to hold. Screening
                the expression and the replacements with
                :func:`validate_logical_operands` first reports that
                case as :class:`NonBooleanLogicalOperandError` instead.
        """
        return self.rebuild_with_visit_children(
            tuple(child.substitute(replacements) for child in self.get_visit_children())
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

        Raises:
            ValueError: If ``other`` is a bare Python ``bool`` or has an
                unsupported type.

        """
        return make_binary_expression(BinaryOperation.EQUAL, self, other)

    def not_equals(self, other: Any) -> "BinaryExpression":
        """Create an inequality expression.

        Args:
            other: Other expression.

        Returns:
            Inequality expression.

        Raises:
            ValueError: If ``other`` is a bare Python ``bool`` or has an
                unsupported type.

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
        or a value of ``LiteralType`` other than ``bool``; the same
        coercion rules as the operator dunders apply.

        Args:
            others: Additional operands to AND with ``self``. At least one
                is required (``self`` is always included).

        Returns:
            Right-folded logical AND expression.

        Raises:
            ValueError: If no additional operands are supplied, or if an
                operand is a bare Python ``bool`` or has an unsupported
                type.

        """
        return logical_and(self, *others)

    def logical_or(
        self, *others: "Expression | Identifier | LiteralType"
    ) -> "BinaryExpression":
        """Create a logical OR expression over ``self`` and ``others``.

        Each item in ``others`` may be an ``Expression``, an ``Identifier``,
        or a value of ``LiteralType`` other than ``bool``; the same
        coercion rules as the operator dunders apply.

        Args:
            others: Additional operands to OR with ``self``. At least one
                is required (``self`` is always included).

        Returns:
            Right-folded logical OR expression.

        Raises:
            ValueError: If no additional operands are supplied, or if an
                operand is a bare Python ``bool`` or has an unsupported
                type.

        """
        return logical_or(self, *others)

    @staticmethod
    def piecewise(
        *cases: tuple[
            "Expression | Identifier | LiteralType",
            "Expression | Identifier | LiteralType",
        ],
        otherwise: "Expression | Identifier | LiteralType",
    ) -> "PiecewiseExpression":
        """Build a ``PiecewiseExpression``; delegates to the free :func:`piecewise`.

        See :func:`piecewise` for the operand coercion rules and the
        conditions under which construction raises.
        """
        return piecewise(*cases, otherwise=otherwise)

    @staticmethod
    def call(
        function_name: str,
        *arguments: "Expression | Identifier | LiteralType",
    ) -> "CallExpression":
        """Build a ``CallExpression`` from a name and positional arguments.

        Each argument may be an ``Expression``, an ``Identifier``, or a
        value of ``LiteralType`` other than ``bool``; the same coercion
        rules as the operator dunders apply.

        Args:
            function_name: Registry key of the function being called.
            arguments: Positional argument operands.

        Returns:
            A ``CallExpression`` over the coerced arguments.

        Raises:
            ValueError: If an argument is a bare Python ``bool`` or has
                an unsupported type.

        """
        return call(function_name, *arguments)

    @staticmethod
    def _get_expression_from_other(other: Any) -> "Expression":
        if isinstance(other, Expression):
            return other
        elif isinstance(other, Identifier):
            return IdentifierExpression(other)
        elif type(other) is bool:
            raise _make_bare_bool_coercion_error("Operand", other)
        elif type(other) in (int, float, str):
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


UNARY_OPERATION_SYMBOLS: immutabledict[UnaryOperation, str] = immutabledict(
    {
        UnaryOperation.NEGATE: "-",
        UnaryOperation.POSITIVE: "+",
        UnaryOperation.LOGICAL_NOT: "!",
    }
)
UNARY_SYMBOL_OPERATIONS: immutabledict[str, UnaryOperation] = invert_frozen_dict(
    UNARY_OPERATION_SYMBOLS
)


@register_serializable(type_id="unary_expression")
@dataclass(frozen=True, eq=False)
class UnaryExpression(Expression, HasOperands[Expression]):
    """Unary expression."""

    operation: UnaryOperation
    operand: Expression

    @override
    def get_operands(self) -> tuple[Expression]:
        return (self.operand,)

    @override
    def get_visit_children(self) -> tuple["Expression", ...]:
        return (self.operand,)

    @override
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


BINARY_OPERATION_SYMBOLS: immutabledict[BinaryOperation, str] = immutabledict(
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
BINARY_SYMBOL_OPERATIONS: immutabledict[str, BinaryOperation] = invert_frozen_dict(
    BINARY_OPERATION_SYMBOLS
)


@register_serializable(type_id="binary_expression")
@dataclass(frozen=True, eq=False)
class BinaryExpression(Expression, HasOperands[Expression]):
    """Binary expression."""

    operation: BinaryOperation
    left: Expression
    right: Expression

    @override
    def get_operands(self) -> tuple[Expression, Expression]:
        return (self.left, self.right)

    @override
    def get_visit_children(self) -> tuple["Expression", ...]:
        return (self.left, self.right)

    @override
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

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.identifier})

    @override
    def substitute(self, replacements: Mapping[Identifier, Term]) -> "Expression":
        replacement = replacements.get(self.identifier)
        if isinstance(replacement, Expression):
            return replacement
        if replacement is not None:
            raise TypeError(
                f"Cannot substitute {self.identifier!r} with a non-Expression "
                f"term of type {type(replacement).__name__}."
            )
        return self


_INTEGER_LITERAL_PATTERN = re.compile(r"\d+")
_FLOAT_LITERAL_PATTERN = re.compile(r"\d+\.\d*|\.\d+")


_LiteralBucket: TypeAlias = tuple[str, "bool | int | float | str | Decimal"]

_INTEGER_LITERAL_BUCKET = "int"
_BOOLEAN_LITERAL_BUCKET = "bool"
_FLOAT_BINARY_LITERAL_BUCKET = "float-binary"
_FLOAT_DECIMAL_LITERAL_BUCKET = "float-decimal"

_CANONICAL_NAN_FORM = "nan"
"""Canonical form every NaN ``float`` shares in the float-binary bucket."""


def _normalize_decimal_exactly(value: Decimal) -> Decimal:
    """Return ``value`` with its trailing coefficient zeros stripped, unrounded.

    ``Decimal.normalize`` strips trailing zeros but first rounds to the
    context precision, 28 digits by default, which would merge distinct
    long decimals. A context exactly as wide as the coefficient, with
    the widest exponent range, leaves nothing to round, so numerically
    equal decimals come out identical and unequal ones stay apart.
    """
    precision = len(value.as_tuple().digits)
    return value.normalize(Context(prec=precision, Emax=MAX_EMAX, Emin=MIN_EMIN))


def _classify_literal_value(value: LiteralType) -> _LiteralBucket:
    """Return the (bucket, canonical-form) pair used for literal equivalence.

    Two literals are equivalent exactly when their pairs are equal, and
    within a bucket equal canonical forms are identical values, which is
    what lets :func:`build_literal_equivalence_key` render the pair as
    text. A binary ``float`` gains zero, folding ``-0.0`` into the
    ``0.0`` it already equals. A NaN equals nothing, itself included, so
    every NaN takes one shared token instead of its own value; without
    it, whether two NaN literals were equivalent would depend on
    whether they held the same ``float`` object. An exact decimal is
    stripped of trailing zeros without rounding, so ``"1.5"`` and
    ``"1.50"`` coincide.
    """
    if isinstance(value, bool):
        return (_BOOLEAN_LITERAL_BUCKET, value)
    elif isinstance(value, int):
        return (_INTEGER_LITERAL_BUCKET, value)
    elif isinstance(value, float):
        if math.isnan(value):
            return (_FLOAT_BINARY_LITERAL_BUCKET, _CANONICAL_NAN_FORM)
        return (_FLOAT_BINARY_LITERAL_BUCKET, value + 0.0)
    elif _INTEGER_LITERAL_PATTERN.fullmatch(value):
        return (_INTEGER_LITERAL_BUCKET, int(value))
    else:
        return (
            _FLOAT_DECIMAL_LITERAL_BUCKET,
            _normalize_decimal_exactly(Decimal(value)),
        )


def build_literal_equivalence_key(value: LiteralType) -> str:
    """Return text two literal values share exactly when they are equivalent.

    Renders the bucket and canonical form :class:`LiteralExpression`
    compares by, so the key cannot disagree with structural equivalence
    in either direction: ``5``, ``"5"``, and ``"05"`` share a key, as do
    ``"1.5"`` and ``"1.50"``, ``0.0`` and ``-0.0``, and every NaN, while
    a ``bool`` keys apart from every integer and an exact-decimal string
    apart from the binary ``float`` with the same digits. The key is a
    pure function of the value, so it is identical in every process and
    across a serialization or pickle round trip.

    Args:
        value: Value stored on a :class:`LiteralExpression`.

    Returns:
        Bucket-prefixed text, such as ``"int:5"`` or
        ``"float-binary:nan"``.

    """
    bucket, canonical = _classify_literal_value(value)
    return f"{bucket}:{canonical}"


def is_integer_valued_literal(value: LiteralType) -> bool:
    """Return whether a literal value falls in the integer bucket.

    Reads the same bucket :class:`LiteralExpression` compares by, so the
    answer is constant on structural-equivalence classes of literals: a
    Python ``int`` and an integer-grammar ``str`` denoting the same
    integer (``5``, ``"5"``, ``"05"``) all answer True, while a ``bool``,
    a Python ``float``, and a float-grammar ``str`` all answer False.
    Whenever the answer is True, ``int(value)`` recovers that integer
    exactly.

    Every consumer that has to decide "is this literal an integer?" asks
    here, which is what establishes the invariant that two structurally
    equivalent literals lower to the same Z3 sort and the same SymPy
    number kind, and are classified identically by the solver's int/float
    hazard screen. A consumer applying its own rule breaks that
    congruence, leaving one equivalence class part decided and part
    refused.

    Args:
        value: Value stored on a :class:`LiteralExpression`.

    Returns:
        True when the value denotes an integer.

    """
    bucket, _ = _classify_literal_value(value)
    return bucket == _INTEGER_LITERAL_BUCKET


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
    - float-binary (Python ``float``): by value, except that every NaN
      is equivalent to every other NaN, keeping the relation reflexive
      for the one value IEEE-754 makes unequal to itself; ``-0.0`` and
      ``0.0`` are equivalent because they compare equal.
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

    @override
    def serialize_data_to_dict(self) -> SerializedDict:
        return {"value": self.value}

    @classmethod
    @override
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "LiteralExpression":
        if not _is_valid_literal_expression_data(data):
            raise DeserializationDictStructureError(
                cls, _LiteralExpressionData.__annotations__, data
            )
        return cls(data["value"])


@register_serializable(type_id="piecewise_expression")
@dataclass(frozen=True, eq=False)
class PiecewiseExpression(Expression, HasOperands[Expression]):
    """Mathematical piecewise expression: ordered first-match cases with a fallback.

    The expression denotes the value of the first case whose condition
    holds; if no condition holds it denotes ``otherwise``, which makes
    the expression a total function. Overlapping conditions are legal:
    first match wins. Cases are stored as two parallel tuples rather
    than a tuple of pairs so the fields derive automatic serialization,
    matching :class:`CallExpression`; :meth:`get_cases` reconstructs the
    ``(condition, value)`` pairs for callers that want case-shaped
    iteration. The form does not imply lazy evaluation at the IR level.

    Attributes:
        conditions: Scalar boolean case conditions, in evaluation order.
        values: Case result values, positionally paired with
            ``conditions``.
        otherwise: Result when no condition holds; makes the function
            total.

    Raises:
        ValueError: If no case is supplied, if ``conditions`` and
            ``values`` differ in length, if any element of
            ``conditions``, any element of ``values``, or
            ``otherwise`` is not an :class:`Expression` instance, or if
            a condition is a :class:`LiteralExpression` whose value is
            not a Python ``bool``.

    """

    conditions: tuple[Expression, ...]
    values: tuple[Expression, ...]
    otherwise: Expression

    def __post_init__(self) -> None:
        conditions = tuple(self.conditions)
        values = tuple(self.values)
        if not conditions:
            raise ValueError(
                "PiecewiseExpression requires at least one case; a piecewise "
                "with only `otherwise` is meaningless -- use the value directly."
            )
        if len(conditions) != len(values):
            raise ValueError(
                "PiecewiseExpression.conditions and .values must have equal "
                f"length, but got {len(conditions)} conditions and "
                f"{len(values)} values."
            )
        for condition in conditions:
            if not isinstance(condition, Expression):
                raise ValueError(
                    "PiecewiseExpression.conditions must contain only "
                    f"Expression instances, but got value {condition!r} of "
                    f"type {type(condition).__name__}."
                )
            if (
                isinstance(condition, LiteralExpression)
                and type(condition.value) is not bool
            ):
                raise ValueError(
                    "PiecewiseExpression condition literal must be a boolean "
                    f"literal, but got literal value {condition.value!r} of "
                    f"type {type(condition.value).__name__}. A non-literal "
                    "condition (an identifier, a call, ...) has no such "
                    "restriction here, since its type is not known without "
                    "surrounding type context."
                )
        for value in values:
            if not isinstance(value, Expression):
                raise ValueError(
                    "PiecewiseExpression.values must contain only Expression "
                    f"instances, but got value {value!r} of type "
                    f"{type(value).__name__}."
                )
        if not isinstance(self.otherwise, Expression):
            raise ValueError(
                "PiecewiseExpression.otherwise must be an Expression "
                f"instance, but got value {self.otherwise!r} of type "
                f"{type(self.otherwise).__name__}."
            )
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "values", values)

    def get_cases(self) -> tuple[tuple[Expression, Expression], ...]:
        """Return the ``(condition, value)`` case pairs in evaluation order.

        Returns:
            The zipped ``(condition, value)`` pairs.

        """
        return tuple(zip(self.conditions, self.values, strict=True))

    @override
    def get_operands(self) -> tuple[Expression, ...]:
        return self.get_visit_children()

    @override
    def get_visit_children(self) -> tuple["Expression", ...]:
        interleaved: list[Expression] = []
        for condition, value in self.get_cases():
            interleaved.append(condition)
            interleaved.append(value)
        interleaved.append(self.otherwise)
        return tuple(interleaved)

    @override
    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "PiecewiseExpression":
        flat = tuple(new_children)
        expected = 2 * len(self.conditions) + 1
        if len(flat) != expected:
            raise ValueError(
                "PiecewiseExpression.rebuild_with_visit_children requires one "
                f"child per condition and value plus otherwise ({expected} for "
                f"{len(self.conditions)} cases), but got {len(flat)}. Accepting "
                "any odd count would silently rebuild the node with a different "
                "number of cases."
            )
        case_children = flat[:-1]
        conditions = case_children[0::2]
        values = case_children[1::2]
        otherwise = flat[-1]
        return PiecewiseExpression(conditions, values, otherwise)


@register_serializable(type_id="call_expression")
@dataclass(frozen=True, eq=False)
class CallExpression(Expression, HasOperands[Expression]):
    """Reference to a registered function applied to argument expressions.

    The node stores the function's registry key and the argument
    expressions. Arity is not validated at construction time; resolution
    and arity checking happen in the type-checker and the inliner so the
    AST itself can be built without the registry having to be loaded.

    Attributes:
        function_name: Key under which the called function is registered.
        arguments: Positional argument expressions, in declared order.

    Raises:
        ValueError: If ``function_name`` is empty, or if any element of
            ``arguments`` is not an :class:`Expression` instance.

    """

    function_name: str
    arguments: tuple[Expression, ...]

    def __post_init__(self) -> None:
        if not self.function_name:
            raise ValueError("CallExpression.function_name must be non-empty.")
        arguments = tuple(self.arguments)
        for argument in arguments:
            if not isinstance(argument, Expression):
                raise ValueError(
                    "CallExpression.arguments must contain only Expression "
                    f"instances, but got value {argument!r} of type "
                    f"{type(argument).__name__}."
                )
        object.__setattr__(self, "arguments", arguments)

    @override
    def get_operands(self) -> tuple[Expression, ...]:
        return self.arguments

    @override
    def get_visit_children(self) -> tuple["Expression", ...]:
        return self.arguments

    @override
    def rebuild_with_visit_children(
        self, new_children: Sequence["Expression"]
    ) -> "CallExpression":
        return CallExpression(self.function_name, tuple(new_children))


_LOGICAL_CONNECTIVE_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR}
)
"""Binary operations that denote a Boolean connective over their operands."""

_ARITHMETIC_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.ADD,
        BinaryOperation.SUBTRACT,
        BinaryOperation.MULTIPLY,
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    }
)
"""Binary operations that denote arithmetic, so their result is a number."""


def _get_native_constant_sort(identifier: Identifier) -> FunctionSort | None:
    """Return the declared sort of the native constant ``identifier`` denotes.

    Returns ``None`` unless ``identifier`` is a registered native
    constant's canonical identifier; one that merely shares a constant's
    name is an ordinary variable.
    """
    # Deferred import: the registry's entry types import this module, so
    # importing the registry at module scope here would form a cycle.
    from .registry import try_get_native_constant_for_identifier  # noqa: PLC0415

    constant = try_get_native_constant_for_identifier(identifier)
    return None if constant is None else constant.sort


def _is_identifier_provably_non_boolean(
    identifier: Identifier, environment: Mapping[Identifier, Expression]
) -> bool:
    """Return whether ``identifier`` provably denotes a number.

    A native constant's canonical identifier takes the constant's declared
    sort, whatever ``environment`` binds it to. Any other identifier takes
    its binding's classification, and an unbound one answers False.
    """
    constant_sort = _get_native_constant_sort(identifier)
    if constant_sort is not None:
        return constant_sort is not FunctionSort.BOOL
    bound = environment.get(identifier)
    return bound is not None and _is_provably_non_boolean(bound, {})


def _is_provably_non_boolean(
    expression: Expression, environment: Mapping[Identifier, Expression]
) -> bool:
    """Return whether ``expression`` provably denotes a number, not a Boolean.

    Answers conservatively: a node whose sort cannot be read off the tree
    -- an identifier with no ``environment`` binding, and a call, whose
    result sort lives in the registry rather than in the node -- answers
    False, so a caller screening on "provably non-Boolean" refuses only
    what it can prove. A registered native constant's canonical
    identifier is the exception: it denotes the constant's value, so it
    takes the constant's declared sort, which is REAL for every built-in
    constant.

    Args:
        expression: Node whose sort is wanted.
        environment: Values bound to identifiers before the expression is
            handed to a backend; a bound identifier takes its value's
            classification. The value is classified on its own, with no
            binding applied to it in turn, matching the simultaneous
            non-chaining semantics of substitution. A native constant's
            canonical identifier is classified by the constant's sort
            whatever it is bound to: it names a value rather than a
            variable, and the SymPy bridge lowers it to that value.

    Returns:
        True when the node denotes a number.

    """
    if isinstance(expression, LiteralExpression):
        bucket, _ = _classify_literal_value(expression.value)
        return bucket != _BOOLEAN_LITERAL_BUCKET
    elif isinstance(expression, IdentifierExpression):
        return _is_identifier_provably_non_boolean(expression.identifier, environment)
    elif isinstance(expression, UnaryExpression):
        return expression.operation is not UnaryOperation.LOGICAL_NOT
    elif isinstance(expression, BinaryExpression):
        return expression.operation in _ARITHMETIC_BINARY_OPERATIONS
    elif isinstance(expression, PiecewiseExpression):
        return all(
            _is_provably_non_boolean(branch, environment)
            for branch in (*expression.values, expression.otherwise)
        )
    return False


def _get_boolean_position_operands(expression: Expression) -> tuple[Expression, ...]:
    """Return the children of ``expression`` that sit in a Boolean position.

    Those are the operands of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
    ``LOGICAL_NOT`` node and the case conditions of a piecewise, which
    selects its branch by truth. Every other child may be of any sort as
    far as its parent is concerned.
    """
    if (
        isinstance(expression, UnaryExpression)
        and expression.operation is UnaryOperation.LOGICAL_NOT
    ) or (
        isinstance(expression, BinaryExpression)
        and expression.operation in _LOGICAL_CONNECTIVE_BINARY_OPERATIONS
    ):
        return expression.get_operands()
    if isinstance(expression, PiecewiseExpression):
        return expression.conditions
    return ()


def _find_non_boolean_logical_operand(
    expression: Expression, environment: Mapping[Identifier, Expression]
) -> tuple[Expression, Expression] | None:
    """Return the first numeric Boolean-position operand paired with its parent."""
    for operand in _get_boolean_position_operands(expression):
        if _is_provably_non_boolean(operand, environment):
            return expression, operand
    for child in expression.get_visit_children():
        found = _find_non_boolean_logical_operand(child, environment)
        if found is not None:
            return found
    return None


def validate_logical_operands(
    expression: Expression,
    environment: Mapping[Identifier, Expression] | None = None,
) -> None:
    """Raise unless every Boolean position in ``expression`` holds a Boolean.

    A Boolean position is an operand of ``LOGICAL_AND``, ``LOGICAL_OR``,
    or ``LOGICAL_NOT``, or a piecewise case condition. No symbolic
    backend gives a number there a faithful meaning: SymPy's ``&``/``|``
    are bitwise on ``sympy.Integer``, its ``Not`` coerces by truthiness,
    and its ``Piecewise`` rejects a numeric condition or, once a
    substitution has put a number there, reads it as a truth value;
    Z3 rejects the sort outright. The whole tree is screened, so a
    numeric operand nested anywhere under the root is found.

    Only a provably numeric operand is refused. A registered native
    constant's canonical identifier counts as the constant's declared
    sort, which is REAL for every built-in constant, so ``pi`` under a
    connective is refused. Any other identifier with no ``environment``
    binding, and a call, keep their sort off the tree, so they pass: the
    screen refuses what it can prove ill-typed rather than everything it
    cannot prove well-typed.

    Args:
        expression: Expression about to be lowered to a symbolic backend,
            or to have ``environment`` substituted into it.
        environment: Values bound to identifiers before lowering, so an
            identifier bound to a number is screened as one. A binding
            for a native constant's canonical identifier is not
            consulted: the identifier names a value rather than a
            variable. Defaults to ``None``, meaning no identifier is
            bound.

    Raises:
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, provably denotes a number.

    """
    found = _find_non_boolean_logical_operand(expression, environment or {})
    if found is None:
        return
    connective, operand = found
    if isinstance(connective, PiecewiseExpression):
        raise NonBooleanLogicalOperandError(
            f"{connective!r} takes {operand!r} as a case condition, which "
            "provably denotes a number; the expression is ill-typed and no "
            "symbolic backend lowers it faithfully."
        )
    raise NonBooleanLogicalOperandError(
        f"{connective!r} applies a Boolean connective to the operand "
        f"{operand!r}, which provably denotes a number; the expression is "
        f"ill-typed and no symbolic backend lowers it faithfully."
    )
