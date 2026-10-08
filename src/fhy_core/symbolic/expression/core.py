"""General expression tree.

The expression classes are backed by the Rust core, ``fhy_core::expression``,
and take its semantics:

- ``==`` and ``hash`` are structural: two separately built expressions of
  the same structure are equal and hash alike, and each expression computes
  its hash once;
- a literal's value is normalized to a ``bool``, an ``int``, a ``float`` or
  a ``decimal.Decimal``: ``LiteralExpression("05").value`` is ``5`` and
  ``LiteralExpression("1.50").value`` is ``Decimal("1.5")``;
- a conjunction or disjunction is one n-ary :class:`LogicalExpression`,
  whose :class:`LogicalOperation` is ``AND`` or ``OR``;
- built-in function names are reserved: a call of ``max`` always calls the
  built-in ``max``;
- ``str``, ``repr`` and :func:`~fhy_core.symbolic.expression.pformat_expression`
  print the core's text (``true``, ``NaN``, ``(a && b && c)``).

``BinaryOperation.MODULO`` keeps its name and means the remainder of floor
division; its value is the core's name, ``"floor_mod"``. Payloads keep the
``__type__``/``__data__`` envelope.
"""

__all__ = [
    "BINARY_OPERATION_SYMBOLS",
    "BINARY_SYMBOL_OPERATIONS",
    "LOGICAL_OPERATION_SYMBOLS",
    "UNARY_OPERATION_SYMBOLS",
    "UNARY_SYMBOL_OPERATIONS",
    "BinaryExpression",
    "BinaryOperation",
    "CallExpression",
    "Expression",
    "IdentifierExpression",
    "LiteralExpression",
    "LiteralType",
    "LogicalExpression",
    "LogicalOperation",
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
    "validate_predicate",
]

import math
import re
from collections.abc import Mapping
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal
from enum import StrEnum
from typing import TypeAlias

from immutabledict import immutabledict

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.serialization import WrappedFamilySerializable, register_serializable
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import AlphaEquivalenceMixin
from fhy_core.traits import FrozenMixin, RewritableMixin, VisitableMixin
from fhy_core.utils import invert_frozen_dict

LiteralType: TypeAlias = str | float | int | bool | Decimal
"""A value :class:`LiteralExpression` accepts.

A literal's ``value`` is one of the normalized forms: a ``bool``, an
``int``, a ``float`` or a ``decimal.Decimal``; a numeric ``str`` is
normalized into an ``int`` or a ``Decimal``.
"""


def _make_bare_bool_coercion_error(position: str, value: bool) -> ValueError:
    """Return the error refusing to coerce a bare Python ``bool``.

    ``Expression.__eq__`` compares two expressions structurally and
    returns a Python ``bool``, so ``expr == k`` is a ``bool`` rather
    than an IR equality node. Lifting that ``bool`` would plant a
    constant where the caller meant a comparison, so no builder coerces
    one; a Boolean constant is written ``LiteralExpression(True)`` or
    ``LiteralExpression(False)``.

    Args:
        position: Where the ``bool`` was supplied; leads the message.
        value: The refused ``bool``.

    Returns:
        ``ValueError`` naming the position and both intended spellings.

    """
    return ValueError(
        f"{position} is a bare Python bool ({value!r}), not an expression. "
        "This is almost always the accidental result of `expr == k`, which "
        "compares two expressions structurally and returns a Python bool "
        "rather than building an equality; use `.equals()` or "
        "`.not_equals()` to build an equality, or write "
        f"`LiteralExpression({value!r})` for a Boolean constant."
    )


class UnaryOperation(StrEnum):
    """Unary operation.

    Each member's value is the Rust operation's name (``NEGATE`` ->
    ``"negate"``), which is also its serialized form.
    """

    NEGATE = "negate"
    POSITIVE = "positive"
    LOGICAL_NOT = "logical_not"


class BinaryOperation(StrEnum):
    """Binary operation.

    Each member's value is the Rust operation's name (``ADD`` ->
    ``"add"``), which is also its serialized form. ``MODULO`` is the
    remainder of floor division, the core's ``floor_mod``, whose sign
    follows the divisor as Python's ``%`` does. Conjunction and
    disjunction are not binary operations: see :class:`LogicalOperation`.
    """

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    FLOOR_DIVIDE = "floor_divide"
    MODULO = "floor_mod"
    POWER = "power"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"


class LogicalOperation(StrEnum):
    """Connective of a :class:`LogicalExpression`.

    Each member's value is the Rust operation's name (``AND`` ->
    ``"and"``), which is also its serialized form.
    """

    AND = "and"
    OR = "or"


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
BINARY_OPERATION_SYMBOLS: immutabledict[BinaryOperation, str] = immutabledict(
    {
        BinaryOperation.ADD: "+",
        BinaryOperation.SUBTRACT: "-",
        BinaryOperation.MULTIPLY: "*",
        BinaryOperation.DIVIDE: "/",
        BinaryOperation.FLOOR_DIVIDE: "//",
        BinaryOperation.MODULO: "%",
        BinaryOperation.POWER: "**",
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
LOGICAL_OPERATION_SYMBOLS: immutabledict[LogicalOperation, str] = immutabledict(
    {LogicalOperation.AND: "&&", LogicalOperation.OR: "||"}
)


class Expression(
    _rs.Expression,
    WrappedFamilySerializable,
    AlphaEquivalenceMixin,
    RewritableMixin["Expression"],
):
    """Abstract base class for expressions.

    Backed by the Rust implementation: ``fhy_core._rs.Expression`` holds
    the Rust expression and implements equality, hashing, the
    operators, ``str``, ``repr``, free identifiers, substitution and
    structural and alpha equivalence; each node's ``_rs`` class
    implements its fields, its children and its data payload. The
    classes mix in the stateless Python protocols, including the
    ``WrappedFamilySerializable`` envelope, and are registered as
    virtual subclasses of ``FrozenMixin`` and ``VisitableMixin``, whose
    members ``_rs.Expression`` implements: inheriting the mixin would
    make ``typing``'s protocol metaclass the classes' metaclass, whose
    ``isinstance`` misses run Python code. Expressions are immutable,
    and pickle as a call of their class with their fields.

    ``==`` and ``hash`` are structural, so structurally equal
    expressions are equal dict keys and set members. Comparison
    operators build nodes: ``<``, ``<=``, ``>`` and ``>=`` build
    comparisons, and :meth:`equals` and :meth:`not_equals` build an
    equality and an inequality. No operator or builder coerces a bare
    Python ``bool`` operand, since ``expr == k`` is one.

    An expression has no truth value: ``bool(expression)`` raises
    ``TypeError``. Python evaluates a chained comparison such as
    ``0 <= x <= 5`` as ``(0 <= x) and (x <= 5)``, and ``and`` and ``or``
    choose an operand by truth, so a truthy expression would silently
    drop a conjunct. Build connectives with :func:`logical_and`,
    :func:`logical_or` and :func:`logical_not`.

    Expressions are :class:`~fhy_core.term.Term` instances: they bind
    no identifiers, so every referenced identifier is free and
    substitution is capture-free. Alpha equivalence under an
    :class:`~fhy_core.term.AlphaRenaming` follows its binder frames,
    as when a binder term compares its body.
    """

    @staticmethod
    def piecewise(
        *cases: tuple[
            "Expression | Identifier | LiteralType",
            "Expression | Identifier | LiteralType",
        ],
        otherwise: "Expression | Identifier | LiteralType",
    ) -> "PiecewiseExpression":
        """Build a ``PiecewiseExpression``; delegates to :func:`piecewise`.

        See :func:`piecewise` for the operand coercion rules and the
        conditions under which construction raises.
        """
        return piecewise(*cases, otherwise=otherwise)

    @staticmethod
    def call(
        function_name: str,
        *arguments: "Expression | Identifier | LiteralType",
    ) -> "CallExpression":
        """Build a ``CallExpression``; delegates to :func:`call`.

        See :func:`call` for the operand coercion rules.
        """
        return call(function_name, *arguments)


FrozenMixin.register(Expression)
VisitableMixin.register(Expression)
Expression._register_public_class()


@register_serializable(type_id="unary_expression")
class UnaryExpression(_rs.UnaryExpression, Expression):
    """Unary operation applied to one operand.

    ``operation`` must be a :class:`UnaryOperation` and ``operand`` an
    :class:`Expression`; a wrong type raises ``TypeError``, an unknown
    operation ``ValueError``.

    Attributes:
        operation: The operation.
        operand: The operand.

    """

    __match_args__ = ("operation", "operand")


UnaryExpression._register_public_class()


@register_serializable(type_id="binary_expression")
class BinaryExpression(_rs.BinaryExpression, Expression):
    """Binary operation applied to a left and a right operand.

    ``operation`` must be a :class:`BinaryOperation` and both operands
    :class:`Expression` instances; a wrong type raises ``TypeError``, an
    unknown operation ``ValueError``.

    Attributes:
        operation: The operation.
        left: The left operand.
        right: The right operand.

    """

    __match_args__ = ("operation", "left", "right")


BinaryExpression._register_public_class()


@register_serializable(type_id="logical_expression")
class LogicalExpression(_rs.LogicalExpression, Expression):
    """Conjunction or disjunction of two or more operands, in order.

    ``operation`` must be a :class:`LogicalOperation` and ``operands``
    an iterable of at least two :class:`Expression` instances, stored as
    a tuple. A nested logical node is kept as an operand, never spliced
    into its parent. Fewer than two operands, or an unknown operation,
    raise ``ValueError``, and a wrong type ``TypeError``.

    Attributes:
        operation: The connective.
        operands: The operands, in order.

    """

    __match_args__ = ("operation", "operands")


LogicalExpression._register_public_class()


@register_serializable(type_id="identifier_expression")
class IdentifierExpression(_rs.IdentifierExpression, Expression):
    """Reference to an identifier.

    ``identifier`` must be an :class:`~fhy_core.identifier.Identifier`,
    and raises ``TypeError`` otherwise.

    Attributes:
        identifier: The identifier referred to.

    """

    __match_args__ = ("identifier",)


IdentifierExpression._register_public_class()


@register_serializable(type_id="literal_expression")
class LiteralExpression(_rs.LiteralExpression, Expression):
    r"""Constant, normalized.

    ``value`` may be:

    - a ``bool``, held as the Boolean;
    - an ``int``, or a subclass of ``int`` other than ``bool`` such as
      an ``IntEnum`` member, held as the exact integer;
    - a ``float``, or a subclass such as NumPy's ``float64``, held as
      the exact float, NaN and the infinities included;
    - a finite, non-negative ``decimal.Decimal``, held as the exact
      decimal (a negative decimal is the negation of a literal);
    - a ``str`` of ASCII digits, the integer it spells, or of ASCII
      digits with one decimal point (``"1.5"``, ``"1."``, ``".5"``), the
      exact decimal it spells.

    No spelling is kept: ``value`` is the normalized ``bool``, ``int``,
    ``float`` or ``decimal.Decimal``, so ``LiteralExpression("05")``
    equals ``LiteralExpression(5)`` and ``LiteralExpression("1.50")``
    has the value ``Decimal("1.5")``. Literals of different kinds are
    unequal: ``1``, ``1.0``, ``Decimal("1")`` and ``True`` are pairwise
    unequal. Every NaN equals every NaN, and ``-0.0`` equals ``0.0``.

    Any other ``str`` raises ``ValueError``, and any other type
    ``TypeError``, including NumPy's ``int64``, which does not subclass
    ``int``.

    Attributes:
        value: The normalized value.

    """

    __match_args__ = ("value",)


LiteralExpression._register_public_class()


@register_serializable(type_id="piecewise_expression")
class PiecewiseExpression(_rs.PiecewiseExpression, Expression):
    """First-match-wins choice among ordered cases, with a fallback.

    The expression denotes the value of the first case whose condition
    holds, or ``otherwise`` if none does. The cases are given as two
    parallel iterables, stored as tuples; :meth:`get_cases` pairs them.
    There must be at least one case, the two iterables must have equal
    lengths, and a literal condition must be a Boolean, each raising
    ``ValueError`` otherwise; an item that is not an
    :class:`Expression` raises ``TypeError``.

    Attributes:
        conditions: The case conditions, in evaluation order.
        values: The case values, paired with ``conditions``.
        otherwise: The value when no condition holds.

    """

    __match_args__ = ("conditions", "values", "otherwise")


PiecewiseExpression._register_public_class()


@register_serializable(type_id="call_expression")
class CallExpression(_rs.CallExpression, Expression):
    """Function applied to argument expressions.

    ``function_name`` is the callee's name: a built-in function's name
    calls the built-in (the names are reserved), and any other
    non-empty name a user function. ``arguments`` is an iterable of
    :class:`Expression` instances, stored as a tuple. Arity is not
    checked. An empty name raises ``ValueError``, and a wrong type
    ``TypeError``.

    Attributes:
        function_name: The callee's name.
        arguments: The arguments, in order.

    """

    __match_args__ = ("function_name", "arguments")


CallExpression._register_public_class()


def make_binary_expression(
    operation: BinaryOperation,
    left: "Expression | Identifier | LiteralType",
    right: "Expression | Identifier | LiteralType",
) -> BinaryExpression:
    """Build a ``BinaryExpression`` from two coercible operands.

    Each operand may be an ``Expression`` (used as-is), an
    ``Identifier`` (wrapped in ``IdentifierExpression``), or a value of
    ``LiteralType`` other than ``bool`` (wrapped in
    ``LiteralExpression``); the same coercion rules as the operator
    dunders apply.

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
    operation: UnaryOperation,
    operand: "Expression | Identifier | LiteralType",
) -> UnaryExpression:
    """Build a ``UnaryExpression`` from one coercible operand.

    The operand is coerced as the operator dunders coerce theirs.

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


def logical_not(
    expression: "Expression | Identifier | LiteralType",
) -> UnaryExpression:
    """Wrap ``expression`` in a ``LOGICAL_NOT`` unary expression.

    Args:
        expression: Operand to negate, coerced as the operator dunders
            coerce theirs.

    Returns:
        ``LOGICAL_NOT`` unary expression over the coerced operand.

    Raises:
        ValueError: If the operand is a bare Python ``bool`` or has an
            unsupported type.

    """
    return make_unary_expression(UnaryOperation.LOGICAL_NOT, expression)


def _build_logical_expression(
    operation: LogicalOperation,
    builder_name: str,
    expressions: tuple["Expression | Identifier | LiteralType", ...],
) -> LogicalExpression:
    if len(expressions) < 2:  # noqa: PLR2004
        raise ValueError(
            f"{builder_name} requires at least two expressions, but got "
            f"{len(expressions)}."
        )
    return LogicalExpression(
        operation,
        tuple(
            Expression._get_expression_from_other(expression)
            for expression in expressions
        ),
    )


def logical_and(
    *expressions: "Expression | Identifier | LiteralType",
) -> LogicalExpression:
    """Build one ``AND`` ``LogicalExpression`` over two or more operands.

    The operands are kept in order, each coerced as the operator
    dunders coerce theirs; a nested conjunction stays one operand.

    Args:
        expressions: Operands to AND together. Must be at least two.

    Returns:
        The conjunction of the operands.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand is a bare Python ``bool`` or has an unsupported type.

    """
    return _build_logical_expression(LogicalOperation.AND, "logical_and", expressions)


def logical_or(
    *expressions: "Expression | Identifier | LiteralType",
) -> LogicalExpression:
    """Build one ``OR`` ``LogicalExpression`` over two or more operands.

    The operands are kept in order, each coerced as the operator
    dunders coerce theirs; a nested disjunction stays one operand.

    Args:
        expressions: Operands to OR together. Must be at least two.

    Returns:
        The disjunction of the operands.

    Raises:
        ValueError: If fewer than two operands are supplied, or if an
            operand is a bare Python ``bool`` or has an unsupported type.

    """
    return _build_logical_expression(LogicalOperation.OR, "logical_or", expressions)


def piecewise(
    *cases: tuple[
        "Expression | Identifier | LiteralType",
        "Expression | Identifier | LiteralType",
    ],
    otherwise: "Expression | Identifier | LiteralType",
) -> PiecewiseExpression:
    """Build a ``PiecewiseExpression`` from ``(condition, value)`` pairs.

    Each element of each pair, and ``otherwise``, is coerced as the
    operator dunders coerce their operands.

    Args:
        cases: One or more ``(condition, value)`` pairs, in evaluation
            order.
        otherwise: Result when no case's condition holds.

    Returns:
        A ``PiecewiseExpression`` over the coerced cases and
        ``otherwise``.

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
                f"piecewise case {index} must be a 2-tuple of (condition, "
                f"value), got {case!r}."
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
) -> CallExpression:
    """Build a ``CallExpression`` from a name and positional arguments.

    Each argument is coerced as the operator dunders coerce their
    operands.

    Args:
        function_name: The callee's name; a built-in function's name
            calls the built-in.
        arguments: Positional argument operands.

    Returns:
        A ``CallExpression`` over the coerced arguments.

    Raises:
        ValueError: If an argument is a bare Python ``bool`` or has an
            unsupported type, or the name is empty.

    """
    return CallExpression(
        function_name,
        tuple(
            Expression._get_expression_from_other(argument) for argument in arguments
        ),
    )


def validate_logical_operands(
    expression: Expression,
    environment: Mapping[Identifier, Expression] | None = None,
    *,
    symbol_types: Mapping[Identifier, SymbolType] | None = None,
) -> None:
    """Raise unless every Boolean position in ``expression`` holds a Boolean.

    A Boolean position is an operand of a ``LogicalExpression`` or of a
    ``LOGICAL_NOT``, or a piecewise case condition; a piecewise in a
    Boolean position also puts its case values and ``otherwise`` in
    one. The whole tree is screened, by the Rust core's screen.

    Only a provably numeric operand is refused: a non-Boolean literal,
    an arithmetic node, a piecewise whose branches all are numeric, a
    call whose result sort is not ``BOOL``, a native constant's
    canonical identifier of a non-``BOOL`` sort, and an identifier
    ``symbol_types`` declares ``INT`` or ``REAL``. A built-in call's
    result sort is the built-in's; a user function's comes from the
    registry, and an unregistered one passes.

    Args:
        expression: Expression about to be lowered to a symbolic
            backend, or to have ``environment`` substituted into it.
        environment: Expressions bound to identifiers. A bound value is
            screened in the identifier's place, without applying
            another binding to it; a binding for a native constant's
            canonical identifier is not consulted.
        symbol_types: Sorts declared for the identifiers left free once
            ``environment`` is substituted.

    Raises:
        NonBooleanLogicalOperandError: If a Boolean position holds an
            operand that provably denotes a number. The message names
            the position, the operand and the node taking it.

    """
    _rs.validate_logical_operands(expression, environment, symbol_types=symbol_types)


def validate_predicate(
    expression: Expression,
    environment: Mapping[Identifier, Expression] | None = None,
    *,
    symbol_types: Mapping[Identifier, SymbolType] | None = None,
) -> None:
    """Raise unless ``expression`` can be used as a predicate.

    The root is itself a Boolean position: it must not provably denote
    a number, and every Boolean position within it is screened as
    :func:`validate_logical_operands` screens them, with the root's own
    case values and ``otherwise`` included when it is a piecewise.

    Args:
        expression: Expression that is itself supposed to denote a
            Boolean.
        environment: As accepted by :func:`validate_logical_operands`.
        symbol_types: As accepted by :func:`validate_logical_operands`.

    Raises:
        NonBooleanLogicalOperandError: If the root, or an operand in a
            Boolean position, provably denotes a number.

    """
    _rs.validate_predicate(expression, environment, symbol_types=symbol_types)


_LiteralBucket: TypeAlias = tuple[str, "bool | int | float | str | Decimal"]

_INTEGER_LITERAL_PATTERN = re.compile(r"[0-9]+")
_INTEGER_LITERAL_BUCKET = "int"
_BOOLEAN_LITERAL_BUCKET = "bool"
_FLOAT_BINARY_LITERAL_BUCKET = "float-binary"
_FLOAT_DECIMAL_LITERAL_BUCKET = "float-decimal"
_CANONICAL_NAN_FORM = "nan"


def _normalize_decimal_exactly(value: Decimal) -> Decimal:
    """Return ``value`` with its trailing coefficient zeros stripped, unrounded."""
    precision = max(len(value.as_tuple().digits), 1)
    return value.normalize(Context(prec=precision, Emax=MAX_EMAX, Emin=MIN_EMIN))


# One early return per literal kind reads clearest here.
def _classify_literal_value(value: LiteralType) -> _LiteralBucket:  # noqa: PLR0911
    """Return the (bucket, canonical-form) pair of a literal value.

    The four buckets are the core's four literal kinds: a Boolean, an
    integer (an ``int``, or an integer-grammar ``str``), a binary float
    (every NaN shares one form, and ``-0.0`` is ``0.0``), and an exact
    decimal (a ``Decimal``, or a decimal-grammar ``str``, stripped of
    trailing zeros without rounding).
    """
    if isinstance(value, bool):
        return (_BOOLEAN_LITERAL_BUCKET, value)
    elif isinstance(value, int):
        return (_INTEGER_LITERAL_BUCKET, int(value))
    elif isinstance(value, float):
        if math.isnan(value):
            return (_FLOAT_BINARY_LITERAL_BUCKET, _CANONICAL_NAN_FORM)
        return (_FLOAT_BINARY_LITERAL_BUCKET, value + 0.0)
    elif isinstance(value, Decimal):
        return (_FLOAT_DECIMAL_LITERAL_BUCKET, _normalize_decimal_exactly(value))
    elif _INTEGER_LITERAL_PATTERN.fullmatch(value):
        return (_INTEGER_LITERAL_BUCKET, int(value))
    else:
        return (
            _FLOAT_DECIMAL_LITERAL_BUCKET,
            _normalize_decimal_exactly(Decimal(value)),
        )


def build_literal_equivalence_key(value: LiteralType) -> str:
    """Return text two literal values share exactly when their literals are equal.

    Renders the kind and canonical form of the literal the value
    builds, so the key agrees with ``==`` of the literals in both
    directions: ``5``, ``"5"`` and ``"05"`` share a key, as do
    ``"1.5"``, ``"1.50"`` and ``Decimal("1.5")``, ``0.0`` and ``-0.0``,
    and every NaN, while a ``bool`` keys apart from every integer and
    a decimal apart from the binary ``float`` with the same digits.

    Args:
        value: A value :class:`LiteralExpression` accepts, or a
            literal's ``value``.

    Returns:
        Bucket-prefixed text, such as ``"int:5"`` or
        ``"float-binary:nan"``.

    """
    bucket, canonical = _classify_literal_value(value)
    return f"{bucket}:{canonical}"


def is_integer_valued_literal(value: LiteralType) -> bool:
    """Return whether a literal value is an integer literal's.

    True for an ``int`` and an integer-grammar ``str``, whose literals
    are the core's integers; False for a ``bool``, a ``float``, a
    ``Decimal`` and a decimal-grammar ``str``, even when the value has
    no fractional part. Whenever the answer is True, ``int(value)``
    recovers the integer exactly.

    Args:
        value: A value :class:`LiteralExpression` accepts, or a
            literal's ``value``.

    Returns:
        True when the value denotes an integer literal.

    """
    bucket, _ = _classify_literal_value(value)
    return bucket == _INTEGER_LITERAL_BUCKET
