"""Backend-agnostic entry point for symbolic queries over expressions.

Simplification is answered by the SymPy bridge; satisfiability,
implication, and universal validity by the Z3 bridge. Backend
selection is explicit: asking a backend for a query kind it cannot
answer raises ``SolverCapabilityError``. Each query kind currently has
exactly one capable backend.

The two bridges agree on what a literal denotes. Each of
``LiteralExpression``'s numeric forms carries its own precision contract
-- a Python ``float`` is IEEE-754 binary, a float-grammar ``str`` is
exact decimal -- and both bridges lower each form to that exact value, so
no ground comparison is decided one way by ``simplify_expression`` and the
other way by the Z3 questions below.

Known divergences: the Z3 and SymPy bridges disagree with each other and
with the type checker on integer division and floor-division/modulo
Euclidean semantics. The two bridges also treat native constants
differently: the SymPy bridge resolves ``pi``, ``e``, ``inf``, and
``nan`` to their values, while the Z3 bridge lowers a constant's
canonical identifier as an ordinary variable of whatever sort
``symbol_types`` gives it. This module routes to each bridge unchanged;
it does not reconcile that math. The hazard screens below refuse the
division-like shapes rather than let a bridge decide one of them.

The Z3-question entry points (``check_expression_satisfiability``,
``does_expression_imply``, ``holds_for_all_free_assignments``, and
their strict ``assert_*`` companions) additionally screen every
expression argument before it is lowered, refusing three node shapes
the Z3 bridge cannot lower soundly: a Boolean operand reaching a
numeric context, where the Z3 Python bindings silently rewrite it to
``If(b, 1, 0)`` and collapse this package's type-strict Boolean/numeric
distinction; a ``DIVIDE``/``FLOOR_DIVIDE``/``MODULO`` node whose divisor
is not provably safe for its operation -- a finite nonzero literal for
``DIVIDE``, since the satisfiability encoding around a possibly-zero
divisor is unsound, or a finite strictly positive literal for
``FLOOR_DIVIDE``/``MODULO``, since Z3 lowers both to Euclidean
division, which disagrees with this package's floor semantics for a
zero, negative, or non-finite (``nan``/``inf``) divisor; and an
``EQUAL``/``NOT_EQUAL`` comparison mixing an INT-sorted operand with a
float-valued literal, since Z3's ``ToReal`` rationalization of the
INT-sorted side collapses this package's type-strict int/float
distinction. A refused expression is never lowered: the lenient entry
points report the same ``None`` they use for a Z3 ``unknown`` result,
and the strict ``assert_*`` companions raise the same
``UndecidableError`` they raise for one, so the screen protects every
caller of this seam the same way regardless of entry point.

An ill-typed expression is a separate matter from an undecidable one and
is reported separately: a provably numeric operand of a logical
connective, or a provably numeric piecewise case condition, raises
``NonBooleanLogicalOperandError`` from every entry point here,
simplification included, since no backend and no timeout gives that
expression a meaning to report.
"""

__all__ = [
    "SolverBackend",
    "SolverCapabilityError",
    "SolverQueryKind",
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "check_expression_satisfiability",
    "does_expression_imply",
    "get_backend_capabilities",
    "holds_for_all_free_assignments",
    "simplify_expression",
    "validate_timeout_milliseconds",
]

import math
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
from enum import Enum, auto

from immutabledict import immutabledict

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.utils import StrEnum, format_comma_separated_list

from .expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    UndecidableError,
    is_integer_valued_literal,
)
from .expression.passes.sympy import simplify_expression as _sympy_simplify_expression
from .expression.passes.z3 import (
    assert_expression_implies as _z3_assert_expression_implies,
)
from .expression.passes.z3 import (
    assert_holds_for_all_free_assignments as _z3_assert_holds_for_all_free_assignments,
)
from .expression.passes.z3 import does_expression_imply as _z3_does_expression_imply
from .expression.passes.z3 import (
    holds_for_all_free_assignments as _z3_holds_for_all_free_assignments,
)
from .symbol_type import SymbolType

_LOGGER = get_logger(__name__)


class SolverBackend(StrEnum):
    """Symbolic engine selectable for a solver query."""

    SYMPY = "sympy"
    Z3 = "z3"


class SolverQueryKind(StrEnum):
    """Kind of question a solver backend can be asked."""

    SIMPLIFICATION = "simplification"
    SATISFIABILITY = "satisfiability"
    IMPLICATION = "implication"
    UNIVERSAL_VALIDITY = "universal_validity"


@register_error
class SolverCapabilityError(ValueError):
    """Raised when the requested backend cannot answer the requested query kind."""


_BACKEND_CAPABILITIES: immutabledict[SolverBackend, frozenset[SolverQueryKind]] = (
    immutabledict(
        {
            SolverBackend.SYMPY: frozenset({SolverQueryKind.SIMPLIFICATION}),
            SolverBackend.Z3: frozenset(
                {
                    SolverQueryKind.SATISFIABILITY,
                    SolverQueryKind.IMPLICATION,
                    SolverQueryKind.UNIVERSAL_VALIDITY,
                }
            ),
        }
    )
)


def get_backend_capabilities(backend: SolverBackend) -> frozenset[SolverQueryKind]:
    """Return the query kinds the given backend can answer.

    Args:
        backend: Backend to look up.

    Returns:
        The backend's supported query kinds, or an empty set for a
        backend with no capability table entry -- so an unrecognized
        backend is reported by the caller's capability check as a
        ``SolverCapabilityError`` rather than escaping as a ``KeyError``.

    """
    return _BACKEND_CAPABILITIES.get(backend, frozenset())


def _validate_backend_capability(
    backend: SolverBackend, query_kind: SolverQueryKind
) -> None:
    if query_kind not in get_backend_capabilities(backend):
        raise SolverCapabilityError(
            f"Backend {backend!r} cannot answer {query_kind!r} queries; "
            f"it supports {sorted(get_backend_capabilities(backend))}."
        )


def validate_timeout_milliseconds(timeout_milliseconds: int | None) -> None:
    """Raise unless ``timeout_milliseconds`` is ``None`` or a positive integer.

    Public so a caller that decides an outcome without reaching the solver
    -- and therefore never passes the value on -- can still hold up the
    same precondition the solver entry points enforce.

    Args:
        timeout_milliseconds: Candidate bound, in milliseconds.

    Raises:
        ValueError: If the value is not ``None`` and not positive.

    """
    if timeout_milliseconds is not None and timeout_milliseconds <= 0:
        raise ValueError(
            "timeout_milliseconds must be None or a positive integer, but got "
            f"{timeout_milliseconds!r}."
        )


def simplify_expression(
    expression: Expression,
    environment: dict[Identifier, Expression] | None = None,
    *,
    backend: SolverBackend = SolverBackend.SYMPY,
) -> Expression:
    """Simplify an expression, optionally substituting an environment first.

    With an environment binding every free identifier, simplification is
    evaluation: the result is a ``LiteralExpression`` whenever the backend
    can decide the value. Delegates to the SymPy bridge, which performs
    all conversion and simplification math.

    Args:
        expression: Expression to simplify.
        environment: Environment to substitute into the expression before
            simplifying. Defaults to ``None``.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.SYMPY``, the only SIMPLIFICATION-capable
            backend today.

    Returns:
        Simplified expression.

    Raises:
        SolverCapabilityError: If ``backend`` is not SIMPLIFICATION-capable
            (currently: any backend other than SYMPY).
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, provably denotes a number, counting an operand
            ``environment`` binds to one. SymPy's ``&``/``|`` are bitwise on
            ``sympy.Integer``, so the shape is refused rather than folded to
            a numerically wrong literal.
        PassExecutionError: If the SymPy bridge's lowering or lifting pass
            fails internally, for example when simplification yields a
            ``sympy.Piecewise`` whose final branch condition is not
            ``sympy.true``, or when it yields ``sympy.zoo``, which a
            quotient by zero folds to. The pass infrastructure wraps the
            originating error (e.g. ``PartialPiecewiseError``,
            ``ComplexInfinityLiftError``) as ``__cause__`` rather than
            letting it propagate directly.

    """
    _validate_backend_capability(backend, SolverQueryKind.SIMPLIFICATION)
    # INVARIANT: _BACKEND_CAPABILITIES grants SIMPLIFICATION to exactly one
    # backend (SYMPY), so delegating straight to the SymPy bridge is valid
    # without dispatching on `backend`. Adding a second SIMPLIFICATION-capable
    # backend requires real dispatch here.
    return _sympy_simplify_expression(expression, environment)


# =============================================================================
# Lowering hazard screens
#
# The Z3 bridge mis-lowers three expression shapes: it cannot be trusted to
# decide an outcome for them, so every Z3-question entry point below screens
# its expression argument(s) for these shapes before lowering, rather than
# letting the bridge decide something it cannot decide soundly.
# =============================================================================


class _LoweredSort(Enum):
    """Z3 sort an expression node lowers to, as far as it can be determined.

    ``UNDETERMINED`` covers the nodes whose lowered sort this module
    cannot read off the tree: an identifier with no ``symbol_types``
    entry, a call (which the Z3 bridge refuses outright), an unrecognized
    ``Expression`` subclass, and a piecewise whose branches disagree.
    """

    BOOLEAN = auto()
    NUMERIC = auto()
    UNDETERMINED = auto()


_NUMERIC_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
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
"""Binary operations the Z3 bridge lowers to arithmetic on a numeric sort."""

_COMPARISON_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.EQUAL,
        BinaryOperation.NOT_EQUAL,
        BinaryOperation.LESS,
        BinaryOperation.LESS_EQUAL,
        BinaryOperation.GREATER,
        BinaryOperation.GREATER_EQUAL,
    }
)
"""Binary operations the Z3 bridge lowers to a comparison of two operands."""

_LOGICAL_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {BinaryOperation.LOGICAL_AND, BinaryOperation.LOGICAL_OR}
)
"""Binary operations the Z3 bridge lowers to ``z3.And``/``z3.Or``."""


# One early return per node kind reads clearest here; the alternative is a
# lookup table that would have to be threaded through `symbol_types` anyway.
def _classify_lowered_sort(  # noqa: PLR0911
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> _LoweredSort:
    """Return the Z3 sort ``expression`` lowers to, per ``passes.z3``.

    Mirrors ``ExpressionToZ3Converter``: a ``bool``-valued literal becomes
    a ``BoolVal`` and every other literal an ``IntVal``/``RealVal``; an
    identifier takes the sort named by its ``symbol_types`` entry; a
    comparison, a logical operation, and a logical negation are Boolean;
    arithmetic is numeric; a piecewise takes the sort its branches agree
    on.

    Args:
        expression: Node whose lowered sort is wanted.
        symbol_types: Z3 sort for each identifier of the enclosing
            expression.

    Returns:
        The node's ``_LoweredSort``.

    """
    if isinstance(expression, LiteralExpression):
        if type(expression.value) is bool:
            return _LoweredSort.BOOLEAN
        return _LoweredSort.NUMERIC
    elif isinstance(expression, IdentifierExpression):
        symbol_type = symbol_types.get(expression.identifier)
        if symbol_type is SymbolType.BOOL:
            return _LoweredSort.BOOLEAN
        elif symbol_type is None:
            return _LoweredSort.UNDETERMINED
        return _LoweredSort.NUMERIC
    elif isinstance(expression, BinaryExpression):
        if expression.operation in _NUMERIC_BINARY_OPERATIONS:
            return _LoweredSort.NUMERIC
        elif expression.operation in (
            _COMPARISON_BINARY_OPERATIONS | _LOGICAL_BINARY_OPERATIONS
        ):
            return _LoweredSort.BOOLEAN
        return _LoweredSort.UNDETERMINED
    elif isinstance(expression, UnaryExpression):
        if expression.operation is UnaryOperation.LOGICAL_NOT:
            return _LoweredSort.BOOLEAN
        return _LoweredSort.NUMERIC
    elif isinstance(expression, PiecewiseExpression):
        return _join_lowered_sorts(
            _classify_lowered_sort(branch, symbol_types)
            for branch in (*expression.values, expression.otherwise)
        )
    elif isinstance(expression, CallExpression):
        return _LoweredSort.UNDETERMINED
    return _LoweredSort.UNDETERMINED


def _join_lowered_sorts(sorts: Iterator[_LoweredSort]) -> _LoweredSort:
    """Return the sort every input agrees on, or ``UNDETERMINED`` if they differ."""
    distinct = set(sorts)
    if len(distinct) == 1:
        return distinct.pop()
    return _LoweredSort.UNDETERMINED


def _does_node_coerce_a_bool_operand(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether this one node makes Z3 coerce a Boolean operand to an integer.

    The Z3 Python bindings rewrite a Boolean operand of a numeric context
    into ``If(b, 1, 0)``: ``True`` becomes ``1`` and ``False`` becomes
    ``0``. That collapses this package's type-strict Boolean/numeric
    distinction, so a decided outcome read back through such a lowering
    can be provably wrong.

    Three contexts coerce:

    - An arithmetic operation with any Boolean operand.
    - A comparison whose two operands mix a Boolean and a numeric sort. A
      comparison of two Booleans lowers faithfully and is not flagged.
    - A piecewise whose branch values mix a Boolean and a numeric sort,
      since ``z3.If`` forces its two arms to a single sort.

    ``z3.And``/``z3.Or``/``z3.Not`` and unary arithmetic negation do not
    coerce, so a Boolean operand under one of those lowers faithfully and
    is not flagged. The mirror-image mismatch -- a *numeric* operand
    under a logical connective -- is not this screen's business either:
    both bridges refuse that shape before lowering, with
    ``NonBooleanLogicalOperandError``, so it never reaches a lowering
    whose answer could be read back wrong.

    Args:
        expression: Node to screen. Children are not visited.
        symbol_types: Z3 sort for each identifier of the enclosing
            expression.

    Returns:
        True if this node's own lowering coerces a Boolean operand.

    """
    if isinstance(expression, BinaryExpression):
        operand_sorts = {
            _classify_lowered_sort(operand, symbol_types)
            for operand in expression.get_operands()
        }
        if expression.operation in _NUMERIC_BINARY_OPERATIONS:
            return _LoweredSort.BOOLEAN in operand_sorts
        elif expression.operation in _COMPARISON_BINARY_OPERATIONS:
            return operand_sorts >= {_LoweredSort.BOOLEAN, _LoweredSort.NUMERIC}
        return False
    elif isinstance(expression, PiecewiseExpression):
        branch_sorts = {
            _classify_lowered_sort(branch, symbol_types)
            for branch in (*expression.values, expression.otherwise)
        }
        return branch_sorts >= {_LoweredSort.BOOLEAN, _LoweredSort.NUMERIC}
    return False


def _find_bool_sort_hazard(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> Expression | None:
    """Return the first node of ``expression`` that coerces a Boolean operand.

    Screens the tree that is actually handed to the solver, so the check
    covers every way a ``BoolVal`` can reach a numeric context: a
    ``bool`` literal written directly into the expression, and a
    ``bool`` value substituted into it before the check (for example by
    a caller resolving a partial assignment). The screen is per-site: a
    Boolean literal consumed by a logical operation leaves the rest of
    the tree decidable.

    Args:
        expression: Expression about to be lowered to Z3.
        symbol_types: Z3 sort for each free identifier of ``expression``.

    Returns:
        The offending node, or ``None`` when every node lowers faithfully.
        The node is returned rather than a flag so the caller can name the
        site it refused to lower.

    """
    if _does_node_coerce_a_bool_operand(expression, symbol_types):
        return expression
    for child in expression.get_visit_children():
        hazard = _find_bool_sort_hazard(child, symbol_types)
        if hazard is not None:
            return hazard
    return None


def _render_identifier_sorts(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> str:
    """Return each free identifier of ``expression`` paired with its declared sort.

    Args:
        expression: Node whose free identifiers are described.
        symbol_types: Z3 sort for each free identifier.

    Returns:
        A comma-separated ``identifier: SORT`` listing, ordered by
        identifier id; ``"none"`` when the node has no free identifier,
        and ``"unknown"`` in place of a sort ``symbol_types`` does not
        carry.

    """
    identifiers = sorted(
        expression.get_free_identifiers(), key=lambda identifier: identifier.id
    )
    if not identifiers:
        return "none"
    rendered: list[str] = []
    for identifier in identifiers:
        symbol_type = symbol_types.get(identifier)
        sort_name = "unknown" if symbol_type is None else symbol_type.name
        rendered.append(f"{identifier!r}: {sort_name}")
    return format_comma_separated_list(rendered)


_PARTIAL_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {
        BinaryOperation.DIVIDE,
        BinaryOperation.FLOOR_DIVIDE,
        BinaryOperation.MODULO,
        BinaryOperation.POWER,
    }
)
"""Binary operations Z3 lowers to a function that is partial or divergent.

Each is underspecified somewhere in its domain (division and modulo at a
zero divisor, exponentiation at ``0 ** 0`` and at every negative
exponent) or lowers to an operation whose semantics differ from this
package's. The satisfiability encoding does not bind a partial function
outside the domain where it is defined, so a term reaching one of those
points admits a decided answer that no concrete assignment supports.
"""


_EUCLIDEAN_DIVERGENT_BINARY_OPERATIONS: frozenset[BinaryOperation] = frozenset(
    {BinaryOperation.FLOOR_DIVIDE, BinaryOperation.MODULO}
)
"""Binary operations Z3 lowers to Euclidean division.

Euclidean division agrees with this package's floor semantics only
when the divisor is positive, so these two operations need a stricter
divisor screen than plain ``DIVIDE`` (true division has no such
sign-dependent divergence).
"""


def _is_safe_nonzero_divisor(node: Expression) -> bool:
    """Return whether ``node`` is provably a finite nonzero numeric literal.

    A ``bool`` value, a float-grammar string-form literal, and a
    non-finite float (``nan``/``inf``) are not safe divisors: none
    carries the provably-nonzero, finite guarantee the division hazard
    screen requires, even when the float is a constructible
    ``LiteralExpression`` value. An integer-valued literal is safe in
    either of its forms, so the screen answers alike for every member of
    one equivalence class.

    """
    if not isinstance(node, LiteralExpression):
        return False
    value = node.value
    if is_integer_valued_literal(value):
        return int(value) != 0
    if isinstance(value, float):
        return math.isfinite(value) and value != 0
    return False


def _is_safe_positive_divisor(node: Expression) -> bool:
    """Return whether ``node`` is a finite positive numeric literal.

    Required for ``FLOOR_DIVIDE``/``MODULO``: a merely nonzero (but
    possibly negative) literal is not enough for these two operations,
    since Z3's Euclidean lowering disagrees with this package's floor
    semantics whenever the divisor is not positive.

    """
    if not isinstance(node, LiteralExpression):
        return False
    value = node.value
    if is_integer_valued_literal(value):
        return int(value) > 0
    if isinstance(value, float):
        return math.isfinite(value) and value > 0
    return False


def _does_operand_lower_to_real_sort(
    node: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether ``node`` provably lowers to a Z3 REAL sort.

    Answers conservatively: an operand this cannot prove real is treated
    as not real, so a caller screening on "provably real" refuses the
    cases it cannot classify rather than trusting them. Arithmetic is
    real-sorted as soon as one side is, matching the Z3 bridge's
    coercion.

    Args:
        node: Operand about to be lowered to Z3.
        symbol_types: Z3 sort for each free identifier of the expression.

    Returns:
        True when ``node`` is a REAL-typed identifier, a float-valued
        literal, or arithmetic with a provably real operand.

    """
    if isinstance(node, IdentifierExpression):
        return symbol_types.get(node.identifier) is SymbolType.REAL
    if isinstance(node, LiteralExpression):
        return _is_float_valued_literal(node)
    if (
        isinstance(node, BinaryExpression)
        and node.operation in _NUMERIC_BINARY_OPERATIONS
    ):
        return _does_operand_lower_to_real_sort(
            node.left, symbol_types
        ) or _does_operand_lower_to_real_sort(node.right, symbol_types)
    if isinstance(node, UnaryExpression) and node.operation is UnaryOperation.NEGATE:
        return _does_operand_lower_to_real_sort(node.operand, symbol_types)
    return False


def _is_safe_true_division(
    expression: BinaryExpression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether a ``DIVIDE`` node lowers to real division Z3 agrees on.

    Two conditions, both required. The divisor must be a finite nonzero
    literal, or Z3's division at zero is underspecified. One operand must
    also lower to a REAL sort: Z3 divides two INT-sorted operands with
    truncating integer division, so ``7 / 2`` is ``3`` there and ``3.5``
    here, and the seam would otherwise decide the negation of what
    ``simplify_expression`` decides for the same node.

    """
    if not _is_safe_nonzero_divisor(expression.right):
        return False
    return _does_operand_lower_to_real_sort(
        expression.left, symbol_types
    ) or _does_operand_lower_to_real_sort(expression.right, symbol_types)


def _is_safe_exponent(node: Expression) -> bool:
    """Return whether ``node`` is an exponent Z3 raises to totally.

    Requires an integer-valued literal of at least one, in either of the
    forms the IR treats as that integer. A negative exponent makes
    exponentiation a division, underspecified at a zero base and
    rational-valued on integers; a zero exponent leaves ``0 ** 0``
    underspecified; a non-integer exponent lowers to a real power that is
    undefined for a negative base; and a symbolic exponent cannot be
    classified at all.

    """
    if not isinstance(node, LiteralExpression):
        return False
    value = node.value
    return is_integer_valued_literal(value) and int(value) >= 1


def _does_node_use_an_unsafe_partial_operation(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether this one node applies a partial operation off its safe domain."""
    if not (
        isinstance(expression, BinaryExpression)
        and expression.operation in _PARTIAL_BINARY_OPERATIONS
    ):
        return False
    if expression.operation is BinaryOperation.POWER:
        return not _is_safe_exponent(expression.right)
    if expression.operation in _EUCLIDEAN_DIVERGENT_BINARY_OPERATIONS:
        return not _is_safe_positive_divisor(expression.right)
    return not _is_safe_true_division(expression, symbol_types)


def _find_partial_operation_hazard(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> Expression | None:
    """Return the first node of ``expression`` applying an unsafe partial operation.

    Screens the whole tree, mirroring ``_find_bool_sort_hazard``. A
    ``DIVIDE`` node is refused unless its divisor is a finite nonzero
    literal and one operand lowers to a REAL sort, since the encoding
    around a zero divisor is unsound and Z3 divides two INT-sorted
    operands with truncating integer division. A
    ``FLOOR_DIVIDE``/``MODULO`` node is refused unless its divisor is a
    finite *positive* literal, since Z3 lowers both to Euclidean
    division, which disagrees with this package's floor semantics for a
    zero, negative, or non-finite divisor. A ``POWER`` node is refused
    unless its exponent is a literal integer of at least one, since Z3's
    exponentiation is underspecified at ``0 ** 0`` and at every negative
    exponent.

    Args:
        expression: Expression about to be lowered to Z3.
        symbol_types: Z3 sort for each free identifier of ``expression``.

    Returns:
        The offending node, or ``None`` when every partial operation in
        the tree stays inside the domain its lowering is sound on.

    """
    if _does_node_use_an_unsafe_partial_operation(expression, symbol_types):
        return expression
    for child in expression.get_visit_children():
        hazard = _find_partial_operation_hazard(child, symbol_types)
        if hazard is not None:
            return hazard
    return None


def _is_float_valued_literal(node: Expression) -> bool:
    """Return whether ``node`` is a ``LiteralExpression`` in a float bucket.

    Covers a Python ``float`` value and a float-grammar string-form
    literal (e.g. ``"1.5"``). A literal is bucketed as Boolean,
    integer-valued, or float-valued, so the answer is what is left once
    the first two are ruled out through the IR's own integer predicate:
    an integer-grammar string is not float-valued, matching the INT sort
    the Z3 bridge lowers it to.

    """
    if not isinstance(node, LiteralExpression):
        return False
    value = node.value
    if isinstance(value, bool):
        return False
    return not is_integer_valued_literal(value)


def _is_int_sorted_operand(
    node: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether ``node`` is an INT-typed identifier or an integer literal."""
    if isinstance(node, IdentifierExpression):
        return symbol_types.get(node.identifier) is SymbolType.INT
    return _is_int_sorted_literal(node)


def _is_int_sorted_literal(node: Expression) -> bool:
    """Return whether ``node`` is a literal the Z3 bridge lowers to the INT sort."""
    return isinstance(node, LiteralExpression) and is_integer_valued_literal(node.value)


def _does_node_mix_int_and_float_equality(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> bool:
    """Return whether this node's ``EQUAL``/``NOT_EQUAL`` mixes INT and float sorts.

    Screens both directions of the mismatch, since Z3 rationalizes
    whichever side is INT-sorted and then compares numerically, in either
    arrangement collapsing the type-strict int/float distinction this
    package draws between ``1`` and ``1.0``.

    """
    if not (
        isinstance(expression, BinaryExpression)
        and expression.operation in (BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL)
    ):
        return False
    left, right = expression.left, expression.right
    return (
        (_is_float_valued_literal(left) and _is_int_sorted_operand(right, symbol_types))
        or (
            _is_float_valued_literal(right)
            and _is_int_sorted_operand(left, symbol_types)
        )
        or (
            _is_int_sorted_literal(left)
            and _does_operand_lower_to_real_sort(right, symbol_types)
        )
        or (
            _is_int_sorted_literal(right)
            and _does_operand_lower_to_real_sort(left, symbol_types)
        )
    )


def _find_int_float_equality_hazard(
    expression: Expression, symbol_types: Mapping[Identifier, SymbolType]
) -> Expression | None:
    """Return the first node whose equality mixes an INT and a REAL sort.

    Z3's ``ToReal`` rationalization of the INT-sorted operand collapses
    this package's type-strict int/float distinction, so an ``EQUAL``/
    ``NOT_EQUAL`` node mixing the two is refused in either arrangement:
    a float-valued literal against an INT-sorted operand, and a
    strict-int literal against a REAL-sorted one. Both carry the same
    hazard, and the second is what a type-strict set constraint lowers
    to when an integer member is screened against a real-valued
    parameter. Ordering comparisons (``<``, ``<=``, ``>``, ``>=``) are
    not screened: mixed-sort ordering stays mathematically meaningful.

    Args:
        expression: Expression about to be lowered to Z3.
        symbol_types: Z3 sort for each free identifier of ``expression``.

    Returns:
        The offending node, or ``None`` when no ``EQUAL``/``NOT_EQUAL``
        node mixes the two sorts.

    """
    if _does_node_mix_int_and_float_equality(expression, symbol_types):
        return expression
    for child in expression.get_visit_children():
        hazard = _find_int_float_equality_hazard(child, symbol_types)
        if hazard is not None:
            return hazard
    return None


def _log_bool_coercion_hazard(
    hazard: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    context: str,
) -> None:
    _LOGGER.warning(
        "%s: node %r lowers a Boolean operand into a numeric context, where "
        "the Z3 bindings rewrite it to If(b, 1, 0) and collapse this "
        "package's type-strict semantics; identifier sorts at that node: "
        "%s. The expression is not handed to the solver; bounding "
        "timeout_milliseconds cannot change this outcome.",
        context,
        hazard,
        _render_identifier_sorts(hazard, symbol_types),
    )


def _log_partial_operation_hazard(
    hazard: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    context: str,
) -> None:
    _LOGGER.warning(
        "%s: node %r applies an operation Z3 lowers partially or with "
        "divergent semantics off the domain the seam can trust (DIVIDE "
        "needs a finite nonzero literal divisor and one REAL-sorted "
        "operand, since Z3 divides two INT-sorted operands with "
        "truncating integer division; FLOOR_DIVIDE/MODULO need a finite "
        "strictly positive literal divisor, since Z3 lowers those to "
        "Euclidean division, which disagrees with floor semantics for a "
        "zero, negative, or non-finite divisor; POWER needs a literal "
        "integer exponent of at least one, since exponentiation is "
        "underspecified at 0 ** 0 and at every negative exponent), and "
        "the seam's satisfiability encoding does not bind a partial "
        "function outside its domain; identifier sorts at that node: "
        "%s. The expression is not handed to the solver; bounding "
        "timeout_milliseconds cannot change this outcome.",
        context,
        hazard,
        _render_identifier_sorts(hazard, symbol_types),
    )


def _log_int_float_equality_hazard(
    hazard: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    context: str,
) -> None:
    _LOGGER.warning(
        "%s: node %r compares an INT-sorted operand against a float-valued "
        "literal with EQUAL/NOT_EQUAL, where the Z3 bridge's ToReal "
        "rationalization of the INT-sorted side collapses this package's "
        "type-strict int/float distinction; identifier sorts at that "
        "node: %s. The expression is not handed to the solver; bounding "
        "timeout_milliseconds cannot change this outcome.",
        context,
        hazard,
        _render_identifier_sorts(hazard, symbol_types),
    )


def _find_and_log_hazard(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    context: str,
) -> bool:
    """Screen ``expression`` for a hazard the Z3 bridge cannot lower soundly.

    Checks, in order, the Boolean-coercion hazard, the partial-operation
    hazard (division and exponentiation off the domain their lowering is
    sound on), and the int/float ``EQUAL``/``NOT_EQUAL`` sort-mixing
    hazard; the first one found is logged at ``WARNING`` and
    short-circuits the remaining checks.

    Args:
        expression: Expression about to be lowered to Z3.
        symbol_types: Z3 sort for each free identifier of ``expression``.
        context: Name of the seam entry point the outcome is reported
            under, used to attribute the warning.

    Returns:
        True if a hazard was found (and logged); False if ``expression``
        lowers soundly.

    """
    hazard = _find_bool_sort_hazard(expression, symbol_types)
    if hazard is not None:
        _log_bool_coercion_hazard(hazard, symbol_types, context=context)
        return True
    hazard = _find_partial_operation_hazard(expression, symbol_types)
    if hazard is not None:
        _log_partial_operation_hazard(hazard, symbol_types, context=context)
        return True
    hazard = _find_int_float_equality_hazard(expression, symbol_types)
    if hazard is not None:
        _log_int_float_equality_hazard(hazard, symbol_types, context=context)
        return True
    return False


_HAZARD_SCREEN_REASON: str = "hazard_screen"
"""``UndecidableError.reason`` for a screen-refused expression.

No Z3 solver call happens for a screened expression, so there is no
``reason_unknown()`` text to report; this fixed marker lets a caller
tell a hazard-screen refusal (undecidable in principle -- a larger
``timeout_milliseconds`` will not change the outcome) apart from a
genuine Z3 ``unknown`` result such as ``"timeout"``.
"""


def _validate_symbol_types_cover_free_identifiers(
    free_identifiers: frozenset[Identifier],
    symbol_types: Mapping[Identifier, SymbolType],
) -> None:
    """Raise unless every one of ``free_identifiers`` has a ``symbol_types`` entry.

    Mirrors the check the Z3 bridge's own conversion performs, run ahead
    of the hazard screen above so a missing entry still raises even when
    the same expression is also refused by that screen: without this, an
    expression that is both hazardous and missing a sort would
    short-circuit to a screened ``None``/``UndecidableError`` before the
    bridge ever got a chance to raise.

    Args:
        free_identifiers: Identifiers that must each have a
            ``symbol_types`` entry.
        symbol_types: Z3 sort supplied for each identifier.

    Raises:
        KeyError: If ``symbol_types`` lacks an entry for one or more of
            ``free_identifiers``.

    """
    missing = free_identifiers - set(symbol_types)
    if not missing:
        return
    sorted_missing = sorted(missing, key=lambda identifier: identifier.id)
    raise KeyError(f"symbol_types is missing entries for identifiers: {sorted_missing}")


def check_expression_satisfiability(
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether some assignment to the free identifiers satisfies the expression.

    True if a satisfying assignment provably exists; False if provably
    none exists; None if the solver returns unknown, or if ``expression``
    is refused by the hazard screen documented on the module docstring
    (logged at WARNING, naming this function and the offending node).
    Implemented as the inversion of ``does_expression_imply(expression,
    false)``: an expression implies false exactly when nothing satisfies
    it.

    Args:
        expression: Expression to check.
        symbol_types: Z3 sort to use for each free identifier of
            ``expression``.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only SATISFIABILITY-capable backend
            today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True, False, or None per the truth table above.

    Raises:
        SolverCapabilityError: If ``backend`` is not SATISFIABILITY-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier.
            Checked ahead of the hazard screen, so the precondition raises
            even for an expression the screen would otherwise refuse.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number. Such an
            expression is ill-typed rather than undecidable, so it raises
            instead of reporting ``None``.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _validate_backend_capability(backend, SolverQueryKind.SATISFIABILITY)
    validate_timeout_milliseconds(timeout_milliseconds)
    _validate_symbol_types_cover_free_identifiers(
        expression.get_free_identifiers(), symbol_types
    )
    if _find_and_log_hazard(
        expression, symbol_types, context="check_expression_satisfiability"
    ):
        return None
    # INVARIANT: _BACKEND_CAPABILITIES grants SATISFIABILITY to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second SATISFIABILITY-capable
    # backend requires real dispatch here.
    implies_false = _z3_does_expression_imply(
        expression,
        LiteralExpression(False),
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
    if implies_false is None:
        return None
    return not implies_false


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the antecedent logically implies the consequent.

    Delegates to the Z3 bridge, which performs all conversion and
    decision math; None means the solver returned unknown, or that
    either side was refused by the hazard screen documented on the
    module docstring (logged at WARNING, naming this function and the
    offending node; the antecedent is screened before the consequent).

    Args:
        antecedent: The premise expression.
        consequent: The conclusion expression.
        symbol_types: Z3 sort to use for each identifier referenced by
            either expression.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only IMPLICATION-capable backend
            today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True if the implication holds for every assignment; False if a
        counterexample exists; None if the solver returns unknown or
        either side is screened out.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier
            of either expression. Checked ahead of the hazard screen, so
            the precondition raises even for a pair the screen would
            otherwise refuse.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in either expression provably denotes a number. Such
            a pair is ill-typed rather than undecidable, so it raises
            instead of reporting ``None``.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _validate_backend_capability(backend, SolverQueryKind.IMPLICATION)
    validate_timeout_milliseconds(timeout_milliseconds)
    _validate_symbol_types_cover_free_identifiers(
        antecedent.get_free_identifiers() | consequent.get_free_identifiers(),
        symbol_types,
    )
    if _find_and_log_hazard(
        antecedent, symbol_types, context="does_expression_imply"
    ) or _find_and_log_hazard(
        consequent, symbol_types, context="does_expression_imply"
    ):
        return None
    # INVARIANT: _BACKEND_CAPABILITIES grants IMPLICATION to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second IMPLICATION-capable backend
    # requires real dispatch here.
    return _z3_does_expression_imply(
        antecedent,
        consequent,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether the expression holds for every free assignment.

    Args:
        considered_identifiers: Identifiers existentially quantified by
            the check; identifiers in the expression but not in this set
            are treated as free (universally quantified).
        expression: Expression to check.
        symbol_types: Z3 sort to use for each identifier appearing in
            the expression.
        backend: Solver backend to route the query to. Defaults to
            ``SolverBackend.Z3``, the only UNIVERSAL_VALIDITY-capable
            backend today.
        timeout_milliseconds: Optional bound, in milliseconds, on the
            underlying Z3 solver invocation. ``None`` (the default)
            leaves the solver unbounded.

    Returns:
        True if the expression has a witness for every free assignment;
        False if some free assignment has no witness; None if the solver
        returns unknown, or ``expression`` is refused by the hazard
        screen documented on the module docstring (logged at WARNING,
        naming this function and the offending node).

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
        KeyError: If ``symbol_types`` lacks an entry for a free or
            considered identifier. Checked ahead of the hazard screen, so
            the precondition raises even for an expression the screen
            would otherwise refuse.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number. Such an
            expression is ill-typed rather than undecidable, so it raises
            instead of reporting ``None``.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _validate_backend_capability(backend, SolverQueryKind.UNIVERSAL_VALIDITY)
    validate_timeout_milliseconds(timeout_milliseconds)
    _validate_symbol_types_cover_free_identifiers(
        expression.get_free_identifiers() | set(considered_identifiers), symbol_types
    )
    if _find_and_log_hazard(
        expression, symbol_types, context="holds_for_all_free_assignments"
    ):
        return None
    # INVARIANT: _BACKEND_CAPABILITIES grants UNIVERSAL_VALIDITY to exactly
    # one backend (Z3), so delegating straight to the Z3 bridge is valid
    # without dispatching on `backend`. Adding a second
    # UNIVERSAL_VALIDITY-capable backend requires real dispatch here.
    return _z3_holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check universal validity, raising ``UndecidableError`` on ``unknown``.

    Args:
        considered_identifiers: As for
            :func:`holds_for_all_free_assignments`.
        expression: As for :func:`holds_for_all_free_assignments`.
        symbol_types: As for :func:`holds_for_all_free_assignments`.
        backend: As for :func:`holds_for_all_free_assignments`.
        timeout_milliseconds: As for :func:`holds_for_all_free_assignments`.

    Returns:
        The decided ``bool`` result.

    Raises:
        SolverCapabilityError: If ``backend`` is not UNIVERSAL_VALIDITY-capable.
        UndecidableError: When the solver returns unknown, or when
            ``expression`` is refused by the hazard screen documented on
            the module docstring (logged at WARNING, naming this function
            and the offending node).
        KeyError: If ``symbol_types`` lacks an entry for a free or
            considered identifier. Checked ahead of the hazard screen, so
            the precondition raises even for an expression the screen
            would otherwise refuse.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number. Reported
            as its own error rather than as ``UndecidableError``: the
            expression is ill-typed, so no ``timeout_milliseconds`` makes it
            decidable.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _validate_backend_capability(backend, SolverQueryKind.UNIVERSAL_VALIDITY)
    validate_timeout_milliseconds(timeout_milliseconds)
    _validate_symbol_types_cover_free_identifiers(
        expression.get_free_identifiers() | set(considered_identifiers), symbol_types
    )
    if _find_and_log_hazard(
        expression, symbol_types, context="assert_holds_for_all_free_assignments"
    ):
        raise UndecidableError(
            "assert_holds_for_all_free_assignments: the expression was "
            "refused by the solver seam's hazard screen before Z3 was "
            "consulted; see the WARNING logged just above for the "
            "offending node. The property is undecidable with the "
            "current solver configuration.",
            reason=_HAZARD_SCREEN_REASON,
        )
    # INVARIANT: _BACKEND_CAPABILITIES grants UNIVERSAL_VALIDITY to exactly
    # one backend (Z3), so delegating straight to the Z3 bridge is valid
    # without dispatching on `backend`. Adding a second
    # UNIVERSAL_VALIDITY-capable backend requires real dispatch here.
    return _z3_assert_holds_for_all_free_assignments(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
    *,
    backend: SolverBackend = SolverBackend.Z3,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check the implication, raising ``UndecidableError`` on ``unknown``.

    Raises ``UndecidableError`` instead of returning ``None`` when the
    solver's result is unknown, or when either expression is refused by
    the hazard screen documented on the module docstring (logged at
    WARNING, naming this function and the offending node; the antecedent
    is screened before the consequent).

    Args:
        antecedent: As for :func:`does_expression_imply`.
        consequent: As for :func:`does_expression_imply`.
        symbol_types: As for :func:`does_expression_imply`.
        backend: As for :func:`does_expression_imply`.
        timeout_milliseconds: As for :func:`does_expression_imply`.

    Returns:
        The decided ``bool`` result.

    Raises:
        SolverCapabilityError: If ``backend`` is not IMPLICATION-capable.
        UndecidableError: When the solver returns unknown, or when either
            expression is screened out.
        KeyError: If ``symbol_types`` lacks an entry for a free identifier
            of either expression. Checked ahead of the hazard screen, so
            the precondition raises even for a pair the screen would
            otherwise refuse.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in either expression provably denotes a number.
            Reported as its own error rather than as ``UndecidableError``:
            the pair is ill-typed, so no ``timeout_milliseconds`` makes it
            decidable.
        ValueError: If ``timeout_milliseconds`` is not None and not positive.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _validate_backend_capability(backend, SolverQueryKind.IMPLICATION)
    validate_timeout_milliseconds(timeout_milliseconds)
    _validate_symbol_types_cover_free_identifiers(
        antecedent.get_free_identifiers() | consequent.get_free_identifiers(),
        symbol_types,
    )
    if _find_and_log_hazard(
        antecedent, symbol_types, context="assert_expression_implies"
    ) or _find_and_log_hazard(
        consequent, symbol_types, context="assert_expression_implies"
    ):
        raise UndecidableError(
            "assert_expression_implies: the expression was refused by the "
            "solver seam's hazard screen before Z3 was consulted; see the "
            "WARNING logged just above for the offending node. The "
            "implication is undecidable with the current solver "
            "configuration.",
            reason=_HAZARD_SCREEN_REASON,
        )
    # INVARIANT: _BACKEND_CAPABILITIES grants IMPLICATION to exactly one
    # backend (Z3), so delegating straight to the Z3 bridge is valid without
    # dispatching on `backend`. Adding a second IMPLICATION-capable backend
    # requires real dispatch here.
    return _z3_assert_expression_implies(
        antecedent,
        consequent,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
