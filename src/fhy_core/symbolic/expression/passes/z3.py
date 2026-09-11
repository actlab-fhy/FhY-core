"""Expression passes that interface with Z3."""

from fhy_core.utils.override import override

__all__ = [
    "assert_expression_implies",
    "assert_holds_for_all_free_assignments",
    "convert_expression_to_z3_expression",
    "does_expression_imply",
    "holds_for_all_free_assignments",
]

import operator
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from typing import Any

import z3  # type: ignore
from immutabledict import immutabledict

from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.pass_infrastructure import (
    PassExecutionError,
    VisitablePass,
    register_pass,
)

from ...symbol_type import SymbolType
from ..core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    is_integer_valued_literal,
    logical_and,
    logical_not,
    validate_logical_operands,
    validate_predicate,
)
from ..errors import NativeConstantLoweringError, UndecidableError
from ..registry import (
    EntryLookupError,
    NativeConstant,
    RegisteredFunction,
    get_registered_entry,
    try_get_native_constant_for_identifier,
)

_LOGGER = get_logger(__name__)


def _z3_floor_divide(left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    """Lower floor division, real operands by truncation toward negative infinity.

    The two branches do not agree on negative operands. A real-sorted
    quotient goes through ``ToInt``, which is a true floor; an int-sorted
    one is Z3's own ``div``, which is Euclidean and disagrees with this
    package's floor semantics whenever the divisor is not positive.
    ``MODULO`` lowers to Z3's Euclidean ``mod`` for the same reason.

    Nothing here compensates for that divergence. The solver seam screens
    it instead, refusing to lower floor division or modulo without a
    finite strictly positive literal divisor, so a caller reaching this
    function through the seam has already been narrowed to the operands
    both semantics agree on.

    """
    expr: z3.ArithRef = left / right
    if expr.is_real():
        return z3.ToInt(expr)
    elif expr.is_int():
        return expr
    else:
        raise ValueError(f"Unsupported floor divide expression type: {expr}")


@register_pass(
    "fhy_core.symbolic.expression.to_z3",
    "Lower expression IR into an equivalent Z3 expression.",
)
class ExpressionToZ3Converter(VisitablePass[Expression, z3.ExprRef]):
    """Transforms an expression into a Z3 expression."""

    _UNARY_OPERATION_Z3_OPERATORS: immutabledict[
        UnaryOperation, Callable[[Any], Any]
    ] = immutabledict(
        {
            UnaryOperation.NEGATE: operator.neg,
            UnaryOperation.POSITIVE: operator.pos,
            UnaryOperation.LOGICAL_NOT: z3.Not,
        }
    )
    _BINARY_OPERATION_Z3_OPERATORS: immutabledict[
        BinaryOperation, Callable[[Any, Any], Any]
    ] = immutabledict(
        {
            BinaryOperation.ADD: operator.add,
            BinaryOperation.SUBTRACT: operator.sub,
            BinaryOperation.MULTIPLY: operator.mul,
            BinaryOperation.DIVIDE: operator.truediv,
            BinaryOperation.FLOOR_DIVIDE: _z3_floor_divide,
            BinaryOperation.MODULO: operator.mod,
            BinaryOperation.POWER: operator.pow,
            BinaryOperation.LOGICAL_AND: z3.And,
            BinaryOperation.LOGICAL_OR: z3.Or,
            BinaryOperation.EQUAL: operator.eq,
            BinaryOperation.NOT_EQUAL: operator.ne,
            BinaryOperation.LESS: operator.lt,
            BinaryOperation.LESS_EQUAL: operator.le,
            BinaryOperation.GREATER: operator.gt,
            BinaryOperation.GREATER_EQUAL: operator.ge,
        }
    )

    _symbol_types: immutabledict[Identifier, SymbolType]
    _identifier_to_z3_expression: dict[Identifier, z3.ExprRef]

    def __init__(self, symbol_types: Mapping[Identifier, SymbolType]) -> None:
        super().__init__()
        self._symbol_types = immutabledict(symbol_types)
        self._identifier_to_z3_expression = {}

    @property
    def identifier_to_z3_expression(self) -> immutabledict[Identifier, z3.ExprRef]:
        return immutabledict(self._identifier_to_z3_expression)

    def visit_binary_expression(
        self, binary_expression: BinaryExpression
    ) -> z3.ExprRef:
        left = self.visit(binary_expression.left)
        right = self.visit(binary_expression.right)
        operation_function = self._BINARY_OPERATION_Z3_OPERATORS[
            binary_expression.operation
        ]
        return operation_function(left, right)

    def visit_unary_expression(self, unary_expression: UnaryExpression) -> z3.ExprRef:
        operand = self.visit(unary_expression.operand)
        operation_function = self._UNARY_OPERATION_Z3_OPERATORS[
            unary_expression.operation
        ]
        return operation_function(operand)

    def visit_identifier_expression(
        self, identifier_expression: IdentifierExpression
    ) -> z3.ExprRef:
        identifier_name = self.format_identifier(identifier_expression.identifier)
        identifier_type = self._symbol_types[identifier_expression.identifier]
        if identifier_type == SymbolType.REAL:
            result = z3.Real(identifier_name)
        elif identifier_type == SymbolType.INT:
            result = z3.Int(identifier_name)
        elif identifier_type == SymbolType.BOOL:
            result = z3.Bool(identifier_name)
        else:
            raise ValueError(f"Unsupported identifier type: {identifier_type}.")
        self._identifier_to_z3_expression[identifier_expression.identifier] = result
        return result

    def visit_piecewise_expression(
        self, piecewise_expression: PiecewiseExpression
    ) -> z3.ExprRef:
        """Lower to a right-folded ``z3.If`` chain.

        Conditions and values are visited in case order, then
        ``otherwise`` -- deterministic left-to-right for the identifier
        cache side effects. Z3 has no n-ary conditional; the right-folded
        nested ``If`` is first-match-wins by construction. Every
        condition must be boolean-valued; Z3 raises
        ``z3.z3types.Z3Exception`` for a condition it cannot convert to
        a Z3 Boolean.
        """
        case_values = [
            (self.visit(condition), self.visit(value))
            for condition, value in piecewise_expression.get_cases()
        ]
        result = self.visit(piecewise_expression.otherwise)
        for condition_z3, value_z3 in reversed(case_values):
            result = z3.If(condition_z3, value_z3, result)
        return result

    def visit_call_expression(self, call_expression: CallExpression) -> z3.ExprRef:
        name = call_expression.function_name
        try:
            entry = get_registered_entry(name)
        except EntryLookupError as exc:
            raise TypeError(
                f"cannot lower call to unknown function {name!r} to Z3"
            ) from exc
        if isinstance(entry, RegisteredFunction):
            raise TypeError(
                f"Cannot lower an expression-bodied function call to Z3; "
                f"call `inline_functions` first to expand {name!r}."
            )
        if isinstance(entry, NativeConstant):
            raise TypeError(
                f"call to {name!r} is to a registered native constant, which "
                f"is not callable"
            )
        raise TypeError(
            f"Z3 does not support native function calls; {name!r} cannot be lowered"
        )

    def visit_literal_expression(
        self, literal_expression: LiteralExpression
    ) -> z3.ExprRef:
        """Lower a literal to the Z3 numeral its IR form denotes exactly.

        ``LiteralExpression`` gives each literal form its own precision
        contract, and each form reaches Z3 as the exact value that
        contract names:

        - ``bool`` becomes a ``BoolVal``.
        - An integer -- a Python ``int`` or an integer-grammar ``str`` --
          becomes an ``IntVal``. Both are in one equivalence class, so
          both have to reach Z3 in one sort; ``RealVal`` for the string
          form would let the solver decide for one member of a class what
          it refuses for another.
        - A Python ``float`` is an IEEE-754 binary value, so it becomes
          the rational its bits denote, taken from
          ``float.as_integer_ratio``. Handing the ``float`` itself to
          ``z3.RealVal`` would instead reinterpret its shortest repr as
          exact decimal text, so ``0.1`` would reach the solver as
          ``1/10`` -- a different number from the one the literal stores.
        - A float-grammar ``str`` is exact decimal, and ``RealVal`` reads
          decimal text exactly, so the text goes to Z3 unconverted.

        The SymPy bridge lowers each of those forms to the same value, so
        a single literal denotes the same number on both bridges. Past a
        single literal the two diverge: ``simplify_expression`` evaluates
        binary-float arithmetic in SymPy's binary floating point, while
        the solver seam reasons over it in exact rational arithmetic, so
        a ground comparison that does float arithmetic on both sides can
        come out differently -- for example, ``(1e16 + 1.0) == 1e16``
        simplifies to ``True`` but the solver seam finds it ``False``.

        A non-finite ``float`` has no rational value at all:
        ``as_integer_ratio`` raises ``OverflowError`` for an infinity and
        ``ValueError`` for a NaN, and the pass infrastructure surfaces
        either as a ``PassExecutionError``. The solver seam refuses every
        non-finite literal before it reaches this bridge. An unsupported
        literal type raises ``TypeError``.
        """
        value = literal_expression.value
        if isinstance(value, bool):
            return z3.BoolVal(value)
        elif isinstance(value, int):
            return z3.IntVal(value)
        elif isinstance(value, float):
            return z3.RatVal(*value.as_integer_ratio())
        elif isinstance(value, str):
            if is_integer_valued_literal(value):
                return z3.IntVal(int(value))
            else:
                return z3.RealVal(value)
        else:
            raise TypeError(f"Unsupported literal type: {type(value)}")

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"

    @override
    def get_noop_output(self, ir: Expression) -> z3.ExprRef:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def _find_referenced_native_constant_identifiers(
    expression: Expression,
) -> frozenset[Identifier]:
    """Return the canonical identifiers of native constants ``expression`` references.

    Args:
        expression: Expression about to be lowered to Z3.

    Returns:
        The referenced canonical identifiers; empty when ``expression``
        references none.

    """
    return frozenset(
        identifier
        for identifier in expression.get_free_identifiers()
        if try_get_native_constant_for_identifier(identifier) is not None
    )


def _raise_if_missing_z3_symbol_types(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    constant_identifiers: AbstractSet[Identifier],
) -> None:
    """Raise unless ``symbol_types`` covers every non-constant free identifier.

    A registered native constant's canonical identifier is exempt: it
    names a value rather than a variable, and the Z3 bridge never reads
    a sort for it.

    Args:
        expression: Expression about to be lowered to Z3.
        symbol_types: Z3 sort declared for each free identifier.
        constant_identifiers: Referenced native constants' canonical
            identifiers, exempt from the coverage requirement.

    Raises:
        KeyError: If ``symbol_types`` lacks an entry for a free
            identifier of ``expression`` other than one in
            ``constant_identifiers``.

    """
    missing_identifiers = (
        expression.get_free_identifiers() - constant_identifiers - symbol_types.keys()
    )
    if missing_identifiers:
        sorted_missing = sorted(missing_identifiers, key=lambda i: i.id)
        raise KeyError(
            f"symbol_types is missing entries for identifiers: {sorted_missing}"
        )


def convert_expression_to_z3_expression(
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType] | None = None,
) -> tuple[z3.ExprRef, immutabledict[Identifier, z3.ExprRef]]:
    """Convert an expression to a Z3 expression.

    Screens the expression before lowering: a provably numeric operand of
    a logical connective, or a provably numeric piecewise case condition,
    is refused here rather than handed to Z3,
    which reports the sort mismatch as a backend exception the pass
    infrastructure would then wrap. An identifier ``symbol_types``
    declares INT or REAL counts as numeric, since that is the sort it
    would be lowered with. A registered native constant's
    canonical identifier is refused as well: the bridge has no lowering
    for a constant, and lowering the identifier as a variable would let
    Z3 choose the constant's value.

    Args:
        expression: Expression to convert.
        symbol_types: Symbol types. A native constant's canonical
            identifier names a value rather than a variable, so it needs
            no entry.

    Returns:
        Z3 expression and mapping of identifiers to Z3 expressions.

    Raises:
        KeyError: If ``symbol_types`` is missing an entry for any
            identifier referenced by ``expression`` other than a native
            constant's canonical identifier.
        NonBooleanLogicalOperandError: If an operand of a ``LOGICAL_AND``,
            ``LOGICAL_OR``, or ``LOGICAL_NOT`` node, or a piecewise case
            condition, in ``expression`` provably denotes a number, counting
            an identifier ``symbol_types`` declares INT or REAL. Screened
            after the ``symbol_types`` precondition, so a missing entry
            raises first.
        NativeConstantLoweringError: If ``expression`` references a
            registered native constant's canonical identifier. Checked
            after both errors above, so a missing entry for a variable or
            an ill-typed operand is reported first.

    """
    resolved_symbol_types = symbol_types or {}
    constant_identifiers = _find_referenced_native_constant_identifiers(expression)
    _raise_if_missing_z3_symbol_types(
        expression, resolved_symbol_types, constant_identifiers
    )
    validate_logical_operands(expression, symbol_types=resolved_symbol_types)
    if constant_identifiers:
        sorted_constants = sorted(constant_identifiers, key=lambda i: i.id)
        raise NativeConstantLoweringError(
            f"Cannot lower the native constant(s) {sorted_constants} to Z3: the "
            "bridge has no term for a constant's value, and lowering its "
            "canonical identifier as a variable would let the solver choose it."
        )
    converter = ExpressionToZ3Converter(resolved_symbol_types)
    z3_expression = converter(expression)
    return z3_expression, converter.identifier_to_z3_expression


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Check whether the expression has a witness for every free assignment.

    Returns True iff
    ``forall <free identifiers>. exists <considered_identifiers>. expression``:
    for every assignment to the free identifiers (those *not* in
    ``considered_identifiers``), there exists some assignment to the
    considered identifiers that satisfies the expression. When
    ``considered_identifiers`` is empty, the check degenerates to "the
    expression holds for every assignment to its free identifiers" -- i.e.,
    the expression is universally valid.

    Note:
        This is not the same as standard satisfiability
        (``exists <all vars>. expression``). For an implication-style check
        ``antecedent -> consequent``, prefer :func:`does_expression_imply`.

    Args:
        considered_identifiers: Identifiers existentially quantified by
            the check. Identifiers in the expression but not in this set
            are treated as free (universally quantified).
        expression: Expression to check.
        symbol_types: Z3 sort to use for each identifier appearing in
            the expression. Every free or considered identifier
            referenced by the expression must have an entry.
        timeout_milliseconds: Optional bound on the underlying
            ``z3.Solver`` invocation, in milliseconds. ``None`` (the
            default) leaves the solver unbounded. Callers are expected
            to have already validated this is ``None`` or positive.

    Returns:
        True if the expression has a witness for every free assignment;
        False if some free assignment has no witness; None if Z3 returns
        ``unknown``.

    Raises:
        KeyError: If ``symbol_types`` lacks an entry for a free
            identifier of ``expression`` other than a native constant's
            canonical identifier.
        NonBooleanLogicalOperandError: If ``expression``'s root, an
            operand of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an identifier ``symbol_types``
            declares INT or REAL, which Z3 has no faithful lowering for.
        NativeConstantLoweringError: If ``expression`` references a
            registered native constant's canonical identifier.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result, _ = _holds_for_all_free_assignments_with_reason(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
    return result


def _holds_for_all_free_assignments_with_reason(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> tuple[bool | None, str | None]:
    """Run the check behind :func:`holds_for_all_free_assignments`, with Z3's reason.

    Computes the same tri-valued result as
    :func:`holds_for_all_free_assignments`, paired with Z3's
    ``reason_unknown()`` text when the result is ``None`` (and ``None``
    otherwise) so :func:`assert_holds_for_all_free_assignments` can
    report the reason without widening the public function's return type.

    ``expression`` is itself a predicate here -- there is no connective
    or piecewise condition above it the way the implication encoding
    wraps one -- so it is screened with :func:`validate_predicate`
    before it reaches :func:`convert_expression_to_z3_expression`, ahead
    of that function's own :func:`validate_logical_operands` screen, so
    a numeric root is refused rather than reaching Z3's ``Not`` directly.

    Raises:
        KeyError: If ``symbol_types`` lacks an entry for a free
            identifier of ``expression`` other than a native constant's
            canonical identifier.
        NonBooleanLogicalOperandError: If ``expression``'s root, or an
            operand of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an identifier ``symbol_types``
            declares INT or REAL. Checked after the ``symbol_types``
            precondition and ahead of the native-constant refusal.
        NativeConstantLoweringError: If ``expression`` references a
            registered native constant's canonical identifier.

    """
    constant_identifiers = _find_referenced_native_constant_identifiers(expression)
    _raise_if_missing_z3_symbol_types(expression, symbol_types, constant_identifiers)
    validate_predicate(expression, symbol_types=symbol_types)
    z3_expression, identifier_to_z3_expression = convert_expression_to_z3_expression(
        expression, symbol_types
    )
    z3_expression = z3.Not(z3_expression)
    quantified_variables = [
        identifier_to_z3_expression[identifier]
        for identifier in considered_identifiers
        if identifier in identifier_to_z3_expression
    ]
    if quantified_variables:
        z3_expression = z3.ForAll(quantified_variables, z3_expression)

    solver = z3.Solver()
    if timeout_milliseconds is not None:
        solver.set(timeout=timeout_milliseconds)
    solver.add(z3_expression)

    _LOGGER.debug(
        "calling Z3 solver.check (referenced_identifiers=%d, considered=%d, "
        "quantified=%d)",
        len(identifier_to_z3_expression),
        len(considered_identifiers),
        len(quantified_variables),
    )
    result = solver.check()
    if result == z3.unsat:
        return True, None
    elif result == z3.sat:
        return False, None
    elif result == z3.unknown:
        reason = solver.reason_unknown()
        _LOGGER.warning(
            "Z3 returned `unknown` (considered=%d, quantified=%d): %s; returning None",
            len(considered_identifiers),
            len(quantified_variables),
            reason,
        )
        return None, reason
    else:
        raise RuntimeError(f"Unexpected Z3 result: {result!r}.")


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> bool | None:
    """Return whether ``antecedent`` logically implies ``consequent``.

    Returns True iff ``forall <all identifiers>. antecedent -> consequent``,
    i.e. there is no assignment to the free identifiers of either
    expression that satisfies ``antecedent`` but not ``consequent``.

    Args:
        antecedent: The premise expression.
        consequent: The conclusion expression.
        symbol_types: Z3 sort to use for each identifier referenced by
            either expression.
        timeout_milliseconds: As for
            :func:`holds_for_all_free_assignments`.

    Returns:
        True if the implication holds for every assignment; False if a
        counterexample exists; None if Z3 returns ``unknown``.

    Raises:
        KeyError: If ``symbol_types`` is missing an entry for any
            identifier referenced by either expression.
        NonBooleanLogicalOperandError: If either expression's root, an
            operand of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an identifier ``symbol_types``
            declares INT or REAL. The check runs over the conjunction the
            implication is encoded as, so a numeric ``antecedent`` or
            ``consequent`` is caught too.
        NativeConstantLoweringError: If either expression references a
            registered native constant's canonical identifier.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result, _ = _does_expression_imply_with_reason(
        antecedent, consequent, symbol_types, timeout_milliseconds=timeout_milliseconds
    )
    return result


def _does_expression_imply_with_reason(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> tuple[bool | None, str | None]:
    """Run the check behind :func:`does_expression_imply`, with Z3's reason.

    Computes the same tri-valued result as :func:`does_expression_imply`,
    paired with Z3's ``reason_unknown()`` text when the result is
    ``None`` (and ``None`` otherwise) so :func:`assert_expression_implies`
    can report the reason without widening the public function's return
    type.

    """
    _LOGGER.debug("antecedent=%r, consequent=%r", antecedent, consequent)
    combined = logical_and(antecedent, logical_not(consequent))
    all_identifiers = combined.get_free_identifiers()
    has_counterexample, reason = _holds_for_all_free_assignments_with_reason(
        all_identifiers,
        combined,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
    if has_counterexample is None:
        return None, reason
    return not has_counterexample, None


def assert_holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check universal validity, raising :class:`UndecidableError` on ``unknown``.

    Raises :class:`UndecidableError` when the underlying solver returns
    ``unknown`` instead of returning ``None``. Callers that need a
    decided ``True``/``False`` should prefer this companion; the
    lenient variant remains available for callers that want to handle
    ``unknown`` themselves.

    Args:
        considered_identifiers: As for
            :func:`holds_for_all_free_assignments`.
        expression: As for :func:`holds_for_all_free_assignments`.
        symbol_types: As for :func:`holds_for_all_free_assignments`.
        timeout_milliseconds: As for
            :func:`holds_for_all_free_assignments`.

    Returns:
        The decided ``bool`` result.

    Raises:
        UndecidableError: When Z3 returns ``unknown``. The message
            includes Z3's ``reason_unknown()`` text.
        KeyError: If ``symbol_types`` is missing an entry.
        NonBooleanLogicalOperandError: If ``expression``'s root, an
            operand of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an identifier ``symbol_types``
            declares INT or REAL.
        NativeConstantLoweringError: If ``expression`` references a
            registered native constant's canonical identifier.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result, reason = _holds_for_all_free_assignments_with_reason(
        considered_identifiers,
        expression,
        symbol_types,
        timeout_milliseconds=timeout_milliseconds,
    )
    if result is None:
        raise UndecidableError(
            "Z3 returned `unknown` for `holds_for_all_free_assignments` "
            f"({reason}); the property is undecidable with the current "
            "solver configuration.",
            reason=reason if reason is not None else "",
        )
    return result


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: Mapping[Identifier, SymbolType],
    *,
    timeout_milliseconds: int | None = None,
) -> bool:
    """Check the implication, raising :class:`UndecidableError` on ``unknown``.

    Raises :class:`UndecidableError` when the underlying solver returns
    ``unknown`` instead of returning ``None``.

    Args:
        antecedent: As for :func:`does_expression_imply`.
        consequent: As for :func:`does_expression_imply`.
        symbol_types: As for :func:`does_expression_imply`.
        timeout_milliseconds: As for :func:`does_expression_imply`.

    Returns:
        The decided ``bool`` result.

    Raises:
        UndecidableError: When Z3 returns ``unknown``. The message
            includes Z3's ``reason_unknown()`` text.
        KeyError: If ``symbol_types`` is missing an entry.
        NonBooleanLogicalOperandError: If either expression's root, an
            operand of a ``LOGICAL_AND``, ``LOGICAL_OR``, or
            ``LOGICAL_NOT`` node, or a piecewise case condition, provably
            denotes a number, counting an identifier ``symbol_types``
            declares INT or REAL.
        NativeConstantLoweringError: If either expression references a
            registered native constant's canonical identifier.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result, reason = _does_expression_imply_with_reason(
        antecedent, consequent, symbol_types, timeout_milliseconds=timeout_milliseconds
    )
    if result is None:
        raise UndecidableError(
            "Z3 returned `unknown` for `does_expression_imply` "
            f"({reason}); the implication is undecidable with the "
            "current solver configuration.",
            reason=reason if reason is not None else "",
        )
    return result
