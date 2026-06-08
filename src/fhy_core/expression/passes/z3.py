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
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from typing import Any

import z3  # type: ignore
from frozendict import frozendict

from fhy_core.expression.core import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    logical_and,
    logical_not,
)
from fhy_core.expression.errors import UndecidableError
from fhy_core.expression.registry import (
    EntryLookupError,
    NativeConstant,
    RegisteredFunction,
    get_registered_entry,
)
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.pass_infrastructure import (
    PassExecutionError,
    VisitablePass,
    register_pass,
)
from fhy_core.symbol_type import SymbolType

_LOGGER = get_logger(__name__)


def _z3_floor_divide(left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    expr: z3.ArithRef = left / right
    if expr.is_real():
        return z3.ToInt(expr)
    elif expr.is_int():
        return expr
    else:
        raise ValueError(f"Unsupported floor divide expression type: {expr}")


@register_pass(
    "fhy_core.expression.to_z3",
    "Lower expression IR into an equivalent Z3 expression.",
)
class ExpressionToZ3Converter(VisitablePass[Expression, z3.ExprRef]):
    """Transforms an expression into a Z3 expression."""

    _UNARY_OPERATION_Z3_OPERATORS: frozendict[UnaryOperation, Callable[[Any], Any]] = (
        frozendict(
            {
                UnaryOperation.NEGATE: operator.neg,
                UnaryOperation.POSITIVE: operator.pos,
                UnaryOperation.LOGICAL_NOT: z3.Not,
            }
        )
    )
    _BINARY_OPERATION_Z3_OPERATORS: frozendict[
        BinaryOperation, Callable[[Any, Any], Any]
    ] = frozendict(
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

    _symbol_types: dict[Identifier, SymbolType]
    _identifier_to_z3_expression: dict[Identifier, z3.ExprRef]

    def __init__(self, symbol_types: dict[Identifier, SymbolType]) -> None:
        super().__init__()
        self._symbol_types = symbol_types
        self._identifier_to_z3_expression = {}

    @property
    def identifier_to_z3_expression(self) -> frozendict[Identifier, z3.ExprRef]:
        return frozendict(self._identifier_to_z3_expression)

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

    def visit_ternary_expression(
        self, ternary_expression: TernaryExpression
    ) -> z3.ExprRef:
        condition = self.visit(ternary_expression.condition)
        true_value = self.visit(ternary_expression.true_value)
        false_value = self.visit(ternary_expression.false_value)
        return z3.If(condition, true_value, false_value)

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
        if isinstance(literal_expression.value, bool):
            return z3.BoolVal(literal_expression.value)
        elif isinstance(literal_expression.value, int):
            return z3.IntVal(literal_expression.value)
        elif isinstance(literal_expression.value, float):
            return z3.RealVal(literal_expression.value)
        elif isinstance(literal_expression.value, str):
            if literal_expression.value == "True":
                return z3.BoolVal(True)
            elif literal_expression.value == "False":
                return z3.BoolVal(False)
            else:
                return z3.RealVal(literal_expression.value)
        else:
            raise TypeError(
                f"Unsupported literal type: {type(literal_expression.value)}"
            )

    @staticmethod
    def format_identifier(identifier: Identifier) -> str:
        return f"{identifier.name_hint}_{identifier.id}"

    @override
    def get_noop_output(self, ir: Expression) -> z3.ExprRef:
        raise PassExecutionError(
            f'Pass "{self.get_pass_name()}" does not define noop output.'
        )


def convert_expression_to_z3_expression(
    expression: Expression, symbol_types: dict[Identifier, SymbolType] | None = None
) -> tuple[z3.ExprRef, frozendict[Identifier, z3.ExprRef]]:
    """Convert an expression to a Z3 expression.

    Args:
        expression: Expression to convert.
        symbol_types: Symbol types.

    Returns:
        Z3 expression and mapping of identifiers to Z3 expressions.

    Raises:
        KeyError: If ``symbol_types`` is missing an entry for any
            identifier referenced by ``expression``.

    """
    resolved_symbol_types = symbol_types or {}
    referenced_identifiers = expression.get_free_identifiers()
    missing_identifiers = referenced_identifiers - resolved_symbol_types.keys()
    if missing_identifiers:
        sorted_missing = sorted(missing_identifiers, key=lambda i: i.id)
        raise KeyError(
            f"symbol_types is missing entries for identifiers: {sorted_missing}"
        )
    converter = ExpressionToZ3Converter(resolved_symbol_types)
    z3_expression = converter(expression)
    return z3_expression, converter.identifier_to_z3_expression


def holds_for_all_free_assignments(
    considered_identifiers: AbstractSet[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
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

    Returns:
        True if the expression has a witness for every free assignment;
        False if some free assignment has no witness; None if Z3 returns
        ``unknown``.

    Raises:
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
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
        return True
    elif result == z3.sat:
        return False
    elif result == z3.unknown:
        _LOGGER.warning(
            "Z3 returned `unknown` (considered=%d, quantified=%d); returning None",
            len(considered_identifiers),
            len(quantified_variables),
        )
        return None
    else:
        raise RuntimeError("Unexpected Z3 result.")


def does_expression_imply(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
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

    Returns:
        True if the implication holds for every assignment; False if a
        counterexample exists; None if Z3 returns ``unknown``.

    Raises:
        KeyError: If ``symbol_types`` is missing an entry for any
            identifier referenced by either expression.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    _LOGGER.debug("antecedent=%r, consequent=%r", antecedent, consequent)
    combined = logical_and(antecedent, logical_not(consequent))
    all_identifiers = combined.get_free_identifiers()
    has_counterexample = holds_for_all_free_assignments(
        all_identifiers, combined, symbol_types
    )
    if has_counterexample is None:
        return None
    return not has_counterexample


def assert_holds_for_all_free_assignments(
    considered_identifiers: set[Identifier],
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
) -> bool:
    """Strict variant of :func:`holds_for_all_free_assignments`.

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

    Returns:
        The decided ``bool`` result.

    Raises:
        UndecidableError: When Z3 returns ``unknown``.
        KeyError: If ``symbol_types`` is missing an entry.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result = holds_for_all_free_assignments(
        considered_identifiers, expression, symbol_types
    )
    if result is None:
        raise UndecidableError(
            "Z3 returned `unknown` for "
            "`holds_for_all_free_assignments`; the property is undecidable "
            "with the current solver configuration."
        )
    return result


def assert_expression_implies(
    antecedent: Expression,
    consequent: Expression,
    symbol_types: dict[Identifier, SymbolType],
) -> bool:
    """Strict variant of :func:`does_expression_imply`.

    Raises :class:`UndecidableError` when the underlying solver returns
    ``unknown`` instead of returning ``None``.

    Args:
        antecedent: As for :func:`does_expression_imply`.
        consequent: As for :func:`does_expression_imply`.
        symbol_types: As for :func:`does_expression_imply`.

    Returns:
        The decided ``bool`` result.

    Raises:
        UndecidableError: When Z3 returns ``unknown``.
        KeyError: If ``symbol_types`` is missing an entry.
        RuntimeError: If the underlying solver returns an unrecognized
            result.

    """
    result = does_expression_imply(antecedent, consequent, symbol_types)
    if result is None:
        raise UndecidableError(
            "Z3 returned `unknown` for `does_expression_imply`; the "
            "implication is undecidable with the current solver configuration."
        )
    return result
