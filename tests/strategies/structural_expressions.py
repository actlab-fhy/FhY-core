"""Hypothesis strategy for the wide grammar used by round-trip and pattern tests.

Shares its leaf-budget machinery with `tests/strategies/expressions.py`
(`_draw_parts_within_leaf_budget`, `_draw_piecewise_expression`, and the
consecutive-wraps bound described there): every node still spends its own
share of the leaf budget by construction, and a `UnaryExpression` or a
single-argument `CallExpression` still counts against the same
"wraps remaining" counter, since neither spends a leaf of its own.

Wider than the gate grammar in that sibling module: numeric leaves draw
from every numeric literal kind (integer, finite float, decimal string),
binary nodes include `DIVIDE` and `POWER`, and calls reach any registered
function whose declared parameter sorts fit the position, not only
`INTEGER_RESULT_NATIVE_FUNCTIONS`. A Boolean node can also be an
`==`/`!=` between two Boolean subtrees. The tree is never evaluated, so
an unrestricted (possibly zero) divisor is fine, and one identifier pool
serves both sorts.
"""

from collections.abc import Callable, Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryOperation,
    Expression,
    FunctionSort,
    IdentifierExpression,
    NativeFunction,
    RegisteredFunction,
    UnaryOperation,
    call,
    get_registered_entries,
    make_binary_expression,
    make_unary_expression,
)

from .expressions import (
    _MAX_CONSECUTIVE_WRAPS,
    _MIN_LEAVES_FOR_TWO_CHILDREN,
    _MIN_PIECEWISE_LEAVES,
    BOOLEAN_EQUALITY_OPERATIONS,
    COMPARISON_OPERATIONS,
    LOGICAL_BINARY_OPERATIONS,
    NUMERIC_DIVISION_OPERATIONS,
    NUMERIC_GATE_OPERATIONS,
    _draw_parts_within_leaf_budget,
    _draw_piecewise_expression,
)
from .identifiers import build_identifier_strategy
from .literals import (
    build_boolean_literal_strategy,
    build_decimal_string_literal_strategy,
    build_finite_float_literal_strategy,
    build_integer_literal_strategy,
)

__all__ = ["build_structural_expression_strategy"]

_STRUCTURAL_NUMERIC_BINARY_OPERATIONS: Final = (
    *NUMERIC_GATE_OPERATIONS,
    *NUMERIC_DIVISION_OPERATIONS,
    BinaryOperation.DIVIDE,
    BinaryOperation.POWER,
)

# (function name, parameter sorts) for every registered entry callable from
# an expression tree (a `RegisteredFunction` or `NativeFunction`; a
# `NativeConstant` has no parameters and is never called). Computed once:
# the builtin registrations run at `fhy_core.symbolic.expression` import
# time, which has already happened by the time this module is imported.
_CALLABLE_SIGNATURES: Final = tuple(
    (name, entry.parameter_sorts, entry.result_sort)
    for name, entry in get_registered_entries().items()
    if isinstance(entry, (RegisteredFunction, NativeFunction))
)
_STRUCTURAL_NUMERIC_CALL_SIGNATURES: Final = tuple(
    (name, sorts)
    for name, sorts, result_sort in _CALLABLE_SIGNATURES
    if result_sort is not FunctionSort.BOOL
)
_STRUCTURAL_BOOLEAN_CALL_SIGNATURES: Final = tuple(
    (name, sorts)
    for name, sorts, result_sort in _CALLABLE_SIGNATURES
    if result_sort is FunctionSort.BOOL
)


def _build_structural_numeric_leaf_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a wide numeric leaf: a numeric literal or an identifier."""
    leaves: list[st.SearchStrategy[Expression]] = [
        build_integer_literal_strategy(),
        build_finite_float_literal_strategy(),
        build_decimal_string_literal_strategy(),
    ]
    if identifiers:
        leaves.append(build_identifier_strategy(identifiers).map(IdentifierExpression))
    return st.one_of(*leaves)


def _build_structural_boolean_leaf_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a boolean leaf: a boolean literal or an identifier."""
    leaves: list[st.SearchStrategy[Expression]] = [build_boolean_literal_strategy()]
    if identifiers:
        leaves.append(build_identifier_strategy(identifiers).map(IdentifierExpression))
    return st.one_of(*leaves)


def _filter_signatures_by_arity(
    signatures: Sequence[tuple[str, tuple[FunctionSort, ...]]], max_leaves: int
) -> tuple[tuple[str, tuple[FunctionSort, ...]], ...]:
    """Return the signatures whose arity fits within ``max_leaves`` leaves."""
    return tuple(
        signature for signature in signatures if len(signature[1]) <= max_leaves
    )


def _build_structural_argument_strategy(
    identifiers: Sequence[Identifier], sort: FunctionSort
) -> Callable[[int], st.SearchStrategy[Expression]]:
    """Return a per-budget strategy factory for a call argument of the given sort."""
    if sort is FunctionSort.BOOL:
        return lambda budget: _build_structural_boolean_strategy(identifiers, budget)
    return lambda budget: _build_structural_numeric_strategy(identifiers, budget)


def _draw_structural_call(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
    signatures: Sequence[tuple[str, tuple[FunctionSort, ...]]],
) -> Expression:
    """Draw a call to one of ``signatures``, one argument per parameter sort."""
    name, parameter_sorts = draw(st.sampled_from(signatures))
    factories = tuple(
        _build_structural_argument_strategy(identifiers, sort)
        for sort in parameter_sorts
    )
    arguments = _draw_parts_within_leaf_budget(draw, factories, max_leaves)
    return call(name, *arguments)


@st.composite
def _draw_structural_numeric_expression(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
    wraps_remaining: int,
) -> Expression:
    """Draw a wide numeric-sorted structural expression tree within a leaf budget."""
    eligible_calls = _filter_signatures_by_arity(
        _STRUCTURAL_NUMERIC_CALL_SIGNATURES, max_leaves
    )
    kinds: list[str] = ["leaf"]
    if wraps_remaining > 0:
        kinds.append("unary")
        if eligible_calls:
            kinds.append("call")
    if max_leaves >= _MIN_LEAVES_FOR_TWO_CHILDREN:
        kinds.append("binary")
    if max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "leaf":
        return draw(_build_structural_numeric_leaf_strategy(identifiers))
    if kind == "unary":
        unary_operation = draw(
            st.sampled_from((UnaryOperation.NEGATE, UnaryOperation.POSITIVE))
        )
        operand = draw(
            _draw_structural_numeric_expression(
                identifiers, max_leaves, wraps_remaining - 1
            )
        )
        return make_unary_expression(unary_operation, operand)
    if kind == "call":
        return _draw_structural_call(draw, identifiers, max_leaves, eligible_calls)
    if kind == "binary":
        binary_operation = draw(st.sampled_from(_STRUCTURAL_NUMERIC_BINARY_OPERATIONS))
        left, right = _draw_parts_within_leaf_budget(
            draw,
            (
                lambda budget: _build_structural_numeric_strategy(identifiers, budget),
                lambda budget: _build_structural_numeric_strategy(identifiers, budget),
            ),
            max_leaves,
        )
        return make_binary_expression(binary_operation, left, right)
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda budget: st.deferred(
            lambda: _build_structural_boolean_strategy(identifiers, budget)
        ),
        lambda budget: _build_structural_numeric_strategy(identifiers, budget),
    )


def _draw_structural_boolean_binary(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_leaves: int, kind: str
) -> Expression:
    """Draw a Boolean binary node of ``kind`` within a leaf budget.

    ``kind`` is ``"comparison"`` (numeric operands), ``"boolean_comparison"``
    (``==``/``!=`` over Boolean operands), or ``"and_or"``.
    """

    def build_numeric_operand(budget: int) -> st.SearchStrategy[Expression]:
        return _build_structural_numeric_strategy(identifiers, budget)

    def build_boolean_operand(budget: int) -> st.SearchStrategy[Expression]:
        return _build_structural_boolean_strategy(identifiers, budget)

    operations, build_operand = {
        "comparison": (COMPARISON_OPERATIONS, build_numeric_operand),
        "boolean_comparison": (BOOLEAN_EQUALITY_OPERATIONS, build_boolean_operand),
        "and_or": (LOGICAL_BINARY_OPERATIONS, build_boolean_operand),
    }[kind]
    operation = draw(st.sampled_from(operations))
    left, right = _draw_parts_within_leaf_budget(
        draw, (build_operand, build_operand), max_leaves
    )
    return make_binary_expression(operation, left, right)


@st.composite
def _draw_structural_boolean_expression(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
    wraps_remaining: int,
) -> Expression:
    """Draw a wide boolean-sorted structural expression tree within a leaf budget."""
    eligible_calls = _filter_signatures_by_arity(
        _STRUCTURAL_BOOLEAN_CALL_SIGNATURES, max_leaves
    )
    kinds: list[str] = ["bool_literal"]
    if max_leaves >= _MIN_LEAVES_FOR_TWO_CHILDREN:
        kinds.append("comparison")
        kinds.append("boolean_comparison")
        kinds.append("and_or")
        if eligible_calls:
            kinds.append("call")
    if wraps_remaining > 0:
        kinds.append("not")
    if max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "bool_literal":
        return draw(_build_structural_boolean_leaf_strategy(identifiers))
    if kind == "call":
        return _draw_structural_call(draw, identifiers, max_leaves, eligible_calls)
    if kind == "not":
        operand = draw(
            _draw_structural_boolean_expression(
                identifiers, max_leaves, wraps_remaining - 1
            )
        )
        return make_unary_expression(UnaryOperation.LOGICAL_NOT, operand)
    if kind != "piecewise":
        return _draw_structural_boolean_binary(draw, identifiers, max_leaves, kind)
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda budget: _build_structural_boolean_strategy(identifiers, budget),
        lambda budget: _build_structural_boolean_strategy(identifiers, budget),
    )


def _build_structural_numeric_strategy(
    identifiers: Sequence[Identifier], max_leaves: int
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a wide numeric-sorted structural tree within a budget."""
    return _draw_structural_numeric_expression(
        identifiers, max_leaves, _MAX_CONSECUTIVE_WRAPS
    )


def _build_structural_boolean_strategy(
    identifiers: Sequence[Identifier], max_leaves: int
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a wide boolean-sorted structural tree within a budget."""
    return _draw_structural_boolean_expression(
        identifiers, max_leaves, _MAX_CONSECUTIVE_WRAPS
    )


def build_structural_expression_strategy(
    identifiers: Sequence[Identifier], max_leaves: int = 10
) -> st.SearchStrategy[Expression]:
    """Return a strategy for wide, sort-correct trees for round-trip and pattern tests.

    Wider than `build_numeric_expression_strategy` and
    `build_boolean_expression_strategy` in `tests/strategies/expressions.py`:
    numeric leaves draw from every numeric literal kind, binary nodes
    include ``DIVIDE`` and ``POWER``, calls reach any registered function
    whose declared sorts fit the position, and a Boolean node can be an
    ``==``/``!=`` between two Boolean subtrees. The tree is never
    evaluated, so an unrestricted (possibly zero) divisor is fine.

    Args:
        identifiers: Pool identifiers may be drawn from as leaves.
        max_leaves: Most literal and identifier leaves the tree may hold.

    Returns:
        A strategy drawing a sort-correct :class:`Expression`.

    """
    return st.one_of(
        _build_structural_numeric_strategy(identifiers, max_leaves),
        _build_structural_boolean_strategy(identifiers, max_leaves),
    )
