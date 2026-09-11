"""Hypothesis strategies for sort-aware, leaf-budget-bounded expression trees.

Every strategy in this module builds its leaf budget by construction,
the way ``tests/symbolic/expression/test_piecewise_properties.py`` does:
each part of a multi-part node draws its own budget from the leaves
still unspent, less one held back for every part not yet drawn, and the
leaves a part actually uses (not its allocated budget) are subtracted
from what remains, so unused leaves roll over and no draw is ever
discarded for being too large.

Two node kinds spend no leaf of their own (``UnaryExpression`` and a
single-argument ``CallExpression``): recursing into either with the same
leaf budget would not shrink it, so termination is bounded separately
with a "wraps remaining" counter that decrements on those two kinds
alone and forces a leaf, or a leaf-spending node, once exhausted.

The numeric and boolean gate strategies are sort-aware by construction:
a numeric operand never reaches a Boolean position, so no draw is ever
refused by ``validate_logical_operands``. They are mutually recursive
(a numeric piecewise's conditions are boolean; a boolean comparison's
operands are numeric); each cross-sort reference goes through
``hypothesis.strategies.deferred`` so either side can refer to the other.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    make_binary_expression,
    make_unary_expression,
    piecewise,
)

from .identifiers import build_identifier_strategy
from .literals import (
    build_boolean_literal_strategy,
    build_integer_literal_strategy,
)

__all__ = [
    "COMPARISON_OPERATIONS",
    "INTEGER_RESULT_NATIVE_FUNCTIONS",
    "LOGICAL_BINARY_OPERATIONS",
    "NUMERIC_DIVISION_OPERATIONS",
    "NUMERIC_GATE_OPERATIONS",
    "SYMPY_STABLE_CALL_FUNCTIONS",
    "build_any_sort_expression_strategy",
    "build_boolean_expression_strategy",
    "build_integer_environment_strategy",
    "build_numeric_expression_strategy",
    "build_sympy_stable_expression_strategy",
    "count_expression_leaves",
    "draw_boolean_tree_with_environment",
    "draw_numeric_tree_with_environment",
    "evaluate_with_python",
]

NUMERIC_GATE_OPERATIONS: Final = (
    BinaryOperation.ADD,
    BinaryOperation.SUBTRACT,
    BinaryOperation.MULTIPLY,
)
NUMERIC_DIVISION_OPERATIONS: Final = (
    BinaryOperation.FLOOR_DIVIDE,
    BinaryOperation.MODULO,
)
COMPARISON_OPERATIONS: Final = (
    BinaryOperation.EQUAL,
    BinaryOperation.NOT_EQUAL,
    BinaryOperation.LESS,
    BinaryOperation.LESS_EQUAL,
    BinaryOperation.GREATER,
    BinaryOperation.GREATER_EQUAL,
)
LOGICAL_BINARY_OPERATIONS: Final = (
    BinaryOperation.LOGICAL_AND,
    BinaryOperation.LOGICAL_OR,
)
INTEGER_RESULT_NATIVE_FUNCTIONS: Final = ("floor", "ceil", "round")

_NONZERO_DIVISORS: Final = (*range(-8, 0), *range(1, 9))
_MIN_LEAVES_FOR_TWO_CHILDREN: Final = 2
_MIN_PIECEWISE_LEAVES: Final = 3
_MAX_PIECEWISE_CASES: Final = 3
_MAX_CONSECUTIVE_WRAPS: Final = 4
"""Consecutive zero-leaf-cost wraps (unary op, single-argument call, logical
not) allowed before a draw is forced to pick a leaf-spending kind. Bounds
recursion depth independently of the leaf budget, since these node kinds
do not shrink it."""

SYMPY_STABLE_CALL_FUNCTIONS: Final = ("floor", "ceil", "round")
"""Natives whose call on an identifier round-trips through SymPy unchanged.

Every member of `INTEGER_RESULT_NATIVE_FUNCTIONS` qualifies: each lowers
to a SymPy node the lifting pass maps back to the same native name.
The list is explicit on purpose: a probe that dropped whatever failed
would hide a regression in the bridge instead of surfacing it.
"""


def count_expression_leaves(expression: Expression) -> int:
    """Return how many literal or identifier leaves ``expression`` holds.

    Generalizes over every node kind through
    :meth:`~fhy_core.symbolic.expression.Expression.get_visit_children`:
    a node with no children (a literal or an identifier) is one leaf; any
    other node's leaf count is the sum of its children's.

    Args:
        expression: Expression to measure.

    Returns:
        Total literal and identifier leaves in the tree.

    """
    children = expression.get_visit_children()
    if not children:
        return 1
    return sum(count_expression_leaves(child) for child in children)


def _draw_parts_within_leaf_budget(
    draw: st.DrawFn,
    build_part_strategies: Sequence[Callable[[int], st.SearchStrategy[Expression]]],
    max_leaves: int,
) -> tuple[Expression, ...]:
    """Draw one expression per factory, spending at most ``max_leaves`` leaves total.

    Args:
        draw: The active Hypothesis draw function.
        build_part_strategies: One budget-to-strategy factory per part, in
            the order the parts are drawn.
        max_leaves: Total leaves every drawn part may spend together.
            Must be at least ``len(build_part_strategies)``.

    Returns:
        One expression per factory, in order.

    """
    num_parts = len(build_part_strategies)
    parts: list[Expression] = []
    unspent = max_leaves
    for index, build_part_strategy in enumerate(build_part_strategies):
        available = unspent - (num_parts - index - 1)
        budget = draw(st.integers(min_value=1, max_value=available))
        part = draw(build_part_strategy(budget))
        unspent -= count_expression_leaves(part)
        parts.append(part)
    return tuple(parts)


def _draw_piecewise_expression(
    draw: st.DrawFn,
    max_leaves: int,
    build_condition_strategy: Callable[[int], st.SearchStrategy[Expression]],
    build_value_strategy: Callable[[int], st.SearchStrategy[Expression]],
) -> PiecewiseExpression:
    """Draw a piecewise node spending at most ``max_leaves`` leaves over all its parts.

    The case count is capped by how many (condition, value) pairs plus a
    final ``otherwise`` fit in the budget; every part then draws its own
    share through :func:`_draw_parts_within_leaf_budget`.
    """
    max_cases = min(_MAX_PIECEWISE_CASES, (max_leaves - 1) // 2)
    num_cases = draw(st.integers(min_value=1, max_value=max_cases))
    part_factories: list[Callable[[int], st.SearchStrategy[Expression]]] = []
    for _ in range(num_cases):
        part_factories.append(build_condition_strategy)
        part_factories.append(build_value_strategy)
    part_factories.append(build_value_strategy)
    parts = _draw_parts_within_leaf_budget(draw, part_factories, max_leaves)
    conditions = parts[0:-1:2]
    values = parts[1:-1:2]
    otherwise = parts[-1]
    cases = tuple(zip(conditions, values, strict=True))
    return piecewise(*cases, otherwise=otherwise)


def _build_numeric_leaf_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a numeric leaf: an integer literal or a pool identifier."""
    leaves: list[st.SearchStrategy[Expression]] = [build_integer_literal_strategy()]
    if identifiers:
        leaves.append(build_identifier_strategy(identifiers).map(IdentifierExpression))
    return st.one_of(*leaves)


def _build_nonzero_divisor_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy for a non-zero integer divisor literal in [-8,-1] | [1,8]."""
    return st.sampled_from(_NONZERO_DIVISORS).map(LiteralExpression)


@st.composite
def _draw_numeric_expression(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
    include_division: bool,
    include_calls: bool,
    native_functions: Sequence[str],
    include_piecewise: bool,
    wraps_remaining: int,
) -> Expression:
    """Draw a numeric-sorted expression tree spending at most ``max_leaves`` leaves."""
    kinds: list[str] = ["leaf"]
    if wraps_remaining > 0:
        kinds.append("unary")
        if include_calls and native_functions:
            kinds.append("call")
    if max_leaves >= _MIN_LEAVES_FOR_TWO_CHILDREN:
        kinds.append("binary")
        if include_division:
            kinds.append("division")
    if include_piecewise and max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "leaf":
        return draw(_build_numeric_leaf_strategy(identifiers))
    if kind == "unary":
        unary_operation = draw(
            st.sampled_from((UnaryOperation.NEGATE, UnaryOperation.POSITIVE))
        )
        operand = draw(
            _draw_numeric_expression(
                identifiers,
                max_leaves,
                include_division,
                include_calls,
                native_functions,
                include_piecewise,
                wraps_remaining - 1,
            )
        )
        return make_unary_expression(unary_operation, operand)
    if kind == "call":
        function_name = draw(st.sampled_from(native_functions))
        argument = draw(
            _draw_numeric_expression(
                identifiers,
                max_leaves,
                include_division,
                include_calls,
                native_functions,
                include_piecewise,
                wraps_remaining - 1,
            )
        )
        return call(function_name, argument)
    if kind == "binary":
        binary_operation = draw(st.sampled_from(NUMERIC_GATE_OPERATIONS))
        left, right = _draw_parts_within_leaf_budget(
            draw,
            (
                lambda budget: build_numeric_expression_strategy(
                    identifiers,
                    budget,
                    include_division=include_division,
                    include_calls=include_calls,
                    native_functions=native_functions,
                    include_piecewise=include_piecewise,
                ),
                lambda budget: build_numeric_expression_strategy(
                    identifiers,
                    budget,
                    include_division=include_division,
                    include_calls=include_calls,
                    native_functions=native_functions,
                    include_piecewise=include_piecewise,
                ),
            ),
            max_leaves,
        )
        return make_binary_expression(binary_operation, left, right)
    if kind == "division":
        division_operation = draw(st.sampled_from(NUMERIC_DIVISION_OPERATIONS))
        dividend, divisor = _draw_parts_within_leaf_budget(
            draw,
            (
                lambda budget: build_numeric_expression_strategy(
                    identifiers,
                    budget,
                    include_division=include_division,
                    include_calls=include_calls,
                    native_functions=native_functions,
                    include_piecewise=include_piecewise,
                ),
                lambda _budget: _build_nonzero_divisor_strategy(),
            ),
            max_leaves,
        )
        return make_binary_expression(division_operation, dividend, divisor)
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda budget: st.deferred(
            lambda: build_boolean_expression_strategy(
                identifiers,
                budget,
                include_calls=include_calls,
                include_division=include_division,
                native_functions=native_functions,
                include_piecewise=include_piecewise,
            )
        ),
        lambda budget: build_numeric_expression_strategy(
            identifiers,
            budget,
            include_division=include_division,
            include_calls=include_calls,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        ),
    )


@st.composite
def _draw_boolean_expression(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
    include_calls: bool,
    include_division: bool,
    native_functions: Sequence[str],
    include_piecewise: bool,
    wraps_remaining: int,
) -> Expression:
    """Draw a boolean-sorted expression tree spending at most ``max_leaves`` leaves."""
    kinds: list[str] = ["bool_literal"]
    if max_leaves >= _MIN_LEAVES_FOR_TWO_CHILDREN:
        kinds.append("comparison")
        kinds.append("and_or")
    if wraps_remaining > 0:
        kinds.append("not")
    if include_piecewise and max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "bool_literal":
        return draw(build_boolean_literal_strategy())
    if kind == "comparison":
        operation = draw(st.sampled_from(COMPARISON_OPERATIONS))
        left, right = _draw_parts_within_leaf_budget(
            draw,
            (
                lambda budget: st.deferred(
                    lambda: build_numeric_expression_strategy(
                        identifiers,
                        budget,
                        include_division=include_division,
                        include_calls=include_calls,
                        native_functions=native_functions,
                        include_piecewise=include_piecewise,
                    )
                ),
                lambda budget: st.deferred(
                    lambda: build_numeric_expression_strategy(
                        identifiers,
                        budget,
                        include_division=include_division,
                        include_calls=include_calls,
                        native_functions=native_functions,
                        include_piecewise=include_piecewise,
                    )
                ),
            ),
            max_leaves,
        )
        return make_binary_expression(operation, left, right)
    if kind == "not":
        operand = draw(
            _draw_boolean_expression(
                identifiers,
                max_leaves,
                include_calls,
                include_division,
                native_functions,
                include_piecewise,
                wraps_remaining - 1,
            )
        )
        return make_unary_expression(UnaryOperation.LOGICAL_NOT, operand)
    if kind == "and_or":
        operation = draw(st.sampled_from(LOGICAL_BINARY_OPERATIONS))
        left, right = _draw_parts_within_leaf_budget(
            draw,
            (
                lambda budget: build_boolean_expression_strategy(
                    identifiers,
                    budget,
                    include_calls=include_calls,
                    include_division=include_division,
                    native_functions=native_functions,
                    include_piecewise=include_piecewise,
                ),
                lambda budget: build_boolean_expression_strategy(
                    identifiers,
                    budget,
                    include_calls=include_calls,
                    include_division=include_division,
                    native_functions=native_functions,
                    include_piecewise=include_piecewise,
                ),
            ),
            max_leaves,
        )
        return make_binary_expression(operation, left, right)
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda budget: build_boolean_expression_strategy(
            identifiers,
            budget,
            include_calls=include_calls,
            include_division=include_division,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        ),
        lambda budget: build_boolean_expression_strategy(
            identifiers,
            budget,
            include_calls=include_calls,
            include_division=include_division,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        ),
    )


def build_numeric_expression_strategy(
    identifiers: Sequence[Identifier],
    max_leaves: int = 8,
    *,
    include_division: bool = True,
    include_calls: bool = True,
    native_functions: Sequence[str] = INTEGER_RESULT_NATIVE_FUNCTIONS,
    include_piecewise: bool = True,
) -> st.SearchStrategy[Expression]:
    """Return a strategy for numeric-sorted expression trees within a leaf budget.

    Every option is forwarded to every numeric and boolean subtree the
    tree draws (binary and division operands, call arguments, and both
    the condition and value parts of a nested piecewise), so one setting
    governs the whole tree, not just its root.

    Args:
        identifiers: Pool identifiers may be drawn from as leaves.
        max_leaves: Most literal and identifier leaves the tree may hold.
        include_division: Whether ``FLOOR_DIVIDE``/``MODULO`` nodes (with
            a non-zero literal divisor) may appear.
        include_calls: Whether a call node may appear at all. A call
            always targets one of ``native_functions``; false or an
            empty ``native_functions`` both mean no call node is drawn.
        native_functions: The only function names a call node may
            target. Defaults to :data:`INTEGER_RESULT_NATIVE_FUNCTIONS`.
        include_piecewise: Whether piecewise nodes may appear.

    Returns:
        A strategy drawing a numeric-sorted :class:`Expression`.

    """
    return _draw_numeric_expression(
        identifiers,
        max_leaves,
        include_division,
        include_calls,
        native_functions,
        include_piecewise,
        _MAX_CONSECUTIVE_WRAPS,
    )


def build_boolean_expression_strategy(
    identifiers: Sequence[Identifier],
    max_leaves: int = 8,
    *,
    include_calls: bool = True,
    include_division: bool = True,
    native_functions: Sequence[str] = INTEGER_RESULT_NATIVE_FUNCTIONS,
    include_piecewise: bool = True,
) -> st.SearchStrategy[Expression]:
    """Return a strategy for boolean-sorted expression trees within a leaf budget.

    ``include_calls``, ``include_division``, and ``native_functions`` are
    forwarded to every numeric subtree this strategy draws (a
    comparison's two operands, and any numeric piecewise value nested
    under a boolean piecewise condition), so they govern the whole tree,
    not only its own boolean nodes.

    Args:
        identifiers: Pool identifiers a nested numeric comparison operand
            may be drawn from.
        max_leaves: Most literal and identifier leaves the tree may hold.
        include_calls: Whether a numeric comparison operand may hold a
            call node. Forwarded to every numeric subtree.
        include_division: Whether a numeric comparison operand may hold
            a division node. Forwarded to every numeric subtree.
        native_functions: The only function names a numeric comparison
            operand's call node may target.
        include_piecewise: Whether piecewise nodes may appear.

    Returns:
        A strategy drawing a boolean-sorted :class:`Expression`.

    """
    return _draw_boolean_expression(
        identifiers,
        max_leaves,
        include_calls,
        include_division,
        native_functions,
        include_piecewise,
        _MAX_CONSECUTIVE_WRAPS,
    )


def build_any_sort_expression_strategy(
    identifiers: Sequence[Identifier],
    max_leaves: int = 8,
    *,
    include_division: bool = True,
    include_calls: bool = True,
    native_functions: Sequence[str] = INTEGER_RESULT_NATIVE_FUNCTIONS,
    include_piecewise: bool = True,
) -> st.SearchStrategy[Expression]:
    """Return a strategy drawing either a numeric-sorted or a boolean-sorted tree.

    Every keyword is forwarded to both the numeric and the boolean
    strategy; see their docstrings for what each option controls.
    """
    return st.one_of(
        build_numeric_expression_strategy(
            identifiers,
            max_leaves,
            include_division=include_division,
            include_calls=include_calls,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        ),
        build_boolean_expression_strategy(
            identifiers,
            max_leaves,
            include_calls=include_calls,
            include_division=include_division,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        ),
    )


def _build_sympy_stable_numeric_leaf_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a numeric leaf: an integer literal or a pool identifier."""
    leaves: list[st.SearchStrategy[Expression]] = [build_integer_literal_strategy()]
    if identifiers:
        leaves.append(build_identifier_strategy(identifiers).map(IdentifierExpression))
    return st.one_of(*leaves)


def _build_sympy_stable_boolean_leaf_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a boolean leaf: a boolean literal or a pool identifier."""
    leaves: list[st.SearchStrategy[Expression]] = [build_boolean_literal_strategy()]
    if identifiers:
        leaves.append(build_identifier_strategy(identifiers).map(IdentifierExpression))
    return st.one_of(*leaves)


def _build_sympy_stable_condition_strategy(
    identifiers: Sequence[Identifier],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a piecewise condition that survives a SymPy round trip.

    Always a bare identifier, never a literal: SymPy's ``Piecewise``
    decides a constant-valued condition eagerly and drops the case (or
    the whole node) on the way through, which would restore something
    other than the original ``PiecewiseExpression``. Requires a
    non-empty ``identifiers`` pool; callers only offer the "piecewise"
    kind when the pool is non-empty.
    """
    return build_identifier_strategy(identifiers).map(IdentifierExpression)


@st.composite
def _draw_sympy_stable_numeric_expression(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int,
) -> Expression:
    """Draw a numeric tree whose SymPy lowering then lifting is structurally stable."""
    kinds: list[str] = ["leaf"]
    if identifiers and SYMPY_STABLE_CALL_FUNCTIONS:
        kinds.append("call")
    if identifiers and max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "leaf":
        return draw(_build_sympy_stable_numeric_leaf_strategy(identifiers))
    if kind == "call":
        # The argument is always a bare identifier, never a literal: SymPy
        # folds a native call on a concrete literal eagerly (`floor(3)`
        # simplifies to `3` on the way through), which would restore a
        # `LiteralExpression` instead of the original `CallExpression`.
        function_name = draw(st.sampled_from(SYMPY_STABLE_CALL_FUNCTIONS))
        identifier = draw(build_identifier_strategy(identifiers))
        return call(function_name, IdentifierExpression(identifier))
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda _budget: _build_sympy_stable_condition_strategy(identifiers),
        lambda budget: _build_sympy_stable_numeric_strategy(identifiers, budget),
    )


@st.composite
def _draw_sympy_stable_boolean_expression(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_leaves: int
) -> Expression:
    """Draw a boolean tree whose SymPy lowering then lifting is structurally stable."""
    kinds: list[str] = ["leaf"]
    if identifiers and max_leaves >= _MIN_PIECEWISE_LEAVES:
        kinds.append("piecewise")
    kind = draw(st.sampled_from(kinds))

    if kind == "leaf":
        return draw(_build_sympy_stable_boolean_leaf_strategy(identifiers))
    return _draw_piecewise_expression(
        draw,
        max_leaves,
        lambda _budget: _build_sympy_stable_condition_strategy(identifiers),
        lambda budget: _build_sympy_stable_boolean_strategy(identifiers, budget),
    )


def _build_sympy_stable_numeric_strategy(
    identifiers: Sequence[Identifier], max_leaves: int
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a SymPy-stable numeric-sorted tree within a leaf budget."""
    return _draw_sympy_stable_numeric_expression(identifiers, max_leaves)


def _build_sympy_stable_boolean_strategy(
    identifiers: Sequence[Identifier], max_leaves: int
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a SymPy-stable boolean-sorted tree within a leaf budget."""
    return _draw_sympy_stable_boolean_expression(identifiers, max_leaves)


def build_sympy_stable_expression_strategy(
    identifiers: Sequence[Identifier], max_leaves: int = 8
) -> st.SearchStrategy[Expression]:
    """Return a strategy for trees whose SymPy lowering then lifting is stable.

    Restricted to integer and boolean literals, identifiers, calls to
    natives, and piecewise over those: no arithmetic ``BinaryExpression``,
    since SymPy re-associates and folds those on the way through.

    Args:
        identifiers: Pool identifiers may be drawn from as leaves.
        max_leaves: Most literal and identifier leaves the tree may hold.

    Returns:
        A strategy drawing a SymPy-round-trip-stable :class:`Expression`.

    """
    return st.one_of(
        _build_sympy_stable_numeric_strategy(identifiers, max_leaves),
        _build_sympy_stable_boolean_strategy(identifiers, max_leaves),
    )


def build_integer_environment_strategy(
    identifiers: Sequence[Identifier], min_value: int = -16, max_value: int = 16
) -> st.SearchStrategy[dict[Identifier, int]]:
    """Return a strategy binding every identifier in ``identifiers`` to an int.

    Args:
        identifiers: Every identifier the returned environment binds.
        min_value: Least value a binding may take.
        max_value: Greatest value a binding may take.

    Returns:
        A strategy drawing a ``dict`` with one entry per identifier in
        ``identifiers``.

    """
    return st.fixed_dictionaries(
        {
            identifier: st.integers(min_value=min_value, max_value=max_value)
            for identifier in identifiers
        }
    )


@st.composite
def draw_numeric_tree_with_environment(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int = 8,
    *,
    include_division: bool = True,
    include_calls: bool = True,
    native_functions: Sequence[str] = INTEGER_RESULT_NATIVE_FUNCTIONS,
    include_piecewise: bool = True,
) -> tuple[Expression, dict[Identifier, int]]:
    """Draw a numeric gate-grammar tree together with bindings for every identifier.

    Every keyword is forwarded to :func:`build_numeric_expression_strategy`.
    """
    expression = draw(
        build_numeric_expression_strategy(
            identifiers,
            max_leaves,
            include_division=include_division,
            include_calls=include_calls,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        )
    )
    environment = draw(build_integer_environment_strategy(identifiers))
    return expression, environment


@st.composite
def draw_boolean_tree_with_environment(
    draw: st.DrawFn,
    identifiers: Sequence[Identifier],
    max_leaves: int = 8,
    *,
    include_calls: bool = True,
    include_division: bool = True,
    native_functions: Sequence[str] = INTEGER_RESULT_NATIVE_FUNCTIONS,
    include_piecewise: bool = True,
) -> tuple[Expression, dict[Identifier, int]]:
    """Draw a boolean gate-grammar tree together with bindings for every identifier.

    Every keyword is forwarded to :func:`build_boolean_expression_strategy`.
    """
    expression = draw(
        build_boolean_expression_strategy(
            identifiers,
            max_leaves,
            include_calls=include_calls,
            include_division=include_division,
            native_functions=native_functions,
            include_piecewise=include_piecewise,
        )
    )
    environment = draw(build_integer_environment_strategy(identifiers))
    return expression, environment


_PYTHON_BINARY_EVALUATORS: Final[
    dict[BinaryOperation, Callable[[int | bool, int | bool], "int | bool"]]
] = {
    BinaryOperation.ADD: lambda left, right: left + right,
    BinaryOperation.SUBTRACT: lambda left, right: left - right,
    BinaryOperation.MULTIPLY: lambda left, right: left * right,
    BinaryOperation.FLOOR_DIVIDE: lambda left, right: left // right,
    BinaryOperation.MODULO: lambda left, right: left % right,
    BinaryOperation.LOGICAL_AND: lambda left, right: bool(left) and bool(right),
    BinaryOperation.LOGICAL_OR: lambda left, right: bool(left) or bool(right),
    BinaryOperation.EQUAL: lambda left, right: left == right,
    BinaryOperation.NOT_EQUAL: lambda left, right: left != right,
    BinaryOperation.LESS: lambda left, right: left < right,
    BinaryOperation.LESS_EQUAL: lambda left, right: left <= right,
    BinaryOperation.GREATER: lambda left, right: left > right,
    BinaryOperation.GREATER_EQUAL: lambda left, right: left >= right,
}
"""Plain-Python implementation of every gate-grammar ``BinaryOperation``."""


def _evaluate_binary_with_python(
    operation: BinaryOperation, left: int | bool, right: int | bool
) -> "int | bool":
    """Evaluate one gate-grammar ``BinaryOperation`` with plain Python semantics."""
    evaluator = _PYTHON_BINARY_EVALUATORS.get(operation)
    if evaluator is None:
        raise NotImplementedError(
            f"evaluate_with_python does not support binary operation {operation!r}."
        )
    return evaluator(left, right)


def _evaluate_native_call_with_python(function_name: str, argument: int) -> int:
    """Evaluate a call to one of ``INTEGER_RESULT_NATIVE_FUNCTIONS`` on an int."""
    if function_name in ("floor", "ceil", "round"):
        return argument
    raise NotImplementedError(
        f"evaluate_with_python does not support call to {function_name!r}."
    )


def _evaluate_literal_with_python(expression: LiteralExpression) -> "int | bool":
    """Evaluate a gate-grammar ``LiteralExpression`` (int or bool valued only)."""
    value = expression.value
    if isinstance(value, (bool, int)):
        return value
    raise NotImplementedError(
        f"evaluate_with_python does not support literal value {value!r}."
    )


def _evaluate_call_with_python(
    expression: CallExpression, environment: Mapping[Identifier, int]
) -> int:
    """Evaluate a gate-grammar call: one native function on one int argument."""
    (argument_expression,) = expression.arguments
    argument = evaluate_with_python(argument_expression, environment)
    if not isinstance(argument, int):
        raise NotImplementedError(
            "evaluate_with_python only supports native calls on int arguments."
        )
    return _evaluate_native_call_with_python(expression.function_name, argument)


def _evaluate_piecewise_with_python(
    expression: PiecewiseExpression, environment: Mapping[Identifier, int]
) -> "int | bool":
    """Evaluate a gate-grammar piecewise: first true condition wins, else otherwise."""
    for condition, value_expression in expression.get_cases():
        if evaluate_with_python(condition, environment):
            return evaluate_with_python(value_expression, environment)
    return evaluate_with_python(expression.otherwise, environment)


def _evaluate_unary_with_python(
    expression: UnaryExpression, environment: Mapping[Identifier, int]
) -> "int | bool":
    """Evaluate a gate-grammar ``UnaryExpression`` with plain Python semantics."""
    operand = evaluate_with_python(expression.operand, environment)
    if expression.operation is UnaryOperation.NEGATE:
        return -operand
    if expression.operation is UnaryOperation.POSITIVE:
        return +operand
    if expression.operation is UnaryOperation.LOGICAL_NOT:
        return not operand
    raise NotImplementedError(
        f"evaluate_with_python does not support unary operation "
        f"{expression.operation!r}."
    )


def evaluate_with_python(
    expression: Expression, environment: Mapping[Identifier, int]
) -> "int | bool":
    """Evaluate a numeric or boolean gate-grammar expression with Python semantics.

    The test-side reference evaluator for gate trees only, used as the
    oracle ``evaluate_expression_with_numpy`` is checked against.
    Supports literals, identifiers, ``NEGATE``/``POSITIVE``/
    ``LOGICAL_NOT``, the arithmetic and logical ``BinaryOperation``s,
    comparisons, piecewise (first true condition wins, else
    ``otherwise``), and floor/ceil/round on ints (floor/ceil/round of an
    int is the int itself).

    Args:
        expression: A gate-grammar expression: no ``DIVIDE``, ``POWER``,
            or call other than to ``INTEGER_RESULT_NATIVE_FUNCTIONS``.
        environment: Values bound to every free identifier in
            ``expression``.

    Returns:
        The int or bool ``expression`` denotes under ``environment``.

    Raises:
        NotImplementedError: If ``expression`` holds a node, operation,
            or call outside the gate grammar this oracle supports.

    """
    if isinstance(expression, LiteralExpression):
        return _evaluate_literal_with_python(expression)
    if isinstance(expression, IdentifierExpression):
        return environment[expression.identifier]
    if isinstance(expression, UnaryExpression):
        return _evaluate_unary_with_python(expression, environment)
    if isinstance(expression, BinaryExpression):
        left = evaluate_with_python(expression.left, environment)
        right = evaluate_with_python(expression.right, environment)
        return _evaluate_binary_with_python(expression.operation, left, right)
    if isinstance(expression, CallExpression):
        return _evaluate_call_with_python(expression, environment)
    if isinstance(expression, PiecewiseExpression):
        return _evaluate_piecewise_with_python(expression, environment)
    raise NotImplementedError(
        f"evaluate_with_python does not support node type {type(expression).__name__}."
    )
