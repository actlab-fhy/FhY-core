"""Hypothesis property tests for ``PiecewiseExpression``.

Covers three invariants that must hold for arbitrary piecewise trees:
serialization round-trips under structural equivalence, the NumPy
lowering's first-match-wins selection matches a pointwise Python fold,
and the SymPy lowering/lifting round trip reconstructs the whole node.
"""

import pytest

# Hypothesis is only in the `property` dependency group, not `test`; the
# `tests` lane (CI's `tests` job, `nox -s tests`) syncs only `test`, so this
# module must be import-skippable there instead of failing collection.
pytest.importorskip("hypothesis")

from typing import Final

from hypothesis import example, given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression_with_numpy,
    piecewise,
)

from ...strategies.literals import (
    build_boolean_literal_strategy,
    build_integer_literal_strategy,
)
from .conftest import mock_identifier

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")


# =============================================================================
# Random piecewise-tree generation
# =============================================================================

_MAX_TREE_LEAVES = 24
"""Most leaves a generated tree may hold, spent as the tree is built."""

_MAX_CASES = 3
"""Most cases a generated piecewise node holds."""

_MIN_PIECEWISE_LEAVES = 3
"""Leaves the smallest piecewise node needs: a condition, a value, ``otherwise``."""


def _leaf_expressions() -> st.SearchStrategy[Expression]:
    """Return a strategy for scalar leaf expressions (int or bool literals)."""
    return st.one_of(
        build_integer_literal_strategy(min_value=-1000, max_value=1000),
        build_boolean_literal_strategy(),
    )


def _coerce_to_valid_condition(expression: Expression) -> Expression:
    """Coerce a drawn expression into one valid as a piecewise condition.

    A literal condition must be boolean-valued, so a non-boolean literal
    maps to an equivalent boolean literal (nonzero is truthy); a
    non-literal expression (a nested piecewise node, here) carries no
    such restriction and passes through unchanged. Mapping rather than
    filtering keeps every draw usable, so the strategy never relies on
    rejection sampling.
    """
    if isinstance(expression, LiteralExpression) and not isinstance(
        expression.value, bool
    ):
        return LiteralExpression(bool(expression.value))
    return expression


def _count_leaves(expression: Expression) -> int:
    """Return how many leaves ``expression`` has; a non-piecewise node is one."""
    if isinstance(expression, PiecewiseExpression):
        parts = (*expression.conditions, *expression.values, expression.otherwise)
        return sum(_count_leaves(part) for part in parts)
    return 1


@st.composite
def _draw_expression(draw: st.DrawFn, max_leaves: int) -> Expression:
    """Draw a leaf or, when ``max_leaves`` affords one, a piecewise node."""
    if max_leaves >= _MIN_PIECEWISE_LEAVES and draw(st.booleans()):
        return draw(_draw_piecewise(max_leaves))
    return draw(_leaf_expressions())


@st.composite
def _draw_piecewise(
    draw: st.DrawFn, max_leaves: int = _MAX_TREE_LEAVES
) -> PiecewiseExpression:
    """Draw a piecewise node whose whole tree has at most ``max_leaves`` leaves.

    The bound is kept by construction rather than by rejection. Each part
    is drawn under a budget taken from the leaves still unspent, less one
    held back for every part not yet drawn, and whatever a part leaves
    unused passes on to the parts after it, so no draw is ever discarded
    for being too large. Drawing each part's budget, rather than handing
    it every leaf left, spreads the trees across the whole range of sizes
    instead of piling them up at the bound.
    """
    num_cases = draw(
        st.integers(min_value=1, max_value=min(_MAX_CASES, (max_leaves - 1) // 2))
    )
    num_parts = 2 * num_cases + 1
    parts: list[Expression] = []
    unspent = max_leaves
    for index in range(num_parts):
        available = unspent - (num_parts - index - 1)
        budget = draw(st.integers(min_value=1, max_value=available))
        part = draw(_draw_expression(budget))
        unspent -= _count_leaves(part)
        parts.append(part)
    conditions = tuple(_coerce_to_valid_condition(part) for part in parts[0:-1:2])
    return PiecewiseExpression(conditions, tuple(parts[1:-1:2]), parts[-1])


# =============================================================================
# Serialization: round trip under structural equivalence
# =============================================================================


@settings(max_examples=50, deadline=None)
@given(_draw_piecewise())
def test_random_piecewise_tree_round_trips_through_dict_serialization(
    expression: PiecewiseExpression,
) -> None:
    """Test any generated piecewise tree survives a dict round trip."""
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert restored.is_structurally_equivalent(expression)


# =============================================================================
# NumPy lowering: first-match-wins matches a pointwise Python fold
# =============================================================================


def _create_bounded_float_strategy() -> st.SearchStrategy[float]:
    """Return a strategy for finite floats bounded to [-1000.0, 1000.0]."""
    return st.floats(
        min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
    )


@settings(max_examples=50, deadline=None)
@given(
    cases=st.lists(
        st.tuples(
            _create_bounded_float_strategy(),
            st.integers(min_value=-1000, max_value=1000),
        ),
        min_size=1,
        max_size=4,
    ),
    otherwise_value=st.integers(min_value=-1000, max_value=1000),
    sample_values=st.lists(_create_bounded_float_strategy(), min_size=1, max_size=20),
)
def test_numpy_evaluation_matches_pointwise_first_match_fold(
    cases: list[tuple[float, int]],
    otherwise_value: int,
    sample_values: list[float],
) -> None:
    """Test the NumPy lowering matches an independent per-element first-match fold.

    Each case's condition is ``x > threshold``; overlapping thresholds are
    common (a large ``x`` can satisfy every threshold), so first-match-wins
    is genuinely exercised, not just the degenerate single-case path.
    """
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = piecewise(
        *((x_expression > threshold, value) for threshold, value in cases),
        otherwise=otherwise_value,
    )
    xs = np.array(sample_values)

    result = evaluate_expression_with_numpy(expression, {x: xs})

    def _reference_fold(sample: float) -> int:
        for threshold, value in cases:
            if sample > threshold:
                return value
        return otherwise_value

    expected = np.array([_reference_fold(sample) for sample in sample_values])
    assert np.array_equal(result, expected)


# =============================================================================
# SymPy round trip: the whole node is preserved
# =============================================================================


@st.composite
def _draw_distinct_integers(draw: st.DrawFn, count: int) -> list[int]:
    """Draw ``count`` distinct integers from ``[-1000, 1000]`` without rejection.

    Each value is drawn as a position among the integers not yet taken
    and mapped onto that integer, so a repeat cannot be drawn and nothing
    is filtered out. A unique list would instead redraw every duplicate
    and abandon the example after too many, which Hypothesis's leaning
    toward small, repeated integers makes common.
    """
    taken: list[int] = []
    for already_taken in range(count):
        value = draw(st.integers(min_value=-1000, max_value=1000 - already_taken))
        for previous in sorted(taken):
            if value >= previous:
                value += 1
        taken.append(value)
    return taken


@st.composite
def _draw_piecewise_with_distinct_case_values(draw: st.DrawFn) -> PiecewiseExpression:
    """Draw a piecewise node with distinct integer case values and ``otherwise``.

    Every case value and ``otherwise`` come from one draw of distinct
    integers, with ``otherwise`` split off its end, so no draw is
    discarded for colliding with another value; a lowering/lifting bug
    that reordered the cases or paired a value with the wrong condition
    -- not just dropped one -- is still caught by comparing the whole
    restored tree structurally.
    """
    num_cases = draw(st.integers(min_value=1, max_value=4))
    *values, otherwise_value = draw(_draw_distinct_integers(num_cases + 1))
    conditions = tuple(
        IdentifierExpression(mock_identifier(f"property_case_{i}", i))
        for i in range(num_cases)
    )
    value_expressions = tuple(LiteralExpression(value) for value in values)
    return PiecewiseExpression(
        conditions, value_expressions, LiteralExpression(otherwise_value)
    )


# Welded from test_sympy_pass.py, deleted there in the same change: both
# use a comparison (rather than a bare identifier) as a case condition,
# which _draw_piecewise_with_distinct_case_values never draws.
_COMPARISON_CONDITION_IDENTIFIER: Final = mock_identifier("x", 0)
_COMPARISON_CONDITION_SYMBOL: Final = IdentifierExpression(
    _COMPARISON_CONDITION_IDENTIFIER
)
# Welded from
# test_sympy_pass.py::test_single_case_piecewise_expression_round_trips_through_sympy:
# both operands are leaves whose SymPy lowerings preserve their shape.
_SINGLE_COMPARISON_CASE_PIECEWISE: Final = PiecewiseExpression(
    (
        BinaryExpression(
            BinaryOperation.GREATER, _COMPARISON_CONDITION_SYMBOL, LiteralExpression(0)
        ),
    ),
    (_COMPARISON_CONDITION_SYMBOL,),
    LiteralExpression(0),
)
# Welded from test_sympy_pass.py::
# test_multi_case_piecewise_expression_round_trips_with_full_order_and_content:
# a 3-case chain of comparisons with distinguishable literal values.
_MULTI_COMPARISON_CASE_PIECEWISE: Final = PiecewiseExpression(
    (
        _COMPARISON_CONDITION_SYMBOL > 0,
        _COMPARISON_CONDITION_SYMBOL > 10,
        _COMPARISON_CONDITION_SYMBOL > 20,
    ),
    (LiteralExpression(10), LiteralExpression(20), LiteralExpression(30)),
    LiteralExpression(99),
)


@settings(max_examples=50)
@example(expression=_SINGLE_COMPARISON_CASE_PIECEWISE)
@example(expression=_MULTI_COMPARISON_CASE_PIECEWISE)
@given(expression=_draw_piecewise_with_distinct_case_values())
def test_sympy_round_trip_preserves_the_whole_piecewise(
    expression: PiecewiseExpression,
) -> None:
    """Test lowering then lifting through SymPy reconstructs an equivalent node.

    Asserting only the case count would pass for a bridge that reordered
    the cases or paired a value with the wrong condition, so the restored
    node is compared against the original structurally.
    """
    sympy_expression = convert_expression_to_sympy_expression(expression)
    restored = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(restored, PiecewiseExpression)
    assert len(restored.conditions) == len(expression.conditions)
    assert restored.is_structurally_equivalent(expression)
