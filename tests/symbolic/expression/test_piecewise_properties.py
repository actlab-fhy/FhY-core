"""Hypothesis property tests for ``PiecewiseExpression``.

Covers three invariants that must hold for arbitrary piecewise trees:
serialization round-trips under structural equivalence, the NumPy
lowering's first-match-wins selection matches a pointwise Python fold,
and the SymPy lowering/lifting round trip preserves case count.
"""

from collections.abc import Sequence

import pytest

# Hypothesis is only in the `property` dependency group, not `test`; the
# `tests` lane (CI's `tests` job, `nox -s tests`) syncs only `test`, so this
# module must be import-skippable there instead of failing collection.
pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression_with_numpy,
    piecewise,
)

from .conftest import mock_identifier

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")


# =============================================================================
# Random piecewise-tree generation
# =============================================================================


def _leaf_expressions() -> st.SearchStrategy[Expression]:
    """Return a strategy for scalar leaf expressions (int or bool literals)."""
    return st.one_of(
        st.integers(min_value=-1000, max_value=1000).map(LiteralExpression),
        st.booleans().map(LiteralExpression),
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


def _condition_expressions(
    children: st.SearchStrategy[Expression],
) -> st.SearchStrategy[Expression]:
    """Return ``children``, mapped so every draw is valid as a piecewise condition."""
    return children.map(_coerce_to_valid_condition)


def _extend_with_piecewise(
    children: st.SearchStrategy[Expression],
) -> st.SearchStrategy[PiecewiseExpression]:
    """Build a strategy for a piecewise node whose parts are drawn from ``children``."""
    condition_children = _condition_expressions(children)
    cases = st.lists(st.tuples(condition_children, children), min_size=1, max_size=3)

    def _build(
        case_list: Sequence[tuple[Expression, Expression]], otherwise: Expression
    ) -> PiecewiseExpression:
        conditions = tuple(condition for condition, _ in case_list)
        values = tuple(value for _, value in case_list)
        return PiecewiseExpression(conditions, values, otherwise)

    return st.builds(_build, cases, children)


def _random_expressions() -> st.SearchStrategy[Expression]:
    """Return a strategy for expressions, possibly nesting piecewise trees."""
    return st.recursive(_leaf_expressions(), _extend_with_piecewise, max_leaves=8)


def _random_piecewise_expressions() -> st.SearchStrategy[PiecewiseExpression]:
    """Return a strategy whose top-level result is always a ``PiecewiseExpression``."""
    return _extend_with_piecewise(_random_expressions())


# =============================================================================
# Serialization: round trip under structural equivalence
# =============================================================================


@settings(max_examples=50, deadline=None)
@given(_random_piecewise_expressions())
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


@settings(max_examples=50, deadline=None)
@given(data=st.data(), num_cases=st.integers(min_value=1, max_value=4))
def test_sympy_round_trip_preserves_the_whole_piecewise(
    data: st.DataObject, num_cases: int
) -> None:
    """Test lowering then lifting through SymPy reconstructs an equivalent node.

    Asserting only the case count would pass for a bridge that reordered
    the cases or paired a value with the wrong condition, so the values
    are drawn distinct and the restored node is compared structurally.
    """
    values = data.draw(
        st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=num_cases,
            max_size=num_cases,
            unique=True,
        )
    )
    otherwise_value = data.draw(
        st.integers(min_value=-1000, max_value=1000).filter(
            lambda candidate: candidate not in values
        )
    )
    conditions = tuple(
        IdentifierExpression(mock_identifier(f"property_case_{i}", i))
        for i in range(num_cases)
    )
    value_expressions = tuple(LiteralExpression(value) for value in values)
    expression = PiecewiseExpression(
        conditions, value_expressions, LiteralExpression(otherwise_value)
    )

    sympy_expression = convert_expression_to_sympy_expression(expression)
    restored = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(restored, PiecewiseExpression)
    assert len(restored.conditions) == num_cases
    assert restored.is_structurally_equivalent(expression)
