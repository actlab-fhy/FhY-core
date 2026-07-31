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

from hypothesis import given, settings  # type: ignore[import-not-found]
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


def _extend_with_piecewise(
    children: st.SearchStrategy[Expression],
) -> st.SearchStrategy[PiecewiseExpression]:
    """Build a strategy for a piecewise node whose parts are drawn from ``children``."""
    cases = st.lists(st.tuples(children, children), min_size=1, max_size=3)

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


@settings(max_examples=50, deadline=None)  # type: ignore[untyped-decorator]
@given(_random_piecewise_expressions())  # type: ignore[untyped-decorator]
def test_random_piecewise_tree_round_trips_through_dict_serialization(
    expression: PiecewiseExpression,
) -> None:
    """Test any generated piecewise tree survives a dict round trip."""
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert restored.is_structurally_equivalent(expression)


# =============================================================================
# NumPy lowering: first-match-wins matches a pointwise Python fold
# =============================================================================


_FLOAT_KWARGS = {
    "min_value": -1000.0,
    "max_value": 1000.0,
    "allow_nan": False,
    "allow_infinity": False,
}


@settings(max_examples=50, deadline=None)  # type: ignore[untyped-decorator]
@given(  # type: ignore[untyped-decorator]
    cases=st.lists(
        st.tuples(
            st.floats(**_FLOAT_KWARGS),
            st.integers(min_value=-1000, max_value=1000),
        ),
        min_size=1,
        max_size=4,
    ),
    otherwise_value=st.integers(min_value=-1000, max_value=1000),
    sample_values=st.lists(st.floats(**_FLOAT_KWARGS), min_size=1, max_size=20),
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
# SymPy round trip: case count is preserved
# =============================================================================


@settings(max_examples=50, deadline=None)  # type: ignore[untyped-decorator]
@given(  # type: ignore[untyped-decorator]
    num_cases=st.integers(min_value=1, max_value=4),
    values=st.lists(
        st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=4
    ),
    otherwise_value=st.integers(min_value=-1000, max_value=1000),
)
def test_sympy_round_trip_preserves_case_count(
    num_cases: int, values: list[int], otherwise_value: int
) -> None:
    """Test lowering then lifting through SymPy preserves the original case count."""
    values = (values * num_cases)[:num_cases]
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
