"""Hypothesis property tests for `Param.is_subset` and `Param.check_subset`.

Covers two shapes named in `Param.is_subset`'s own docstring: a finite-set
domain (ordinal, categorical) decides the relation by enumeration, and a
numeric domain (integer, natural, real, interval-integer) may defer to the
Z3 bridge.
"""

from typing import Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import ConstraintOutcome
from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_ordinal_param,
)

from ...strategies.params import (
    build_categorical_value_set_strategy,
    build_ordinal_value_set_strategy,
    draw_bounded_integer_param,
    draw_bounded_real_param,
    draw_interval_integer_param,
    draw_natural_param,
)

pytestmark = pytest.mark.property

# Every property runs without a hypothesis deadline. Every example takes
# about a millisecond and no draw is filtered, so a deadline here would time
# only the scheduler: on a contended machine one example can be descheduled
# for hundreds of milliseconds, failing a subset law for a reason unrelated
# to it. The `dev`/`thorough` profiles already set `deadline=None`.

_CANDIDATE_LIMIT: Final = 40


@st.composite
def draw_ordinal_subset_case(
    draw: st.DrawFn,
) -> tuple[Param[int], Param[int], frozenset[int]]:
    """Draw two ordinal params, plus the union of their declared value sets."""
    left_values = draw(build_ordinal_value_set_strategy())
    right_values = draw(build_ordinal_value_set_strategy())
    left = create_ordinal_param(left_values)
    right = create_ordinal_param(right_values)
    universe = frozenset(left_values) | frozenset(right_values)
    return left, right, universe


@st.composite
def draw_categorical_subset_case(
    draw: st.DrawFn,
) -> tuple[Param[str], Param[str], frozenset[str]]:
    """Draw two categorical params, plus the union of their declared value sets."""
    left_values = draw(build_categorical_value_set_strategy())
    right_values = draw(build_categorical_value_set_strategy())
    left = create_categorical_param(left_values)
    right = create_categorical_param(right_values)
    universe = frozenset(left_values) | frozenset(right_values)
    return left, right, universe


@st.composite
def draw_integer_family_pair_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[int], Param[int], int]:
    """Draw two integer-family params (any interval kind), plus a candidate."""
    family = st.one_of(
        draw_interval_integer_param(),
        draw_bounded_integer_param(),
        draw_natural_param(),
    )
    left = draw(family)
    right = draw(family)
    candidate = draw(
        st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT)
    )
    return left, right, candidate


@st.composite
def draw_real_family_pair_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[str | float], Param[str | float], float]:
    """Draw two independent bounded-real params with a candidate."""
    left = draw(draw_bounded_real_param())
    right = draw(draw_bounded_real_param())
    candidate = float(
        draw(st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT))
    )
    return left, right, candidate


# =============================================================================
# Property: is_subset equals enumerated value-set inclusion, finite kinds
# =============================================================================


@given(case=st.one_of(draw_ordinal_subset_case(), draw_categorical_subset_case()))
def test_is_subset_matches_enumerated_value_set_inclusion_for_finite_kinds(
    case: tuple[Param[object], Param[object], frozenset[object]],
) -> None:
    """Test ``A.is_subset(B)`` equals "every value valid in A is valid in B".

    A finite-set domain (ordinal or categorical) decides subset by
    enumeration, per ``Param.check_subset``'s own docstring, so the
    oracle enumerates the union of both operands' declared value sets
    and checks each with ``is_value_valid`` -- the same admissibility
    check the implementation itself is built from.
    """
    left, right, universe = case

    expected = all(
        right.is_value_valid(value) for value in universe if left.is_value_valid(value)
    )

    assert left.is_subset(right) == expected


# =============================================================================
# Property: check_subset is sound for numeric (interval) kinds
# =============================================================================


@pytest.mark.z3
# Z3-backed: numeric operands may route check_subset through the solver.
@settings(max_examples=50)
@given(
    case=st.one_of(
        draw_integer_family_pair_with_candidate(),
        draw_real_family_pair_with_candidate(),
    )
)
def test_check_subset_is_sound_against_a_sampled_counterexample(
    case: tuple[Param[object], Param[object], object],
) -> None:
    """Test a counterexample decides ``check_subset`` against SATISFIED.

    If a sampled ``x`` is valid for ``A`` but invalid for ``B``, it
    witnesses that ``A``'s feasible set is not contained in ``B``'s, so
    ``check_subset(A, B)`` must not report ``SATISFIED`` and
    ``is_subset(A, B)`` must be ``False``. Numeric operands may route
    through the Z3 bridge, unlike the finite-kind property above.
    """
    left, right, candidate = case

    if left.is_value_valid(candidate) and not right.is_value_valid(candidate):
        assert left.check_subset(right) is not ConstraintOutcome.SATISFIED
        assert not left.is_subset(right)
