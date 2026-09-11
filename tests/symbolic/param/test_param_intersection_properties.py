"""Hypothesis property tests for `create_intersection_param`.

Split out of ``test_param_intersection.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.

``create_intersection_param`` accepts every domain kind: ordinal,
categorical, permutation, bounded integer, natural, bounded real, and
interval-integer, per its own docstring and the ``ParamDomain.compute_intersection``
overrides (a mixed interval-integer/plain-integer pair is also accepted, via
coercion, but that pairing is exercised by the interval-arithmetic property
files instead). Most properties below draw across all seven kinds.
"""

from typing import Any, Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import InSetConstraint
from fhy_core.symbolic.param import (
    Param,
    create_intersection_param,
    create_ordinal_param,
    create_union_param,
)

from ...strategies.params import (
    draw_overlapping_intersection_eligible_group,
    draw_overlapping_union_eligible_group,
)

# Most properties here intersect numeric operands (bounded integer, real,
# interval-integer), which routes an empty-intersection check through the Z3
# bridge (`Param.check_feasibility`) even when the operands themselves carry
# no explicit solver-only constraint. Every test in this file is therefore
# marked z3, whether or not the specific example it runs happens to draw a
# numeric kind.
pytestmark = [pytest.mark.property, pytest.mark.z3]

# Every property runs without a hypothesis deadline. Every example takes
# about a millisecond and no draw is filtered -- the overlap the membership
# and algebraic-law properties need is built into the strategies rather than
# filtered for -- so a deadline here would time only the scheduler: on a
# contended machine one example can be descheduled for hundreds of
# milliseconds, failing a law for a reason unrelated to it. The
# `dev`/`thorough` profiles already set `deadline=None`.

_ORDINAL_LIMIT: Final = 12


@st.composite
def draw_overlapping_constrained_ordinal_pair_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[int], Param[int], int]:
    """Draw two ordinal params each further restricted by an in-set constraint.

    Both the declared value sets and the added in-set constraints share
    one pivot member, so the restricted intersection stays non-empty by
    construction rather than by filtering for overlap.
    """
    pivot = draw(st.integers(min_value=-_ORDINAL_LIMIT, max_value=_ORDINAL_LIMIT))
    integers = st.integers(min_value=-_ORDINAL_LIMIT, max_value=_ORDINAL_LIMIT)
    left_values = sorted({pivot} | draw(st.sets(integers, max_size=5)))
    right_values = sorted({pivot} | draw(st.sets(integers, max_size=5)))
    left = create_ordinal_param(left_values)
    right = create_ordinal_param(right_values)
    left_restricted = {pivot} | draw(
        st.sets(st.sampled_from(left_values), max_size=len(left_values))
    )
    right_restricted = {pivot} | draw(
        st.sets(st.sampled_from(right_values), max_size=len(right_values))
    )
    left = left.add_constraint(InSetConstraint(left.variable, sorted(left_restricted)))
    right = right.add_constraint(
        InSetConstraint(right.variable, sorted(right_restricted))
    )
    candidate = draw(
        st.integers(min_value=-_ORDINAL_LIMIT - 5, max_value=_ORDINAL_LIMIT + 5)
    )
    return left, right, candidate


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@given(case=draw_overlapping_intersection_eligible_group(size=2))
def test_intersection_membership_law_holds_across_every_supported_domain_kind(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test a value is valid for the intersection iff valid for both operands.

    Holds across every domain kind ``create_intersection_param`` accepts,
    and over each operand's own admissibility and constraints via
    ``is_value_valid`` rather than its raw value set. The operands are
    drawn sharing a member, so the intersection is non-empty by
    construction -- an empty one would make the factory raise rather
    than return a param.
    """
    (left, right), candidate = case

    result = create_intersection_param(left, right)

    assert result.is_value_valid(candidate) == (
        left.is_value_valid(candidate) and right.is_value_valid(candidate)
    )


@given(case=draw_overlapping_constrained_ordinal_pair_with_candidate())
def test_intersection_membership_law_follows_each_operands_constrained_set(
    case: tuple[Param[int], Param[int], int],
) -> None:
    """Test intersection membership follows each operand's own constrained set.

    Each operand carries its own added in-set constraint, so a value is
    valid for the intersection iff it is valid for both operands'
    declared value sets *and* their added constraints, not merely their
    declared sets.
    """
    left, right, candidate = case

    result = create_intersection_param(left, right)

    assert result.is_value_valid(candidate) == (
        left.is_value_valid(candidate) and right.is_value_valid(candidate)
    )


# =============================================================================
# Property: algebraic laws of intersection
# =============================================================================


@given(case=draw_overlapping_intersection_eligible_group(size=2))
def test_intersection_is_commutative(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test ``A & B`` and ``B & A`` agree on every candidate's membership."""
    (left, right), candidate = case

    forward = create_intersection_param(left, right)
    backward = create_intersection_param(right, left)

    assert forward.is_value_valid(candidate) == backward.is_value_valid(candidate)


@given(case=draw_overlapping_intersection_eligible_group(size=3))
def test_intersection_is_associative(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test ``(A & B) & C`` and ``A & (B & C)`` agree on every candidate.

    Membership-equal, not identical objects: associativity is checked as
    a law over ``is_value_valid``, the same terms the membership law
    above states in.
    """
    (a, b, c), candidate = case

    left_first = create_intersection_param(create_intersection_param(a, b), c)
    right_first = create_intersection_param(a, create_intersection_param(b, c))

    assert left_first.is_value_valid(candidate) == right_first.is_value_valid(candidate)


@given(case=draw_overlapping_intersection_eligible_group(size=1))
def test_intersection_is_idempotent(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test ``A & A`` agrees with ``A`` on every candidate's membership."""
    (left,), candidate = case

    result = create_intersection_param(left, left)

    assert result.is_value_valid(candidate) == left.is_value_valid(candidate)


@given(case=draw_overlapping_union_eligible_group(size=2))
def test_intersection_absorbs_union(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test ``A & (A | B)`` agrees with ``A`` on every candidate's membership.

    Restricted to ordinal and categorical, the two kinds
    ``create_union_param`` accepts, since the law needs union defined on
    the same operands.
    """
    (a, b), candidate = case

    union = create_union_param(a, b)
    absorbed = create_intersection_param(a, union)

    assert absorbed.is_value_valid(candidate) == a.is_value_valid(candidate)
