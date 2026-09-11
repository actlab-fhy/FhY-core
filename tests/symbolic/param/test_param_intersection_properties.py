"""Hypothesis property tests for `create_intersection_param`.

Split out of ``test_param_intersection.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import InSetConstraint
from fhy_core.symbolic.param import create_intersection_param, create_ordinal_param

pytestmark = pytest.mark.property

# Both properties run without a hypothesis deadline. Every example takes
# about a millisecond and no draw is filtered -- the overlap the membership
# law needs is built into the strategies rather than filtered for -- so a
# deadline here would time only the scheduler: on a contended machine one
# example can be descheduled for hundreds of milliseconds, failing a
# membership law for a reason unrelated to it.


_VALUES = st.integers(min_value=0, max_value=12)


@st.composite
def _draw_overlapping_value_sets(draw: st.DrawFn) -> tuple[set[int], set[int], int]:
    """Draw two ordinal value sets plus a member they are built to share.

    The shared member is drawn first and unioned into both sets, so the
    intersection is non-empty by construction. Building the overlap in
    rather than drawing freely and discarding disjoint pairs keeps every
    generated input usable: filtering for overlap rejects most pairs of
    small sets, which starves generation and skews what does get through.
    """
    pivot = draw(_VALUES)
    left_rest = draw(st.sets(_VALUES, max_size=5))
    right_rest = draw(st.sets(_VALUES, max_size=5))
    return {pivot} | left_rest, {pivot} | right_rest, pivot


@st.composite
def _draw_overlapping_narrowed_sets(
    draw: st.DrawFn,
) -> tuple[set[int], set[int], set[int], set[int]]:
    """Draw two value sets with in-set narrowings that share a member.

    Each narrowing is a non-empty subset of its own value set containing
    the shared pivot, so the narrowed sets overlap by construction.
    """
    left_values, right_values, pivot = draw(_draw_overlapping_value_sets())
    left_narrowed = {pivot} | draw(
        st.sets(st.sampled_from(sorted(left_values)), max_size=5)
    )
    right_narrowed = {pivot} | draw(
        st.sets(st.sampled_from(sorted(right_values)), max_size=5)
    )
    return left_values, right_values, left_narrowed, right_narrowed


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@settings(deadline=None)
@given(value_sets=_draw_overlapping_value_sets(), candidate=st.integers(0, 15))
def test_intersection_membership_law_holds_for_random_ordinal_sets(
    value_sets: tuple[set[int], set[int], int], candidate: int
) -> None:
    """Test a value is valid for the intersection iff valid for both operands.

    Holds for arbitrary (non-empty) ordinal value sets; when the sets happen
    to be disjoint the intersection is empty, which `create_intersection_param`
    signals by raising `ParamError` rather than returning a param -- covered
    separately by the disjoint-set unit tests, so the generated sets are
    built to share a member and keep the assertion meaningful.
    """
    left_values, right_values, _ = value_sets
    left = create_ordinal_param(sorted(left_values))
    right = create_ordinal_param(sorted(right_values))

    result = create_intersection_param(left, right)

    expected = candidate in left_values and candidate in right_values
    assert result.is_value_valid(candidate) == expected


@settings(deadline=None)
@given(narrowed_sets=_draw_overlapping_narrowed_sets(), candidate=st.integers(0, 15))
def test_intersection_membership_law_follows_each_operands_narrowed_set(
    narrowed_sets: tuple[set[int], set[int], set[int], set[int]], candidate: int
) -> None:
    """Test intersection membership is decided by both operands' narrowed sets.

    Each operand is narrowed by an in-set constraint drawn from its own
    members, so a value is valid for the intersection iff it lies in both
    narrowed sets, not merely in both declared value sets. As above, the
    narrowings are built to share a member, since an empty intersection
    raises rather than returning a param.
    """
    left_values, right_values, left_narrowed, right_narrowed = narrowed_sets
    left = create_ordinal_param(sorted(left_values))
    left = left.add_constraint(InSetConstraint(left.variable, left_narrowed))
    right = create_ordinal_param(sorted(right_values))
    right = right.add_constraint(InSetConstraint(right.variable, right_narrowed))

    result = create_intersection_param(left, right)

    expected = candidate in left_narrowed and candidate in right_narrowed
    assert result.is_value_valid(candidate) == expected
