"""Hypothesis property tests for `create_union_param`.

Split out of ``test_param_union.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.

``create_union_param`` supports exactly two domain kinds -- ordinal and
categorical, per its own docstring and the ``ParamDomain.compute_union``
overrides -- so every property here draws from those two kinds only.
"""

from typing import Any, Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.param import Param, create_intersection_param, create_union_param

from ...strategies.params import (
    draw_categorical_param,
    draw_ordinal_param,
    draw_overlapping_union_eligible_group,
    draw_same_kind_param_pair_with_candidate,
)

pytestmark = pytest.mark.property

# Every property runs without a hypothesis deadline. Every example takes
# about a millisecond and no draw is filtered, so a deadline here would time
# only the scheduler: on a contended machine one example can be descheduled
# for hundreds of milliseconds, failing a membership or algebraic law for a
# reason unrelated to it. The `dev`/`thorough` profiles already set
# `deadline=None`.

_ORDINAL_CANDIDATE_LIMIT: Final = 17
_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")


@st.composite
def draw_same_kind_param_triple_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[Any], Param[Any], Param[Any], Any]:
    """Draw three params of one union-eligible kind with a shared-superset candidate.

    Mirrors ``draw_same_kind_param_pair_with_candidate`` one operand
    wider, for the associativity law, which needs three operands.
    """
    if draw(st.booleans()):
        a: Param[Any] = draw(draw_ordinal_param())
        b: Param[Any] = draw(draw_ordinal_param())
        c: Param[Any] = draw(draw_ordinal_param())
        candidate: Any = draw(
            st.integers(
                min_value=-_ORDINAL_CANDIDATE_LIMIT, max_value=_ORDINAL_CANDIDATE_LIMIT
            )
        )
    else:
        a = draw(draw_categorical_param())
        b = draw(draw_categorical_param())
        c = draw(draw_categorical_param())
        candidate = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return a, b, c, candidate


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@given(case=draw_same_kind_param_pair_with_candidate())
def test_union_membership_law_holds_across_every_supported_domain_kind(
    case: tuple[Param[Any], Param[Any], Any],
) -> None:
    """Test a value is valid for the union iff valid for either operand.

    Holds over both domain kinds ``create_union_param`` accepts (ordinal
    and categorical), and over each operand's own admissibility and
    constraints via ``is_value_valid`` rather than its raw value set, so
    a constrained operand is covered the same way an unconstrained one is.
    """
    left, right, candidate = case

    result = create_union_param(left, right)

    assert result.is_value_valid(candidate) == (
        left.is_value_valid(candidate) or right.is_value_valid(candidate)
    )


# =============================================================================
# Property: algebraic laws of union
# =============================================================================


@given(case=draw_same_kind_param_pair_with_candidate())
def test_union_is_commutative(case: tuple[Param[Any], Param[Any], Any]) -> None:
    """Test ``A | B`` and ``B | A`` agree on every candidate's membership."""
    left, right, candidate = case

    forward = create_union_param(left, right)
    backward = create_union_param(right, left)

    assert forward.is_value_valid(candidate) == backward.is_value_valid(candidate)


@given(case=draw_same_kind_param_triple_with_candidate())
def test_union_is_associative(
    case: tuple[Param[Any], Param[Any], Param[Any], Any],
) -> None:
    """Test ``(A | B) | C`` and ``A | (B | C)`` agree on every candidate.

    Membership-equal, not identical objects: associativity is checked as
    a law over ``is_value_valid``, the same terms the membership law
    above states in.
    """
    a, b, c, candidate = case

    left_first = create_union_param(create_union_param(a, b), c)
    right_first = create_union_param(a, create_union_param(b, c))

    assert left_first.is_value_valid(candidate) == right_first.is_value_valid(candidate)


@given(case=draw_same_kind_param_pair_with_candidate())
def test_union_is_idempotent(case: tuple[Param[Any], Param[Any], Any]) -> None:
    """Test ``A | A`` agrees with ``A`` on every candidate's membership."""
    left, _, candidate = case

    result = create_union_param(left, left)

    assert result.is_value_valid(candidate) == left.is_value_valid(candidate)


@given(case=draw_overlapping_union_eligible_group(size=2))
def test_union_absorbs_intersection(
    case: tuple[tuple[Param[Any], ...], Any],
) -> None:
    """Test ``A | (A & B)`` agrees with ``A`` on every candidate's membership.

    ``A`` and ``B`` are drawn sharing a member, so ``A & B`` is
    non-empty by construction rather than by filtering; an empty
    intersection would make ``create_intersection_param`` raise.
    """
    (a, b), candidate = case

    intersection = create_intersection_param(a, b)
    absorbed = create_union_param(a, intersection)

    assert absorbed.is_value_valid(candidate) == a.is_value_valid(candidate)
