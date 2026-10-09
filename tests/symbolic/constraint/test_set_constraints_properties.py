"""Hypothesis property tests for set-constraint member matching.

Checks `InSetConstraint`/`NotInSetConstraint.evaluate_with_bindings`
against a reference matcher built directly from the type-strict matching
rules documented on `_SetConstraint` and `InSetConstraint`: equality is
type-strict (`1`, `"1"`, `1.0`, and `True` are four distinct members),
and nested tuple members preserve that same type-strict comparison at
every depth. A `float` equal to zero matches regardless of its sign,
since Python's `==` already treats `-0.0` and `0.0` as equal.
"""

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    ConstraintMember,
    ConstraintOutcome,
    InSetConstraint,
    NotInSetConstraint,
)

from ...strategies.constraints import build_set_member_strategy
from .conftest import mock_identifier

pytestmark = pytest.mark.property

_X = mock_identifier("x", 0)


def _matches_member(probe: ConstraintMember, member: ConstraintMember) -> bool:
    """Return whether `probe` type-strictly matches a declared `member`.

    Mirrors the documented matching rule: two values match only when
    their types match exactly (so `1`, `"1"`, `1.0`, and `True` never
    match each other) and, for a matching pair of tuples, every
    corresponding element matches the same way, recursively.
    """
    if isinstance(probe, tuple) and isinstance(member, tuple):
        return len(probe) == len(member) and all(
            _matches_member(probe_item, member_item)
            for probe_item, member_item in zip(probe, member, strict=True)
        )
    return type(probe) is type(member) and probe == member


@st.composite
def _draw_members_and_probe(
    draw: st.DrawFn,
) -> tuple[list[ConstraintMember], ConstraintMember]:
    """Draw up to 4 set members and a probe value from the same strategy."""
    members = draw(st.lists(build_set_member_strategy(), min_size=0, max_size=4))
    probe = draw(build_set_member_strategy())
    return members, probe


@example(drawn=([0.0], -0.0))
@given(drawn=_draw_members_and_probe())
def test_in_set_constraint_matches_the_reference_matcher(
    drawn: tuple[list[ConstraintMember], ConstraintMember],
) -> None:
    """Test InSetConstraint is SATISFIED iff the reference matcher finds a hit."""
    members, probe = drawn
    constraint = InSetConstraint(_X, members)
    bindings: dict[Identifier, Any] = {_X: probe}

    outcome = constraint.evaluate_with_bindings(bindings)

    expected_hit = any(_matches_member(probe, member) for member in members)
    assert outcome is (
        ConstraintOutcome.SATISFIED if expected_hit else ConstraintOutcome.VIOLATED
    )


@example(drawn=([0.0], -0.0))
@given(drawn=_draw_members_and_probe())
def test_not_in_set_constraint_is_the_complement(
    drawn: tuple[list[ConstraintMember], ConstraintMember],
) -> None:
    """Test NotInSetConstraint is SATISFIED iff the reference matcher finds no hit."""
    members, probe = drawn
    constraint = NotInSetConstraint(_X, members)
    bindings: dict[Identifier, Any] = {_X: probe}

    outcome = constraint.evaluate_with_bindings(bindings)

    expected_hit = any(_matches_member(probe, member) for member in members)
    assert outcome is (
        ConstraintOutcome.VIOLATED if expected_hit else ConstraintOutcome.SATISFIED
    )
