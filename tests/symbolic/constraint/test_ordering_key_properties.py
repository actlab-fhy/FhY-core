"""Hypothesis property tests for `ConstraintSystem`'s canonical member order.

`ConstraintSystem` sorts its members by `Constraint.build_ordering_key`,
so construction order must not matter (permuting the constructor's
arguments yields a structurally equivalent system) and the stored order
must match sorting the given members by that same public key.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import BinaryOperation, make_binary_expression

from ...strategies.constraints import (
    draw_bound_equation_constraint,
    draw_in_set_constraint,
    draw_not_in_set_constraint,
)
from ...strategies.identifiers import build_identifier_pool
from .conftest import mock_identifier

pytestmark = pytest.mark.property

_POOL = build_identifier_pool(2)

_WELD_X = mock_identifier("x", 0)
_WELD_Y = mock_identifier("y", 1)
_WELD_Z = mock_identifier("z", 2)
_WELD_MEMBERS: list[Constraint] = [
    NotInSetConstraint(_WELD_Z, {5, 6}),
    EquationConstraint(make_binary_expression(BinaryOperation.LESS, _WELD_X, _WELD_Y)),
    InSetConstraint(_WELD_X, {1, 2}),
]
"""The fixed member list the deleted `test_ordering_key.py` test exercised."""


@st.composite
def _draw_constraint_over_pool(draw: st.DrawFn) -> Constraint:
    """Draw one equation, in-set, or not-in-set constraint over `_POOL`."""
    variable = draw(st.sampled_from(_POOL))
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        return draw(draw_bound_equation_constraint(variable))
    if kind == 1:
        return draw(draw_in_set_constraint(variable))
    return draw(draw_not_in_set_constraint(variable))


@st.composite
def _draw_member_list(draw: st.DrawFn) -> list[Constraint]:
    """Draw 2 to 4 constraints over a 2-identifier pool; identifiers may repeat."""
    count = draw(st.integers(min_value=2, max_value=4))
    return [draw(_draw_constraint_over_pool()) for _ in range(count)]


@example(members=_WELD_MEMBERS)
@given(members=_draw_member_list())
def test_constraint_system_member_order_matches_sorting_by_the_public_key(
    members: list[Constraint],
) -> None:
    """Test a system's canonical member tuple equals members sorted by the key."""
    system = create_constraint_system(*members)

    assert list(system.constraints) == sorted(
        members, key=lambda member: member.build_ordering_key()
    )


@st.composite
def _draw_members_with_permutation(
    draw: st.DrawFn,
) -> tuple[list[Constraint], list[Constraint]]:
    """Draw a member list together with one permutation of it."""
    members = draw(_draw_member_list())
    permutation = draw(st.permutations(members))
    return members, permutation


@given(drawn=_draw_members_with_permutation())
def test_constraint_system_is_invariant_to_construction_order(
    drawn: tuple[list[Constraint], list[Constraint]],
) -> None:
    """Test create_constraint_system yields an equivalent system for any order."""
    members, permutation = drawn

    original = create_constraint_system(*members)
    reordered = create_constraint_system(*permutation)

    assert original.is_structurally_equivalent(reordered)
