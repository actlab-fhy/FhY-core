"""Hypothesis property tests for constraint and constraint-system serialization.

Both `Constraint` and `ConstraintSystem` are declared `eq=False`, so `==`
falls back to object identity; a round trip is checked with
`is_structurally_equivalent` instead, never `==`.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.serialization import SerializationFormat
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintSystem,
)

from ...strategies.constraints import (
    draw_bound_equation_constraint,
    draw_constraint_system,
    draw_in_set_constraint,
    draw_not_in_set_constraint,
)
from ...strategies.identifiers import build_identifier_pool

pytestmark = pytest.mark.property

_X = build_identifier_pool(1)[0]


def _assert_round_trips_structurally(instance: Constraint | ConstraintSystem) -> None:
    """Assert `instance` round-trips through DICT, JSON, and BINARY.

    Uses `is_structurally_equivalent` rather than `==`: both `Constraint`
    and `ConstraintSystem` are `eq=False`, so `==` would compare object
    identity and could never pass for a freshly deserialized instance.
    """
    for fmt in SerializationFormat:
        rebuilt = type(instance).deserialize(instance.serialize(fmt), fmt)
        assert instance.is_structurally_equivalent(rebuilt)


@st.composite
def _draw_constraint(draw: st.DrawFn) -> Constraint:
    """Draw one equation, in-set, or not-in-set constraint over `_X`."""
    kind = draw(st.integers(min_value=0, max_value=2))
    if kind == 0:
        return draw(draw_bound_equation_constraint(_X))
    if kind == 1:
        return draw(draw_in_set_constraint(_X))
    return draw(draw_not_in_set_constraint(_X))


@given(constraint=_draw_constraint())
def test_constraint_round_trips_in_all_formats(constraint: Constraint) -> None:
    """Test an equation, in-set, or not-in-set constraint round-trips structurally."""
    _assert_round_trips_structurally(constraint)


@given(drawn=draw_constraint_system(build_identifier_pool(3)))
def test_constraint_system_round_trips_in_all_formats(
    drawn: tuple[ConstraintSystem, tuple[Constraint, ...]],
) -> None:
    """Test a constraint system round-trips structurally."""
    system, _ = drawn
    _assert_round_trips_structurally(system)
