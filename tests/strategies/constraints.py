"""Hypothesis strategies for `Constraint` and `ConstraintSystem` instances.

Every constraint drawn here is scoped to a caller-supplied variable
(usually one identifier from a pool), the same shape
``tests/symbolic/constraint/conftest.py``'s example builders use.
"""

from collections.abc import Sequence
from typing import Final

from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintMember,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import make_binary_expression

from .expressions import COMPARISON_OPERATIONS
from .identifiers import build_identifier_strategy

__all__ = [
    "build_integer_bindings_strategy",
    "build_set_member_strategy",
    "draw_bound_equation_constraint",
    "draw_constraint_system",
    "draw_in_set_constraint",
    "draw_integer_set_constraint",
    "draw_not_in_set_constraint",
]

_SHORT_TEXT_ALPHABET: Final = "abc"


def build_set_member_strategy() -> st.SearchStrategy[ConstraintMember]:
    """Return a strategy for a scalar or small-tuple constraint member.

    Draws ints in ``[-5, 10]``, short strings, booleans, small finite
    floats, and 2-tuples of ints: a representative sample of the
    ``ConstraintMember`` primitive kinds.
    """
    return st.one_of(
        st.integers(min_value=-5, max_value=10),
        st.text(alphabet=_SHORT_TEXT_ALPHABET, min_size=0, max_size=3),
        st.booleans(),
        st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        st.tuples(
            st.integers(min_value=-5, max_value=10),
            st.integers(min_value=-5, max_value=10),
        ),
    )


@st.composite
def draw_in_set_constraint(
    draw: st.DrawFn, variable: Identifier, max_size: int = 4
) -> InSetConstraint:
    """Draw an ``InSetConstraint`` scoped to ``variable``."""
    members = draw(st.lists(build_set_member_strategy(), min_size=0, max_size=max_size))
    return InSetConstraint(variable, members)


@st.composite
def draw_not_in_set_constraint(
    draw: st.DrawFn, variable: Identifier, max_size: int = 4
) -> NotInSetConstraint:
    """Draw a ``NotInSetConstraint`` scoped to ``variable``."""
    members = draw(st.lists(build_set_member_strategy(), min_size=0, max_size=max_size))
    return NotInSetConstraint(variable, members)


@st.composite
def draw_bound_equation_constraint(
    draw: st.DrawFn, variable: Identifier, limit: int = 10
) -> EquationConstraint:
    """Draw an ``EquationConstraint`` of the form ``variable <cmp> literal``."""
    operation = draw(st.sampled_from(COMPARISON_OPERATIONS))
    bound = draw(st.integers(min_value=-limit, max_value=limit))
    return EquationConstraint(make_binary_expression(operation, variable, bound))


@st.composite
def draw_integer_set_constraint(
    draw: st.DrawFn, variable: Identifier, limit: int = 10
) -> InSetConstraint | NotInSetConstraint:
    """Draw an integer-only ``InSetConstraint`` or ``NotInSetConstraint``.

    Restricted to integer members (unlike :func:`draw_in_set_constraint`),
    so a brute-force property can enumerate the same domain the
    constraint draws its members from.
    """
    members = draw(
        st.lists(
            st.integers(min_value=-limit, max_value=limit),
            min_size=0,
            max_size=4,
            unique=True,
        )
    )
    if draw(st.booleans()):
        return InSetConstraint(variable, members)
    return NotInSetConstraint(variable, members)


@st.composite
def draw_constraint_system(
    draw: st.DrawFn, identifiers: Sequence[Identifier], max_members: int = 4
) -> tuple[ConstraintSystem, tuple[Constraint, ...]]:
    """Draw a constraint system together with its members, in draw order.

    Members are integer set constraints and bound equations over
    ``identifiers``.

    Args:
        draw: The active Hypothesis draw function.
        identifiers: Non-empty pool a member's variable is drawn from.
        max_members: Most members the system may hold.

    Returns:
        The canonicalized system and the member tuple in the order they
        were drawn (which need not match the system's canonical order).

    """
    num_members = draw(st.integers(min_value=0, max_value=max_members))
    members: list[Constraint] = []
    for _ in range(num_members):
        variable = draw(build_identifier_strategy(identifiers))
        if draw(st.booleans()):
            members.append(draw(draw_integer_set_constraint(variable)))
        else:
            members.append(draw(draw_bound_equation_constraint(variable)))
    system = create_constraint_system(*members)
    return system, tuple(members)


def build_integer_bindings_strategy(
    identifiers: Sequence[Identifier], min_value: int = -5, max_value: int = 10
) -> st.SearchStrategy[dict[Identifier, int]]:
    """Return a strategy binding every identifier in ``identifiers`` to an int."""
    return st.fixed_dictionaries(
        {
            identifier: st.integers(min_value=min_value, max_value=max_value)
            for identifier in identifiers
        }
    )
