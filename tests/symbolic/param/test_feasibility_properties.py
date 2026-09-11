"""Hypothesis property tests for `Param.check_feasibility` over finite domains.

Covers ordinal, categorical, and width-bounded plain-integer parameters,
each carrying zero to two extra constraints, against a brute-force
oracle that enumerates the finite domain directly.
"""

from collections.abc import Sequence
from typing import Any, Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintOutcome,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param_between,
    create_ordinal_param,
)

from ...strategies.constraints import draw_bound_equation_constraint
from ...strategies.params import (
    build_categorical_value_set_strategy,
    build_ordinal_value_set_strategy,
)

# `check_feasibility` on the plain-integer case may route through the Z3
# bridge whenever the drawn constraints do not include an `InSetConstraint`
# (see `_numeric_has_feasible_value`), so this whole file is marked z3 and
# capped at 50 examples, per the common Z3-backed-property convention.
pytestmark = [pytest.mark.property, pytest.mark.z3]

# Every property runs without a hypothesis deadline; the `dev`/`thorough`
# profiles already set `deadline=None`. `max_examples=50` on the property
# below caps the more expensive, solver-backed examples.

_WIDTH_LIMIT: Final = 12
_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")


@st.composite
def draw_param_with_extra_constraints(
    draw: st.DrawFn,
    param: Param[Any],
    member_pool: Sequence[Any],
    *,
    constraint_kinds: Sequence[str],
    bound_limit: int,
) -> Param[Any]:
    """Draw zero to two extra constraints, from ``constraint_kinds``, and add them.

    Each extra constraint is an in-set or not-in-set constraint drawn
    from ``member_pool`` (built to hold values inside and outside the
    domain), or a bound equation ``variable <cmp> literal``. Callers
    choose which of ``"in_set"``, ``"not_in_set"``, and ``"bound"`` are
    admissible for their own domain kind: ordinal and categorical
    domains permit only the first two; see the module docstring's linked
    xfail for why the numeric branch below excludes ``"not_in_set"``.
    """
    num_constraints = draw(st.integers(min_value=0, max_value=2))
    for _ in range(num_constraints):
        kind = draw(st.sampled_from(constraint_kinds))
        constraint: Constraint
        if kind == "bound":
            constraint = draw(
                draw_bound_equation_constraint(param.variable, limit=bound_limit)
            )
        else:
            members = draw(
                st.lists(st.sampled_from(member_pool), max_size=4, unique=True)
            )
            constraint_cls = InSetConstraint if kind == "in_set" else NotInSetConstraint
            constraint = constraint_cls(param.variable, members)
        param = param.add_constraint(constraint)
    return param


@st.composite
def draw_finite_ordinal_case(draw: st.DrawFn) -> tuple[Param[int], tuple[int, ...]]:
    """Draw an ordinal param carrying 0-2 extra constraints, with its domain."""
    values = draw(build_ordinal_value_set_strategy())
    param: Param[int] = create_ordinal_param(values)
    outside = [
        v for v in range(-_WIDTH_LIMIT - 5, _WIDTH_LIMIT + 6) if v not in values
    ][:4]
    param = draw(
        draw_param_with_extra_constraints(
            param,
            values + outside,
            constraint_kinds=("in_set", "not_in_set"),
            bound_limit=0,
        )
    )
    return param, tuple(values)


@st.composite
def draw_finite_categorical_case(draw: st.DrawFn) -> tuple[Param[str], tuple[str, ...]]:
    """Draw a categorical param carrying 0-2 extra constraints, with its domain."""
    values = draw(build_categorical_value_set_strategy())
    param: Param[str] = create_categorical_param(values)
    outside = [letter for letter in _CATEGORICAL_ALPHABET if letter not in values]
    param = draw(
        draw_param_with_extra_constraints(
            param,
            values + outside,
            constraint_kinds=("in_set", "not_in_set"),
            bound_limit=0,
        )
    )
    return param, tuple(values)


@st.composite
def draw_finite_bounded_integer_case(
    draw: st.DrawFn,
) -> tuple[Param[int], tuple[int, ...]]:
    """Draw a plain-integer param bounded to width <= 12, with its domain.

    Uses the plain integer domain rather than the interval-integer one:
    only the plain domain permits in-set and bound constraints together.
    Draws only ``"in_set"`` and ``"bound"``, never ``"not_in_set"``: see
    the module-level comment above the xfail near the bottom of this
    file for why a not-in-set constraint on this solver-backed path is
    excluded rather than covered here.
    """
    lower = draw(st.integers(min_value=-20, max_value=20))
    width = draw(st.integers(min_value=0, max_value=_WIDTH_LIMIT))
    upper = lower + width
    param: Param[int] = create_integer_param_between(lower, upper)
    domain = tuple(range(lower, upper + 1))
    outside = [lower - 3, lower - 1, upper + 1, upper + 3]
    param = draw(
        draw_param_with_extra_constraints(
            param,
            list(domain) + outside,
            constraint_kinds=("in_set", "bound"),
            bound_limit=max(abs(lower), abs(upper)) + 5,
        )
    )
    return param, domain


@st.composite
def draw_finite_domain_case(draw: st.DrawFn) -> tuple[Param[Any], tuple[Any, ...]]:
    """Draw a param over a finite domain (ordinal, categorical, or bounded integer)."""
    result: tuple[Param[Any], tuple[Any, ...]] = draw(
        st.one_of(
            draw_finite_ordinal_case(),
            draw_finite_categorical_case(),
            draw_finite_bounded_integer_case(),
        )
    )
    return result


# =============================================================================
# Property: check_feasibility matches brute-force enumeration
# =============================================================================


# Z3-backed: some drawn cases route check_feasibility through the solver.
@settings(max_examples=50)
@given(case=draw_finite_domain_case())
def test_check_feasibility_matches_brute_force_over_finite_domains(
    case: tuple[Param[Any], tuple[Any, ...]],
) -> None:
    """Test ``check_feasibility`` agrees with brute force over a finite domain.

    ``SATISFIED`` iff some candidate in the domain is valid, ``VIOLATED``
    iff none is; a finite domain (ordinal, categorical, or a
    width-bounded integer interval) always decides, so ``UNDECIDED``
    must not occur. ``is_feasible`` and ``is_empty`` must agree with the
    decided outcome.
    """
    param, domain = case

    any_valid = any(param.is_value_valid(candidate) for candidate in domain)
    outcome = param.check_feasibility()

    assert outcome is not ConstraintOutcome.UNDECIDED, (
        f"check_feasibility should always decide over a finite domain, "
        f"got UNDECIDED for {param!r}"
    )
    assert outcome is (
        ConstraintOutcome.SATISFIED if any_valid else ConstraintOutcome.VIOLATED
    )
    assert param.is_feasible() == (outcome is ConstraintOutcome.SATISFIED)
    assert param.is_empty() == (outcome is ConstraintOutcome.VIOLATED)


# =============================================================================
# Known discrepancy: any not-in-set constraint spuriously undecides
# =============================================================================
#
# A plain-integer parameter carrying any `NotInSetConstraint` -- even one
# whose member lies entirely outside the domain -- and no `InSetConstraint`
# (which would force enumeration instead) goes to the Z3-screening path
# (`fhy_core.symbolic.param.domains._numeric_has_feasible_value`). There,
# `_build_screened_constraint_system_with_fidelity` compares the screened
# constraint to the original with `is not` to decide whether the screen
# excluded anything. `_screen_not_in_set_constraint` always returns a
# *freshly constructed* `NotInSetConstraint`, even when every member lifted
# and nothing was excluded, so that comparison is `True` unconditionally and
# the system is marked inexact for every not-in-set constraint. The
# solver's own SATISFIED answer is then conservatively downgraded to
# UNDECIDED. The parameter below is, in fact, provably feasible.
# `draw_finite_bounded_integer_case` draws only "in_set" and "bound" extra
# constraints so the property above never trips this discrepancy; this test
# covers the excluded not-in-set shape on its own.


@pytest.mark.xfail(
    strict=True,
    reason=(
        "domains._build_screened_constraint_system_with_fidelity compares "
        "a screened NotInSetConstraint to the original with `is not`, but "
        "_screen_not_in_set_constraint always rebuilds a fresh object even "
        "when no member was excluded, so fidelity is False -- and the "
        "solver's SATISFIED answer is downgraded to UNDECIDED -- for every "
        "not-in-set constraint, not only one whose screen actually "
        "excluded a member."
    ),
)
def test_check_feasibility_decides_singleton_with_a_harmless_constraint() -> None:
    """Test a satisfiable singleton with an irrelevant not-in-set constraint decides.

    ``create_integer_param_between(0, 0)`` constrained by
    ``NotInSetConstraint(variable, [5])`` is provably ``SATISFIED``: ``5``
    is outside the domain entirely, so the constraint excludes nothing
    ``0`` needed. ``check_feasibility`` instead reports ``UNDECIDED``.
    """
    param = create_integer_param_between(0, 0)
    param = param.add_constraint(NotInSetConstraint(param.variable, [5]))

    assert param.is_value_valid(0)
    assert param.check_feasibility() is ConstraintOutcome.SATISFIED
