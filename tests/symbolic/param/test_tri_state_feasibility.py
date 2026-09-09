"""Tests for tri-state `Param.check_feasibility` and `Param.check_subset`.

The tri-state queries are the real implementation; `is_feasible`,
`is_empty`, and `is_subset` are thin wrappers that fold an `UNDECIDED`
outcome into the optimistic boolean. These tests pin down all three
outcomes on the tri-state side and the exact fold on the boolean side, so
the wrappers' documented optimism cannot drift away from the outcome it is
derived from.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
)

from .conftest import mock_identifier


def _create_undecided_param(name: str = "x", identifier_id: int = 1) -> Param[int]:
    """Create an integer parameter whose feasibility the solver cannot decide.

    `x / x != 1` trips the division hazard screen, since the divisor is
    not a nonzero literal, so the seam refuses to lower it and reports no
    decision at all.
    """
    variable = mock_identifier(name, identifier_id)
    hazardous = (
        IdentifierExpression(variable) / IdentifierExpression(variable)
    ).not_equals(1)
    return create_integer_param(
        name=variable, constraints=[EquationConstraint(hazardous)]
    )


# =============================================================================
# check_feasibility: all three outcomes
# =============================================================================


def test_check_feasibility_reports_satisfied_for_a_reachable_interval() -> None:
    """Test a satisfiable integer interval reports `SATISFIED`."""
    param = create_integer_param_between(1, 5)

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED


def test_check_feasibility_reports_violated_for_contradictory_constraints() -> None:
    """Test contradictory bounds report `VIOLATED`."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < 0),
            EquationConstraint(IdentifierExpression(x) > 0),
        ],
    )

    assert param.check_feasibility() is ConstraintOutcome.VIOLATED


def test_check_feasibility_reports_undecided_when_the_solver_refuses() -> None:
    """Test a hazardous equation the seam refuses reports `UNDECIDED`.

    This is the outcome the boolean wrappers cannot express: it is neither
    a proof of feasibility nor a proof of emptiness.
    """
    param = _create_undecided_param()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED


# =============================================================================
# is_feasible / is_empty fold UNDECIDED optimistically
# =============================================================================


def test_is_feasible_folds_undecided_to_true() -> None:
    """Test `is_feasible` reads an undecided outcome as "not disproven"."""
    param = _create_undecided_param()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert param.is_feasible()


def test_is_empty_folds_undecided_to_false() -> None:
    """Test `is_empty` reads an undecided outcome as "not proven empty"."""
    param = _create_undecided_param()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert not param.is_empty()


def test_is_feasible_and_is_empty_are_both_false_only_when_undecided() -> None:
    """Test the two wrappers are complements, so neither is lost to the fold.

    An undecided parameter reports feasible and not empty, which is the
    documented optimism; a decided one reports exactly one of the two.
    """
    undecided = _create_undecided_param()
    feasible = create_integer_param_between(1, 5)
    x = mock_identifier("x", 2)
    empty = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < 0),
            EquationConstraint(IdentifierExpression(x) > 0),
        ],
    )

    assert (undecided.is_feasible(), undecided.is_empty()) == (True, False)
    assert (feasible.is_feasible(), feasible.is_empty()) == (True, False)
    assert (empty.is_feasible(), empty.is_empty()) == (False, True)


# =============================================================================
# check_subset: all three outcomes
# =============================================================================


def test_check_subset_reports_satisfied_for_a_narrower_interval() -> None:
    """Test a narrower integer interval reports `SATISFIED` against a wider one."""
    narrower = create_integer_param_between(2, 3)
    wider = create_integer_param_between(1, 5)

    assert narrower.check_subset(wider) is ConstraintOutcome.SATISFIED


def test_check_subset_reports_violated_for_a_wider_interval() -> None:
    """Test a wider integer interval reports `VIOLATED` against a narrower one."""
    wider = create_integer_param_between(1, 5)
    narrower = create_integer_param_between(2, 3)

    assert wider.check_subset(narrower) is ConstraintOutcome.VIOLATED


def test_check_subset_reports_violated_across_value_spaces() -> None:
    """Test an integer parameter reports `VIOLATED` against a real one.

    Value-space gating is a decided answer, not an undecided one: the two
    parameters range over different value spaces, so no subset relation
    can hold.
    """
    integer: Param[Any] = create_integer_param_between(1, 5)
    real: Param[Any] = create_real_param()

    assert integer.check_subset(real) is ConstraintOutcome.VIOLATED


def test_check_subset_reports_violated_across_finite_families() -> None:
    """Test an ordinal parameter reports `VIOLATED` against a categorical one."""
    ordinal = create_ordinal_param([1, 2])
    categorical = create_categorical_param({1, 2})

    assert ordinal.check_subset(categorical) is ConstraintOutcome.VIOLATED


def test_check_subset_reports_undecided_when_the_solver_refuses() -> None:
    """Test a hazardous antecedent the seam refuses reports `UNDECIDED`."""
    undecided = _create_undecided_param()
    wider = create_integer_param(name=undecided.variable)

    assert undecided.check_subset(wider) is ConstraintOutcome.UNDECIDED


# =============================================================================
# is_subset folds UNDECIDED optimistically
# =============================================================================


def test_is_subset_folds_undecided_to_true() -> None:
    """Test `is_subset` reads an undecided implication as "not disproven"."""
    undecided = _create_undecided_param()
    wider = create_integer_param(name=undecided.variable)

    assert undecided.check_subset(wider) is ConstraintOutcome.UNDECIDED
    assert undecided.is_subset(wider)


def test_is_subset_reports_false_only_for_a_violated_outcome() -> None:
    """Test only a reported counterexample folds `is_subset` to `False`."""
    wider = create_integer_param_between(1, 5)
    narrower = create_integer_param_between(2, 3)

    assert wider.check_subset(narrower) is ConstraintOutcome.VIOLATED
    assert not wider.is_subset(narrower)
    assert narrower.is_subset(wider)


# =============================================================================
# Finite-set domains enumerate, so they never report UNDECIDED
# =============================================================================


@pytest.mark.parametrize(
    "create_param",
    [
        pytest.param(lambda: create_ordinal_param([1, 2, 3]), id="ordinal"),
        pytest.param(lambda: create_categorical_param({"a", "b"}), id="categorical"),
        pytest.param(lambda: create_permutation_param(["n", "c"]), id="permutation"),
    ],
)
def test_finite_set_check_feasibility_always_decides(
    create_param: Callable[[], Param[Any]],
) -> None:
    """Test a finite-set parameter's feasibility is never undecided."""
    param = create_param()

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED


def test_finite_set_check_feasibility_reports_violated_for_an_empty_narrowing() -> None:
    """Test an ordinal parameter narrowed to no admissible member is `VIOLATED`."""
    param = create_ordinal_param([1, 2, 3])
    narrowed = param.add_constraint(InSetConstraint(param.variable, {9}))

    assert narrowed.check_feasibility() is ConstraintOutcome.VIOLATED
    assert narrowed.is_empty()
