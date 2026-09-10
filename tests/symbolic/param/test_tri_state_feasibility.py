"""Tests for tri-state `Param.check_feasibility` and `Param.check_subset`.

The tri-state queries are the real implementation; `is_feasible`,
`is_empty`, and `is_subset` are thin wrappers that report `True` only for
a proven answer, so an `UNDECIDED` outcome folds to `False` in all three.
These tests pin down all three outcomes on the tri-state side and the
exact fold on the boolean side, so the wrappers cannot drift away from the
outcome they are derived from.
"""

import logging
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    logical_and,
)
from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
)

from .conftest import build_case_condition_constraint, mock_identifier


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
# is_feasible / is_empty report True only for a proven answer
# =============================================================================


def test_is_feasible_folds_undecided_to_false() -> None:
    """Test `is_feasible` reads an undecided outcome as "not proven feasible"."""
    param = _create_undecided_param()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert not param.is_feasible()


def test_is_empty_folds_undecided_to_false() -> None:
    """Test `is_empty` reads an undecided outcome as "not proven empty"."""
    param = _create_undecided_param()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert not param.is_empty()


def test_is_feasible_and_is_empty_are_both_false_only_when_undecided() -> None:
    """Test the two wrappers are not complements: `UNDECIDED` makes both `False`.

    Each reports `True` only for its own proof, so an undecided parameter
    is reported neither feasible nor empty, while a decided one reports
    exactly one of the two. Deriving either wrapper as the negation of the
    other would claim a proof the solver never gave.
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

    assert (undecided.is_feasible(), undecided.is_empty()) == (False, False)
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
# is_subset reports True only for a proven relation
# =============================================================================


def test_is_subset_folds_undecided_to_false() -> None:
    """Test `is_subset` reads an undecided implication as "not proven"."""
    undecided = _create_undecided_param()
    wider = create_integer_param(name=undecided.variable)

    assert undecided.check_subset(wider) is ConstraintOutcome.UNDECIDED
    assert not undecided.is_subset(wider)


def test_is_subset_reports_true_only_for_a_satisfied_outcome() -> None:
    """Test only a proven relation folds `is_subset` to `True`."""
    wider = create_integer_param_between(1, 5)
    narrower = create_integer_param_between(2, 3)

    assert narrower.check_subset(wider) is ConstraintOutcome.SATISFIED
    assert narrower.is_subset(wider)
    assert wider.check_subset(narrower) is ConstraintOutcome.VIOLATED
    assert not wider.is_subset(narrower)


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


# =============================================================================
# Numeric in-set enumeration carries each candidate's outcome
# =============================================================================


_DOMAINS_LOGGER = "fhy_core.symbolic.param.domains"


def _find_domain_warnings(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    """Return the domains module's records emitted at exactly `WARNING`."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == _DOMAINS_LOGGER
    ]


def _create_in_set_param_with_undecided_members() -> Param[int]:
    """Create `x in {1, 2, 3}` with `x + y > 0`, leaving every member undecided.

    `y` is foreign to the parameter, so binding any member of the set
    leaves the equation unresolved.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(
        IdentifierExpression(x) + IdentifierExpression(y) > 0
    )
    return create_integer_param(
        name=x, constraints=[InSetConstraint(x, (1, 2, 3)), dependent]
    )


def _create_in_set_param_with_one_decided_member() -> Param[int]:
    """Create `x in {0, 1}` with `x * y == 0`, deciding only the member `0`.

    Binding `0` reduces the product to a literal, so that member is
    decided; binding `1` leaves `y == 0` unresolved.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(
        (IdentifierExpression(x) * IdentifierExpression(y)).equals(0)
    )
    return create_integer_param(
        name=x, constraints=[InSetConstraint(x, (0, 1)), dependent]
    )


def _create_integer_param_with_bound(
    identifier_id: int, build_bound: Callable[[IdentifierExpression], Any]
) -> Param[int]:
    """Create an integer parameter `z` constrained by `build_bound(z)`."""
    z = mock_identifier("z", identifier_id)
    return create_integer_param(
        name=z, constraints=[EquationConstraint(build_bound(IdentifierExpression(z)))]
    )


def test_check_feasibility_reports_undecided_when_no_in_set_candidate_is_decided() -> (
    None
):
    """Test enumeration reports `UNDECIDED` when no candidate is decided either way."""
    param = _create_in_set_param_with_undecided_members()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED


def test_check_feasibility_reports_satisfied_when_one_in_set_candidate_is_decided() -> (
    None
):
    """Test one decided-feasible candidate outweighs an undecided sibling."""
    param = _create_in_set_param_with_one_decided_member()

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED


def test_in_set_enumeration_decides_violated_when_the_solver_refuses() -> None:
    """Test enumeration decides `VIOLATED` for a hazard the solver seam refuses.

    Binding each member of `{1, 2, 3}` reduces `x / x != 1` to a false
    literal, so the enumeration decides what the solver would not.
    """
    undecided = _create_undecided_param()
    narrowed = undecided.add_constraint(InSetConstraint(undecided.variable, (1, 2, 3)))

    assert narrowed.check_feasibility() is ConstraintOutcome.VIOLATED


def test_check_subset_reports_undecided_when_the_rejected_candidate_is_undecided() -> (
    None
):
    """Test a rejected candidate `own` cannot place in its own set is no counterexample.

    `y <= 2` empties `own`, so the relation holds; `y >= 3` places `3` in
    `own`, which `other` rejects. Neither is decided.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(
        IdentifierExpression(x) + IdentifierExpression(y) > 5
    )
    own = create_integer_param(
        name=x, constraints=[InSetConstraint(x, (1, 2, 3)), dependent]
    )
    other = _create_integer_param_with_bound(3, lambda z: z <= 2)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED


def test_check_subset_reports_undecided_beside_an_accepted_decided_candidate() -> None:
    """Test an accepted decided candidate leaves an undecided rejected one open."""
    own = _create_in_set_param_with_one_decided_member()
    other = _create_integer_param_with_bound(3, lambda z: z <= 0)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED


def test_check_subset_reports_violated_when_a_decided_candidate_is_rejected() -> None:
    """Test a candidate decided into `own` and out of `other` is a counterexample."""
    own = _create_in_set_param_with_one_decided_member()
    other = _create_integer_param_with_bound(3, lambda z: z >= 1)

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_check_subset_reports_satisfied_when_other_accepts_every_candidate() -> None:
    """Test `other` accepting every candidate decides `SATISFIED` regardless of `own`.

    A candidate `other` accepts cannot break the relation whether or not
    it actually lies in `own`.
    """
    own = _create_in_set_param_with_undecided_members()
    other = _create_integer_param_with_bound(3, lambda z: z >= 1)

    assert own.check_subset(other) is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_subset_reports_violated_when_own_exceeds_an_undecided_finite_other() -> (
    None
):
    """Test an infinite `own` exceeding an undecided finite `other` is a counterexample.

    `other` admits at most `{1, 2, 3}` however its dependent constraint
    resolves, and `own` admits `4`, so the relation is decided from proof.
    """
    own = _create_integer_param_with_bound(3, lambda z: z >= 1)
    other = _create_in_set_param_with_undecided_members()

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_is_feasible_and_is_empty_fold_an_undecided_enumeration() -> None:
    """Test the boolean wrappers read an undecided enumeration as unproven.

    No candidate is decided either way, so neither feasibility nor
    emptiness is proven and both wrappers report `False`.
    """
    param = _create_in_set_param_with_undecided_members()

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert not param.is_feasible()
    assert not param.is_empty()


def test_is_subset_folds_an_undecided_enumeration_to_false() -> None:
    """Test `is_subset` reads an undecided enumeration as "not proven"."""
    own = _create_in_set_param_with_one_decided_member()
    other = _create_integer_param_with_bound(3, lambda z: z <= 0)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED
    assert not own.is_subset(other)


def test_undecided_feasibility_enumeration_logs_one_warning_naming_the_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an undecided feasibility enumeration logs one WARNING naming candidates."""
    param = _create_in_set_param_with_undecided_members()

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = param.check_feasibility()

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_domain_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "1, 2, 3" in message
    assert repr(param.variable) in message


def test_decided_feasibility_enumeration_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a feasibility enumeration decided by one candidate logs no WARNING."""
    param = _create_in_set_param_with_one_decided_member()

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = param.check_feasibility()

    assert outcome is ConstraintOutcome.SATISFIED
    assert _find_domain_warnings(caplog) == []


def test_undecided_subset_enumeration_logs_one_warning_naming_the_candidates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an undecided subset enumeration logs one WARNING naming its candidates.

    `other` accepts `1` outright, so only `2` and `3`, each undecided on
    the own side and rejected by `other`, leave the relation open.
    """
    own = _create_in_set_param_with_undecided_members()
    other = _create_integer_param_with_bound(3, lambda z: z <= 1)

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = own.check_subset(other)

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_domain_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "2, 3" in message
    assert "1, 2, 3" not in message
    assert repr(own.variable) in message


# =============================================================================
# A weakened screened system cannot carry an unproven answer
# =============================================================================


def _create_integer_param_with_dependent_constraint(
    *build_bounds: Callable[[IdentifierExpression], Any],
) -> Param[int]:
    """Create `x` carrying the dependent `x < y` and each `build_bound(x)`.

    `y` is foreign to the parameter, so screening drops `x < y` before
    the solver is asked and the screened system is inexact.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    x_expression = IdentifierExpression(x)
    dependent = EquationConstraint(x_expression < IdentifierExpression(y))
    bounds = [
        EquationConstraint(build_bound(x_expression)) for build_bound in build_bounds
    ]
    return create_integer_param(name=x, constraints=[dependent, *bounds])


def _find_downgrade_warnings(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    """Return the domains module's WARNING records that report `UNDECIDED`."""
    return [
        record
        for record in _find_domain_warnings(caplog)
        if "UNDECIDED" in record.getMessage()
    ]


def test_check_feasibility_reports_undecided_for_a_satisfiable_weakened_system() -> (
    None
):
    """Test a dropped constraint downgrades the solver's `SATISFIED` to `UNDECIDED`."""
    param = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED


def test_check_feasibility_reports_undecided_for_a_satisfiable_narrowed_system() -> (
    None
):
    """Test a narrowed not-in-set constraint downgrades `SATISFIED` to `UNDECIDED`."""
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[NotInSetConstraint(x, {5, "a"})])

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED


def test_check_feasibility_keeps_violated_for_an_unsatisfiable_weakened_system() -> (
    None
):
    """Test `x < 0 and x > 0` stays `VIOLATED` beside a dropped constraint."""
    param = _create_integer_param_with_dependent_constraint(
        lambda x: x > 0, lambda x: x < 0
    )

    assert param.check_feasibility() is ConstraintOutcome.VIOLATED


def test_check_subset_reports_undecided_for_a_counterexample_to_a_weakened_antecedent() -> (  # noqa: E501
    None
):
    """Test a counterexample against a weakened antecedent is not trusted.

    The screened own side is only `x > 0`, which admits `6` outside
    `[0, 5]`; the dropped `x < y` might forbid it, so the relation is
    reported `UNDECIDED` rather than `VIOLATED`.
    """
    own = _create_integer_param_with_dependent_constraint(lambda x: x > 0)
    other = create_integer_param_between(0, 5)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED


def test_check_subset_reports_undecided_for_an_implication_into_a_weakened_consequent() -> (  # noqa: E501
    None
):
    """Test an implication into a weakened consequent is not trusted.

    `[1, 3]` implies the screened `x > 0`, but the dropped `x < y` may
    reject part of it, so the relation is reported `UNDECIDED` rather
    than `SATISFIED`.
    """
    own = create_integer_param_between(1, 3)
    other = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED


def test_check_subset_keeps_satisfied_when_only_the_antecedent_is_weakened() -> None:
    """Test `SATISFIED` survives a weakened antecedent.

    The screened own side `x > 0` admits every value the original does,
    so its inclusion in `z >= 0` proves the original's inclusion too.
    """
    own = _create_integer_param_with_dependent_constraint(lambda x: x > 0)
    other = _create_integer_param_with_bound(3, lambda z: z >= 0)

    assert own.check_subset(other) is ConstraintOutcome.SATISFIED


def test_check_subset_keeps_violated_when_only_the_consequent_is_weakened() -> None:
    """Test `VIOLATED` survives a weakened consequent.

    `-5` lies in the exact own side and outside the screened `x > 0`,
    which admits every value the original consequent does, so it lies
    outside the original as well.
    """
    own = create_integer_param_between(-5, 3)
    other = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_is_feasible_and_is_empty_fold_a_weakened_feasibility_to_false() -> None:
    """Test the boolean wrappers prove neither answer for a weakened system.

    The solver's `SATISFIED` rests on a dropped constraint, so feasibility
    is unproven, and nothing proves emptiness either.
    """
    param = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    assert param.is_feasible() is False
    assert param.is_empty() is False


def test_is_subset_folds_an_untrusted_counterexample_to_false() -> None:
    """Test `is_subset` reports `False` when the only counterexample is untrusted.

    An untrusted counterexample leaves the relation `UNDECIDED`, which
    proves neither that it holds nor that it fails.
    """
    own = _create_integer_param_with_dependent_constraint(lambda x: x > 0)
    other = create_integer_param_between(0, 5)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED
    assert own.is_subset(other) is False


def test_is_subset_folds_an_untrusted_implication_to_false() -> None:
    """Test `is_subset` reports `False` for an undecided weakened implication."""
    own = create_integer_param_between(1, 3)
    other = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    assert own.check_subset(other) is ConstraintOutcome.UNDECIDED
    assert own.is_subset(other) is False


def test_weakened_feasibility_downgrade_logs_one_warning_naming_the_variable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test downgrading a weakened `SATISFIED` logs one WARNING naming the variable."""
    param = _create_integer_param_with_dependent_constraint(lambda x: x > 0)

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = param.check_feasibility()

    assert outcome is ConstraintOutcome.UNDECIDED
    downgrades = _find_downgrade_warnings(caplog)
    assert len(downgrades) == 1
    assert repr(param.variable) in downgrades[0].getMessage()


@pytest.mark.parametrize(
    ("build_own", "build_other"),
    [
        pytest.param(
            lambda: _create_integer_param_with_dependent_constraint(lambda x: x > 0),
            lambda: create_integer_param_between(0, 5),
            id="weakened-antecedent",
        ),
        pytest.param(
            lambda: create_integer_param_between(1, 3),
            lambda: _create_integer_param_with_dependent_constraint(lambda x: x > 0),
            id="weakened-consequent",
        ),
    ],
)
def test_weakened_subset_downgrade_logs_one_warning_naming_both_variables(
    build_own: Callable[[], Param[int]],
    build_other: Callable[[], Param[int]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test downgrading a weakened subset answer logs one WARNING naming both sides."""
    own = build_own()
    other = build_other()

    with caplog.at_level(logging.WARNING, logger=_DOMAINS_LOGGER):
        outcome = own.check_subset(other)

    assert outcome is ConstraintOutcome.UNDECIDED
    downgrades = _find_downgrade_warnings(caplog)
    assert len(downgrades) == 1
    message = downgrades[0].getMessage()
    assert repr(own.variable) in message
    assert repr(other.variable) in message


# =============================================================================
# True means proven, even where a valid value exists
# =============================================================================


def _create_integer_param_with_equation_literal(
    literal: float | int | str,
) -> Param[int]:
    """Create an integer parameter `v` constrained by `v == literal`."""
    v = mock_identifier("v", 1)
    return create_integer_param(
        name=v,
        constraints=[EquationConstraint(IdentifierExpression(v).equals(literal))],
    )


def _create_integer_param_restricted_to(members: set[float]) -> Param[int]:
    """Create an integer parameter `v` constrained by `v in members`."""
    v = mock_identifier("v", 1)
    return create_integer_param(name=v, constraints=[InSetConstraint(v, members)])


@pytest.mark.parametrize(
    ("build_param", "outcome", "is_feasible", "is_empty"),
    [
        pytest.param(
            lambda: _create_integer_param_with_equation_literal(1.5),
            ConstraintOutcome.UNDECIDED,
            False,
            False,
            id="equals-float-1.5",
        ),
        pytest.param(
            lambda: _create_integer_param_with_equation_literal("1.5"),
            ConstraintOutcome.UNDECIDED,
            False,
            False,
            id="equals-decimal-string-1.5",
        ),
        pytest.param(
            lambda: _create_integer_param_with_equation_literal(2.0),
            ConstraintOutcome.UNDECIDED,
            False,
            False,
            id="equals-float-2.0",
        ),
        pytest.param(
            lambda: _create_integer_param_with_equation_literal(2),
            ConstraintOutcome.SATISFIED,
            True,
            False,
            id="equals-int-2",
        ),
        pytest.param(
            lambda: _create_integer_param_restricted_to({1.5}),
            ConstraintOutcome.VIOLATED,
            False,
            True,
            id="in-set-float-1.5",
        ),
    ],
)
def test_integer_param_wrappers_report_true_only_for_the_outcome_proving_them(
    build_param: Callable[[], Param[int]],
    outcome: ConstraintOutcome,
    is_feasible: bool,
    is_empty: bool,
) -> None:
    """Test each wrapper reports `True` only for the outcome that proves it.

    The solver refuses to compare an integer variable with a float-valued
    literal, so the three float-literal equations are `UNDECIDED` and are
    reported neither feasible nor empty. `v == 1.5` and `v == "1.5"` admit
    no integer at all, while `v == 2.0` admits `2`; reporting that last one
    not feasible is the accepted cost of `True` meaning proven. The in-set
    member `1.5` is decided by enumeration: no integer is a float, so the
    parameter is proven empty.
    """
    param = build_param()

    assert param.check_feasibility() is outcome
    assert param.is_feasible() is is_feasible
    assert param.is_empty() is is_empty


def test_is_feasible_reports_false_for_a_float_equation_an_integer_satisfies() -> None:
    """Test `v == 2.0` is not reported feasible, though `v = 2` is valid.

    Bound to `2`, evaluation decides `2 == 2.0`; unbound, the solver refuses
    the int/float comparison, so feasibility is `UNDECIDED`. `is_feasible`
    therefore reports `False` for a parameter that has a valid value: the
    accepted cost of never reporting an unproven parameter feasible.
    """
    param = _create_integer_param_with_equation_literal(2.0)

    assert param.is_value_valid(2)
    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert param.is_feasible() is False
    assert param.is_empty() is False


# =============================================================================
# An ill-typed constraint raises rather than reporting UNDECIDED
# =============================================================================


def _create_param_conditioned_on_its_own_value() -> Param[int]:
    """Create `x in {1, 2}` whose other constraint takes `x` as a case condition.

    The in-set constraint makes feasibility an enumeration, which binds
    each member to `x` and so puts a number in the case condition.
    """
    x = mock_identifier("x", 1)
    return create_integer_param(
        name=x,
        constraints=[
            InSetConstraint(x, (1, 2)),
            build_case_condition_constraint(IdentifierExpression(x)),
        ],
    )


def _create_param_conditioned_on_arithmetic() -> Param[int]:
    """Create `x` whose constraint takes `x + 1` as a case condition.

    With no in-set constraint the question goes to the solver, whose seam
    refuses `x + 1` in a Boolean position as provably numeric.
    """
    x = mock_identifier("x", 1)
    return create_integer_param(
        name=x,
        constraints=[build_case_condition_constraint(IdentifierExpression(x) + 1)],
    )


def _create_unconstrained_param() -> Param[int]:
    """Create a well-typed integer parameter `y` with no constraints."""
    return create_integer_param(name=mock_identifier("y", 2))


def _create_well_typed_in_set_param() -> Param[int]:
    """Create a well-typed integer parameter `y in {1, 2}`."""
    y = mock_identifier("y", 2)
    return create_integer_param(name=y, constraints=[InSetConstraint(y, (1, 2))])


@pytest.mark.parametrize(
    "build_param",
    [
        pytest.param(_create_param_conditioned_on_its_own_value, id="enumeration"),
        pytest.param(_create_param_conditioned_on_arithmetic, id="solver"),
    ],
)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(Param.check_feasibility, id="check_feasibility"),
        pytest.param(Param.is_feasible, id="is_feasible"),
        pytest.param(Param.is_empty, id="is_empty"),
    ],
)
def test_feasibility_raises_for_a_number_in_a_case_condition(
    build_param: Callable[[], Param[int]], query: Callable[[Param[int]], object]
) -> None:
    """Test an ill-typed constraint raises instead of reporting `UNDECIDED`.

    No backend, bound, or timeout gives a number in a Boolean position a
    meaning, so folding it into an undecided answer would invite a caller
    to retry a question that cannot succeed. The typed error propagates
    from the enumeration and the solver path alike.
    """
    param = build_param()

    with pytest.raises(NonBooleanLogicalOperandError):
        query(param)


@pytest.mark.parametrize(
    ("build_own", "build_other"),
    [
        pytest.param(
            _create_param_conditioned_on_its_own_value,
            _create_unconstrained_param,
            id="enumerating-own",
        ),
        pytest.param(
            _create_unconstrained_param,
            _create_param_conditioned_on_its_own_value,
            id="enumerating-other",
        ),
        pytest.param(
            _create_param_conditioned_on_arithmetic,
            _create_unconstrained_param,
            id="solver-antecedent",
        ),
        pytest.param(
            _create_unconstrained_param,
            _create_param_conditioned_on_arithmetic,
            id="solver-consequent",
        ),
        pytest.param(
            _create_param_conditioned_on_arithmetic,
            _create_well_typed_in_set_param,
            id="solver-witness-outside-a-finite-other",
        ),
    ],
)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(Param.check_subset, id="check_subset"),
        pytest.param(Param.is_subset, id="is_subset"),
    ],
)
def test_subset_raises_for_a_number_in_a_case_condition(
    build_own: Callable[[], Param[int]],
    build_other: Callable[[], Param[int]],
    query: Callable[[Param[int], Param[int]], object],
) -> None:
    """Test an ill-typed constraint on either side raises from the subset query.

    Covers each path that evaluates a constraint: enumerating this
    parameter's candidates, enumerating the other side's, the solver's
    implication in either direction, and the solver's search for a value
    outside a finite other side.
    """
    own = build_own()
    other = build_other()

    with pytest.raises(NonBooleanLogicalOperandError):
        query(own, other)


_BOOLEAN_POSITION_CONSTRAINTS = [
    pytest.param(
        lambda variable: EquationConstraint(
            logical_and(IdentifierExpression(variable), LiteralExpression(True))
        ),
        id="and",
    ),
    pytest.param(
        lambda variable: build_case_condition_constraint(
            IdentifierExpression(variable)
        ),
        id="case_condition",
    ),
]

_NUMERIC_PARAM_FACTORIES = [
    pytest.param(create_integer_param, id="integer"),
    pytest.param(create_real_param, id="real"),
]


@pytest.mark.parametrize("create_param", _NUMERIC_PARAM_FACTORIES)
@pytest.mark.parametrize("build_constraint", _BOOLEAN_POSITION_CONSTRAINTS)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(Param.check_feasibility, id="check_feasibility"),
        pytest.param(Param.is_feasible, id="is_feasible"),
        pytest.param(Param.is_empty, id="is_empty"),
    ],
)
def test_feasibility_raises_for_the_variable_itself_in_a_boolean_position(
    create_param: Callable[..., Param[Any]],
    build_constraint: Callable[[Any], EquationConstraint],
    query: Callable[[Param[Any]], object],
) -> None:
    """Test a numeric parameter's own variable in a Boolean position raises.

    The domain declares the variable INT or REAL to the solver, so the
    variable under a connective or as a case condition is ill-typed. Z3
    used to reject the sort mismatch itself, which escaped a tri-state
    query as a `PassExecutionError`.
    """
    x = mock_identifier("x", 1)
    param = create_param(name=x, constraints=[build_constraint(x)])

    with pytest.raises(NonBooleanLogicalOperandError):
        query(param)


@pytest.mark.parametrize("create_param", _NUMERIC_PARAM_FACTORIES)
@pytest.mark.parametrize("build_constraint", _BOOLEAN_POSITION_CONSTRAINTS)
@pytest.mark.parametrize(
    "is_own_ill_typed", [True, False], ids=["antecedent", "consequent"]
)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(Param.check_subset, id="check_subset"),
        pytest.param(Param.is_subset, id="is_subset"),
    ],
)
def test_subset_raises_for_the_variable_itself_in_a_boolean_position(
    create_param: Callable[..., Param[Any]],
    build_constraint: Callable[[Any], EquationConstraint],
    is_own_ill_typed: bool,
    query: Callable[[Param[Any], Param[Any]], object],
) -> None:
    """Test the solver's implication raises for either side's ill-typed variable."""
    x = mock_identifier("x", 1)
    ill_typed = create_param(name=x, constraints=[build_constraint(x)])
    plain = create_param(name=mock_identifier("y", 2))
    own, other = (ill_typed, plain) if is_own_ill_typed else (plain, ill_typed)

    with pytest.raises(NonBooleanLogicalOperandError):
        query(own, other)


@pytest.mark.parametrize("build_constraint", _BOOLEAN_POSITION_CONSTRAINTS)
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(Param.check_subset, id="check_subset"),
        pytest.param(Param.is_subset, id="is_subset"),
    ],
)
def test_subset_witness_search_raises_for_the_variable_in_a_boolean_position(
    build_constraint: Callable[[Any], EquationConstraint],
    query: Callable[[Param[int], Param[int]], object],
) -> None:
    """Test the search for a value outside a finite other side raises too."""
    x = mock_identifier("x", 1)
    own = create_integer_param(name=x, constraints=[build_constraint(x)])

    with pytest.raises(NonBooleanLogicalOperandError):
        query(own, _create_well_typed_in_set_param())
