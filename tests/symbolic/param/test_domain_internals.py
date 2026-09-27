"""Tests for the screening and rescoping rules of `fhy_core.symbolic.param.domains`.

A `Param`'s own constraints are always scoped to its own variable
(`Param.validate_constraint` enforces this at construction), so a
constraint scoped to a foreign variable never reaches the screening or the
rescoping through `Param.is_subset`/`is_feasible`. The domains' own
methods take any constraints and variables, so each test below drives the
rule through them: `compute_intersection` rescopes both sides' constraints
onto the result variable, `has_feasible_value` asks the solver about the
screened system of a numeric side, and `compute_constraint_implication_subset`
screens both sides of an implication. (The private helpers
`_rename_constraint_variable` and `_build_screened_constraint_system` these
tests called were deleted when the procedures moved to the Rust core, S16;
the rules are pinned here and in `rust/fhy-core/tests/it/param/`.)

Also covers the `WARNING` logging screening emits for every constraint or
member excluded, and for every solver `UNDECIDED` outcome, which a boolean
wrapper reports as unproven.
"""

import logging
from collections.abc import Callable, Sequence

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import IntegerDomain, create_integer_param
from fhy_core.symbolic.param.domains import compute_constraint_implication_subset
from fhy_core.symbolic.solver import SatResult
from fhy_core.symbolic.symbol_type import SymbolType

from ..conftest import RecordingSmtSolver
from .conftest import mock_identifier

_DOMAINS_LOGGER = "fhy_core.symbolic.param.domains"

PlugSmtSolver = Callable[[SatResult], RecordingSmtSolver]


def _find_records(
    caplog: pytest.LogCaptureFixture, level: int
) -> list[logging.LogRecord]:
    """Return the domains module's records emitted at exactly `level`."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name == _DOMAINS_LOGGER
    ]


def _rescope(
    constraint: Constraint, old_variable: Identifier, new_variable: Identifier
) -> Constraint:
    """Return `constraint` rescoped from `old_variable` onto `new_variable`.

    An intersection rescopes each side's constraints onto the result
    variable; the other side here carries none.
    """
    other_variable = Identifier("other")
    _, constraints = IntegerDomain().compute_intersection(
        (constraint,), old_variable, IntegerDomain(), (), other_variable, new_variable
    )
    assert len(constraints) == 1
    return constraints[0]


def _screen_for_feasibility(
    plug_smt_solver: PlugSmtSolver,
    constraints: Sequence[Constraint],
    variable: Identifier,
) -> tuple[ConstraintOutcome, list[str]]:
    """Return the feasibility of `constraints` for `variable`, and the scripts asked.

    The solver answers `sat`, so a screened system that lost a constraint
    reports `UNDECIDED`, and one that lost nothing `SATISFIED`; a screened
    system with no constraint is satisfied without a question.
    """
    backend = plug_smt_solver(SatResult.SAT)
    outcome = IntegerDomain().has_feasible_value(tuple(constraints), variable)
    return outcome, [script for script, _ in backend.checks]


def _screen_as_consequent(
    plug_smt_solver: PlugSmtSolver,
    constraint: Constraint,
    variable: Identifier,
) -> list[str]:
    """Return the scripts an implication into `constraint` over `variable` asks.

    The solver answers `unsat`, so no witness is found and the implication
    is asked, with the consequent screened.
    """
    backend = plug_smt_solver(SatResult.UNSAT)
    compute_constraint_implication_subset(
        IntegerDomain(),
        (),
        Identifier("own"),
        IntegerDomain(),
        (constraint,),
        variable,
        SymbolType.INT,
    )
    return [script for script, _ in backend.checks]


# =============================================================================
# Rescoping: explicit scope precondition for set constraints
# =============================================================================


@pytest.mark.parametrize(
    "constraint_type",
    [
        pytest.param(InSetConstraint, id="in_set"),
        pytest.param(NotInSetConstraint, id="not_in_set"),
    ],
)
def test_rescoping_rejects_a_scope_mismatch(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test rescoping raises when the constraint is not scoped to `old_variable`.

    Passing `old=w` against a constraint scoped to `z` must not silently
    relabel it to `x`.
    """
    z = mock_identifier("z", 1)
    w = mock_identifier("w", 2)
    x = mock_identifier("x", 3)
    constraint = constraint_type(z, {1, 2})

    with pytest.raises(ConstraintError, match="scoped"):
        _rescope(constraint, w, x)


@pytest.mark.parametrize(
    "constraint_type",
    [
        pytest.param(InSetConstraint, id="in_set"),
        pytest.param(NotInSetConstraint, id="not_in_set"),
    ],
)
def test_rescoping_renames_a_matching_set_constraint(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test rescoping a correctly-scoped set constraint relabels its variable."""
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    constraint = constraint_type(w, {1, 2})

    renamed = _rescope(constraint, w, x)

    assert isinstance(renamed, constraint_type)
    assert renamed.variable == x
    assert renamed.members == (1, 2)


def test_rescoping_substitutes_a_matching_equation_constraint() -> None:
    """Test rescoping an equation constraint substitutes the identifier in place."""
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    constraint = EquationConstraint(IdentifierExpression(w) >= 0)

    renamed = _rescope(constraint, w, x)

    assert isinstance(renamed, EquationConstraint)
    assert renamed.get_free_identifiers() == frozenset((x,))


def test_rescoping_an_equation_constraint_ignores_an_absent_old_variable() -> None:
    """Test rescoping an equation constraint not referencing `old_variable` is a no-op.

    Contrasts with the set-constraint precondition above: substitution
    only rewrites occurrences of `old_variable` inside the expression
    tree, so a mismatch cannot mislabel an equation constraint the way it
    could a set constraint's unconditional `variable` field.
    """
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    z = mock_identifier("z", 3)
    constraint = EquationConstraint(IdentifierExpression(z) >= 0)

    renamed = _rescope(constraint, w, x)

    assert isinstance(renamed, EquationConstraint)
    assert renamed.get_free_identifiers() == frozenset((z,))


# =============================================================================
# Screening: what the solver is asked
# =============================================================================


def test_screening_excludes_a_scope_mismatched_not_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening drops a set constraint scoped to a different variable."""
    x = mock_identifier("x", 1)
    z = mock_identifier("z", 2)

    outcome, scripts = _screen_for_feasibility(
        plug_smt_solver, [NotInSetConstraint(z, {1, 2})], x
    )

    assert outcome is ConstraintOutcome.UNDECIDED
    assert scripts == []


def test_screening_excludes_a_scope_mismatched_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening drops an in-set constraint scoped to a different variable."""
    x = mock_identifier("x", 1)
    z = mock_identifier("z", 2)

    scripts = _screen_as_consequent(plug_smt_solver, InSetConstraint(z, {1, 7}), x)

    assert " 7)" not in scripts[-1]


def test_screening_excludes_a_dependent_equation_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening drops an equation constraint that reaches beyond `variable`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))

    outcome, scripts = _screen_for_feasibility(plug_smt_solver, [dependent], x)

    assert outcome is ConstraintOutcome.UNDECIDED
    assert scripts == []


def test_screening_includes_a_correctly_scoped_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening keeps a fully liftable in-set constraint scoped to `variable`."""
    x = mock_identifier("x", 1)

    scripts = _screen_as_consequent(plug_smt_solver, InSetConstraint(x, {1, 7}), x)

    assert len(scripts) == 2
    assert " 7)" in scripts[-1]


def test_screening_excludes_an_unliftable_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening drops an in-set constraint when any member fails to lift.

    An in-set constraint's members combine with OR, so narrowing to only
    the liftable members would shrink the admissible set; the whole
    constraint is excluded instead.
    """
    x = mock_identifier("x", 1)

    scripts = _screen_as_consequent(plug_smt_solver, InSetConstraint(x, {7, "a"}), x)

    # The first question looks for a witness outside the admissible
    # candidates; the implication's consequent lost the in-set constraint.
    assert len(scripts) == 2
    assert " 7)" not in scripts[-1]


def test_screening_narrows_a_partially_liftable_not_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening narrows a not-in-set constraint to its liftable members.

    A not-in-set constraint's members combine with AND, so dropping the
    non-liftable member only widens the admissible set.
    """
    x = mock_identifier("x", 1)

    outcome, scripts = _screen_for_feasibility(
        plug_smt_solver, [NotInSetConstraint(x, {5, "a"})], x
    )

    assert outcome is ConstraintOutcome.UNDECIDED
    assert len(scripts) == 1
    assert " 5)" in scripts[0]


def test_screening_drops_a_wholly_unliftable_not_in_set_constraint(
    plug_smt_solver: PlugSmtSolver,
) -> None:
    """Test screening drops a not-in-set constraint when no member lifts."""
    x = mock_identifier("x", 1)

    outcome, scripts = _screen_for_feasibility(
        plug_smt_solver, [NotInSetConstraint(x, {"a", "b"})], x
    )

    assert outcome is ConstraintOutcome.UNDECIDED
    assert scripts == []


# =============================================================================
# WARNING content: every exclusion names the constraint and the variable
# =============================================================================


def _screen_logging(
    caplog: pytest.LogCaptureFixture,
    plug_smt_solver: PlugSmtSolver,
    constraint: Constraint,
    variable: Identifier,
) -> list[logging.LogRecord]:
    """Return the domains module's WARNINGs while screening `constraint`."""
    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        if isinstance(constraint, InSetConstraint):
            _screen_as_consequent(plug_smt_solver, constraint, variable)
        else:
            _screen_for_feasibility(plug_smt_solver, [constraint], variable)
    return _find_records(caplog, logging.WARNING)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda x: EquationConstraint(
                IdentifierExpression(x) < IdentifierExpression(Identifier("y"))
            ),
            id="dependent_equation",
        ),
        pytest.param(
            lambda x: NotInSetConstraint(mock_identifier("z", 2), {1, 2}),
            id="scope_mismatched_set",
        ),
        pytest.param(lambda x: InSetConstraint(x, {1, "a"}), id="unliftable_in_set"),
        pytest.param(
            lambda x: NotInSetConstraint(x, {5, "a"}), id="narrowed_not_in_set"
        ),
        pytest.param(
            lambda x: NotInSetConstraint(x, {"a", "b"}),
            id="wholly_unliftable_not_in_set",
        ),
    ],
)
def test_screening_logs_warning_naming_the_constraint_and_the_variable(
    caplog: pytest.LogCaptureFixture,
    plug_smt_solver: PlugSmtSolver,
    build: Callable[[Identifier], Constraint],
) -> None:
    """Test each exclusion or narrowing logs a WARNING naming it and the variable."""
    x = mock_identifier("x", 1)
    constraint = build(x)

    warnings = _screen_logging(caplog, plug_smt_solver, constraint, x)

    assert warnings, "expected a WARNING naming the screened constraint"
    message = warnings[0].getMessage()
    assert repr(constraint) in message
    assert repr(x) in message


# =============================================================================
# WARNING content: an UNDECIDED outcome the boolean wrappers report unproven
# =============================================================================


def test_is_feasible_logs_warning_when_satisfiability_is_undecided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an unproven feasibility logs a WARNING naming the variable."""
    x = mock_identifier("x", 1)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    param = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        result = param.is_feasible()

    assert result is False
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the undecided variable"
    assert any(repr(x) in record.getMessage() for record in warnings)


def test_is_subset_logs_warning_when_implication_is_undecided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an unproven subset relation logs a WARNING naming the variable."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    own = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])
    other = create_integer_param(name=y)

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        result = own.is_subset(other)

    assert result is False
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the undecided comparison"
    assert any(repr(x) in record.getMessage() for record in warnings)
