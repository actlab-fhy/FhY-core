"""Tests for private screening/renaming helpers in `fhy_core.symbolic.param.domains`.

`_rename_constraint_variable`'s scope precondition and
`_build_screened_constraint_system`'s scope/liftability screening cannot be
exercised through `Param`'s public API: a `Param`'s own constraints are
always already scoped to its own variable (`Param.validate_constraint`
enforces this at construction), so a mismatched `old_variable`/constraint
pair, or a constraint scoped to a foreign variable, never reaches these
helpers through `Param.is_subset`/`is_feasible`. Each test below calls the
private helper directly to pin the precondition and the screening behavior
that guards it, following the same pattern as `test_core_internals.py` and
`test_bound_internals.py`.

Also covers the `WARNING` logging `_build_screened_constraint_system` and
its callers emit for every constraint or member excluded, and for every
solver `UNDECIDED` outcome collapsed to the optimistic default.
"""

import logging

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintError,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import create_integer_param
from fhy_core.symbolic.param.domains import (
    _build_screened_constraint_system,
    _rename_constraint_variable,
)

from .conftest import mock_identifier

_DOMAINS_LOGGER = "fhy_core.symbolic.param.domains"


def _find_records(
    caplog: pytest.LogCaptureFixture, level: int
) -> list[logging.LogRecord]:
    """Return the domains module's records emitted at exactly `level`."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name == _DOMAINS_LOGGER
    ]


# =============================================================================
# `_rename_constraint_variable`: explicit scope precondition for set constraints
# =============================================================================


@pytest.mark.parametrize(
    "constraint_type",
    [
        pytest.param(InSetConstraint, id="in_set"),
        pytest.param(NotInSetConstraint, id="not_in_set"),
    ],
)
def test_rename_constraint_variable_rejects_a_scope_mismatch(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test renaming raises when the constraint is not scoped to `old_variable`.

    Passing `old=w` against a constraint scoped to `z` must not silently
    relabel it to `x`.
    """
    z = mock_identifier("z", 1)
    w = mock_identifier("w", 2)
    x = mock_identifier("x", 3)
    constraint = constraint_type(z, {1, 2})

    with pytest.raises(ConstraintError, match="scoped"):
        _rename_constraint_variable(constraint, w, x)


@pytest.mark.parametrize(
    "constraint_type",
    [
        pytest.param(InSetConstraint, id="in_set"),
        pytest.param(NotInSetConstraint, id="not_in_set"),
    ],
)
def test_rename_constraint_variable_renames_a_matching_set_constraint(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test renaming a correctly-scoped set constraint relabels its variable."""
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    constraint = constraint_type(w, {1, 2})

    renamed = _rename_constraint_variable(constraint, w, x)

    assert isinstance(renamed, constraint_type)
    assert renamed.variable == x
    assert renamed.members == (1, 2)


def test_rename_constraint_variable_substitutes_a_matching_equation_constraint() -> (
    None
):
    """Test renaming an equation constraint substitutes the identifier in place."""
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    constraint = EquationConstraint(IdentifierExpression(w) >= 0)

    renamed = _rename_constraint_variable(constraint, w, x)

    assert isinstance(renamed, EquationConstraint)
    assert renamed.get_free_identifiers() == frozenset((x,))


def test_rename_equation_constraint_ignores_an_absent_old_variable() -> None:
    """Test renaming an equation constraint not referencing `old_variable` is a no-op.

    Contrasts with the set-constraint precondition above: substitution
    only rewrites occurrences of `old_variable` inside the expression
    tree, so a mismatch cannot mislabel an equation constraint the way it
    could a set constraint's unconditional `variable` field.
    """
    w = mock_identifier("w", 1)
    x = mock_identifier("x", 2)
    z = mock_identifier("z", 3)
    constraint = EquationConstraint(IdentifierExpression(z) >= 0)

    renamed = _rename_constraint_variable(constraint, w, x)

    assert isinstance(renamed, EquationConstraint)
    assert renamed.get_free_identifiers() == frozenset((z,))


# =============================================================================
# `_build_screened_constraint_system`: scope screening for set constraints
# =============================================================================


@pytest.mark.parametrize(
    "constraint_type",
    [
        pytest.param(InSetConstraint, id="in_set"),
        pytest.param(NotInSetConstraint, id="not_in_set"),
    ],
)
def test_screening_excludes_a_scope_mismatched_set_constraint(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test screening drops a set constraint scoped to a different variable."""
    x = mock_identifier("x", 1)
    z = mock_identifier("z", 2)
    mismatched = constraint_type(z, {1, 2})

    system = _build_screened_constraint_system([mismatched], x)

    assert system.constraints == ()


def test_screening_excludes_a_dependent_equation_constraint() -> None:
    """Test screening drops an equation constraint that reaches beyond `variable`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))

    system = _build_screened_constraint_system([dependent], x)

    assert system.constraints == ()


def test_screening_includes_a_correctly_scoped_in_set_constraint() -> None:
    """Test screening keeps a fully liftable in-set constraint scoped to `variable`."""
    x = mock_identifier("x", 1)
    constraint = InSetConstraint(x, {1, 2})

    system = _build_screened_constraint_system([constraint], x)

    assert system.constraints == (constraint,)


def test_screening_excludes_an_unliftable_in_set_constraint() -> None:
    """Test screening drops an in-set constraint when any member fails to lift.

    An in-set constraint's members combine with OR, so narrowing to only
    the liftable members would shrink the admissible set; the whole
    constraint is excluded instead.
    """
    x = mock_identifier("x", 1)
    constraint = InSetConstraint(x, {1, "a"})

    system = _build_screened_constraint_system([constraint], x)

    assert system.constraints == ()


def test_screening_narrows_a_partially_liftable_not_in_set_constraint() -> None:
    """Test screening narrows a not-in-set constraint to its liftable members.

    A not-in-set constraint's members combine with AND, so dropping the
    non-liftable member only widens the admissible set.
    """
    x = mock_identifier("x", 1)
    constraint = NotInSetConstraint(x, {5, "a"})

    system = _build_screened_constraint_system([constraint], x)

    assert len(system.constraints) == 1
    narrowed = system.constraints[0]
    assert isinstance(narrowed, NotInSetConstraint)
    assert narrowed.members == (5,)


def test_screening_drops_a_wholly_unliftable_not_in_set_constraint() -> None:
    """Test screening drops a not-in-set constraint when no member lifts."""
    x = mock_identifier("x", 1)
    constraint = NotInSetConstraint(x, {"a", "b"})

    system = _build_screened_constraint_system([constraint], x)

    assert system.constraints == ()


# =============================================================================
# WARNING content: every exclusion names the constraint and the variable
# =============================================================================


def test_screening_logs_warning_naming_the_dependent_equation_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test excluding a dependent equation constraint logs a WARNING naming it."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        _build_screened_constraint_system([dependent], x)

    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the excluded constraint"
    message = warnings[0].getMessage()
    assert repr(dependent) in message
    assert repr(x) in message


def test_screening_logs_warning_naming_a_scope_mismatched_set_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test excluding a scope-mismatched set constraint logs a WARNING naming it."""
    x = mock_identifier("x", 1)
    z = mock_identifier("z", 2)
    mismatched = InSetConstraint(z, {1, 2})

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        _build_screened_constraint_system([mismatched], x)

    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the excluded constraint"
    message = warnings[0].getMessage()
    assert repr(mismatched) in message
    assert repr(x) in message


def test_screening_logs_warning_naming_an_unliftable_in_set_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test excluding an unliftable in-set constraint logs a WARNING naming it."""
    x = mock_identifier("x", 1)
    constraint = InSetConstraint(x, {1, "a"})

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        _build_screened_constraint_system([constraint], x)

    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the excluded constraint"
    message = warnings[0].getMessage()
    assert repr(constraint) in message
    assert repr(x) in message


def test_screening_logs_warning_naming_a_narrowed_not_in_set_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test narrowing a not-in-set constraint logs a WARNING naming it."""
    x = mock_identifier("x", 1)
    constraint = NotInSetConstraint(x, {5, "a"})

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        _build_screened_constraint_system([constraint], x)

    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the narrowed constraint"
    message = warnings[0].getMessage()
    assert repr(constraint) in message
    assert repr(x) in message


def test_screening_logs_warning_naming_a_wholly_unliftable_not_in_set_constraint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test dropping a wholly unliftable not-in-set constraint logs a WARNING."""
    x = mock_identifier("x", 1)
    constraint = NotInSetConstraint(x, {"a", "b"})

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        _build_screened_constraint_system([constraint], x)

    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the excluded constraint"
    message = warnings[0].getMessage()
    assert repr(constraint) in message
    assert repr(x) in message


# =============================================================================
# WARNING content: an UNDECIDED outcome collapsed to the optimistic default
# =============================================================================


def test_is_feasible_logs_warning_when_satisfiability_is_undecided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the optimistic feasible default logs a WARNING naming the variable."""
    x = mock_identifier("x", 1)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    param = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        result = param.is_feasible()

    assert result is True
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the undecided variable"
    assert any(repr(x) in record.getMessage() for record in warnings)


def test_is_subset_logs_warning_when_implication_is_undecided(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the optimistic subset default logs a WARNING naming the variable."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    own = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])
    other = create_integer_param(name=y)

    with caplog.at_level(logging.DEBUG, logger=_DOMAINS_LOGGER):
        result = own.is_subset(other)

    assert result is True
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the undecided comparison"
    assert any(repr(x) in record.getMessage() for record in warnings)
