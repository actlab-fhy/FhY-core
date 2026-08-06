"""Tests for `Param.check_feasibility`, `Param.is_feasible`, and `Param.is_empty`.

A parameter is feasible when at least one value satisfies its domain and all of
its constraints; it is empty otherwise. Construction never rejects a jointly
empty parameter, so these tests confirm the empty constructions build before
querying feasibility. `check_feasibility` is tri-state (`True`/`False`/`None`);
`is_feasible` and `is_empty` are conservative boolean wrappers over it.
"""

import logging

import pytest

from fhy_core.symbolic.constraint import (
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import (
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_integer_param_with_lower_bound,
    create_ordinal_param,
    create_permutation_param,
)
from fhy_core.symbolic.param import domains as param_domains

from .conftest import mock_identifier

# =============================================================================
# Feasible cases
# =============================================================================


def test_unconstrained_integer_param_is_feasible() -> None:
    """Test an unconstrained integer parameter is feasible."""
    param = create_integer_param(name=mock_identifier("x", 1))

    assert param.is_feasible()
    assert not param.is_empty()


@pytest.mark.z3
def test_bounded_integer_param_is_feasible() -> None:
    """Test a bounded integer parameter with a non-empty interval is feasible."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 2))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_ordinal_param_is_feasible() -> None:
    """Test an ordinal parameter with no narrowing constraints is feasible."""
    param = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 3))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_categorical_param_is_feasible() -> None:
    """Test a categorical parameter with no narrowing constraints is feasible."""
    param = create_categorical_param(["a", "b", "c"], name=mock_identifier("x", 4))

    assert param.is_feasible()
    assert not param.is_empty()


def test_unnarrowed_permutation_param_is_feasible() -> None:
    """Test a permutation parameter with no narrowing constraints is feasible."""
    param = create_permutation_param(["a", "b", "c"], name=mock_identifier("x", 5))

    assert param.is_feasible()
    assert not param.is_empty()


def test_categorical_param_with_admitting_constraint_is_feasible() -> None:
    """Test a categorical param stays feasible when a constraint admits a value."""
    param = create_categorical_param(["a", "b", "c"], name=mock_identifier("x", 6))
    narrowed = param.add_constraint(InSetConstraint(param.variable, {"a", "b"}))

    assert narrowed.is_feasible()
    assert not narrowed.is_empty()


def test_permutation_param_with_admitting_constraint_is_feasible() -> None:
    """Test a permutation param stays feasible when a constraint admits a value."""
    param = create_permutation_param(["a", "b", "c"], name=mock_identifier("x", 7))
    narrowed = param.add_constraint(InSetConstraint(param.variable, {("a", "b", "c")}))

    assert narrowed.is_feasible()
    assert not narrowed.is_empty()


# =============================================================================
# Empty / infeasible cases
# =============================================================================


def test_categorical_param_with_out_of_domain_in_set_is_empty() -> None:
    """Test a categorical param whose in-set allows only outside values is empty."""
    param = create_categorical_param(["a", "b"], name=mock_identifier("x", 8))
    # An in-set constraint that permits only a value the domain does not admit.
    # This constructs without error; the emptiness is only discoverable by query.
    narrowed = param.add_constraint(InSetConstraint(param.variable, {"z"}))

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


@pytest.mark.z3
def test_integer_param_with_contradictory_bounds_is_empty() -> None:
    """Test a plain integer param with contradictory bounds is empty."""
    param = create_integer_param(name=mock_identifier("x", 9))
    # A plain integer domain permits arbitrary bound constraints, so these
    # jointly-empty bounds construct without error.
    narrowed = param.add_lower_bound_constraint(10).add_upper_bound_constraint(5)

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_ordinal_param_excluding_all_members_is_empty() -> None:
    """Test an ordinal param whose not-in-set excludes every member is empty."""
    param = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 10))
    narrowed = param.add_constraint(NotInSetConstraint(param.variable, {1, 2, 3}))

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_permutation_param_excluding_all_permutations_is_empty() -> None:
    """Test a permutation param excluding every permutation is empty."""
    param = create_permutation_param(["a", "b"], name=mock_identifier("x", 11))
    narrowed = param.add_constraint(
        NotInSetConstraint(param.variable, {("a", "b"), ("b", "a")})
    )

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


# =============================================================================
# Dependent (multi-variable) constraints are rejected at construction
# =============================================================================


def test_dependent_constraint_is_rejected_at_construction_not_is_feasible() -> None:
    """Test a multi-variable constraint is rejected at construction, not query time.

    ``Param`` is strictly single-variable: the numeric feasibility routing
    only supplies a Z3 sort for the parameter's own variable. A constraint
    referencing a second free identifier is rejected by `validate_constraint`
    with a typed `ParamError` before `is_feasible` can be called.
    """
    x = mock_identifier("x", 12)
    y = mock_identifier("y", 13)
    dependent_constraint = EquationConstraint(
        x, IdentifierExpression(x) < IdentifierExpression(y)
    )

    with pytest.raises(ParamError, match="ConstraintSystem"):
        create_integer_param(name=x, constraints=[dependent_constraint])


# =============================================================================
# `check_feasibility` tri-state: proven True / proven False / undecided
# =============================================================================


@pytest.mark.z3
def test_check_feasibility_returns_true_for_a_proven_feasible_parameter() -> None:
    """Test `check_feasibility` returns ``True`` when the solver proves feasibility."""
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 20))

    assert param.check_feasibility() is True


@pytest.mark.z3
def test_check_feasibility_returns_false_for_a_proven_infeasible_parameter() -> None:
    """Test `check_feasibility` returns ``False`` for a proven-infeasible parameter."""
    param = create_integer_param(name=mock_identifier("x", 21))
    narrowed = param.add_lower_bound_constraint(10).add_upper_bound_constraint(5)

    assert narrowed.check_feasibility() is False


@pytest.mark.z3
def test_check_feasibility_returns_none_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_feasibility` returns ``None`` when Z3 cannot decide.

    Mirrors the monkeypatch technique in
    `test_intersection_accepts_result_when_z3_returns_unknown`
    (`tests/symbolic/param/test_param_intersection.py`): patches
    `check_expression_satisfiability` -- the seam `_numeric_has_feasible_value`
    calls -- to always return ``None`` (Z3's "unknown" outcome).
    """
    param = create_integer_param_with_lower_bound(0, name=mock_identifier("x", 22))
    monkeypatch.setattr(
        param_domains, "check_expression_satisfiability", lambda *args, **kwargs: None
    )

    assert param.check_feasibility() is None


# =============================================================================
# `is_feasible` / `is_empty`: conservative wrappers, not complements, under
# an undecided `check_feasibility`
# =============================================================================


@pytest.mark.z3
def test_is_feasible_is_false_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `is_feasible` reports ``False`` (not proven) when Z3 cannot decide."""
    param = create_integer_param_with_lower_bound(0, name=mock_identifier("x", 23))
    monkeypatch.setattr(
        param_domains, "check_expression_satisfiability", lambda *args, **kwargs: None
    )

    assert not param.is_feasible()


@pytest.mark.z3
def test_is_empty_is_also_false_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `is_empty` is not the complement of `is_feasible` when Z3 is undecided.

    Both `is_feasible` and `is_empty` report ``False`` for the same undecided
    parameter: neither claims a definite answer without proof.
    """
    param = create_integer_param_with_lower_bound(0, name=mock_identifier("x", 24))
    monkeypatch.setattr(
        param_domains, "check_expression_satisfiability", lambda *args, **kwargs: None
    )

    assert not param.is_feasible()
    assert not param.is_empty()


# =============================================================================
# Z3 `unknown` diagnostic logging
# =============================================================================


@pytest.mark.z3
def test_check_feasibility_logs_debug_record_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test `check_feasibility` logs a debug record when Z3 cannot decide.

    The logged record is diagnostic only: it must not claim that feasibility
    was assumed, since `check_feasibility` reports the undecided outcome
    honestly as ``None`` rather than collapsing it to a guess.
    """
    param = create_integer_param_with_lower_bound(0, name=mock_identifier("x", 25))
    monkeypatch.setattr(
        param_domains, "check_expression_satisfiability", lambda *args, **kwargs: None
    )

    with caplog.at_level(logging.DEBUG, logger="fhy_core.symbolic.param.domains"):
        result = param.check_feasibility()

    assert result is None
    assert any(
        record.levelno == logging.DEBUG and "unknown" in record.getMessage().lower()
        for record in caplog.records
    ), "expected a DEBUG-level log record naming the unknown outcome"
    assert not any(
        "assum" in record.getMessage().lower() for record in caplog.records
    ), "the log must not claim feasibility was assumed"
