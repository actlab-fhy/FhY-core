"""Tests for the multi-variable bindings evaluation API on `Constraint`."""

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    call,
    make_binary_expression,
)
from fhy_core.utils.override import override

from .conftest import ALL_KINDS, SET_KINDS, mock_identifier

SetConstraintFactory = Callable[[Identifier, object], Constraint]
ConstraintFactory = Callable[[Identifier], Constraint]

# =============================================================================
# Base-default bindings evaluation (set constraints)
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_bound_value_matches_evaluate(
    factory: SetConstraintFactory,
) -> None:
    """Test a bound value under the constrained variable matches `evaluate`."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({x: 2}) == constraint.evaluate(2)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_missing_variable_is_undecided(
    factory: SetConstraintFactory,
) -> None:
    """Test a bindings mapping missing the constrained variable is UNDECIDED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({y: 2}) is ConstraintOutcome.UNDECIDED


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_unwraps_literal_expression_binding(
    factory: SetConstraintFactory,
) -> None:
    """Test a `LiteralExpression`-valued binding unwraps to match `evaluate`."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings(
        {x: LiteralExpression(2)}
    ) == constraint.evaluate(2)


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_symbolic_expression_binding_is_undecided(
    factory: SetConstraintFactory,
) -> None:
    """Test a non-literal `Expression` binding cannot be decided by the default."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    outcome = constraint.evaluate_with_bindings({x: IdentifierExpression(y)})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_ignores_extraneous_keys(
    factory: SetConstraintFactory,
) -> None:
    """Test identifiers the constraint does not reference are ignored."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({x: 2, y: 999}) == constraint.evaluate(2)


def test_in_set_constraint_bindings_are_type_strict_for_bool_vs_int() -> None:
    """Test a bound `True` is distinct from a member `1` under type-strict rules."""
    x = mock_identifier("x", 0)
    constraint = InSetConstraint(x, {1})

    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.VIOLATED


def test_not_in_set_constraint_bindings_are_type_strict_for_bool_vs_int() -> None:
    """Test a bound `True` is distinct from a forbidden member `1`."""
    x = mock_identifier("x", 0)
    constraint = NotInSetConstraint(x, {1})

    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.SATISFIED


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_propagates_type_error_for_unhashable_value(
    factory: SetConstraintFactory,
) -> None:
    """Test an unhashable bound value propagates `TypeError`, matching `evaluate`."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    with pytest.raises(TypeError):
        constraint.evaluate_with_bindings({x: [1, 2]})  # type: ignore[dict-item]


# =============================================================================
# Base-default `evaluate_with_bindings` reads the mapping once (snapshot)
# =============================================================================


class _SingleReadMapping(Mapping[Identifier, Any]):
    """A mapping that raises if any key is read from it more than once.

    Used to prove the base ``evaluate_with_bindings`` default takes a
    single snapshot of the caller's mapping rather than performing a
    membership check and a separate lookup against the live mapping
    (which would read the same key twice).
    """

    def __init__(self, data: dict[Identifier, Any]) -> None:
        self._data = dict(data)
        self._read_counts: dict[Identifier, int] = {}

    @override
    def __iter__(self) -> Iterator[Identifier]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, key: Identifier) -> Any:
        self._read_counts[key] = self._read_counts.get(key, 0) + 1
        if self._read_counts[key] > 1:
            raise AssertionError(f"{key!r} was read more than once from the mapping")
        return self._data[key]


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_bindings_reads_the_mapping_only_once(
    factory: SetConstraintFactory,
) -> None:
    """Test the base default snapshots the mapping instead of re-reading it."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})
    bindings = _SingleReadMapping({x: 2})

    outcome = constraint.evaluate_with_bindings(bindings)

    assert outcome == constraint.evaluate(2)


# =============================================================================
# `EquationConstraint.evaluate_with_bindings` override
# =============================================================================


def _build_sum_less_than_ten(x: Identifier, y: Identifier) -> EquationConstraint:
    expression = make_binary_expression(
        BinaryOperation.LESS,
        make_binary_expression(BinaryOperation.ADD, x, y),
        10,
    )
    return EquationConstraint(x, expression)


def test_equation_constraint_bindings_full_assignment_satisfied() -> None:
    """Test a full multi-variable assignment that holds reports SATISFIED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = _build_sum_less_than_ten(x, y)

    outcome = constraint.evaluate_with_bindings({x: 3, y: 5})

    assert outcome is ConstraintOutcome.SATISFIED


def test_equation_constraint_bindings_full_assignment_violated() -> None:
    """Test a full multi-variable assignment that fails reports VIOLATED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = _build_sum_less_than_ten(x, y)

    outcome = constraint.evaluate_with_bindings({x: 20, y: 1})

    assert outcome is ConstraintOutcome.VIOLATED


def test_equation_constraint_bindings_partial_assignment_is_undecided() -> None:
    """Test binding only one of two free identifiers is UNDECIDED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = _build_sum_less_than_ten(x, y)

    outcome = constraint.evaluate_with_bindings({x: 3})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_equation_constraint_bindings_empty_undecided_for_open_expression() -> None:
    """Test an empty bindings mapping is UNDECIDED for an open expression."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = _build_sum_less_than_ten(x, y)

    outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_equation_constraint_bindings_empty_satisfied_for_closed_expression() -> None:
    """Test an empty bindings mapping decides a closed (variable-free) expression."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(True))

    outcome = constraint.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.SATISFIED


def test_equation_constraint_bindings_symbolic_binding_can_decide() -> None:
    """Test a symbolic (non-literal) binding can still decide the outcome."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = make_binary_expression(BinaryOperation.GREATER, x, y)
    constraint = EquationConstraint(x, expression)
    successor_of_y = make_binary_expression(
        BinaryOperation.ADD, IdentifierExpression(y), 1
    )

    outcome = constraint.evaluate_with_bindings({x: successor_of_y})

    assert outcome is ConstraintOutcome.SATISFIED


def test_equation_constraint_bindings_ignores_extraneous_keys() -> None:
    """Test bindings for identifiers outside the expression do not affect the result."""
    x = mock_identifier("x", 0)
    z = mock_identifier("z", 2)
    constraint = EquationConstraint(x, LiteralExpression(True))

    outcome = constraint.evaluate_with_bindings({z: 999})

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# DEBUG-vs-WARNING logging split
# =============================================================================


def test_equation_constraint_bindings_logs_debug_when_free_identifiers_unbound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test partial bindings log at DEBUG, not WARNING."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = _build_sum_less_than_ten(x, y)

    with caplog.at_level(logging.DEBUG, logger="fhy_core.symbolic.constraint"):
        outcome = constraint.evaluate_with_bindings({x: 3})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "fhy_core.symbolic.constraint"
        for record in caplog.records
    ), "expected a DEBUG-level log record from fhy_core.symbolic.constraint"
    assert not any(record.levelno == logging.WARNING for record in caplog.records), (
        "a partial (expected) UNDECIDED case must not log a warning"
    )


def test_equation_constraint_bindings_logs_debug_for_symbolic_residual(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a symbolic binding that leaves the residual open logs at DEBUG.

    Every free identifier of the *original* expression is bound here, but
    the bound value is itself a symbolic ``Expression`` referencing a new
    identifier, so the substituted-and-simplified residual still has a
    free identifier. This is classified the same as an ordinary partial
    assignment (DEBUG), not the fully-grounded anomaly case (WARNING).
    """
    x = mock_identifier("x", 0)
    w = mock_identifier("w", 1)
    constraint = EquationConstraint(
        x, make_binary_expression(BinaryOperation.LESS, x, 10)
    )

    with caplog.at_level(logging.DEBUG, logger="fhy_core.symbolic.constraint"):
        outcome = constraint.evaluate_with_bindings({x: IdentifierExpression(w)})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert any(
        record.levelno == logging.DEBUG
        and record.name == "fhy_core.symbolic.constraint"
        for record in caplog.records
    ), "expected a DEBUG-level log record from fhy_core.symbolic.constraint"
    assert not any(record.levelno == logging.WARNING for record in caplog.records), (
        "a residual that still has a free identifier must not log a warning"
    )


def test_equation_constraint_bindings_logs_warning_when_fully_bound_but_irreducible(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a fully bound, fully closed, yet irreducible residual logs at WARNING.

    ``arcsin`` is not registered as a native-constant-foldable function
    for an out-of-domain argument, so ``simplify_expression`` returns an
    unevaluated ``CallExpression`` with no free identifiers at all: every
    free identifier was bound, but the simplifier still failed to reduce
    the residual to a literal -- the genuine anomaly case.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, call("arcsin", IdentifierExpression(y)))

    with caplog.at_level(logging.DEBUG, logger="fhy_core.symbolic.constraint"):
        outcome = constraint.evaluate_with_bindings({y: 2})

    assert outcome is ConstraintOutcome.UNDECIDED
    assert any(
        record.levelno == logging.WARNING
        and record.name == "fhy_core.symbolic.constraint"
        for record in caplog.records
    ), "expected a WARNING-level log record from fhy_core.symbolic.constraint"
    assert not any(record.levelno == logging.DEBUG for record in caplog.records), (
        "a fully-bound, fully-closed irreducible residual must not log at debug"
    )


# =============================================================================
# `evaluate` / `evaluate_with_bindings` equivalence
# =============================================================================


@pytest.mark.parametrize("factory", ALL_KINDS)
@pytest.mark.parametrize("value", [1, True, 2.5])
def test_evaluate_matches_evaluate_with_bindings_for_raw_value(
    factory: ConstraintFactory, value: LiteralType
) -> None:
    """Test `evaluate(v)` matches `evaluate_with_bindings({variable: v})`."""
    x = mock_identifier("x", 0)
    constraint = factory(x)

    assert constraint.evaluate(value) == constraint.evaluate_with_bindings({x: value})


def test_equation_constraint_evaluate_matches_bindings_for_expression_value() -> None:
    """Test the equivalence also holds when the single binding is an `Expression`."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, IdentifierExpression(x))
    value = LiteralExpression(True)

    assert constraint.evaluate(value) == constraint.evaluate_with_bindings({x: value})


# =============================================================================
# `is_satisfied_with_bindings`
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_is_satisfied_with_bindings_folds_undecided_to_false(
    factory: SetConstraintFactory,
) -> None:
    """Test an UNDECIDED bindings outcome maps to `False`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({y: 2}) is ConstraintOutcome.UNDECIDED
    assert constraint.is_satisfied_with_bindings({y: 2}) is False


def test_equation_constraint_is_satisfied_with_bindings_folds_undecided_to_false() -> (
    None
):
    """Test an UNDECIDED bindings outcome maps to `False` for an equation."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, IdentifierExpression(y))

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.UNDECIDED
    assert constraint.is_satisfied_with_bindings({}) is False


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_is_satisfied_with_bindings_true_when_satisfied(
    factory: SetConstraintFactory,
) -> None:
    """Test a decided SATISFIED bindings outcome maps to `True`."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})

    assert constraint.evaluate_with_bindings({x: 2}) == constraint.evaluate(2)
    assert constraint.is_satisfied_with_bindings({x: 2}) == constraint.is_satisfied(2)


# =============================================================================
# `get_free_identifiers`
# =============================================================================


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_free_identifiers_is_just_the_variable(
    factory: SetConstraintFactory,
) -> None:
    """Test a set constraint's free identifiers is exactly its variable."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})

    assert constraint.get_free_identifiers() == frozenset({x})


def test_equation_constraint_free_identifiers_unions_variable_and_expression() -> None:
    """Test an equation's free identifiers include the variable and the expression."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, IdentifierExpression(y))

    assert constraint.get_free_identifiers() == frozenset({x, y})


def test_equation_constraint_free_identifiers_include_absent_variable() -> None:
    """Test the variable is always included even when absent from the expression."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(True))

    assert constraint.get_free_identifiers() == frozenset({x})
