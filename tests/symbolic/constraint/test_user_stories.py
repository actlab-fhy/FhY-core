"""User-story tests for the scope-based constraint API.

Each test walks an end-to-end scenario a real caller would run, rather
than exercising one method in isolation. Param-layer stories (a
dependent-parameter scenario validated jointly via bindings) belong to
`tests/symbolic/param/**`, out of this sub-package's scope; these stories
stay entirely within `fhy_core.symbolic.constraint`.
"""

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    SymbolicPredicate,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType

from .conftest import mock_identifier


def test_dependent_constraint_scenario_matches_the_documented_walkthrough() -> None:
    """Test the design doc's dependent-constraint example end to end.

    A dependent constraint (`x < y`) is a first-class `Constraint`: it has
    no designated variable, its scope is both identifiers, and its
    outcome tracks whichever of `x`/`y` bindings are supplied.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    c = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))

    assert c.get_free_identifiers() == frozenset({x, y})
    assert c.evaluate_with_bindings({x: 3, y: 5}) is ConstraintOutcome.SATISFIED
    assert c.evaluate_with_bindings({x: 3}) is ConstraintOutcome.UNDECIDED
    assert c.evaluate_with_bindings({x: 5, y: 3}) is ConstraintOutcome.VIOLATED


def test_unary_set_constraint_scenario_matches_the_documented_walkthrough() -> None:
    """Test the design doc's `InSetConstraint` example end to end."""
    x = mock_identifier("x", 0)
    s = InSetConstraint(x, {1, 2, 3})

    assert s.get_free_identifiers() == frozenset({x})
    assert s.is_satisfied_with_bindings({x: 2}) is True
    assert s.is_satisfied_with_bindings({x: True}) is False
    assert s.evaluate_with_bindings({}) is ConstraintOutcome.UNDECIDED


def test_one_generic_consumer_handles_both_a_leaf_and_a_system() -> None:
    """Test a single function typed over `SymbolicPredicate` serves both shapes."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    c = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    s = InSetConstraint(x, {1, 2, 3})
    system = create_constraint_system(c, s)

    def count_undecided(predicate: SymbolicPredicate) -> bool:
        return predicate.evaluate_with_bindings({}) is ConstraintOutcome.UNDECIDED

    assert count_undecided(c) is True
    assert count_undecided(system) is True


@pytest.mark.z3
def test_chain_of_three_dependent_inequalities_is_jointly_satisfiable() -> None:
    """Test a realistic three-variable chain of dependent inequalities.

    Mirrors how a caller would model `x < y < z < 100`: three
    `EquationConstraint`s over overlapping scopes, checked jointly as one
    `ConstraintSystem`, then narrowed by partially binding one variable.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, z)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, z, 100)),
    )
    symbol_types = {x: SymbolType.INT, y: SymbolType.INT, z: SymbolType.INT}

    assert system.check_satisfiability(symbol_types) is ConstraintOutcome.SATISFIED
    # Narrowing x to a value near the ceiling leaves no room for y, z.
    outcome = system.check_satisfiability_with_bindings(
        {x: 99}, {y: SymbolType.INT, z: SymbolType.INT}
    )
    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_migration_shaped_scenario_matches_the_old_suites_outcomes() -> None:
    """Test a system shaped like the pre-rewrite suite's examples decides the same way.

    Builds the same mixed set-and-equation system the old
    `test_constraint_system.py` used to pin
    (`test_check_satisfiability_mixed_set_and_equation_system_is_satisfiable`)
    through the new constructor shapes, and confirms the satisfiability
    outcome is unchanged: this behavior was never variable-attachment
    dependent, so the rewrite must not have disturbed it.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2, 3}),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


def test_ground_constraint_scenario_matches_the_documented_edge_case() -> None:
    """Test a ground equation constraint's documented edge-case behavior.

    A constraint with no free identifiers has an empty scope and is
    decidable under empty bindings -- the case the design doc calls out
    as valid in a bare `ConstraintSystem` even though a `Param` would
    reject attaching it (scope-membership rule, tested at the param layer).
    """
    ground = EquationConstraint(LiteralExpression(True))

    assert ground.get_free_identifiers() == frozenset()
    assert ground.evaluate_with_bindings({}) is ConstraintOutcome.SATISFIED
    assert not create_constraint_system(ground).get_free_identifiers()
