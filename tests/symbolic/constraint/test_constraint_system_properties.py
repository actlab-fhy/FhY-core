"""Hypothesis property tests for `ConstraintSystem`.

Kept in a dedicated module so a test environment without hypothesis
installed (the CI `tests` lane syncs only the `test` dependency group)
can still collect this package cleanly: the `pytest.importorskip` below
skips the whole module before the `hypothesis` import is attempted.
"""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st

from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import BinaryOperation, make_binary_expression
from fhy_core.symbolic.symbol_type import SymbolType

from .conftest import mock_identifier

pytestmark = pytest.mark.property


@given(  # type: ignore[untyped-decorator]
    x_value=st.integers(min_value=-5, max_value=10),
    y_value=st.integers(min_value=-5, max_value=10),
)
def test_evaluate_with_bindings_matches_fold_of_member_outcomes(
    x_value: int, y_value: int
) -> None:
    """Test the conjunction outcome matches folding each member's own outcome."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    members: tuple[Constraint, ...] = (
        InSetConstraint(x, {1, 2, 3, 4}),
        InSetConstraint(y, {0, 1, 2}),
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
    )
    system = create_constraint_system(*members)
    bindings = {x: x_value, y: y_value}

    outcome = system.evaluate_with_bindings(bindings)

    member_outcomes = [member.evaluate_with_bindings(bindings) for member in members]
    if any(o is ConstraintOutcome.VIOLATED for o in member_outcomes):
        expected = ConstraintOutcome.VIOLATED
    elif all(o is ConstraintOutcome.SATISFIED for o in member_outcomes):
        expected = ConstraintOutcome.SATISFIED
    else:
        expected = ConstraintOutcome.UNDECIDED
    assert outcome is expected


@pytest.mark.z3
@given(threshold=st.integers(min_value=0, max_value=10))  # type: ignore[untyped-decorator]
def test_check_satisfiability_matches_brute_force_enumeration(threshold: int) -> None:
    """Test z3-backed satisfiability agrees with brute-force enumeration."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    domain = tuple(range(6))
    link_expression = make_binary_expression(
        BinaryOperation.EQUAL,
        make_binary_expression(BinaryOperation.ADD, x, threshold),
        y,
    )
    system = create_constraint_system(
        InSetConstraint(x, set(domain)),
        InSetConstraint(y, set(domain)),
        EquationConstraint(x, link_expression),
    )

    brute_force_satisfiable = any(a + threshold == b for a in domain for b in domain)

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    expected = (
        ConstraintOutcome.SATISFIED
        if brute_force_satisfiable
        else ConstraintOutcome.VIOLATED
    )
    assert outcome is expected
