"""Pins for API surface deliberately removed by the scope-based rewrite.

The old designated-`variable` unary contract (`evaluate(value)`,
`is_satisfied(value)`, `__call__(value)`, and a base-level `variable`
declaration) no longer exists. These tests pin its absence so a future
change cannot silently reintroduce it.
"""

import pytest

from fhy_core.symbolic.constraint import Constraint, EquationConstraint
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression

from .conftest import mock_identifier


def test_equation_constraint_rejects_zero_arguments() -> None:
    """Test `EquationConstraint()` with no arguments raises `TypeError`."""
    with pytest.raises(TypeError):
        EquationConstraint()  # type: ignore[call-arg]


def test_equation_constraint_rejects_the_old_two_argument_signature() -> None:
    """Test the old `EquationConstraint(variable, expression)` shape is rejected."""
    x = mock_identifier("x", 0)

    with pytest.raises(TypeError):
        EquationConstraint(x, LiteralExpression(True))  # type: ignore[arg-type, call-arg]  # test: old two-argument signature


def test_equation_constraint_accepts_exactly_one_positional_argument() -> None:
    """Test the constructor accepts a single positional `expression` argument."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert isinstance(constraint, EquationConstraint)


def test_equation_constraint_instance_has_no_variable_attribute() -> None:
    """Test an `EquationConstraint` instance carries no `variable` field.

    The old designated-variable attachment key is gone; a constraint's
    scope is `get_free_identifiers()`, never a single privileged field.
    """
    constraint = EquationConstraint(IdentifierExpression(mock_identifier("x", 0)))

    assert not hasattr(constraint, "variable")


def test_constraint_base_class_exposes_no_evaluate_method() -> None:
    """Test `Constraint` no longer declares a unary `evaluate` method."""
    assert not hasattr(Constraint, "evaluate")


def test_constraint_base_class_exposes_no_is_satisfied_method() -> None:
    """Test `Constraint` no longer declares a unary `is_satisfied` method."""
    assert not hasattr(Constraint, "is_satisfied")


def test_equation_constraint_instance_exposes_no_evaluate_method() -> None:
    """Test a constructed leaf instance has no `evaluate` attribute either."""
    constraint = EquationConstraint(LiteralExpression(True))

    assert not hasattr(constraint, "evaluate")
    assert not hasattr(constraint, "is_satisfied")


def test_equation_constraint_instance_is_not_callable() -> None:
    """Test a constraint instance is not callable; `__call__` sugar is removed."""
    constraint = EquationConstraint(LiteralExpression(True))

    with pytest.raises(TypeError, match="not callable"):
        constraint(True)  # type: ignore[operator]
