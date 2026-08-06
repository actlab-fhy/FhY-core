"""Pickle and deepcopy round-trips for Identifier-holding frozen types.

These tests deliberately use real ``Identifier`` instances rather than the
conftest ``mock_identifier``: restoring a pickled real ``Identifier`` inside
a frozen container is the behavior under test, and ``unittest.mock`` objects
do not survive pickling.
"""

import copy
import pickle

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression
from fhy_core.symbolic.param import create_integer_param


def _make_constraint() -> EquationConstraint:
    variable = Identifier("x")
    return EquationConstraint(
        variable, IdentifierExpression(variable) < LiteralExpression(5)
    )


def test_equation_constraint_round_trips_through_pickle() -> None:
    """Test a constraint holding a real identifier survives pickling frozen."""
    constraint = _make_constraint()

    restored = pickle.loads(pickle.dumps(constraint))

    assert restored.variable == constraint.variable
    assert restored.is_frozen
    assert restored.is_structurally_equivalent(constraint)


def test_equation_constraint_round_trips_through_deepcopy() -> None:
    """Test a constraint holding a real identifier survives deep copy frozen."""
    constraint = _make_constraint()

    duplicate = copy.deepcopy(constraint)

    assert duplicate is not constraint
    assert duplicate.variable == constraint.variable
    assert duplicate.is_frozen
    assert duplicate.is_structurally_equivalent(constraint)


def test_param_round_trips_through_pickle() -> None:
    """Test an integer parameter survives pickling with its binder intact."""
    param = create_integer_param(name=Identifier("p"))

    restored = pickle.loads(pickle.dumps(param))

    assert restored.variable == param.variable
    assert restored.is_frozen
    assert restored.is_structurally_equivalent(param)


def test_param_round_trips_through_deepcopy() -> None:
    """Test an integer parameter survives deep copy with its binder intact."""
    param = create_integer_param(name=Identifier("p"))

    duplicate = copy.deepcopy(param)

    assert duplicate is not param
    assert duplicate.variable == param.variable
    assert duplicate.is_frozen
    assert duplicate.is_structurally_equivalent(param)
