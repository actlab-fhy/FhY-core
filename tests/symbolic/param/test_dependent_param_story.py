"""User-story test: dependent tile-size parameters linked by shared constraints.

Mirrors a real caller building two related integer parameters -- tile
dimensions `x` and `y` -- whose joint admissibility only makes sense as a
system: `x * y <= 64` (a memory-budget bound) and `x < y` (a shape
preference). Neither constraint is decidable from either parameter alone;
this is the dependent-constraint story the scope-based rewrite makes first
class. Both constraints stay inside decidable arithmetic (multiplication and
strict ordering only, no division/modulo, no bool coercion, no mixed
int/float equality), so the joint system is confidently SATISFIED or
VIOLATED rather than UNDECIDED.
"""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    ConstraintOutcome,
    EquationConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import Param, create_integer_param
from fhy_core.symbolic.symbol_type import SymbolType

from .conftest import mock_identifier

pytestmark = pytest.mark.integration


def _build_tile_size_params() -> tuple[Param[int], Param[int]]:
    """Build two dependent tile-size params sharing a budget and a shape constraint."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    budget = EquationConstraint(
        (IdentifierExpression(x) * IdentifierExpression(y)) <= 64
    )
    shape = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    x_param = create_integer_param(name=x, constraints=[budget, shape])
    y_param = create_integer_param(name=y, constraints=[budget, shape])
    return x_param, y_param


def test_shared_constraints_attach_to_both_dependent_tile_params() -> None:
    """Test the shared budget and shape constraints attach to both tile params."""
    x_param, y_param = _build_tile_size_params()

    assert len(x_param.constraints) == 2
    assert len(y_param.constraints) == 2


@pytest.mark.parametrize(
    ("x_value", "y_value"),
    [pytest.param(4, 8, id="4-by-8"), pytest.param(2, 16, id="2-by-16")],
)
def test_satisfying_tile_candidate_validates_jointly_via_bindings(
    x_value: int, y_value: int
) -> None:
    """Test a satisfying `(x, y)` candidate validates through cross-param bindings."""
    x_param, y_param = _build_tile_size_params()

    assert x_param.is_value_valid(x_value, bindings={y_param.variable: y_value})
    assert y_param.is_value_valid(y_value, bindings={x_param.variable: x_value})


@pytest.mark.parametrize(
    ("x_value", "y_value"),
    [pytest.param(10, 20, id="over-budget"), pytest.param(8, 2, id="wrong-shape")],
)
def test_violating_tile_candidate_fails_joint_validation_via_bindings(
    x_value: int, y_value: int
) -> None:
    """Test a violating `(x, y)` candidate fails cross-param binding validation."""
    x_param, y_param = _build_tile_size_params()

    assert not x_param.is_value_valid(x_value, bindings={y_param.variable: y_value})
    assert not y_param.is_value_valid(y_value, bindings={x_param.variable: x_value})


@pytest.mark.z3
def test_joint_tile_size_system_is_satisfiable_over_integers() -> None:
    """Test the shared tile-size system is satisfiable as a whole."""
    x_param, y_param = _build_tile_size_params()
    system = create_constraint_system(*x_param.constraints, *y_param.constraints)
    symbol_types: dict[Identifier, SymbolType] = {
        x_param.variable: SymbolType.INT,
        y_param.variable: SymbolType.INT,
    }

    outcome = system.check_satisfiability(symbol_types)

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_joint_tile_size_system_becomes_violated_after_adding_x_over_budget() -> None:
    """Test adding `x > 64` makes the joint tile-size system unsatisfiable.

    Combined with `x < y`, `x > 64` forces `y > 64` too, which contradicts
    `x * y <= 64`.
    """
    x_param, y_param = _build_tile_size_params()
    x_over_budget = EquationConstraint(IdentifierExpression(x_param.variable) > 64)
    system = create_constraint_system(
        *x_param.constraints, *y_param.constraints, x_over_budget
    )
    symbol_types: dict[Identifier, SymbolType] = {
        x_param.variable: SymbolType.INT,
        y_param.variable: SymbolType.INT,
    }

    outcome = system.check_satisfiability(symbol_types)

    assert outcome is ConstraintOutcome.VIOLATED
