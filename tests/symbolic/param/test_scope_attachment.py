"""Tests for scope-based `Constraint` attachment on `Param`.

Under the scope-based rewrite, a constraint no longer carries a designated
`variable`; instead a constraint attaches to a parameter exactly when the
parameter's variable is a member of the constraint's scope
(`get_free_identifiers()`). These tests exercise that attachment rule for
equation constraints (including dependent, multi-variable ones and ground,
variable-free ones) and confirm set constraints keep their simpler
single-variable attachment rule. They also pin the constraint tuple's
canonical order to the new public `build_constraint_ordering_key`.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.symbolic.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression, LiteralExpression
from fhy_core.symbolic.param import Param, ParamError, create_integer_param

from .conftest import mock_identifier

_SET_CONSTRAINT_TYPES = [
    pytest.param(InSetConstraint, id="in_set"),
    pytest.param(NotInSetConstraint, id="not_in_set"),
]

# =============================================================================
# EquationConstraint: dependent (multi-variable) attachment by scope
# =============================================================================


def test_dependent_equation_constraint_attaches_via_constructor() -> None:
    """Test a dependent equation constraint attaches when its variable is in scope."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))

    param = create_integer_param(name=x, constraints=[dependent])

    assert dependent in param.constraints


def test_dependent_equation_constraint_attaches_via_add_constraint() -> None:
    """Test `add_constraint` accepts a dependent equation constraint by scope."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x)

    updated = param.add_constraint(dependent)

    assert dependent in updated.constraints


@pytest.mark.parametrize(
    "attach",
    [
        pytest.param(
            lambda param, constraint: create_integer_param(
                name=param.variable, constraints=[constraint]
            ),
            id="constructor",
        ),
        pytest.param(
            lambda param, constraint: param.add_constraint(constraint),
            id="add_constraint",
        ),
    ],
)
def test_equation_constraint_with_foreign_only_scope_is_rejected(
    attach: Callable[[Param[Any], Constraint], Param[Any]],
) -> None:
    """Test a constraint whose scope excludes the parameter's variable is rejected."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    foreign_only = EquationConstraint(IdentifierExpression(y) < IdentifierExpression(z))
    param = create_integer_param(name=x)

    with pytest.raises(ParamError):
        attach(param, foreign_only)


def test_validate_constraint_rejects_foreign_only_scope_directly() -> None:
    """Test `validate_constraint` itself rejects a foreign-only-scope constraint."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    foreign_only = EquationConstraint(IdentifierExpression(y) < IdentifierExpression(z))
    param = create_integer_param(name=x)

    with pytest.raises(ParamError):
        param.validate_constraint(foreign_only)


# =============================================================================
# EquationConstraint: ground (variable-free) constraints
# =============================================================================


def test_ground_equation_constraint_is_rejected_on_attach() -> None:
    """Test a ground (variable-free) equation constraint cannot attach to a param.

    A constraint that references no variable constrains nothing, so its
    empty scope never contains the parameter's variable.
    """
    x = mock_identifier("x", 1)
    ground = EquationConstraint(LiteralExpression(True))
    param = create_integer_param(name=x)

    with pytest.raises(ParamError):
        param.add_constraint(ground)


def test_ground_equation_constraint_has_empty_scope() -> None:
    """Test a ground equation constraint reports an empty free-identifier scope."""
    ground = EquationConstraint(LiteralExpression(True))

    assert ground.get_free_identifiers() == frozenset()


# =============================================================================
# Set constraints: single-variable attachment
# =============================================================================


@pytest.mark.parametrize("constraint_type", _SET_CONSTRAINT_TYPES)
def test_set_constraint_attaches_when_variable_matches_param(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test a set constraint attaches exactly when its variable is the param's."""
    x = mock_identifier("x", 1)
    matching = constraint_type(x, {1, 2, 3})
    param = create_integer_param(name=x)

    updated = param.add_constraint(matching)

    assert matching in updated.constraints


@pytest.mark.parametrize("constraint_type", _SET_CONSTRAINT_TYPES)
def test_set_constraint_rejected_when_variable_differs_from_param(
    constraint_type: type[InSetConstraint] | type[NotInSetConstraint],
) -> None:
    """Test a set constraint over a different variable is rejected on attach."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    mismatched = constraint_type(y, {1, 2, 3})
    param = create_integer_param(name=x)

    with pytest.raises(ParamError):
        param.add_constraint(mismatched)


# =============================================================================
# Canonical constraint ordering
# =============================================================================


def test_param_constraint_tuple_matches_build_constraint_ordering_key_order() -> None:
    """Test a param's constraint tuple order matches `build_constraint_ordering_key`.

    Builds the same three constraints in two different insertion orders and
    confirms both parameters converge on the same order: the order the
    public ordering key independently derives. `build_constraint_ordering_key`
    is imported locally because it does not exist on the current module
    surface; a module-level import would break collection of this whole
    file rather than just this test.
    """
    # test: not on the module surface yet
    from fhy_core.symbolic.constraint import (  # noqa: PLC0415
        build_constraint_ordering_key,
    )

    x_first = mock_identifier("x", 1)
    x_second = mock_identifier("x", 1)
    in_set = InSetConstraint(x_first, {3, 4})
    lower = EquationConstraint(IdentifierExpression(x_first) >= 0)
    upper = EquationConstraint(IdentifierExpression(x_first) <= 10)

    forward = create_integer_param(name=x_first).add_constraints([in_set, lower, upper])
    reverse = create_integer_param(name=x_second).add_constraints(
        [upper, lower, in_set]
    )

    expected_order = tuple(
        sorted((in_set, lower, upper), key=build_constraint_ordering_key)
    )
    assert forward.constraints == expected_order
    assert reverse.constraints == expected_order
