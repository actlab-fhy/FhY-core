"""Tests for bindings-aware `Param` validation of dependent constraints.

A dependent constraint's scope extends beyond the owning parameter's own
variable, so deciding it needs values for the other identifiers it
references. `validate_value`, `is_value_valid`, `is_constraints_satisfied`,
and `assign` all gain a keyword-only `bindings` mapping that supplies those
values; the evaluation environment is `{self.variable: value}` merged with
`bindings`. Without bindings a dependent constraint stays UNDECIDED, which
these methods conservatively treat as invalid/unsatisfied, mirroring the
existing conservative treatment of any undecided constraint.
"""

from collections.abc import Callable

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import EquationConstraint
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import Param, ParamError, create_integer_param

from .conftest import mock_identifier


def _build_dependent_param() -> tuple[Param[int], Identifier, Identifier]:
    """Build an integer param whose sole constraint is the dependent `x < y`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])
    return param, x, y


# =============================================================================
# `is_value_valid` / `is_constraints_satisfied` with bindings
# =============================================================================


def test_is_value_valid_without_bindings_is_false_for_dependent_constraint() -> None:
    """Test an unbound dependent constraint conservatively fails `is_value_valid`."""
    param, _x, _y = _build_dependent_param()

    assert not param.is_value_valid(3)


def test_is_value_valid_with_satisfying_binding_is_true() -> None:
    """Test a binding that satisfies the dependent constraint validates the value."""
    param, _x, y = _build_dependent_param()

    assert param.is_value_valid(3, bindings={y: 5})


def test_is_value_valid_with_violating_binding_is_false() -> None:
    """Test a binding that violates the dependent constraint invalidates the value."""
    param, _x, y = _build_dependent_param()

    assert not param.is_value_valid(3, bindings={y: 2})


def test_is_constraints_satisfied_uses_bindings_to_decide_dependent_constraint() -> (
    None
):
    """Test `is_constraints_satisfied` decides a dependent constraint via bindings."""
    param, _x, y = _build_dependent_param()

    assert not param.is_constraints_satisfied(3)
    assert param.is_constraints_satisfied(3, bindings={y: 5})
    assert not param.is_constraints_satisfied(3, bindings={y: 2})


@pytest.mark.parametrize(
    ("value", "extra_binding_value", "expect_valid"),
    [
        pytest.param(3, 5, True, id="satisfying-with-irrelevant-binding"),
        pytest.param(3, 2, False, id="violating-with-irrelevant-binding"),
    ],
)
def test_bindings_for_unreferenced_identifiers_are_ignored(
    value: int, extra_binding_value: int, expect_valid: bool
) -> None:
    """Test a binding for an identifier outside the constraint's scope is ignored.

    An unrelated third identifier `z` is bound alongside `y`; the outcome
    must depend only on the `y` binding, since `z` never appears in the
    dependent constraint's scope.
    """
    param, _x, y = _build_dependent_param()
    z = mock_identifier("z", 3)

    result = param.is_value_valid(value, bindings={y: extra_binding_value, z: 999})

    assert result is expect_valid


# =============================================================================
# `validate_value` message distinguishes undecided from violated
# =============================================================================


def test_validate_value_accepts_value_when_binding_proves_satisfaction() -> None:
    """Test `validate_value` does not raise once a binding decides the constraint."""
    param, _x, y = _build_dependent_param()

    param.validate_value(3, bindings={y: 5})


def test_validate_value_raises_could_not_be_verified_without_bindings() -> None:
    """Test `validate_value` reports an undecided dependent constraint distinctly."""
    param, _x, _y = _build_dependent_param()

    with pytest.raises(ParamError, match="could not be verified"):
        param.validate_value(3)


def test_validate_value_raises_violates_with_a_violating_binding() -> None:
    """Test `validate_value` reports a binding-proven violation distinctly."""
    param, _x, y = _build_dependent_param()

    with pytest.raises(ParamError, match="violates constraint"):
        param.validate_value(3, bindings={y: 2})


def test_validate_value_message_distinguishes_undecided_from_violated() -> None:
    """Test the undecided and violated error messages do not cross-match."""
    param, _x, y = _build_dependent_param()

    with pytest.raises(ParamError) as undecided_info:
        param.validate_value(3)
    with pytest.raises(ParamError) as violated_info:
        param.validate_value(3, bindings={y: 2})

    assert "could not be verified" not in str(violated_info.value)
    assert "violates constraint" not in str(undecided_info.value)


# =============================================================================
# Rebinding the parameter's own variable is ambiguous
# =============================================================================


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda param, x, value: param.validate_value(value, bindings={x: 5}),
            id="validate_value",
        ),
        pytest.param(
            lambda param, x, value: param.is_value_valid(value, bindings={x: 5}),
            id="is_value_valid",
        ),
        pytest.param(
            lambda param, x, value: param.is_constraints_satisfied(
                value, bindings={x: 5}
            ),
            id="is_constraints_satisfied",
        ),
        pytest.param(
            lambda param, x, value: param.assign(value, bindings={x: 5}),
            id="assign",
        ),
    ],
)
def test_bindings_for_own_variable_raises_param_error(
    call: Callable[[Param[int], Identifier, int], object],
) -> None:
    """Test a `bindings` entry for the parameter's own variable is rejected.

    Binding the parameter's own variable through `bindings` while also
    passing it as the positional `value` is an ambiguous call the caller
    should never make; every bindings-aware method raises `ParamError`
    instead of silently picking one of the two.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x)

    with pytest.raises(ParamError):
        call(param, x, 3)


# =============================================================================
# `assign` with bindings
# =============================================================================


def test_assign_with_satisfying_bindings_matches_plain_assignment_value() -> None:
    """Test `assign` with bindings yields the same value as an unconstrained assign."""
    dependent, _x, y = _build_dependent_param()
    independent = create_integer_param(name=mock_identifier("x", 1))

    dependent_assignment = dependent.assign(3, bindings={y: 5})
    independent_assignment = independent.assign(3)

    assert dependent_assignment.value == independent_assignment.value == 3


def test_assign_without_bindings_raises_for_dependent_constraint() -> None:
    """Test `assign` raises `ParamError` when the dependent constraint is undecided."""
    param, _x, _y = _build_dependent_param()

    with pytest.raises(ParamError):
        param.assign(3)


def test_assign_with_violating_binding_raises() -> None:
    """Test `assign` raises `ParamError` when a binding proves a violation."""
    param, _x, y = _build_dependent_param()

    with pytest.raises(ParamError):
        param.assign(3, bindings={y: 2})
