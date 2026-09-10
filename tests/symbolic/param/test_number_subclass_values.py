"""Tests for parameters over a value whose type subclasses ``int`` or ``float``.

The integer domains admit an ``IntEnum`` member, which is an ``int``, and
the real domain admits a NumPy ``float64``, which is a ``float``. A value
a domain admits has to be one every query can lift into a literal: as the
parameter's own value, as a bound, and as a set-constraint member. The
literal holds the exact number the value denotes, and a set member is that
exact number too, so an answer the solver reaches from the lifted
expression agrees with membership.
"""

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import (
    IntegerDomain,
    IntervalIntegerDomain,
    Param,
    ParamAssignment,
    ParamError,
    RealDomain,
    create_integer_param,
    create_integer_param_between,
    create_intersection_param,
    create_interval_integer_param_between,
    create_real_param,
)

from .conftest import mock_identifier


class _Level(IntEnum):
    """An ``int`` subclass, which a literal holds as the ``int`` it denotes."""

    HIGH = 3


class _Measure(float):
    """A ``float`` subclass, which a literal holds as the ``float`` it denotes."""


def _create_integer_param_below_ten(variable: Identifier) -> Param[Any]:
    return create_integer_param(
        name=variable,
        constraints=[EquationConstraint(IdentifierExpression(variable) < 10)],
    )


def _create_interval_integer_param_up_to_ten(variable: Identifier) -> Param[Any]:
    return create_interval_integer_param_between(0, 10, name=variable)


def _create_real_param_below_ten(variable: Identifier) -> Param[Any]:
    return create_real_param(
        name=variable,
        constraints=[EquationConstraint(IdentifierExpression(variable) < 10.0)],
    )


# =============================================================================
# The parameter's own value
# =============================================================================


@pytest.mark.parametrize(
    ("create_param", "value"),
    [
        pytest.param(_create_integer_param_below_ten, _Level.HIGH, id="integer"),
        pytest.param(
            _create_interval_integer_param_up_to_ten,
            _Level.HIGH,
            id="interval_integer",
        ),
        pytest.param(_create_real_param_below_ten, _Measure(1.5), id="real"),
    ],
)
def test_param_validates_an_admitted_number_subclass_value(
    create_param: Callable[[Identifier], Param[Any]], value: float
) -> None:
    """Test a value the domain admits is one every validating method can lift."""
    param = create_param(mock_identifier("x", 1))

    assert param.is_value_admissible(value)
    assert param.is_value_valid(value)
    param.validate_value(value)
    assert param.assign(value).value == value
    assert ParamAssignment(param, value).value == value


def test_param_rejects_a_number_subclass_value_that_violates_a_constraint() -> None:
    """Test such a value is decided invalid rather than refused as unliftable."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x, constraints=[EquationConstraint(IdentifierExpression(x) < 3)]
    )

    assert not param.is_value_valid(_Level.HIGH)
    with pytest.raises(ParamError, match="violates"):
        param.validate_value(_Level.HIGH)


def test_real_param_validates_a_numpy_float64_value() -> None:
    """Test NumPy's ``float64``, which the real domain admits, lifts as a float."""
    np = pytest.importorskip("numpy")
    param = _create_real_param_below_ten(mock_identifier("x", 1))
    value = np.float64(1.5)

    assert param.is_value_valid(value)
    assert param.assign(value).value == value
    assert ParamAssignment(param, value).value == value


@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_real_domain_still_refuses_a_non_finite_numpy_float64(text: str) -> None:
    """Test a non-finite ``float64`` stays inadmissible, like any non-finite float."""
    np = pytest.importorskip("numpy")
    value = np.float64(text)
    param = create_real_param(name=mock_identifier("x", 1))

    assert not RealDomain().is_value_admissible(value)
    assert not param.is_value_valid(value)
    with pytest.raises(ParamError, match="not admissible"):
        param.validate_value(value)


def test_numpy_int64_is_refused_consistently() -> None:
    """Test NumPy's ``int64``, which subclasses no Python number, is refused everywhere.

    No domain admits it, no set constraint takes it as a member, and no
    constraint takes it as a binding, so it is never admitted by one path
    and refused by another.
    """
    np = pytest.importorskip("numpy")
    value = np.int64(3)
    x = mock_identifier("x", 1)
    param = _create_integer_param_below_ten(x)

    for domain in (IntegerDomain(), IntervalIntegerDomain(), RealDomain()):
        assert not domain.is_value_admissible(value)
    assert not param.is_value_valid(value)
    with pytest.raises(ParamError, match="not admissible"):
        param.validate_value(value)
    with pytest.raises(ConstraintError):
        InSetConstraint(x, {value})
    with pytest.raises(ConstraintError):
        InSetConstraint(x, {3}).evaluate_with_bindings({x: value})
    with pytest.raises(ConstraintError):
        param.constraint_system.evaluate_with_bindings({x: value})


# =============================================================================
# Bounds
# =============================================================================


def test_bound_factories_lift_an_int_subclass_bound() -> None:
    """Test a bound the factories accept as a strict integer reaches a literal.

    Each factory checks its bounds are strict integers, which an
    ``IntEnum`` member is, and then builds its bound constraints from them.
    """
    x = mock_identifier("x", 1)

    for param in (
        create_integer_param_between(0, _Level.HIGH, name=x),
        create_interval_integer_param_between(0, _Level.HIGH, name=x),
    ):
        assert param.is_value_valid(3)
        assert not param.is_value_valid(4)


def test_interval_arithmetic_lifts_an_int_subclass_operand() -> None:
    """Test interval addition takes an ``IntEnum`` operand as the ``int`` it is."""
    shifted = create_interval_integer_param_between(0, 10) + _Level.HIGH

    assert shifted.is_value_valid(13)
    assert not shifted.is_value_valid(14)


# =============================================================================
# Set-constraint members
# =============================================================================


def test_in_set_int_subclass_member_decides_feasibility_subsets_and_intersection() -> (
    None
):
    """Test the queries that lift an in-set member each decide from the number."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            InSetConstraint(x, {_Level.HIGH}),
            EquationConstraint(IdentifierExpression(x) < 10),
        ],
    )
    bounded = create_integer_param_between(0, 10, name=mock_identifier("y", 2))

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED
    assert param.check_subset(bounded) is ConstraintOutcome.SATISFIED
    assert bounded.check_subset(param) is ConstraintOutcome.VIOLATED
    intersection = create_intersection_param(
        param, bounded, name=mock_identifier("z", 3)
    )
    assert intersection.is_value_valid(3)
    assert not intersection.is_value_valid(4)


def test_in_set_float_subclass_member_decides_feasibility_and_membership() -> None:
    """Test a real parameter over a ``float`` subclass member decides from it."""
    x = mock_identifier("x", 1)
    param = create_real_param(
        name=x,
        constraints=[
            InSetConstraint(x, {_Measure(1.5)}),
            EquationConstraint(IdentifierExpression(x) < 10.0),
        ],
    )

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED
    assert param.is_value_valid(1.5)
    assert param.is_value_valid(_Measure(1.5))


def test_not_in_set_int_subclass_member_excludes_the_int_it_denotes() -> None:
    """Test feasibility and validity agree on a not-in-set subclass member.

    The member lifts to ``x != 3``, so the solver finds no value in
    ``[3, 3]``. Membership has to exclude ``3`` too, or ``3`` would be a
    valid value of a parameter decided to have none.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param_between(3, 3, name=x).add_constraint(
        NotInSetConstraint(x, {_Level.HIGH})
    )

    assert param.check_feasibility() is ConstraintOutcome.VIOLATED
    assert not param.is_value_valid(3)
    assert not param.is_value_valid(_Level.HIGH)


def test_subset_into_an_int_subclass_in_set_member_agrees_with_membership() -> None:
    """Test a subset decided through the lifted member holds for membership too.

    ``[3, 3]`` is decided a subset of ``{_Level.HIGH}`` because the member
    lifts to ``y == 3``, so the finite side has to accept ``3``, or ``3``
    would be a counterexample to a relation decided to hold.
    """
    exact = create_integer_param_between(3, 3, name=mock_identifier("x", 1))
    y = mock_identifier("y", 2)
    finite = create_integer_param(
        name=y, constraints=[InSetConstraint(y, {_Level.HIGH})]
    )

    assert exact.check_subset(finite) is ConstraintOutcome.SATISFIED
    assert exact.is_value_valid(3)
    assert finite.is_value_valid(3)
