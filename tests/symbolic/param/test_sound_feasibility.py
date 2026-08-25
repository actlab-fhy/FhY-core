"""Integration tests for sound `Param` feasibility, emptiness, and subset decisions.

Covers the audit-C6 reproductions (feasibility/subset for type-strict set
members that Python's `==` conflates but this package's domains do not),
screened equation-constraint feasibility, and foreign-identifier
degradation for dependent constraints. `Param.is_feasible`/`is_empty`/
`is_subset` stay boolean; what changes is that the boolean answer is now
provably correct, or an honestly documented optimistic default, instead of
a provably wrong decided answer.
"""

import pytest

from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.expression import IdentifierExpression
from fhy_core.symbolic.param import create_integer_param, create_integer_param_between

from .conftest import mock_identifier

# =============================================================================
# C6: type-strict set-constraint feasibility, decided by enumeration
# =============================================================================


def test_integer_param_restricted_to_bool_literal_is_infeasible() -> None:
    """Test an integer param restricted to `{True}` is infeasible.

    `True` is not a strict `int` even though Python considers `True == 1`;
    enumeration over the domain's admissible members must decide this
    without lowering to the solver.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[InSetConstraint(x, {True})])

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_param_restricted_to_float_literal_is_infeasible() -> None:
    """Test an integer param restricted to `{1.0}` is infeasible."""
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[InSetConstraint(x, {1.0})])

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_param_restricted_to_string_member_is_infeasible_without_raising() -> (
    None
):
    """Test a string set member on an integer param is infeasible, not a crash.

    `InSetConstraint.convert_to_expression` raises `ConstraintError` for a
    string member; the enumeration-based feasibility path must decide
    membership directly against the domain instead of routing through it.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[InSetConstraint(x, {"5"})])

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_singleton_one_is_not_subset_of_integer_singleton_float_one() -> None:
    """Test integer `{1}` is not a subset of integer `{1.0}`.

    `1` and `1.0` are distinct type-strict members even under a shared
    `SymbolType.INT` domain.
    """
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    ones = create_integer_param(name=x1, constraints=[InSetConstraint(x1, {1})])
    float_ones = create_integer_param(name=x2, constraints=[InSetConstraint(x2, {1.0})])

    assert not ones.is_subset(float_ones)


def test_integer_singleton_one_is_not_subset_of_integer_singleton_true() -> None:
    """Test integer `{1}` is not a subset of integer `{True}`."""
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    ones = create_integer_param(name=x1, constraints=[InSetConstraint(x1, {1})])
    bool_ones = create_integer_param(name=x2, constraints=[InSetConstraint(x2, {True})])

    assert not ones.is_subset(bool_ones)


@pytest.mark.parametrize(
    ("smaller_members", "larger_members", "expect_subset"),
    [
        pytest.param({1, 2}, {1, 2, 3}, True, id="strict-subset-holds"),
        pytest.param({1, 2, 3}, {1, 2}, False, id="strict-superset-does-not-hold"),
    ],
)
def test_integer_in_set_subset_sanity_checks(
    smaller_members: set[int], larger_members: set[int], expect_subset: bool
) -> None:
    """Test ordinary integer in-set subset relations still hold as expected."""
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    smaller = create_integer_param(
        name=x1, constraints=[InSetConstraint(x1, smaller_members)]
    )
    larger = create_integer_param(
        name=x2, constraints=[InSetConstraint(x2, larger_members)]
    )

    assert smaller.is_subset(larger) is expect_subset


# =============================================================================
# Screened equation-constraint feasibility
# =============================================================================


def test_integer_param_with_consistent_bounds_is_feasible() -> None:
    """Test `x >= 0 and x <= 10` is feasible."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) >= 0),
            EquationConstraint(IdentifierExpression(x) <= 10),
        ],
    )

    assert param.is_feasible()
    assert not param.is_empty()


def test_integer_param_with_contradictory_equation_constraints_is_infeasible() -> None:
    """Test `x < 0 and x > 0` is infeasible."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < 0),
            EquationConstraint(IdentifierExpression(x) > 0),
        ],
    )

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_param_with_division_by_variable_stays_optimistically_feasible() -> (
    None
):
    """Test a division-by-the-same-variable equation degrades to the optimistic default.

    `x / x != 1` triggers the division hazard screen (the divisor is not a
    nonzero literal); the screen reports `UNDECIDED` rather than a wrong
    decided outcome, and `is_feasible` documents `UNDECIDED` degrading to
    the optimistic `True` (feasible-unless-disproven) instead of crashing
    or silently returning a wrong `False`.
    """
    x = mock_identifier("x", 1)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    param = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])

    assert param.is_feasible()


# =============================================================================
# Bound-constraint round trip through the new `EquationConstraint` shape
# =============================================================================


def test_bounded_integer_param_round_trip_stays_feasible() -> None:
    """Test `create_integer_param_between` stays feasible through the new shape.

    Anchors the bound-constraint round trip against the same healthy case
    the pre-rewrite suite pins in `test_feasibility.py`.
    """
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 1))

    assert param.is_feasible()
    assert not param.is_empty()
    param.validate_value(5)


def test_integer_param_with_contradictory_bounds_is_still_empty() -> None:
    """Test added lower/upper bound constraints can be jointly empty.

    Anchors the same case `test_feasibility.py` pins for the pre-rewrite
    constructor.
    """
    param = create_integer_param(name=mock_identifier("x", 1))
    narrowed = param.add_lower_bound_constraint(10).add_upper_bound_constraint(5)

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_narrower_bounded_param_is_subset_of_wider_bounded_param() -> None:
    """Test a narrower bound-constrained param is a subset of a wider one.

    Anchors the same relation `test_subset_relations.py` pins for the
    pre-rewrite constructor.
    """
    wider = create_integer_param_between(0, 10, name=mock_identifier("x", 1))
    narrower = create_integer_param_between(2, 8, name=mock_identifier("x", 1))

    assert narrower.is_subset(wider)
    assert not wider.is_subset(narrower)


# =============================================================================
# Dependent (foreign-identifier) constraints degrade instead of crashing
# =============================================================================


def test_is_feasible_with_foreign_identifier_constraint_does_not_raise() -> None:
    """Test `is_feasible` degrades to the optimistic default instead of raising.

    A constraint whose scope includes an identifier foreign to this
    parameter cannot be jointly decided by this parameter alone; it must
    be excluded from the decided conjunction rather than crash with a raw
    `KeyError`.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])

    assert param.is_feasible() is True


def test_is_empty_with_foreign_identifier_constraint_does_not_raise() -> None:
    """Test `is_empty` degrades to the optimistic default instead of raising."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])

    assert param.is_empty() is False


def test_is_subset_with_dependent_constraint_is_two_sided_and_safe() -> None:
    """Test `is_subset` stays boolean in both directions for a dependent constraint."""
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x1) < IdentifierExpression(y))
    with_dependent = create_integer_param(name=x1, constraints=[dependent])
    plain = create_integer_param(name=x2)

    forward = with_dependent.is_subset(plain)
    backward = plain.is_subset(with_dependent)

    assert forward is True
    assert backward is True
