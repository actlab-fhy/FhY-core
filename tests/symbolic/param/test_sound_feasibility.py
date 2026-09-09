"""Integration tests for sound `Param` feasibility, emptiness, and subset decisions.

Covers feasibility/subset decisions for type-strict set members that
Python's `==` conflates but this package's domains do not, screened
equation-constraint feasibility, foreign-identifier degradation for
dependent constraints, one-sided in-set subset screening, and partial
not-in-set member liftability. `Param.is_feasible`/`is_empty`/`is_subset`
stay boolean; what changes is that the boolean answer is provably
correct, or an honestly documented optimistic default, rather than a
provably wrong decided answer.
"""

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintMember,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import (
    ParamError,
    create_integer_param,
    create_integer_param_between,
)

from .conftest import mock_identifier

# =============================================================================
# Type-strict set-constraint feasibility, decided by enumeration
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
# is_subset: an in-set constraint on only one side must still narrow that side
# =============================================================================


def test_bounded_integer_param_is_not_subset_of_disjoint_in_set_param() -> None:
    """Test a bounded integer param is not a subset of a disjoint in-set param.

    The in-set constraint lives on the `other` side only, so the
    comparison falls through to the screened-system branch (`own` has no
    in-set constraint of its own). That branch must still fold `other`'s
    in-set constraint into `other`'s screened system instead of silently
    dropping it.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param_between(1, 10, name=x)
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {100, 200})])

    assert not own.is_subset(other)


def test_bounded_integer_param_is_not_subset_of_partly_covering_in_set_param() -> None:
    """Test a bounded integer param is not a subset of a partly-covering in-set param.

    `{1..10}` is not a subset of `{1, 2, 3}`: most of the interval falls
    outside the in-set constraint's three members.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param_between(1, 10, name=x)
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {1, 2, 3})])

    assert not own.is_subset(other)


def test_bounded_integer_param_is_subset_of_in_set_param_covering_its_range() -> None:
    """Test a bounded integer param is a subset of a fully-covering in-set param."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param_between(1, 10, name=x)
    other = create_integer_param(
        name=y, constraints=[InSetConstraint(y, set(range(20)))]
    )

    assert own.is_subset(other)


def test_in_set_param_is_subset_of_bounded_integer_param_covering_it() -> None:
    """Test an in-set param is a subset of a bounded integer param covering it.

    Here the in-set constraint is on `own`, so this takes the enumeration
    branch rather than the screened-system branch, unlike the one-sided
    cases above.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param(name=x, constraints=[InSetConstraint(x, {1, 2, 3})])
    other = create_integer_param_between(1, 10, name=y)

    assert own.is_subset(other)


def test_unconstrained_integer_param_is_not_subset_of_singleton_in_set_param() -> None:
    """Test an unconstrained integer param is not a subset of a singleton in-set one."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param(name=x)
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {7})])

    assert not own.is_subset(other)


def test_lower_bounded_integer_param_is_not_subset_of_in_set_param() -> None:
    """Test a lower-bounded integer param is not a subset of a small in-set param."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param(
        name=x, constraints=[EquationConstraint(IdentifierExpression(x) >= 0)]
    )
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {1, 2})])

    assert not own.is_subset(other)


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
# A single unliftable not-in-set member must narrow, not erase, the constraint
# =============================================================================


def test_integer_param_with_conflicting_not_in_set_string_member_is_infeasible() -> (
    None
):
    """Test a not-in-set constraint with a string member still narrows correctly.

    `NotInSetConstraint(x, {5, "a"})` alone would leave every integer but
    `5` admissible; combined with `x == 5` the two are contradictory. The
    unliftable string member `"a"` must not cause the whole not-in-set
    constraint to be dropped and the contradiction missed.
    """
    x = mock_identifier("x", 1)
    equals_five = EquationConstraint(IdentifierExpression(x).equals(5))
    excludes_five_and_a = NotInSetConstraint(x, {5, "a"})
    param = create_integer_param(name=x, constraints=[equals_five, excludes_five_and_a])

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_param_with_conflicting_not_in_set_container_member_is_infeasible() -> (
    None
):
    """Test a not-in-set constraint with a container member still narrows correctly."""
    x = mock_identifier("x", 1)
    equals_five = EquationConstraint(IdentifierExpression(x).equals(5))
    excludes_five_and_pair = NotInSetConstraint(x, {5, (1, 2)})
    param = create_integer_param(
        name=x, constraints=[equals_five, excludes_five_and_pair]
    )

    assert not param.is_feasible()
    assert param.is_empty()


def test_integer_param_with_unliftable_not_in_set_member_stays_otherwise_feasible() -> (
    None
):
    """Test a not-in-set constraint with one unliftable member still allows others.

    Narrowing `NotInSetConstraint(x, {5, "a"})` to its liftable member
    `{5}` only widens the admissible set relative to the full two-member
    constraint, so with no other constraint present the parameter stays
    feasible through any integer other than `5`.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[NotInSetConstraint(x, {5, "a"})])

    assert param.is_feasible()
    assert not param.is_empty()


def test_integer_param_with_only_unliftable_not_in_set_members_stays_feasible() -> None:
    """Test a not-in-set constraint with no liftable members is dropped, not fatal."""
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x, constraints=[NotInSetConstraint(x, {"a", "b"})]
    )

    assert param.is_feasible()
    assert not param.is_empty()


# =============================================================================
# Bound-constraint round trip through `EquationConstraint`
# =============================================================================


def test_bounded_integer_param_round_trip_stays_feasible() -> None:
    """Test `create_integer_param_between` produces a feasible, non-empty param.

    Anchors the same healthy bound-constraint case covered in
    `test_feasibility.py`.
    """
    param = create_integer_param_between(0, 10, name=mock_identifier("x", 1))

    assert param.is_feasible()
    assert not param.is_empty()
    param.validate_value(5)


def test_integer_param_with_contradictory_bounds_is_still_empty() -> None:
    """Test added lower/upper bound constraints can be jointly empty.

    Anchors the same case covered in `test_feasibility.py`.
    """
    param = create_integer_param(name=mock_identifier("x", 1))
    narrowed = param.add_lower_bound_constraint(10).add_upper_bound_constraint(5)

    assert not narrowed.is_feasible()
    assert narrowed.is_empty()


def test_narrower_bounded_param_is_subset_of_wider_bounded_param() -> None:
    """Test a narrower bound-constrained param is a subset of a wider one.

    Anchors the same relation covered in `test_subset_relations.py`.
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


@pytest.mark.z3
@pytest.mark.parametrize(
    "members",
    [{1.0}, {True}, {"5"}],
    ids=["float_member", "bool_member", "string_member"],
)
def test_unbounded_param_is_not_subset_of_a_provably_empty_in_set_param(
    members: set[ConstraintMember],
) -> None:
    """Test an infinite parameter is not a subset of an empty set-constrained one.

    The other side's members are inadmissible for an integer domain, so
    it admits nothing, while the own side admits every non-negative
    integer. The enumeration gate previously keyed only on the own side
    carrying an `InSetConstraint`, so both sides went to the implication
    branch, where the unliftable member set is dropped whole from the
    consequent -- weakening it to `True` -- and the undecided result
    collapsed optimistically to a wrong `True`.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    own = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    IdentifierExpression(x),
                    LiteralExpression(0),
                )
            )
        ],
    )
    other = create_integer_param(
        name=y, constraints=[InSetConstraint(y, frozenset(members))]
    )

    assert other.is_feasible() is False
    assert own.is_subset(other) is False


@pytest.mark.z3
def test_unbounded_param_is_not_subset_of_a_narrower_in_set_param() -> None:
    """Test an infinite parameter admitting an outside value is not a subset.

    Generalizes the empty-superset case: the own side admits `3`, which
    lies outside the other side's `{0, 1, 2}`, so the relation is decided
    `False` from a genuine counterexample rather than left to the
    optimistic default.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    own = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    IdentifierExpression(x),
                    LiteralExpression(0),
                )
            )
        ],
    )
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {0, 1, 2})])

    assert own.is_subset(other) is False


@pytest.mark.z3
def test_unbounded_param_stays_subset_when_no_counterexample_is_provable() -> None:
    """Test the optimistic default survives when no counterexample can be proven.

    The own side is pinned to a single value inside the other side's
    finite set, so no admitted value lies outside it and the relation
    must not be decided `False`.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    own = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    IdentifierExpression(x),
                    LiteralExpression(1),
                )
            )
        ],
    )
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {0, 1, 2})])

    assert own.is_subset(other) is True


@pytest.mark.parametrize("constant_name", ["e", "pi"])
def test_param_named_after_a_native_constant_is_not_decided_infeasible(
    constant_name: str,
) -> None:
    """Test a parameter whose name collides with a native constant is not decided.

    The expression bridge resolves an identifier whose `name_hint` names a
    registered native constant to that constant rather than to a
    substitutable symbol, so the candidate binding is silently dropped and
    the constraint is evaluated against the constant's value instead.
    Reporting `False` there is a proof claim the backend never
    established -- and it flipped with the presence of an unrelated in-set
    constraint, since only the enumeration path routed through the bridge
    this way.
    """
    constant = mock_identifier(constant_name, 1)
    equation = EquationConstraint(
        BinaryExpression(
            BinaryOperation.EQUAL, IdentifierExpression(constant), LiteralExpression(3)
        )
    )
    enumerated = create_integer_param(
        name=constant, constraints=[InSetConstraint(constant, {3}), equation]
    )
    solver_decided = create_integer_param(name=constant, constraints=[equation])

    assert enumerated.is_feasible() is True
    assert enumerated.is_empty() is False
    assert solver_decided.is_feasible() == enumerated.is_feasible()


def test_bridge_failure_degrades_instead_of_escaping_a_boolean_api() -> None:
    """Test an expression the bridge cannot lower degrades rather than raising.

    `x / 3` over the integers leaves a rational the SymPy bridge refuses
    to lift, raising `PassExecutionError` from deep inside evaluation.
    Every parameter-level entry point here returns `bool` or raises
    `ParamError`, so the backend failing to answer must degrade to the
    documented optimistic default instead of escaping as an unrelated
    exception type.
    """
    x = mock_identifier("x", 1)
    unliftable = EquationConstraint(
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), LiteralExpression(3)
        )
    )
    param = create_integer_param(
        name=x, constraints=[InSetConstraint(x, {2, 4}), unliftable]
    )

    assert param.is_feasible() is True
    assert param.is_value_valid(2) is False
    with pytest.raises(ParamError, match="could not be verified"):
        param.validate_value(2)
