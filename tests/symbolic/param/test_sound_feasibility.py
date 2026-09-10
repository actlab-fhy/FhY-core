"""Integration tests for sound `Param` feasibility, emptiness, and subset decisions.

Covers feasibility/subset decisions for type-strict set members that
Python's `==` conflates but this package's domains do not, screened
equation-constraint feasibility, foreign-identifier degradation for
dependent constraints, one-sided in-set subset screening, and partial
not-in-set member liftability. `Param.is_feasible`/`is_empty`/`is_subset`
stay boolean, and each reports `True` only for a proven answer: a question
the checker leaves undecided answers `False` rather than a provably wrong
decided answer.
"""

import math
from typing import cast

import pytest

from fhy_core.symbolic.constraint import (
    ConstraintMember,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    call,
    logical_not,
    piecewise,
)
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_integer_param,
    create_integer_param_between,
    create_real_param,
    create_real_param_between,
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

    assert ones.check_subset(float_ones) is ConstraintOutcome.VIOLATED


def test_integer_singleton_one_is_not_subset_of_integer_singleton_true() -> None:
    """Test integer `{1}` is not a subset of integer `{True}`."""
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    ones = create_integer_param(name=x1, constraints=[InSetConstraint(x1, {1})])
    bool_ones = create_integer_param(name=x2, constraints=[InSetConstraint(x2, {True})])

    assert ones.check_subset(bool_ones) is ConstraintOutcome.VIOLATED


@pytest.mark.parametrize(
    ("smaller_members", "larger_members", "expected"),
    [
        pytest.param(
            {1, 2}, {1, 2, 3}, ConstraintOutcome.SATISFIED, id="strict-subset-holds"
        ),
        pytest.param(
            {1, 2, 3},
            {1, 2},
            ConstraintOutcome.VIOLATED,
            id="strict-superset-does-not-hold",
        ),
    ],
)
def test_integer_in_set_subset_sanity_checks(
    smaller_members: set[int],
    larger_members: set[int],
    expected: ConstraintOutcome,
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

    assert smaller.check_subset(larger) is expected


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

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_bounded_integer_param_is_not_subset_of_partly_covering_in_set_param() -> None:
    """Test a bounded integer param is not a subset of a partly-covering in-set param.

    `{1..10}` is not a subset of `{1, 2, 3}`: most of the interval falls
    outside the in-set constraint's three members.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param_between(1, 10, name=x)
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {1, 2, 3})])

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


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

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_lower_bounded_integer_param_is_not_subset_of_in_set_param() -> None:
    """Test a lower-bounded integer param is not a subset of a small in-set param."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param(
        name=x, constraints=[EquationConstraint(IdentifierExpression(x) >= 0)]
    )
    other = create_integer_param(name=y, constraints=[InSetConstraint(y, {1, 2})])

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


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


def test_integer_param_with_division_by_variable_is_neither_feasible_nor_empty() -> (
    None
):
    """Test a division-by-the-same-variable equation degrades to an unproven answer.

    `x / x != 1` triggers the division hazard screen (the divisor is not a
    nonzero literal); the screen reports `UNDECIDED` rather than a wrong
    decided outcome, so neither wrapper has a proof to report: the
    parameter is reported neither feasible nor empty, and nothing crashes.
    """
    x = mock_identifier("x", 1)
    hazardous = (IdentifierExpression(x) / IdentifierExpression(x)).not_equals(1)
    param = create_integer_param(name=x, constraints=[EquationConstraint(hazardous)])

    assert param.is_feasible() is False
    assert param.is_empty() is False


def test_real_param_with_in_set_and_a_numeric_rooted_equation_raises() -> None:
    """Test enumeration over in-set candidates does not silently decide a numeric root.

    ``x / 2.0`` is a numeric root, not a predicate: reducing it to
    ``0.5`` or ``1.0`` for each enumerated candidate is not a decided
    VIOLATED, so the enumeration this in-set membership makes possible
    must not paper over the ill-typedness with a proof of emptiness.
    """
    x = mock_identifier("x", 1)
    equation = EquationConstraint(
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), LiteralExpression(2.0)
        )
    )
    param = create_real_param(
        name=x, constraints=[InSetConstraint(x, {1.0, 2.0}), equation]
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        param.is_empty()
    with pytest.raises(NonBooleanLogicalOperandError):
        param.check_feasibility()
    with pytest.raises(NonBooleanLogicalOperandError):
        param.is_value_valid(1.0)


def test_real_param_with_a_numeric_rooted_equation_raises_on_the_solver_path() -> None:
    """Test the same numeric-rooted equation raises without an in-set candidate set.

    With no `InSetConstraint` to make the domain finite, feasibility goes
    to the solver instead of enumeration; the refusal has to hold on
    that path too.
    """
    x = mock_identifier("x", 1)
    equation = EquationConstraint(
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), LiteralExpression(2.0)
        )
    )
    param = create_real_param(name=x, constraints=[equation])

    with pytest.raises(NonBooleanLogicalOperandError):
        param.is_empty()


def test_integer_param_with_in_set_and_a_numeric_result_call_under_not_raises() -> None:
    """Test enumeration does not decide a connective over a numeric-result call.

    ``floor`` is registered with an INT result sort, so
    ``logical_not(floor(x))`` is ill-typed for every enumerated candidate.
    """
    x = mock_identifier("x", 1)
    equation = EquationConstraint(logical_not(call("floor", IdentifierExpression(x))))
    param = create_integer_param(
        name=x, constraints=[InSetConstraint(x, {1, 2}), equation]
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        param.check_feasibility()


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


def test_integer_param_with_unliftable_not_in_set_member_admits_other_values() -> None:
    """Test a not-in-set constraint with one unliftable member still allows others.

    Narrowing `NotInSetConstraint(x, {5, "a"})` to its liftable member
    `{5}` only widens the admissible set, so the parameter is never
    reported empty and any integer other than `5` stays valid. The
    narrowed system is inexact, so the solver's answer does not prove
    feasibility and `is_feasible` reports `False`.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(name=x, constraints=[NotInSetConstraint(x, {5, "a"})])

    assert not param.is_empty()
    assert param.is_value_valid(6)
    assert not param.is_value_valid(5)
    assert not param.is_feasible()


def test_integer_param_with_only_unliftable_not_in_set_members_is_not_empty() -> None:
    """Test a not-in-set constraint with no liftable members is dropped, not fatal.

    Dropping the constraint leaves an inexact system, so feasibility is
    unproven and `is_feasible` reports `False`; nothing proves emptiness,
    and every integer stays valid, since no string can equal one.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x, constraints=[NotInSetConstraint(x, {"a", "b"})]
    )

    assert not param.is_empty()
    assert param.is_value_valid(5)
    assert not param.is_feasible()


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
    assert wider.check_subset(narrower) is ConstraintOutcome.VIOLATED


# =============================================================================
# Dependent (foreign-identifier) constraints degrade instead of crashing
# =============================================================================


def test_is_feasible_with_foreign_identifier_constraint_does_not_raise() -> None:
    """Test `is_feasible` degrades to an unproven `False` instead of raising.

    A constraint whose scope includes an identifier foreign to this
    parameter cannot be jointly decided by this parameter alone; it must
    be excluded from the decided conjunction rather than crash with a raw
    `KeyError`, and the weakened conjunction proves no satisfying value.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])

    assert param.is_feasible() is False


def test_is_empty_with_foreign_identifier_constraint_does_not_raise() -> None:
    """Test `is_empty` degrades to an unproven `False` instead of raising."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])

    assert param.is_empty() is False


def test_is_subset_with_dependent_constraint_is_two_sided_and_safe() -> None:
    """Test `is_subset` stays boolean in both directions for a dependent constraint.

    Dropping the dependent constraint only widens its side. Forward, that
    side is the antecedent, and even widened it lies inside the
    unconstrained consequent, which proves the relation. Backward, it is
    the consequent, and inclusion in a widened consequent proves nothing,
    so the relation is unproven and reported `False`.
    """
    x1 = mock_identifier("x", 1)
    x2 = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x1) < IdentifierExpression(y))
    with_dependent = create_integer_param(name=x1, constraints=[dependent])
    plain = create_integer_param(name=x2)

    forward = with_dependent.is_subset(plain)
    backward = plain.is_subset(with_dependent)

    assert forward is True
    assert backward is False


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
    integer, so the own side has a counterexample and the relation is
    decided `VIOLATED`. Posed to the implication branch instead, the
    unliftable member set would be dropped whole from the consequent --
    weakening it to `True` -- and the relation left undecided.
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
    assert other.is_empty() is True
    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_unbounded_param_is_not_subset_of_a_narrower_in_set_param() -> None:
    """Test an infinite parameter admitting an outside value is not a subset.

    Generalizes the empty-superset case: the own side admits `3`, which
    lies outside the other side's `{0, 1, 2}`, so the relation is decided
    `VIOLATED` from a genuine counterexample rather than left undecided.
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

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_unbounded_param_stays_subset_when_no_counterexample_is_provable() -> None:
    """Test a relation with no counterexample is still proven to hold.

    The own side is pinned to a single value inside the other side's
    finite set, so no admitted value lies outside it: the counterexample
    search finds none, and the implication then proves the relation.
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
def test_param_named_after_a_native_constant_binds_like_any_other(
    constant_name: str,
) -> None:
    """Test a parameter named after a native constant takes its candidate binding.

    The expression bridge resolves a native constant by the canonical
    identifier the registry minted for it, so a parameter that merely
    shares a constant's `name_hint` is an ordinary variable: the
    candidate binding reaches the constraint, and both the enumeration
    path and the solver path decide from it rather than from the
    constant's value.
    """
    named_like_constant = mock_identifier(constant_name, 1)
    equation = EquationConstraint(
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(named_like_constant),
            LiteralExpression(3),
        )
    )
    satisfied = create_integer_param(
        name=named_like_constant,
        constraints=[InSetConstraint(named_like_constant, {3}), equation],
    )
    violated = create_integer_param(
        name=named_like_constant,
        constraints=[InSetConstraint(named_like_constant, {4}), equation],
    )
    solver_decided = create_integer_param(
        name=named_like_constant, constraints=[equation]
    )

    assert satisfied.is_feasible() is True
    assert satisfied.is_empty() is False
    assert violated.is_feasible() is False
    assert violated.is_empty() is True
    assert solver_decided.is_feasible() is True


def test_bridge_failure_degrades_instead_of_escaping_a_boolean_api() -> None:
    """Test an expression the bridge cannot lift degrades rather than raising.

    The branch guarded by the unbound `y` divides by zero at both in-set
    candidates, so the SymPy bridge refuses to lift the complex infinity
    it folds to, raising `PassExecutionError` from deep inside
    evaluation. Every parameter-level entry point here returns `bool` or
    raises `ParamError`, so the backend failing to answer must degrade to
    an unproven answer instead of escaping as an unrelated exception type.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    denominator = BinaryExpression(
        BinaryOperation.MULTIPLY,
        BinaryExpression(
            BinaryOperation.SUBTRACT, IdentifierExpression(x), LiteralExpression(2)
        ),
        BinaryExpression(
            BinaryOperation.SUBTRACT, IdentifierExpression(x), LiteralExpression(4)
        ),
    )
    guarded = piecewise(
        (
            BinaryExpression(
                BinaryOperation.GREATER, IdentifierExpression(y), LiteralExpression(0)
            ),
            BinaryExpression(BinaryOperation.DIVIDE, LiteralExpression(1), denominator),
        ),
        otherwise=LiteralExpression(1),
    )
    unliftable = EquationConstraint(
        BinaryExpression(BinaryOperation.GREATER, guarded, LiteralExpression(0))
    )
    param = create_integer_param(
        name=x, constraints=[InSetConstraint(x, {2, 4}), unliftable]
    )

    assert param.is_feasible() is False
    assert param.is_empty() is False
    assert param.is_value_valid(2) is False
    with pytest.raises(ParamError, match="could not be verified"):
        param.validate_value(2)


def test_is_value_valid_degrades_instead_of_raising_for_a_nan_dependent_binding() -> (
    None
):
    """Test a NaN dependent binding reports `False` rather than raising.

    Substituting NaN for `y` under a strict comparison makes the SymPy
    bridge raise `PassExecutionError` from deep inside substitution;
    `evaluate_system_outcome` degrades that to `UNDECIDED`, and
    `is_value_valid` reports `False` for an undecided answer rather than
    propagating the exception.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    param = create_real_param(
        name=x,
        constraints=[
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(x),
                    IdentifierExpression(y),
                )
            )
        ],
    )

    assert param.is_value_valid(1.0, bindings={y: math.nan}) is False


def test_is_value_valid_degrades_instead_of_raising_for_a_zero_divisor_binding() -> (
    None
):
    """Test a zero-divisor dependent binding reports `False` rather than raising."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    param = create_real_param(
        name=x,
        constraints=[
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER,
                    BinaryExpression(
                        BinaryOperation.DIVIDE,
                        IdentifierExpression(x),
                        IdentifierExpression(y),
                    ),
                    LiteralExpression(1),
                )
            )
        ],
    )

    assert param.is_value_valid(1.0, bindings={y: 0.0}) is False


# =============================================================================
# Enumeration combines every constraint kind, not just a lone in-set constraint
# =============================================================================


def test_two_in_set_constraints_intersect_under_type_strict_equality() -> None:
    """Test a second in-set constraint narrows the candidate set.

    Enumeration seeds from the first in-set constraint's members and
    intersects every subsequent one. Without the intersection step the
    seed set survives whole and a value only one constraint permits is
    reported feasible.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[InSetConstraint(x, {1, 2, 3}), InSetConstraint(x, {3, 4, 5})],
    )

    assert param.is_feasible() is True
    assert param.is_value_valid(3) is True
    assert param.is_value_valid(1) is False
    assert param.is_value_valid(4) is False


def test_not_in_set_constraint_can_empty_an_in_set_constrained_param() -> None:
    """Test a disjoint not-in-set constraint subtracts the whole candidate set.

    Enumeration removes every not-in-set member from the candidates. With
    the subtraction dropped, a parameter that provably admits nothing
    reports itself feasible.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            InSetConstraint(x, {1, 2, 3}),
            NotInSetConstraint(x, {1, 2, 3}),
        ],
    )

    assert param.is_feasible() is False
    assert param.is_empty() is True


def test_equation_constraint_excludes_in_set_candidates_it_violates() -> None:
    """Test an equation constraint removes candidates it provably violates.

    `{1, 2}` with `x > 5` admits nothing. Without the per-candidate
    equation filter the in-set members survive unexamined and the
    parameter reports itself feasible.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param(
        name=x,
        constraints=[
            InSetConstraint(x, {1, 2}),
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER,
                    IdentifierExpression(x),
                    LiteralExpression(5),
                )
            ),
        ],
    )

    assert param.is_feasible() is False
    assert param.is_empty() is True


# =============================================================================
# The other side of a subset query rejects on its domain and its not-in-set set
# =============================================================================


def test_in_set_param_is_not_subset_when_other_side_forbids_a_candidate() -> None:
    """Test the other side's not-in-set constraint rejects a candidate.

    Every candidate the own side admits must be accepted by the other
    side, which checks its own not-in-set members. Without that arm,
    `{1, 2, 3}` is reported a subset of a parameter that explicitly
    forbids exactly those values.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_integer_param(name=x, constraints=[InSetConstraint(x, {1, 2, 3})])
    other = create_integer_param(name=y, constraints=[NotInSetConstraint(y, {1, 2, 3})])

    assert own.check_subset(other) is ConstraintOutcome.VIOLATED


def test_real_in_set_param_is_not_subset_of_an_integer_param() -> None:
    """Test the other side's domain admissibility rejects a candidate.

    A real parameter restricted to `{1.5, 2.5}` cannot be a subset of any
    integer parameter, since neither member is an admissible integer.
    Without the domain-admissibility arm the candidates pass unexamined.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    own = create_real_param(name=x, constraints=[InSetConstraint(x, {1.5, 2.5})])
    other = create_integer_param(name=y)

    # `Param[_T]` is an advisory call-site hint; the domain enforces the
    # admissible type at runtime, and a cross-domain query is exactly what
    # this asserts returns False.
    outcome = own.check_subset(cast("Param[str | float]", other))

    assert outcome is ConstraintOutcome.VIOLATED


# =============================================================================
# Real domain: a not-in-set constraint's numeric member does not decide alone
# =============================================================================


def test_real_param_singleton_excluding_its_own_float_value_is_undecided() -> None:
    """Test excluding a real singleton's own float value degrades to UNDECIDED.

    Z3 lowers the excluded `float` member and the singleton's bounds to
    the same rational, so the solver reports the parameter emptied.
    Type-strict membership excludes only the `float` kind of `0.5` and
    still admits the decimal-string kind that denotes the same value, so
    the proof does not actually hold and must be reported UNDECIDED
    rather than VIOLATED.
    """
    x = mock_identifier("x", 1)
    param = create_real_param_between(0.5, 0.5, name=x).add_constraint(
        NotInSetConstraint(x, {0.5})
    )

    assert param.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert param.is_empty() is False


def test_real_param_singleton_excluding_its_own_float_still_admits_string_kind() -> (
    None
):
    """Test the excluded real singleton still admits its decimal-string kind.

    Companion witness to the feasibility test above: the not-in-set
    member is the `float` `0.5`, so the type-strict decimal-string
    `"0.5"` denoting the same real value is a distinct member and stays a
    valid value for the parameter.
    """
    x = mock_identifier("x", 1)
    param = create_real_param_between(0.5, 0.5, name=x).add_constraint(
        NotInSetConstraint(x, {0.5})
    )

    assert param.is_value_valid("0.5") is True


def test_integer_param_singleton_excluding_its_own_value_stays_violated() -> None:
    """Test an integer singleton excluding its own value stays decided VIOLATED.

    The integer domain admits only `int`, so there is no second kind of
    `5` for the exclusion to miss: the solver's proof is sound and must
    not be downgraded the way the real-domain case above is.
    """
    x = mock_identifier("x", 1)
    param = create_integer_param_between(5, 5, name=x).add_constraint(
        NotInSetConstraint(x, {5})
    )

    assert param.check_feasibility() is ConstraintOutcome.VIOLATED
    assert param.is_empty() is True


def test_real_param_violated_without_set_constraints_stays_violated() -> None:
    """Test a real VIOLATED that rests on no set member stays decided."""
    x = mock_identifier("x", 1)
    param = create_real_param_between(0.0, 1.0, name=x).add_constraint(
        EquationConstraint(
            BinaryExpression(
                BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(2.0)
            )
        )
    )

    assert param.check_feasibility() is ConstraintOutcome.VIOLATED
    assert param.is_empty() is True


def test_real_param_in_set_float_member_is_feasible() -> None:
    """Test a real param restricted to a float in-set member is feasible.

    Enumeration over the in-set candidate compares type-strictly and is
    unaffected by the solver's kind conflation, so this stays a decided
    SATISFIED.
    """
    x = mock_identifier("x", 1)
    param = create_real_param(name=x, constraints=[InSetConstraint(x, {0.5})])

    assert param.check_feasibility() is ConstraintOutcome.SATISFIED
