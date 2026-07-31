"""Tests for `create_intersection_param` and `Param.__and__`.

Intersection is supported for every domain kind, with a uniform emptiness
rule: `create_intersection_param` raises `ParamError` whenever the result is
provably empty. Finite-set kinds detect emptiness by baking an empty member
set; permutation and numeric kinds are checked by the factory's
`is_feasible()` call after construction.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_integer_param_with_lower_bound,
    create_integer_param_with_upper_bound,
    create_intersection_param,
    create_interval_integer_param,
    create_interval_integer_param_between,
    create_interval_integer_param_with_lower_bound,
    create_interval_integer_param_with_upper_bound,
    create_interval_natural_param,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
    create_real_param_with_lower_bound,
    create_union_param,
)
from fhy_core.symbolic.param import domains as param_domains
from fhy_core.symbolic.param.domains import (
    CategoricalDomain,
    IntegerDomain,
    OrdinalDomain,
)
from fhy_core.traits.derived_equivalence import compared_as_reference
from fhy_core.utils.override import override

from .conftest import (
    assert_all_satisfied,
    assert_all_valid,
    assert_none_satisfied,
    assert_none_valid,
    assert_param_round_trips_in_all_formats,
    mock_identifier,
)

# =============================================================================
# Finite-set intersection (baked)
# =============================================================================


def test_categorical_intersection_bakes_common_members() -> None:
    """Test categorical intersection keeps only members common to both operands."""
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})

    result = create_intersection_param(left, right)

    assert isinstance(result.domain, CategoricalDomain)
    assert_all_valid(result, ["b", "c"])
    assert_none_valid(result, ["a", "d"])


def test_categorical_intersection_of_disjoint_sets_raises_param_error() -> None:
    """Test categorical intersection of disjoint sets raises ``ParamError``."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


def test_categorical_intersection_of_strictly_distinct_kinds_is_empty() -> None:
    """Test ``{1} & {True}`` is empty: ``1`` and ``True`` are distinct kinds."""
    left: Param[Any] = create_categorical_param({1})
    right = create_categorical_param({True})

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


def test_ordinal_intersection_bakes_common_values() -> None:
    """Test ordinal intersection keeps only values common to both operands."""
    left = create_ordinal_param([1, 2, 3])
    right = create_ordinal_param([2, 3, 4])

    result = create_intersection_param(left, right)

    assert isinstance(result.domain, OrdinalDomain)
    assert_all_valid(result, [2, 3])
    assert_none_valid(result, [1, 4])


def test_ordinal_intersection_of_disjoint_sets_raises_param_error() -> None:
    """Test ordinal intersection of disjoint sets raises ``ParamError``."""
    left = create_ordinal_param([1, 2])
    right = create_ordinal_param([3, 4])

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


def test_finite_set_intersection_result_carries_no_constraints() -> None:
    """Test the baked finite-set intersection result carries no constraints."""
    left = create_categorical_param({"a", "b", "c"})
    left = left.add_constraint(InSetConstraint(left.variable, {"a", "b"}))
    right = create_categorical_param({"b", "c"})

    result = create_intersection_param(left, right)

    assert result.constraints == ()


# =============================================================================
# Permutation intersection
# =============================================================================


def test_permutation_intersection_of_equal_member_sets_conjoins_constraints() -> None:
    """Test permutation intersection with equal member sets conjoins constraints.

    ``left`` is unconstrained (both orderings of ``{"a","b"}`` are valid);
    ``right`` narrows to a single ordering. The intersection admits only the
    ordering both agree on.
    """
    left = create_permutation_param(["a", "b"])
    right = create_permutation_param(["a", "b"])
    right = right.add_constraint(InSetConstraint(right.variable, {("a", "b")}))

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [("a", "b")])
    assert_none_satisfied(result, [("b", "a")])


def test_permutation_intersection_keeps_left_operand_ordered_members() -> None:
    """Test the permutation intersection keeps the left operand's member order."""
    left = create_permutation_param(["x", "y"])
    right = create_permutation_param(["y", "x"])

    result = create_intersection_param(left, right)

    assert result.domain.ordered_members == ("x", "y")  # type: ignore[attr-defined]


def test_permutation_intersection_with_differing_member_sets_raises_param_error() -> (
    None
):
    """Test permutation intersection with unequal member sets raises ``ParamError``."""
    left = create_permutation_param(["a", "b"])
    right = create_permutation_param(["a", "c"])

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


def test_permutation_intersection_with_differing_member_counts_raises_param_error() -> (
    None
):
    """Test permutation intersection with differently-sized member sets raises."""
    left = create_permutation_param(["a", "b"])
    right = create_permutation_param(["a", "b", "c"])

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


def test_permutation_intersection_of_mutually_exclusive_constraints_is_empty() -> None:
    """Test permutation intersection whose combined constraints admit nothing.

    Equal member sets (so the member-set gate passes), but ``left`` and
    ``right`` each pin down a different, non-overlapping single ordering, so
    the conjunction is empty.
    """
    left = create_permutation_param(["a", "b"])
    left = left.add_constraint(InSetConstraint(left.variable, {("a", "b")}))
    right = create_permutation_param(["a", "b"])
    right = right.add_constraint(InSetConstraint(right.variable, {("b", "a")}))

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


# =============================================================================
# Interval-integer intersection (bound tightening)
# =============================================================================


@pytest.mark.z3
def test_interval_intersection_tightens_bounds_via_and_operator() -> None:
    """Test ``[0,10] & [5,20]`` tightens to ``[5,10]``."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = left & right

    assert_all_satisfied(result, [5, 10])
    assert_none_satisfied(result, [4, 11])


@pytest.mark.z3
def test_interval_intersection_tightens_bounds_via_factory() -> None:
    """Test ``create_intersection_param`` tightens bounds the same as ``&``."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [5, 10])
    assert_none_satisfied(result, [4, 11])


@pytest.mark.z3
def test_interval_intersection_of_disjoint_intervals_raises_param_error() -> None:
    """Test interval intersection of disjoint intervals raises ``ParamError``."""
    left = create_interval_integer_param_between(0, 5)
    right = create_interval_integer_param_between(10, 15)

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


@pytest.mark.z3
def test_interval_intersection_merges_non_negative_attribute() -> None:
    """Test the interval intersection's ``non_negative`` is the OR of operands."""
    left = create_interval_integer_param()
    left = left.add_lower_bound_constraint(0).add_upper_bound_constraint(10)
    right = create_interval_natural_param()
    right = right.add_upper_bound_constraint(10)

    result = create_intersection_param(left, right)

    assert result.domain.non_negative  # type: ignore[attr-defined]


# =============================================================================
# Mixed interval-integer / plain-integer intersection (coercion)
# =============================================================================


@pytest.mark.z3
def test_intersection_coerces_plain_integer_operand_on_right() -> None:
    """Test intersection coerces a plain integer param (right) into interval form."""
    left = create_interval_integer_param_between(0, 10)
    right = create_integer_param_between(5, 20)

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [5, 10])
    assert_none_satisfied(result, [4, 11])


@pytest.mark.z3
def test_intersection_coerces_plain_integer_operand_on_left() -> None:
    """Test intersection coerces a plain integer param (left) into interval form."""
    left = create_integer_param_between(5, 20)
    right = create_interval_integer_param_between(0, 10)

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [5, 10])
    assert_none_satisfied(result, [4, 11])


# =============================================================================
# Integer / real intersection (attribute merge, constraint conjunction)
# =============================================================================


@pytest.mark.z3
def test_integer_intersection_conjoins_bound_constraints() -> None:
    """Test plain-integer intersection conjoins both operands' bound constraints."""
    left = create_integer_param_with_lower_bound(0)
    right = create_integer_param_with_upper_bound(10)

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [0, 5, 10])
    assert_none_satisfied(result, [-1, 11])


@pytest.mark.z3
def test_integer_intersection_merges_non_negative_attribute() -> None:
    """Test integer intersection merges ``non_negative``/``zero_included`` attributes.

    ``left`` is a natural (non-negative, zero-included) param upper-bounded at
    10; ``right`` is a plain integer param lower-bounded at -5. The merged
    domain is non-negative (OR rule), so the result is exactly ``[0, 10]``.
    """
    left = create_natural_param()
    left = left.add_upper_bound_constraint(10)
    right = create_integer_param_with_lower_bound(-5)

    result = create_intersection_param(left, right)

    assert isinstance(result.domain, IntegerDomain)
    assert result.domain.non_negative
    assert_all_satisfied(result, [0, 10])
    assert_none_satisfied(result, [-1, 11])


@pytest.mark.z3
def test_real_intersection_conjoins_bound_constraints() -> None:
    """Test real-param intersection conjoins both operands' bound constraints."""
    left = create_real_param_with_lower_bound(0.0)
    right = create_real_param()
    right = right.add_upper_bound_constraint(10.0)

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [0.0, 5.0, 10.0])
    assert_none_satisfied(result, [-1.0, 11.0])


# =============================================================================
# Constraint rebinding: alpha-equivalence against a hand-built parameter
# =============================================================================


@pytest.mark.z3
def test_intersection_constraints_rebind_to_result_variable() -> None:
    """Test rebound intersection constraints are alpha-equivalent to a hand-built param.

    Each operand here carries exactly one bound constraint (a lower bound on
    the left, an upper bound on the right), so the intersection's rebound
    conjunction is exactly the same two constraints a directly-built
    ``[5,10]`` interval-integer param would carry -- alpha-equivalent up to
    renaming of the (fresh, distinct) result variable.
    """
    left = create_interval_integer_param_with_lower_bound(5)
    right = create_interval_integer_param_with_upper_bound(10)
    hand_built = create_interval_integer_param_between(
        5, 10, name=mock_identifier("expected", 99)
    )

    result = create_intersection_param(left, right)

    assert result.is_alpha_equivalent(hand_built)
    assert not result.is_structurally_equivalent(hand_built)


def test_intersection_categorical_constraints_rebind_to_result_variable() -> None:
    """Test a baked categorical intersection is alpha-equivalent to a hand-built one."""
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})
    hand_built = create_categorical_param(
        {"b", "c"}, name=mock_identifier("expected", 99)
    )

    result = create_intersection_param(left, right)

    assert result.is_alpha_equivalent(hand_built)


# =============================================================================
# Emptiness detected via Z3 (marked, solver required)
# =============================================================================


@pytest.mark.z3
def test_integer_intersection_z3_proven_empty_raises_param_error() -> None:
    """Test a Z3-decidable infeasible integer conjunction raises ``ParamError``."""
    left = create_integer_param_with_lower_bound(6)
    right = create_integer_param_with_upper_bound(3)

    with pytest.raises(ParamError):
        create_intersection_param(left, right)


@pytest.mark.z3
def test_intersection_accepts_result_when_z3_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test intersection is accepted (not raised) when Z3 cannot decide feasibility.

    Monkeypatches ``check_expression_satisfiability`` -- the seam
    ``_numeric_has_feasible_value`` calls -- to always return ``None`` (Z3's
    "unknown" outcome), so ``is_feasible()``'s optimistic-on-unknown
    convention is exercised: the factory must not treat an undecidable
    conjunction as proven empty. Asserting the tightened bounds (rather than
    just ``result is not None``, which a ``Param``-or-raise factory can never
    fail) confirms the returned param is the real, live intersection and not
    some degenerate stand-in.
    """
    left = create_integer_param_with_lower_bound(0)
    right = create_integer_param_with_upper_bound(10)
    monkeypatch.setattr(
        param_domains, "check_expression_satisfiability", lambda *args, **kwargs: None
    )

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [0, 5, 10])
    assert_none_satisfied(result, [-1, 11])


# =============================================================================
# Unknown constraint kind
# =============================================================================


@dataclass(frozen=True, eq=False)
class _UnrebindableConstraint(Constraint):
    """A `Constraint` subclass with no rebinding rule, for negative testing."""

    variable: Identifier = field(metadata=compared_as_reference())

    @override
    def evaluate(self, value: object) -> ConstraintOutcome:
        del value
        return ConstraintOutcome.SATISFIED

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def __repr__(self) -> str:
        return "UnrebindableConstraint"

    @override
    def __str__(self) -> str:
        return "UnrebindableConstraint"


def test_intersection_with_unrebindable_constraint_kind_raises_type_error() -> None:
    """Test intersection raises ``TypeError`` when a carried constraint can't rebind."""
    left = create_integer_param()
    left = left.add_constraint(_UnrebindableConstraint(left.variable))
    right = create_integer_param()

    with pytest.raises(TypeError):
        create_intersection_param(left, right)


# =============================================================================
# Dependent (multi-variable) constraints are rejected at construction
# =============================================================================


def test_dependent_constraint_operand_is_rejected_at_construction() -> None:
    """Test a multi-variable-constrained operand is rejected before intersection runs.

    A constraint referencing a second free identifier used to slip past
    `Param.validate_constraint` on the left operand and only blow up later,
    with a raw `KeyError` from the Z3 bridge, from `create_intersection_param`'s
    internal `result.is_feasible()` call. `validate_constraint` must reject it
    up front (while constructing the offending operand) with a typed
    `ParamError` instead.
    """
    x = mock_identifier("x", 200)
    y = mock_identifier("y", 201)
    dependent_constraint = EquationConstraint(
        x, IdentifierExpression(x) < IdentifierExpression(y)
    )

    with pytest.raises(ParamError, match="ConstraintSystem"):
        create_integer_param(name=x, constraints=[dependent_constraint])


# =============================================================================
# Result variable
# =============================================================================


def test_intersection_result_uses_fresh_default_variable() -> None:
    """Test the intersection uses a fresh ``Identifier`` when ``name`` is omitted."""
    left = create_categorical_param({"a", "b"}, name=mock_identifier("x", 1))
    right = create_categorical_param({"b"}, name=mock_identifier("y", 2))

    result = create_intersection_param(left, right)

    assert result.variable is not left.variable
    assert result.variable is not right.variable
    assert isinstance(result.variable, Identifier)


def test_intersection_result_uses_given_name() -> None:
    """Test the intersection result uses the ``name`` argument when supplied."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"b"})
    given = mock_identifier("z", 3)

    result = create_intersection_param(left, right, name=given)

    assert result.variable is given


# =============================================================================
# `__and__` dunder delegation
# =============================================================================


def test_and_dunder_delegates_to_create_intersection_param() -> None:
    """Test ``&`` produces the same intersection as calling the factory directly."""
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})

    result = left & right

    assert_all_valid(result, ["b", "c"])
    assert_none_valid(result, ["a", "d"])


def test_and_dunder_with_non_param_operand_raises_type_error() -> None:
    """Test ``&`` raises ``TypeError`` (via ``NotImplemented``) for a non-`Param`."""
    left = create_categorical_param({"a"})

    with pytest.raises(TypeError):
        _ = left & "not a param"


# =============================================================================
# Cross-kind mismatch
# =============================================================================


def test_intersection_across_categorical_and_ordinal_raises_type_error() -> None:
    """Test intersection across categorical and ordinal domains raises ``TypeError``."""
    left: Param[Any] = create_categorical_param({"a", "b"})
    right = create_ordinal_param([1, 2])

    with pytest.raises(TypeError):
        create_intersection_param(left, right)


def test_intersection_of_permutation_and_categorical_raises_type_error() -> None:
    """Test intersection of permutation and categorical domains raises ``TypeError``."""
    left: Param[Any] = create_permutation_param(["a", "b"])
    right = create_categorical_param({"a", "b"})

    with pytest.raises(TypeError):
        create_intersection_param(left, right)


def test_intersection_of_integer_and_real_raises_type_error() -> None:
    """Test intersection across integer and real domains raises ``TypeError``."""
    left: Param[Any] = create_integer_param()
    right = create_real_param()

    with pytest.raises(TypeError):
        create_intersection_param(left, right)


# =============================================================================
# Set-membership law (intersection)
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("a", False),
        ("b", True),
        ("c", True),
        ("d", False),
    ],
)
def test_intersection_membership_law_holds_for_categorical_operands(
    value: str, expected: bool
) -> None:
    """Test a value is valid for the intersection iff valid for both operands.

    Left is ``{"a","b","c"}``, right is ``{"b","c","d"}``.
    """
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})

    result = create_intersection_param(left, right)

    assert result.is_value_valid(value) == expected


def test_intersection_type_annotation_is_param() -> None:
    """Test `create_intersection_param` returns a `Param` (public-surface check)."""
    left = create_categorical_param({"a"})
    right = create_categorical_param({"a"})

    result = create_intersection_param(left, right)

    assert isinstance(result, Param)


# =============================================================================
# Integration: serialization, subset/feasibility/assign interop, chaining
# =============================================================================


@pytest.mark.z3
def test_intersection_result_round_trips_through_serialization() -> None:
    """Test an intersection result round-trips through DICT, JSON, and BINARY."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = create_intersection_param(left, right)

    assert_param_round_trips_in_all_formats(result)


def test_categorical_intersection_result_round_trips_through_serialization() -> None:
    """Test a baked categorical intersection result round-trips in every format."""
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})

    result = create_intersection_param(left, right)

    assert_param_round_trips_in_all_formats(result)


@pytest.mark.z3
def test_intersection_result_interoperates_with_is_subset() -> None:
    """Test an intersection result's feasible set is a subset of each operand's."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = create_intersection_param(left, right)

    assert result.is_subset(left)
    assert result.is_subset(right)
    assert not left.is_subset(result)


@pytest.mark.z3
def test_intersection_result_interoperates_with_is_feasible() -> None:
    """Test a non-empty intersection result reports feasible."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = create_intersection_param(left, right)

    assert result.is_feasible()
    assert not result.is_empty()


@pytest.mark.z3
def test_intersection_result_interoperates_with_assign() -> None:
    """Test a value valid for both operands can be assigned to the result."""
    left = create_interval_integer_param_between(0, 10)
    right = create_interval_integer_param_between(5, 20)

    result = create_intersection_param(left, right)
    assignment = result.assign(7)

    assert assignment.value == 7
    with pytest.raises(ParamError):
        result.assign(11)


def test_chained_union_then_intersection() -> None:
    """Test ``(a | b) & c``: union two categorical operands, then intersect a third."""
    a = create_categorical_param({"a", "b"})
    b = create_categorical_param({"b", "c"})
    c = create_categorical_param({"b", "c", "d"})

    result = create_intersection_param(create_union_param(a, b), c)

    assert_all_valid(result, ["b", "c"])
    assert_none_valid(result, ["a", "d"])


def test_or_and_dunders_chain_the_same_as_the_factories() -> None:
    """Test ``(a | b) & c`` via dunders matches the equivalent factory chain."""
    a = create_categorical_param({"a", "b"})
    b = create_categorical_param({"b", "c"})
    c = create_categorical_param({"b", "c", "d"})

    result = (a | b) & c

    assert_all_valid(result, ["b", "c"])
    assert_none_valid(result, ["a", "d"])
