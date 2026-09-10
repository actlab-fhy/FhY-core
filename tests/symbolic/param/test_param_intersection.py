"""Tests for `create_intersection_param` and `Param.__and__`.

Intersection is supported for every domain kind, with a uniform emptiness
rule: `create_intersection_param` raises `ParamError` whenever the result is
provably empty. Finite-set kinds detect emptiness by baking an empty member
set; permutation and numeric kinds are checked by the factory after
construction: a `VIOLATED` conjunction raises, an `UNDECIDED` conjunction
raises only when an operand is itself proven infeasible, and otherwise the
undecided conjunction survives as a live parameter.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
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
    create_real_param_between,
    create_real_param_with_lower_bound,
    create_union_param,
)
from fhy_core.symbolic.param.domains import (
    CategoricalDomain,
    IntegerDomain,
    IntervalIntegerDomain,
    OrdinalDomain,
)
from fhy_core.term import compared_as_reference
from fhy_core.utils.override import override

from .conftest import (
    assert_all_satisfied,
    assert_all_valid,
    assert_none_satisfied,
    assert_none_valid,
    assert_param_round_trips_in_all_formats,
    build_case_condition_constraint,
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
    """Test a baked finite-set intersection folds operand constraints into its set.

    ``left`` declares ``{"a","b","c"}`` but its in-set constraint narrows it
    to ``{"a","b"}``; against ``{"b","c"}`` only ``"b"`` survives. The
    result carries that narrowing in its baked member set and no
    constraints at all.
    """
    left = create_categorical_param({"a", "b", "c"})
    left = left.add_constraint(InSetConstraint(left.variable, {"a", "b"}))
    right = create_categorical_param({"b", "c"})

    result = create_intersection_param(left, right)

    assert result.constraints == ()
    assert_all_valid(result, ["b"])
    assert_none_valid(result, ["a", "c"])


def test_ordinal_intersection_folds_not_in_set_and_in_set_constraints() -> None:
    """Test ordinal intersection bakes each side's set constraint into the result.

    ``left`` declares ``[1..5]`` minus ``{1, 5}``, so ``{2,3,4}``; ``right``
    declares ``[2..6]`` narrowed to ``{3,4,5,6}``. Only ``{3, 4}`` is
    admitted by both, and the result bakes exactly that.
    """
    left = create_ordinal_param([1, 2, 3, 4, 5])
    left = left.add_constraint(NotInSetConstraint(left.variable, {1, 5}))
    right = create_ordinal_param([2, 3, 4, 5, 6])
    right = right.add_constraint(InSetConstraint(right.variable, {3, 4, 5, 6}))

    result = create_intersection_param(left, right)

    assert isinstance(result.domain, OrdinalDomain)
    assert result.domain.sorted_values == (3, 4)
    assert result.constraints == ()
    assert_all_valid(result, [3, 4])
    assert_none_valid(result, [1, 2, 5, 6])


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
def test_interval_intersection_of_strict_subset_yields_the_subset_bounds() -> None:
    """Test ``[1,10] & [3,5]`` yields exactly ``[3,5]``: a strict-subset case."""
    left = create_interval_integer_param_between(1, 10)
    right = create_interval_integer_param_between(3, 5)

    result = left & right

    assert_all_satisfied(result, [3, 5])
    assert_none_satisfied(result, [2, 6])


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


@pytest.mark.z3
def test_interval_intersection_rendering_follows_left_operand_prefer_inclusive() -> (
    None
):
    """Test the intersection's ``prefer_inclusive`` follows the LEFT operand's.

    Each pair mixes operands with DIFFERING ``prefer_inclusive`` flags, so
    a result that took the flag from the right operand, or reconciled the
    two, is distinguishable from one that takes it from the left. The flag
    only decides how later arithmetic renders bounds; both results admit
    the same values.
    """
    x_incl = create_interval_integer_param_between(0, 10, prefer_inclusive=True)
    y_excl = create_interval_integer_param_between(5, 20, prefer_inclusive=False)
    x_excl = create_interval_integer_param_between(0, 10, prefer_inclusive=False)
    y_incl = create_interval_integer_param_between(5, 20, prefer_inclusive=True)

    z_left_incl = x_incl & y_excl
    z_left_excl = x_excl & y_incl

    for v in range(0, 25):
        assert z_left_incl.is_constraints_satisfied(
            v
        ) == z_left_excl.is_constraints_satisfied(v)
    assert isinstance(z_left_incl.domain, IntervalIntegerDomain)
    assert isinstance(z_left_excl.domain, IntervalIntegerDomain)
    assert z_left_incl.domain.prefer_inclusive
    assert not z_left_excl.domain.prefer_inclusive


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


def test_intersection_rejects_integer_param_with_non_bound_constraint_on_right() -> (
    None
):
    """Test ``interval & integer`` rejects a plain integer carrying a set constraint.

    A set constraint has no interval form, so the plain integer operand
    cannot be coerced and the intersection raises before any bound
    merging.
    """
    interval = create_interval_integer_param_between(0, 10)
    integer = create_integer_param()
    integer = integer.add_constraint(InSetConstraint(integer.variable, {1, 2, 3}))

    with pytest.raises(
        TypeError, match="Cannot coerce an integer parameter with non-bound constraints"
    ):
        _ = interval & integer


def test_intersection_rejects_integer_param_with_non_bound_constraint_on_left() -> None:
    """Test ``integer & interval`` rejects a set-constrained integer on the left."""
    interval = create_interval_integer_param_between(0, 10)
    integer = create_integer_param()
    integer = integer.add_constraint(InSetConstraint(integer.variable, {1, 2, 3}))

    with pytest.raises(
        TypeError, match="Cannot coerce an integer parameter with non-bound constraints"
    ):
        _ = integer & interval


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
# Non-negative attribute merging: `zero_included`
# =============================================================================


def _build_integer_operand(zero_included: bool | None) -> Param[int]:
    """Build a plain integer param for ``None``, else a natural one with the flag."""
    if zero_included is None:
        return create_integer_param()
    return create_natural_param(zero_included=zero_included)


def _build_interval_integer_operand(zero_included: bool | None) -> Param[int]:
    """Build a plain interval-integer param for ``None``, else a natural one."""
    if zero_included is None:
        return create_interval_integer_param()
    return create_interval_natural_param(zero_included=zero_included)


_NATURAL_OPERAND_BUILDERS = [
    pytest.param(_build_integer_operand, IntegerDomain, id="integer"),
    pytest.param(
        _build_interval_integer_operand, IntervalIntegerDomain, id="interval-integer"
    ),
]

_ZERO_EXCLUDING_OPERAND_PAIRS = [
    pytest.param(False, True, id="zero-excluded-and-zero-included"),
    pytest.param(True, False, id="zero-included-and-zero-excluded"),
    pytest.param(None, False, id="plain-integer-and-zero-excluded"),
    pytest.param(False, None, id="zero-excluded-and-plain-integer"),
]

_ZERO_KEEPING_OPERAND_PAIRS = [
    pytest.param(True, None, id="zero-included-and-plain-integer"),
    pytest.param(None, True, id="plain-integer-and-zero-included"),
]


@pytest.mark.z3
@pytest.mark.parametrize("build_operand, domain_type", _NATURAL_OPERAND_BUILDERS)
@pytest.mark.parametrize(
    "left_zero_included, right_zero_included", _ZERO_EXCLUDING_OPERAND_PAIRS
)
def test_intersection_domain_excludes_zero_when_a_natural_operand_does(
    build_operand: Callable[[bool | None], Param[int]],
    domain_type: type[IntegerDomain] | type[IntervalIntegerDomain],
    left_zero_included: bool | None,
    right_zero_included: bool | None,
) -> None:
    """Test the merged integer domain itself excludes zero if any operand's does.

    The result also carries the zero-excluding operand's own ``> 0``
    constraint, which rejects zero regardless of the merged domain, so
    the admissibility check is made with the carried constraints
    stripped: re-canonicalizing an empty constraint set leaves exactly
    what the merged domain implies on its own.
    """
    left = build_operand(left_zero_included)
    right = build_operand(right_zero_included)

    result = create_intersection_param(left, right)
    domain_alone = result.replace_constraints(())

    assert isinstance(result.domain, domain_type)
    assert result.domain.non_negative
    assert not result.domain.zero_included
    assert not domain_alone.is_value_valid(0)
    assert domain_alone.is_value_valid(1)


@pytest.mark.z3
@pytest.mark.parametrize("build_operand, domain_type", _NATURAL_OPERAND_BUILDERS)
@pytest.mark.parametrize(
    "left_zero_included, right_zero_included", _ZERO_KEEPING_OPERAND_PAIRS
)
def test_intersection_domain_keeps_zero_when_no_natural_operand_excludes_it(
    build_operand: Callable[[bool | None], Param[int]],
    domain_type: type[IntegerDomain] | type[IntervalIntegerDomain],
    left_zero_included: bool | None,
    right_zero_included: bool | None,
) -> None:
    """Test the merged integer domain keeps zero when no natural operand excludes it.

    The merged domain is still non-negative (one operand is natural), so
    with the carried constraints stripped it admits zero but not ``-1``.
    """
    left = build_operand(left_zero_included)
    right = build_operand(right_zero_included)

    result = create_intersection_param(left, right)
    domain_alone = result.replace_constraints(())

    assert isinstance(result.domain, domain_type)
    assert result.domain.non_negative
    assert result.domain.zero_included
    assert domain_alone.is_value_valid(0)
    assert not domain_alone.is_value_valid(-1)


# =============================================================================
# Constraint rescoping: alpha-equivalence against a hand-built parameter
# =============================================================================


@pytest.mark.z3
def test_intersection_constraints_rescope_to_result_variable() -> None:
    """Test rescoped intersection constraints match a hand-built param up to renaming.

    Each operand here carries exactly one bound constraint (a lower bound on
    the left, an upper bound on the right), so the intersection's rescoped
    conjunction is exactly the same two constraints a directly-built
    ``[5,10]`` interval-integer param would carry -- alpha-equivalent up to
    renaming of the result's own variable, which every rescoped constraint
    names.
    """
    left = create_interval_integer_param_with_lower_bound(5)
    right = create_interval_integer_param_with_upper_bound(10)
    hand_built = create_interval_integer_param_between(5, 10)

    result = create_intersection_param(left, right)

    assert result.is_alpha_equivalent(hand_built)
    for constraint in result.constraints:
        assert constraint.get_free_identifiers() == frozenset((result.variable,))


def test_intersection_categorical_constraints_rescope_to_result_variable() -> None:
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
    ``ConstraintSystem.check_satisfiability`` calls on behalf of the
    domain's ``has_feasible_value`` -- to always return ``None`` (Z3's
    "unknown" outcome), so the factory sees ``UNDECIDED`` and must not read
    it as a proven-empty intersection. Asserting the tightened bounds
    (rather than just ``result is not None``, which a ``Param``-or-raise
    factory can never fail) confirms the returned param is the real, live
    intersection and not some degenerate stand-in.
    """
    left = create_integer_param_with_lower_bound(0)
    right = create_integer_param_with_upper_bound(10)
    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.system.check_expression_satisfiability",
        lambda *args, **kwargs: None,
    )

    result = create_intersection_param(left, right)

    assert_all_satisfied(result, [0, 5, 10])
    assert_none_satisfied(result, [-1, 11])


# =============================================================================
# Unknown constraint kind
# =============================================================================


@dataclass(frozen=True, eq=False)
class _UnrescopableConstraint(Constraint):
    """A `Constraint` subclass with no rescoping rule, for negative testing."""

    variable: Identifier = field(metadata=compared_as_reference())

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.variable})

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        del bindings
        return ConstraintOutcome.SATISFIED

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def build_ordering_key(self) -> str:
        return f"_UnrescopableConstraint|{self.variable.id}"

    @override
    def __repr__(self) -> str:
        return "UnrescopableConstraint"

    @override
    def __str__(self) -> str:
        return "UnrescopableConstraint"


def test_intersection_with_unrescopable_constraint_kind_raises_constraint_error() -> (
    None
):
    """Test intersection raises ``ConstraintError`` for an unrescopable constraint."""
    left = create_integer_param()
    left = left.add_constraint(_UnrescopableConstraint(left.variable))
    right = create_integer_param()

    with pytest.raises(ConstraintError, match="Cannot rename an unexpected"):
        create_intersection_param(left, right)


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


def test_and_dunder_result_is_alpha_equivalent_to_create_intersection_param() -> None:
    """Test ``left & right`` is alpha-equivalent to the factory's result.

    Both calls mint a fresh default result variable, so the two results are
    not structurally equivalent (different variable identity) but must be
    alpha-equivalent.
    """
    left = create_categorical_param({"a", "b", "c"})
    right = create_categorical_param({"b", "c", "d"})

    dunder_result = left & right
    factory_result = create_intersection_param(left, right)

    assert dunder_result.is_alpha_equivalent(factory_result)
    assert not dunder_result.is_structurally_equivalent(factory_result)


def test_and_dunder_with_non_param_operand_raises_type_error() -> None:
    """Test ``&`` raises ``TypeError`` (via ``NotImplemented``) for a non-`Param`."""
    left = create_categorical_param({"a"})

    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = left & "not a param"  # type: ignore[operator]  # test: non-Param operand


# =============================================================================
# Cross-kind mismatch
# =============================================================================


def test_intersection_across_categorical_and_ordinal_raises_type_error() -> None:
    """Test intersection across categorical and ordinal domains raises ``TypeError``."""
    left: Param[Any] = create_categorical_param({"a", "b"})
    right = create_ordinal_param([1, 2])

    with pytest.raises(
        TypeError, match="Cannot intersect a CategoricalDomain with a domain of type"
    ):
        create_intersection_param(left, right)


def test_intersection_across_ordinal_and_categorical_raises_type_error() -> None:
    """Test intersection across ordinal and categorical domains raises ``TypeError``.

    Mirrors ``test_intersection_across_categorical_and_ordinal_raises_type_error``
    with the operand order swapped, closing the directional gap that
    ``test_param_union.py`` already covers for its own categorical/ordinal
    mismatch (``test_union_across_categorical_and_ordinal_kinds_raises_type_error``
    and its reversed sibling).
    """
    left: Param[Any] = create_ordinal_param([1, 2])
    right = create_categorical_param({"a", "b"})

    with pytest.raises(
        TypeError, match="Cannot intersect an OrdinalDomain with a domain of type"
    ):
        create_intersection_param(left, right)


def test_intersection_of_permutation_and_categorical_raises_type_error() -> None:
    """Test intersection of permutation and categorical domains raises ``TypeError``."""
    left: Param[Any] = create_permutation_param(["a", "b"])
    right = create_categorical_param({"a", "b"})

    with pytest.raises(
        TypeError, match="Cannot intersect a PermutationDomain with a domain of type"
    ):
        create_intersection_param(left, right)


def test_intersection_of_integer_and_real_raises_type_error() -> None:
    """Test intersection across integer and real domains raises ``TypeError``."""
    left: Param[Any] = create_integer_param()
    right = create_real_param()

    with pytest.raises(
        TypeError, match="Cannot intersect an IntegerDomain with a domain of type"
    ):
        create_intersection_param(left, right)


def test_intersection_of_real_and_integer_raises_type_error() -> None:
    """Test intersection with the real domain on the left raises ``TypeError``."""
    left: Param[Any] = create_real_param()
    right: Param[Any] = create_integer_param()

    with pytest.raises(
        TypeError, match="Cannot intersect a RealDomain with a domain of type"
    ):
        create_intersection_param(left, right)


def test_intersection_of_interval_integer_and_real_raises_type_error() -> None:
    """Test intersection with the interval-integer domain on the left raises."""
    left: Param[Any] = create_interval_integer_param_between(1, 5)
    right: Param[Any] = create_real_param()

    with pytest.raises(
        TypeError,
        match="Cannot intersect an IntervalIntegerDomain with a domain of type",
    ):
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
    """Test `create_intersection_param` returns a `Param` (public-surface check).

    Also confirms the actual resulting value set for this identical-operand
    case: both operands are the same singleton set, so the intersection must
    still resolve to exactly ``{"a"}``, not merely satisfy the return-type
    annotation.
    """
    left = create_categorical_param({"a"})
    right = create_categorical_param({"a"})

    result = create_intersection_param(left, right)

    assert isinstance(result, Param)
    assert_all_valid(result, ["a"])
    assert result.domain.categories == ("a",)  # type: ignore[attr-defined]


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
    assert left.check_subset(result) is ConstraintOutcome.VIOLATED


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


# =============================================================================
# Operand variables are unified onto the result variable
# =============================================================================


def _create_undecided_integer_param(name: str, identifier_id: int) -> Param[int]:
    """Create an integer parameter whose feasibility the solver cannot decide.

    `v / v != 1` trips the division hazard screen, since the divisor is
    not a nonzero literal, so the seam refuses to lower it and reports no
    decision at all.
    """
    variable = mock_identifier(name, identifier_id)
    hazardous = (
        IdentifierExpression(variable) / IdentifierExpression(variable)
    ).not_equals(1)
    return create_integer_param(
        name=variable, constraints=[EquationConstraint(hazardous)]
    )


def _create_empty_integer_param(name: str, identifier_id: int) -> Param[int]:
    """Create an integer parameter the solver proves infeasible: `v < 0 and v > 0`."""
    variable = mock_identifier(name, identifier_id)
    expression = IdentifierExpression(variable)
    return create_integer_param(
        name=variable,
        constraints=[
            EquationConstraint(expression < 0),
            EquationConstraint(expression > 0),
        ],
    )


def _assert_scoped_to_result_variable(result: Param[Any]) -> None:
    """Assert every constraint of `result` mentions only `result.variable`."""
    for constraint in result.constraints:
        assert constraint.get_free_identifiers() == frozenset({result.variable})


@pytest.mark.z3
def test_intersection_unifies_the_other_operands_variable_into_the_result() -> None:
    """Test `x <= y` & `y >= 10` yields constraints decided without bindings.

    The result variable stands for both operands' quantities, so `y`
    inside the left operand's constraint is renamed to it as well; the
    carried constraints then mention only the result variable.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    left = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) <= IdentifierExpression(y))
        ],
    )
    right = create_integer_param(
        name=y, constraints=[EquationConstraint(IdentifierExpression(y) >= 10)]
    )

    result = create_intersection_param(left, right)

    _assert_scoped_to_result_variable(result)
    assert_all_valid(result, [10, 12])
    assert_none_valid(result, [9])


@pytest.mark.z3
@pytest.mark.parametrize(
    "is_dependent_operand_left",
    [True, False],
    ids=["dependent-left", "dependent-right"],
)
def test_intersection_raises_when_the_unified_conjunction_is_provably_empty(
    is_dependent_operand_left: bool,
) -> None:
    """Test `x < y` & `y >= 10` unifies to `param < param` and raises `ParamError`."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
        ],
    )
    bounded = create_integer_param(
        name=y, constraints=[EquationConstraint(IdentifierExpression(y) >= 10)]
    )
    left, right = (
        (dependent, bounded) if is_dependent_operand_left else (bounded, dependent)
    )

    with pytest.raises(ParamError, match="empty"):
        create_intersection_param(left, right)


@pytest.mark.z3
def test_intersection_leaves_a_third_party_identifier_free() -> None:
    """Test an identifier that is neither operand's variable stays free in the result.

    `z` remains substitutable through bindings, and the result comes back
    live with feasibility undecided, since the constraint mentioning `z`
    cannot be posed to the solver.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    z = mock_identifier("z", 3)
    left = create_integer_param(
        name=x,
        constraints=[
            EquationConstraint(IdentifierExpression(x) < IdentifierExpression(z))
        ],
    )
    right = create_integer_param(
        name=y, constraints=[EquationConstraint(IdentifierExpression(y) >= 10)]
    )

    result = create_intersection_param(left, right)

    assert z in result.constraint_system.get_free_identifiers()
    assert result.check_feasibility() is ConstraintOutcome.UNDECIDED
    assert result.is_value_valid(12, bindings={z: 50})
    assert not result.is_value_valid(12, bindings={z: 5})


@pytest.mark.z3
def test_intersection_of_operands_sharing_one_variable_conjoins_constraints() -> None:
    """Test operands over the same variable object intersect to their conjunction."""
    x = mock_identifier("x", 1)
    left = create_integer_param(
        name=x, constraints=[EquationConstraint(IdentifierExpression(x) < 20)]
    )
    right = create_integer_param(
        name=x, constraints=[EquationConstraint(IdentifierExpression(x) >= 10)]
    )

    result = create_intersection_param(left, right)

    _assert_scoped_to_result_variable(result)
    assert_all_valid(result, [10, 19])
    assert_none_valid(result, [9, 20])


# =============================================================================
# An undecided conjunction still raises when an operand is provably empty
# =============================================================================


@pytest.mark.z3
@pytest.mark.parametrize(
    "is_empty_operand_left", [True, False], ids=["empty-left", "empty-right"]
)
def test_intersection_with_a_provably_empty_operand_raises_despite_undecided_result(
    is_empty_operand_left: bool,
) -> None:
    """Test an undecided conjunction raises `ParamError` when an operand is empty.

    The hazardous conjunct leaves the whole conjunction undecided, so the
    factory consults each operand's own feasibility; an intersection with
    an empty set is empty.
    """
    undecided = _create_undecided_integer_param("a", 1)
    empty = _create_empty_integer_param("e", 2)
    left, right = (empty, undecided) if is_empty_operand_left else (undecided, empty)

    with pytest.raises(ParamError, match="empty"):
        create_intersection_param(left, right)


@pytest.mark.z3
def test_intersection_of_two_undecided_operands_is_returned_live() -> None:
    """Test an undecided conjunction of undecided operands is returned, not raised."""
    left = _create_undecided_integer_param("a", 1)
    right = _create_undecided_integer_param("b", 2)

    result = create_intersection_param(left, right)

    assert result.check_feasibility() is ConstraintOutcome.UNDECIDED


# =============================================================================
# Real domain: a not-in-set constraint's numeric member does not decide emptiness
# =============================================================================


@pytest.mark.z3
def test_real_intersection_excluding_a_shared_float_value_is_returned_live() -> None:
    """Test a real intersection narrowed by its own float value is returned live.

    ``c`` is the real singleton `0.5` and ``d`` excludes the `float`
    member `0.5`; Z3 lowers both to the same rational, so the solver
    reports the conjunction emptied. Type-strict membership still admits
    the decimal-string kind of `0.5` for both operands, so the
    conjunction is not actually empty and the factory must return a live
    result whose feasibility is UNDECIDED rather than raise `ParamError`.
    """
    c = create_real_param_between(0.5, 0.5, name=mock_identifier("c", 1))
    w = mock_identifier("w", 2)
    d = create_real_param(name=w, constraints=[NotInSetConstraint(w, {0.5})])

    result = create_intersection_param(c, d)

    assert result.check_feasibility() is ConstraintOutcome.UNDECIDED


# =============================================================================
# Mixed coercion: a set member that does not lift is a non-bound constraint
# =============================================================================


@pytest.mark.parametrize(
    "is_plain_operand_left", [True, False], ids=["plain-left", "plain-right"]
)
def test_intersection_with_a_non_liftable_set_member_on_the_plain_operand_raises(
    is_plain_operand_left: bool,
) -> None:
    """Test a plain integer operand whose set member does not lift raises `TypeError`.

    The member cannot be lifted to an expression at all, so the operand
    has no interval form; the lifting `ConstraintError` is chained as the
    cause.
    """
    i = mock_identifier("i", 1)
    plain = create_integer_param(name=i, constraints=[InSetConstraint(i, ("s",))])
    interval = create_interval_integer_param_between(0, 5)
    left, right = (plain, interval) if is_plain_operand_left else (interval, plain)

    with pytest.raises(TypeError, match="non-bound constraints") as excinfo:
        create_intersection_param(left, right)

    assert isinstance(excinfo.value.__cause__, ConstraintError)


# =============================================================================
# An ill-typed conjunction raises rather than being returned live
# =============================================================================


def test_intersection_raises_for_a_number_in_a_case_condition() -> None:
    """Test an ill-typed operand's error propagates from the emptiness check.

    `create_intersection_param` decides emptiness through
    `check_feasibility`, which refuses a number in a Boolean position.
    Returning the conjunction live as undecided would hand back a
    parameter no query can answer.
    """
    x = mock_identifier("x", 1)
    ill_typed = create_integer_param(
        name=x,
        constraints=[build_case_condition_constraint(IdentifierExpression(x) + 1)],
    )
    well_typed = create_integer_param(name=mock_identifier("y", 2))

    with pytest.raises(NonBooleanLogicalOperandError):
        create_intersection_param(ill_typed, well_typed)
