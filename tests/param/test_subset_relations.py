"""Tests for `Param.is_subset` and `Param.is_value_set_subset` across param types."""

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.param import (
    CategoricalParam,
    IntParam,
    OrdinalParam,
    PermParam,
    RealParam,
)

# =============================================================================
# Same-class subset relations - RealParam / IntParam
# =============================================================================


def test_unconstrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test two unconstrained `RealParam`s are mutual subsets of each other."""
    left = RealParam()
    right = RealParam()
    assert left.is_subset(right)
    assert right.is_subset(left)


def test_constrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test a constrained `RealParam` is a subset of an unconstrained one only."""
    constrained = RealParam()
    constrained = constrained.add_constraint(
        EquationConstraint(constrained.variable, constrained.variable_expression > 0)
    )
    unconstrained = RealParam()
    assert constrained.is_subset(unconstrained)
    assert not unconstrained.is_subset(constrained)


def test_narrower_interval_real_param_is_subset_of_wider_interval_real_param() -> None:
    """Test a narrower-interval `RealParam` is a subset of a wider-interval one."""
    wider = RealParam()
    wider = wider.add_constraint(
        EquationConstraint(wider.variable, wider.variable_expression >= 0)
    )
    wider = wider.add_constraint(
        EquationConstraint(wider.variable, wider.variable_expression <= 3)
    )
    narrower = RealParam()
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable, narrower.variable_expression >= 0)
    )
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable, narrower.variable_expression <= 2)
    )
    assert not wider.is_subset(narrower)
    assert narrower.is_subset(wider)


# =============================================================================
# Cross-class subset relations
# =============================================================================


def test_is_subset_returns_false_for_different_param_classes() -> None:
    """Test `is_subset` returns ``False`` between different `Param` subclasses.

    Pins down the early-return guard against cross-class subset queries: an
    `IntParam` is not a subset of a `RealParam` even when both are unconstrained.
    """
    assert not IntParam().is_subset(RealParam())  # type: ignore[arg-type]  # test: cross class
    assert not RealParam().is_subset(IntParam())  # type: ignore[arg-type]  # test: cross class


def test_ordinal_param_is_subset_returns_false_against_categorical_param() -> None:
    """Test cross-family `is_subset` is ``False`` even with identical value sets."""
    ordinal: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    categorical: CategoricalParam[int] = CategoricalParam({1, 2, 3})
    assert not ordinal.is_subset(categorical)
    assert not categorical.is_subset(ordinal)


# =============================================================================
# is_value_set_subset - RealParam / IntParam
# =============================================================================


def test_real_param_is_value_set_subset_is_true_for_other_real_param() -> None:
    """Test `RealParam.is_value_set_subset` returns ``True`` for any other `RealParam`.

    `RealParam`'s value set is the full reals; narrowing happens through
    constraints, not through the value-set domain.
    """
    assert RealParam().is_value_set_subset(RealParam())
    assert RealParam.with_lower_bound(0.0).is_value_set_subset(
        RealParam.with_upper_bound(10.0)
    )


def test_int_param_is_value_set_subset_is_true_for_other_int_param() -> None:
    """Test `IntParam.is_value_set_subset` returns ``True`` for any other `IntParam`.

    `IntParam`'s value set is the full integers; narrowing happens through
    constraints, not through the value-set domain.
    """
    assert IntParam().is_value_set_subset(IntParam())
    assert IntParam.with_lower_bound(0).is_value_set_subset(
        IntParam.with_upper_bound(10)
    )


def test_real_param_is_value_set_subset_returns_false_against_non_real() -> None:
    """Test `RealParam.is_value_set_subset` returns ``False`` for a non-`RealParam`."""
    assert not RealParam().is_value_set_subset(IntParam())  # type: ignore[arg-type]  # test: cross class


def test_int_param_is_value_set_subset_returns_false_against_non_int() -> None:
    """Test `IntParam.is_value_set_subset` returns ``False`` for a non-`IntParam`."""
    assert not IntParam().is_value_set_subset(RealParam())  # type: ignore[arg-type]  # test: cross class


# =============================================================================
# OrdinalParam subset relations
# =============================================================================


def test_ordinal_param_with_equal_value_sets_is_mutual_subset() -> None:
    """Test two `OrdinalParam`s with equal value sets are mutual subsets."""
    left: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    right: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert left.is_subset(right)
    assert right.is_subset(left)


def test_ordinal_param_strict_value_set_subset_is_one_directional() -> None:
    """Test a strict `OrdinalParam` value-set subset is one-directional."""
    smaller: OrdinalParam[int] = OrdinalParam([1, 2])
    larger: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_ordinal_param_disjoint_value_sets_are_not_subsets() -> None:
    """Test two `OrdinalParam`s with disjoint value sets are not subsets."""
    left: OrdinalParam[int] = OrdinalParam([1, 2])
    right: OrdinalParam[int] = OrdinalParam([3, 4])
    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_ordinal_param_with_extra_constraint_is_strict_subset() -> None:
    """Test an `OrdinalParam` with an extra in-set constraint is a strict subset."""
    base: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    narrowed = base.add_constraint(InSetConstraint(base.variable, {1, 2}))
    assert narrowed.is_subset(base)
    assert not base.is_subset(narrowed)


def test_ordinal_param_is_value_set_subset_returns_true_for_equal_sets() -> None:
    """Test `OrdinalParam.is_value_set_subset` returns ``True`` for equal value sets."""
    left: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    right: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert left.is_value_set_subset(right)


def test_ordinal_param_is_value_set_subset_returns_true_for_strict_subset() -> None:
    """Test `OrdinalParam.is_value_set_subset` returns ``True`` for a strict subset."""
    smaller: OrdinalParam[int] = OrdinalParam([1, 2])
    larger: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert smaller.is_value_set_subset(larger)


def test_ordinal_param_is_value_set_subset_returns_false_for_strict_superset() -> None:
    """Test `OrdinalParam.is_value_set_subset` is ``False`` for a strict superset."""
    smaller: OrdinalParam[int] = OrdinalParam([1, 2])
    larger: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert not larger.is_value_set_subset(smaller)


def test_ordinal_param_is_value_set_subset_returns_false_for_disjoint_sets() -> None:
    """Test `OrdinalParam.is_value_set_subset` returns ``False`` for disjoint sets."""
    left: OrdinalParam[int] = OrdinalParam([1, 2])
    right: OrdinalParam[int] = OrdinalParam([3, 4])
    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_ordinal_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test `OrdinalParam.is_value_set_subset` honors the bool/int distinction.

    A `True` value set is not a subset of a `1` value set, despite ``True == 1``.
    Pins the per-element bool/int mismatch check at the value-set level.
    """
    bool_param: OrdinalParam[bool] = OrdinalParam([True, False])
    int_param: OrdinalParam[int] = OrdinalParam([0, 1])
    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: bool vs int
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: bool vs int


def test_ordinal_param_is_value_set_subset_uses_value_equality_not_identity() -> None:
    """Test `OrdinalParam.is_value_set_subset` matches values by ``==``, not ``is``.

    Constructs two equal-but-non-identical ``float`` instances via
    ``float("1.5")`` so the value-set match must use ``==``. An identity-based
    comparison would report the two sets as disjoint.
    """
    left: OrdinalParam[float] = OrdinalParam([float("1.5"), float("2.5")])
    right: OrdinalParam[float] = OrdinalParam([float("1.5"), float("2.5")])
    assert left.is_value_set_subset(right)
    assert right.is_value_set_subset(left)


def test_ordinal_param_is_value_set_subset_returns_false_against_non_ordinal() -> None:
    """Test `OrdinalParam.is_value_set_subset` rejects a non-`OrdinalParam` other.

    The defensive `isinstance` guard inside the override returns ``False`` so
    direct callers (bypassing `is_subset`) see a deterministic answer.
    """
    ordinal: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    not_ordinal: CategoricalParam[int] = CategoricalParam({1, 2, 3})
    assert not ordinal.is_value_set_subset(not_ordinal)


# =============================================================================
# CategoricalParam subset relations
# =============================================================================


def test_categorical_param_with_equal_categories_is_mutual_subset() -> None:
    """Test two `CategoricalParam`s with equal categories are mutual subsets."""
    left: CategoricalParam[str] = CategoricalParam({"a", "b"})
    right: CategoricalParam[str] = CategoricalParam({"a", "b"})
    assert left.is_subset(right)
    assert right.is_subset(left)


def test_categorical_param_strict_category_subset_is_one_directional() -> None:
    """Test a strict `CategoricalParam` category subset is one-directional."""
    smaller: CategoricalParam[str] = CategoricalParam({"a", "b"})
    larger: CategoricalParam[str] = CategoricalParam({"a", "b", "c"})
    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_disjoint_categories_are_not_subsets() -> None:
    """Test two `CategoricalParam`s with disjoint categories are not subsets."""
    left: CategoricalParam[str] = CategoricalParam({"a", "b"})
    right: CategoricalParam[str] = CategoricalParam({"c", "d"})
    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_categorical_param_is_value_set_subset_returns_true_for_equal_categories() -> (
    None
):
    """Test `CategoricalParam.is_value_set_subset` returns ``True`` for equal sets."""
    left: CategoricalParam[str] = CategoricalParam({"a", "b"})
    right: CategoricalParam[str] = CategoricalParam({"a", "b"})
    assert left.is_value_set_subset(right)


def test_categorical_param_is_value_set_subset_returns_true_for_strict_subset() -> None:
    """Test `CategoricalParam.is_value_set_subset` is ``True`` for a strict subset."""
    smaller: CategoricalParam[str] = CategoricalParam({"a", "b"})
    larger: CategoricalParam[str] = CategoricalParam({"a", "b", "c"})
    assert smaller.is_value_set_subset(larger)


def test_categorical_param_is_value_set_subset_returns_false_for_strict_superset() -> (
    None
):
    """Test `CategoricalParam.is_value_set_subset` returns ``False`` for a superset."""
    smaller: CategoricalParam[str] = CategoricalParam({"a", "b"})
    larger: CategoricalParam[str] = CategoricalParam({"a", "b", "c"})
    assert not larger.is_value_set_subset(smaller)


def test_categorical_param_is_value_set_subset_returns_false_for_disjoint() -> None:
    """Test `CategoricalParam.is_value_set_subset` is ``False`` for disjoint sets."""
    left: CategoricalParam[str] = CategoricalParam({"a", "b"})
    right: CategoricalParam[str] = CategoricalParam({"c", "d"})
    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_categorical_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test `CategoricalParam.is_value_set_subset` honors the bool/int distinction."""
    bool_param: CategoricalParam[bool] = CategoricalParam({True, False})
    int_param: CategoricalParam[int] = CategoricalParam({0, 1})
    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: bool vs int
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: bool vs int


def test_categorical_param_is_value_set_subset_returns_false_against_non_categorical() -> (  # noqa: E501
    None
):
    """Test `CategoricalParam.is_value_set_subset` rejects a non-`CategoricalParam`.

    The defensive `isinstance` guard inside the override returns ``False`` so
    direct callers (bypassing `is_subset`) see a deterministic answer.
    """
    categorical: CategoricalParam[int] = CategoricalParam({1, 2, 3})
    not_categorical: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert not categorical.is_value_set_subset(not_categorical)


# =============================================================================
# PermParam subset relations
# =============================================================================


def test_perm_param_with_same_member_set_is_mutual_subset() -> None:
    """Test two `PermParam`s with the same member set are mutual subsets.

    The set of admissible permutations depends only on the *set* of members,
    not their order; two `PermParam`s with the same members in different
    orders generate the same set of admissible permutations.
    """
    left: PermParam[int] = PermParam([1, 2, 3])
    right: PermParam[int] = PermParam([3, 2, 1])
    assert left.is_subset(right)
    assert right.is_subset(left)


def test_perm_param_with_disjoint_members_is_not_subset() -> None:
    """Test two `PermParam`s with disjoint members are not subsets."""
    left: PermParam[int] = PermParam([1, 2, 3])
    right: PermParam[int] = PermParam([4, 5, 6])
    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_perm_param_with_subset_members_is_not_subset() -> None:
    """Test a `PermParam` with a subset of members is not a permutation subset.

    Permutations of ``{1, 2}`` are length-2 tuples; permutations of
    ``{1, 2, 3}`` are length-3 tuples. The two sets are disjoint, so neither
    is a subset of the other even though the member sets nest.
    """
    smaller: PermParam[int] = PermParam([1, 2])
    larger: PermParam[int] = PermParam([1, 2, 3])
    assert not smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_perm_param_is_value_set_subset_returns_true_for_same_member_set() -> None:
    """Test `PermParam.is_value_set_subset` returns ``True`` for the same member set."""
    left: PermParam[int] = PermParam([1, 2, 3])
    right: PermParam[int] = PermParam([3, 2, 1])
    assert left.is_value_set_subset(right)
    assert right.is_value_set_subset(left)


def test_perm_param_is_value_set_subset_returns_false_for_disjoint_members() -> None:
    """Test `PermParam.is_value_set_subset` returns ``False`` for disjoint members."""
    left: PermParam[int] = PermParam([1, 2, 3])
    right: PermParam[int] = PermParam([4, 5, 6])
    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_perm_param_is_value_set_subset_returns_false_for_subset_members() -> None:
    """Test `PermParam.is_value_set_subset` returns ``False`` for subset members.

    Member sets that nest still produce disjoint permutation sets, so the
    value-set subset check must reject the inclusion.
    """
    smaller: PermParam[int] = PermParam([1, 2])
    larger: PermParam[int] = PermParam([1, 2, 3])
    assert not smaller.is_value_set_subset(larger)
    assert not larger.is_value_set_subset(smaller)


def test_perm_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test `PermParam.is_value_set_subset` honors the bool/int distinction."""
    bool_param: PermParam[bool] = PermParam([True, False])
    int_param: PermParam[int] = PermParam([0, 1])
    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: bool vs int
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: bool vs int


def test_perm_param_is_value_set_subset_returns_false_against_non_perm() -> None:
    """Test `PermParam.is_value_set_subset` rejects a non-`PermParam` other.

    The defensive `isinstance` guard inside the override returns ``False`` so
    direct callers (bypassing `is_subset`) see a deterministic answer.
    """
    perm: PermParam[int] = PermParam([1, 2, 3])
    not_perm: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    assert not perm.is_value_set_subset(not_perm)  # type: ignore[arg-type]  # test: cross class


# =============================================================================
# Discrete-set subset semantics (no Z3 path)
# =============================================================================


def test_categorical_param_subset_with_string_values_does_not_raise() -> None:
    """Test `CategoricalParam.is_subset` works for non-numeric string categories."""
    smaller: CategoricalParam[str] = CategoricalParam(["red", "blue"])
    larger: CategoricalParam[str] = CategoricalParam(["red", "blue", "green"])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_subset_with_int_values() -> None:
    """Test ``CategoricalParam.is_subset`` over int categories using set semantics."""
    smaller: CategoricalParam[int] = CategoricalParam([1, 2])
    larger: CategoricalParam[int] = CategoricalParam([1, 2, 3])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_subset_respects_in_set_constraint() -> None:
    """Test ``CategoricalParam.is_subset`` honors an ``InSetConstraint`` narrowing."""
    universe: CategoricalParam[int] = CategoricalParam([1, 2, 3])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {1, 2}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_ordinal_param_subset_with_int_values() -> None:
    """Test ``OrdinalParam.is_subset`` over integer values using set semantics."""
    smaller: OrdinalParam[int] = OrdinalParam([1, 2])
    larger: OrdinalParam[int] = OrdinalParam([1, 2, 3])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_ordinal_param_subset_respects_in_set_constraint() -> None:
    """Test ``OrdinalParam.is_subset`` honors an ``InSetConstraint`` narrowing."""
    universe: OrdinalParam[int] = OrdinalParam([1, 2, 3])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {1, 2}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_perm_param_subset_distinguishes_different_member_sets() -> None:
    """Test ``PermParam.is_subset`` returns False when member sets differ.

    Different universes produce disjoint permutation sets, so neither
    direction holds.
    """
    smaller: PermParam[int] = PermParam([1, 2])
    larger: PermParam[int] = PermParam([1, 2, 3])

    assert not smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_perm_param_subset_respects_in_set_constraint() -> None:
    """Test ``PermParam.is_subset`` honors an ``InSetConstraint`` narrowing."""
    universe: PermParam[int] = PermParam([1, 2])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {(1, 2)}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_discrete_params_reject_cross_class_subset_check() -> None:
    """Test ``is_subset`` returns False across distinct discrete-param classes."""
    cat: CategoricalParam[int] = CategoricalParam([1])
    ordinal: OrdinalParam[int] = OrdinalParam([1])
    perm: PermParam[int] = PermParam([1])

    assert not cat.is_subset(ordinal)
    assert not cat.is_subset(perm)  # type: ignore[arg-type]  # test: cross class
    assert not ordinal.is_subset(cat)
    assert not ordinal.is_subset(perm)  # type: ignore[arg-type]  # test: cross class
    assert not perm.is_subset(cat)  # type: ignore[arg-type]  # test: cross class
    assert not perm.is_subset(ordinal)  # type: ignore[arg-type]  # test: cross class
