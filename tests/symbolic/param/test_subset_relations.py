"""Tests for `Param.is_subset` and `Param.is_value_set_subset` across param kinds."""

from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.param import (
    create_categorical_param,
    create_integer_param,
    create_integer_param_with_lower_bound,
    create_integer_param_with_upper_bound,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
    create_real_param_with_lower_bound,
    create_real_param_with_upper_bound,
)

# =============================================================================
# Same-kind subset relations - real / integer params
# =============================================================================


def test_unconstrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test two unconstrained real params are mutual subsets of each other."""
    left = create_real_param()
    right = create_real_param()

    assert left.is_subset(right)
    assert right.is_subset(left)


def test_constrained_real_param_is_subset_of_unconstrained_real_param() -> None:
    """Test a constrained real param is a subset of an unconstrained one only."""
    constrained = create_real_param()
    constrained = constrained.add_constraint(
        EquationConstraint(constrained.variable_expression > 0)
    )
    unconstrained = create_real_param()

    assert constrained.is_subset(unconstrained)
    assert not unconstrained.is_subset(constrained)


def test_narrower_interval_real_param_is_subset_of_wider_interval_real_param() -> None:
    """Test a narrower-interval real param is a subset of a wider-interval one."""
    wider = create_real_param()
    wider = wider.add_constraint(EquationConstraint(wider.variable_expression >= 0))
    wider = wider.add_constraint(EquationConstraint(wider.variable_expression <= 3))
    narrower = create_real_param()
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable_expression >= 0)
    )
    narrower = narrower.add_constraint(
        EquationConstraint(narrower.variable_expression <= 2)
    )

    assert not wider.is_subset(narrower)
    assert narrower.is_subset(wider)


# =============================================================================
# Cross-space / cross-family subset relations
# =============================================================================


def test_is_subset_returns_false_across_value_spaces() -> None:
    """Test `is_subset` returns ``False`` across distinct numeric value spaces.

    Pins down value-space gating: an integer param and a real param occupy
    different Z3 sorts (``INT`` vs ``REAL``), so neither is a subset of the
    other even when both are unconstrained. This is symmetric.
    """
    assert not create_integer_param().is_subset(create_real_param())  # type: ignore[arg-type]  # test: cross-value-space comparison
    assert not create_real_param().is_subset(create_integer_param())  # type: ignore[arg-type]  # test: cross-value-space comparison


def test_ordinal_param_is_subset_returns_false_against_categorical_param() -> None:
    """Test cross-family `is_subset` is ``False`` even with identical value sets."""
    ordinal = create_ordinal_param([1, 2, 3])
    categorical = create_categorical_param({1, 2, 3})

    assert not ordinal.is_subset(categorical)
    assert not categorical.is_subset(ordinal)


# =============================================================================
# is_value_set_subset - real / integer params
# =============================================================================


def test_real_param_is_value_set_subset_is_true_for_other_real_param() -> None:
    """Test a real param's value set is a subset of any other real param's.

    A real param's value set is the full reals; narrowing happens through
    constraints, not through the value-set domain.
    """
    assert create_real_param().is_value_set_subset(create_real_param())
    assert create_real_param_with_lower_bound(0.0).is_value_set_subset(
        create_real_param_with_upper_bound(10.0)
    )


def test_int_param_is_value_set_subset_is_true_for_other_int_param() -> None:
    """Test an integer param's value set is a subset of any other integer param's.

    An integer param's value set is the full integers; narrowing happens
    through constraints, not through the value-set domain.
    """
    assert create_integer_param().is_value_set_subset(create_integer_param())
    assert create_integer_param_with_lower_bound(0).is_value_set_subset(
        create_integer_param_with_upper_bound(10)
    )


def test_real_param_is_value_set_subset_returns_false_against_non_real() -> None:
    """Test a real param's value set is not a subset of an integer param's.

    Integer and real domains occupy different value spaces, so the value-set
    subset check rejects the cross-space comparison.
    """
    assert not create_real_param().is_value_set_subset(create_integer_param())  # type: ignore[arg-type]  # test: cross-value-space comparison


def test_int_param_is_value_set_subset_returns_false_against_non_int() -> None:
    """Test an integer param's value set is not a subset of a real param's.

    Integer and real domains occupy different value spaces, so the value-set
    subset check rejects the cross-space comparison.
    """
    assert not create_integer_param().is_value_set_subset(create_real_param())  # type: ignore[arg-type]  # test: cross-value-space comparison


# =============================================================================
# Ordinal param subset relations
# =============================================================================


def test_ordinal_param_with_equal_value_sets_is_mutual_subset() -> None:
    """Test two ordinal params with equal value sets are mutual subsets."""
    left = create_ordinal_param([1, 2, 3])
    right = create_ordinal_param([1, 2, 3])

    assert left.is_subset(right)
    assert right.is_subset(left)


def test_ordinal_param_strict_value_set_subset_is_one_directional() -> None:
    """Test a strict ordinal value-set subset is one-directional."""
    smaller = create_ordinal_param([1, 2])
    larger = create_ordinal_param([1, 2, 3])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_ordinal_param_disjoint_value_sets_are_not_subsets() -> None:
    """Test two ordinal params with disjoint value sets are not subsets."""
    left = create_ordinal_param([1, 2])
    right = create_ordinal_param([3, 4])

    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_ordinal_param_with_extra_constraint_is_strict_subset() -> None:
    """Test an ordinal param with an extra in-set constraint is a strict subset."""
    base = create_ordinal_param([1, 2, 3])
    narrowed = base.add_constraint(InSetConstraint(base.variable, {1, 2}))

    assert narrowed.is_subset(base)
    assert not base.is_subset(narrowed)


def test_ordinal_param_is_value_set_subset_returns_true_for_equal_sets() -> None:
    """Test an ordinal param's value set is a subset of an equal value set."""
    left = create_ordinal_param([1, 2, 3])
    right = create_ordinal_param([1, 2, 3])

    assert left.is_value_set_subset(right)


def test_ordinal_param_is_value_set_subset_returns_true_for_strict_subset() -> None:
    """Test an ordinal param's value set is a subset of a strict superset."""
    smaller = create_ordinal_param([1, 2])
    larger = create_ordinal_param([1, 2, 3])

    assert smaller.is_value_set_subset(larger)


def test_ordinal_param_is_value_set_subset_returns_false_for_strict_superset() -> None:
    """Test an ordinal param's value set is not a subset of a strict subset."""
    smaller = create_ordinal_param([1, 2])
    larger = create_ordinal_param([1, 2, 3])

    assert not larger.is_value_set_subset(smaller)


def test_ordinal_param_is_value_set_subset_returns_false_for_disjoint_sets() -> None:
    """Test an ordinal param's value set is not a subset of a disjoint set."""
    left = create_ordinal_param([1, 2])
    right = create_ordinal_param([3, 4])

    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_ordinal_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test the ordinal value-set subset check honors the bool/int distinction.

    A `True` value set is not a subset of a `1` value set, despite ``True == 1``.
    Pins the per-element bool/int mismatch check at the value-set level.
    """
    bool_param = create_ordinal_param([True, False])
    int_param = create_ordinal_param([0, 1])

    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison


def test_ordinal_param_is_value_set_subset_uses_value_equality_not_identity() -> None:
    """Test the ordinal value-set subset check matches values by ``==``, not ``is``.

    Constructs two equal-but-non-identical ``float`` instances via
    ``float("1.5")`` so the value-set match must use ``==``. An identity-based
    comparison would report the two sets as disjoint.
    """
    left = create_ordinal_param([float("1.5"), float("2.5")])
    right = create_ordinal_param([float("1.5"), float("2.5")])

    assert left.is_value_set_subset(right)
    assert right.is_value_set_subset(left)


def test_ordinal_param_is_value_set_subset_returns_false_against_non_ordinal() -> None:
    """Test the ordinal value-set subset check rejects a non-ordinal other.

    The defensive domain-kind guard returns ``False`` so direct callers
    (bypassing `is_subset`) see a deterministic answer.
    """
    ordinal = create_ordinal_param([1, 2, 3])
    not_ordinal = create_categorical_param({1, 2, 3})

    assert not ordinal.is_value_set_subset(not_ordinal)


# =============================================================================
# Categorical param subset relations
# =============================================================================


def test_categorical_param_with_equal_categories_is_mutual_subset() -> None:
    """Test two categorical params with equal categories are mutual subsets."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"a", "b"})

    assert left.is_subset(right)
    assert right.is_subset(left)


def test_categorical_param_strict_category_subset_is_one_directional() -> None:
    """Test a strict categorical category subset is one-directional."""
    smaller = create_categorical_param({"a", "b"})
    larger = create_categorical_param({"a", "b", "c"})

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_disjoint_categories_are_not_subsets() -> None:
    """Test two categorical params with disjoint categories are not subsets."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_categorical_param_is_value_set_subset_returns_true_for_equal_categories() -> (
    None
):
    """Test the categorical value-set subset check is ``True`` for equal sets."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"a", "b"})

    assert left.is_value_set_subset(right)


def test_categorical_param_is_value_set_subset_returns_true_for_strict_subset() -> None:
    """Test the categorical value-set subset check is ``True`` for a strict subset."""
    smaller = create_categorical_param({"a", "b"})
    larger = create_categorical_param({"a", "b", "c"})

    assert smaller.is_value_set_subset(larger)


def test_categorical_param_is_value_set_subset_returns_false_for_strict_superset() -> (
    None
):
    """Test the categorical value-set subset check is ``False`` for a superset."""
    smaller = create_categorical_param({"a", "b"})
    larger = create_categorical_param({"a", "b", "c"})

    assert not larger.is_value_set_subset(smaller)


def test_categorical_param_is_value_set_subset_returns_false_for_disjoint() -> None:
    """Test the categorical value-set subset check is ``False`` for disjoint sets."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_categorical_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test the categorical value-set subset check honors the bool/int distinction."""
    bool_param = create_categorical_param({True, False})
    int_param = create_categorical_param({0, 1})

    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison


def test_categorical_param_is_value_set_subset_returns_false_against_non_categorical() -> (  # noqa: E501
    None
):
    """Test the categorical value-set subset check rejects a non-categorical other.

    The defensive domain-kind guard returns ``False`` so direct callers
    (bypassing `is_subset`) see a deterministic answer.
    """
    categorical = create_categorical_param({1, 2, 3})
    not_categorical = create_ordinal_param([1, 2, 3])

    assert not categorical.is_value_set_subset(not_categorical)


# =============================================================================
# Permutation param subset relations
# =============================================================================


def test_perm_param_with_same_member_set_is_mutual_subset() -> None:
    """Test two permutation params with the same member set are mutual subsets.

    The set of admissible permutations depends only on the *set* of members,
    not their order; two permutation params with the same members in different
    orders generate the same set of admissible permutations.
    """
    left = create_permutation_param([1, 2, 3])
    right = create_permutation_param([3, 2, 1])

    assert left.is_subset(right)
    assert right.is_subset(left)


def test_perm_param_with_disjoint_members_is_not_subset() -> None:
    """Test two permutation params with disjoint members are not subsets."""
    left = create_permutation_param([1, 2, 3])
    right = create_permutation_param([4, 5, 6])

    assert not left.is_subset(right)
    assert not right.is_subset(left)


def test_perm_param_with_subset_members_is_not_subset() -> None:
    """Test a permutation param with a subset of members is not a subset.

    Permutations of ``{1, 2}`` are length-2 tuples; permutations of
    ``{1, 2, 3}`` are length-3 tuples. The two sets are disjoint, so neither
    is a subset of the other even though the member sets nest.
    """
    smaller = create_permutation_param([1, 2])
    larger = create_permutation_param([1, 2, 3])

    assert not smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_perm_param_is_value_set_subset_returns_true_for_same_member_set() -> None:
    """Test the permutation value-set subset check is ``True`` for the same members."""
    left = create_permutation_param([1, 2, 3])
    right = create_permutation_param([3, 2, 1])

    assert left.is_value_set_subset(right)
    assert right.is_value_set_subset(left)


def test_perm_param_is_value_set_subset_returns_false_for_disjoint_members() -> None:
    """Test the permutation value-set subset check is ``False`` for disjoint members."""
    left = create_permutation_param([1, 2, 3])
    right = create_permutation_param([4, 5, 6])

    assert not left.is_value_set_subset(right)
    assert not right.is_value_set_subset(left)


def test_perm_param_is_value_set_subset_returns_false_for_subset_members() -> None:
    """Test the permutation value-set subset check is ``False`` for subset members.

    Member sets that nest still produce disjoint permutation sets, so the
    value-set subset check must reject the inclusion.
    """
    smaller = create_permutation_param([1, 2])
    larger = create_permutation_param([1, 2, 3])

    assert not smaller.is_value_set_subset(larger)
    assert not larger.is_value_set_subset(smaller)


def test_perm_param_is_value_set_subset_distinguishes_bool_from_int() -> None:
    """Test the permutation value-set subset check honors the bool/int distinction."""
    bool_param = create_permutation_param([True, False])
    int_param = create_permutation_param([0, 1])

    assert not bool_param.is_value_set_subset(int_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison
    assert not int_param.is_value_set_subset(bool_param)  # type: ignore[arg-type]  # test: cross-type bool/int comparison


def test_perm_param_is_value_set_subset_returns_false_against_non_perm() -> None:
    """Test the permutation value-set subset check rejects a non-permutation other.

    The defensive domain-kind guard returns ``False`` so direct callers
    (bypassing `is_subset`) see a deterministic answer.
    """
    perm = create_permutation_param([1, 2, 3])
    not_perm = create_ordinal_param([1, 2, 3])

    assert not perm.is_value_set_subset(not_perm)  # type: ignore[arg-type]  # test: cross-family comparison


# =============================================================================
# Discrete-set subset semantics (no Z3 path)
# =============================================================================


def test_categorical_param_subset_with_string_values_does_not_raise() -> None:
    """Test categorical `is_subset` works for non-numeric string categories."""
    smaller = create_categorical_param(["red", "blue"])
    larger = create_categorical_param(["red", "blue", "green"])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_subset_with_int_values() -> None:
    """Test categorical `is_subset` over int categories using set semantics."""
    smaller = create_categorical_param([1, 2])
    larger = create_categorical_param([1, 2, 3])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_categorical_param_subset_respects_in_set_constraint() -> None:
    """Test categorical `is_subset` honors an ``InSetConstraint`` narrowing."""
    universe = create_categorical_param([1, 2, 3])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {1, 2}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_ordinal_param_subset_with_int_values() -> None:
    """Test ordinal `is_subset` over integer values using set semantics."""
    smaller = create_ordinal_param([1, 2])
    larger = create_ordinal_param([1, 2, 3])

    assert smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_ordinal_param_subset_respects_in_set_constraint() -> None:
    """Test ordinal `is_subset` honors an ``InSetConstraint`` narrowing."""
    universe = create_ordinal_param([1, 2, 3])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {1, 2}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_perm_param_subset_distinguishes_different_member_sets() -> None:
    """Test permutation `is_subset` returns False when member sets differ.

    Different universes produce disjoint permutation sets, so neither
    direction holds.
    """
    smaller = create_permutation_param([1, 2])
    larger = create_permutation_param([1, 2, 3])

    assert not smaller.is_subset(larger)
    assert not larger.is_subset(smaller)


def test_perm_param_subset_respects_in_set_constraint() -> None:
    """Test permutation `is_subset` honors an ``InSetConstraint`` narrowing."""
    universe = create_permutation_param([1, 2])
    restricted = universe.add_constraint(InSetConstraint(universe.variable, {(1, 2)}))

    assert restricted.is_subset(universe)
    assert not universe.is_subset(restricted)


def test_discrete_params_reject_cross_family_subset_check() -> None:
    """Test `is_subset` returns False across distinct discrete-param families."""
    cat = create_categorical_param([1])
    ordinal = create_ordinal_param([1])
    perm = create_permutation_param([1])

    assert not cat.is_subset(ordinal)
    assert not cat.is_subset(perm)  # type: ignore[arg-type]  # test: cross-family comparison
    assert not ordinal.is_subset(cat)
    assert not ordinal.is_subset(perm)  # type: ignore[arg-type]  # test: cross-family comparison
    assert not perm.is_subset(cat)  # type: ignore[arg-type]  # test: cross-family comparison
    assert not perm.is_subset(ordinal)  # type: ignore[arg-type]  # test: cross-family comparison
