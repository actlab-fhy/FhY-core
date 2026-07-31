"""Tests for `create_union_param` and `Param.__or__`.

Union is supported only for the finite-set kinds that can bake both operands'
effective value sets into a fresh member set: ``CategoricalDomain`` and
``OrdinalDomain``. Every other kind (permutation, all numeric kinds, and any
cross-kind pair) raises ``TypeError`` because a union is a disjunction, which
the conjunction-of-constraints model cannot represent.
"""

from typing import Any

import pytest
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import InSetConstraint, NotInSetConstraint
from fhy_core.symbolic.param import (
    Param,
    ParamError,
    create_categorical_param,
    create_integer_param,
    create_integer_param_between,
    create_intersection_param,
    create_interval_integer_param_between,
    create_ordinal_param,
    create_permutation_param,
    create_real_param,
    create_union_param,
)
from fhy_core.symbolic.param.domains import CategoricalDomain, OrdinalDomain

from .conftest import (
    assert_all_valid,
    assert_none_valid,
    assert_param_round_trips_in_all_formats,
    mock_identifier,
)

# =============================================================================
# Categorical union
# =============================================================================


def test_categorical_union_merges_both_category_sets() -> None:
    """Test categorical union merges both operands' category sets."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    result = create_union_param(left, right)

    assert isinstance(result.domain, CategoricalDomain)
    assert_all_valid(result, ["a", "b", "c", "d"])


def test_categorical_union_bakes_not_in_set_constraint_before_merging() -> None:
    """Test a `NotInSetConstraint` on the left operand is folded into the merge.

    ``{"a","b","c"}`` narrowed by ``NotInSet{"c"}`` has effective set
    ``{"a","b"}``; unioned with ``{"c","d"}`` (unnarrowed, so "c" survives on
    the right), the result is ``{"a","b","c","d"}``.
    """
    left = create_categorical_param({"a", "b", "c"})
    left = left.add_constraint(NotInSetConstraint(left.variable, {"c"}))
    right = create_categorical_param({"c", "d"})

    result = create_union_param(left, right)

    assert_all_valid(result, ["a", "b", "c", "d"])


def test_categorical_union_excludes_value_narrowed_out_of_both_operands() -> None:
    """Test a value excluded from both operands' effective sets is absent."""
    left = create_categorical_param({"a", "b"})
    left = left.add_constraint(NotInSetConstraint(left.variable, {"b"}))
    right = create_categorical_param({"b", "c"})
    right = right.add_constraint(NotInSetConstraint(right.variable, {"b"}))

    result = create_union_param(left, right)

    assert_all_valid(result, ["a", "c"])
    assert_none_valid(result, ["b"])


def test_categorical_union_preserves_strict_bool_int_distinction() -> None:
    """Test union keeps ``True`` and ``1`` as distinct categorical members."""
    left: Param[Any] = create_categorical_param({True})
    right = create_categorical_param({1})

    result = create_union_param(left, right)

    assert_all_valid(result, [True, 1])
    assert result.domain.is_value_admissible(True)
    assert result.domain.is_value_admissible(1)
    assert len(result.domain.categories) == 2  # type: ignore[attr-defined]


def test_categorical_union_result_carries_no_constraints() -> None:
    """Test the union result carries no constraints (baked, not conjoined)."""
    left = create_categorical_param({"a", "b"})
    left = left.add_constraint(InSetConstraint(left.variable, {"a"}))
    right = create_categorical_param({"c"})

    result = create_union_param(left, right)

    assert result.constraints == ()


def test_categorical_union_of_two_constraint_emptied_operands_raises_param_error() -> (
    None
):
    """Test a union of two operands whose effective value sets are both empty raises.

    ``left`` is narrowed by ``NotInSet{"a"}`` to the empty effective set;
    ``right`` is narrowed by ``NotInSet{"b"}`` likewise. Neither operand
    contributes any value, so the merged set is empty -- this must raise a
    clear "union is empty" error, not the internal
    ``build_categorical_domain`` "Categories must be non-empty" message.
    """
    left = create_categorical_param({"a"})
    left = left.add_constraint(NotInSetConstraint(left.variable, {"a"}))
    right = create_categorical_param({"b"})
    right = right.add_constraint(NotInSetConstraint(right.variable, {"b"}))

    with pytest.raises(ParamError, match="is empty"):
        create_union_param(left, right)


# =============================================================================
# Ordinal union
# =============================================================================


def test_ordinal_union_merges_and_resorts_values() -> None:
    """Test ordinal union merges both operands' values and re-sorts them."""
    left = create_ordinal_param([1, 2])
    right = create_ordinal_param([2, 3])

    result = create_union_param(left, right)

    assert isinstance(result.domain, OrdinalDomain)
    assert_all_valid(result, [1, 2, 3])
    assert result.domain.sorted_values == (1, 2, 3)


def test_ordinal_union_of_incomparable_values_raises_type_error() -> None:
    """Test an ordinal union whose merged values are not mutually comparable raises."""
    left: Param[Any] = create_ordinal_param([1, 2])
    right = create_ordinal_param(["a"])

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_ordinal_union_of_two_constraint_emptied_operands_raises_param_error() -> None:
    """Test a union of two constraint-emptied ordinal operands raises ``ParamError``.

    Mirrors the categorical case: both operands' effective value sets are
    narrowed to empty by a ``NotInSetConstraint``, so the merged set is empty.
    """
    left = create_ordinal_param([1])
    left = left.add_constraint(NotInSetConstraint(left.variable, {1}))
    right = create_ordinal_param([2])
    right = right.add_constraint(NotInSetConstraint(right.variable, {2}))

    with pytest.raises(ParamError, match="is empty"):
        create_union_param(left, right)


# =============================================================================
# Unsupported domain kinds
# =============================================================================


def test_union_of_permutation_params_raises_type_error() -> None:
    """Test union of two permutation parameters raises ``TypeError``."""
    left = create_permutation_param(["a", "b"])
    right = create_permutation_param(["a", "b"])

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_of_integer_params_raises_type_error() -> None:
    """Test union of two plain integer parameters raises ``TypeError``."""
    left = create_integer_param()
    right = create_integer_param()

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_of_real_params_raises_type_error() -> None:
    """Test union of two real parameters raises ``TypeError``."""
    left = create_real_param()
    right = create_real_param()

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_of_interval_integer_params_raises_type_error() -> None:
    """Test union of two interval-integer parameters raises ``TypeError``.

    Interval union as a convex hull is deliberately excluded (silent
    over-approximation); interval-integer params are the ones users most
    likely try to combine arithmetically, so this case is worth pinning
    down explicitly rather than relying only on the plain-integer case.
    """
    left = create_interval_integer_param_between(0, 5)
    right = create_interval_integer_param_between(3, 8)

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_of_mixed_interval_integer_and_plain_integer_raises_type_error() -> None:
    """Test union across an interval-integer and a plain-integer param raises."""
    left = create_interval_integer_param_between(0, 5)
    right = create_integer_param_between(3, 8)

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_across_categorical_and_ordinal_kinds_raises_type_error() -> None:
    """Test union across categorical and ordinal domains raises ``TypeError``."""
    left: Param[Any] = create_categorical_param({"a", "b"})
    right = create_ordinal_param([1, 2])

    with pytest.raises(TypeError):
        create_union_param(left, right)


def test_union_across_ordinal_and_categorical_kinds_raises_type_error() -> None:
    """Test union across ordinal/categorical domains raises ``TypeError`` (reversed)."""
    left: Param[Any] = create_ordinal_param([1, 2])
    right = create_categorical_param({"a", "b"})

    with pytest.raises(TypeError):
        create_union_param(left, right)


# =============================================================================
# Result variable
# =============================================================================


def test_union_result_uses_fresh_default_variable() -> None:
    """Test the union result uses a fresh ``Identifier`` when ``name`` is omitted."""
    left = create_categorical_param({"a"}, name=mock_identifier("x", 1))
    right = create_categorical_param({"b"}, name=mock_identifier("y", 2))

    result = create_union_param(left, right)

    assert result.variable is not left.variable
    assert result.variable is not right.variable
    assert isinstance(result.variable, Identifier)


def test_union_result_uses_given_name() -> None:
    """Test the union result uses the ``name`` argument when supplied."""
    left = create_categorical_param({"a"})
    right = create_categorical_param({"b"})
    given = mock_identifier("z", 3)

    result = create_union_param(left, right, name=given)

    assert result.variable is given


# =============================================================================
# `__or__` dunder delegation
# =============================================================================


def test_or_dunder_delegates_to_create_union_param() -> None:
    """Test ``|`` produces the same union as calling `create_union_param` directly."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c"})

    result = left | right

    assert_all_valid(result, ["a", "b", "c"])


def test_or_dunder_with_non_param_operand_raises_type_error() -> None:
    """Test ``|`` raises ``TypeError`` (via ``NotImplemented``) for a non-`Param`."""
    left = create_categorical_param({"a"})

    with pytest.raises(TypeError):
        _ = left | "not a param"


# =============================================================================
# Integration: serialization, subset/feasibility/assign interop
# =============================================================================


def test_union_result_round_trips_through_serialization() -> None:
    """Test a union result round-trips through DICT, JSON, and BINARY formats."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"b", "c"})

    result = create_union_param(left, right)

    assert_param_round_trips_in_all_formats(result)


def test_union_result_interoperates_with_is_subset() -> None:
    """Test a union result's feasible set is a superset of each operand's."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"b", "c"})

    result = create_union_param(left, right)

    assert left.is_subset(result)
    assert right.is_subset(result)
    assert not result.is_subset(left)


def test_union_result_interoperates_with_is_feasible() -> None:
    """Test a non-empty union result reports feasible."""
    left = create_categorical_param({"a"})
    right = create_categorical_param({"b"})

    result = create_union_param(left, right)

    assert result.is_feasible()
    assert not result.is_empty()


def test_union_result_interoperates_with_assign() -> None:
    """Test a value valid for either operand can be assigned to the union result."""
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    result = create_union_param(left, right)
    assignment = result.assign("c")

    assert assignment.value == "c"
    with pytest.raises(ParamError):
        result.assign("e")


# =============================================================================
# Set-membership law (union)
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        ("a", True),
        ("b", True),
        ("c", True),
        ("d", True),
        ("e", False),
    ],
)
def test_union_membership_law_holds_for_categorical_operands(
    value: str, expected: bool
) -> None:
    """Test a value is valid for the union iff valid for either operand.

    Left is ``{"a","b"}``, right is ``{"c","d"}``.
    """
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"c", "d"})

    result = create_union_param(left, right)

    assert result.is_value_valid(value) == expected


def test_union_is_not_intersection() -> None:
    """Test union and intersection disagree on a value present in only one operand.

    Sanity check that the two operations are not accidentally swapped.
    """
    left = create_categorical_param({"a", "b"})
    right = create_categorical_param({"b", "c"})

    union = create_union_param(left, right)
    intersection = create_intersection_param(left, right)

    assert union.is_value_valid("a")
    assert not intersection.is_value_valid("a")


# =============================================================================
# Property: finite-set membership law
# =============================================================================


@pytest.mark.property
@given(  # type: ignore[untyped-decorator]
    left_values=st.sets(st.integers(min_value=0, max_value=12), min_size=1, max_size=6),
    right_values=st.sets(
        st.integers(min_value=0, max_value=12), min_size=1, max_size=6
    ),
    candidate=st.integers(min_value=0, max_value=15),
)
def test_union_membership_law_holds_for_random_ordinal_sets(
    left_values: set[int], right_values: set[int], candidate: int
) -> None:
    """Test a value is valid for the union iff valid for either operand.

    Holds for arbitrary (non-empty) ordinal value sets, not just the fixed
    examples above.
    """
    left = create_ordinal_param(sorted(left_values))
    right = create_ordinal_param(sorted(right_values))

    result = create_union_param(left, right)

    expected = candidate in left_values or candidate in right_values
    assert result.is_value_valid(candidate) == expected
