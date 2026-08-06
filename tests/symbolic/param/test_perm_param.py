"""Tests for permutation parameters."""

import pytest

from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.param import ParamError, create_permutation_param
from fhy_core.symbolic.param.core import Param
from fhy_core.symbolic.param.domains import PermutationDomain

from .conftest import (
    SerializableEqualHashable,
    assert_all_satisfied,
    assert_none_satisfied,
    mock_identifier,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_perm_param_initializes_from_sequence_of_members() -> None:
    """Test permutation param initializes from a sequence of permutation members."""
    param = create_permutation_param(["n", "c", "h", "w"])

    assert isinstance(param, Param)
    assert isinstance(param.domain, PermutationDomain)


def test_perm_param_init_rejects_duplicate_members() -> None:
    """Test permutation param rejects duplicate members with `ParamError`."""
    with pytest.raises(ParamError):
        create_permutation_param(["n", "c", "h", "n"])


def test_perm_param_init_rejects_empty_members() -> None:
    """Test permutation param rejects an empty member sequence with `ParamError`."""
    with pytest.raises(ParamError, match="non-empty"):
        create_permutation_param([])


def test_perm_param_init_detects_adjacent_duplicate_members() -> None:
    """Test permutation param detects an adjacent duplicate with no later occurrences.

    Pins down the inner-loop bound of the unsorted uniqueness walk: a
    duplicate at indices ``0`` and ``1`` must be detected even though only
    a single later index needs to be inspected.
    """
    with pytest.raises(ParamError):
        create_permutation_param([1, 1, 2])


def test_perm_param_init_uses_equality_not_identity_to_detect_duplicates() -> None:
    """Test permutation param detects duplicates by ``==``, not identity.

    Two equal but non-identical `Serializable` values must be rejected as
    duplicates.
    """
    first = SerializableEqualHashable(1)
    second = SerializableEqualHashable(1)
    assert first is not second
    assert first == second

    with pytest.raises(ParamError):
        create_permutation_param([first, second])  # type: ignore[type-var]  # test: bespoke `Serializable` value


def test_perm_param_init_rejects_non_primitive_non_serializable_members() -> None:
    """Test permutation param rejects members that are not primitive or `Serializable`.

    Plain tuples are neither primitive nor `Serializable`; the per-member
    validation pass must reject them rather than letting them through.
    """
    with pytest.raises(TypeError):
        create_permutation_param([(1, 2), (3, 4)])  # type: ignore[type-var]  # test: invalid input


# =============================================================================
# Properties
# =============================================================================


def test_perm_param_members_is_a_property() -> None:
    """Test the permutation domain's ``ordered_members`` is a property, not a method."""
    param = create_permutation_param(["n", "c", "h", "w"])

    assert isinstance(param.domain, PermutationDomain)
    assert not callable(param.domain.ordered_members)
    assert param.domain.ordered_members == ("n", "c", "h", "w")


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_perm_param_assigns_a_permutation_of_its_members(
    perm_param_nchw: Param[tuple[str, ...]],
) -> None:
    """Test permutation param assign accepts a permutation of its members."""
    assignment = perm_param_nchw.assign(["c", "n", "w", "h"])  # type: ignore[arg-type]  # test: list as permutation value

    assert assignment.is_value_set()


def test_perm_param_assign_rejects_non_permutation_value(
    perm_param_nchw: Param[tuple[str, ...]],
) -> None:
    """Test permutation param assign raises `ParamError` for a non-permutation value."""
    with pytest.raises(ParamError):
        perm_param_nchw.assign(["n", "c", "h", "n"])  # type: ignore[arg-type]  # test: list as permutation value


def test_perm_param_admissibility_rejects_string_value() -> None:
    """Test permutation param is_value_admissible rejects a plain string value."""
    param = create_permutation_param(["n", "c", "h", "w"])

    assert not param.is_value_admissible("nchw")


def test_perm_param_admissibility_rejects_value_outside_member_set() -> None:
    """Test permutation param is_value_admissible rejects values outside the member set.

    A candidate sequence containing a value that is a valid permutation-member
    type but not in the param's allowed members must be rejected. Pins down
    the per-element membership check in ``_is_valid_permutation``.
    """
    param = create_permutation_param([1, 2, 3])

    assert not param.is_value_admissible((1, 2, 5))


def test_perm_param_does_not_define_get_symbol_type() -> None:
    """Test permutation param's ``symbol_type`` is ``None`` (non-numeric domain)."""
    param = create_permutation_param([1, 2, 3])

    assert param.symbol_type is None


def test_perm_param_str_lists_members() -> None:
    """Test ``str`` of a permutation param lists the members inside ``{...}``."""
    text = str(create_permutation_param([1, 2, 3]))

    assert "1" in text and "2" in text and "3" in text
    assert "{" in text and "}" in text


# =============================================================================
# Constraints
# =============================================================================


def test_perm_param_add_constraint_combines_with_existing_membership(
    perm_param_nchw: Param[tuple[str, ...]],
) -> None:
    """Test permutation param add_constraint further restricts the admissible set."""
    param = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )

    assert_all_satisfied(param, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    assert_none_satisfied(param, [["n", "c", "w", "h"]])


def test_perm_param_rejects_non_set_constraint(
    perm_param_nchw: Param[tuple[str, ...]],
) -> None:
    """Test permutation ``add_constraint`` raises `ParamError` for equation constraints.

    Equation constraints are not valid for permutation params.
    """
    with pytest.raises(ParamError):
        perm_param_nchw.add_constraint(
            EquationConstraint(
                perm_param_nchw.variable, perm_param_nchw.variable_expression > 1
            )
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_perm_param_is_structurally_equivalent_to_self() -> None:
    """Test permutation param is_structurally_equivalent is reflexive."""
    param = create_permutation_param(["n", "c", "h", "w"])

    assert param.is_structurally_equivalent(param)


def test_perm_param_is_not_structurally_equivalent_when_member_orders_differ() -> None:
    """Test is_structurally_equivalent distinguishes member orderings.

    Permutation member tuples are stored positionally, so reversing them must
    produce a non-equivalent param in *both* directions. Asserting both
    directions pins down ``==`` against ordered comparisons: a one-directional
    assertion would still pass under ``<=`` (when the self-tuple is the
    smaller one) or under ``>=`` (when it is the larger one).
    """
    left: Param[tuple[int, ...]] = create_permutation_param(
        [1, 2, 3], name=mock_identifier("x", 1)
    )
    right: Param[tuple[int, ...]] = create_permutation_param(
        [3, 2, 1], name=mock_identifier("x", 1)
    )

    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


def test_perm_param_is_not_structurally_equivalent_to_non_perm_object() -> None:
    """Test is_structurally_equivalent is ``False`` for a non-``Param`` object."""
    param = create_permutation_param(["n", "c", "h", "w"])

    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_perm_param_serialization_round_trip_preserves_constraints(
    perm_param_nchw: Param[tuple[str, ...]],
) -> None:
    """Test permutation param round-trips through dict serialization."""
    constrained = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )

    dictionary = constrained.serialize_to_dict()
    restored: Param[tuple[str, ...]] = Param.deserialize_from_dict(dictionary)

    assert_all_satisfied(restored, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    assert_none_satisfied(restored, [["n", "c", "w", "h"]])
