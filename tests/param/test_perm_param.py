"""Tests for `PermParam`."""

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import SymbolType
from fhy_core.identifier import Identifier
from fhy_core.param import PermParam

from .conftest import (
    SerializableEqualHashable,
    assert_all_satisfied,
    assert_none_satisfied,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_perm_param_initializes_from_sequence_of_members() -> None:
    """Test `PermParam` initializes from a sequence of permutation members."""
    param = PermParam(["n", "c", "h", "w"])
    assert isinstance(param, PermParam)


def test_perm_param_init_rejects_duplicate_members() -> None:
    """Test `PermParam` rejects duplicate members with `ParamError`."""
    with pytest.raises(ValueError):
        PermParam(["n", "c", "h", "n"])


def test_perm_param_init_detects_adjacent_duplicate_members() -> None:
    """Test `PermParam` detects an adjacent duplicate with no later occurrences.

    Pins down the inner-loop bound of the unsorted uniqueness walk: a
    duplicate at indices ``0`` and ``1`` must be detected even though only
    a single later index needs to be inspected.
    """
    with pytest.raises(ValueError):
        PermParam([1, 1, 2])


def test_perm_param_init_uses_equality_not_identity_to_detect_duplicates() -> None:
    """Test `PermParam` detects duplicates by ``==``, not identity.

    Two equal but non-identical `Serializable` values must be rejected as
    duplicates.
    """
    first = SerializableEqualHashable(1)
    second = SerializableEqualHashable(1)
    assert first is not second
    assert first == second
    with pytest.raises(ValueError):
        PermParam([first, second])


def test_perm_param_init_rejects_non_primitive_non_serializable_members() -> None:
    """Test `PermParam` rejects members that are neither primitive nor `Serializable`.

    Plain tuples are neither primitive nor `Serializable`; the per-member
    validation pass must reject them rather than letting them through.
    """
    with pytest.raises(TypeError):
        PermParam([(1, 2), (3, 4)])  # type: ignore[type-var]  # test: invalid input


# =============================================================================
# Properties
# =============================================================================


def test_perm_param_members_is_a_property() -> None:
    """Test `PermParam.members` is a property, not a method."""
    param = PermParam(["n", "c", "h", "w"])
    assert not callable(param.members)
    assert param.members == ("n", "c", "h", "w")


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_perm_param_assigns_a_permutation_of_its_members(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test `PermParam.assign` accepts a permutation of its members."""
    assignment = perm_param_nchw.assign(["c", "n", "w", "h"])
    assert assignment.is_value_set()


def test_perm_param_assign_rejects_non_permutation_value(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test `PermParam.assign` rejects a value that is not a permutation."""
    with pytest.raises(ValueError):
        perm_param_nchw.assign(["n", "c", "h", "n"])


def test_perm_param_admissibility_rejects_string_value() -> None:
    """Test `PermParam.is_value_admissible` rejects a plain string value."""
    param = PermParam(["n", "c", "h", "w"])
    assert not param.is_value_admissible("nchw")


def test_perm_param_admissibility_rejects_value_outside_member_set() -> None:
    """Test `PermParam.is_value_admissible` rejects values outside the member set.

    A candidate sequence containing a value that is a valid permutation-member
    type but not in the param's allowed members must be rejected. Pins down
    the per-element ``and _contains_param_value`` clause in the membership
    walk.
    """
    param = PermParam([1, 2, 3])
    assert not param.is_value_admissible((1, 2, 5))


def test_perm_param_get_symbol_type_is_real() -> None:
    """Test `PermParam.get_symbol_type` returns ``SymbolType.REAL``."""
    assert PermParam([1, 2, 3]).get_symbol_type() == SymbolType.REAL


def test_perm_param_str_lists_members() -> None:
    """Test `str(PermParam(...))` lists the members inside ``{...}``."""
    text = str(PermParam([1, 2, 3]))
    assert "1" in text and "2" in text and "3" in text
    assert "{" in text and "}" in text


# =============================================================================
# Constraints
# =============================================================================


def test_perm_param_add_constraint_combines_with_existing_membership(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test `PermParam.add_constraint` further restricts the admissible set."""
    param = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )
    assert_all_satisfied(param, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    assert_none_satisfied(param, [["n", "c", "w", "h"]])


def test_perm_param_rejects_non_set_constraint(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test `PermParam.add_constraint` rejects equation constraints."""
    with pytest.raises(ValueError):
        perm_param_nchw.add_constraint(
            EquationConstraint(
                perm_param_nchw.variable, perm_param_nchw.variable_expression > 1
            )
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_perm_param_is_structurally_equivalent_to_self() -> None:
    """Test `PermParam.is_structurally_equivalent` is reflexive."""
    param = PermParam(["n", "c", "h", "w"])
    assert param.is_structurally_equivalent(param)


def test_perm_param_is_not_structurally_equivalent_when_member_orders_differ() -> None:
    """Test `PermParam.is_structurally_equivalent` distinguishes member orderings.

    Permutation member tuples are stored positionally, so reversing them must
    produce a non-equivalent param in *both* directions. Asserting both
    directions pins down ``==`` against ordered comparisons: a one-directional
    assertion would still pass under ``<=`` (when the self-tuple is the
    smaller one) or under ``>=`` (when it is the larger one).
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    left: PermParam[int] = PermParam([1, 2, 3], name=shared_name)
    right: PermParam[int] = PermParam([3, 2, 1], name=shared_name_copy)
    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


def test_perm_param_is_not_structurally_equivalent_to_non_perm_object() -> None:
    """Test `PermParam.is_structurally_equivalent` is ``False`` for non-`PermParam`."""
    param = PermParam(["n", "c", "h", "w"])
    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_perm_param_serialization_round_trip_preserves_constraints(
    perm_param_nchw: PermParam[str],
) -> None:
    """Test `PermParam` round-trips through dict serialization with its constraints."""
    constrained = perm_param_nchw.add_constraint(
        InSetConstraint(
            perm_param_nchw.variable, {("n", "c", "h", "w"), ("c", "n", "w", "h")}
        )
    )
    dictionary = constrained.serialize_to_dict()
    restored: PermParam[str] = PermParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [["n", "c", "h", "w"], ["c", "n", "w", "h"]])
    assert_none_satisfied(restored, [["n", "c", "w", "h"]])
