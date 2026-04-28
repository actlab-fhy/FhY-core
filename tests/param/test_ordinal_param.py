"""Tests for `OrdinalParam`."""

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import SymbolType
from fhy_core.identifier import Identifier
from fhy_core.param import OrdinalParam
from fhy_core.serialization import DeserializationValueError

from .conftest import (
    SerializableEqualNoOrder,
    assert_all_satisfied,
    assert_none_satisfied,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_ordinal_param_initializes_from_sequence_of_values() -> None:
    """Test `OrdinalParam` initializes from a sequence of orderable values."""
    param = OrdinalParam([5, 6, 7])
    assert isinstance(param, OrdinalParam)


def test_ordinal_param_init_rejects_duplicate_values() -> None:
    """Test `OrdinalParam` rejects duplicate values with `ParamError`."""
    with pytest.raises(ValueError):
        OrdinalParam([1, 2, 1])


def test_ordinal_param_init_detects_duplicate_in_middle() -> None:
    """Test `OrdinalParam` detects a duplicate that lands adjacent after sorting.

    Pins down the adjacency-pair walk in the uniqueness check: a sequence that
    sorts to ``(1, 2, 2)`` must be rejected, killing both off-by-one and
    range-shrinking mutations on the inner ``range(len(values) - 1)`` walk.
    """
    with pytest.raises(ValueError):
        OrdinalParam([1, 2, 2])


def test_ordinal_param_init_detects_duplicate_in_two_value_sequence() -> None:
    """Test `OrdinalParam` rejects a two-value sequence whose values are equal.

    Pins down the uniqueness check on the smallest non-trivial input, killing
    mutations that shrink the comparison range to zero iterations (such as
    ``range(len(values) - 2)`` when ``len(values) == 2``).
    """
    with pytest.raises(ValueError):
        OrdinalParam([1, 1])


def test_ordinal_param_init_detects_distinct_float_objects_as_duplicates() -> None:
    """Test `OrdinalParam` detects equal but non-identical float duplicates.

    Constructs the two duplicates via ``float("1.5")`` so they are distinct
    objects (CPython folds repeated ``1.5`` literals to a single object,
    which would mask an identity-based comparison). The uniqueness check
    must use ``==``, not ``is``, on adjacent sorted values.
    """
    with pytest.raises(ValueError):
        OrdinalParam([float("1.5"), float("1.5")])


def test_ordinal_param_init_rejects_value_without_orderable_semantics() -> None:
    """Test `OrdinalParam` rejects wrapped-leaf values without ordering semantics."""
    with pytest.raises(TypeError):
        OrdinalParam(  # type: ignore[type-var]  # test: invalid input
            [SerializableEqualNoOrder(1), SerializableEqualNoOrder(2)]
        )


def test_ordinal_param_init_rejects_non_primitive_non_serializable_values() -> None:
    """Test `OrdinalParam` rejects values that are neither primitive nor `Serializable`.

    Plain tuples are orderable but neither primitive nor `Serializable`; they
    must be rejected during the per-value validation pass before the sort.
    """
    with pytest.raises(TypeError):
        OrdinalParam([(1, 2), (3, 4)])  # type: ignore[type-var]  # test: invalid input


def test_ordinal_param_init_with_unsortable_mixture_raises_specific_message() -> None:
    """Test `OrdinalParam` re-raises `TypeError` with a sort-failure message."""
    with pytest.raises(TypeError, match="mutually comparable"):
        OrdinalParam([1, "a"])  # type: ignore[type-var]  # test: invalid input


# =============================================================================
# Properties
# =============================================================================


def test_ordinal_param_possible_values_is_a_property() -> None:
    """Test `OrdinalParam.possible_values` is a property, not a method."""
    param = OrdinalParam([1, 2, 3])
    assert not callable(param.possible_values)
    assert param.possible_values == (1, 2, 3)


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_ordinal_param_assigns_values_in_the_possible_set(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test `OrdinalParam.assign` accepts values in the possible-value set."""
    assert ordinal_param_123.assign(1).is_value_set()
    assert ordinal_param_123.assign(3).is_value_set()


def test_ordinal_param_assign_rejects_values_outside_the_possible_set(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test `OrdinalParam.assign` rejects values outside the possible-value set."""
    with pytest.raises(ValueError):
        ordinal_param_123.assign(4)


def test_ordinal_param_admissibility_distinguishes_bool_from_numeric_values() -> None:
    """Test `OrdinalParam` does not treat ``bool`` as interchangeable with `int`."""
    param = OrdinalParam([1, 2, 3])
    assert not param.is_value_admissible(True)


def test_ordinal_param_get_symbol_type_is_real() -> None:
    """Test `OrdinalParam.get_symbol_type` returns ``SymbolType.REAL``."""
    assert OrdinalParam([1, 2, 3]).get_symbol_type() == SymbolType.REAL


def test_ordinal_param_str_lists_possible_values() -> None:
    """Test `str(OrdinalParam(...))` lists the possible values inside ``{...}``."""
    text = str(OrdinalParam([1, 2, 3]))
    assert "1" in text and "2" in text and "3" in text
    assert "{" in text and "}" in text


# =============================================================================
# Constraints
# =============================================================================


def test_ordinal_param_add_constraint_combines_with_existing_membership(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test `OrdinalParam.add_constraint` further restricts the admissible set."""
    param = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )
    assert_all_satisfied(param, [1, 2])
    assert_none_satisfied(param, [3])


def test_ordinal_param_rejects_non_set_constraint(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test `OrdinalParam.add_constraint` rejects equation constraints."""
    with pytest.raises(ValueError):
        ordinal_param_123.add_constraint(
            EquationConstraint(
                ordinal_param_123.variable,
                ordinal_param_123.variable_expression > 1,
            )
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_ordinal_param_is_structurally_equivalent_to_self() -> None:
    """Test `OrdinalParam.is_structurally_equivalent` is reflexive."""
    param = OrdinalParam([1, 2, 3])
    assert param.is_structurally_equivalent(param)


def test_ordinal_param_is_not_structurally_equivalent_when_possible_values_differ() -> (
    None
):
    """Test `OrdinalParam.is_structurally_equivalent` distinguishes possible-value sets.

    Two `OrdinalParam`s sharing variable and constraints but holding different
    possible-value tuples must compare non-equivalent in *both* directions.
    Asserting both directions pins down ``==`` against ordered comparisons:
    a one-directional assertion would still pass under ``<=`` (when the
    self-tuple is the smaller one) or under ``>=`` (when it is the larger one).
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    left: OrdinalParam[int] = OrdinalParam([1, 2, 3], name=shared_name)
    right: OrdinalParam[int] = OrdinalParam([1, 2, 4], name=shared_name_copy)
    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


def test_ordinal_param_is_not_structurally_equivalent_to_non_ordinal_object() -> None:
    """Test `OrdinalParam.is_structurally_equivalent` is ``False`` off-class."""
    param = OrdinalParam([1, 2, 3])
    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_ordinal_param_serialization_round_trip_preserves_constraints(
    ordinal_param_123: OrdinalParam[int],
) -> None:
    """Test `OrdinalParam` round-trips through dict serialization with constraints."""
    constrained = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )
    dictionary = constrained.serialize_to_dict()
    restored: OrdinalParam[int] = OrdinalParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, [1, 2])
    assert_none_satisfied(restored, [3])


def test_ordinal_param_deserialize_rejects_unwrapped_possible_values() -> None:
    """Test `OrdinalParam.deserialize_from_dict` rejects raw possible values."""
    payload = OrdinalParam([1, 2, 3]).serialize_to_dict()
    payload["__data__"]["possible_values"] = [1, 2, 3]  # type: ignore[index]  # test: modify serialized
    with pytest.raises(DeserializationValueError):
        OrdinalParam.deserialize_from_dict(payload)
