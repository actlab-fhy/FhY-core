"""Tests for ordinal parameters."""

from typing import Any

import pytest

from fhy_core.serialization import (
    DeserializationValueError,
    serialize_registry_wrapped_value,
)
from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.param import ParamError, create_ordinal_param
from fhy_core.symbolic.param.core import Param
from fhy_core.symbolic.param.domains import OrdinalDomain

from .conftest import (
    SerializableEqualNoOrder,
    assert_all_satisfied,
    assert_none_satisfied,
    mock_identifier,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_ordinal_param_initializes_from_sequence_of_values() -> None:
    """Test ordinal param initializes from a sequence of orderable values."""
    param = create_ordinal_param([5, 6, 7])

    assert isinstance(param, Param)
    assert isinstance(param.domain, OrdinalDomain)


def test_ordinal_param_init_rejects_duplicate_values() -> None:
    """Test ordinal param rejects duplicate values with `ParamError`."""
    with pytest.raises(ParamError):
        create_ordinal_param([1, 2, 1])


def test_ordinal_param_init_rejects_empty_values() -> None:
    """Test ordinal param rejects an empty value set with `ParamError`."""
    with pytest.raises(ParamError, match="non-empty"):
        create_ordinal_param([])


def test_ordinal_param_init_detects_duplicate_in_middle() -> None:
    """Test ordinal param rejects a value set with a duplicate.

    ``[1, 2, 2]`` holds a duplicate; the uniqueness check must reject it
    regardless of where the repeated value falls in the input sequence.
    """
    with pytest.raises(ParamError):
        create_ordinal_param([1, 2, 2])


def test_ordinal_param_init_detects_duplicate_in_two_value_sequence() -> None:
    """Test ordinal param rejects a two-value sequence whose values are equal.

    Exercises the uniqueness check on the smallest input that can hold a
    duplicate: a two-element sequence must still be rejected when both
    elements are equal.
    """
    with pytest.raises(ParamError):
        create_ordinal_param([1, 1])


def test_ordinal_param_init_detects_distinct_float_objects_as_duplicates() -> None:
    """Test ordinal param detects equal but non-identical float duplicates.

    Constructs the two duplicates via ``float("1.5")`` so they are distinct
    objects (CPython folds repeated ``1.5`` literals to a single object,
    which would mask an identity-based comparison). The uniqueness check
    must compare values with ``==``, not ``is``.
    """
    with pytest.raises(ParamError):
        create_ordinal_param([float("1.5"), float("1.5")])


def test_ordinal_param_rejects_negative_and_positive_zero_with_int_between() -> None:
    """Test ordinal param rejects -0.0 and 0.0 even with an int sorting between them.

    ``0`` sorts between ``-0.0`` and ``0.0`` by ``repr``, so a uniqueness
    check that only compares neighbours after sorting never places the two
    floats next to each other.
    """
    with pytest.raises(ParamError, match="unique"):
        create_ordinal_param([-0.0, 0, 0.0])


def test_ordinal_param_init_accepts_negative_zero_alongside_the_int_zero() -> None:
    """Test ordinal param accepts -0.0 and the int 0 as distinct kinds."""
    param = create_ordinal_param([-0.0, 0])

    assert isinstance(param, Param)


def test_ordinal_param_init_rejects_negative_and_positive_zero() -> None:
    """Test ordinal param rejects -0.0 and 0.0 as duplicate float members."""
    with pytest.raises(ParamError, match="unique"):
        create_ordinal_param([-0.0, 0.0])


def test_ordinal_param_init_rejects_value_without_orderable_semantics() -> None:
    """Test ordinal param rejects wrapped-leaf values without ordering semantics."""
    with pytest.raises(TypeError):
        create_ordinal_param(  # type: ignore[type-var]  # test: invalid input
            [SerializableEqualNoOrder(1), SerializableEqualNoOrder(2)]
        )


def test_ordinal_param_init_rejects_non_primitive_non_serializable_values() -> None:
    """Test ordinal param rejects values that are neither primitive nor `Serializable`.

    Plain tuples are orderable but neither primitive nor `Serializable`; they
    must be rejected during the per-value validation pass before the sort.
    """
    with pytest.raises(TypeError):
        create_ordinal_param([(1, 2), (3, 4)])  # type: ignore[type-var]  # test: invalid input


def test_ordinal_param_init_with_unsortable_mixture_raises_specific_message() -> None:
    """Test ordinal param re-raises `TypeError` with a sort-failure message."""
    with pytest.raises(TypeError, match="mutually comparable"):
        create_ordinal_param([1, "a"])  # type: ignore[type-var]  # test: invalid input


# =============================================================================
# Properties
# =============================================================================


def test_ordinal_param_possible_values_is_a_property() -> None:
    """Test the ordinal domain's ``sorted_values`` is a property, not a method."""
    param = create_ordinal_param([1, 2, 3])

    assert isinstance(param.domain, OrdinalDomain)
    assert not callable(param.domain.sorted_values)
    assert param.domain.sorted_values == (1, 2, 3)


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_ordinal_param_assigns_values_in_the_possible_set(
    ordinal_param_123: Param[int],
) -> None:
    """Test ordinal param assign accepts values in the possible-value set."""
    assert ordinal_param_123.assign(1).is_value_set()
    assert ordinal_param_123.assign(3).is_value_set()


def test_ordinal_param_assign_rejects_values_outside_the_possible_set(
    ordinal_param_123: Param[int],
) -> None:
    """Test ordinal param assign raises `ParamError` for values outside the set."""
    with pytest.raises(ParamError):
        ordinal_param_123.assign(4)


def test_ordinal_param_admissibility_distinguishes_bool_from_numeric_values() -> None:
    """Test ordinal param does not treat ``bool`` as interchangeable with `int`."""
    param = create_ordinal_param([1, 2, 3])

    assert not param.is_value_admissible(True)


def test_ordinal_param_does_not_define_get_symbol_type() -> None:
    """Test ordinal param's ``symbol_type`` is ``None`` (non-numeric domain)."""
    param = create_ordinal_param([1, 2, 3])

    assert param.symbol_type is None


def test_ordinal_param_str_lists_possible_values() -> None:
    """Test ``str`` of an ordinal param lists the possible values inside ``{...}``."""
    text = str(create_ordinal_param([1, 2, 3]))

    assert "1" in text and "2" in text and "3" in text
    assert "{" in text and "}" in text


# =============================================================================
# Constraints
# =============================================================================


def test_ordinal_param_add_constraint_combines_with_existing_membership(
    ordinal_param_123: Param[int],
) -> None:
    """Test ordinal param add_constraint further restricts the admissible set."""
    param = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )

    assert_all_satisfied(param, [1, 2])
    assert_none_satisfied(param, [3])


def test_ordinal_param_rejects_non_set_constraint(
    ordinal_param_123: Param[int],
) -> None:
    """Test ordinal param add_constraint raises for equation constraints."""
    with pytest.raises(ParamError):
        ordinal_param_123.add_constraint(
            EquationConstraint(ordinal_param_123.variable_expression > 1)
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_ordinal_param_is_structurally_equivalent_to_self() -> None:
    """Test ordinal param is_structurally_equivalent is reflexive."""
    param = create_ordinal_param([1, 2, 3])

    assert param.is_structurally_equivalent(param)


def test_ordinal_param_is_not_structurally_equivalent_when_possible_values_differ() -> (
    None
):
    """Test is_structurally_equivalent distinguishes possible-value sets.

    Two ordinal params sharing variable and constraints but holding different
    possible-value tuples must compare non-equivalent in *both* directions.
    Asserting both directions pins down ``==`` against ordered comparisons:
    a one-directional assertion would still pass under ``<=`` (when the
    self-tuple is the smaller one) or under ``>=`` (when it is the larger one).
    """
    left: Param[int] = create_ordinal_param([1, 2, 3], name=mock_identifier("x", 1))
    right: Param[int] = create_ordinal_param([1, 2, 4], name=mock_identifier("x", 1))

    assert not left.is_structurally_equivalent(right)
    assert not right.is_structurally_equivalent(left)


def test_ordinal_param_is_equivalent_to_a_separately_built_equal_param() -> None:
    """Test equivalence holds in both directions for two equal value sets.

    Guards the strict index-wise comparison against over-rejection: two params
    built separately over the same values must still compare equivalent, and the
    relation must stay reflexive and symmetric.
    """
    left: Param[int] = create_ordinal_param([1, 2], name=mock_identifier("x", 1))
    right: Param[int] = create_ordinal_param([1, 2], name=mock_identifier("x", 1))

    assert left.is_structurally_equivalent(left)
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_ordinal_param_bool_and_int_value_sets_are_not_equivalent() -> None:
    """Test structural equivalence keeps ``bool`` and ``int`` value sets distinct.

    ``{1, 2}`` and ``{True, 2}`` admit different values, so they must compare
    non-equivalent in both directions. Native ``tuple`` equality would report them
    equal because ``True == 1``.
    """
    int_param: Param[int] = create_ordinal_param([1, 2], name=mock_identifier("x", 1))
    bool_param: Param[int] = create_ordinal_param(
        [True, 2], name=mock_identifier("x", 1)
    )

    assert not int_param.is_structurally_equivalent(bool_param)
    assert not bool_param.is_structurally_equivalent(int_param)


def test_ordinal_param_int_and_float_value_sets_are_not_equivalent() -> None:
    """Test structural equivalence keeps ``int`` and ``float`` value sets distinct.

    ``1`` and ``1.0`` are distinct ordinal members, so ``{1, 2}`` and ``{1.0, 2}``
    must compare non-equivalent in both directions despite ``1 == 1.0``.
    """
    int_param: Param[int] = create_ordinal_param([1, 2], name=mock_identifier("x", 1))
    float_param: Param[float] = create_ordinal_param(
        [1.0, 2], name=mock_identifier("x", 1)
    )

    assert not int_param.is_structurally_equivalent(float_param)
    assert not float_param.is_structurally_equivalent(int_param)


def test_ordinal_param_mixed_kind_value_set_is_equivalent_in_either_order() -> None:
    """Test a mixed ``int``/``bool`` value set compares equivalent either way round.

    ``1`` and ``True`` cannot be separated by the ascending sort, so equivalence
    holds only because both construction orders canonicalize identically.
    """
    forward: Param[int] = create_ordinal_param([1, True], name=mock_identifier("x", 1))
    reverse: Param[int] = create_ordinal_param([True, 1], name=mock_identifier("x", 1))

    assert forward.is_structurally_equivalent(reverse)
    assert reverse.is_structurally_equivalent(forward)


def test_ordinal_params_reported_equivalent_agree_on_value_validity() -> None:
    """Test equivalent ordinal params accept exactly the same values.

    The downstream contract of structural equivalence: params that report
    equivalent must not disagree about any value. A value-blind equivalence check
    breaks it, because ``{1, 2}`` and ``{True, 2}`` would compare equivalent while
    only one of them accepts ``True``.
    """
    params: list[Param[Any]] = [
        create_ordinal_param(values, name=mock_identifier("x", 1))
        for values in ([1, 2], [True, 2], [1.0, 2])
    ]
    candidate_values: tuple[Any, ...] = (True, 1, 1.0, 2)

    for left in params:
        for right in params:
            if not left.is_structurally_equivalent(right):
                continue
            for value in candidate_values:
                assert left.is_value_valid(value) == right.is_value_valid(value)


def test_ordinal_param_is_not_structurally_equivalent_to_non_ordinal_object() -> None:
    """Test is_structurally_equivalent is ``False`` for a non-``Param`` object."""
    param = create_ordinal_param([1, 2, 3])

    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_ordinal_param_serialization_round_trip_preserves_constraints(
    ordinal_param_123: Param[int],
) -> None:
    """Test ordinal param round-trips through dict serialization with constraints."""
    constrained = ordinal_param_123.add_constraint(
        InSetConstraint(ordinal_param_123.variable, {1, 2})
    )

    dictionary = constrained.serialize_to_dict()
    restored: Param[int] = Param.deserialize_from_dict(dictionary)

    assert_all_satisfied(restored, [1, 2])
    assert_none_satisfied(restored, [3])


def test_ordinal_param_deserialize_rejects_unwrapped_possible_values() -> None:
    """Test ordinal param deserialize_from_dict rejects raw (unwrapped) values.

    Under the derived format the value list lives at
    ``payload["domain"]["__data__"]["sorted_values"]`` as wrapped registry
    dicts; replacing them with bare integers must be rejected by the wrapped-leaf
    sequence codec.
    """
    payload = create_ordinal_param([1, 2, 3]).serialize_to_dict()
    payload["domain"]["__data__"]["sorted_values"] = [1, 2, 3]  # type: ignore[index,call-overload]  # test: modify serialized

    with pytest.raises(DeserializationValueError):
        Param.deserialize_from_dict(payload)


def test_ordinal_param_deserialize_rejects_a_nan_value() -> None:
    """Test a payload carrying a NaN value is refused, as construction refuses it."""
    payload = create_ordinal_param([1.0, 2.0]).serialize_to_dict()
    payload["domain"]["__data__"]["sorted_values"] = [  # type: ignore[index,call-overload]  # test: modify serialized
        serialize_registry_wrapped_value(float("nan")),
        serialize_registry_wrapped_value(1.0),
    ]

    with pytest.raises(DeserializationValueError, match="NaN"):
        Param.deserialize_from_dict(payload)
