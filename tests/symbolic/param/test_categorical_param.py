"""Tests for categorical parameters."""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationValueError,
    serialize_registry_wrapped_value,
)
from fhy_core.symbolic.constraint import EquationConstraint, InSetConstraint
from fhy_core.symbolic.param import (
    OrdinalDomain,
    ParamError,
    PermutationDomain,
    create_categorical_param,
    create_single_valid_value_param,
)
from fhy_core.symbolic.param.core import Param
from fhy_core.symbolic.param.domains import CategoricalDomain

from .conftest import (
    SerializableEqualHashable,
    SerializableHashOnly,
    assert_all_satisfied,
    assert_none_satisfied,
    mock_identifier,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_categorical_param_initializes_from_set_of_values() -> None:
    """Test categorical param initializes from a set of categorical values."""
    param = create_categorical_param({"a", "b", "c"})

    assert isinstance(param, Param)
    assert isinstance(param.domain, CategoricalDomain)


def test_categorical_param_init_rejects_duplicate_values() -> None:
    """Test categorical param rejects duplicate values with `ParamError`."""
    with pytest.raises(ParamError):
        create_categorical_param([1, 1])


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param(set(), id="empty-set"),
        pytest.param(frozenset(), id="empty-frozenset"),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
    ],
)
def test_categorical_param_init_rejects_empty_categories(empty: object) -> None:
    """Test categorical param rejects an empty collection with `ParamError`."""
    with pytest.raises(ParamError, match="non-empty"):
        create_categorical_param(empty)  # type: ignore[arg-type]  # test: invalid input


def test_categorical_param_init_rejects_value_without_equal_semantics() -> None:
    """Test categorical param rejects wrapped-leaf values without equal semantics."""
    with pytest.raises(TypeError):
        create_categorical_param(  # type: ignore[type-var]  # test: invalid input
            {SerializableHashOnly(1), SerializableHashOnly(2)}
        )


# =============================================================================
# Properties
# =============================================================================


def test_categorical_param_categories_is_a_property() -> None:
    """Test the categorical domain's ``categories`` is a property, not a method."""
    param = create_categorical_param({"a", "b"})

    assert isinstance(param.domain, CategoricalDomain)
    assert not callable(param.domain.categories)
    assert set(param.domain.categories) == {"a", "b"}


# =============================================================================
# Single-value helper
# =============================================================================


def test_create_single_valid_value_param_builds_one_category_param() -> None:
    """Test `create_single_valid_value_param` builds a one-category param."""
    param = create_single_valid_value_param("only")

    assert isinstance(param, Param)
    assert isinstance(param.domain, CategoricalDomain)
    assert param.domain.categories == ("only",)


def test_create_single_valid_value_param_assigns_the_single_value() -> None:
    """Test `create_single_valid_value_param` assigns its single admissible value."""
    param = create_single_valid_value_param("only")

    assignment = param.assign("only")

    assert assignment.is_value_set()
    assert assignment.value == "only"


def test_create_single_valid_value_param_rejects_any_other_value() -> None:
    """Test `create_single_valid_value_param` rejects a value other than its own."""
    param = create_single_valid_value_param("only")

    with pytest.raises(ParamError):
        param.assign("different")


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_categorical_param_assigns_values_in_the_category_set(
    categorical_param_abc: Param[str],
) -> None:
    """Test categorical param assign accepts values in the category set."""
    assert categorical_param_abc.assign("a").is_value_set()
    assert categorical_param_abc.assign("c").is_value_set()


def test_categorical_param_assign_rejects_values_outside_the_category_set(
    categorical_param_abc: Param[str],
) -> None:
    """Test categorical param assign raises `ParamError` for values outside the set."""
    with pytest.raises(ParamError):
        categorical_param_abc.assign("d")


def test_categorical_param_admissibility_distinguishes_bool_from_int_categories() -> (
    None
):
    """Test categorical param does not treat ``bool`` as interchangeable with `int`."""
    param = create_categorical_param([1, 2, 3])

    assert not param.is_value_admissible(True)


def test_categorical_param_does_not_define_get_symbol_type() -> None:
    """Test categorical param's ``symbol_type`` is ``None`` (non-numeric domain)."""
    param = create_categorical_param({"a", "b"})

    assert param.symbol_type is None


def test_categorical_param_str_lists_categories() -> None:
    """Test ``str`` of a categorical param lists the categories inside ``{...}``."""
    text = str(create_categorical_param({"a", "b"}))

    assert "a" in text and "b" in text
    assert "{" in text and "}" in text


def test_categorical_param_admissibility_distinguishes_int_from_bool_categories() -> (
    None
):
    """Test a categorical param of booleans rejects an integer probe.

    Pins down the ``bool``/``int`` mismatch check in both directions: a
    candidate that is plainly ``int`` must not be admitted by a ``bool``-only
    category set.
    """
    param = create_categorical_param([True])

    assert not param.is_value_admissible(1)


def test_categorical_param_admissibility_uses_equality_not_identity() -> None:
    """Test categorical param admissibility uses ``==``, not ``is``.

    Constructs two equal but non-identical `Serializable` values and asserts a
    candidate is admitted by a category set that contains an equal-but-distinct
    object. Pins down value-equality matching against identity-only matching.
    """
    category = SerializableEqualHashable(42)
    candidate = SerializableEqualHashable(42)
    assert category is not candidate
    assert category == candidate
    param: Param[SerializableEqualHashable] = create_categorical_param([category])  # type: ignore[type-var]  # test: bespoke `Serializable` value

    assert param.is_value_admissible(candidate)


# =============================================================================
# Constraints
# =============================================================================


def test_categorical_param_add_constraint_combines_with_existing_membership(
    categorical_param_abc: Param[str],
) -> None:
    """Test categorical param add_constraint further restricts the admissible set."""
    param = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )

    assert_all_satisfied(param, ["a", "b"])
    assert_none_satisfied(param, ["c"])


def test_categorical_param_rejects_non_set_constraint(
    categorical_param_abc: Param[str],
) -> None:
    """Test categorical param add_constraint raises for equation constraints."""
    with pytest.raises(ParamError):
        categorical_param_abc.add_constraint(
            EquationConstraint(categorical_param_abc.variable_expression > 1)
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_categorical_param_is_structurally_equivalent_to_self() -> None:
    """Test categorical param is_structurally_equivalent is reflexive."""
    param = create_categorical_param({"a", "b"})

    assert param.is_structurally_equivalent(param)


def test_categorical_param_is_not_structurally_equivalent_to_subset_categories() -> (
    None
):
    """Test is_structurally_equivalent rejects a subset categories.

    Equivalence must require equal-size, mutually matching category sets rather
    than one-directional containment: two categorical params where one's
    categories are a strict subset of the other's must compare non-equivalent.
    """
    smaller: Param[int] = create_categorical_param({1, 2}, name=mock_identifier("x", 1))
    larger: Param[int] = create_categorical_param(
        {1, 2, 3}, name=mock_identifier("x", 1)
    )

    assert not smaller.is_structurally_equivalent(larger)
    assert not larger.is_structurally_equivalent(smaller)


def test_categorical_param_is_not_structurally_equivalent_for_disjoint_categories() -> (
    None
):
    """Test is_structurally_equivalent rejects disjoint sets."""
    left: Param[int] = create_categorical_param({1, 2}, name=mock_identifier("x", 1))
    right: Param[int] = create_categorical_param({3, 4}, name=mock_identifier("x", 1))

    assert not left.is_structurally_equivalent(right)


def test_categorical_param_is_not_equivalent_to_non_categorical_object() -> None:
    """Test categorical equivalence is ``False`` for a non-``Param`` object."""
    param: Param[str] = create_categorical_param({"a", "b"})

    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_categorical_param_serialization_round_trip_preserves_constraints(
    categorical_param_abc: Param[str],
) -> None:
    """Test categorical param round-trips with constraints through dict."""
    constrained = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )

    dictionary = constrained.serialize_to_dict()
    restored: Param[str] = Param.deserialize_from_dict(dictionary)

    assert_all_satisfied(restored, ["a", "b"])
    assert_none_satisfied(restored, ["c"])


@pytest.mark.usefixtures("v1_wire")
def test_categorical_param_deserialize_rejects_wrapped_non_leaf_values() -> None:
    """Test categorical deserialize rejects wrapped container values.

    Under the derived format the value list lives at
    ``payload["domain"]["__data__"]["categories"]``; a wrapped tuple holding a
    float is not a valid category (a tuple of leaf values is, but a float is
    refused inside one too) and must be rejected.
    """
    payload = create_categorical_param({"a", "b"}).serialize_to_dict()
    payload["domain"]["__data__"]["categories"] = [  # type: ignore[index,call-overload]  # test: modify serialized
        serialize_registry_wrapped_value(("a", 1.5))
    ]

    with pytest.raises(DeserializationValueError):
        Param.deserialize_from_dict(payload)


def test_categorical_param_round_trips_with_serializable_value_type() -> None:
    """Test categorical param round-trips when values are `Serializable` instances."""
    param: Param[Identifier] = create_categorical_param(
        [Identifier("a"), Identifier("b")]
    )

    data = param.serialize_to_dict()
    restored: Param[Identifier] = Param.deserialize_from_dict(data)

    assert param.is_structurally_equivalent(restored)


def test_categorical_param_keeps_bool_and_int_categories_distinct() -> None:
    """Test a categorical param keeps ``True`` and ``1`` as two distinct categories.

    Native ``frozenset`` storage would collapse ``True`` and ``1`` into one
    element because ``True == 1``; the tuple storage keeps both and admits each.
    """
    param: Param[int] = create_categorical_param([True, 1])

    assert isinstance(param.domain, CategoricalDomain)
    assert len(param.domain.categories) == 2
    assert param.is_value_admissible(True)
    assert param.is_value_admissible(1)


def test_categorical_param_bool_and_int_category_sets_are_not_equivalent() -> None:
    """Test structural equivalence keeps ``bool`` and ``int`` category sets distinct.

    ``(True,)`` and ``(1,)`` must not compare equivalent even though ``True == 1``,
    so the equivalence check cannot fall back on native ``tuple`` equality.
    """
    bool_param: Param[bool] = create_categorical_param(
        [True], name=mock_identifier("x", 1)
    )
    int_param: Param[int] = create_categorical_param([1], name=mock_identifier("x", 1))

    assert not bool_param.is_structurally_equivalent(int_param)
    assert not int_param.is_structurally_equivalent(bool_param)


def test_categorical_param_bool_and_int_categories_round_trip_distinctly() -> None:
    """Test a mixed ``bool``/``int`` categorical param round-trips both kinds.

    Serialization must not collapse ``True`` and ``1`` during the round trip, so
    the restored param still admits both distinct categories.
    """
    param: Param[int] = create_categorical_param([True, 1])

    restored: Param[int] = Param.deserialize_from_dict(param.serialize_to_dict())

    assert isinstance(restored.domain, CategoricalDomain)
    assert len(restored.domain.categories) == 2
    assert restored.is_value_admissible(True)
    assert restored.is_value_admissible(1)


# =============================================================================
# Tuple and frozen-set categories
# =============================================================================


def _tile_shapes() -> Any:
    """Return the tile shapes `(4, 4)` and `(8, 8)`, untyped as categories."""
    return ((8, 8), (4, 4))


def test_categorical_domain_accepts_tuple_categories() -> None:
    """Test a domain over tile shapes builds and keeps them as tuples."""
    domain = CategoricalDomain(_tile_shapes())

    assert set(domain.categories) == {(4, 4), (8, 8)}
    assert all(isinstance(category, tuple) for category in domain.categories)


def test_tuple_categories_come_back_in_a_canonical_order() -> None:
    """Test the categories' order does not depend on the order they were given in."""
    ascending: Any = ((4, 4), (8, 8))
    descending: Any = ((8, 8), (4, 4))

    forward = CategoricalDomain(ascending).categories
    backward = CategoricalDomain(descending).categories

    assert forward == backward == ((4, 4), (8, 8))


def test_tuple_categories_may_mix_leaf_kinds_and_nest() -> None:
    """Test a tuple of a string and an int, and a nested tuple, are categories."""
    categories: Any = (("x", 4), (1, (2, 3)), (True, "y"))

    domain = CategoricalDomain(categories)

    assert set(domain.categories) == set(categories)
    assert domain.is_value_admissible(("x", 4))
    assert domain.is_value_admissible((1, (2, 3)))


def test_a_frozen_set_is_a_category() -> None:
    """Test a frozen set of leaf values is a category."""
    categories: Any = (frozenset({1, 2}), frozenset({3}))

    domain = CategoricalDomain(categories)

    assert set(domain.categories) == {frozenset({1, 2}), frozenset({3})}
    assert domain.is_value_admissible(frozenset({3}))


def test_a_categorical_param_over_tuples_admits_exactly_them() -> None:
    """Test a param admits each tile shape and nothing else."""
    param = create_categorical_param(_tile_shapes())

    assert param.is_value_admissible((8, 8))
    assert param.is_value_admissible((4, 4))
    assert not param.is_value_admissible((4, 8))
    assert not param.is_value_admissible((8,))
    assert not param.is_value_admissible(8)


def test_a_categorical_param_over_tuples_assigns_a_tuple() -> None:
    """Test assigning `(8, 8)` keeps the tuple, assigning `(5, 5)` is refused."""
    param = create_categorical_param(_tile_shapes())

    assignment = param.assign((8, 8))

    assert assignment.value == (8, 8)
    with pytest.raises(ParamError):
        param.assign((5, 5))


def test_tuple_categories_that_repeat_are_refused() -> None:
    """Test two equal tuples are one category twice: `ParamError`."""
    repeated: Any = ((4, 4), (4, 4))

    with pytest.raises(ParamError, match="unique"):
        CategoricalDomain(repeated)


def test_a_tuple_category_round_trips_through_its_payload() -> None:
    """Test a param over tuples serializes and reads back admitting the same."""
    param = create_categorical_param(_tile_shapes())

    restored: Param[Any] = Param.deserialize_from_dict(param.serialize_to_dict())

    assert isinstance(restored.domain, CategoricalDomain)
    assert restored.domain.categories == ((4, 4), (8, 8))
    assert restored.is_value_admissible((8, 8))
    assert not restored.is_value_admissible((4, 8))


@pytest.mark.parametrize(
    "category",
    [
        pytest.param((4, 4.5), id="float_in_a_tuple"),
        pytest.param((4, (4, 4.5)), id="float_nested"),
        pytest.param(4.5, id="float"),
        pytest.param(frozenset({4.5}), id="float_in_a_frozen_set"),
    ],
)
def test_a_float_category_is_refused_inside_a_tuple_too(category: Any) -> None:
    """Test a float, bare or inside a composite category, is a `TypeError`."""
    categories: Any = ((4, 4), category)

    with pytest.raises(TypeError):
        CategoricalDomain(categories)


def test_ordinal_and_permutation_domains_still_refuse_tuples() -> None:
    """Test composite values are for categorical domains only."""
    values: Any = ((4, 4), (8, 8))

    with pytest.raises(TypeError):
        OrdinalDomain(values)
    with pytest.raises(TypeError):
        PermutationDomain(values)
