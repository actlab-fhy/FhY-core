"""Tests for `CategoricalParam`."""

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.expression import SymbolType
from fhy_core.identifier import Identifier
from fhy_core.param import (
    CategoricalParam,
    create_single_valid_value_param,
)
from fhy_core.serialization import (
    DeserializationValueError,
    serialize_registry_wrapped_value,
)

from .conftest import (
    SerializableEqualHashable,
    SerializableHashOnly,
    assert_all_satisfied,
    assert_none_satisfied,
)

# =============================================================================
# Construction & uniqueness
# =============================================================================


def test_categorical_param_initializes_from_set_of_values() -> None:
    """Test `CategoricalParam` initializes from a set of categorical values."""
    param = CategoricalParam({"a", "b", "c"})
    assert isinstance(param, CategoricalParam)


def test_categorical_param_init_rejects_duplicate_values() -> None:
    """Test `CategoricalParam` rejects duplicate values with `ParamError`."""
    with pytest.raises(ValueError):
        CategoricalParam([1, 1])


def test_categorical_param_init_rejects_value_without_equal_semantics() -> None:
    """Test `CategoricalParam` rejects wrapped-leaf values without equal semantics."""
    with pytest.raises(TypeError):
        CategoricalParam(  # type: ignore[type-var]  # test: invalid input
            {SerializableHashOnly(1), SerializableHashOnly(2)}
        )


# =============================================================================
# Properties
# =============================================================================


def test_categorical_param_categories_is_a_property() -> None:
    """Test `CategoricalParam.categories` is a property, not a method."""
    param = CategoricalParam({"a", "b"})
    assert not callable(param.categories)
    assert param.categories == frozenset({"a", "b"})


# =============================================================================
# Single-value helper
# =============================================================================


def test_create_single_valid_value_param_constrains_to_one_value() -> None:
    """Test `create_single_valid_value_param` constrains the parameter to one value."""
    param = create_single_valid_value_param("only")
    assert isinstance(param, CategoricalParam)
    assert param.get_possible_values() == {"only"}

    assignment = param.assign("only")
    assert assignment.is_value_set()
    assert assignment.value == "only"

    with pytest.raises(ValueError):
        param.assign("different")


# =============================================================================
# Admissibility & assignment
# =============================================================================


def test_categorical_param_assigns_values_in_the_category_set(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test `CategoricalParam.assign` accepts values in the category set."""
    assert categorical_param_abc.assign("a").is_value_set()
    assert categorical_param_abc.assign("c").is_value_set()


def test_categorical_param_assign_rejects_values_outside_the_category_set(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test `CategoricalParam.assign` rejects values outside the category set."""
    with pytest.raises(ValueError):
        categorical_param_abc.assign("d")


def test_categorical_param_admissibility_distinguishes_bool_from_int_categories() -> (
    None
):
    """Test `CategoricalParam` does not treat ``bool`` as interchangeable with `int`."""
    param = CategoricalParam([1, 2, 3])
    assert not param.is_value_admissible(True)


def test_categorical_param_get_symbol_type_is_real() -> None:
    """Test `CategoricalParam.get_symbol_type` returns ``SymbolType.REAL``."""
    assert CategoricalParam({"a", "b"}).get_symbol_type() == SymbolType.REAL


def test_categorical_param_str_lists_categories() -> None:
    """Test `str(CategoricalParam(...))` lists the categories inside ``{...}``."""
    text = str(CategoricalParam({"a", "b"}))
    assert "a" in text and "b" in text
    assert "{" in text and "}" in text


def test_categorical_param_admissibility_distinguishes_int_from_bool_categories() -> (
    None
):
    """Test a `CategoricalParam` of booleans rejects an integer probe.

    Pins down the ``bool``/``int`` mismatch check in both directions: a
    candidate that is plainly ``int`` must not be admitted by a ``bool``-only
    category set.
    """
    param = CategoricalParam([True])
    assert not param.is_value_admissible(1)


def test_categorical_param_admissibility_uses_equality_not_identity() -> None:
    """Test `CategoricalParam` admissibility uses ``==``, not ``is``.

    Constructs two equal but non-identical `Serializable` values and asserts a
    candidate is admitted by a category set that contains an equal-but-distinct
    object. Pins down value-equality matching against identity-only matching.
    """
    category = SerializableEqualHashable(42)
    candidate = SerializableEqualHashable(42)
    assert category is not candidate
    assert category == candidate
    param: CategoricalParam[SerializableEqualHashable] = CategoricalParam([category])
    assert param.is_value_admissible(candidate)


# =============================================================================
# Constraints
# =============================================================================


def test_categorical_param_add_constraint_combines_with_existing_membership(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test `CategoricalParam.add_constraint` further restricts the admissible set."""
    param = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )
    assert_all_satisfied(param, ["a", "b"])
    assert_none_satisfied(param, ["c"])


def test_categorical_param_rejects_non_set_constraint(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test `CategoricalParam.add_constraint` rejects equation constraints."""
    with pytest.raises(ValueError):
        categorical_param_abc.add_constraint(
            EquationConstraint(
                categorical_param_abc.variable,
                categorical_param_abc.variable_expression > 1,
            )
        )


# =============================================================================
# Structural equivalence
# =============================================================================


def test_categorical_param_is_structurally_equivalent_to_self() -> None:
    """Test `CategoricalParam.is_structurally_equivalent` is reflexive."""
    param = CategoricalParam({"a", "b"})
    assert param.is_structurally_equivalent(param)


def test_categorical_param_is_not_structurally_equivalent_to_subset_categories() -> (
    None
):
    """Test `CategoricalParam.is_structurally_equivalent` rejects a subset categories.

    A frozenset comparison must use ``==`` rather than ``<=`` or ``>=``: two
    `CategoricalParam`s where one's categories are a strict subset of the
    other's must compare non-equivalent.
    """
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    smaller: CategoricalParam[int] = CategoricalParam({1, 2}, name=shared_name)
    larger: CategoricalParam[int] = CategoricalParam({1, 2, 3}, name=shared_name_copy)
    assert not smaller.is_structurally_equivalent(larger)
    assert not larger.is_structurally_equivalent(smaller)


def test_categorical_param_is_not_structurally_equivalent_for_disjoint_categories() -> (
    None
):
    """Test `CategoricalParam.is_structurally_equivalent` rejects disjoint sets."""
    shared_name = Identifier("x")
    shared_name_copy = Identifier.deserialize_from_dict(shared_name.serialize_to_dict())
    left: CategoricalParam[int] = CategoricalParam({1, 2}, name=shared_name)
    right: CategoricalParam[int] = CategoricalParam({3, 4}, name=shared_name_copy)
    assert not left.is_structurally_equivalent(right)


def test_categorical_param_is_not_equivalent_to_non_categorical_object() -> None:
    """Test categorical equivalence is ``False`` for a non-`CategoricalParam` other."""
    param: CategoricalParam[str] = CategoricalParam({"a", "b"})
    assert not param.is_structurally_equivalent("not a param")
    assert not param.is_structurally_equivalent(object())


# =============================================================================
# Serialization
# =============================================================================


def test_categorical_param_serialization_round_trip_preserves_constraints(
    categorical_param_abc: CategoricalParam[str],
) -> None:
    """Test `CategoricalParam` round-trips with constraints through dict."""
    constrained = categorical_param_abc.add_constraint(
        InSetConstraint(categorical_param_abc.variable, {"a", "b"})
    )
    dictionary = constrained.serialize_to_dict()
    restored: CategoricalParam[str] = CategoricalParam.deserialize_from_dict(dictionary)
    assert_all_satisfied(restored, ["a", "b"])
    assert_none_satisfied(restored, ["c"])


def test_categorical_param_deserialize_rejects_wrapped_non_leaf_values() -> None:
    """Test categorical deserialize rejects wrapped container values."""
    payload = CategoricalParam({"a", "b"}).serialize_to_dict()
    payload["__data__"]["possible_values"] = [  # type: ignore[index]  # test: modify serialized
        serialize_registry_wrapped_value(("a", "b"))
    ]
    with pytest.raises(DeserializationValueError):
        CategoricalParam.deserialize_from_dict(payload)


def test_categorical_param_round_trips_with_serializable_value_type() -> None:
    """Test `CategoricalParam` round-trips when values are `Serializable` instances."""
    param: CategoricalParam[Identifier] = CategoricalParam(
        [Identifier("a"), Identifier("b")]
    )
    data = param.serialize_to_dict()
    restored: CategoricalParam[Identifier] = CategoricalParam.deserialize_from_dict(
        data
    )
    assert param.is_structurally_equivalent(restored)
