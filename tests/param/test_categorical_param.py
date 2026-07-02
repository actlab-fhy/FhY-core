"""Tests for categorical parameters."""

import pytest

from fhy_core.constraint import EquationConstraint, InSetConstraint
from fhy_core.identifier import Identifier
from fhy_core.param import (
    ParamError,
    create_categorical_param,
    create_single_valid_value_param,
)
from fhy_core.param.core import Param
from fhy_core.param.domains import CategoricalDomain
from fhy_core.serialization import (
    DeserializationValueError,
    serialize_registry_wrapped_value,
)

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
    assert param.domain.categories == frozenset({"a", "b"})


# =============================================================================
# Single-value helper
# =============================================================================


def test_create_single_valid_value_param_builds_one_category_param() -> None:
    """Test `create_single_valid_value_param` builds a one-category param."""
    param = create_single_valid_value_param("only")

    assert isinstance(param, Param)
    assert isinstance(param.domain, CategoricalDomain)
    assert param.domain.categories == frozenset({"only"})


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
            EquationConstraint(
                categorical_param_abc.variable,
                categorical_param_abc.variable_expression > 1,
            )
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

    A frozenset comparison must use ``==`` rather than ``<=`` or ``>=``: two
    categorical params where one's categories are a strict subset of the
    other's must compare non-equivalent.
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


def test_categorical_param_deserialize_rejects_wrapped_non_leaf_values() -> None:
    """Test categorical deserialize rejects wrapped container values.

    Under the derived format the value list lives at
    ``payload["domain"]["__data__"]["categories"]``; a wrapped tuple is not a
    valid categorical leaf value and must be rejected.
    """
    payload = create_categorical_param({"a", "b"}).serialize_to_dict()
    payload["domain"]["__data__"]["categories"] = [  # type: ignore[index,call-overload]  # test: modify serialized
        serialize_registry_wrapped_value(("a", "b"))
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
