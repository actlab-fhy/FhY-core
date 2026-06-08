"""Tests for `fhy_core.op_attribute`.

Covers the public surface of the open `OpAttribute` registry:

- Construction with the documented trait stack and immutability.
- Interning semantics keyed on the `Identifier` instance (id-equality).
- The four canonical algebraic/semantic attributes and their registration.
- Structural equivalence ignoring `description`.
- Hashability and frozenset use.
- Dict and binary round-trips, including canonical-lookup behavior during
  deserialization.
"""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.op_attribute import (
    ASSOCIATIVE,
    COMMUTATIVE,
    ELEMENTWISE,
    PURE,
    OpAttribute,
)
from fhy_core.serialization import (
    Serializable,
    SerializedDict,
)
from fhy_core.traits import (
    Frozen,
    FrozenMutationError,
    HasIdentifier,
    Interned,
    StructuralEquivalence,
)

# =============================================================================
# Construction & traits
# =============================================================================


def test_op_attribute_constructs_with_name_and_description() -> None:
    """Test `OpAttribute` stores its `name` and `description`."""
    name = Identifier("x")
    attribute = OpAttribute(name, "an attribute")
    assert attribute.name is name
    assert attribute.description == "an attribute"


def test_op_attribute_satisfies_documented_protocols() -> None:
    """Test instances satisfy `HasIdentifier`, `Frozen`, `Interned`, and
    `StructuralEquivalence` runtime protocols."""
    attribute = OpAttribute(Identifier("x"), "desc")
    assert isinstance(attribute, HasIdentifier)
    assert isinstance(attribute, Frozen)
    assert isinstance(attribute, Interned)
    assert isinstance(attribute, StructuralEquivalence)


def test_op_attribute_get_identifier_returns_name() -> None:
    """Test `get_identifier` returns the same `Identifier` passed at construction."""
    name = Identifier("x")
    attribute = OpAttribute(name, "desc")
    assert attribute.get_identifier() is name


def test_op_attribute_get_intern_key_returns_name() -> None:
    """Test `get_intern_key` returns the same `Identifier` passed at construction."""
    name = Identifier("x")
    attribute = OpAttribute(name, "desc")
    assert attribute.get_intern_key() is name


def test_op_attribute_is_frozen_after_construction() -> None:
    """Test the dataclass is frozen and reports `is_frozen` True."""
    attribute = OpAttribute(Identifier("x"), "desc")
    assert attribute.is_frozen


def test_op_attribute_blocks_attribute_mutation() -> None:
    """Test attribute assignment on a frozen `OpAttribute` raises."""
    attribute = OpAttribute(Identifier("x"), "desc")
    with pytest.raises((FrozenMutationError, AttributeError)):
        attribute.description = "rewritten"  # type: ignore[misc]


# =============================================================================
# Interning
# =============================================================================


def test_op_attribute_first_constructed_with_key_is_canonical() -> None:
    """Test `get_interned` returns the first instance registered under a name."""
    name = Identifier("x")
    first = OpAttribute(name, "first")
    second = OpAttribute(name, "second")
    canonical = OpAttribute.get_interned(name)
    assert canonical is first
    assert canonical is not second


def test_op_attribute_distinct_identifiers_intern_separately() -> None:
    """Test two `Identifier`s with the same `name_hint` are distinct intern keys."""
    name_a = Identifier("dup")
    name_b = Identifier("dup")
    attribute_a = OpAttribute(name_a, "a")
    attribute_b = OpAttribute(name_b, "b")
    assert OpAttribute.get_interned(name_a) is attribute_a
    assert OpAttribute.get_interned(name_b) is attribute_b
    assert attribute_a is not attribute_b


# =============================================================================
# Canonical instances
# =============================================================================


def test_canonical_commutative_is_registered() -> None:
    """Test `COMMUTATIVE` is the canonical entry for its name."""
    assert OpAttribute.get_interned(COMMUTATIVE.name) is COMMUTATIVE


def test_canonical_associative_is_registered() -> None:
    """Test `ASSOCIATIVE` is the canonical entry for its name."""
    assert OpAttribute.get_interned(ASSOCIATIVE.name) is ASSOCIATIVE


def test_canonical_pure_is_registered() -> None:
    """Test `PURE` is the canonical entry for its name."""
    assert OpAttribute.get_interned(PURE.name) is PURE


def test_canonical_elementwise_is_registered() -> None:
    """Test `ELEMENTWISE` is the canonical entry for its name."""
    assert OpAttribute.get_interned(ELEMENTWISE.name) is ELEMENTWISE


def test_canonical_attributes_are_distinct() -> None:
    """Test all four canonical attributes are distinct entries."""
    canonicals = [COMMUTATIVE, ASSOCIATIVE, PURE, ELEMENTWISE]
    for i, left in enumerate(canonicals):
        for right in canonicals[i + 1 :]:
            assert left is not right
            assert left.name != right.name


def test_canonical_attributes_carry_descriptions() -> None:
    """Test all four canonical attributes carry non-empty descriptions."""
    for canonical in (COMMUTATIVE, ASSOCIATIVE, PURE, ELEMENTWISE):
        assert isinstance(canonical.description, str)
        assert canonical.description.strip() != ""


# =============================================================================
# Structural equivalence
# =============================================================================


def test_op_attribute_structurally_equivalent_when_names_match() -> None:
    """Test two instances with the same `name` are equivalent regardless of
    `description`."""
    name = Identifier("x")
    left = OpAttribute(name, "first description")
    right = OpAttribute(name, "second description")
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


# =============================================================================
# Hashability & frozenset use
# =============================================================================


def test_op_attribute_is_hashable() -> None:
    """Test `OpAttribute` instances are hashable."""
    attribute = OpAttribute(Identifier("x"), "desc")
    assert hash(attribute) == hash(attribute)


def test_op_attribute_equality_ignores_description() -> None:
    """Test `__eq__` compares on `name` only and ignores `description`."""
    name = Identifier("x")
    left = OpAttribute(name, "first description")
    right = OpAttribute(name, "second description")
    assert left == right


def test_op_attribute_hash_ignores_description() -> None:
    """Test `__hash__` is keyed on `name` and ignores `description`."""
    name = Identifier("x")
    left = OpAttribute(name, "first description")
    right = OpAttribute(name, "second description")
    assert hash(left) == hash(right)


def test_op_attribute_unequal_when_names_differ() -> None:
    """Test instances with different `name`s compare unequal."""
    assert OpAttribute(Identifier("a"), "desc") != OpAttribute(Identifier("b"), "desc")


def test_op_attribute_works_in_frozenset() -> None:
    """Test canonical attributes can be assembled into a frozenset."""
    tags = frozenset({COMMUTATIVE, PURE})
    assert COMMUTATIVE in tags
    assert PURE in tags
    assert ASSOCIATIVE not in tags
    assert len(tags) == 2


def test_op_attribute_canonical_appears_once_in_frozenset() -> None:
    """Test the same canonical instance collapses to a single frozenset entry."""
    tags = frozenset({COMMUTATIVE, COMMUTATIVE, PURE})
    assert tags == frozenset({COMMUTATIVE, PURE})


# =============================================================================
# Serialization
# =============================================================================


def test_op_attribute_deserialize_returns_canonical_for_registered_name() -> None:
    """Test deserialization returns the canonical interned instance when one
    is registered for the deserialized name."""
    data = PURE.serialize_to_dict()
    restored = OpAttribute.deserialize_from_dict(data)
    assert restored is PURE


def test_op_attribute_deserialize_constructs_fresh_for_unregistered_name() -> None:
    """Test deserialization constructs a fresh instance for an unseen identifier."""
    unregistered_name = Identifier("never-registered-op-attribute")
    assert OpAttribute.get_interned(unregistered_name) is None
    data: SerializedDict = {
        "name": unregistered_name.serialize_to_dict(),
        "description": "fresh from deserialize",
    }
    restored = OpAttribute.deserialize_from_dict(data)
    assert isinstance(restored, OpAttribute)
    assert restored.description == "fresh from deserialize"
    assert OpAttribute.get_interned(restored.name) is restored


def test_op_attribute_canonical_binary_round_trip_returns_canonical() -> None:
    """Test the binary envelope of a canonical attribute deserializes back to the
    canonical interned instance via the registry lookup."""
    blob = COMMUTATIVE.to_bytes()
    restored = Serializable.from_bytes(blob)
    assert restored is COMMUTATIVE


# =============================================================================
# Description-mismatch deserialization warning
# =============================================================================


def test_op_attribute_deserialize_warns_on_description_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test deserializing a canonical name with a different description warns."""
    canonical = OpAttribute(Identifier("mismatch-warning-attr"), "original description")
    payload: SerializedDict = {
        "name": canonical.name.serialize_to_dict(),
        "description": "divergent description",
    }
    with caplog.at_level("WARNING", logger="fhy_core.traits.interned"):
        restored = OpAttribute.deserialize_from_dict(payload)

    assert restored is canonical
    assert restored.description == "original description"
    assert any(
        "already canonical" in record.getMessage()
        and "divergent description" in record.getMessage()
        for record in caplog.records
    )


def test_op_attribute_deserialize_does_not_warn_when_descriptions_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test deserializing a canonical name with matching description does not warn."""
    canonical = OpAttribute(
        Identifier("matching-description-attr"), "matching description"
    )
    payload = canonical.serialize_to_dict()
    with caplog.at_level("WARNING", logger="fhy_core.traits.interned"):
        restored = OpAttribute.deserialize_from_dict(payload)

    assert restored is canonical
    assert not any(
        "already canonical" in record.getMessage() for record in caplog.records
    )


# =============================================================================
# register_default_instances restores module-level canonicals
# =============================================================================


def test_register_default_instances_restores_module_level_op_attributes() -> None:
    """Test ``register_default_instances`` re-canonicalizes shipped defaults."""
    try:
        OpAttribute.clear_interned_registry()
        OpAttribute.register_default_instances()

        for canonical in (COMMUTATIVE, ASSOCIATIVE, PURE, ELEMENTWISE):
            restored = OpAttribute.deserialize_from_dict(canonical.serialize_to_dict())
            assert restored is canonical
    finally:
        OpAttribute.register_default_instances()
