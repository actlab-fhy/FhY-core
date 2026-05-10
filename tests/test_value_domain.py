"""Tests for `fhy_core.value_domain`.

Covers the public surface of the open `ValueDomain` registry:

- Construction with the documented trait stack and immutability.
- Interning semantics keyed on the `Identifier` instance (id-equality).
- The canonical `DATA_DOMAIN` and `ADDRESS_DOMAIN` registrations.
- Structural equivalence ignoring `description` and walking the `parent` chain.
- `is_subdomain_of` over the parent chain.
- Dict, JSON, and binary round-trips, including canonical-lookup behavior
  during deserialization.
"""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    Serializable,
    SerializationFormat,
    SerializedDict,
)
from fhy_core.trait import (
    Frozen,
    FrozenMutationError,
    HasIdentifier,
    Interned,
    StructuralEquivalence,
)
from fhy_core.value_domain import (
    ADDRESS_DOMAIN,
    DATA_DOMAIN,
    ValueDomain,
)

# =============================================================================
# Construction & traits
# =============================================================================


def test_value_domain_constructs_with_name_and_description() -> None:
    """Test `ValueDomain` stores its `name`, `description`, and a `None` parent."""
    name = Identifier("x")
    domain = ValueDomain(name, "a domain")
    assert domain.name is name
    assert domain.description == "a domain"
    assert domain.parent is None


def test_value_domain_satisfies_documented_protocols() -> None:
    """Test instances satisfy `HasIdentifier`, `Frozen`, `Interned`, and
    `StructuralEquivalence` runtime protocols."""
    domain = ValueDomain(Identifier("x"), "desc")
    assert isinstance(domain, HasIdentifier)
    assert isinstance(domain, Frozen)
    assert isinstance(domain, Interned)
    assert isinstance(domain, StructuralEquivalence)


def test_value_domain_get_identifier_returns_name() -> None:
    """Test `get_identifier` returns the same `Identifier` passed at construction."""
    name = Identifier("x")
    domain = ValueDomain(name, "desc")
    assert domain.get_identifier() is name


def test_value_domain_get_intern_key_returns_name() -> None:
    """Test `get_intern_key` returns the same `Identifier` passed at construction."""
    name = Identifier("x")
    domain = ValueDomain(name, "desc")
    assert domain.get_intern_key() is name


def test_value_domain_is_frozen_after_construction() -> None:
    """Test the dataclass is frozen and reports `is_frozen` True."""
    domain = ValueDomain(Identifier("x"), "desc")
    assert domain.is_frozen


def test_value_domain_blocks_attribute_mutation() -> None:
    """Test attribute assignment on a frozen `ValueDomain` raises."""
    domain = ValueDomain(Identifier("x"), "desc")
    with pytest.raises((FrozenMutationError, AttributeError)):
        domain.description = "rewritten"  # type: ignore[misc]


# =============================================================================
# Interning
# =============================================================================


def test_value_domain_first_constructed_with_key_is_canonical() -> None:
    """Test `get_interned` returns the first instance registered under a name."""
    name = Identifier("x")
    first = ValueDomain(name, "first")
    second = ValueDomain(name, "second")
    canonical = ValueDomain.get_interned(name)
    assert canonical is first
    assert canonical is not second


def test_value_domain_distinct_identifiers_intern_separately() -> None:
    """Test two `Identifier`s with the same `name_hint` are distinct intern keys."""
    name_a = Identifier("dup")
    name_b = Identifier("dup")
    domain_a = ValueDomain(name_a, "a")
    domain_b = ValueDomain(name_b, "b")
    assert ValueDomain.get_interned(name_a) is domain_a
    assert ValueDomain.get_interned(name_b) is domain_b
    assert domain_a is not domain_b


# =============================================================================
# Canonical instances
# =============================================================================


def test_canonical_data_domain_is_registered() -> None:
    """Test `DATA_DOMAIN` is the canonical entry for its name."""
    assert ValueDomain.get_interned(DATA_DOMAIN.name) is DATA_DOMAIN


def test_canonical_address_domain_is_registered() -> None:
    """Test `ADDRESS_DOMAIN` is the canonical entry for its name."""
    assert ValueDomain.get_interned(ADDRESS_DOMAIN.name) is ADDRESS_DOMAIN


def test_canonical_domains_are_distinct() -> None:
    """Test the canonical data and address domains are distinct entries."""
    assert DATA_DOMAIN is not ADDRESS_DOMAIN
    assert DATA_DOMAIN.name != ADDRESS_DOMAIN.name


def test_canonical_data_domain_has_no_parent() -> None:
    """Test `DATA_DOMAIN` is registered at the top of its chain."""
    assert DATA_DOMAIN.parent is None


def test_canonical_address_domain_has_no_parent() -> None:
    """Test `ADDRESS_DOMAIN` is registered at the top of its chain."""
    assert ADDRESS_DOMAIN.parent is None


# =============================================================================
# Structural equivalence
# =============================================================================


def test_value_domain_structurally_equivalent_when_name_and_parent_match() -> None:
    """Test two instances with equivalent `name` and `parent` are equivalent
    regardless of `description`."""
    name = Identifier("x")
    left = ValueDomain(name, "first description")
    right = ValueDomain(name, "second description")
    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_value_domain_not_equivalent_when_names_differ() -> None:
    """Test instances with different `name`s are not structurally equivalent."""
    left = ValueDomain(Identifier("a"), "desc")
    right = ValueDomain(Identifier("b"), "desc")
    assert not left.is_structurally_equivalent(right)


def test_value_domain_not_equivalent_when_parents_differ() -> None:
    """Test instances with different `parent`s are not structurally equivalent."""
    name = Identifier("child")
    parented = ValueDomain(name, "desc", parent=DATA_DOMAIN)
    orphan = ValueDomain(name, "desc")
    assert not parented.is_structurally_equivalent(orphan)
    assert not orphan.is_structurally_equivalent(parented)


def test_value_domain_not_equivalent_to_non_value_domain() -> None:
    """Test structural equivalence with a non-`ValueDomain` value is False."""
    domain = ValueDomain(Identifier("x"), "desc")
    assert not domain.is_structurally_equivalent("x")
    assert not domain.is_structurally_equivalent(None)


def test_value_domain_structural_equivalence_recurses_into_parent() -> None:
    """Test parent comparison is structural rather than reference equality."""
    parent_name = Identifier("p")
    parent_left = ValueDomain(parent_name, "left parent")
    parent_right = ValueDomain(parent_name, "right parent")
    child_name = Identifier("c")
    left = ValueDomain(child_name, "left", parent=parent_left)
    right = ValueDomain(child_name, "right", parent=parent_right)
    assert left.is_structurally_equivalent(right)


# =============================================================================
# Parent chain & is_subdomain_of
# =============================================================================


def test_value_domain_is_subdomain_of_itself() -> None:
    """Test `is_subdomain_of` is reflexive."""
    child = ValueDomain(Identifier("c"), "child", parent=DATA_DOMAIN)
    assert child.is_subdomain_of(child)


def test_value_domain_is_subdomain_of_direct_parent() -> None:
    """Test a child reports itself as a subdomain of its direct parent."""
    child = ValueDomain(Identifier("c"), "child", parent=DATA_DOMAIN)
    assert child.is_subdomain_of(DATA_DOMAIN)


def test_value_domain_is_subdomain_of_distant_ancestor() -> None:
    """Test `is_subdomain_of` follows the full parent chain."""
    middle = ValueDomain(Identifier("m"), "middle", parent=DATA_DOMAIN)
    leaf = ValueDomain(Identifier("l"), "leaf", parent=middle)
    assert leaf.is_subdomain_of(DATA_DOMAIN)


def test_value_domain_not_subdomain_of_sibling() -> None:
    """Test unrelated domains are not subdomains."""
    child = ValueDomain(Identifier("c"), "child", parent=DATA_DOMAIN)
    assert not child.is_subdomain_of(ADDRESS_DOMAIN)


def test_value_domain_root_not_subdomain_of_child() -> None:
    """Test the parent direction is one-way."""
    child = ValueDomain(Identifier("c"), "child", parent=DATA_DOMAIN)
    assert not DATA_DOMAIN.is_subdomain_of(child)


# =============================================================================
# Serialization
# =============================================================================


def test_value_domain_serialize_to_dict_includes_required_fields() -> None:
    """Test `serialize_to_dict` emits `name`, `description`, and `parent` keys."""
    domain = ValueDomain(Identifier("x"), "desc")
    data = domain.serialize_to_dict()
    assert set(data.keys()) == {"name", "description", "parent"}
    assert data["description"] == "desc"
    assert data["parent"] is None


def test_value_domain_serialize_to_dict_inlines_parent() -> None:
    """Test the parent field is serialized as a nested dict when present."""
    child = ValueDomain(Identifier("c"), "child", parent=DATA_DOMAIN)
    data = child.serialize_to_dict()
    parent_data = data["parent"]
    assert isinstance(parent_data, dict)
    assert "name" in parent_data and "description" in parent_data


def test_value_domain_dict_round_trip_with_no_parent_is_structurally_equivalent() -> (
    None
):
    """Test parent-less dict round-trips reconstruct a structurally-equivalent
    domain."""
    original = ValueDomain(Identifier("fresh-no-parent"), "round trip")
    data = original.serialize_to_dict()
    restored = ValueDomain.deserialize_from_dict(data)
    assert restored.is_structurally_equivalent(original)
    assert restored.description == "round trip"


def test_value_domain_dict_round_trip_with_parent_is_structurally_equivalent() -> None:
    """Test parented dict round-trips preserve the parent chain."""
    original = ValueDomain(
        Identifier("fresh-with-parent"), "round trip", parent=DATA_DOMAIN
    )
    data = original.serialize_to_dict()
    restored = ValueDomain.deserialize_from_dict(data)
    assert restored.is_structurally_equivalent(original)
    assert restored.parent is not None
    assert restored.parent.is_structurally_equivalent(DATA_DOMAIN)


def test_value_domain_deserialize_rejects_malformed_dict() -> None:
    """Test deserializing a dict missing required fields raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        ValueDomain.deserialize_from_dict({"name": {"id": 0, "name_hint": "x"}})


def test_value_domain_deserialize_returns_canonical_for_registered_name() -> None:
    """Test deserialization returns the canonical interned instance when one
    is registered for the deserialized name."""
    data = DATA_DOMAIN.serialize_to_dict()
    restored = ValueDomain.deserialize_from_dict(data)
    assert restored is DATA_DOMAIN


def test_value_domain_deserialize_constructs_fresh_for_unregistered_name() -> None:
    """Test deserialization constructs a fresh instance for an unseen identifier."""
    unregistered_name = Identifier("never-registered-value-domain")
    # The identifier exists but no `ValueDomain` was constructed against it,
    # so the registry has no canonical entry for this name.
    assert ValueDomain.get_interned(unregistered_name) is None
    data: SerializedDict = {
        "name": unregistered_name.serialize_to_dict(),
        "description": "fresh from deserialize",
        "parent": None,
    }
    restored = ValueDomain.deserialize_from_dict(data)
    assert isinstance(restored, ValueDomain)
    assert restored.description == "fresh from deserialize"
    assert restored.parent is None
    # The constructed instance is now the canonical entry for that name.
    assert ValueDomain.get_interned(restored.name) is restored


def test_value_domain_binary_round_trip_via_from_bytes() -> None:
    """Test the binary envelope round-trips a fresh domain to a
    structurally-equivalent instance through the registered type id."""
    original = ValueDomain(Identifier("fresh-binary"), "binary round trip")
    blob = original.to_bytes()
    restored = Serializable.from_bytes(blob)
    assert isinstance(restored, ValueDomain)
    assert restored.is_structurally_equivalent(original)


def test_value_domain_json_round_trip_via_deserialize() -> None:
    """Test the JSON form is round-trippable through the public `deserialize` API."""
    original = ValueDomain(Identifier("fresh-json"), "json round trip")
    payload = original.serialize(SerializationFormat.JSON)
    restored = ValueDomain.deserialize(payload, SerializationFormat.JSON)
    assert restored.is_structurally_equivalent(original)
