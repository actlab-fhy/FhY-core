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


# =============================================================================
# Equality & hash
# =============================================================================


def test_value_domain_equality_ignores_description() -> None:
    """Test `__eq__` ignores `description` (metadata)."""
    name = Identifier("x")
    left = ValueDomain(name, "first")
    right = ValueDomain(name, "second")
    assert left == right


def test_value_domain_hash_ignores_description() -> None:
    """Test `__hash__` ignores `description` (metadata)."""
    name = Identifier("x")
    left = ValueDomain(name, "first")
    right = ValueDomain(name, "second")
    assert hash(left) == hash(right)


def test_value_domain_unequal_when_names_differ() -> None:
    """Test instances with different `name`s compare unequal."""
    assert ValueDomain(Identifier("a"), "desc") != ValueDomain(Identifier("b"), "desc")


def test_value_domain_unequal_when_parents_differ() -> None:
    """Test `__eq__` distinguishes domains with the same `name` but different
    `parent`s, keeping equality aligned with structural equivalence."""
    name = Identifier("child")
    parented = ValueDomain(name, "desc", parent=DATA_DOMAIN)
    orphan = ValueDomain(name, "desc")
    assert parented != orphan


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


# =============================================================================
# Description-mismatch deserialization warning
# =============================================================================


def test_value_domain_deserialize_warns_on_description_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test deserializing a canonical name with a different description warns."""
    canonical = ValueDomain(
        Identifier("mismatch-warning-domain"), "original description"
    )
    payload: SerializedDict = {
        "name": canonical.name.serialize_to_dict(),
        "description": "divergent description",
        "parent": None,
    }
    with caplog.at_level("WARNING", logger="fhy_core.trait.interned"):
        restored = ValueDomain.deserialize_from_dict(payload)

    assert restored is canonical
    assert restored.description == "original description"
    assert any(
        "already canonical" in record.getMessage()
        and "divergent description" in record.getMessage()
        for record in caplog.records
    )


def test_value_domain_deserialize_does_not_warn_when_descriptions_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test deserializing a canonical name with matching description does not warn."""
    canonical = ValueDomain(
        Identifier("matching-description-domain"), "matching description"
    )
    payload = canonical.serialize_to_dict()
    with caplog.at_level("WARNING", logger="fhy_core.trait.interned"):
        restored = ValueDomain.deserialize_from_dict(payload)

    assert restored is canonical
    assert not any(
        "already canonical" in record.getMessage() for record in caplog.records
    )


# =============================================================================
# register_default_instances restores module-level canonicals
# =============================================================================


def test_clearing_registry_desyncs_module_level_constants_without_default_restore() -> (
    None
):
    """Test clearing the registry desyncs the module-level constants."""
    try:
        ValueDomain.clear_interned_registry()
        payload = DATA_DOMAIN.serialize_to_dict()
        restored = ValueDomain.deserialize_from_dict(payload)
        assert restored is not DATA_DOMAIN
    finally:
        ValueDomain.register_default_instances()


def test_register_default_instances_restores_module_level_canonicals() -> None:
    """Test ``register_default_instances`` re-canonicalizes shipped defaults."""
    try:
        ValueDomain.clear_interned_registry()
        ValueDomain.register_default_instances()

        restored_data = ValueDomain.deserialize_from_dict(
            DATA_DOMAIN.serialize_to_dict()
        )
        restored_address = ValueDomain.deserialize_from_dict(
            ADDRESS_DOMAIN.serialize_to_dict()
        )
        assert restored_data is DATA_DOMAIN
        assert restored_address is ADDRESS_DOMAIN
    finally:
        ValueDomain.register_default_instances()
