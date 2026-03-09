"""Tests the identifier."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
)
from fhy_core.trait import Equal, PartialEqual


def test_identifier_initialization():
    """Test that the identifier is initialized correctly."""
    identifier = Identifier("test_name")
    assert identifier.name_hint == "test_name"
    assert isinstance(identifier.id, int)


@patch.object(Identifier, "_next_id", 0)
def test_unique_id_generation():
    """Test that the identifier generates a unique ID."""
    id1 = Identifier("")
    id2 = Identifier("")
    id3 = Identifier("")

    assert id1.id == 0
    assert id2.id == 1
    assert id3.id == 2


def test_unique_id_generation_thread_safe():
    """Test that IDs remain unique under concurrent creation."""
    num_identifiers = 2000
    with patch.object(Identifier, "_next_id", 0):
        with ThreadPoolExecutor(max_workers=32) as executor:
            identifiers = list(
                executor.map(lambda _: Identifier(""), range(num_identifiers))
            )

    ids = [identifier.id for identifier in identifiers]
    assert len(set(ids)) == num_identifiers
    assert min(ids) == 0
    assert max(ids) == num_identifiers - 1


def test_equality():
    """Test that the identifier equality is based on the ID."""
    id1 = Identifier("name")
    id2 = Identifier("name")
    id3 = Identifier("bar")

    assert id1 == id1  # noqa: PLR0124
    assert id1 != id2
    assert id2 != id3


def test_identifier_supports_equal_traits() -> None:
    """Test `Identifier` satisfies equality trait protocols."""
    identifier = Identifier("name")
    assert isinstance(identifier, PartialEqual)
    assert isinstance(identifier, Equal)
    assert identifier.supports_partial_equality is True
    assert identifier.supports_equality is True


def test_string_representation():
    """Test that the identifier string representation is correct."""
    identifier = Identifier("test_repr")
    assert str(identifier) == "test_repr"
    assert repr(identifier) == f"test_repr::{identifier.id}"


def test_hash():
    """Test that the identifier hash is based on the ID."""
    identifier = Identifier("hash_test")
    assert hash(identifier) == hash(identifier.id)


def test_dict_serialization():
    """Test the identifier can be serialized/deserialized via a dictionary."""
    identifier = Identifier("serialization_test")
    serialized = identifier.serialize_to_dict()
    deserialized = Identifier.deserialize_from_dict(serialized)

    assert deserialized == identifier

    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"invalid": "data"})
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"id": "not_an_int", "name_hint": "test"})
    with pytest.raises(DeserializationDictStructureError):
        Identifier.deserialize_from_dict({"id": 1, "name_hint": 123})
    with pytest.raises(DeserializationValueError):
        Identifier.deserialize_from_dict({"id": -1, "name_hint": "test"})
