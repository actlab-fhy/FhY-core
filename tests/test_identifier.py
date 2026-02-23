"""Tests the identifier."""

from copy import copy
from unittest.mock import patch

import pytest
from fhy_core.identifier import Identifier, SerializationError


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


def test_equality():
    """Test that the identifier equality is based on the ID."""
    id1 = Identifier("name")
    id2 = Identifier("name")
    id3 = Identifier("bar")

    assert id1 == id1  # noqa: PLR0124
    assert id1 != id2
    assert id2 != id3


def test_copy():
    """Test that the identifier can be copied."""
    identifier = Identifier("copy_test")
    copy_identifier = copy(identifier)

    assert copy_identifier.id == identifier.id
    assert copy_identifier.name_hint == identifier.name_hint
    assert copy_identifier == identifier


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

    with pytest.raises(SerializationError):
        Identifier.deserialize_from_dict({"invalid": "data"})
    with pytest.raises(SerializationError):
        Identifier.deserialize_from_dict({"id": "not_an_int", "name_hint": "test"})
    with pytest.raises(SerializationError):
        Identifier.deserialize_from_dict({"id": 1, "name_hint": 123})
    with pytest.raises(SerializationError):
        Identifier.deserialize_from_dict({"id": -1, "name_hint": "test"})
