"""Tests for shared fixtures defined in `tests/conftest.py`."""

from .conftest import mock_identifier


def test_mock_identifier_deserialize_recovers_name_hint_and_id_from_dict() -> None:
    """Test the mock's ``deserialize_from_dict`` lambda restores correct fields."""
    original = mock_identifier("alpha", 7)
    serialized = original.serialize_to_dict()

    restored = original.deserialize_from_dict(serialized)

    assert restored.name_hint == "alpha"
    assert restored.id == 7
