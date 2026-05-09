"""Tests for shared fixtures defined in `tests/conftest.py`.

These tests exist so a future caller of ``mock_identifier(...).deserialize_from_dict``
gets a working mock instead of one with name and id swapped (see [F-009] in
the ``expression-data-and-io`` audit).
"""

from .conftest import mock_identifier


def test_mock_identifier_deserialize_recovers_name_hint_and_id_from_dict() -> None:
    """Test the mock's ``deserialize_from_dict`` lambda restores the right fields.

    Pre-fix, the lambda passed ``data["id"]`` to the ``name_hint`` slot and
    ``data["name_hint"]`` to the id slot. The bug was latent because no test
    invoked the lambda; this test exercises it so the failure mode surfaces
    if the swap regresses.
    """
    original = mock_identifier("alpha", 7)
    serialized = original.serialize_to_dict()

    restored = original.deserialize_from_dict(serialized)

    assert restored.name_hint == "alpha"
    assert restored.id == 7
