"""Tests for shared fixtures defined in `tests/conftest.py`."""

import pytest

from fhy_core.symbolic.expression.registry import (
    get_native_constant_identifier,
    register_native_constant,
)
from fhy_core.symbolic.expression.sort import FunctionSort

from .conftest import MockIdentifierAliasError, mock_identifier


def test_mock_identifier_deserialize_recovers_name_hint_and_id_from_dict() -> None:
    """Test the mock's ``deserialize_from_dict`` lambda restores correct fields."""
    original = mock_identifier("alpha", 7)
    serialized = original.serialize_to_dict()

    restored = original.deserialize_from_dict(serialized)

    assert restored.name_hint == "alpha"
    assert restored.id == 7


@pytest.mark.parametrize("constant_name", ["pi", "e", "inf", "nan"])
def test_mock_identifier_refuses_the_id_of_a_native_constant(
    constant_name: str,
) -> None:
    """Test a mock cannot take the id of a native constant's identifier.

    A mock compares and hashes by id, as the real `Identifier` does, so a
    mock holding that id would be the constant to every registry lookup,
    and a test using it as a variable would silently ask about the
    constant instead.
    """
    constant_id = get_native_constant_identifier(constant_name).id

    with pytest.raises(MockIdentifierAliasError, match=repr(constant_name)):
        mock_identifier("x", constant_id)


def test_mock_identifier_refuses_the_id_of_a_constant_registered_in_the_test(
    function_registry_snapshot: None,
) -> None:
    """Test the guard reads the registry when the mock is made, not at import."""
    register_native_constant("test_mock_guard_tau", FunctionSort.REAL, 6.25)
    constant_id = get_native_constant_identifier("test_mock_guard_tau").id

    with pytest.raises(MockIdentifierAliasError, match="'test_mock_guard_tau'"):
        mock_identifier("x", constant_id)
