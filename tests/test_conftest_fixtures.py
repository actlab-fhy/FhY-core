"""Tests for shared fixtures defined in `tests/conftest.py`."""

import copy
from typing import cast
from unittest.mock import Mock

import pytest

from fhy_core.symbolic.expression.registry import (
    get_native_constant_identifier,
    register_native_constant,
)
from fhy_core.symbolic.expression.sort import FunctionSort

from .conftest import MockIdentifierAliasError, mock_identifier


def test_mock_identifier_equals_and_hashes_like_a_mock_with_the_same_id() -> None:
    """Test two mocks with one id are equal and hash alike, as identifiers do."""
    identifier = mock_identifier("v", 1)
    twin = mock_identifier("v", 1)

    assert identifier == twin
    assert hash(identifier) == hash(twin)


def test_mock_identifier_differs_from_a_mock_with_another_id() -> None:
    """Test mocks with different ids are unequal."""
    assert mock_identifier("v", 1) != mock_identifier("v", 2)


def test_mock_identifier_differs_from_a_value_without_an_id() -> None:
    """Test a mock is unequal to a value that carries no ``id``."""
    assert mock_identifier("v", 1) != 1


def test_mock_identifier_renders_as_name_hint_and_id() -> None:
    """Test ``repr`` renders ``<name_hint>::<id>`` as an identifier does."""
    assert repr(mock_identifier("v", 1)) == "v::1"


def test_mock_identifier_deepcopy_equals_the_original() -> None:
    """Test a deep copy compares, hashes, and renders like the original."""
    identifier = mock_identifier("v", 1)

    duplicate = copy.deepcopy(identifier)

    assert duplicate == identifier
    assert hash(duplicate) == hash(identifier)
    assert repr(duplicate) == "v::1"


def test_mock_identifier_records_no_calls_when_compared_hashed_or_rendered() -> None:
    """Test comparing, hashing, and rendering a mock leave no call history.

    Pools of mock identifiers are shared across every example a worker
    runs, so a history that grew with each comparison would make every
    later deep copy of an expression slower than the last.
    """
    identifier = mock_identifier("v", 1)
    twin = mock_identifier("v", 1)

    observed = [identifier == twin, hash(identifier), repr(identifier)]

    assert observed == [True, hash(1), "v::1"]
    assert cast(Mock, identifier).mock_calls == []


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
