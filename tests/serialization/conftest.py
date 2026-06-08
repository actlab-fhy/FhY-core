"""Shared assertion helpers for serialization tests."""

import pytest

from fhy_core.serialization import (
    Serializable,
    SerializationError,
    SerializationFormat,
    SerializedDict,
)

__all__ = [
    "assert_dict_round_trip",
    "assert_round_trips_in_all_formats",
    "assert_rejects_malformed_dict",
]


def assert_dict_round_trip(instance: Serializable) -> None:
    """Assert ``instance`` survives a dict serialize/deserialize round trip.

    Args:
        instance: A serializable object expected to equal itself after a
            ``serialize_to_dict`` / ``deserialize_from_dict`` round trip.
    """
    rebuilt = type(instance).deserialize_from_dict(instance.serialize_to_dict())
    assert rebuilt == instance


def assert_round_trips_in_all_formats(instance: Serializable) -> None:
    """Assert ``instance`` round-trips through every serialization format.

    Exercises DICT, JSON, and BINARY via the public ``serialize`` /
    ``deserialize`` entry points. BINARY requires the instance's class to be
    registered with ``register_serializable``.

    Args:
        instance: A serializable object expected to equal itself after a
            round trip in each format.
    """
    cls = type(instance)
    for fmt in SerializationFormat:
        assert cls.deserialize(instance.serialize(fmt), fmt) == instance


def assert_rejects_malformed_dict(
    cls: type[Serializable],
    data: SerializedDict,
    *,
    expected_error: type[Exception] = SerializationError,
) -> None:
    """Assert deserializing ``data`` into ``cls`` raises a serialization error.

    Args:
        cls: The serializable class under test.
        data: A malformed serialized dict (e.g. a missing, extra, or
            wrong-typed field).
        expected_error: The exception type expected to be raised. Defaults to
            ``SerializationError`` (the base of every serialization failure).
    """
    with pytest.raises(expected_error):
        cls.deserialize_from_dict(data)
