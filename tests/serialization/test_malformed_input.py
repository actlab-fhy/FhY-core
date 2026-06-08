"""Tests that malformed deserialization input stays within ``SerializationError``.

The module documents its default deserialization paths as safe against
untrusted input, and the JSON/binary entry points advertise
``Raises: SerializationError``. These tests pin that contract: invalid UTF-8,
invalid JSON, and a constructor that raises ``TypeError`` must all surface as a
``SerializationError`` rather than a raw ``UnicodeDecodeError`` /
``json.JSONDecodeError`` / ``TypeError``.
"""

from dataclasses import dataclass

import pytest

from fhy_core.serialization import (
    _HEADER_STRUCT,
    BinaryPayloadCodec,
    DeserializationValueError,
    MalformedPayloadError,
    Serializable,
    SerializationError,
    register_serializable,
)
from fhy_core.traits.frozen import FrozenMixin


@register_serializable(type_id="_test_malformed_point")
@dataclass(frozen=True)
class _Point(Serializable, FrozenMixin):
    x: int
    y: int


@register_serializable(type_id="_test_malformed_typeerror")
@dataclass(frozen=True)
class _RaisesTypeError(Serializable, FrozenMixin):
    x: int

    def __post_init__(self) -> None:
        raise TypeError("constructor rejects this payload")


@register_serializable(type_id="_test_malformed_serialerror")
@dataclass(frozen=True)
class _RaisesSerializationError(Serializable, FrozenMixin):
    x: int

    def __post_init__(self) -> None:
        raise DeserializationValueError("constructor raised a serialization error")


# ============================================================================
# Malformed payload error type
# ============================================================================


def test_malformed_payload_error_is_a_serialization_error() -> None:
    """Test ``MalformedPayloadError`` is catchable as ``SerializationError``."""
    assert issubclass(MalformedPayloadError, SerializationError)


# ============================================================================
# from_json
# ============================================================================


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param("{not valid json", "not valid JSON", id="invalid-json-text"),
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="invalid-utf8-bytes"),
    ],
)
def test_from_json_wraps_malformed_input(payload: str | bytes, match: str) -> None:
    """Test ``from_json`` reports malformed input as ``MalformedPayloadError``."""
    with pytest.raises(MalformedPayloadError, match=match):
        _Point.from_json(payload)


# ============================================================================
# deserialize_from_binary
# ============================================================================


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param(b"\xff\xfe", "not valid UTF-8", id="invalid-utf8-payload"),
        pytest.param(b"{not json", "not valid JSON", id="invalid-json-payload"),
    ],
)
def test_deserialize_from_binary_wraps_malformed_payload(
    payload: bytes, match: str
) -> None:
    """Test ``deserialize_from_binary`` reports a malformed JSON payload."""
    with pytest.raises(MalformedPayloadError, match=match):
        _Point.deserialize_from_binary(payload, codec=BinaryPayloadCodec.JSON)


# ============================================================================
# from_bytes (binary envelope)
# ============================================================================


def test_from_bytes_wraps_non_utf8_type_id() -> None:
    """Test a binary envelope with a non-UTF-8 type_id stays a serialization error."""
    blob = bytearray(_Point(1, 2).to_bytes())
    # The type_id bytes immediately follow the fixed-size header; 0xFF is never
    # a valid UTF-8 leading byte, so decoding the type_id must fail.
    blob[_HEADER_STRUCT.size] = 0xFF

    with pytest.raises(SerializationError):
        Serializable.from_bytes(bytes(blob))


# ============================================================================
# Constructor errors surface through the SerializationError hierarchy
# ============================================================================


def test_constructor_type_error_surfaces_as_serialization_error() -> None:
    """Test a ``TypeError`` from a constructor surfaces as a serialization error."""
    # The constructor raises unconditionally, so the payload is a structurally
    # valid literal rather than a round-tripped instance.
    with pytest.raises(DeserializationValueError):
        _RaisesTypeError.deserialize_from_dict({"x": 0})


def test_constructor_serialization_error_propagates_unwrapped() -> None:
    """Test a constructor's ``SerializationError`` propagates without re-wrapping."""
    with pytest.raises(
        DeserializationValueError, match="constructor raised a serialization error"
    ) as exc_info:
        _RaisesSerializationError.deserialize_from_dict({"x": 0})

    assert str(exc_info.value) == "constructor raised a serialization error"
