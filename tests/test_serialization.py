"""Tests the serialization module."""

import importlib
import json
import struct
import types
from abc import ABC
from dataclasses import dataclass
from typing import Any, Mapping

import pytest
from fhy_core import serialization


@serialization.register_serializable
@dataclass(frozen=True)
class _DummySpan(serialization.Serializable):
    lo: int
    hi: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_DummySpan":
        return cls(int(data["lo"]), int(data["hi"]))


@serialization.register_serializable
@dataclass(frozen=True)
class _CompactSpan(serialization.Serializable):
    """A span that uses a compact custom binary payload (8 bytes)."""

    lo: int
    hi: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_CompactSpan":
        return cls(int(data["lo"]), int(data["hi"]))

    @classmethod
    def get_binary_codec(cls) -> serialization.BinaryPayloadCodec:
        return serialization.BinaryPayloadCodec.CUSTOM

    def serialize_to_binary(self, *, codec: serialization.BinaryPayloadCodec) -> bytes:
        if codec is not serialization.BinaryPayloadCodec.CUSTOM:
            raise serialization.CodecMismatchError("Expected CUSTOM codec.")
        return struct.pack("!ii", self.lo, self.hi)

    @classmethod
    def deserialize_from_binary(
        cls,
        payload: bytes,
        *,
        codec: serialization.BinaryPayloadCodec,
    ) -> "_CompactSpan":
        if codec is not serialization.BinaryPayloadCodec.CUSTOM:
            raise serialization.CodecMismatchError("Expected CUSTOM codec.")
        lo, hi = struct.unpack("!ii", payload)
        return cls(lo=lo, hi=hi)


@serialization.register_serializable(type_id="compiler_core.CustomIdNode")
@dataclass(frozen=True)
class _CustomIdNode(serialization.Serializable):
    x: int

    TYPE_ID = "compiler_core.CustomIdNode"

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"x": self.x}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_CustomIdNode":
        return cls(int(data["x"]))


class _CustomNodeBase(serialization.WrappedFamilySerializable, ABC):
    pass


class _CustomNode(_CustomNodeBase):
    value: int

    def __init__(self, value: int):
        self.value = value

    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "_CustomNode":
        return cls(data["value"])


def test_dict_round_trip_via_classmethod():
    """Test that `serialize`/`deserialize` via dict format works."""
    span = _DummySpan(1, 9)
    dictionary = span.serialize(serialization.SerializationFormat.DICT)
    assert dictionary == {"lo": 1, "hi": 9}

    span2 = _DummySpan.deserialize(dictionary, serialization.SerializationFormat.DICT)
    assert span2 == span


def test_json_round_trip_via_classmethod():
    """Test that `serialize`/`deserialize` via JSON format works."""
    span = _DummySpan(2, 7)
    json_str = span.serialize(serialization.SerializationFormat.JSON)
    assert isinstance(json_str, str)

    span2 = _DummySpan.deserialize(json_str, serialization.SerializationFormat.JSON)
    assert span2 == span

    span3 = _DummySpan.deserialize(
        json_str.encode("utf-8"), serialization.SerializationFormat.JSON
    )
    assert span3 == span


def test_serialize_dispatch_and_invalid_format():
    """Test that `serialize` returns the correct format/rejects invalid formats."""
    span = _DummySpan(0, 1)
    assert span.serialize(serialization.SerializationFormat.DICT) == {"lo": 0, "hi": 1}
    assert isinstance(span.serialize(serialization.SerializationFormat.JSON), str)
    assert isinstance(
        span.serialize(serialization.SerializationFormat.BINARY), (bytes, bytearray)
    )

    with pytest.raises(serialization.SerializationError):
        span.serialize("nope")


def test_deserialize_type_mismatch_raises():
    """Test that deserializing with a type that doesn't match raises an error."""
    blob = _DummySpan(1, 2).to_bytes()
    with pytest.raises(serialization.SerializationError):
        _CustomIdNode.deserialize(blob, serialization.SerializationFormat.BINARY)


def test_raise_error_if_deserialization_data_invalid_success():
    """Test that valid required/optional fields do not raise."""
    data = {"name": "node", "count": 2, "meta": {"tag": "ok"}}

    _DummySpan.raise_error_if_deserialization_from_dict_data_invalid(
        data,
        required_fields={"name": str, "count": lambda v: isinstance(v, int) and v >= 0},
        optional_fields={"meta": dict},
    )


@pytest.mark.parametrize(
    ("data", "required_fields", "optional_fields", "error_type"),
    [
        (
            {},
            {"name": str},
            None,
            serialization.SerializationError,
        ),
        (
            {"name": 42},
            {"name": str},
            None,
            serialization.SerializationError,
        ),
        (
            {"count": -1},
            {"count": lambda v: isinstance(v, int) and v >= 0},
            None,
            serialization.SerializationError,
        ),
        (
            {"name": "ok", "meta": "not-a-dict"},
            {"name": str},
            {"meta": dict},
            serialization.SerializationError,
        ),
        (
            {"name": "ok", "meta": {}},
            {"name": str},
            {"meta": object()},
            ValueError,
        ),
    ],
)
def test_raise_error_if_deserialization_data_invalid_failure_cases(
    data: Mapping[str, Any],
    required_fields: dict[str, type | Any],
    optional_fields: dict[str, type | Any] | None,
    error_type: type[Exception],
):
    """Test that invalid input triggers the expected failures."""
    with pytest.raises(error_type):
        _DummySpan.raise_error_if_deserialization_from_dict_data_invalid(
            data,
            required_fields=required_fields,
            optional_fields=optional_fields,
        )


def test_wrapped_dict_round_trip():
    """Test that `get_wrapper_dict` and `unwrap_wrapper_dict` work together."""
    span = _DummySpan(3, 4)
    wrapper_dict = serialization.get_wrapper_dict(span)
    assert wrapper_dict["__type__"] == _DummySpan.get_class_type_id()
    assert wrapper_dict["__data__"] == {"lo": 3, "hi": 4}

    span2 = serialization.unwrap_wrapper_dict(wrapper_dict)
    assert isinstance(span2, _DummySpan)
    assert span2 == span


def test_unwrap_dict_invalid_payload_raises():
    """Test that `unwrap_wrapper_dict` raises if fields are not in the right format."""
    with pytest.raises(serialization.SerializationError):
        serialization.unwrap_wrapper_dict({"__type__": "x", "__data__": 123})
    with pytest.raises(serialization.SerializationError):
        serialization.unwrap_wrapper_dict({"__type__": 123, "__data__": {}})


def test_wrapped_family_serializable_round_trip():
    """Test that a `WrappedFamilySerializable` can round-trip."""
    node = _CustomNode(42)

    dictionary = node.serialize(serialization.SerializationFormat.DICT)
    assert dictionary == {
        "__type__": node.get_class_type_id(),
        "__data__": {"value": 42},
    }

    node2 = _CustomNodeBase.deserialize(
        dictionary, serialization.SerializationFormat.DICT
    )
    assert isinstance(node2, _CustomNode)
    assert node2.value == 42

    node3 = _CustomNode.deserialize(dictionary, serialization.SerializationFormat.DICT)
    assert isinstance(node3, _CustomNode)
    assert node3.value == 42


def _parse_envelope(blob: bytes) -> tuple[bytes, int, int, str, bytes]:
    header_struct = struct.Struct("!4sBBH")
    payload_len_struct = struct.Struct("!I")

    magic, version, codec_u8, type_id_len = header_struct.unpack(
        blob[: header_struct.size]
    )
    offset = header_struct.size

    type_id = blob[offset : offset + type_id_len].decode("utf-8")
    offset += type_id_len

    (payload_len,) = payload_len_struct.unpack(
        blob[offset : offset + payload_len_struct.size]
    )
    offset += payload_len_struct.size

    payload = blob[offset : offset + payload_len]
    return magic, version, codec_u8, type_id, payload


def test_binary_round_trip_auto_reconstructs_class_default_json_codec():
    """Test that serializing to binary and deserializing reconstructs the object."""
    span = _DummySpan(10, 20)
    blob = span.to_bytes()

    obj = serialization.Serializable.from_bytes(blob)
    assert isinstance(obj, _DummySpan)
    assert obj == span


def test_binary_round_trip_auto_reconstructs_class_custom_compact_codec():
    """Test that custom compact binary payload round-trips and auto-reconstructs."""
    span = _CompactSpan(10, 20)
    blob = span.to_bytes()

    obj = serialization.Serializable.from_bytes(blob)
    assert isinstance(obj, _CompactSpan)
    assert obj == span


def test_binary_envelope_contains_expected_fields_json_codec():
    """Test that the envelope contains the expected fields for JSON codec payloads."""
    span = _DummySpan(5, 6)
    blob = span.to_bytes()

    magic, version, codec_u8, type_id, payload = _parse_envelope(blob)
    assert magic == b"FhYS"
    assert version == 1
    assert codec_u8 == 1  # JSON
    assert type_id == _DummySpan.get_class_type_id()

    payload_obj = json.loads(payload.decode("utf-8"))
    assert payload_obj == {"lo": 5, "hi": 6}


def test_binary_envelope_contains_expected_fields_custom_codec():
    """Test that the envelope contains the expected fields for CUSTOM codec payloads."""
    span = _CompactSpan(5, 6)
    blob = span.to_bytes()

    magic, version, codec_u8, type_id, payload = _parse_envelope(blob)
    assert magic == b"FhYS"
    assert version == 1
    assert codec_u8 == 2  # CUSTOM
    assert type_id == _CompactSpan.get_class_type_id()

    lo, hi = struct.unpack("!ii", payload)
    assert (lo, hi) == (5, 6)


def test_binary_custom_type_id_is_used():
    """Test that a custom type id is used in the binary envelope."""
    node = _CustomIdNode(123)
    blob = node.to_bytes()
    _, _, codec_u8, type_id, payload = _parse_envelope(blob)

    assert codec_u8 == 1  # default JSON codec
    assert type_id == "compiler_core.CustomIdNode"
    assert json.loads(payload.decode("utf-8")) == {"x": 123}

    obj = serialization.Serializable.from_bytes(blob)
    assert isinstance(obj, _CustomIdNode)
    assert obj == node


def test_binary_bad_magic_raises():
    """Test that a blob with a bad magic value raises an error."""
    blob = _DummySpan(1, 2).to_bytes()
    corrupted = b"NOPE" + blob[4:]
    with pytest.raises(serialization.SerializationError):
        serialization.Serializable.from_bytes(corrupted)


def test_binary_version_mismatch_raises():
    """Test that a blob with an unsupported version raises a `VersionMismatchError`."""
    blob = _DummySpan(1, 2).to_bytes()

    corrupted = bytearray(blob)
    corrupted[4] = 99  # version byte
    with pytest.raises(serialization.VersionMismatchError):
        serialization.Serializable.from_bytes(bytes(corrupted))


def test_binary_unknown_codec_raises():
    """Test that a blob with an unknown codec code raises an `UnknownCodecError`."""
    blob = _DummySpan(1, 2).to_bytes()
    corrupted = bytearray(blob)
    corrupted[5] = 250  # codec byte (unknown)
    with pytest.raises(serialization.UnknownCodecError):
        serialization.Serializable.from_bytes(bytes(corrupted))


def test_binary_codec_mismatch_raises_on_custom_class_with_json_envelope():
    """Test that codec mismatch is rejected (CUSTOM class, JSON envelope)."""
    span = _CompactSpan(1, 2)
    blob = span.to_bytes()
    corrupted = bytearray(blob)
    corrupted[5] = 1  # force JSON codec in envelope
    with pytest.raises(serialization.CodecMismatchError):
        serialization.Serializable.from_bytes(bytes(corrupted))


def test_binary_truncated_header_raises():
    """Test that a blob with a truncated header raises an error."""
    blob = _DummySpan(1, 2).to_bytes()
    with pytest.raises(serialization.SerializationError):
        serialization.Serializable.from_bytes(blob[:2])


def test_binary_truncated_type_id_or_payload_raises():
    """Test that a blob with a truncated type id or payload raises an error."""
    blob = _DummySpan(1, 2).to_bytes()
    with pytest.raises(serialization.SerializationError):
        serialization.Serializable.from_bytes(blob[:-1])


def test_binary_unknown_type_id_raises():
    """Test that a blob with an unknown type id raises an `UnknownTypeIdError`."""
    blob = _DummySpan(1, 2).to_bytes()
    magic, version, codec_u8, _, payload = _parse_envelope(blob)

    header_struct = struct.Struct("!4sBBH")
    payload_len_struct = struct.Struct("!I")

    fake_type_id = "does.not.Exist".encode("utf-8")
    rebuilt = (
        header_struct.pack(magic, version, codec_u8, len(fake_type_id))
        + fake_type_id
        + payload_len_struct.pack(len(payload))
        + payload
    )

    with pytest.raises(serialization.UnknownTypeIdError):
        serialization.Serializable.from_bytes(rebuilt)


def test_registry_resolution_prefers_registered_class():
    """Test that the registry resolution prefers registered classes over importlib."""
    cls = serialization._resolve_type_id(_DummySpan.get_class_type_id())
    assert cls is _DummySpan


def test_fallback_import_resolution_via_mock(monkeypatch):
    """Test that the fallback resolution works when the type is not in the registry."""
    mod = types.SimpleNamespace()

    class _Outer(serialization.Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {"ok": True}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Outer":
            return cls()

    setattr(mod, "_Outer", _Outer)

    def fake_import(name: str):
        assert name == "fakepkg.fakemod"
        return mod

    ty_id = "fakepkg.fakemod._Outer"
    monkeypatch.setattr(importlib, "import_module", fake_import)

    resolved = serialization._resolve_type_id(ty_id)
    assert resolved is _Outer


def test_fallback_import_resolution_rejects_non_serializable(monkeypatch):
    """Test that the fallback resolution rejects types that are not serializable."""
    mod = types.SimpleNamespace()

    class _NotSerializable:
        pass

    setattr(mod, "_NotSerializable", _NotSerializable)

    def fake_import(name: str):
        assert name == "fakepkg.fakemod"
        return mod

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(serialization.UnknownTypeIdError):
        serialization._resolve_type_id("fakepkg.fakemod._NotSerializable")


def test_from_json_non_object_raises():
    """Test that from_json raises if the JSON is not an object."""
    with pytest.raises(serialization.SerializationError):
        _DummySpan.from_json("[]")


def test_deserialize_invalid_payload_type_raises():
    """Test that deserializing with an invalid payload type raises an error."""
    with pytest.raises(serialization.SerializationError):
        _DummySpan.deserialize(123, serialization.SerializationFormat.DICT)
    with pytest.raises(serialization.SerializationError):
        _DummySpan.deserialize(123, serialization.SerializationFormat.JSON)
    with pytest.raises(serialization.SerializationError):
        _DummySpan.deserialize("not-bytes", serialization.SerializationFormat.BINARY)
