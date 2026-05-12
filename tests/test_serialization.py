"""Tests for `fhy_core.serialization`.

Drives the public surface of the serialization module:

- The ``is_serialized_*`` / ``is_registry_wrapped_*`` type-guards.
- ``serialize_registry_wrapped_value`` / ``deserialize_registry_wrapped_value``.
- The public error classes (constructed and registered via ``@register_error``).
- ``register_serializable`` and the type-id resolution it drives.
- ``Serializable`` round-trips across DICT, JSON, and BINARY formats.
- The binary envelope contract through ``to_bytes`` / ``from_bytes``.
- ``WrappedFamilySerializable`` family resolution.

Tests reach internal branches only via the public API so behavioral
mutations (off-by-one boundaries, flipped predicates, swapped operators)
surface as observable test failures.
"""

import importlib
import json
import logging
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from fhy_core.error import _COMPILER_ERRORS
from fhy_core.serialization import (
    BinaryPayloadCodec,
    CodecMismatchError,
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializationError,
    SerializationFormat,
    SerializationPayloadTypeError,
    SerializationTypeError,
    SerializationValueError,
    SerializedDict,
    UnknownCodecError,
    UnknownTypeIdError,
    VersionMismatchError,
    WrappedFamilySerializable,
    deserialize_registry_wrapped_value,
    is_registry_wrapped_value,
    is_registry_wrapped_value_leaf,
    is_serialized_dict,
    is_serialized_value,
    register_serializable,
    serialize_registry_wrapped_value,
)

# =============================================================================
# Registered serializable classes used by the test suite
# =============================================================================


@register_serializable
@dataclass(frozen=True)
class _DummySpan(Serializable):
    """Default-codec serializable with two integer fields."""

    lo: int
    hi: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_DummySpan":
        return cls(int(data["lo"]), int(data["hi"]))


@register_serializable
@dataclass(frozen=True)
class _CompactSpan(Serializable):
    """Custom-codec serializable that emits an 8-byte payload."""

    lo: int
    hi: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_CompactSpan":
        return cls(int(data["lo"]), int(data["hi"]))

    @classmethod
    def get_binary_codec(cls) -> BinaryPayloadCodec:
        return BinaryPayloadCodec.CUSTOM

    def serialize_to_binary(self, *, codec: BinaryPayloadCodec) -> bytes:
        if codec is not BinaryPayloadCodec.CUSTOM:
            raise CodecMismatchError("Expected CUSTOM codec.")
        return struct.pack("!ii", self.lo, self.hi)

    @classmethod
    def deserialize_from_binary(
        cls, payload: bytes, *, codec: BinaryPayloadCodec
    ) -> "_CompactSpan":
        if codec is not BinaryPayloadCodec.CUSTOM:
            raise CodecMismatchError("Expected CUSTOM codec.")
        lo, hi = struct.unpack("!ii", payload)
        return cls(lo=lo, hi=hi)


@register_serializable(type_id="compiler_core.CustomIdNode")
@dataclass(frozen=True)
class _CustomIdNode(Serializable):
    """Default-codec serializable with an explicit canonical ``type_id``."""

    x: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"x": self.x}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_CustomIdNode":
        return cls(int(data["x"]))


@dataclass(frozen=True)
class _UnregisteredSpan(Serializable):
    """Identical to ``_DummySpan`` but never registered.

    Reached only via the ``allow_import_fallback`` path of ``from_bytes``.
    """

    lo: int
    hi: int

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_UnregisteredSpan":
        return cls(int(data["lo"]), int(data["hi"]))


class _CustomNodeBase(WrappedFamilySerializable):
    """Family base for `WrappedFamilySerializable` round-trip tests."""


@register_serializable
@dataclass(frozen=True)
class _CustomNode(_CustomNodeBase):
    """Concrete family member used by the wrapped-family tests."""

    value: int

    def serialize_data_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def deserialize_data_from_dict(cls, data: Mapping[str, Any]) -> "_CustomNode":
        return cls(int(data["value"]))


@dataclass(frozen=True)
class _NonLeafHashable:
    """Frozen dataclass used as a hashable but non-leaf wrapped-value element."""

    x: int


# =============================================================================
# Constants and helper functions
# =============================================================================

_HEADER_STRUCT = struct.Struct("!4sBBH")
_PAYLOAD_LEN_STRUCT = struct.Struct("!I")


def _parse_envelope(blob: bytes) -> tuple[bytes, int, int, str, bytes]:
    """Decode the binary envelope into its named fields."""
    magic, version, codec_u8, type_id_len = _HEADER_STRUCT.unpack(
        blob[: _HEADER_STRUCT.size]
    )
    offset = _HEADER_STRUCT.size
    type_id = blob[offset : offset + type_id_len].decode("utf-8")
    offset += type_id_len
    (payload_len,) = _PAYLOAD_LEN_STRUCT.unpack(
        blob[offset : offset + _PAYLOAD_LEN_STRUCT.size]
    )
    offset += _PAYLOAD_LEN_STRUCT.size
    payload = blob[offset : offset + payload_len]
    return magic, version, codec_u8, type_id, payload


def _pack_envelope(
    *,
    magic: bytes = b"FhYS",
    version: int = 1,
    codec_u8: int = 1,
    type_id: bytes,
    payload: bytes,
) -> bytes:
    """Pack a synthetic binary envelope for negative-path tests."""
    return (
        _HEADER_STRUCT.pack(magic, version, codec_u8, len(type_id))
        + type_id
        + _PAYLOAD_LEN_STRUCT.pack(len(payload))
        + payload
    )


def _replace_byte(data: bytes, index: int, value: int) -> bytes:
    """Return ``data`` with the byte at ``index`` replaced by ``value``."""
    out = bytearray(data)
    out[index] = value
    return bytes(out)


def _decode_envelope_payload(payload: bytes, obj: Serializable) -> dict[str, int]:
    """Decode an envelope payload using the codec advertised by ``obj``."""
    if type(obj).get_binary_codec() is BinaryPayloadCodec.JSON:
        decoded: dict[str, int] = json.loads(payload.decode("utf-8"))
        return decoded
    lo, hi = struct.unpack("!ii", payload)
    return {"lo": lo, "hi": hi}


_CODEC_TO_U8 = {BinaryPayloadCodec.JSON: 1, BinaryPayloadCodec.CUSTOM: 2}


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def dummy_span() -> _DummySpan:
    """Return a registered default-codec serializable instance."""
    return _DummySpan(1, 2)


@pytest.fixture
def compact_span() -> _CompactSpan:
    """Return a registered custom-codec serializable instance."""
    return _CompactSpan(1, 2)


@pytest.fixture
def custom_id_node() -> _CustomIdNode:
    """Return a serializable with an explicit canonical ``type_id``."""
    return _CustomIdNode(123)


@pytest.fixture
def unregistered_span() -> _UnregisteredSpan:
    """Return an instance of a ``Serializable`` that was never registered."""
    return _UnregisteredSpan(1, 2)


@pytest.fixture
def custom_node() -> _CustomNode:
    """Return a concrete ``WrappedFamilySerializable`` instance."""
    return _CustomNode(7)


@pytest.fixture
def shorter_than_header_blob() -> bytes:
    """Return a blob shorter than the binary envelope header."""
    return b"\x00" * (_HEADER_STRUCT.size - 1)


@pytest.fixture
def header_only_blob() -> bytes:
    """Return a header-sized blob with ``type_id_len`` set to zero."""
    return _HEADER_STRUCT.pack(b"FhYS", 1, 1, 0)


@pytest.fixture
def truncated_payload_blob(dummy_span: _DummySpan) -> bytes:
    """Return a valid blob truncated by one byte inside the payload."""
    return bytes(bytearray(dummy_span.to_bytes())[:-1])


@pytest.fixture
def bad_magic_blob(dummy_span: _DummySpan) -> bytes:
    """Return a blob whose magic header has been corrupted."""
    return b"NOPE" + dummy_span.to_bytes()[4:]


@pytest.fixture
def unsupported_version_blob(dummy_span: _DummySpan) -> bytes:
    """Return a blob whose version byte is unsupported."""
    return _replace_byte(dummy_span.to_bytes(), 4, 99)


@pytest.fixture
def unknown_codec_blob(dummy_span: _DummySpan) -> bytes:
    """Return a blob whose codec byte is not recognized."""
    return _replace_byte(dummy_span.to_bytes(), 5, 250)


@pytest.fixture
def codec_mismatch_blob(compact_span: _CompactSpan) -> bytes:
    """Return a blob whose envelope codec disagrees with the class codec."""
    return _replace_byte(compact_span.to_bytes(), 5, 1)


@pytest.fixture
def unknown_type_id_blob() -> bytes:
    """Return a syntactically valid blob that names an unregistered class."""
    return _pack_envelope(type_id=b"does.not.Exist", payload=json.dumps({}).encode())


# =============================================================================
# is_serialized_value / is_serialized_dict
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        # Accepted scalar leaves.
        pytest.param(None, True, id="none"),
        pytest.param("text", True, id="str"),
        pytest.param(0, True, id="int_zero"),
        pytest.param(-3, True, id="int_negative"),
        pytest.param(1.5, True, id="float"),
        pytest.param(True, True, id="bool_true"),
        pytest.param(False, True, id="bool_false"),
        # Accepted nested structure.
        pytest.param([1, "a", {"k": [True, None, 1.25]}], True, id="nested"),
        # Rejected bytes-like values.
        pytest.param(b"", False, id="empty_bytes"),
        pytest.param(b"abc", False, id="bytes"),
        pytest.param(bytearray(b"abc"), False, id="bytearray"),
        pytest.param(memoryview(b"abc"), False, id="memoryview"),
        # Rejected unsupported types.
        pytest.param(object(), False, id="bare_object"),
        pytest.param(set(), False, id="empty_set"),
        pytest.param({1, 2, 3}, False, id="non_empty_set"),
        pytest.param(frozenset({1}), False, id="frozenset"),
        # Rejected sequence with one bad element.
        pytest.param([1, "a", object()], False, id="list_with_bad_element"),
    ],
)
def test_is_serialized_value_classifies_value(value: Any, expected: bool) -> None:
    """Test `is_serialized_value` accepts/rejects each input as expected."""
    assert is_serialized_value(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param({"k": 1, "j": [1.0, None]}, True, id="string_keys_leaf_values"),
        pytest.param("not a mapping", False, id="string_not_mapping"),
        pytest.param([("k", 1)], False, id="list_of_pairs_not_mapping"),
        pytest.param({1: "a"}, False, id="non_string_key"),
        pytest.param({"k": object()}, False, id="bad_value"),
        # Locks down per-pair iteration: any later bad entry must reject.
        pytest.param({"a": 1, 2: "b"}, False, id="late_bad_key"),
        pytest.param({"a": 1, "b": object()}, False, id="late_bad_value"),
    ],
)
def test_is_serialized_dict_classifies_value(value: Any, expected: bool) -> None:
    """Test `is_serialized_dict` accepts/rejects each mapping as expected."""
    assert is_serialized_dict(value) is expected


# =============================================================================
# Registry-wrapped value type guards
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(True, True, id="bool"),
        pytest.param(0, True, id="int"),
        pytest.param(-7, True, id="int_negative"),
        pytest.param("text", True, id="str"),
        pytest.param(2.5, True, id="float"),
        pytest.param(_DummySpan(0, 1), True, id="serializable"),
        pytest.param(b"", False, id="bytes"),
        pytest.param([1, 2], False, id="list"),
        pytest.param((), False, id="tuple"),
        pytest.param(frozenset(), False, id="frozenset"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_registry_wrapped_value_leaf_classifies_value(
    value: Any, expected: bool
) -> None:
    """Test `is_registry_wrapped_value_leaf` accepts/rejects each input."""
    assert is_registry_wrapped_value_leaf(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # Accepted leaves and containers.
        pytest.param(1, True, id="int_leaf"),
        pytest.param((1, 2), True, id="tuple_of_ints"),
        pytest.param((1, "a", _DummySpan(0, 1)), True, id="tuple_of_mixed_leaves"),
        pytest.param(frozenset({1, 2}), True, id="frozenset_of_ints"),
        pytest.param(((1,), frozenset({"a"})), True, id="nested_containers"),
        # Rejected non-wrapped containers.
        pytest.param([1, 2], False, id="list_not_tuple"),
        pytest.param({"a": 1}, False, id="dict"),
        pytest.param(set(), False, id="set_not_frozenset"),
        pytest.param(None, False, id="none"),
        pytest.param(b"x", False, id="bytes"),
        # Containers with invalid inner element.
        pytest.param((1, object()), False, id="tuple_with_invalid_element"),
        pytest.param(
            frozenset({_NonLeafHashable(1)}),
            False,
            id="frozenset_with_invalid_element",
        ),
    ],
)
def test_is_registry_wrapped_value_classifies_value(value: Any, expected: bool) -> None:
    """Test `is_registry_wrapped_value` accepts/rejects each input as expected."""
    assert is_registry_wrapped_value(value) is expected


# =============================================================================
# Registry-wrapped value round-trips
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(7, id="int"),
        pytest.param("value", id="str"),
        pytest.param(2.5, id="float"),
        pytest.param(_DummySpan(4, 9), id="serializable"),
        pytest.param((1, "a", True), id="tuple"),
        pytest.param(frozenset({1, 2, 3}), id="frozenset"),
        pytest.param(
            (1, (2, 3), frozenset({4, 5}), _DummySpan(6, 7)),
            id="nested_containers",
        ),
    ],
)
def test_registry_wrapped_value_round_trips(value: Any) -> None:
    """Test wrapped-value helpers round-trip the value to an equal object."""
    wrapped = serialize_registry_wrapped_value(value)
    assert deserialize_registry_wrapped_value(wrapped) == value


def test_registry_wrapped_value_round_trips_bool_preserving_type() -> None:
    """Test wrapped-value helpers round-trip bools without coercing to int."""
    restored = deserialize_registry_wrapped_value(
        serialize_registry_wrapped_value(True)
    )
    assert restored is True
    assert isinstance(restored, bool)


def test_registry_wrapped_value_serialize_rejects_unsupported_value_type() -> None:
    """Test `serialize_registry_wrapped_value` raises on unsupported value types."""
    with pytest.raises(SerializationTypeError):
        serialize_registry_wrapped_value(object())  # type: ignore[arg-type]


# =============================================================================
# Registry-wrapped deserialization error paths
# =============================================================================


@pytest.mark.parametrize(
    "data, expected_error",
    [
        # Envelope structure errors.
        pytest.param(
            {"bad": "data"},
            DeserializationDictStructureError,
            id="missing_type_and_data",
        ),
        pytest.param(
            {"__type__": "builtins.int"},
            DeserializationDictStructureError,
            id="missing_data",
        ),
        pytest.param(
            {"__data__": 1},
            DeserializationDictStructureError,
            id="missing_type",
        ),
        pytest.param(
            {"__type__": 7, "__data__": 1},
            DeserializationDictStructureError,
            id="non_string_type",
        ),
        pytest.param(
            {"__type__": "builtins.int", "__data__": object()},
            DeserializationDictStructureError,
            id="non_serialized_data",
        ),
        # Leaf payload type errors.
        pytest.param(
            {"__type__": "builtins.int", "__data__": True},
            DeserializationValueError,
            id="bool_payload_for_int_leaf",
        ),
        pytest.param(
            {"__type__": "builtins.bool", "__data__": 1},
            DeserializationValueError,
            id="int_payload_for_bool_leaf",
        ),
        pytest.param(
            {"__type__": "builtins.str", "__data__": 1},
            DeserializationValueError,
            id="non_string_payload_for_str_leaf",
        ),
        pytest.param(
            {"__type__": "builtins.float", "__data__": "1.0"},
            DeserializationValueError,
            id="non_float_payload_for_float_leaf",
        ),
        # Container payload errors.
        pytest.param(
            {"__type__": "builtins.tuple", "__data__": "not-a-list"},
            DeserializationValueError,
            id="non_list_container_payload",
        ),
        pytest.param(
            {"__type__": "builtins.tuple", "__data__": [1, 2]},
            DeserializationValueError,
            id="unwrapped_container_items",
        ),
        # Object-payload error: non-dict data for a registered class.
        pytest.param(
            {
                "__type__": _DummySpan.get_serialization_class_type_id(),
                "__data__": "not-a-dict",
            },
            DeserializationValueError,
            id="non_dict_payload_for_object_leaf",
        ),
        # Unknown type id (registry miss + fallback off).
        pytest.param(
            {"__type__": "no.such.Class", "__data__": {}},
            UnknownTypeIdError,
            id="unknown_type_id",
        ),
    ],
)
def test_deserialize_registry_wrapped_value_rejects_invalid_data(
    data: dict[str, Any], expected_error: type[Exception]
) -> None:
    """Test `deserialize_registry_wrapped_value` raises the expected error."""
    with pytest.raises(expected_error):
        deserialize_registry_wrapped_value(data)


def test_deserialize_registry_wrapped_value_rejects_unhashable_frozenset_item() -> None:
    """Test wrapped frozenset payload with unhashable inner items raises.

    A `Serializable` whose ``__hash__`` raises forces the constructor to
    catch the ``TypeError`` and re-raise as ``DeserializationValueError``.
    """

    @register_serializable(type_id="tests.UnhashableLeaf")
    class _UnhashableLeaf(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_UnhashableLeaf":
            return cls()

        def __hash__(self) -> int:  # pragma: no cover - explicitly unhashable
            raise TypeError("unhashable")

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _UnhashableLeaf)

    payload: SerializedDict = {
        "__type__": "builtins.frozenset",
        "__data__": [
            {
                "__type__": _UnhashableLeaf.get_serialization_class_type_id(),
                "__data__": {},
            }
        ],
    }
    with pytest.raises(DeserializationValueError):
        deserialize_registry_wrapped_value(payload)


# =============================================================================
# Public error classes
# =============================================================================


@pytest.mark.parametrize(
    "error_class",
    [
        SerializationPayloadTypeError,
        SerializationTypeError,
        SerializationValueError,
        DeserializationDictStructureError,
        DeserializationValueError,
        UnknownTypeIdError,
        VersionMismatchError,
        UnknownCodecError,
        CodecMismatchError,
    ],
)
def test_serialization_error_class_is_registered_in_compiler_error_registry(
    error_class: type[Exception],
) -> None:
    """Test each public error class is registered via `@register_error`."""
    assert error_class in _COMPILER_ERRORS


def test_deserialization_value_error_one_arg_message_uses_arg_verbatim() -> None:
    """Test `DeserializationValueError("msg")` exposes the message verbatim."""
    assert str(DeserializationValueError("a custom message")) == "a custom message"


def test_deserialization_value_error_four_arg_message_includes_all_pieces() -> None:
    """Test the four-arg constructor formats class, field, phrase, and value repr."""

    class _MyClass:
        pass

    text = str(DeserializationValueError(_MyClass, "my_field", "an int", "x"))
    assert "_MyClass" in text
    assert "my_field" in text
    assert "an int" in text
    assert "'x'" in text


@pytest.mark.parametrize(
    "args",
    [
        pytest.param((), id="zero_args"),
        pytest.param((1,), id="one_non_string"),
        pytest.param(("a", "b", "c"), id="three_args"),
        pytest.param(("a", "b", "c", "d"), id="four_args_first_not_type"),
    ],
)
def test_deserialization_value_error_invalid_arity_raises_value_error(
    args: tuple[Any, ...],
) -> None:
    """Test the constructor rejects argument shapes that match neither overload."""
    with pytest.raises(ValueError, match="Invalid arguments"):
        DeserializationValueError(*args)


def test_serialization_value_error_message_includes_phrase_and_value_repr() -> None:
    """Test `SerializationValueError` formats the phrase and value `repr`."""
    text = str(SerializationValueError("a positive int", -3))
    assert "a positive int" in text
    assert "-3" in text


# =============================================================================
# register_serializable
# =============================================================================


def test_register_serializable_assigns_default_type_id_when_not_specified() -> None:
    """Test `register_serializable` assigns the module.qualname default id."""

    @register_serializable
    class _DefaultIdClass(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_DefaultIdClass":
            return cls()

    expected_id = f"{_DefaultIdClass.__module__}.{_DefaultIdClass.__qualname__}"
    assert _DefaultIdClass.get_serialization_class_type_id() == expected_id


def test_register_serializable_assigns_provided_type_id_as_canonical() -> None:
    """Test `register_serializable(type_id=...)` makes the id canonical on the class."""

    @register_serializable(type_id="tests.CanonicalA")
    class _Canon(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Canon":
            return cls()

    assert _Canon.get_serialization_class_type_id() == "tests.CanonicalA"


def test_register_serializable_alias_does_not_override_canonical_id() -> None:
    """Test `alias=True` registers a secondary id without changing the canonical."""

    @register_serializable(type_id="tests.CanonicalMain")
    class _Aliased(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Aliased":
            return cls()

    register_serializable(_Aliased, type_id="tests.LegacyAlias", alias=True)
    assert _Aliased.get_serialization_class_type_id() == "tests.CanonicalMain"

    blob_main = _pack_envelope(
        type_id=b"tests.CanonicalMain",
        payload=json.dumps({}).encode("utf-8"),
    )
    blob_alias = _pack_envelope(
        type_id=b"tests.LegacyAlias",
        payload=json.dumps({}).encode("utf-8"),
    )
    assert isinstance(Serializable.from_bytes(blob_main), _Aliased)
    assert isinstance(Serializable.from_bytes(blob_alias), _Aliased)


def test_register_serializable_rejects_two_classes_under_same_type_id() -> None:
    """Test registering two distinct classes under the same id raises."""

    @register_serializable(type_id="tests.DuplicateId")
    class _First(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_First":
            return cls()

    with pytest.raises(SerializationError):

        @register_serializable(type_id="tests.DuplicateId")
        class _Second(Serializable):
            def serialize_to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Second":
                return cls()


def test_register_serializable_rejects_canonical_id_conflict() -> None:
    """Test re-registering a class with a different canonical id raises.

    Existing id is built from string fragments so it isn't the same string
    object as the comparison target - value-equality must drive the check.
    """
    existing_id = "tests." + "ConstructedId" + "!"

    @register_serializable(type_id=existing_id)
    class _Conflicted(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Conflicted":
            return cls()

    other_id = "tests." + "ConstructedId" + "?"
    with pytest.raises(SerializationError):
        register_serializable(_Conflicted, type_id=other_id)


def test_register_serializable_accepts_matching_id_built_from_fragments() -> None:
    """Test re-registering with a value-equal but distinct-object id is a no-op."""
    canonical = "tests." + "MatchingId" + "$"

    @register_serializable(type_id=canonical)
    class _Match(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Match":
            return cls()

    register_serializable(_Match, type_id="tests." + "MatchingId" + "$")
    assert _Match.get_serialization_class_type_id() == canonical


# =============================================================================
# Serializable.serialize / deserialize format dispatch
# =============================================================================


@pytest.mark.parametrize(
    "obj_fixture",
    [
        pytest.param("dummy_span", id="dummy_span"),
        pytest.param("compact_span", id="compact_span"),
        pytest.param("custom_id_node", id="custom_id_node"),
    ],
)
@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param(SerializationFormat.DICT, id="dict"),
        pytest.param(SerializationFormat.JSON, id="json"),
        pytest.param(SerializationFormat.BINARY, id="binary"),
    ],
)
def test_serializable_round_trip_for_each_format(
    request: pytest.FixtureRequest, obj_fixture: str, fmt: SerializationFormat
) -> None:
    """Test `serialize` then `deserialize` round-trips the object in each format."""
    obj = request.getfixturevalue(obj_fixture)
    rebuilt = type(obj).deserialize(obj.serialize(fmt), fmt)
    assert rebuilt == obj


@pytest.mark.parametrize(
    "fmt, expected_types",
    [
        pytest.param(SerializationFormat.DICT, (dict,), id="dict"),
        pytest.param(SerializationFormat.JSON, (str,), id="json"),
        pytest.param(SerializationFormat.BINARY, (bytes, bytearray), id="binary"),
    ],
)
def test_serializable_serialize_returns_expected_payload_type(
    dummy_span: _DummySpan,
    fmt: SerializationFormat,
    expected_types: tuple[type, ...],
) -> None:
    """Test `serialize(fmt)` returns the appropriate payload type per format."""
    assert isinstance(dummy_span.serialize(fmt), expected_types)


def test_serializable_serialize_rejects_unknown_format(dummy_span: _DummySpan) -> None:
    """Test `serialize` rejects an unrecognized format value."""
    with pytest.raises(SerializationError, match="Unsupported format"):
        dummy_span.serialize("nope")  # type: ignore[arg-type]


def test_serializable_deserialize_rejects_unknown_format() -> None:
    """Test `deserialize` rejects an unrecognized format value."""
    with pytest.raises(SerializationError, match="Unsupported format"):
        _DummySpan.deserialize({"lo": 0, "hi": 1}, "nope")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fmt, bad_payload",
    [
        pytest.param(SerializationFormat.DICT, 123, id="dict"),
        pytest.param(SerializationFormat.JSON, 123, id="json"),
        pytest.param(SerializationFormat.BINARY, "not-bytes", id="binary"),
    ],
)
def test_serializable_deserialize_rejects_payload_with_wrong_type(
    fmt: SerializationFormat, bad_payload: Any
) -> None:
    """Test `deserialize` rejects payloads whose Python type is wrong."""
    with pytest.raises(SerializationPayloadTypeError):
        _DummySpan.deserialize(bad_payload, fmt)


@pytest.mark.parametrize(
    "fmt, payload_factory",
    [
        pytest.param(
            SerializationFormat.JSON,
            lambda obj: obj.to_json().encode("utf-8"),
            id="json_bytes",
        ),
        pytest.param(
            SerializationFormat.BINARY,
            lambda obj: bytearray(obj.to_bytes()),
            id="binary_bytearray",
        ),
        pytest.param(
            SerializationFormat.BINARY,
            lambda obj: memoryview(obj.to_bytes()),
            id="binary_memoryview",
        ),
    ],
)
def test_serializable_deserialize_accepts_alternate_payload_representations(
    dummy_span: _DummySpan,
    fmt: SerializationFormat,
    payload_factory: Callable[[Serializable], Any],
) -> None:
    """Test `deserialize` accepts non-canonical payload representations per format."""
    assert _DummySpan.deserialize(payload_factory(dummy_span), fmt) == dummy_span


def test_serializable_deserialize_binary_rejects_class_mismatch(
    dummy_span: _DummySpan,
) -> None:
    """Test `deserialize(BINARY)` raises when the payload class does not match."""
    with pytest.raises(SerializationError):
        _CustomIdNode.deserialize(dummy_span.to_bytes(), SerializationFormat.BINARY)


# =============================================================================
# to_json / from_json / from_bytes
# =============================================================================


def test_to_json_returns_sorted_keys_by_default(dummy_span: _DummySpan) -> None:
    """Test `to_json()` emits keys in alphabetical order by default."""
    payload = dummy_span.to_json()
    assert payload.index('"hi"') < payload.index('"lo"')


def test_to_json_respects_sort_keys_false() -> None:
    """Test `to_json(sort_keys=False)` does not enforce alphabetical order."""

    @dataclass(frozen=True)
    class _Ordered(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {"z_field": 1, "a_field": 2}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_Ordered":
            return cls()

    payload = _Ordered().to_json(sort_keys=False)
    assert payload.index('"z_field"') < payload.index('"a_field"')


def test_to_json_indent_emits_newlines(dummy_span: _DummySpan) -> None:
    """Test `to_json(indent=...)` produces a multi-line string."""
    assert "\n" in dummy_span.to_json(indent=2)


@pytest.mark.parametrize(
    "payload_factory",
    [
        pytest.param(lambda obj: obj.to_json(), id="str"),
        pytest.param(lambda obj: obj.to_json().encode("utf-8"), id="bytes"),
        pytest.param(
            lambda obj: bytearray(obj.to_json().encode("utf-8")), id="bytearray"
        ),
    ],
)
def test_from_json_round_trips_each_payload_representation(
    dummy_span: _DummySpan,
    payload_factory: Callable[[Serializable], Any],
) -> None:
    """Test `from_json` accepts each documented payload representation."""
    assert _DummySpan.from_json(payload_factory(dummy_span)) == dummy_span


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("[1, 2, 3]", id="array"),
        pytest.param('"a string"', id="scalar_string"),
    ],
)
def test_from_json_rejects_non_object_top_level(payload: str) -> None:
    """Test `from_json` rejects JSON whose top level is not an object."""
    with pytest.raises(SerializationError):
        _DummySpan.from_json(payload)


def test_from_bytes_default_does_not_import_unknown_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `from_bytes` defaults to *no* import fallback for unknown ids."""
    blob = _pack_envelope(
        type_id=b"fakepkg.fakemod._Outer",
        payload=json.dumps({}).encode(),
    )

    def fail_import(name: str) -> None:  # pragma: no cover - asserts on call
        raise AssertionError("import_module should not be called by default")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    with pytest.raises(UnknownTypeIdError):
        Serializable.from_bytes(blob)


def test_from_bytes_with_import_fallback_resolves_unregistered_class(
    unregistered_span: _UnregisteredSpan,
) -> None:
    """Test `from_bytes` with import fallback resolves an unregistered class."""
    rebuilt = Serializable.from_bytes(
        unregistered_span.to_bytes(),
        allow_import_fallback=True,
        allowed_module_prefixes=("tests.",),
    )
    assert rebuilt == unregistered_span


def test_from_bytes_import_fallback_emits_warning(
    unregistered_span: _UnregisteredSpan,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test import fallback path emits a WARNING audit record before importing."""
    with caplog.at_level(logging.WARNING, logger="fhy_core.serialization"):
        Serializable.from_bytes(
            unregistered_span.to_bytes(),
            allow_import_fallback=True,
            allowed_module_prefixes=("tests.",),
        )

    warning_records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "import fallback" in record.getMessage()
    ]
    assert warning_records, "expected WARNING record for import-fallback path"


@pytest.mark.parametrize(
    "type_id, prefixes, match",
    [
        pytest.param(
            _UnregisteredSpan.get_serialization_class_type_id().encode(),
            ("nope.",),
            "Refusing to import",
            id="module_outside_allowed_prefixes",
        ),
        pytest.param(
            b"json.decoder.JSONDecoder",
            ("json.",),
            "did not resolve",
            id="resolved_attribute_not_serializable",
        ),
        pytest.param(
            b"NoModule",
            ("fhy_core.",),
            "no module path",
            id="type_id_without_module_path",
        ),
        pytest.param(
            b"fhy_core.does_not_exist.Class",
            ("fhy_core.",),
            "Could not resolve",
            id="unimportable_module_path",
        ),
    ],
)
def test_from_bytes_fallback_rejects_unresolvable_type_id(
    type_id: bytes, prefixes: tuple[str, ...], match: str
) -> None:
    """Test `from_bytes(allow_import_fallback=True)` rejects unresolvable type ids."""
    blob = _pack_envelope(type_id=type_id, payload=json.dumps({}).encode())
    with pytest.raises(UnknownTypeIdError, match=match):
        Serializable.from_bytes(
            blob,
            allow_import_fallback=True,
            allowed_module_prefixes=prefixes,
        )


# =============================================================================
# Binary envelope contract
# =============================================================================


@pytest.mark.parametrize(
    "obj_fixture",
    [
        pytest.param("dummy_span", id="default_json_codec"),
        pytest.param("compact_span", id="custom_codec"),
    ],
)
def test_binary_envelope_carries_expected_fields_per_codec(
    request: pytest.FixtureRequest, obj_fixture: str
) -> None:
    """Test the binary envelope encodes magic, version, codec, type id, and payload."""
    obj = request.getfixturevalue(obj_fixture)
    magic, version, codec_u8, type_id, payload = _parse_envelope(obj.to_bytes())
    assert magic == b"FhYS"
    assert version == 1
    assert codec_u8 == _CODEC_TO_U8[type(obj).get_binary_codec()]
    assert type_id == type(obj).get_serialization_class_type_id()
    assert _decode_envelope_payload(payload, obj) == {"lo": obj.lo, "hi": obj.hi}


def test_binary_envelope_uses_explicitly_assigned_type_id(
    custom_id_node: _CustomIdNode,
) -> None:
    """Test the binary envelope embeds the canonical type id assigned via decorator."""
    _, _, _, type_id, _ = _parse_envelope(custom_id_node.to_bytes())
    assert type_id == "compiler_core.CustomIdNode"


@pytest.mark.parametrize(
    "blob_fixture, match",
    [
        pytest.param(
            "shorter_than_header_blob",
            "Data too short for header",
            id="shorter_than_header",
        ),
        pytest.param(
            "header_only_blob",
            "Data too short for type_id",
            id="header_only_missing_payload_length",
        ),
        pytest.param(
            "truncated_payload_blob",
            "Data too short for payload",
            id="truncated_within_payload",
        ),
    ],
)
def test_from_bytes_rejects_truncated_blob(
    request: pytest.FixtureRequest, blob_fixture: str, match: str
) -> None:
    """Test `from_bytes` rejects blobs truncated at each envelope boundary."""
    blob = request.getfixturevalue(blob_fixture)
    with pytest.raises(SerializationError, match=match):
        Serializable.from_bytes(blob)


@pytest.mark.parametrize(
    "blob_fixture, expected_error",
    [
        pytest.param("bad_magic_blob", SerializationError, id="bad_magic"),
        pytest.param(
            "unsupported_version_blob", VersionMismatchError, id="unsupported_version"
        ),
        pytest.param("unknown_codec_blob", UnknownCodecError, id="unknown_codec"),
        pytest.param("codec_mismatch_blob", CodecMismatchError, id="codec_mismatch"),
        pytest.param("unknown_type_id_blob", UnknownTypeIdError, id="unknown_type_id"),
    ],
)
def test_from_bytes_rejects_corrupted_blob(
    request: pytest.FixtureRequest,
    blob_fixture: str,
    expected_error: type[Exception],
) -> None:
    """Test `from_bytes` raises the expected error per envelope-corruption case."""
    blob = request.getfixturevalue(blob_fixture)
    with pytest.raises(expected_error):
        Serializable.from_bytes(blob)


def test_default_serialize_to_binary_rejects_non_json_codec(
    dummy_span: _DummySpan,
) -> None:
    """Test the default `serialize_to_binary` raises on a non-JSON codec."""
    with pytest.raises(CodecMismatchError):
        dummy_span.serialize_to_binary(codec=BinaryPayloadCodec.CUSTOM)


def test_default_serialize_to_binary_emits_sorted_json_keys(
    dummy_span: _DummySpan,
) -> None:
    """Test the default `serialize_to_binary` emits JSON keys in sorted order."""
    text = dummy_span.serialize_to_binary(codec=BinaryPayloadCodec.JSON).decode("utf-8")
    assert text.index('"hi"') < text.index('"lo"')


@pytest.mark.parametrize(
    "payload, codec, expected_error",
    [
        pytest.param(
            b"{}", BinaryPayloadCodec.CUSTOM, CodecMismatchError, id="codec_mismatch"
        ),
        pytest.param(
            json.dumps([1, 2, 3]).encode("utf-8"),
            BinaryPayloadCodec.JSON,
            SerializationError,
            id="non_dict_json_payload",
        ),
    ],
)
def test_default_deserialize_from_binary_rejects_invalid_inputs(
    payload: bytes, codec: BinaryPayloadCodec, expected_error: type[Exception]
) -> None:
    """Test the default `deserialize_from_binary` validates codec and payload."""
    with pytest.raises(expected_error):
        _DummySpan.deserialize_from_binary(payload, codec=codec)


def test_default_deserialize_from_binary_rejects_non_json_codec_match() -> None:
    """Test the default `deserialize_from_binary` raises for non-JSON-codec class.

    Reaches the trailing fallback in the default impl: the codec matches
    `get_binary_codec()` (so the mismatch guard passes) but is not JSON,
    and the class did not override the binary hooks.
    """

    @register_serializable(type_id="tests.NoBinaryOverride")
    class _NoBinaryOverride(Serializable):
        @classmethod
        def get_binary_codec(cls) -> BinaryPayloadCodec:
            return BinaryPayloadCodec.CUSTOM

        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_NoBinaryOverride":
            return cls()

    with pytest.raises(UnknownCodecError):
        _NoBinaryOverride.deserialize_from_binary(b"", codec=BinaryPayloadCodec.CUSTOM)


def test_dump_to_binary_rejects_unmappable_codec() -> None:
    """Test `to_bytes` raises `UnknownCodecError` when `get_binary_codec` is unmappable.

    Drives the branch via a public subclass override that returns a value
    outside the supported codec set.
    """

    @register_serializable(type_id="tests.UnmappedCodecSpan")
    class _UnmappedCodecSpan(Serializable):
        @classmethod
        def get_binary_codec(cls) -> BinaryPayloadCodec:
            return "not-a-real-codec"  # type: ignore[return-value]

        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_UnmappedCodecSpan":
            return cls()

    with pytest.raises(UnknownCodecError):
        _UnmappedCodecSpan().to_bytes()


def test_dump_to_binary_rejects_non_bytes_payload() -> None:
    """Test `to_bytes` raises when `serialize_to_binary` returns a non-bytes payload."""

    @register_serializable(type_id="tests.NonBytesPayloadSpan")
    class _NonBytesPayloadSpan(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(
            cls, data: Mapping[str, Any]
        ) -> "_NonBytesPayloadSpan":
            return cls()

        def serialize_to_binary(self, *, codec: BinaryPayloadCodec) -> bytes:
            return 12345  # type: ignore[return-value]

    with pytest.raises(SerializationError, match="bytes-like payload"):
        _NonBytesPayloadSpan().to_bytes()


def test_dump_to_binary_rejects_overlong_type_id() -> None:
    """Test `to_bytes` raises when the encoded `type_id` exceeds the 64KB limit."""
    huge_type_id = "tests." + ("x" * 70000)

    @register_serializable(type_id=huge_type_id)
    class _HugeIdSpan(Serializable):
        def serialize_to_dict(self) -> dict[str, Any]:
            return {}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_HugeIdSpan":
            return cls()

    with pytest.raises(SerializationError, match="type_id too long"):
        _HugeIdSpan().to_bytes()


# =============================================================================
# WrappedFamilySerializable
# =============================================================================


def test_wrapped_family_serialize_emits_type_and_data_envelope(
    custom_node: _CustomNode,
) -> None:
    """Test a `WrappedFamilySerializable` emits a `__type__` / `__data__` envelope."""
    assert custom_node.serialize_to_dict() == {
        "__type__": custom_node.get_serialization_class_type_id(),
        "__data__": {"value": custom_node.value},
    }


@pytest.mark.parametrize(
    "deserializer",
    [
        pytest.param(_CustomNodeBase, id="base_class"),
        pytest.param(_CustomNode, id="concrete_class"),
    ],
)
def test_wrapped_family_dict_round_trip_through_each_class(
    custom_node: _CustomNode, deserializer: type[_CustomNodeBase]
) -> None:
    """Test base and concrete classes both deserialize the wrapped envelope."""
    rebuilt = deserializer.deserialize_from_dict(custom_node.serialize_to_dict())
    assert isinstance(rebuilt, _CustomNode)
    assert rebuilt.value == custom_node.value


def test_wrapped_family_binary_round_trip(custom_node: _CustomNode) -> None:
    """Test a `WrappedFamilySerializable` round-trips through binary."""
    assert Serializable.from_bytes(custom_node.to_bytes()) == custom_node


def test_wrapped_family_accepts_alias_type_id() -> None:
    """Test a `WrappedFamilySerializable` accepts an alias in `__type__`."""
    register_serializable(_CustomNode, type_id="tests.LegacyCustomNode", alias=True)
    rebuilt = _CustomNodeBase.deserialize_from_dict(
        {"__type__": "tests.LegacyCustomNode", "__data__": {"value": 9}}
    )
    assert isinstance(rebuilt, _CustomNode)
    assert rebuilt.value == 9


@pytest.mark.parametrize(
    "payload, match",
    [
        pytest.param(
            {"__type__": 123, "__data__": {}},
            "Not a wrapped dict",
            id="non_string_type",
        ),
        pytest.param(
            {"__type__": "tests.X", "__data__": "not a dict"},
            "Not a wrapped dict",
            id="non_dict_data",
        ),
        pytest.param({"__data__": {}}, "Not a wrapped dict", id="missing_type"),
        pytest.param({"__type__": "tests.X"}, "Not a wrapped dict", id="missing_data"),
        pytest.param(
            {
                "__type__": _DummySpan.get_serialization_class_type_id(),
                "__data__": {"lo": 1, "hi": 2},
            },
            "not a subclass",
            id="resolved_class_outside_family",
        ),
    ],
)
def test_wrapped_family_rejects_invalid_envelope(
    payload: dict[str, Any], match: str
) -> None:
    """Test the base class rejects payloads that aren't a valid family envelope."""
    with pytest.raises(SerializationError, match=match):
        _CustomNodeBase.deserialize_from_dict(payload)


# =============================================================================
# Float precision + NaN/Infinity rejection
# =============================================================================


@register_serializable
@dataclass(frozen=True)
class _FloatBox(Serializable):
    """Serializable holding a single float field for round-trip tests."""

    value: float

    def serialize_to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "_FloatBox":
        return cls(value=float(data["value"]))


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        -1.0,
        3.141592653589793,
        2.718281828459045,
        1e-300,
        1e300,
        1.7976931348623157e308,
        5e-324,
    ],
)
def test_float_round_trips_exactly_through_json_binary_codec(value: float) -> None:
    """Test finite float values round-trip exactly through the JSON binary codec."""
    blob = _FloatBox(value).to_bytes()
    restored = Serializable.from_bytes(blob)
    assert isinstance(restored, _FloatBox)
    assert restored.value == value


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_to_bytes_rejects_nan_and_infinity(value: float) -> None:
    """Test ``to_bytes`` raises ``SerializationValueError`` for NaN/Infinity."""
    with pytest.raises(SerializationValueError, match="JSON-finite"):
        _FloatBox(value).to_bytes()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_to_json_rejects_nan_and_infinity(value: float) -> None:
    """Test ``to_json`` rejects NaN/Infinity with a ``SerializationValueError``."""
    with pytest.raises(SerializationValueError, match="JSON-finite"):
        _FloatBox(value).to_json()
