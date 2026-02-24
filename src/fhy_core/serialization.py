"""Serialization trait module.

This module defines the serialization trait and utilities used across the FhY
compiler stack. It provides:

- A runtime-enforced base class (`Serializable`) that compiler components
    can inherit from to support consistent serialization and deserialization.

- A small set of supported serialization formats (`SerializationFormat`):
    - DICT:   A JSON-friendly Python mapping (useful for tests and debugging).
    - JSON:   UTF-8 JSON text of the DICT representation.
    - BINARY: A self-describing binary envelope suitable for file storage and
        round-tripping objects back into their concrete Python classes.

- A class-family serialization pattern via `WrappedFamilySerializable` that
    simplifies serialization of class hierarchies (e.g., AST nodes) by embedding
    type information in the dict form.

Binary format
-------------
Binary serialization uses a compact envelope:

    `MAGIC(4)` | `VERSION(u8)` | `CODEC(u8)` | `type_id_len(u16)` | `type_id(bytes)`
             | `payload_len(u32)` | `payload(bytes)`

- `MAGIC` identifies the blob as produced by this module.
- `VERSION` allows future evolution of the envelope format.
- `CODEC` identifies how the payload bytes are encoded (e.g. JSON-bytes, custom).
- `type_id` is a stable identifier for the concrete class being serialized.
  By default, it is the fully-qualified name: "<module>.<qualname>".
- `payload` is the encoded representation of the object, determined by CODEC.

Default behavior:
- CODEC=JSON and payload = UTF-8 JSON encoding of `serialize_to_dict()`.

Custom behavior:
- Classes may override `get_binary_codec()`, `serialize_to_binary()`,
  and `deserialize_from_binary()` to implement compact class-specific payloads.

Because type_id is embedded in the envelope, deserialization can reconstruct
the correct class automatically:
    obj = Serializable.from_bytes(blob)

Type registration
-----------------
For deterministic and controlled reconstruction, classes can be registered
into a local registry via the `@register_serializable` decorator.

At decode time, the registry is consulted first. This avoids importing
arbitrary modules during deserialization and keeps reconstruction explicit.
As a convenience, the module also supports a fallback that resolves classes
by importing the module portion of the `type_id`.
"""

from __future__ import annotations

__all__ = [
    "Serializable",
    "SerializationFormat",
    "BinaryPayloadCodec",
    "register_serializable",
    "WrappedFamilySerializable",
    "SerializedDict",
    "SerializedValue",
    "SerializedObject",
    "is_serialized_value",
    "is_serialized_dict",
    "InvalidSerializationDictStructureError",
    "InvalidSerializationDataValueError",
]

import importlib
import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    ClassVar,
    Mapping,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)

from frozendict import frozendict

from .error import register_error
from .utils import StrEnum, format_comma_separated_list

SerializedValue: TypeAlias = Union[
    str, int, float, bool, None, Sequence["SerializedValue"], "SerializedDict"
]
SerializedDict: TypeAlias = dict[str, SerializedValue]
SerializedObject: TypeAlias = SerializedDict | str | bytes


def is_serialized_value(v: Any) -> TypeGuard[SerializedValue]:
    """Return if `v` is a valid `SerializedValue`."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, (bytes, bytearray, memoryview)):
        return False
    if isinstance(v, Mapping):
        return is_serialized_dict(v)
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return all(is_serialized_value(x) for x in v)
    return False


def is_serialized_dict(v: Any) -> TypeGuard[SerializedDict]:
    """Return if `v` is a mapping with `str` keys and `SerializedValue` values."""
    if not isinstance(v, Mapping):
        return False
    for k, val in v.items():
        if not isinstance(k, str):
            return False
        if not is_serialized_value(val):
            return False
    return True


_T = TypeVar("_T", bound="Serializable")


class SerializationFormat(StrEnum):
    """Supported formats (logical formats, not file extensions)."""

    DICT = "dict"
    JSON = "json"
    BINARY = "binary"


class BinaryPayloadCodec(StrEnum):
    """Payload encoding used inside the binary envelope."""

    JSON = "json"
    CUSTOM = "custom"


_CODEC_TO_U8: frozendict[BinaryPayloadCodec, int] = frozendict(
    {
        BinaryPayloadCodec.JSON: 1,
        BinaryPayloadCodec.CUSTOM: 2,
    }
)
_U8_TO_CODEC: frozendict[int, BinaryPayloadCodec] = frozendict(
    {v: k for k, v in _CODEC_TO_U8.items()}
)


class SerializationError(Exception):
    """Base error for serialization failures."""


@register_error
class InvalidSerializationDictStructureError(SerializationError):
    """Raised when dictionary provided for deserialization has an invalid structure."""

    def __init__(  # type: ignore[no-untyped-def]
        self, class_type: type, expected_structure, actual_data: SerializedDict
    ):
        expected_fields_str = format_comma_separated_list(
            f'"{fld}": {ty}' for fld, ty in expected_structure.__annotations__.items()
        )
        actual_fields_str = format_comma_separated_list(
            f'"{fld}": {type(val).__name__}' for fld, val in actual_data.items()
        )
        super().__init__(
            f'Invalid dictionary structure for deserializing "{class_type.__name__}". '
            f"Expected fields: {{{expected_fields_str}}}. "
            f"Actual fields: {{{actual_fields_str}}}."
        )


@register_error
class InvalidSerializationDataValueError(SerializationError, ValueError):
    """Raised when data provided for deserialization has invalid values."""

    def __init__(
        self,
        class_type: type,
        field_name: str,
        expected_description: str,
        actual_value: Any,
    ):
        super().__init__(
            f'Invalid value for field "{field_name}" while deserializing '
            f'"{class_type.__name__}". Expected: {expected_description}. '
            f"Actual: {repr(actual_value)}."
        )


@register_error
class UnknownTypeIdError(SerializationError):
    """Raised when a type_id cannot be resolved to a class."""


@register_error
class VersionMismatchError(SerializationError):
    """Raised when the binary envelope version is unsupported."""


@register_error
class UnknownCodecError(SerializationError):
    """Raised when a binary payload codec code is unknown/unsupported."""


@register_error
class CodecMismatchError(SerializationError):
    """Raised when the envelope codec does not match the expected codec for a class."""


_TYPE_REGISTRY: dict[str, type["Serializable"]] = {}


def _get_default_type_id(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_type_id(
    type_id: str,
    *,
    allow_import_fallback: bool = False,
    allowed_module_prefixes: tuple[str, ...] = ("fhy_core.",),
) -> type["Serializable"]:
    if type_id in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[type_id]

    if not allow_import_fallback:
        raise UnknownTypeIdError(
            f'Type "{type_id}" is not registered. Import fallback is disabled.'
        )

    module_name, _, qual = type_id.rpartition(".")
    if not module_name:
        raise UnknownTypeIdError(f'Invalid type_id "{type_id}" (no module path).')

    if allowed_module_prefixes and not module_name.startswith(allowed_module_prefixes):
        raise UnknownTypeIdError(
            f'Refusing to import "{module_name}" while resolving "{type_id}". '
            f"Allowed prefixes: {allowed_module_prefixes}"
        )

    try:
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in qual.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, type) or not issubclass(obj, Serializable):
            raise UnknownTypeIdError(
                f'type_id "{type_id}" did not resolve to a serializable class.'
            )
        return obj
    except Exception as e:
        raise UnknownTypeIdError(f'Could not resolve type_id "{type_id}": {e}') from e


def register_serializable(
    cls: type[_T] | None = None,
    *,
    type_id: str | None = None,
    alias: bool = False,
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    """Decorator to register a Serializable class under a type_id.

    Args:
        cls: The class to register. Can be provided directly or via a decorator.
        type_id: The type id to register under. If omitted, uses the class' existing
            `_SERIALIZATION_CLASS_TYPE_ID` if present.
        alias: If True, register `type_id` as an additional (legacy) id for the class
            without changing the class' canonical `_SERIALIZATION_CLASS_TYPE_ID`.
            If False (default), `type_id` is treated as the canonical id: it will be
            set on the class (if not already set) and must match if already set.
    """

    def _wrapper(c: type[_T]) -> type[_T]:
        local_existing = c.__dict__.get("_SERIALIZATION_CLASS_TYPE_ID", None)

        if type_id is None:
            ty_id = local_existing or _get_default_type_id(c)
        else:
            ty_id = type_id

        if ty_id in _TYPE_REGISTRY and _TYPE_REGISTRY[ty_id] is not c:
            raise SerializationError(
                f'Duplicate registration for type_id "{ty_id}": '
                f"{_TYPE_REGISTRY[ty_id]} already registered; refusing to override."
            )

        if type_id is not None and not alias:
            if local_existing is None:
                c._SERIALIZATION_CLASS_TYPE_ID = type_id
            elif local_existing != type_id:
                raise SerializationError(
                    f"Class {c} already has "
                    f'_SERIALIZATION_CLASS_TYPE_ID="{local_existing}", '
                    f'cannot register with different type_id="{type_id}".'
                )

        _TYPE_REGISTRY[ty_id] = c
        return c

    return _wrapper(cls) if cls is not None else _wrapper


_MAGIC: bytes = b"FhYS"
_VERSION: int = 1
# MAGIC(4) | VERSION(u8) | CODEC(u8) | type_id_len(u16)
_HEADER_STRUCT = struct.Struct("!4sBBH")
_PAYLOAD_LEN_STRUCT = struct.Struct("!I")


def _dump_to_binary(obj: "Serializable") -> bytes:
    type_id = obj.get_serialization_class_type_id().encode("utf-8")

    codec = obj.get_binary_codec()
    try:
        codec_u8 = _CODEC_TO_U8[codec]
    except KeyError as e:
        raise UnknownCodecError(f'Unsupported binary codec "{codec}".') from e

    payload = obj.serialize_to_binary(codec=codec)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise SerializationError(
            "serialize_to_binary() must return bytes-like payload."
        )

    payload_bytes = bytes(payload)

    if len(type_id) > 65535:  # noqa: PLR2004
        raise SerializationError("type_id too long (>65535 bytes).")
    if len(payload_bytes) > 0xFFFFFFFF:  # noqa: PLR2004
        raise SerializationError("payload too large (>4GB).")

    header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, codec_u8, len(type_id))
    payload_len = _PAYLOAD_LEN_STRUCT.pack(len(payload_bytes))
    return header + type_id + payload_len + payload_bytes


def _loads_from_binary(
    data: bytes | bytearray | memoryview,
    *,
    allow_import_fallback: bool = False,
    allowed_module_prefixes: tuple[str, ...] = ("fhy_core.",),
) -> "Serializable":
    mv = memoryview(data)
    if len(mv) < _HEADER_STRUCT.size:
        raise SerializationError("Data too short for header.")

    magic, version, codec_u8, type_id_len = _HEADER_STRUCT.unpack(
        mv[: _HEADER_STRUCT.size]
    )
    if magic != _MAGIC:
        raise SerializationError(
            "Bad magic header; not a recognized serialization blob."
        )
    if version != _VERSION:
        raise VersionMismatchError(
            f"Unsupported version {version}; expected {_VERSION}."
        )

    try:
        codec = _U8_TO_CODEC[codec_u8]
    except KeyError as e:
        raise UnknownCodecError(f"Unknown codec code {codec_u8}.") from e

    offset = _HEADER_STRUCT.size
    end_type = offset + type_id_len
    if len(mv) < end_type + _PAYLOAD_LEN_STRUCT.size:
        raise SerializationError("Data too short for type_id and payload length.")

    type_id = mv[offset:end_type].tobytes().decode("utf-8")
    offset = end_type

    (payload_len,) = _PAYLOAD_LEN_STRUCT.unpack(
        mv[offset : offset + _PAYLOAD_LEN_STRUCT.size]
    )
    offset += _PAYLOAD_LEN_STRUCT.size

    end_payload = offset + payload_len
    if len(mv) < end_payload:
        raise SerializationError("Data too short for payload.")

    payload_bytes = mv[offset:end_payload].tobytes()

    cls = _resolve_type_id(
        type_id,
        allow_import_fallback=allow_import_fallback,
        allowed_module_prefixes=allowed_module_prefixes,
    )
    return cls.deserialize_from_binary(payload_bytes, codec=codec)


class Serializable(ABC):
    """Serialization trait for compiler objects.

    Required:
      - serialize_to_dict()
      - deserialize_from_dict()

    Optional:
      - get_binary_codec()
      - serialize_to_binary()
      - deserialize_from_binary()

    Notes:
    - The default binary codec stores JSON bytes of the dict form.
    - Override the binary hooks to define a compact canonical binary form.

    """

    _SERIALIZATION_CLASS_TYPE_ID: ClassVar[str | None] = None

    @classmethod
    def get_serialization_class_type_id(cls) -> str:
        """Return the type_id for this class; used for serialization."""
        local = cls.__dict__.get("_SERIALIZATION_CLASS_TYPE_ID", None)
        return local or _get_default_type_id(cls)

    @abstractmethod
    def serialize_to_dict(self) -> SerializedDict:
        """Return a dictionary representation of this object for serialization."""

    @classmethod
    @abstractmethod
    def deserialize_from_dict(cls: type[_T], data: SerializedDict) -> _T:
        """Create an instance of this class from a dictionary representation."""

    @classmethod
    def get_binary_codec(cls) -> BinaryPayloadCodec:
        """Return the payload codec used for this class in BINARY format.

        Default: JSON (payload is JSON bytes of serialize_to_dict()).
        """
        return BinaryPayloadCodec.JSON

    def serialize_to_binary(self, *, codec: BinaryPayloadCodec) -> bytes:
        """Encode this object into payload bytes according to `codec`.

        Default implementation supports JSON only. Classes that return CUSTOM
        from get_binary_codec() should override this to emit compact bytes.

        Args:
            codec: Codec selected for this object's envelope.

        Returns:
            Bytes payload to store inside the binary envelope.

        Raises:
            CodecMismatchError: If the provided codec does not match the expected codec.

        """
        if codec is not BinaryPayloadCodec.JSON:
            raise CodecMismatchError(
                f'Class "{type(self)}" does not implement binary codec "{codec}".'
            )
        payload_dict = self.serialize_to_dict()
        return json.dumps(payload_dict, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    @classmethod
    def deserialize_from_binary(
        cls: type[_T],
        payload: bytes,
        *,
        codec: BinaryPayloadCodec,
    ) -> _T:
        """Decode payload bytes into an instance of this class.

        Default implementation supports JSON only (payload is JSON bytes of dict form).
        Override for CUSTOM/PROTOBUF/etc.

        Args:
            payload: Payload bytes from the envelope.
            codec: Codec stored in the envelope.

        Returns:
            Reconstructed instance of this class.

        Raises:
            CodecMismatchError: If the provided codec does not match the expected codec.
            SerializationError: If the payload cannot be decoded properly.

        """
        expected = cls.get_binary_codec()
        if codec is not expected:
            raise CodecMismatchError(
                f'Codec mismatch for "{cls}": envelope="{codec}" expected="{expected}".'
            )

        if codec is BinaryPayloadCodec.JSON:
            payload_obj = json.loads(payload.decode("utf-8"))
            if not is_serialized_dict(payload_obj):
                raise SerializationError("Payload did not decode to an object/dict.")
            return cls.deserialize_from_dict(payload_obj)

        raise UnknownCodecError(f'Unsupported codec "{codec}".')

    def serialize(self, fmt: SerializationFormat) -> SerializedObject:
        """Serialize this object to the specified format.

        Args:
            fmt: The serialization format to use.

        Returns:
            The serialized representation of this object in the specified format.

        Raises:
            SerializationError: If the format is unsupported or if serialization fails.

        """
        if fmt is SerializationFormat.DICT:
            return self.serialize_to_dict()
        elif fmt is SerializationFormat.JSON:
            return self.to_json()
        elif fmt is SerializationFormat.BINARY:
            return self.to_bytes()
        else:
            raise SerializationError(f"Unsupported format: {fmt}")

    @classmethod
    def deserialize(
        cls: type[_T], payload: SerializedObject, fmt: SerializationFormat
    ) -> _T:
        """Deserialize an object of this class from the given payload and format.

        Args:
            payload: The serialized representation of the object.
            fmt: The format of the serialized payload.

        Returns:
            An instance of this class reconstructed from the payload.

        Raises:
            SerializationError: If the format is unsupported or deserialization fails.

        """
        if fmt is SerializationFormat.DICT:
            if not is_serialized_dict(payload):
                raise SerializationError(
                    "DICT deserialization requires a dictionary payload."
                )
            return cls.deserialize_from_dict(payload)
        elif fmt is SerializationFormat.JSON:
            if not isinstance(payload, (str, bytes, bytearray)):
                raise SerializationError(
                    "JSON deserialization requires str/bytes payload."
                )
            return cls.from_json(payload)
        elif fmt is SerializationFormat.BINARY:
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise SerializationError(
                    "BINARY deserialization requires bytes-like payload."
                )
            obj = _loads_from_binary(payload)
            if not isinstance(obj, cls):
                raise SerializationError(
                    f'Binary payload produced "{type(obj)}"; expected "{cls}".'
                )
            return obj
        else:
            raise SerializationError(f"Unsupported format: {fmt}")

    def to_json(self, *, indent: int | None = None, sort_keys: bool = True) -> str:
        """Serialize this object to a JSON string.

        Args:
            indent: If specified, the JSON string is formatted with this indent level.
            sort_keys: Whether to sort the keys in the JSON output.

        Returns:
            A JSON string representation of this object.

        """
        return json.dumps(self.serialize_to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_json(cls: type[_T], payload: str | bytes | bytearray) -> _T:
        """Deserialize an object of this class from a JSON string or bytes.

        Args:
            payload: A JSON string or bytes representing the object.

        Returns:
            An instance of this class reconstructed from the JSON payload.

        Raises:
            SerializationError: If the payload cannot be decoded properly.

        """
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        payload_obj = json.loads(payload)
        if not is_serialized_dict(payload_obj):
            raise SerializationError("JSON did not decode to an object/dict.")
        return cls.deserialize_from_dict(payload_obj)

    def to_bytes(self) -> bytes:
        """Return a binary serialization of this object."""
        return _dump_to_binary(self)

    @staticmethod
    def from_bytes(
        data: bytes | bytearray | memoryview,
        *,
        allow_import_fallback: bool = False,
        allowed_module_prefixes: tuple[str, ...] = ("fhy_core.",),
    ) -> "Serializable":
        """Deserialize a Serializable object from binary data.

        Args:
            data: The binary data containing the serialized object.
            allow_import_fallback: Whether to allow importing classes from
                external modules.
            allowed_module_prefixes: A tuple of allowed module prefixes for
                import fallback.

        Returns:
            The deserialized Serializable object.

        """
        return _loads_from_binary(
            data,
            allow_import_fallback=allow_import_fallback,
            allowed_module_prefixes=allowed_module_prefixes,
        )


_F = TypeVar("_F", bound="WrappedFamilySerializable")


class WrappedFamilySerializable(Serializable, ABC):
    """Serializable base for class families (e.g., AST nodes).

    Pattern:
      - Base class implements serialize_to_dict() to emit a wrapped dict:
        ```
          {"__type__": <type_id>, "__data__": <data_dict>}
        ```

      - Subclasses implement only the data portion:
          - `serialize_data_to_dict()`
          - `deserialize_data_from_dict()`

      - Base class implements deserialize_from_dict() that:
          - reads `__type__`/`__data__`
          - resolves the concrete subclass from `__type__`
          - calls `subclass.deserialize_data_from_dict(__data__)`

    Example usage:
    ```
    class BaseNode(WrappedFamilySerializable):
        pass

    @register_serializable
    class NodeA(BaseNode):
        value: int

        def serialize_data_to_dict(self):
            return {"value": self.value}

        @classmethod
        def deserialize_data_from_dict(cls, data):
            return cls(value=data["value"])

    """

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "__type__": self.get_serialization_class_type_id(),
            "__data__": self.serialize_data_to_dict(),
        }

    @classmethod
    def deserialize_from_dict(cls: type[_F], data: SerializedDict) -> _F:
        class_type_id = data.get("__type__")
        object_data = data.get("__data__")
        if not isinstance(class_type_id, str) or not is_serialized_dict(object_data):
            raise SerializationError("Not a wrapped dict with __type__ and __data__.")

        concrete_class = _resolve_type_id(class_type_id)

        if not issubclass(concrete_class, cls):
            raise SerializationError(
                f'Wrapped type "{concrete_class}" is not a subclass of expected '
                f'family "{cls}".'
            )

        return concrete_class.deserialize_data_from_dict(object_data)

    @abstractmethod
    def serialize_data_to_dict(self) -> SerializedDict:
        """Serialize only this class's data.

        Returns:
            A dict representing only this class's data (no type wrapper).

        """

    @classmethod
    @abstractmethod
    def deserialize_data_from_dict(cls: type[_F], data: SerializedDict) -> _F:
        """Deserialize only this class's data.

        Args:
            data: A dict representing only this class's data (no type wrapper).

        Returns:
            An instance of this class reconstructed from the data dict.

        """
