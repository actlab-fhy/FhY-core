"""Serialization trait module.

This module defines the serialization trait and utilities used across the FhY
compiler stack. It provides:

- A runtime-enforced base class (`Serializable`) that compiler components
   can inherit from to support consistent serialization and deserialization.

- A small set of supported serialization formats (`SerializationFormat`):
   - DICT:  A JSON-friendly Python mapping (useful for tests and debugging).
   - JSON:  UTF-8 JSON text of the DICT representation.
   - BINARY: A self-describing binary envelope suitable for file storage and
            round-tripping objects back into their concrete Python classes.

Binary format
-------------
Binary serialization uses a compact envelope:

    `MAGIC(4)` | `VERSION(u8)` | `type_id_len(u16)` | `type_id(bytes)`
             | `payload_len(u32)` | `payload(bytes)`

- `MAGIC` identifies the blob as produced by this module.
- `VERSION` allows future evolution of the format.
- `type_id` is a stable identifier for the concrete class being serialized.
  By default, it is the fully-qualified name: "<module>.<qualname>".
- `payload` is the JSON encoding (UTF-8) of the object's DICT representation.

Because type_id is embedded in the envelope, deserialization can reconstruct
the correct class automatically:
```
    obj = Serializable.from_bytes(blob)
```

Type registration
-----------------
For deterministic and controlled reconstruction, classes can be registered
into a local registry via the `@register_serializable` decorator:

```
    @register_serializable
    class Span(Serializable):
        ...
```

At decode time, the registry is consulted first. This avoids importing
arbitrary modules during deserialization and keeps reconstruction explicit.
As a convenience, the module also supports a fallback that resolves classes
by importing the module portion of the `type_id` (this is useful in simple
setups, but explicit registration is recommended).

Usage sketch
------------
```
    @register_serializable
    @dataclass(frozen=True)
    class Span(Serializable):
        lo: int
        hi: int

        def serialize_to_dict(self) -> dict[str, Any]:
            return {"lo": self.lo, "hi": self.hi}

        @classmethod
        def deserialize_from_dict(cls, data: Mapping[str, Any]) -> "Span":
            return cls(int(data["lo"]), int(data["hi"]))

    span = Span(1, 9)

    # Dict / JSON
    dictionary = span.serialize(SerializationFormat.DICT)
    json_str = span.serialize(SerializationFormat.JSON)

    # Binary round-trip with automatic reconstruction
    binary = span.serialize(SerializationFormat.BINARY)
    span2 = Serializable.from_bytes(binary)
    assert span2 == span
```

"""

__all__ = [
    "Serializable",
    "register_serializable",
    "get_wrapper_dict",
    "unwrap_wrapper_dict",
]

import importlib
import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, ClassVar, TypeVar

from .error import register_error
from .utils import StrEnum

_T = TypeVar("_T", bound="Serializable")


class SerializationFormat(StrEnum):
    """Supported formats (logical formats, not file extensions)."""

    DICT = "dict"
    JSON = "json"
    BINARY = "binary"


@register_error
class SerializationError(Exception):
    """Base error for serialization failures."""


@register_error
class UnknownTypeIdError(SerializationError):
    """Raised when a type_id cannot be resolved to a class."""


@register_error
class VersionMismatchError(SerializationError):
    """Raised when the binary envelope version is unsupported."""


_TYPE_REGISTRY: dict[str, type["Serializable"]] = {}


def _get_default_type_id(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_type_id(type_id: str) -> type["Serializable"]:
    if type_id in _TYPE_REGISTRY:
        return _TYPE_REGISTRY[type_id]

    try:
        module_name, _, qual = type_id.rpartition(".")
        if not module_name:
            raise UnknownTypeIdError(f'Invalid type_id "{type_id}" (no module path).')
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in qual.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, type) or not issubclass(obj, Serializable):
            raise UnknownTypeIdError(
                f'type_id "{type_id}" did not resolve to a Serializable class.'
            )
        return obj
    except Exception as e:
        raise UnknownTypeIdError(f'Could not resolve type_id "{type_id}": {e}') from e


def register_serializable(
    cls: type[_T] | None = None,
    *,
    type_id: str | None = None,
) -> type[_T] | Callable[[type[_T]], type[_T]]:
    """Decorator to register a Serializable class under a type_id.

    Usage:
    ```
        @register_serializable
        class Foo(Serializable): ...
    ```
    or:
    ```
        @register_serializable(type_id="core.Foo")
        class Foo(Serializable): ...
    ```

    """

    def _wrapper(c: type[_T]) -> type[_T]:
        ty_id = type_id or _get_default_type_id(c)
        _TYPE_REGISTRY[ty_id] = c
        return c

    return _wrapper(cls) if cls is not None else _wrapper


_MAGIC: bytes = b"FhYS"
_VERSION: int = 1
_HEADER_STRUCT = struct.Struct("!4sBH")
_PAYLOAD_LEN_STRUCT = struct.Struct("!I")


def _dump_to_binary(obj: "Serializable") -> bytes:
    type_id = obj.get_class_type_id().encode("utf-8")
    payload_dict = obj.serialize_to_dict()

    payload = json.dumps(payload_dict, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )

    if len(type_id) > 65535:  # noqa: PLR2004
        raise SerializationError("type_id too long (>65535 bytes).")
    if len(payload) > 0xFFFFFFFF:  # noqa: PLR2004
        raise SerializationError("payload too large (>4GB).")

    header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, len(type_id))
    payload_len = _PAYLOAD_LEN_STRUCT.pack(len(payload))
    return header + type_id + payload_len + payload


def _loads_from_binary(data: bytes | bytearray | memoryview) -> "Serializable":
    mv = memoryview(data)
    if len(mv) < _HEADER_STRUCT.size:
        raise SerializationError("Data too short for header.")

    magic, version, type_id_len = _HEADER_STRUCT.unpack(mv[: _HEADER_STRUCT.size])
    if magic != _MAGIC:
        raise SerializationError(
            "Bad magic header; not a recognized serialization blob."
        )
    if version != _VERSION:
        raise VersionMismatchError(
            f"Unsupported version {version}; expected {_VERSION}."
        )

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
    payload_obj = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload_obj, Mapping):
        raise SerializationError("Payload did not decode to an object/dict.")

    cls = _resolve_type_id(type_id)
    return cls.deserialize_from_dict(payload_obj)


class Serializable(ABC):
    """Serialization trait for compiler objects.

    Implementors MUST define:
      - to_serializable_dict()
      - from_serializable_dict()

    Best practice: keep the dict JSON-friendly (str/int/float/bool/None,
    lists, dicts), and only store stable schema keys.
    """

    TYPE_ID: ClassVar[str | None] = None

    @classmethod
    def get_class_type_id(cls) -> str:
        """Return the type_id for this class; used for serialization."""
        return cls.TYPE_ID or _get_default_type_id(cls)

    @abstractmethod
    def serialize_to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of this object for serialization."""

    @classmethod
    @abstractmethod
    def deserialize_from_dict(cls: type[_T], data: Mapping[str, Any]) -> _T:
        """Create an instance of this class from a dictionary representation.

        Args:
            data: Mapping containing the serialized data for this object.

        Returns:
            Instance of this class reconstructed from the provided data.

        """

    def serialize(self, fmt: SerializationFormat) -> Any:
        """Serialize this object to the specified format.

        Args:
            fmt: Desired serialization format (DICT, JSON, BINARY).

        Returns:
            Serialized representation of this object in the specified format.

        Raises:
            SerializationError: If the specified format is unsupported.

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
    def deserialize(cls: type[_T], payload: Any, fmt: SerializationFormat) -> _T:
        """Deserialize an object of this class from the given payload and format.

        Args:
            payload: Data to deserialize, whose type depends on the format.
            fmt: Format of the payload (DICT, JSON, BINARY).

        Returns:
            Instance of this class reconstructed from the provided payload.

        Raises:
            SerializationError: If the specified format is unsupported or if the
                payload type is incompatible with the expected format.

        """
        if fmt is SerializationFormat.DICT:
            if not isinstance(payload, Mapping):
                raise SerializationError(
                    "DICT deserialization requires a Mapping payload."
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
            indent: If specified, the number of spaces to use for indentation in the
                resulting JSON string. If None, the most compact representation is used.
            sort_keys: Whether to sort the keys in the resulting JSON string.

        Returns:
            JSON string representation of this object.

        """
        return json.dumps(self.serialize_to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_json(cls: type[_T], payload: str | bytes | bytearray) -> _T:
        """Deserialize an object of this class from a JSON string or bytes.

        Args:
            payload: JSON string or bytes containing the serialized
                representation of the object.

        Returns:
            Instance of this class reconstructed from the provided payload.

        Raises:
            SerializationError: If the payload is not valid JSON or if it does
                not represent an object of this class.

        """
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        payload = json.loads(payload)
        if not isinstance(payload, Mapping):
            raise SerializationError("JSON did not decode to an object/dict.")
        return cls.deserialize_from_dict(payload)

    def to_bytes(self) -> bytes:
        """Return a binary serialization of this object.."""
        return _dump_to_binary(self)

    @staticmethod
    def from_bytes(data: bytes | bytearray | memoryview) -> "Serializable":
        """Deserialize a Serializable object from binary data.

        Args:
            data: Binary data containing the serialized representation of the object.

        Returns:
            Instance of a Serializable object reconstructed from the provided data.

        """
        return _loads_from_binary(data)

    def _dataclass_default_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):
            raise SerializationError(
                'Not a dataclass; implement "to_serializable_dict".'
            )
        return asdict(self)


def get_wrapper_dict(obj: Serializable) -> dict[str, Any]:
    """Return a dict containing type information and the dict representation.

    If you sometimes want a self-describing dict for JSON, use this wrapper:
      {"__type__": "...", "__data__": {...}}

    """
    return {"__type__": obj.get_class_type_id(), "__data__": obj.serialize_to_dict()}


def unwrap_wrapper_dict(data: Mapping[str, Any]) -> Serializable:
    """Return the Serializable object represented by a wrapper dict."""
    t = data.get("__type__")
    object_data = data.get("__data__")
    if not isinstance(t, str) or not isinstance(object_data, Mapping):
        raise SerializationError("Not a wrapped dict with __type__ and __data__.")
    cls = _resolve_type_id(t)
    return cls.deserialize_from_dict(object_data)
