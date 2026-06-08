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

Security
--------
Default deserialization paths are safe against untrusted input: every
``type_id`` is looked up only in the in-process registry, and unknown ids
raise ``UnknownTypeIdError`` rather than triggering imports.

``Serializable.from_bytes(..., allow_import_fallback=True)`` opts into
``importlib.import_module`` for ``type_id`` lookups, which executes the
named module's top-level code. The ``allowed_module_prefixes`` parameter
(default ``("fhy_core.",)``) constrains which modules may be loaded, but
even within that prefix, importing has side effects (registry
registrations, logging configuration, etc.). Do **not** enable
``allow_import_fallback`` with blobs whose origin you do not control or
trust.

Float and special-value contract
--------------------------------
Float values round-trip exactly within Python: ``json.dumps`` emits the
shortest round-trip-safe ``repr`` and ``json.loads`` recovers the same
``float``. Cross-language consumers (non-Python JSON parsers) may see
different precision because of their own float-to-string conversion
rules. Applications requiring bit-for-bit cross-runtime round-tripping
should define a non-JSON binary codec by overriding ``get_binary_codec``
and ``serialize_to_binary`` / ``deserialize_from_binary`` on the
relevant class.

``NaN`` and ``+/-Infinity`` are rejected at serialization time: both
``serialize_to_binary`` (JSON codec) and ``to_json`` pass
``allow_nan=False`` to ``json.dumps`` and re-raise the resulting
``ValueError`` as ``SerializationValueError``. This prevents accidentally
emitting non-standard ``"NaN"`` / ``"Infinity"`` tokens that strict
third-party JSON parsers would reject. Applications that need
NaN-tolerant serialization must use a non-JSON codec.

``NaN`` compares unequal to itself by IEEE-754 rule, so two
``LiteralExpression`` values wrapping ``NaN`` are not structurally
equivalent. That is the standard IEEE-754 consequence, not an
``fhy_core`` quirk.
"""

__all__ = [
    "Serializable",
    "SerializationFormat",
    "BinaryPayloadCodec",
    "register_serializable",
    "WrappedFamilySerializable",
    "FieldCodec",
    "register_field_codec",
    "make_field_codec",
    "make_enum_field_codec",
    "make_labeled_enum_field_codec",
    "SerializationDerivationError",
    "SerializedDict",
    "SerializedValue",
    "SerializedObject",
    "is_serialized_value",
    "is_serialized_dict",
    "DeserializationDictStructureError",
    "DeserializationValueError",
    "SerializationTypeError",
    "SerializationValueError",
    "SerializationPayloadTypeError",
    "RegistryWrappedValueLeaf",
    "RegistryWrappedValue",
    "is_registry_wrapped_value_leaf",
    "is_registry_wrapped_value",
    "deserialize_registry_wrapped_value",
    "serialize_registry_wrapped_value",
]

import dataclasses
import enum
import importlib
import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath
from pprint import pformat
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from frozendict import frozendict

from .error import register_error
from .logger import get_logger
from .utils.enum import StrEnum
from .utils.type_hint_utils import (
    get_origin_and_arguments,
    resolve_field_annotations,
    split_optional,
    unwrap_annotation,
)

_LOGGER = get_logger(__name__)

SerializedValue: TypeAlias = Union[
    str, int, float, bool, None, Sequence["SerializedValue"], "SerializedDict"
]
SerializedDict: TypeAlias = dict[str, SerializedValue]
SerializedObject: TypeAlias = SerializedDict | str | bytes
RegistryWrappedValueLeaf: TypeAlias = Union[int, bool, str, float, "Serializable"]
RegistryWrappedValue: TypeAlias = Union[
    RegistryWrappedValueLeaf,
    tuple["RegistryWrappedValue", ...],
    frozenset["RegistryWrappedValue"],
]


def is_serialized_value(v: Any) -> TypeGuard[SerializedValue]:
    """Return if `v` is a valid `SerializedValue`.

    The sequence arm accepts only `list`, matching what `json.loads` produces
    (JSON arrays always decode to lists). A `tuple` is therefore rejected even
    though it satisfies the `Sequence` arm of the type alias; this keeps the
    guard aligned with the actual wire form the engine round-trips. Note that
    `NaN`/`Infinity` floats pass this structural check but are rejected at
    JSON serialization time (see the module-level float contract).
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, (bytes, bytearray, memoryview)):
        return False
    if isinstance(v, Mapping):
        return is_serialized_dict(v)
    if isinstance(v, list):
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


def _format_type(ty: type | UnionType) -> str:
    if isinstance(ty, type):
        return ty.__name__
    return pformat(ty)


def _format_type_structure(structure: dict[str, type | UnionType]) -> str:
    return "\n".join(f'\t"{fld}": {_format_type(ty)}' for fld, ty in structure.items())


@register_error
class SerializationPayloadTypeError(SerializationError, TypeError):
    """Raised when a payload has the wrong Python type for a given format."""

    def __init__(
        self, fmt: SerializationFormat, expected_type: type | UnionType, actual: Any
    ) -> None:
        super().__init__(
            f'Invalid payload type for format "{fmt.value}". '
            f'Expected "{_format_type(expected_type)}", got '
            f'"{_format_type(type(actual))}".'
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


@register_error
class MalformedPayloadError(SerializationError, ValueError):
    """Raised when raw bytes/text cannot be decoded into a serialized object.

    Covers malformed input that fails before structural validation: invalid
    UTF-8 in a binary envelope or JSON payload, and text that is not valid
    JSON. Keeps such failures inside the ``SerializationError`` hierarchy
    rather than leaking raw ``UnicodeDecodeError``/``json.JSONDecodeError``.
    """


@register_error
class SerializationTypeError(SerializationError, TypeError):
    """Raised when serialization fails due to a type error."""

    def __init__(self, actual_type: type) -> None:
        super().__init__(
            f'Unable to serialize object of type "{_format_type(actual_type)}".'
        )


@register_error
class SerializationValueError(SerializationError, ValueError):
    """Raised when serialization fails due to a value error."""

    def __init__(self, expected_description_phrase: str, actual_value: Any) -> None:
        super().__init__(
            f"While serializing, expected {expected_description_phrase}, got "
            f"{repr(actual_value)}"
        )


@register_error
class DeserializationDictStructureError(SerializationError):
    """Raised when dictionary provided for deserialization has an invalid structure."""

    def __init__(
        self,
        class_type: type,
        expected_structure: dict[str, type | UnionType],
        actual_data: SerializedDict,
    ) -> None:
        expected_fields_str = _format_type_structure(expected_structure)
        actual_fields_str = _format_type_structure(
            {k: type(v) for k, v in actual_data.items()}
        )
        super().__init__(
            "Invalid dictionary structure for deserializing to "
            f'"{class_type.__name__}".\nExpected fields: '
            f"{{\n{expected_fields_str}\n}}\nActual fields: "
            f"{{\n{actual_fields_str}\n}}"
        )


@register_error
class DeserializationValueError(SerializationError, ValueError):
    """Raised when data provided for deserialization has invalid values."""

    @overload
    def __init__(self, message: str, /) -> None: ...

    @overload
    def __init__(
        self,
        class_type: type,
        field_name: str,
        expected_description_phrase: str,
        actual_value: Any,
        /,
    ) -> None: ...

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            super().__init__(args[0])
        elif len(args) == 4 and isinstance(args[0], type):  # noqa: PLR2004
            class_type, field_name, expected_description_phrase, actual_value = args
            super().__init__(
                f'Invalid value for field "{field_name}" while deserializing to '
                f'"{class_type.__name__}". Expected {expected_description_phrase}, '
                f"got {repr(actual_value)}"
            )
        else:
            raise ValueError('Invalid arguments for "DeserializationValueError".')


@register_error
class SerializationDerivationError(SerializationError, TypeError):
    """Raised when a class relying on schema derivation cannot be set up.

    Fires on first instantiation of a concrete subclass when the class is not
    a dataclass, a field type cannot be inferred, or it sets ``derive=False``
    without hand-writing the target serialization method(s). The message names
    the specific cause and, for an uninferable field, the available fixes:
    supply a per-field codec via ``field(metadata={"serialize_codec": ...})``,
    implement the serialization method(s) by hand, or pass ``derive=False``.
    """


_TYPE_REGISTRY: dict[str, type["Serializable"]] = {}


def _get_default_type_id(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_type_id(
    type_id: str,
    *,
    allow_import_fallback: bool = False,
    allowed_module_prefixes: tuple[str, ...] = ("fhy_core.",),
) -> type["Serializable"]:
    """Resolve a ``type_id`` string to its registered class, or via import fallback.

    The registry is consulted first; this is always safe. When
    ``allow_import_fallback`` is True and the type is not registered, the
    module named in ``type_id`` is loaded via ``importlib.import_module``,
    subject to ``allowed_module_prefixes``.

    Security:
        ``importlib.import_module`` executes the named module's top-level
        code. Even with the default prefix restriction
        (``("fhy_core.",)``), enabling this path with untrusted blob input
        is dangerous: a blob can choose which ``fhy_core`` module gets
        imported and trigger its registry registrations, logging side
        effects, etc. Keep ``allow_import_fallback=False`` unless the
        ``type_id`` source is fully trusted.
    """
    if type_id in _TYPE_REGISTRY:
        _LOGGER.debug("resolved type_id %r from registry", type_id)
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

    _LOGGER.warning(
        "importing module %r to resolve type_id %r via import fallback "
        "(allowed prefixes=%s); this executes module top-level code",
        module_name,
        type_id,
        allowed_module_prefixes,
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
    except UnknownTypeIdError:
        raise
    except Exception as e:
        raise UnknownTypeIdError(f'Could not resolve type_id "{type_id}": {e}') from e


# Wire keys shared by the registry-wrapped-value encoding and the
# ``WrappedFamilySerializable`` family encoding. The ``_RegistryWrappedValueData``
# TypedDict below must repeat them as literal field names (a language
# requirement); everywhere else, reference these constants to prevent drift.
_WRAPPED_TYPE_KEY: Final[str] = "__type__"
_WRAPPED_DATA_KEY: Final[str] = "__data__"


class _RegistryWrappedValueData(TypedDict):
    __type__: str
    __data__: SerializedValue


def _is_valid_registry_wrapped_value_data(
    data: SerializedDict,
) -> TypeGuard[_RegistryWrappedValueData]:
    return (
        _WRAPPED_TYPE_KEY in data
        and isinstance(data[_WRAPPED_TYPE_KEY], str)
        and _WRAPPED_DATA_KEY in data
        and is_serialized_value(data[_WRAPPED_DATA_KEY])
    )


def is_registry_wrapped_value_leaf(v: Any) -> TypeGuard[RegistryWrappedValueLeaf]:
    """Return if `v` is a valid `RegistryWrappedValueLeaf`."""
    return isinstance(v, (bool, int, str, float, Serializable))


def is_registry_wrapped_value(v: Any) -> TypeGuard[RegistryWrappedValue]:
    """Return if `v` is a valid `RegistryWrappedValue`."""
    if is_registry_wrapped_value_leaf(v):
        return True
    if isinstance(v, tuple):
        return all(is_registry_wrapped_value(x) for x in v)
    if isinstance(v, frozenset):
        return all(is_registry_wrapped_value(x) for x in v)
    return False


_REGISTRY_WRAPPED_BOOL_TYPE_ID: Final[str] = "builtins.bool"
_REGISTRY_WRAPPED_INT_TYPE_ID: Final[str] = "builtins.int"
_REGISTRY_WRAPPED_STR_TYPE_ID: Final[str] = "builtins.str"
_REGISTRY_WRAPPED_FLOAT_TYPE_ID: Final[str] = "builtins.float"
_REGISTRY_WRAPPED_TUPLE_TYPE_ID: Final[str] = "builtins.tuple"
_REGISTRY_WRAPPED_FROZENSET_TYPE_ID: Final[str] = "builtins.frozenset"


def serialize_registry_wrapped_value(value: RegistryWrappedValue) -> SerializedDict:
    """Serialize a scalar/serializable value into a wrapped registry dict.

    Frozenset elements are sorted by ``repr`` so the wrapped form is
    deterministic across processes.

    Raises:
        SerializationTypeError: If ``value`` is not a supported leaf,
            ``tuple``, ``frozenset``, or ``Serializable``.
    """
    if isinstance(value, bool):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_BOOL_TYPE_ID,
            _WRAPPED_DATA_KEY: value,
        }
    elif isinstance(value, int):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_INT_TYPE_ID,
            _WRAPPED_DATA_KEY: value,
        }
    elif isinstance(value, str):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_STR_TYPE_ID,
            _WRAPPED_DATA_KEY: value,
        }
    elif isinstance(value, float):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_FLOAT_TYPE_ID,
            _WRAPPED_DATA_KEY: value,
        }
    elif isinstance(value, tuple):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_TUPLE_TYPE_ID,
            _WRAPPED_DATA_KEY: [serialize_registry_wrapped_value(v) for v in value],
        }
    elif isinstance(value, frozenset):
        return {
            _WRAPPED_TYPE_KEY: _REGISTRY_WRAPPED_FROZENSET_TYPE_ID,
            _WRAPPED_DATA_KEY: sorted(
                [serialize_registry_wrapped_value(v) for v in value], key=repr
            ),
        }
    elif isinstance(value, Serializable):
        return {
            _WRAPPED_TYPE_KEY: value.get_serialization_class_type_id(),
            _WRAPPED_DATA_KEY: value.serialize_to_dict(),
        }
    else:
        raise SerializationTypeError(type(value))


def _deserialize_registry_wrapped_scalar_value(
    type_id: str, value_data: SerializedValue
) -> RegistryWrappedValueLeaf | None:
    if type_id == _REGISTRY_WRAPPED_BOOL_TYPE_ID:
        if not isinstance(value_data, bool):
            raise DeserializationValueError(
                "Expected boolean data for wrapped bool value."
            )
        return value_data
    elif type_id == _REGISTRY_WRAPPED_INT_TYPE_ID:
        if not isinstance(value_data, int) or isinstance(value_data, bool):
            raise DeserializationValueError(
                "Expected integer data for wrapped int value."
            )
        return value_data
    elif type_id == _REGISTRY_WRAPPED_STR_TYPE_ID:
        if not isinstance(value_data, str):
            raise DeserializationValueError(
                "Expected string data for wrapped str value."
            )
        return value_data
    elif type_id == _REGISTRY_WRAPPED_FLOAT_TYPE_ID:
        if not isinstance(value_data, float):
            raise DeserializationValueError(
                "Expected float data for wrapped float value."
            )
        return value_data
    else:
        return None


def _deserialize_registry_wrapped_container_value(
    type_id: str, value_data: SerializedValue
) -> tuple[RegistryWrappedValue, ...] | frozenset[RegistryWrappedValue] | None:
    if type_id not in (
        _REGISTRY_WRAPPED_TUPLE_TYPE_ID,
        _REGISTRY_WRAPPED_FROZENSET_TYPE_ID,
    ):
        return None
    if not isinstance(value_data, list):
        raise DeserializationValueError(
            "Expected list payload for wrapped tuple/frozenset value."
        )

    wrapped_items: list[RegistryWrappedValue] = []
    for value_item in value_data:
        if not is_serialized_dict(value_item):
            raise DeserializationValueError(
                "Expected wrapped dictionary items for wrapped tuple/frozenset value."
            )
        wrapped_items.append(deserialize_registry_wrapped_value(value_item))

    if type_id == _REGISTRY_WRAPPED_TUPLE_TYPE_ID:
        return tuple(wrapped_items)
    else:
        try:
            return frozenset(wrapped_items)
        except TypeError as exc:
            raise DeserializationValueError(
                "Wrapped frozenset value contains unhashable items."
            ) from exc


def deserialize_registry_wrapped_value(data: SerializedDict) -> RegistryWrappedValue:
    """Deserialize a wrapped registry dict to scalar/serializable value.

    Raises:
        DeserializationDictStructureError: If ``data`` is not a valid wrapped
            registry dict.
        DeserializationValueError: If the payload does not match its declared
            wrapped type.
        UnknownTypeIdError: If a wrapped object's ``type_id`` is not registered.
    """
    if not _is_valid_registry_wrapped_value_data(data):
        raise DeserializationDictStructureError(
            Serializable, _RegistryWrappedValueData.__annotations__, data
        )

    # ``data`` is narrowed to the ``_RegistryWrappedValueData`` TypedDict here,
    # whose access requires literal keys (the keys equal the constants above).
    type_id = data["__type__"]
    value_data = data["__data__"]

    scalar_value = _deserialize_registry_wrapped_scalar_value(type_id, value_data)
    if scalar_value is not None:
        return scalar_value

    container_value = _deserialize_registry_wrapped_container_value(type_id, value_data)
    if container_value is not None:
        return container_value

    if not is_serialized_dict(value_data):
        raise DeserializationValueError(
            "Expected dictionary payload for wrapped object."
        )

    cls = _resolve_type_id(type_id)
    obj = cls.deserialize_from_dict(value_data)
    return obj


@overload
def register_serializable(
    cls: type[_T], *, type_id: str | None = ..., alias: bool = ...
) -> type[_T]: ...
@overload
def register_serializable(
    *, type_id: str | None = ..., alias: bool = ...
) -> Callable[[type[_T]], type[_T]]: ...


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
        _LOGGER.debug("registered %r -> %s", ty_id, c.__qualname__)
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
    if len(payload_bytes) > 0xFFFFFFFF:  # noqa: PLR2004  # pragma: no cover
        raise SerializationError("payload too large (>4GB).")

    header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, codec_u8, len(type_id))
    payload_len = _PAYLOAD_LEN_STRUCT.pack(len(payload_bytes))
    return header + type_id + payload_len + payload_bytes


def _decode_utf8(raw: bytes | bytearray | memoryview, *, context: str) -> str:
    """Decode bytes as UTF-8, raising :class:`MalformedPayloadError` on failure."""
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError as e:
        raise MalformedPayloadError(f"{context} is not valid UTF-8.") from e


def _loads_json(text: str, *, context: str) -> Any:
    """Parse a JSON document, raising :class:`MalformedPayloadError` on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise MalformedPayloadError(f"{context} is not valid JSON.") from e


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

    type_id = _decode_utf8(mv[offset:end_type], context="binary envelope type_id")
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
    _LOGGER.debug(
        "decoded binary envelope (type_id=%r, version=%d, codec=%s, payload_len=%d)",
        type_id,
        version,
        codec.value,
        payload_len,
    )
    return cls.deserialize_from_binary(payload_bytes, codec=codec)


# ===========================================================================
# Schema-derived serialization engine
# ===========================================================================
#
# When a Serializable / WrappedFamilySerializable subclass opts in (the default,
# ``derive=True``) and does not hand-write the relevant method(s), the trait
# derives them from the subclass's dataclass fields and resolved type hints.
# Each field maps to one FieldCodec (from ``field.metadata["serialize_codec"]``
# if present, else inferred from the resolved annotation). The plan is built and
# validated at first instantiation, matching the ABC metaclass's "cannot
# instantiate" timing.


@runtime_checkable
class FieldCodec(Protocol):
    """Encode, validate, and decode one field value for schema derivation.

    Every codec the engine infers, and every codec a caller supplies through
    ``field(metadata={"serialize_codec": ...})``, satisfies this contract. Use
    ``make_field_codec``, ``make_enum_field_codec``, or
    ``make_labeled_enum_field_codec`` to build one from plain functions rather
    than implementing it by hand.

    Implementations must round-trip, and ``decode`` must raise
    ``DeserializationDictStructureError`` / ``DeserializationValueError`` on
    malformed input so the trait surfaces uniform errors. ``decode`` is only
    called after ``accepts`` returns ``True``, so it may assume the coarse
    structure already holds; a codec whose ``accepts`` returns ``True``
    unconditionally (such as one built by ``make_field_codec``) must therefore
    do all of its validation in ``decode``.

    Attributes:
        expected: The Python type (or union) of the serialized value, used in
            structure-error messages.
    """

    expected: type | UnionType

    def encode(self, value: Any) -> SerializedValue:
        """Return the JSON-friendly serialized form of ``value``."""

    def accepts(self, data: SerializedValue) -> bool:
        """Return whether ``data`` is structurally valid for this field."""

    def decode(self, data: SerializedValue, *, field_name: str, owner: type) -> Any:
        """Reconstruct the field value from ``data``."""


_SERIALIZE_SETUP_FLAG: Final[str] = "_fhy_serialize_setup_done"
# Serialization plans are built once per class and cached forever; this assumes
# a class's dataclass field schema is frozen after first use (see the matching
# note on ``_PLAN_CACHE`` in ``traits/derived_equivalence.py``).
_SERIALIZE_PLAN_CACHE: dict[type, dict[str, FieldCodec]] = {}
_FIELD_CODEC_REGISTRY: dict[type, FieldCodec] = {}


class _CodecInferenceError(Exception):
    """Internal: no codec could be inferred for a resolved field type."""

    type_repr: str

    def __init__(self, type_obj: Any) -> None:
        self.type_repr = (
            _format_type(type_obj)
            if isinstance(type_obj, (type, UnionType))
            else repr(type_obj)
        )
        super().__init__(self.type_repr)


class _ScalarFieldCodec(FieldCodec):
    """Abstract base for JSON-scalar codecs that store the value unchanged.

    Subclasses provide ``expected`` and ``accepts``. ``_FloatFieldCodec``
    overrides ``decode`` to coerce an integer payload to ``float``.
    """

    expected: type | UnionType

    def encode(self, value: Any) -> SerializedValue:
        return cast(SerializedValue, value)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        return data

    @abstractmethod
    def accepts(self, data: Any) -> bool: ...


class _IntFieldCodec(_ScalarFieldCodec):
    expected: type | UnionType = int

    def accepts(self, data: Any) -> bool:
        return isinstance(data, int) and not isinstance(data, bool)


class _FloatFieldCodec(_ScalarFieldCodec):
    expected: type | UnionType = float

    def accepts(self, data: Any) -> bool:
        return isinstance(data, (int, float)) and not isinstance(data, bool)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        return float(data)


class _StrFieldCodec(_ScalarFieldCodec):
    expected: type | UnionType = str

    def accepts(self, data: Any) -> bool:
        return isinstance(data, str)


class _BoolFieldCodec(_ScalarFieldCodec):
    expected: type | UnionType = bool

    def accepts(self, data: Any) -> bool:
        return isinstance(data, bool)


class _OptionalFieldCodec(FieldCodec):
    expected: type | UnionType
    _inner: FieldCodec

    def __init__(self, inner: FieldCodec) -> None:
        self._inner = inner
        self.expected = inner.expected | None

    def encode(self, value: Any) -> SerializedValue:
        return None if value is None else self._inner.encode(value)

    def accepts(self, data: Any) -> bool:
        return data is None or self._inner.accepts(data)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        if data is None:
            return None
        return self._inner.decode(data, field_name=field_name, owner=owner)


class _SequenceFieldCodec(FieldCodec):
    expected: type | UnionType = list
    _factory: Callable[[Any], Any]
    _inner: FieldCodec

    def __init__(self, factory: Callable[[Any], Any], inner: FieldCodec) -> None:
        self._factory = factory
        self._inner = inner

    def encode(self, value: Any) -> SerializedValue:
        encoded = [self._inner.encode(item) for item in value]
        if self._factory is frozenset:
            # A frozenset iterates in hash order, which varies across processes
            # for str/bytes elements (hash randomization). Sort the encoded
            # items so the wire form is deterministic, matching the registry-
            # wrapped frozenset path.
            encoded.sort(key=repr)
        return encoded

    def accepts(self, data: Any) -> bool:
        return isinstance(data, list) and all(
            self._inner.accepts(item) for item in data
        )

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        return self._factory(
            self._inner.decode(item, field_name=field_name, owner=owner)
            for item in data
        )


class _SerializableFieldCodec(FieldCodec):
    expected: type | UnionType = dict
    _serializable_cls: type["Serializable"]

    def __init__(self, serializable_cls: type["Serializable"]) -> None:
        self._serializable_cls = serializable_cls

    def encode(self, value: Any) -> SerializedValue:
        return cast(SerializedValue, value.serialize_to_dict())

    def accepts(self, data: Any) -> bool:
        return is_serialized_dict(data)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        return self._serializable_cls.deserialize_from_dict(data)


class _PathFieldCodec(FieldCodec):
    expected: type | UnionType = str
    _path_type: type

    def __init__(self, path_type: type) -> None:
        self._path_type = path_type

    def encode(self, value: Any) -> SerializedValue:
        # Emit the POSIX form (forward slashes) rather than ``str(value)``: the
        # latter is OS-dependent (backslashes on Windows), which would make the
        # wire form non-portable across platforms. ``PurePath`` accepts forward
        # slashes on every platform, so decoding round-trips everywhere.
        return cast(str, value.as_posix())

    def accepts(self, data: Any) -> bool:
        return isinstance(data, str)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        return self._path_type(data)


class _EnumByNameFieldCodec(FieldCodec):
    expected: type | UnionType = str
    _enum_type: type[enum.Enum]

    def __init__(self, enum_type: type[enum.Enum]) -> None:
        self._enum_type = enum_type

    def encode(self, value: Any) -> SerializedValue:
        return cast(SerializedValue, value.name)

    def accepts(self, data: Any) -> bool:
        return isinstance(data, str)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        try:
            return self._enum_type[data]
        except KeyError as exc:
            raise DeserializationValueError(
                owner, field_name, f"a valid {self._enum_type.__name__} name", data
            ) from exc


class _EnumByValueFieldCodec(FieldCodec):
    expected: type | UnionType = object
    _enum_type: type[enum.Enum]

    def __init__(self, enum_type: type[enum.Enum]) -> None:
        self._enum_type = enum_type

    def encode(self, value: Any) -> SerializedValue:
        return cast(SerializedValue, value.value)

    def accepts(self, data: Any) -> bool:
        if issubclass(self._enum_type, str):
            return isinstance(data, str)
        if issubclass(self._enum_type, int):
            return isinstance(data, int) and not isinstance(data, bool)
        return True

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        try:
            return self._enum_type(data)
        except (ValueError, TypeError) as exc:
            # ValueError: in-range type but not a member value. TypeError:
            # unhashable payload (e.g. a dict/list) reaching ``Enum(value)``.
            # Both must surface through the serialization error hierarchy.
            raise DeserializationValueError(
                owner, field_name, f"a valid {self._enum_type.__name__} value", data
            ) from exc


class _LabeledEnumFieldCodec(FieldCodec):
    expected: type | UnionType = str
    _enum_type: type[enum.Enum]
    _value_to_label: Mapping[Any, str]
    _label_to_value: dict[str, Any]

    def __init__(self, enum_type: type[enum.Enum], labels: Mapping[Any, str]) -> None:
        self._enum_type = enum_type
        self._value_to_label = labels
        self._label_to_value = {label: member for member, label in labels.items()}

    def encode(self, value: Any) -> SerializedValue:
        try:
            return cast(SerializedValue, self._value_to_label[value])
        except KeyError as exc:
            # The labels were verified to cover every member at build time, so
            # this only fires for an out-of-domain value (e.g. a member of a
            # different enum). Surface it through the serialization hierarchy
            # rather than leaking a bare KeyError.
            raise SerializationValueError(
                f"a labeled {self._enum_type.__name__} member", value
            ) from exc

    def accepts(self, data: Any) -> bool:
        return isinstance(data, str)

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        try:
            return self._label_to_value[data]
        except KeyError as exc:
            raise DeserializationValueError(
                owner, field_name, f"a valid {self._enum_type.__name__} label", data
            ) from exc


class _FunctionFieldCodec(FieldCodec):
    expected: type | UnionType = object
    _encode_fn: Callable[[Any], SerializedValue]
    _decode_fn: Callable[[Any], Any]

    def __init__(
        self,
        encode_fn: Callable[[Any], SerializedValue],
        decode_fn: Callable[[Any], Any],
    ) -> None:
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn

    def encode(self, value: Any) -> SerializedValue:
        return self._encode_fn(value)

    def accepts(self, data: Any) -> bool:
        return True

    def decode(self, data: Any, *, field_name: str, owner: type) -> Any:
        try:
            return self._decode_fn(data)
        except (ValueError, TypeError) as exc:
            raise DeserializationValueError(
                owner, field_name, "a decodable value", data
            ) from exc


_BUILTIN_SCALAR_CODECS: dict[type, FieldCodec] = {
    bool: _BoolFieldCodec(),
    int: _IntFieldCodec(),
    float: _FloatFieldCodec(),
    str: _StrFieldCodec(),
}
_SEQUENCE_FACTORIES: dict[type, Callable[[Any], Any]] = {
    tuple: tuple,
    list: list,
    frozenset: frozenset,
}


def _sequence_element_type(origin: type, arguments: tuple[Any, ...]) -> Any:
    if origin is tuple:
        # Only the homogeneous, variable-length form ``tuple[T, ...]`` is
        # derivable. A fixed-length form such as ``tuple[T]`` or
        # ``tuple[T, U]`` is rejected: the sequence codec does not validate
        # length, so deriving it would silently accept payloads of any arity.
        if len(arguments) == 2 and arguments[1] is Ellipsis:  # noqa: PLR2004
            return arguments[0]
        raise _CodecInferenceError(origin)
    if len(arguments) == 1:
        return arguments[0]
    raise _CodecInferenceError(origin)


def _infer_field_codec(resolved: Any) -> FieldCodec:
    annotation = unwrap_annotation(resolved)
    inner, is_optional = split_optional(annotation)
    if is_optional:
        return _OptionalFieldCodec(_infer_field_codec(inner))

    container = get_origin_and_arguments(inner)
    if container is not None:
        origin, arguments = container
        factory = _SEQUENCE_FACTORIES.get(origin)
        if factory is None:
            raise _CodecInferenceError(inner)
        element = _sequence_element_type(origin, arguments)
        return _SequenceFieldCodec(factory, _infer_field_codec(element))

    if isinstance(inner, type):
        if inner in _FIELD_CODEC_REGISTRY:
            return _FIELD_CODEC_REGISTRY[inner]
        if issubclass(inner, Serializable):
            return _SerializableFieldCodec(inner)
        if issubclass(inner, enum.Enum):
            # A ``StrEnum`` / ``IntEnum`` member is its own serializable scalar
            # value (``str`` / ``int``), so serialize by value -- that is the
            # canonical, stable wire form and never needs a metadata override.
            # A plain ``Enum`` value may be an arbitrary or positional
            # (``auto()``) object that is neither serializable nor stable, so
            # fall back to the always-safe member name. Override per field with
            # ``make_enum_field_codec`` / ``make_labeled_enum_field_codec``.
            if issubclass(inner, (str, int)):
                return _EnumByValueFieldCodec(inner)
            return _EnumByNameFieldCodec(inner)
        if issubclass(inner, PurePath):
            return _PathFieldCodec(inner)
        builtin = _BUILTIN_SCALAR_CODECS.get(inner)
        if builtin is not None:
            return builtin

    raise _CodecInferenceError(inner)


def _build_serialization_plan(cls: type) -> dict[str, FieldCodec]:
    try:
        field_defs = dataclasses.fields(cls)
    except TypeError as exc:
        raise SerializationDerivationError(
            f'Cannot derive serialization for "{cls.__name__}": it is not a '
            f"dataclass. Implement serialize_to_dict / deserialize_from_dict by "
            f"hand, or pass derive=False."
        ) from exc

    resolved_hints = resolve_field_annotations(cls)
    plan: dict[str, FieldCodec] = {}
    for field_def in field_defs:
        codec = field_def.metadata.get("serialize_codec")
        if codec is None:
            if field_def.name not in resolved_hints:
                # ``resolve_field_annotations`` omits fields whose annotation
                # could not be resolved (the underlying cause is logged as a
                # warning from that helper). Surface that as the true cause
                # rather than feeding the raw, unresolved annotation -- often a
                # bare string under ``from __future__ import annotations`` --
                # into inference, which would mislabel it an "unsupported type".
                raise SerializationDerivationError(
                    f"Cannot derive a serialization codec for field "
                    f'"{cls.__name__}.{field_def.name}": its type annotation '
                    f"could not be resolved (e.g. a forward reference to a name "
                    f"that is not importable at the class's module scope; see "
                    f"the preceding annotation-resolution warning for the cause)."
                    f" Fix the annotation, supply field(metadata="
                    f'{{"serialize_codec": ...}}), implement serialize_to_dict / '
                    f"deserialize_from_dict by hand, or pass derive=False."
                )
            resolved = resolved_hints[field_def.name]
            try:
                codec = _infer_field_codec(resolved)
            except _CodecInferenceError as exc:
                raise SerializationDerivationError(
                    f"Cannot derive a serialization codec for field "
                    f'"{cls.__name__}.{field_def.name}" of type {exc.type_repr}. '
                    f'Supply field(metadata={{"serialize_codec": ...}}), implement '
                    f"serialize_to_dict / deserialize_from_dict by hand, or pass "
                    f"derive=False."
                ) from exc
        plan[field_def.name] = codec
    return plan


def _get_or_build_serialization_plan(cls: type) -> dict[str, FieldCodec]:
    plan = _SERIALIZE_PLAN_CACHE.get(cls)
    if plan is None:
        plan = _build_serialization_plan(cls)
        _SERIALIZE_PLAN_CACHE[cls] = plan
    return plan


def _derived_serialize_to_dict(self: "Serializable") -> SerializedDict:
    plan = _get_or_build_serialization_plan(type(self))
    return {name: codec.encode(getattr(self, name)) for name, codec in plan.items()}


def _derived_deserialize_from_dict(
    cls: type["Serializable"], data: SerializedDict
) -> "Serializable":
    plan = _get_or_build_serialization_plan(cls)
    expected_structure = {name: codec.expected for name, codec in plan.items()}
    if set(data) != set(plan) or not all(
        plan[name].accepts(data[name]) for name in plan
    ):
        raise DeserializationDictStructureError(cls, expected_structure, data)
    decoded = {
        name: plan[name].decode(data[name], field_name=name, owner=cls) for name in plan
    }
    try:
        return cls.construct_from_fields(decoded)
    except SerializationError:
        # Already in the serialization hierarchy (e.g. a nested
        # DeserializationValueError); propagate unwrapped.
        raise
    except (ValueError, TypeError) as exc:
        raise DeserializationValueError(str(exc)) from exc


def _is_trait_default_method(cls: type["Serializable"], name: str) -> bool:
    """Return whether ``cls`` resolves ``name`` to the trait's deriving default.

    Walks the MRO and returns ``True`` when the first class defining ``name`` is
    ``Serializable`` or ``WrappedFamilySerializable`` -- i.e. the subclass did
    not hand-write the method and therefore relies on schema derivation.
    """
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass in (Serializable, WrappedFamilySerializable)
    return False


def _validate_serialization_derivation(cls: type["Serializable"]) -> None:
    """Validate, at first instantiation, that a deriving class can be derived.

    A no-op for abstract classes and for classes that hand-write both target
    methods. For a deriving class, builds (and caches) the codec plan, raising
    ``SerializationDerivationError`` if it cannot be built. For a
    ``derive=False`` class that left a target method to the deriving default,
    raises rather than silently deriving.
    """
    if getattr(cls, "__abstractmethods__", frozenset()):
        return
    serialize_name, deserialize_name = cls._SERIALIZE_DERIVE_TARGET
    uses_derivation = _is_trait_default_method(
        cls, serialize_name
    ) or _is_trait_default_method(cls, deserialize_name)
    if not uses_derivation:
        return
    if cls._SERIALIZE_DERIVE:
        _get_or_build_serialization_plan(cls)
    else:
        raise SerializationDerivationError(
            f'"{cls.__name__}" sets derive=False but does not implement '
            f'"{serialize_name}" / "{deserialize_name}". Implement them by hand, '
            f"or remove derive=False to use schema derivation."
        )


class Serializable(ABC):
    """Serialization trait for compiler objects.

    Dict form (derived by default):
      - serialize_to_dict()
      - deserialize_from_dict()
      A ``@dataclass`` subclass derives both from its field schema. Override
      them to hand-write the dict form, or pass ``derive=False`` to require a
      hand-written implementation (a missing one then raises
      ``SerializationDerivationError`` at first instantiation).

    Optional:
      - get_binary_codec()
      - serialize_to_binary()
      - deserialize_from_binary()

    Notes:
    - The default binary codec stores JSON bytes of the dict form.
    - Override the binary hooks to define a compact canonical binary form.

    """

    _SERIALIZATION_CLASS_TYPE_ID: ClassVar[str | None] = None
    _SERIALIZE_DERIVE: ClassVar[bool] = True
    _SERIALIZE_DERIVE_TARGET: ClassVar[tuple[str, str]] = (
        "serialize_to_dict",
        "deserialize_from_dict",
    )

    def __init_subclass__(cls, *, derive: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._SERIALIZE_DERIVE = derive

    if not TYPE_CHECKING:
        # Defined only at runtime so the permissive ``*args, **kwargs`` signature
        # does not disable constructor-call type checking for subclasses (the
        # same reason FrozenMixin guards its ``__new__``). The derivation plan is
        # built and validated here, on first instantiation, because ``@dataclass``
        # finalizes the field list only after ``__init_subclass__`` runs.
        def __new__(cls, *args: Any, **kwargs: Any) -> "Serializable":
            del args, kwargs
            if not cls.__dict__.get(_SERIALIZE_SETUP_FLAG, False):
                _validate_serialization_derivation(cls)
                type.__setattr__(cls, _SERIALIZE_SETUP_FLAG, True)
            return super().__new__(cls)

    @classmethod
    def get_serialization_class_type_id(cls) -> str:
        """Return the type_id for this class; used for serialization."""
        local = cls.__dict__.get("_SERIALIZATION_CLASS_TYPE_ID", None)
        return local or _get_default_type_id(cls)

    @classmethod
    def construct_from_fields(cls: type[_T], fields: dict[str, Any]) -> _T:
        """Build an instance from already-decoded fields.

        Default reconstruction for derived ``deserialize_from_dict``. Override
        for classes whose reconstruction is not a plain constructor call (e.g.
        interning or ``__new__``-based construction).

        Args:
            fields: Mapping of field name to decoded value, in declaration order.

        Returns:
            A reconstructed instance of ``cls``.
        """
        return cls(**fields)

    def serialize_to_dict(self) -> SerializedDict:
        """Return a dictionary representation of this object for serialization.

        By default the dict is derived from this object's dataclass fields and
        their type hints. Override to customize, or pass ``derive=False`` to
        require a hand-written implementation.
        """
        return _derived_serialize_to_dict(self)

    @classmethod
    def deserialize_from_dict(cls: type[_T], data: SerializedDict) -> _T:
        """Create an instance of this class from a dictionary representation.

        By default the fields are derived from this class's dataclass fields and
        their type hints, validated, decoded, and passed to
        ``construct_from_fields``. Override to customize, or pass
        ``derive=False`` to require a hand-written implementation.
        """
        return cast(_T, _derived_deserialize_from_dict(cls, data))

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
        try:
            return json.dumps(
                payload_dict,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except ValueError as exc:
            raise SerializationValueError(
                "a JSON-finite numeric payload (no NaN or Infinity)", payload_dict
            ) from exc

    @classmethod
    def deserialize_from_binary(
        cls: type[_T],
        payload: bytes,
        *,
        codec: BinaryPayloadCodec,
    ) -> _T:
        """Decode payload bytes into an instance of this class.

        Default implementation supports JSON only (payload is JSON bytes of dict form).
        Override for CUSTOM (e.g. a compact binary codec).

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
            text = _decode_utf8(payload, context="binary JSON payload")
            payload_obj = _loads_json(text, context="binary JSON payload")
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

        For ``BINARY``, deserialization is always registry-only (no import
        fallback). Use ``Serializable.from_bytes`` directly when you need to opt
        into ``allow_import_fallback`` for a trusted blob.

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
                raise SerializationPayloadTypeError(fmt, dict, payload)
            return cls.deserialize_from_dict(payload)
        elif fmt is SerializationFormat.JSON:
            if not isinstance(payload, (str, bytes, bytearray)):
                raise SerializationPayloadTypeError(
                    fmt, str | bytes | bytearray, payload
                )
            return cls.from_json(payload)
        elif fmt is SerializationFormat.BINARY:
            if not isinstance(payload, (bytes, bytearray, memoryview)):
                raise SerializationPayloadTypeError(
                    fmt, bytes | bytearray | memoryview, payload
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

        Raises:
            SerializationValueError: If the payload contains ``NaN`` or
                ``Infinity`` values, which JSON does not support per the
                strict spec.

        """
        payload = self.serialize_to_dict()
        try:
            return json.dumps(
                payload,
                indent=indent,
                sort_keys=sort_keys,
                allow_nan=False,
            )
        except ValueError as exc:
            raise SerializationValueError(
                "a JSON-finite numeric payload (no NaN or Infinity)",
                payload,
            ) from exc

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
            payload = _decode_utf8(payload, context="JSON payload")
        payload_obj = _loads_json(payload, context="JSON payload")
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
            allow_import_fallback: When ``True``, ``type_id`` values not
                present in the registered serialization registry trigger an
                ``importlib.import_module`` call to load the module named in
                the ``type_id`` (subject to ``allowed_module_prefixes``).
                **Security: importing a module runs its top-level code;
                enabling this with untrusted blob input lets the blob
                choose which fhy_core module gets imported.** Keep this
                ``False`` unless you control or trust the source of
                ``data``. Defaults to ``False``.
            allowed_module_prefixes: A tuple of allowed module prefixes for
                import fallback. Only consulted when
                ``allow_import_fallback`` is ``True``. Defaults to
                ``("fhy_core.",)`` to confine imports to the core package.

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
      - Base class implements serialize_to_dict() to emit a wrapped dict::

          {"__type__": <type_id>, "__data__": <data_dict>}

      - Subclasses provide only the data portion, by overriding (or, for a
        ``@dataclass`` subclass, letting the engine derive):
          - `serialize_data_to_dict()`
          - `deserialize_data_from_dict()`

      - Base class implements deserialize_from_dict() that:
          - reads `__type__`/`__data__`
          - resolves the concrete subclass from `__type__`
          - calls `subclass.deserialize_data_from_dict(__data__)`

    Example usage (explicit-override form; a ``@dataclass`` subclass could omit
    both methods and let derivation supply them)::

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

    _SERIALIZE_DERIVE_TARGET: ClassVar[tuple[str, str]] = (
        "serialize_data_to_dict",
        "deserialize_data_from_dict",
    )

    def serialize_to_dict(self) -> SerializedDict:
        return {
            _WRAPPED_TYPE_KEY: self.get_serialization_class_type_id(),
            _WRAPPED_DATA_KEY: self.serialize_data_to_dict(),
        }

    @classmethod
    def deserialize_from_dict(cls: type[_F], data: SerializedDict) -> _F:
        expected_keys = {_WRAPPED_TYPE_KEY, _WRAPPED_DATA_KEY}
        if not isinstance(data, dict) or set(data) != expected_keys:
            # Reject missing required keys *and* any unexpected extra keys, so
            # producer/consumer schema drift is loud rather than silently
            # ignored (mirrors the exact-key-set check the derived path uses).
            raise DeserializationDictStructureError(
                cls,
                _RegistryWrappedValueData.__annotations__,
                data if isinstance(data, dict) else {},
            )

        class_type_id = data[_WRAPPED_TYPE_KEY]
        if not isinstance(class_type_id, str):
            raise DeserializationValueError(
                cls, _WRAPPED_TYPE_KEY, "a string type id", class_type_id
            )

        object_data = data[_WRAPPED_DATA_KEY]
        if not is_serialized_dict(object_data):
            raise DeserializationValueError(
                cls, _WRAPPED_DATA_KEY, "a serialized dict payload", object_data
            )

        concrete_class = _resolve_type_id(class_type_id)

        if not issubclass(concrete_class, cls):
            raise SerializationError(
                f'Wrapped type "{concrete_class}" is not a subclass of expected '
                f'family "{cls}".'
            )

        return concrete_class.deserialize_data_from_dict(object_data)

    def serialize_data_to_dict(self) -> SerializedDict:
        """Serialize only this class's data.

        By default the data dict is derived from this object's dataclass fields
        and their type hints. Override to customize, or pass ``derive=False`` to
        require a hand-written implementation.

        Returns:
            A dict representing only this class's data (no type wrapper).

        """
        return _derived_serialize_to_dict(self)

    @classmethod
    def deserialize_data_from_dict(cls: type[_F], data: SerializedDict) -> _F:
        """Deserialize only this class's data.

        By default the fields are derived from this class's dataclass fields and
        their type hints. Override to customize, or pass ``derive=False`` to
        require a hand-written implementation.

        Args:
            data: A dict representing only this class's data (no type wrapper).

        Returns:
            An instance of this class reconstructed from the data dict.

        """
        return cast(_F, _derived_deserialize_from_dict(cls, data))


def register_field_codec(tp: type, codec: FieldCodec) -> None:
    """Register ``codec`` as the inference default for leaf type ``tp``.

    Args:
        tp: The leaf type to match during inference.
        codec: The codec to use for fields resolving to ``tp``.

    Returns:
        None. (Register-style APIs in this package do not return ``self``.)

    Raises:
        SerializationError: If ``tp`` already has a different registered codec.
    """
    existing = _FIELD_CODEC_REGISTRY.get(tp)
    if existing is not None and existing is not codec:
        raise SerializationError(
            f'A field codec is already registered for type "{_format_type(tp)}"; '
            f"refusing to override."
        )
    _FIELD_CODEC_REGISTRY[tp] = codec


def make_field_codec(
    encode: Callable[[Any], SerializedValue],
    decode: Callable[[Any], Any],
) -> FieldCodec:
    """Build a ``FieldCodec`` from an encode/decode function pair.

    Convenience for bespoke per-field overrides supplied through
    ``field(metadata={"serialize_codec": ...})``.

    The resulting codec's ``accepts`` returns ``True`` unconditionally, so all
    validation happens in ``decode``. ``decode`` must therefore raise
    ``ValueError`` or ``TypeError`` on malformed input (the engine wraps those
    as ``DeserializationValueError``); any other exception propagates unwrapped
    and escapes the serialization error hierarchy.

    Args:
        encode: Callable mapping a field value to a ``SerializedValue``.
        decode: Callable mapping a ``SerializedValue`` back to a field value.
            Must raise ``ValueError``/``TypeError`` on malformed input.

    Returns:
        A ``FieldCodec`` wrapping the two callables.
    """
    return _FunctionFieldCodec(encode, decode)


def make_enum_field_codec(
    enum_type: type[enum.Enum], *, by: str = "name"
) -> FieldCodec:
    """Return a codec that serializes an ``Enum`` member by name or value.

    Args:
        enum_type: The ``enum.Enum`` subclass to encode.
        by: ``"name"`` (default) encodes ``member.name``; ``"value"`` encodes
            ``member.value``.

    Returns:
        A ``FieldCodec`` for fields annotated with ``enum_type``.

    Raises:
        ValueError: If ``by`` is not ``"name"`` or ``"value"``.
    """
    if by == "name":
        return _EnumByNameFieldCodec(enum_type)
    if by == "value":
        return _EnumByValueFieldCodec(enum_type)
    raise ValueError(
        f'make_enum_field_codec "by" must be "name" or "value", got {by!r}.'
    )


def make_labeled_enum_field_codec(
    enum_type: type[enum.Enum], labels: Mapping[Any, str]
) -> FieldCodec:
    """Return a codec that serializes an ``Enum`` member via explicit string labels.

    Use this when the on-the-wire form is a bespoke label that is neither the
    member's ``name`` nor its ``value`` -- for example a human-readable
    function name. ``labels`` must assign a distinct label to every member so
    encoding is total and decoding is unambiguous.

    Args:
        enum_type: The ``enum.Enum`` subclass to encode.
        labels: A bijective mapping from each member to its serialized label.

    Returns:
        A ``FieldCodec`` that encodes ``member -> label`` and decodes the
        inverse, raising ``DeserializationValueError`` for an unknown label and
        rejecting non-string payloads as a structure error.

    Raises:
        ValueError: If ``labels`` does not cover every member, or assigns the
            same label to two members.
    """
    members = set(enum_type)
    if set(labels) != members:
        raise ValueError(
            f"make_labeled_enum_field_codec labels must cover every "
            f"{enum_type.__name__} member exactly once."
        )
    if len(set(labels.values())) != len(labels):
        raise ValueError(
            f"make_labeled_enum_field_codec labels for {enum_type.__name__} "
            f"must be distinct."
        )
    return _LabeledEnumFieldCodec(enum_type, labels)
