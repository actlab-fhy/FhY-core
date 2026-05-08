"""Unique identifier for named compiler objects."""

__all__ = ["Identifier"]

from threading import Lock
from typing import Any, ClassVar, TypedDict, TypeGuard, final

from .serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from .trait.equality import EqualMixin


class _IdentifierData(TypedDict):
    id: int
    name_hint: str


_IDENTIFIER_DATA_KEYS: frozenset[str] = frozenset({"id", "name_hint"})


def _is_valid_identifier_data(data: SerializedDict) -> TypeGuard[_IdentifierData]:
    if data.keys() != _IDENTIFIER_DATA_KEYS:
        return False
    id_value = data["id"]
    # `bool` is a subclass of `int` in Python; reject it explicitly so the
    # id space stays integer-only.
    if not isinstance(id_value, int) or isinstance(id_value, bool):
        return False
    return isinstance(data["name_hint"], str)


@final
@register_serializable(type_id="id")
class Identifier(Serializable, EqualMixin):
    """Process-globally unique, named compiler symbol.

    Two ``Identifier`` instances are equal iff they share the same ``id``;
    ``name_hint`` is a debugging aid and is not consulted by ``__eq__`` or
    ``__hash__``. Ids are drawn from a single process-global,
    monotonically-increasing counter and are never reused.

    Construction and deserialization are thread-safe and share the same
    counter: a deserialized id cannot collide with a subsequently
    constructed id, regardless of interleaving. Deserializing an id
    greater than the next-to-be-issued value advances the counter past it.

    ``repr`` of an ``Identifier`` returns ``"<name_hint>::<id>"``. The form
    is for debugging only --- it is not a serialization protocol and is not
    round-trippable through the constructor. Use the structured ``id`` and
    ``name_hint`` properties (or ``serialize_to_dict``) when a parseable
    representation is needed.

    The class is ``@final`` and is not intended to be subclassed; callers
    should treat it as a closed implementation that provides a single
    process-global id space.
    """

    _next_id: ClassVar[int] = 0
    _id_lock: ClassVar[Lock] = Lock()
    _id: int
    _name_hint: str

    def __init__(self, name_hint: str) -> None:
        with Identifier._id_lock:
            self._id = Identifier._next_id
            Identifier._next_id += 1
        self._name_hint = name_hint

    @property
    def name_hint(self) -> str:
        return self._name_hint

    @property
    def id(self) -> int:
        return self._id

    def serialize_to_dict(self) -> SerializedDict:
        return {"id": self._id, "name_hint": self._name_hint}

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Identifier":
        if not _is_valid_identifier_data(data):
            raise DeserializationDictStructureError(
                cls, _IdentifierData.__annotations__, data
            )
        if data["id"] < 0:
            raise DeserializationValueError(
                cls, "id", "a non-negative integer", data["id"]
            )
        identifier = cls.__new__(cls)
        identifier._id = data["id"]
        identifier._name_hint = data["name_hint"]
        with Identifier._id_lock:
            if identifier._id >= Identifier._next_id:
                Identifier._next_id = identifier._id + 1
        return identifier

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __str__(self) -> str:
        return self._name_hint

    def __repr__(self) -> str:
        return f"{self._name_hint}::{self._id}"
