"""Unique identifier for named compiler objects."""

from fhy_core.utils.override import override

__all__ = ["HasIdentifier", "Identifier"]

from threading import Lock
from typing import (
    Any,
    ClassVar,
    Protocol,
    TypedDict,
    TypeGuard,
    final,
    runtime_checkable,
)

from fhy_core.utils import is_strict_int

from .logger import get_logger
from .serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from .traits.equality import EqualMixin
from .traits.frozen import FrozenMixin

_LOGGER = get_logger(__name__)


class _IdentifierData(TypedDict):
    id: int
    name_hint: str


_IDENTIFIER_DATA_KEYS: frozenset[str] = frozenset({"id", "name_hint"})


def _is_valid_identifier_data(data: SerializedDict) -> TypeGuard[_IdentifierData]:
    if data.keys() != _IDENTIFIER_DATA_KEYS:
        return False
    id_value = data["id"]
    if not is_strict_int(id_value):
        return False
    return isinstance(data["name_hint"], str)


@final
@register_serializable(type_id="id")
class Identifier(Serializable, FrozenMixin, EqualMixin, freeze_on_init=True):
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
    is for debugging only. It is not a serialization protocol and is not
    round-trippable through the constructor. Use the structured ``id`` and
    ``name_hint`` properties (or ``serialize_to_dict``) when a structured
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
        """Return the identifier's name hint."""
        return self._name_hint

    @property
    def id(self) -> int:
        """Return the identifier's unique id."""
        return self._id

    @override
    def serialize_to_dict(self) -> SerializedDict:
        return {"id": self._id, "name_hint": self._name_hint}

    @classmethod
    @override
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
        cls._advance_next_id_past(identifier._id, identifier._name_hint)
        identifier.freeze()
        return identifier

    @classmethod
    def _advance_next_id_past(cls, identifier_id: int, name_hint: str) -> None:
        """Advance the global counter so ``identifier_id`` is never re-issued.

        No-op when the counter is already past ``identifier_id``. Thread-safe:
        the check-and-advance runs under the id lock shared with construction.
        """
        advanced = False
        with cls._id_lock:
            if identifier_id >= cls._next_id:
                cls._next_id = identifier_id + 1
                advanced = True
        if advanced:
            _LOGGER.debug(
                "advanced _next_id past %d (name_hint=%r)", identifier_id, name_hint
            )

    @override
    def __setstate__(self, state: Any) -> None:
        # Mirrors `deserialize_from_dict`: an id restored in a process whose
        # counter has not yet reached it must never be re-issued to a later
        # construction, so unpickling advances the counter the same way
        # deserialization does.
        super().__setstate__(state)
        Identifier._advance_next_id_past(self._id, self._name_hint)

    @override
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identifier) and self._id == other._id

    @override
    def __hash__(self) -> int:
        return hash(self._id)

    @override
    def __str__(self) -> str:
        return self._name_hint

    @override
    def __repr__(self) -> str:
        return f"{self._name_hint}::{self._id}"


@runtime_checkable
class HasIdentifier(Protocol):
    """Protocol for objects that have a stable identifier.

    The protocol lives beside :class:`Identifier` rather than in
    :mod:`fhy_core.traits` because its signature names an identifier,
    which makes it vocabulary of this module rather than a generic
    structural contract.
    """

    def get_identifier(self) -> Identifier:
        """Return the object's stable identifier."""
