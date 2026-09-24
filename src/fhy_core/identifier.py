"""Unique identifier for named compiler objects.

An :class:`Identifier` stores its id and name hint itself and draws new ids
from a process-global counter. The counter is the Rust extension's
(``fhy_core._rs``) when the package runs on the Rust backend
(``fhy_core.RUST_BACKEND_SELECTED``) and a pure-Python counter otherwise.
The backend is fixed when the package is imported, so exactly one counter
issues ids in a process.

The class stays in Python on both backends, and only the counter comes from
the extension. A Rust-backed class makes every attribute read, equality
check, and hash cross into the extension, which is significantly slower than
reading plain Python attributes.
"""

from fhy_core.utils.override import override

__all__ = ["HasIdentifier", "Identifier"]

from collections.abc import Callable
from threading import Lock
from typing import (
    Any,
    Final,
    Protocol,
    TypedDict,
    TypeGuard,
    final,
    runtime_checkable,
)

from fhy_core.utils import is_strict_int

from ._backend import IS_RUST_BACKEND_SELECTED
from .serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from .traits.equality import EqualMixin
from .traits.frozen import FrozenMixin

# Ids below this are reserved for the identifiers the Rust extension ships, so
# the counter starts here. Matches the Rust implementation:
# `fhy_core::identifier::RESERVED_ID_COUNT`.
_RESERVED_ID_COUNT: Final[int] = 65_536
_ID_SPACE_SIZE = 2**64
_EXHAUSTED_COUNTER_VALUE = _ID_SPACE_SIZE - 1
_ID_SPACE_EXHAUSTED_MESSAGE = "identifier id space exhausted"
# The messages the Rust extension raises for an id outside ``[0, 2**64)``.
_NEGATIVE_ID_MESSAGE = "can't convert negative int to unsigned"
_OVERSIZED_ID_MESSAGE = "int too big to convert"


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


def _is_utf8_encodable(text: str) -> bool:
    """Return whether ``text`` has no lone surrogates, so it encodes as UTF-8."""
    if text.isascii():
        return True
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@final
class _PythonIdCounter:
    """Lock-protected id counter that never wraps.

    Behaves like the Rust extension's counter: it starts at ``65_536``,
    above the ids reserved for shipped identifiers, ids are in
    ``[0, 2**64)``, and the largest id issued is ``2**64 - 2``. Allocating
    once the counter has reached ``2**64 - 1``, or advancing past
    ``2**64 - 1``, raises ``RuntimeError`` and leaves the counter unchanged.
    """

    def __init__(self, next_id: int = _RESERVED_ID_COUNT) -> None:
        """Create a counter whose first allocation returns ``next_id``.

        Args:
            next_id: The first id to issue. Tests set it to reach the end of
                the id space; the process-global counter keeps the default.

        """
        self._lock = Lock()
        self._next_id = next_id

    def allocate(self) -> int:
        """Return the next id and advance the counter past it.

        Raises:
            RuntimeError: If the counter has reached ``2**64 - 1``.

        """
        with self._lock:
            if self._next_id >= _EXHAUSTED_COUNTER_VALUE:
                raise RuntimeError(_ID_SPACE_EXHAUSTED_MESSAGE)
            identifier_id = self._next_id
            self._next_id += 1
        return identifier_id

    def advance_past(self, identifier_id: int, /) -> None:
        """Advance the counter so ``identifier_id`` is never allocated.

        The counter is left unchanged when it is already past
        ``identifier_id``.

        Args:
            identifier_id: Id in ``[0, 2**64)`` to advance past.

        Raises:
            OverflowError: If ``identifier_id`` is outside ``[0, 2**64)``.
            RuntimeError: If ``identifier_id`` is ``2**64 - 1``, which the
                counter cannot advance past.

        """
        if identifier_id < 0:
            raise OverflowError(_NEGATIVE_ID_MESSAGE)
        if identifier_id >= _ID_SPACE_SIZE:
            raise OverflowError(_OVERSIZED_ID_MESSAGE)
        if identifier_id == _EXHAUSTED_COUNTER_VALUE:
            raise RuntimeError(_ID_SPACE_EXHAUSTED_MESSAGE)
        with self._lock:
            if identifier_id >= self._next_id:
                self._next_id = identifier_id + 1


_allocate_id: Callable[[], int]
_advance_counter_past: Callable[[int], None]
if IS_RUST_BACKEND_SELECTED:
    from . import _rs

    _allocate_id = _rs.allocate_identifier_id
    _advance_counter_past = _rs.advance_identifier_counter_past
else:
    _PYTHON_ID_COUNTER = _PythonIdCounter()
    _allocate_id = _PYTHON_ID_COUNTER.allocate
    _advance_counter_past = _PYTHON_ID_COUNTER.advance_past


@final
@register_serializable(type_id="id")
class Identifier(Serializable, FrozenMixin, EqualMixin, freeze_on_init=True):
    """Process-globally unique, named compiler symbol.

    Two ``Identifier`` instances are equal iff they share the same ``id``;
    ``name_hint`` is a debugging aid and is not consulted by ``__eq__`` or
    ``__hash__``. Ids are drawn from a single process-global,
    monotonically-increasing counter and are never reused. The ids
    ``0..65_536`` are reserved for the identifiers the Rust extension ships,
    so the counter starts at ``65_536`` on both backends.

    Construction and deserialization are thread-safe and share the same
    counter: a deserialized id cannot collide with a subsequently
    constructed id, regardless of interleaving. Deserializing an id
    greater than or equal to the next-to-be-issued value advances the
    counter past it. Unpickling, copying, and deep-copying restore an
    identifier through deserialization, so they advance the counter the
    same way. A pickle holds only the id and the name hint and loads under
    either backend.

    Ids are unsigned 64-bit integers, and the largest id an identifier ever
    holds is ``2**64 - 2``. Deserialization accepts an int ``id`` with
    ``0 <= id < 2**64 - 1`` and raises ``DeserializationValueError`` for any
    other int. Once ``2**64 - 2`` is issued or restored, the counter cannot
    advance without wrapping and re-issuing a live id, so construction
    raises ``RuntimeError("identifier id space exhausted")`` on both
    backends and leaves the counter unchanged.

    A name hint must be a ``str`` encodable as UTF-8: construction raises
    ``TypeError`` for any other type and ``ValueError`` for a string holding
    a lone surrogate code point (U+D800 to U+DFFF), in both cases without
    consuming an id, and deserialization raises
    ``DeserializationValueError`` for such a string.

    ``repr`` of an ``Identifier`` returns ``"<name_hint>::<id>"``. The form
    is for debugging only. It is not a serialization protocol and is not
    round-trippable through the constructor. Use the structured ``id`` and
    ``name_hint`` properties (or ``serialize_to_dict``) when a structured
    representation is needed.

    The class is ``@final`` and is not intended to be subclassed; callers
    should treat it as a closed implementation that provides a single
    process-global id space.
    """

    _id: int
    _name_hint: str

    def __init__(self, name_hint: str) -> None:
        if not isinstance(name_hint, str):
            raise TypeError(
                f"Identifier name hint must be a str, got {type(name_hint).__name__}."
            )
        if not _is_utf8_encodable(name_hint):
            raise ValueError(
                f"Identifier name hint must be encodable as UTF-8, got {name_hint!r}."
            )
        self._id = _allocate_id()
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
        if data["id"] >= _EXHAUSTED_COUNTER_VALUE:
            raise DeserializationValueError(
                cls, "id", "a non-negative integer below 2**64 - 1", data["id"]
            )
        if not _is_utf8_encodable(data["name_hint"]):
            raise DeserializationValueError(
                cls, "name_hint", "a string encodable as UTF-8", data["name_hint"]
            )
        _advance_counter_past(data["id"])
        identifier = cls.__new__(cls)
        identifier._id = data["id"]
        identifier._name_hint = data["name_hint"]
        identifier.freeze()
        return identifier

    @override
    def __reduce__(
        self,
    ) -> tuple[Callable[[SerializedDict], "Identifier"], tuple[SerializedDict]]:
        return (Identifier.deserialize_from_dict, (self.serialize_to_dict(),))

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
