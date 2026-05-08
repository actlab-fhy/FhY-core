"""Diagnostic message types.

This module provides ``Note`` and ``NoteKind`` for structured diagnostic
messages emitted by compiler passes and other tooling. These types were
previously housed in ``fhy_core.provenance``; they have been moved here
because they describe diagnostic content, not the origin of compiler
objects.
"""

__all__ = [
    "Note",
    "NoteKind",
]

from dataclasses import dataclass
from typing import TypedDict, TypeGuard

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from fhy_core.trait.equality import EqualMixin
from fhy_core.trait.frozen import FrozenMixin
from fhy_core.utils import StrEnum


class NoteKind(StrEnum):
    """Structured note kinds so tooling can filter and group notes."""

    OTHER = "other"


class _NoteData(TypedDict):
    message: str
    kind: str


def _is_valid_note_data(data: SerializedDict) -> TypeGuard[_NoteData]:
    return (
        "message" in data
        and isinstance(data["message"], str)
        and "kind" in data
        and isinstance(data["kind"], str)
    )


@register_serializable(type_id="diagnostic_note")
@dataclass(frozen=True, slots=True)
class Note(Serializable, FrozenMixin, EqualMixin):
    """A structured diagnostic message with an optional kind tag."""

    message: str
    kind: NoteKind = NoteKind.OTHER

    def serialize_to_dict(self) -> SerializedDict:
        return {"message": self.message, "kind": self.kind.value}

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Note":
        if not _is_valid_note_data(data):
            raise DeserializationDictStructureError(
                cls, _NoteData.__annotations__, data
            )
        try:
            return cls(data["message"], kind=NoteKind(data["kind"]))
        except ValueError as exc:
            raise DeserializationValueError(f"Invalid note values: {exc}") from exc

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"
