"""Provenance tracking utilities for compiler objects."""

from __future__ import annotations

__all__ = [
    "Note",
    "NoteKind",
    "Position",
    "Provenance",
    "Span",
]

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, TypeGuard

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    is_serialized_dict,
    register_serializable,
)
from fhy_core.utils import StrEnum


class _PositionData(TypedDict):
    line: int
    column: int


def _is_valid_position_data(data: SerializedDict) -> TypeGuard[_PositionData]:
    return (
        "line" in data
        and isinstance(data["line"], int)
        and "column" in data
        and isinstance(data["column"], int)
    )


@register_serializable(type_id="position")
@dataclass(frozen=True, slots=True, order=True)
class Position(Serializable):
    """Line/column position."""

    line: int
    column: int

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError(f'"line" must be >= 1, got {self.line}')
        if self.column < 1:
            raise ValueError(f'"column" must be >= 1, got {self.column}')

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "line": self.line,
            "column": self.column,
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Position":
        if not _is_valid_position_data(data):
            raise DeserializationDictStructureError(
                cls, _PositionData.__annotations__, data
            )
        try:
            return cls(data["line"], data["column"])
        except ValueError as exc:
            raise DeserializationValueError(f"Invalid position values: {exc}") from exc

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


class _SpanData(TypedDict):
    file_path: str
    start_offset: int | None
    end_offset: int | None
    start_position: _PositionData | None
    end_position: _PositionData | None


def _is_valid_span_data(data: SerializedDict) -> TypeGuard[_SpanData]:
    if not is_serialized_dict(data):
        return False
    if "file_path" not in data or not isinstance(data["file_path"], str):
        return False
    if "start_offset" not in data or not (
        isinstance(data["start_offset"], int) or data["start_offset"] is None
    ):
        return False
    if "end_offset" not in data or not (
        isinstance(data["end_offset"], int) or data["end_offset"] is None
    ):
        return False
    start_position_data = data["start_position"] if "start_position" in data else None
    if "start_position" not in data or not (
        (
            is_serialized_dict(start_position_data)
            and _is_valid_position_data(start_position_data)
        )
        or start_position_data is None
    ):
        return False
    end_position_data = data["end_position"] if "end_position" in data else None
    if "end_position" not in data or not (
        (
            is_serialized_dict(end_position_data)
            and _is_valid_position_data(end_position_data)
        )
        or end_position_data is None
    ):
        return False
    return True


@register_serializable(type_id="span")
@dataclass(frozen=True, slots=True)
class Span(Serializable):
    """A region in a source file."""

    file_path: Path
    start_offset: int | None = None
    end_offset: int | None = None
    start_position: Position | None = None
    end_position: Position | None = None

    def __post_init__(self) -> None:
        if self.start_offset is not None and self.start_offset < 0:
            raise ValueError(f'"start_offset" must be >= 0, got {self.start_offset}')
        if self.end_offset is not None and self.end_offset < 0:
            raise ValueError(f'"end_offset" must be >= 0, got {self.end_offset}')
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset < self.start_offset
        ):
            raise ValueError(
                '"end_offset" must be >= "start_offset", got '
                f"{self.end_offset} < {self.start_offset}"
            )
        if (
            self.start_position is not None
            and self.end_position is not None
            and self.end_position < self.start_position
        ):
            raise ValueError(
                '"end_position" must be >= "start_position", got '
                f"{self.end_position} < {self.start_position}"
            )

    def is_unknown(self) -> bool:
        return (
            self.start_offset is None
            and self.end_offset is None
            and self.start_position is None
            and self.end_position is None
        )

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "file_path": str(self.file_path),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_position": self.start_position.serialize_to_dict()
            if self.start_position
            else None,
            "end_position": self.end_position.serialize_to_dict()
            if self.end_position
            else None,
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Span":
        if not _is_valid_span_data(data):
            raise DeserializationDictStructureError(
                cls, _SpanData.__annotations__, data
            )

        start_position_data = data["start_position"]
        end_position_data = data["end_position"]

        try:
            return cls(
                Path(data["file_path"]),
                start_offset=data["start_offset"],
                end_offset=data["end_offset"],
                start_position=(
                    Position.deserialize_from_dict(start_position_data)
                    if is_serialized_dict(start_position_data)
                    else None
                ),
                end_position=(
                    Position.deserialize_from_dict(end_position_data)
                    if is_serialized_dict(end_position_data)
                    else None
                ),
            )
        except ValueError as exc:
            raise DeserializationValueError(f"Invalid span values: {exc}") from exc

    def __str__(self) -> str:
        file_path = str(self.file_path)
        if self.start_position and self.end_position:
            return f"{file_path}:{self.start_position}-{self.end_position}"
        if self.start_offset is not None and self.end_offset is not None:
            return f"{file_path}@{self.start_offset}-{self.end_offset}"
        return file_path


class NoteKind(StrEnum):
    """Structured note kinds so tooling can filter/group notes."""

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


@register_serializable(type_id="provenance_note")
@dataclass(frozen=True, slots=True)
class Note(Serializable):
    """A provenance breadcrumb."""

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


class _ProvenanceData(TypedDict):
    span: _SpanData | None
    origins: list[SerializedDict]
    notes: list[SerializedDict]
    source_ids: list[str]


def _is_valid_provenance_data(data: SerializedDict) -> TypeGuard[_ProvenanceData]:
    if not is_serialized_dict(data):
        return False
    if "span" not in data or not (
        is_serialized_dict(data["span"])
        and _is_valid_span_data(data["span"])
        or data["span"] is None
    ):
        return False
    if "origins" not in data or not isinstance(data["origins"], list):
        return False
    for origin in data["origins"]:
        if not is_serialized_dict(origin) or not _is_valid_span_data(origin):
            return False
    if "notes" not in data or not isinstance(data["notes"], list):
        return False
    for note in data["notes"]:
        if not is_serialized_dict(note) or not _is_valid_note_data(note):
            return False
    if "source_ids" not in data or not isinstance(data["source_ids"], list):
        return False
    for source_id in data["source_ids"]:
        if not isinstance(source_id, str):
            return False
    return True


@register_serializable(type_id="provenance")
@dataclass(frozen=True, slots=True)
class Provenance(Serializable):
    """Immutable provenance for compiler objects."""

    span: Span | None = None
    origins: tuple[Span, ...] = ()
    notes: tuple[Note, ...] = ()
    source_ids: tuple[str, ...] = ()

    @staticmethod
    def unknown() -> "Provenance":
        """Return a provenance with no information."""
        return Provenance()

    def with_span(self, span: Span) -> "Provenance":
        """Return a new provenance with the span, preserving other metadata."""
        return Provenance(
            span=span,
            origins=self.origins,
            notes=self.notes,
            source_ids=self.source_ids,
        )

    def add_origin(self, origin: Span) -> "Provenance":
        """Return a new provenance with the origin, preserving other metadata."""
        return Provenance(
            span=self.span,
            origins=self.origins + (origin,),
            notes=self.notes,
            source_ids=self.source_ids,
        )

    def add_note(self, note: Note) -> "Provenance":
        """Return a new provenance with the note, preserving other metadata."""
        return Provenance(
            span=self.span,
            origins=self.origins,
            notes=self.notes + (note,),
            source_ids=self.source_ids,
        )

    def add_source_id(self, source_id: str) -> "Provenance":
        """Return a new provenance with the source ID, preserving other metadata."""
        return Provenance(
            span=self.span,
            origins=self.origins,
            notes=self.notes,
            source_ids=self.source_ids + (source_id,),
        )

    def merge(self, *others: "Provenance") -> "Provenance":
        """Merge multiple provenances while preserving insertion order."""
        span = self.span
        origins = list(self.origins)
        notes = list(self.notes)
        source_ids = list(self.source_ids)

        for other in others:
            if span is None and other.span is not None:
                span = other.span
            origins.extend(other.origins)
            notes.extend(other.notes)
            source_ids.extend(other.source_ids)

        return Provenance(
            span=span,
            origins=tuple(origins),
            notes=tuple(notes),
            source_ids=tuple(source_ids),
        )

    def serialize_to_dict(self) -> SerializedDict:
        return {
            "span": self.span.serialize_to_dict() if self.span else None,
            "origins": [origin.serialize_to_dict() for origin in self.origins],
            "notes": [note.serialize_to_dict() for note in self.notes],
            "source_ids": list(self.source_ids),
        }

    @classmethod
    def deserialize_from_dict(cls, data: SerializedDict) -> "Provenance":
        if not _is_valid_provenance_data(data):
            raise DeserializationDictStructureError(
                cls, _ProvenanceData.__annotations__, data
            )

        span_data = data["span"]
        return cls(
            span=Span.deserialize_from_dict(span_data)
            if is_serialized_dict(span_data)
            else None,
            origins=tuple(
                Span.deserialize_from_dict(origin) for origin in data["origins"]
            ),
            notes=tuple(Note.deserialize_from_dict(note) for note in data["notes"]),
            source_ids=tuple(data["source_ids"]),
        )
