"""Provenance tracking for compiler objects.

This module exposes a ``Provenance`` abstract base class and the five
canonical variants used by MLIR's ``Location`` attribute family. Each
variant captures one well-defined concept; richer concepts (builtins,
library symbols, lowered objects) are expressed by composition rather than
by adding new variants.

Variants:

- ``UnknownProvenance``: no source information is available.
- ``FileProvenance``: a region within a source code file.
- ``NamedProvenance``: wraps a child provenance with a human-readable name.
- ``CallSiteProvenance``: a value created at a call site (inlining,
    macro expansion).
- ``FusedProvenance``: N provenances combined by a transformation.

Combining provenances during transformations is done through
``Provenance.fuse``, which applies a small set of reduction rules to keep
fusion trees compact.
"""

__all__ = [
    "CallSiteProvenance",
    "FileProvenance",
    "FusedProvenance",
    "NamedProvenance",
    "Position",
    "Provenance",
    "Span",
    "UnknownProvenance",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, TypeGuard

from fhy_core.logger import get_logger
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    WrappedFamilySerializable,
    is_serialized_dict,
    register_serializable,
)
from fhy_core.trait.equality import EqualMixin
from fhy_core.trait.frozen import FrozenMixin
from fhy_core.utils.numeric_utils import is_strict_int

_LOGGER = get_logger(__name__)


class _PositionData(TypedDict):
    line: int
    column: int


def _is_valid_position_data(data: SerializedDict) -> TypeGuard[_PositionData]:
    return (
        "line" in data
        and is_strict_int(data["line"])
        and "column" in data
        and is_strict_int(data["column"])
    )


@register_serializable(type_id="position")
@dataclass(frozen=True, slots=True, order=True)
class Position(Serializable, FrozenMixin, EqualMixin):
    """A 1-indexed line/column position in a source text."""

    line: int
    column: int

    @property
    def supports_ordering(self) -> bool:
        return True

    @property
    def supports_partial_ordering(self) -> bool:
        return True

    def __post_init__(self) -> None:
        if not is_strict_int(self.line):
            raise TypeError(
                f'"line" must be a strict int, got {type(self.line).__name__}'
            )
        if not is_strict_int(self.column):
            raise TypeError(
                f'"column" must be a strict int, got {type(self.column).__name__}'
            )
        if self.line < 1:
            raise ValueError(f'"line" must be >= 1, got {self.line}')
        if self.column < 1:
            raise ValueError(f'"column" must be >= 1, got {self.column}')

    def serialize_to_dict(self) -> SerializedDict:
        return {"line": self.line, "column": self.column}

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
    start_offset: int | None
    end_offset: int | None
    start_position: _PositionData | None
    end_position: _PositionData | None


def _is_valid_optional_int(value: object) -> bool:
    return value is None or is_strict_int(value)


def _is_valid_optional_position_data(
    value: object,
) -> TypeGuard[_PositionData | None]:
    if value is None:
        return True
    return is_serialized_dict(value) and _is_valid_position_data(value)


def _is_valid_span_data(data: SerializedDict) -> TypeGuard[_SpanData]:
    if not is_serialized_dict(data):
        return False
    for offset_key in ("start_offset", "end_offset"):
        if offset_key not in data or not _is_valid_optional_int(data[offset_key]):
            return False
    for position_key in ("start_position", "end_position"):
        if position_key not in data or not _is_valid_optional_position_data(
            data[position_key]
        ):
            return False
    return True


@register_serializable(type_id="span")
@dataclass(frozen=True, slots=True)
class Span(Serializable, FrozenMixin, EqualMixin):
    """A file-agnostic byte/position range.

    Owned by ``FileProvenance``; does not carry a file path itself.
    """

    start_offset: int | None = None
    end_offset: int | None = None
    start_position: Position | None = None
    end_position: Position | None = None

    def __post_init__(self) -> None:
        for offset_name, offset_value in (
            ("start_offset", self.start_offset),
            ("end_offset", self.end_offset),
        ):
            if offset_value is not None and not is_strict_int(offset_value):
                raise TypeError(
                    f'"{offset_name}" must be a strict int or None, got '
                    f"{type(offset_value).__name__}"
                )
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
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_position": (
                self.start_position.serialize_to_dict()
                if self.start_position is not None
                else None
            ),
            "end_position": (
                self.end_position.serialize_to_dict()
                if self.end_position is not None
                else None
            ),
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
        if self.is_unknown():
            return "<unknown>"
        if self.start_position is not None or self.end_position is not None:
            start = str(self.start_position) if self.start_position is not None else "?"
            end = str(self.end_position) if self.end_position is not None else "?"
            return f"{start}-{end}"
        start_offset = str(self.start_offset) if self.start_offset is not None else "?"
        end_offset = str(self.end_offset) if self.end_offset is not None else "?"
        return f"@{start_offset}-{end_offset}"


class Provenance(WrappedFamilySerializable, FrozenMixin, EqualMixin, ABC):
    """Origin information for a compiler object. Abstract base."""

    @abstractmethod
    def __str__(self) -> str: ...

    @staticmethod
    def unknown() -> "Provenance":
        return UnknownProvenance()

    @staticmethod
    def fuse(*provenances: "Provenance", metadata: str | None = None) -> "Provenance":
        flat: list[Provenance] = []
        pending: list[Provenance] = list(reversed(provenances))
        unknowns_dropped = 0
        fused_collapsed = 0
        while pending:
            provenance = pending.pop()
            if isinstance(provenance, UnknownProvenance):
                unknowns_dropped += 1
                continue
            if isinstance(provenance, FusedProvenance) and provenance.metadata is None:
                fused_collapsed += 1
                pending.extend(reversed(provenance.sources))
                continue
            flat.append(provenance)

        if unknowns_dropped or fused_collapsed:
            _LOGGER.debug(
                "reduced input (inputs=%d, unknowns_dropped=%d, "
                "fused_collapsed=%d, final_sources=%d)",
                len(provenances),
                unknowns_dropped,
                fused_collapsed,
                len(flat),
            )

        if not flat:
            return UnknownProvenance()
        if len(flat) == 1 and metadata is None:
            return flat[0]
        return FusedProvenance(sources=tuple(flat), metadata=metadata)


@register_serializable(type_id="provenance.unknown")
@dataclass(frozen=True, slots=True)
class UnknownProvenance(Provenance):
    """Provenance with no source information."""

    def serialize_data_to_dict(self) -> SerializedDict:
        return {}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "UnknownProvenance":
        if data:
            raise DeserializationDictStructureError(cls, {}, data)
        return cls()

    def __str__(self) -> str:
        return "<unknown>"


class _FileProvenanceData(TypedDict):
    file_path: str
    span: SerializedDict | None


@register_serializable(type_id="provenance.file")
@dataclass(frozen=True, slots=True)
class FileProvenance(Provenance):
    """Provenance pointing to a region within a source code file."""

    file_path: Path
    span: Span | None = None

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "file_path": str(self.file_path),
            "span": self.span.serialize_to_dict() if self.span is not None else None,
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "FileProvenance":
        file_path_value = data.get("file_path")
        span_value = data.get("span")
        if not isinstance(file_path_value, str) or not (
            span_value is None or is_serialized_dict(span_value)
        ):
            raise DeserializationDictStructureError(
                cls, _FileProvenanceData.__annotations__, data
            )
        try:
            return cls(
                Path(file_path_value),
                Span.deserialize_from_dict(span_value)
                if is_serialized_dict(span_value)
                else None,
            )
        except ValueError as exc:
            raise DeserializationValueError(
                f"Invalid file provenance values: {exc}"
            ) from exc

    def __str__(self) -> str:
        if self.span is None or self.span.is_unknown():
            return str(self.file_path)
        else:
            return f"{self.file_path}:{self.span}"


class _NamedProvenanceData(TypedDict):
    name: str
    child: SerializedDict


@register_serializable(type_id="provenance.named")
@dataclass(frozen=True, slots=True)
class NamedProvenance(Provenance):
    """Wraps a child provenance with a human-readable label."""

    name: str
    child: Provenance

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('"name" must be non-empty')

    def serialize_data_to_dict(self) -> SerializedDict:
        return {"name": self.name, "child": self.child.serialize_to_dict()}

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "NamedProvenance":
        name_value = data.get("name")
        child_value = data.get("child")
        if not isinstance(name_value, str) or not is_serialized_dict(child_value):
            raise DeserializationDictStructureError(
                cls, _NamedProvenanceData.__annotations__, data
            )
        try:
            return cls(name_value, Provenance.deserialize_from_dict(child_value))
        except ValueError as exc:
            raise DeserializationValueError(
                f"Invalid named provenance values: {exc}"
            ) from exc

    def __str__(self) -> str:
        if isinstance(self.child, UnknownProvenance):
            return self.name
        else:
            return f"{self.name} ({self.child})"


class _CallSiteProvenanceData(TypedDict):
    callee: SerializedDict
    caller: SerializedDict


@register_serializable(type_id="provenance.call_site")
@dataclass(frozen=True, slots=True)
class CallSiteProvenance(Provenance):
    """Provenance for a value created at a call site."""

    callee: Provenance
    caller: Provenance

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "callee": self.callee.serialize_to_dict(),
            "caller": self.caller.serialize_to_dict(),
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "CallSiteProvenance":
        callee_value = data.get("callee")
        caller_value = data.get("caller")
        if not is_serialized_dict(callee_value) or not is_serialized_dict(caller_value):
            raise DeserializationDictStructureError(
                cls, _CallSiteProvenanceData.__annotations__, data
            )
        return cls(
            callee=Provenance.deserialize_from_dict(callee_value),
            caller=Provenance.deserialize_from_dict(caller_value),
        )

    def __str__(self) -> str:
        return f"{self.callee} at {self.caller}"


class _FusedProvenanceData(TypedDict):
    sources: list[SerializedDict]
    metadata: str | None


@register_serializable(type_id="provenance.fused")
@dataclass(frozen=True, slots=True)
class FusedProvenance(Provenance):
    """N provenances combined by a transformation."""

    sources: tuple[Provenance, ...]
    metadata: str | None = None

    def serialize_data_to_dict(self) -> SerializedDict:
        return {
            "sources": [source.serialize_to_dict() for source in self.sources],
            "metadata": self.metadata,
        }

    @classmethod
    def deserialize_data_from_dict(cls, data: SerializedDict) -> "FusedProvenance":
        sources_value = data.get("sources")
        metadata_value = data.get("metadata")
        if not isinstance(sources_value, list) or not (
            metadata_value is None or isinstance(metadata_value, str)
        ):
            raise DeserializationDictStructureError(
                cls, _FusedProvenanceData.__annotations__, data
            )
        for source in sources_value:
            if not is_serialized_dict(source):
                raise DeserializationDictStructureError(
                    cls, _FusedProvenanceData.__annotations__, data
                )
        return cls(
            sources=tuple(
                Provenance.deserialize_from_dict(source) for source in sources_value
            ),
            metadata=metadata_value,
        )

    def __str__(self) -> str:
        label = self.metadata if self.metadata is not None else "fused"
        rendered_sources = ", ".join(str(source) for source in self.sources)
        return f"{label}[{rendered_sources}]"
