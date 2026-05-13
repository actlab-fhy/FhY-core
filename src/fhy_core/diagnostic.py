"""Diagnostic message and report types.

This module provides the layer-agnostic vocabulary for structured
diagnostics:

- :class:`Note` and :class:`NoteKind` describe an individual message.
- :class:`DiagnosticLevel` classifies a message as ERROR, WARNING, or
  INFO.
- :class:`Diagnostic` bundles a level, a :class:`Note`, the source
  identifier of whatever emitted it, and an optional detail string.
- :class:`ValidationReport` aggregates multiple :class:`Diagnostic`
  instances plus an opaque sequence of per-source execution records. The
  ``records`` field is generic so the pass infrastructure can specialize
  it without forcing this module to depend on pass-specific types.
- :class:`ValidationFailedError` is the exception raised when a report
  with ERROR diagnostics is escalated via
  :meth:`ValidationReport.raise_if_failed`.

These types live here, rather than in ``fhy_core.pass_infrastructure``,
because the abstraction belongs to the diagnostic layer: a non-pass
caller (a self-verifying lattice, a symbol table) needs to construct a
report without pulling in pass-execution machinery.
"""

__all__ = [
    "Diagnostic",
    "DiagnosticLevel",
    "Note",
    "NoteKind",
    "ValidationFailedError",
    "ValidationReport",
]

from dataclasses import dataclass, field
from typing import Any, Generic, TypedDict, TypeGuard, TypeVar

from fhy_core.error import register_error
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    Serializable,
    SerializedDict,
    register_serializable,
)
from fhy_core.trait.equality import EqualMixin, PartialEqualMixin
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


class DiagnosticLevel(StrEnum):
    """Severity levels for structured diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Diagnostic(FrozenMixin, PartialEqualMixin):
    """A structured diagnostic emitted by a named source.

    Attributes:
        level: Severity of the diagnostic.
        message: The diagnostic message as a :class:`Note`.
        source: Stable identifier of whatever emitted this diagnostic
            (typically a pass name or a ``<module>.<class>.<method>``
            identifier for non-pass verifiers).
        detail: Optional supplementary string with extended context.

    """

    level: DiagnosticLevel
    message: Note
    source: str
    detail: str | None = None

    @property
    def message_text(self) -> str:
        """Return the underlying message text without the kind prefix."""
        return self.message.message


_RecordT = TypeVar("_RecordT")


@register_error
class ValidationFailedError(RuntimeError):
    """Raised when a :class:`ValidationReport` is escalated and has errors.

    The triggering report is available via :attr:`report`. The exception
    message is the formatted report, so uncaught exceptions print a
    useful summary.
    """

    _report: "ValidationReport[Any]"

    def __init__(self, report: "ValidationReport[Any]") -> None:
        super().__init__(report.format())
        self._report = report

    @property
    def report(self) -> "ValidationReport[Any]":
        """Return the validation report that triggered this failure."""
        return self._report


@dataclass(frozen=True)
class ValidationReport(FrozenMixin, PartialEqualMixin, Generic[_RecordT]):
    """Aggregated diagnostics plus optional per-source execution records.

    ``ValidationReport`` is generic over its record type so the
    diagnostic layer does not depend on pass-execution machinery. The
    pass infrastructure specializes it with ``PassRunRecord``; non-pass
    callers (for example a self-verifying lattice) leave the parameter
    unbound and produce a report with no records.

    Attributes:
        diagnostics: Every diagnostic, in emission order.
        records: Per-source execution metadata, one entry per registered
            source, in pipeline order. Empty for callers that do not run
            a pipeline.

    """

    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    records: tuple[_RecordT, ...] = field(default_factory=tuple)

    def errors(self) -> tuple[Diagnostic, ...]:
        """Return only the ERROR-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.ERROR)

    def warnings(self) -> tuple[Diagnostic, ...]:
        """Return only the WARNING-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.WARNING)

    def infos(self) -> tuple[Diagnostic, ...]:
        """Return only the INFO-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.INFO)

    def has_errors(self) -> bool:
        """Return True when at least one ERROR-level diagnostic is present."""
        return any(d.level == DiagnosticLevel.ERROR for d in self.diagnostics)

    def format(self) -> str:
        """Return a human-readable rendering of every diagnostic.

        Each diagnostic is rendered on its own line as
        ``[LEVEL] <source>: <message>``; optional detail is appended on an
        indented continuation line.
        """
        if not self.diagnostics:
            return "No validation diagnostics."
        lines: list[str] = []
        for diagnostic in self.diagnostics:
            prefix = f"[{diagnostic.level.value.upper()}] {diagnostic.source}: "
            body = diagnostic.message_text
            lines.append(f"{prefix}{body}")
            if diagnostic.detail:
                lines.append(f"    detail: {diagnostic.detail}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise :class:`ValidationFailedError` if any ERROR diagnostics exist.

        No-op when the report contains only warnings/infos or nothing at all.
        """
        if self.has_errors():
            raise ValidationFailedError(self)
