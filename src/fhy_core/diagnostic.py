"""Diagnostic message and report types.

Provides the layer-agnostic vocabulary for structured diagnostics:

- :class:`Note` and :class:`NoteKind` describe an individual message.
- :class:`DiagnosticLevel` classifies a message as ERROR, WARNING, or
  INFO.
- :class:`Diagnostic` bundles a level, a :class:`Note`, the source
  identifier of whatever emitted it, and an optional detail string.
- :class:`ValidationReport` aggregates :class:`Diagnostic` instances
  plus a generic sequence of per-source execution records.
- :class:`ValidationFailedError` is raised when a report with ERROR
  diagnostics is escalated via :meth:`ValidationReport.raise_if_failed`.
"""

from fhy_core.utils.override import override

__all__ = [
    "OTHER_NOTE_KIND",
    "RATIONALE_NOTE_KIND",
    "REMARK_NOTE_KIND",
    "SUGGESTION_NOTE_KIND",
    "Diagnostic",
    "DiagnosticLevel",
    "Note",
    "NoteKind",
    "ValidationFailedError",
    "ValidationReport",
]

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    Serializable,
    register_serializable,
)
from fhy_core.traits import (
    DerivedEquivalenceMixin,
    HasIdentifier,
    InternedMixin,
)
from fhy_core.traits.equality import EqualMixin, PartialEqualMixin
from fhy_core.traits.frozen import FrozenMixin
from fhy_core.utils import StrEnum


@register_serializable(type_id="note_kind")
@dataclass(frozen=True)
class NoteKind(
    HasIdentifier,
    FrozenMixin,
    DerivedEquivalenceMixin,
    InternedMixin[Identifier],
    Serializable,
):
    """Open, registry-backed classification of an explanatory note's role.

    A :class:`Note` is a self-contained, human-readable explanation captured at
    one point in time; it is not part of the IR graph. It holds no live
    references: nothing maintains a note through transformation (unlike
    provenance, which is fused as passes rewrite the IR), so a note must never
    point at a node, span, or definition that a later pass could invalidate. A
    ``NoteKind`` names the *role* of that explanation (why something happened, a
    suggestion, a neutral remark) so tooling can filter and group notes
    regardless of where they are attached (a diagnostic, a pass report, a search
    log, an error).

    It is an open class with canonical interning, like
    :class:`fhy_core.value_domain.ValueDomain`, so a downstream layer registers
    its own kinds without modifying ``fhy_core``. Only the universally
    meaningful, reference-free roles are shipped here; import those names rather
    than constructing a fresh ``NoteKind`` with the same ``name_hint``, since
    ``Identifier`` uses id-equality.

    Note kinds are deliberately distinct from provenance: an object's origin and
    transformation history are tracked authoritatively by
    :mod:`fhy_core.provenance`, not re-encoded as notes.

    ``description`` is human-readable metadata only; it does not participate in
    equality, structural equivalence, hashing, or interning. The first instance
    registered for a given ``Identifier`` becomes canonical; deserializing a
    payload whose description differs from the canonical's emits a warning.

    Attributes:
        name: Stable, process-global identifier for this kind.
        description: Short human-readable description (excluded from
            structural equivalence).

    """

    name: Identifier
    description: str = field(compare=False)

    def __post_init__(self) -> None:
        self.register_interned_instance()

    @override
    def __str__(self) -> str:
        return str(self.name)

    @override
    def get_identifier(self) -> Identifier:
        return self.name

    @override
    def get_intern_key(self) -> Identifier:
        return self.name

    @classmethod
    @override
    def register_default_instances(cls) -> None:
        """Re-register the canonical default note kinds shipped here.

        After :meth:`clear_interned_registry` wipes the registry, call this
        method to restore the module-level constants so they remain canonical.
        """
        for instance in _DEFAULT_NOTE_KINDS:
            instance.register_interned_instance()


RATIONALE_NOTE_KIND: NoteKind = NoteKind(
    Identifier("rationale"),
    "Explains why a decision, transformation, or result occurred.",
)
SUGGESTION_NOTE_KIND: NoteKind = NoteKind(
    Identifier("suggestion"),
    "A suggested fix or course of action.",
)
REMARK_NOTE_KIND: NoteKind = NoteKind(
    Identifier("remark"),
    "A neutral informational observation.",
)
OTHER_NOTE_KIND: NoteKind = NoteKind(
    Identifier("other"),
    "Uncategorized note.",
)

_DEFAULT_NOTE_KINDS: tuple[NoteKind, ...] = (
    RATIONALE_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    REMARK_NOTE_KIND,
    OTHER_NOTE_KIND,
)


@register_serializable(type_id="diagnostic_note")
@dataclass(frozen=True, slots=True)
class Note(Serializable, FrozenMixin, EqualMixin):
    """A structured diagnostic message with an optional kind tag."""

    message: str
    kind: NoteKind = OTHER_NOTE_KIND

    @override
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
        """The underlying message text, without the kind prefix."""
        return self.message.message


_RecordT = TypeVar("_RecordT")


@register_error
class ValidationFailedError(RuntimeError):
    """Raised when a :class:`ValidationReport` is escalated and has errors.

    The triggering report is available via :attr:`report`. The exception
    message is the report's :meth:`ValidationReport.format` output.
    """

    _report: "ValidationReport[Any]"

    def __init__(self, report: "ValidationReport[Any]") -> None:
        super().__init__(report.format())
        self._report = report

    @property
    def report(self) -> "ValidationReport[Any]":
        """The validation report that triggered this failure."""
        return self._report


@dataclass(frozen=True)
class ValidationReport(FrozenMixin, PartialEqualMixin, Generic[_RecordT]):
    """Aggregated diagnostics plus optional per-source execution records.

    Generic over the record type. The pass infrastructure specializes
    it with :class:`PassRunRecord`; non-pass callers leave the parameter
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

        Raises:
            ValidationFailedError: If at least one diagnostic has level
                :attr:`DiagnosticLevel.ERROR`. The error carries this
                report on its :attr:`ValidationFailedError.report`
                attribute.

        """
        if self.has_errors():
            raise ValidationFailedError(self)
