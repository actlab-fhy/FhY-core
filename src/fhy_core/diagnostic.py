"""Diagnostic message and report types.

Provides the layer-agnostic vocabulary for structured diagnostics:

- :class:`Note` and :class:`NoteKind` describe an individual message. The
  shipped note kinds hold fixed identifier ids, the same in every process
  and on either backend. On the Rust backend
  (``fhy_core.RUST_BACKEND_SELECTED``), ``NoteKind`` is backed by the Rust
  implementation, whose registry is the only one in the process:
  constructing a kind whose name is registered returns the canonical
  instance itself, and the registry cannot be cleared.
- :class:`DiagnosticLevel` classifies a message as ERROR, WARNING, or
  INFO.
- :class:`Diagnostic` bundles a level, a :class:`Note`, the source
  identifier of whatever emitted it, and an optional detail string.
- :class:`ValidationReport` aggregates :class:`Diagnostic` instances
  plus a generic sequence of per-source execution records.
- :class:`ValidationFailedError` is raised when a report with ERROR
  diagnostics is escalated via :meth:`ValidationReport.raise_if_failed`.

On the Rust backend, ``Note``, ``Diagnostic`` and ``ValidationReport`` are
backed by the Rust implementation too, with the same API, text and pickles;
their arguments are type-checked at construction. ``DiagnosticLevel`` and
``ValidationFailedError`` stay Python classes on both backends.
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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fhy_core._backend import IS_RUST_BACKEND_SELECTED
from fhy_core.error import register_error
from fhy_core.identifier import (
    _RESERVED_OTHER_NOTE_KIND,
    _RESERVED_RATIONALE_NOTE_KIND,
    _RESERVED_REMARK_NOTE_KIND,
    _RESERVED_SUGGESTION_NOTE_KIND,
    HasIdentifier,
    Identifier,
    _build_reserved_identifier,
)
from fhy_core.serialization import (
    Serializable,
    SerializedDict,
    register_serializable,
)
from fhy_core.term import AlphaEquivalenceMixin, DerivedEquivalenceMixin
from fhy_core.traits import (
    EqualMixin,
    FrozenMixin,
    InternedMixin,
    PartialEqualMixin,
    StructuralEquivalence,
)
from fhy_core.utils import StrEnum

if TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED:

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

        A :class:`Note` is a self-contained, human-readable explanation
        captured at one point in time; it is not part of the IR graph. It holds
        no live references: nothing maintains a note through transformation
        (unlike provenance, which is fused as passes rewrite the IR), so a note
        must never point at a node, span, or definition that a later pass could
        invalidate. A ``NoteKind`` names the *role* of that explanation (why
        something happened, a suggestion, a neutral remark) so tooling can
        filter and group notes regardless of where they are attached (a
        diagnostic, a pass report, a search log, an error).

        It is an open class with canonical interning, like
        :class:`fhy_core.value_domain.ValueDomain`, so a downstream layer
        registers its own kinds without modifying ``fhy_core``. Only the
        universally meaningful, reference-free roles are shipped here; import
        those names rather than constructing a fresh ``NoteKind`` with the same
        ``name_hint``, since ``Identifier`` uses id-equality.

        Note kinds are deliberately distinct from provenance: an object's
        origin and transformation history are tracked authoritatively by
        :mod:`fhy_core.provenance`, not re-encoded as notes.

        ``description`` is human-readable metadata only; it does not
        participate in equality, structural equivalence, hashing, or interning.
        The first instance registered for a given ``Identifier`` becomes
        canonical; deserializing a payload whose description differs from the
        canonical's emits a warning. Pickling a kind stores its payload, so
        unpickling returns the canonical instance.

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

        @override
        def __reduce__(
            self,
        ) -> tuple[Callable[[SerializedDict], "NoteKind"], tuple[SerializedDict]]:
            return (NoteKind.deserialize_from_dict, (self.serialize_to_dict(),))

        @classmethod
        @override
        def register_default_instances(cls) -> None:
            """Re-register the canonical default note kinds shipped here.

            After :meth:`clear_interned_registry` wipes the registry, call this
            method to restore the module-level constants so they remain
            canonical.
            """
            for instance in _DEFAULT_NOTE_KINDS:
                instance.register_interned_instance()

    RATIONALE_NOTE_KIND: NoteKind = NoteKind(
        _build_reserved_identifier(_RESERVED_RATIONALE_NOTE_KIND),
        "Explains why a decision, transformation, or result occurred.",
    )
    SUGGESTION_NOTE_KIND: NoteKind = NoteKind(
        _build_reserved_identifier(_RESERVED_SUGGESTION_NOTE_KIND),
        "A suggested fix or course of action.",
    )
    REMARK_NOTE_KIND: NoteKind = NoteKind(
        _build_reserved_identifier(_RESERVED_REMARK_NOTE_KIND),
        "A neutral informational observation.",
    )
    OTHER_NOTE_KIND: NoteKind = NoteKind(
        _build_reserved_identifier(_RESERVED_OTHER_NOTE_KIND),
        "Uncategorized note.",
    )

else:
    from fhy_core import _rs

    @register_serializable(type_id="note_kind")
    class NoteKind(
        _rs.NoteKind,
        HasIdentifier,
        StructuralEquivalence,
        AlphaEquivalenceMixin,
        Serializable,
    ):
        """Open, registry-backed classification of an explanatory note's role.

        Backed by the Rust implementation: the Rust registry holds the
        canonical kinds, and ``fhy_core._rs.NoteKind`` implements the
        attributes, equality, hashing, interning, payloads and ``str``, which
        renders the name hint. This class mixes in the stateless Python
        protocols, and is registered as a virtual subclass of
        ``InternedMixin`` and ``FrozenMixin``.

        Constructing a kind whose ``name`` is registered returns the canonical
        instance itself, keeping its description. Kinds are immutable, compare
        and hash by ``name``, and pickle as their payload, so unpickling
        returns the canonical instance. The registry is append-only:
        ``clear_interned_registry`` and ``register_default_instances`` raise
        ``NotImplementedError``.

        Attributes:
            name: Stable, process-global identifier for this kind.
            description: Short human-readable description (excluded from
                equality and structural equivalence).

        """

        __slots__ = ()
        __new__ = staticmethod(_rs.NoteKind._new_canonical)

    InternedMixin.register(NoteKind)
    FrozenMixin.register(NoteKind)
    NoteKind._register_public_class()

    RATIONALE_NOTE_KIND = NoteKind.require_interned(
        _build_reserved_identifier(_RESERVED_RATIONALE_NOTE_KIND)
    )
    SUGGESTION_NOTE_KIND = NoteKind.require_interned(
        _build_reserved_identifier(_RESERVED_SUGGESTION_NOTE_KIND)
    )
    REMARK_NOTE_KIND = NoteKind.require_interned(
        _build_reserved_identifier(_RESERVED_REMARK_NOTE_KIND)
    )
    OTHER_NOTE_KIND = NoteKind.require_interned(
        _build_reserved_identifier(_RESERVED_OTHER_NOTE_KIND)
    )

_DEFAULT_NOTE_KINDS: tuple[NoteKind, ...] = (
    RATIONALE_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    REMARK_NOTE_KIND,
    OTHER_NOTE_KIND,
)


class DiagnosticLevel(StrEnum):
    """Severity levels for structured diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


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


if TYPE_CHECKING or not IS_RUST_BACKEND_SELECTED:

    @register_serializable(type_id="diagnostic_note")
    @dataclass(frozen=True, slots=True)
    class Note(Serializable, FrozenMixin, EqualMixin):
        """A structured diagnostic message with an optional kind tag.

        Pickling a note stores a call of its class with its fields, so a
        pickle loads under either backend.
        """

        message: str
        kind: NoteKind = OTHER_NOTE_KIND

        @override
        def __str__(self) -> str:
            return f"{self.kind}: {self.message}"

        @override
        def __reduce__(self) -> tuple[type["Note"], tuple[str, NoteKind]]:
            return (type(self), (self.message, self.kind))

    @dataclass(frozen=True)
    class Diagnostic(FrozenMixin, PartialEqualMixin):
        """A structured diagnostic emitted by a named source.

        Pickling a diagnostic stores a call of its class with its fields, so
        a pickle loads under either backend.

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

        @override
        def __reduce__(
            self,
        ) -> tuple[type["Diagnostic"], tuple[DiagnosticLevel, Note, str, str | None]]:
            return (type(self), (self.level, self.message, self.source, self.detail))

    @dataclass(frozen=True)
    class ValidationReport(FrozenMixin, PartialEqualMixin, Generic[_RecordT]):
        """Aggregated diagnostics plus optional per-source execution records.

        Generic over the record type. The pass infrastructure specializes
        it with :class:`PassRunRecord`; non-pass callers leave the parameter
        unbound and produce a report with no records. Pickling a report
        stores a call of its class with its fields, so a pickle loads under
        either backend.

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
            return tuple(
                d for d in self.diagnostics if d.level == DiagnosticLevel.ERROR
            )

        def warnings(self) -> tuple[Diagnostic, ...]:
            """Return only the WARNING-level diagnostics."""
            return tuple(
                d for d in self.diagnostics if d.level == DiagnosticLevel.WARNING
            )

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

        @override
        def __reduce__(
            self,
        ) -> tuple[
            type["ValidationReport[_RecordT]"],
            tuple[tuple[Diagnostic, ...], tuple[_RecordT, ...]],
        ]:
            return (type(self), (self.diagnostics, self.records))

else:

    @register_serializable(type_id="diagnostic_note")
    class Note(_rs.Note, Serializable, EqualMixin):
        """A structured diagnostic message with an optional kind tag.

        Backed by the Rust implementation: ``fhy_core._rs.Note`` holds the
        Rust note and implements the fields, equality, hashing, ``str``,
        ``repr`` and payloads. This class mixes in the stateless Python
        protocols, and is registered as a virtual subclass of
        ``FrozenMixin``. ``message`` must be a ``str`` and ``kind`` a
        :class:`NoteKind`; either raises ``TypeError`` otherwise. Notes are
        immutable, and pickle as a call of their class with their fields.

        Attributes:
            message: The message text.
            kind: The role the note plays, by default
                :data:`OTHER_NOTE_KIND`.

        """

        __slots__ = ()
        __match_args__ = ("message", "kind")

    FrozenMixin.register(Note)
    Note._register_public_class()

    class Diagnostic(_rs.Diagnostic, PartialEqualMixin):
        """A structured diagnostic emitted by a named source.

        Backed by the Rust implementation: ``fhy_core._rs.Diagnostic`` holds
        the Rust diagnostic and implements the fields, ``message_text``,
        equality, hashing and ``repr``. This class mixes in the stateless
        Python protocols, and is registered as a virtual subclass of
        ``FrozenMixin``. ``level`` is converted to a :class:`DiagnosticLevel`
        as ``DiagnosticLevel(level)`` does; ``message`` must be a
        :class:`Note`, ``source`` a ``str`` and ``detail`` a ``str`` or
        ``None``, and each raises ``TypeError`` otherwise. Diagnostics are
        immutable, and pickle as a call of their class with their fields.

        Attributes:
            level: Severity of the diagnostic.
            message: The diagnostic message as a :class:`Note`.
            source: Stable identifier of whatever emitted this diagnostic
                (typically a pass name or a ``<module>.<class>.<method>``
                identifier for non-pass verifiers).
            detail: Optional supplementary string with extended context.

        """

        __slots__ = ()
        __match_args__ = ("level", "message", "source", "detail")

    FrozenMixin.register(Diagnostic)
    Diagnostic._register_public_class()

    class ValidationReport(_rs.ValidationReport, PartialEqualMixin, Generic[_RecordT]):
        """Aggregated diagnostics plus optional per-source execution records.

        Backed by the Rust implementation: ``fhy_core._rs.ValidationReport``
        holds a Rust report over the Python records and implements the
        fields, ``errors``, ``warnings``, ``infos``, ``has_errors``,
        ``format``, ``raise_if_failed``, equality, hashing and ``repr``. This
        class mixes in the stateless Python protocols, and is registered as a
        virtual subclass of ``FrozenMixin``. Each argument may be any
        iterable and is stored as a tuple; every diagnostic must be a
        :class:`Diagnostic`, and raises ``TypeError`` otherwise. Reports are
        immutable, and pickle as a call of their class with their fields.

        Attributes:
            diagnostics: Every diagnostic, in emission order.
            records: Per-source execution metadata, one entry per registered
                source, in pipeline order. Empty for callers that do not run
                a pipeline.

        """

        __slots__ = ()
        __match_args__ = ("diagnostics", "records")

    FrozenMixin.register(ValidationReport)
    ValidationReport._register_public_class()
