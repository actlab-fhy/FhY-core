"""Diagnostic message and report types.

Provides the layer-agnostic vocabulary for structured diagnostics:

- :class:`Note` and :class:`NoteKind` describe an individual message. The
  shipped note kinds hold fixed identifier ids, the same in every process.
  ``NoteKind`` is backed by the Rust implementation, whose registry is the
  only one in the process: constructing a kind whose name is registered
  returns the canonical instance itself, and the registry cannot be
  cleared.
- :class:`DiagnosticLevel` classifies a message as ERROR, WARNING, or
  INFO.
- :class:`Diagnostic` bundles a level, a :class:`Note`, the source
  identifier of whatever emitted it, and an optional detail string.
- :class:`ValidationReport` aggregates :class:`Diagnostic` instances
  plus a generic sequence of per-source execution records.
- :class:`ValidationFailedError` is raised when a report with ERROR
  diagnostics is escalated via :meth:`ValidationReport.raise_if_failed`.

``Note``, ``Diagnostic`` and ``ValidationReport`` are backed by the Rust
implementation too; their arguments are type-checked at construction.
``DiagnosticLevel`` and ``ValidationFailedError`` are Python classes.
"""

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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fhy_core import _rs
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
from fhy_core.serialization import Serializable, register_serializable
from fhy_core.term import AlphaEquivalenceMixin
from fhy_core.traits import (
    EqualMixin,
    FrozenMixin,
    InternedMixin,
    PartialEqualMixin,
    StructuralEquivalence,
)
from fhy_core.utils import StrEnum
from fhy_core.utils.override import override
from fhy_core.utils.self import Self


@register_serializable(type_id="note_kind")
class NoteKind(
    _rs.NoteKind,
    HasIdentifier,
    StructuralEquivalence,
    AlphaEquivalenceMixin,
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
    ``name_hint``, since ``Identifier`` uses id-equality. Note kinds are
    deliberately distinct from provenance: an object's origin and
    transformation history are tracked by :mod:`fhy_core.provenance`, not
    re-encoded as notes.

    Backed by the Rust implementation: the Rust registry holds the
    canonical kinds, and ``fhy_core._rs.NoteKind`` implements the
    attributes, equality, hashing, interning, payloads and ``str``, which
    renders the name hint. This class mixes in the stateless Python
    protocols, and is registered as a virtual subclass of
    ``InternedMixin`` and ``FrozenMixin``.

    Constructing a kind whose ``name`` is registered returns the canonical
    instance itself, keeping its description; deserializing a payload whose
    description differs logs a warning. Kinds are immutable, compare
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
    if TYPE_CHECKING:
        # The type checkers' view of `_new_canonical`, which the stub cannot
        # type as this class's constructor.
        def __new__(cls, name: Identifier, description: str) -> Self:
            """Return the canonical kind named ``name``."""
            ...

    else:
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

    Generic over the record type. The pass infrastructure specializes it
    with :class:`PassRunRecord`; non-pass callers leave the parameter
    unbound and produce a report with no records.

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

    if TYPE_CHECKING:
        # The stub cannot make `_rs.ValidationReport` generic in the record
        # type, since the extension class cannot be subscripted.
        def __new__(
            cls,
            diagnostics: Iterable[Diagnostic] = (),
            records: Iterable[_RecordT] = (),
        ) -> Self:
            """Return a report of ``diagnostics`` and ``records``."""
            ...

        @property
        @override
        def records(self) -> tuple[_RecordT, ...]:
            """Per-source execution metadata, in pipeline order."""
            ...


FrozenMixin.register(ValidationReport)
ValidationReport._register_public_class()
