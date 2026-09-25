"""Tests the Python interface of the Rust-backed diagnostics.

``Note``, ``Diagnostic`` and ``ValidationReport`` are thin Python subclasses
of the ``fhy_core._rs`` classes over the Rust values, and ``DiagnosticLevel``
converts by value at the boundary (slice S3 of
``docs/design/python-switch.md``). Their behavioral suites cover the
diagnostics' semantics; this suite covers what the binding adds: construction and its
argument checks, the dataclass reprs and ``format()`` text, equality,
payloads and pickles, frozen errors, and the registration of the public
classes.
"""

import base64
import copy
import io
import pickle
import subprocess
import sys
from typing import Any

import pytest

import fhy_core
from fhy_core.diagnostic import (
    OTHER_NOTE_KIND,
    RATIONALE_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    Diagnostic,
    DiagnosticLevel,
    Note,
    NoteKind,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    Serializable,
    SerializationFormat,
    SerializedDict,
)
from fhy_core.traits import (
    EqualMixin,
    FrozenMixin,
    FrozenMutationError,
    PartialEqualMixin,
)
from fhy_core.utils.override import override

_PUBLIC_CLASSES = pytest.mark.parametrize(
    "cls", [Note, Diagnostic, ValidationReport], ids=lambda cls: cls.__name__
)

_OTHER_KIND_REPR = "NoteKind(name=other::3, description='Uncategorized note.')"


def _build_error(message: str = "bound is negative") -> Diagnostic:
    """Return an error diagnostic with detail."""
    return Diagnostic(
        DiagnosticLevel.ERROR, Note(message), "bounds.check", "bound -1 in loop i"
    )


def _build_mixed_report() -> ValidationReport[Any]:
    """Return a report of one diagnostic per level, with records."""
    return ValidationReport(
        (
            _build_error(),
            Diagnostic(DiagnosticLevel.WARNING, Note("slow loop"), "perf"),
            Diagnostic(DiagnosticLevel.INFO, Note("fyi"), "info.source", ""),
        ),
        ("record", 1),
    )


# =============================================================================
# Class structure
# =============================================================================


@_PUBLIC_CLASSES
def test_public_class_is_a_thin_subclass_of_the_rust_class(cls: type[Any]) -> None:
    """Test the public class subclasses its `_rs` class and mixes in protocols."""
    rust_class = getattr(fhy_core._rs, cls.__name__)

    assert cls.__bases__[0] is rust_class
    assert issubclass(cls, PartialEqualMixin)
    assert issubclass(cls, FrozenMixin)
    assert FrozenMixin not in cls.__mro__


def test_note_is_serializable_with_total_equality() -> None:
    """Test `Note` mixes in `Serializable` and `EqualMixin` with its type id."""
    assert issubclass(Note, Serializable)
    assert issubclass(Note, EqualMixin)
    assert Note.get_serialization_class_type_id() == "diagnostic_note"


@pytest.mark.parametrize("cls", [Diagnostic, ValidationReport])
def test_diagnostic_and_report_are_not_serializable(cls: type[Any]) -> None:
    """Test `Diagnostic` and `ValidationReport` stay non-serializable."""
    assert not issubclass(cls, Serializable)
    assert not issubclass(cls, EqualMixin)


# =============================================================================
# Public class registration
# =============================================================================


@_PUBLIC_CLASSES
def test_registering_the_public_class_again_is_a_no_op(cls: type[Any]) -> None:
    """Test re-registering the registered public class succeeds."""
    cls._register_public_class()


@_PUBLIC_CLASSES
def test_registering_another_public_class_raises(cls: type[Any]) -> None:
    """Test a second public class cannot replace the registered one."""
    subclass = type(f"Other{cls.__name__}", (cls,), {"__slots__": ()})

    with pytest.raises(RuntimeError, match="registered already"):
        subclass._register_public_class()  # type: ignore[attr-defined]  # test: Rust-only


# =============================================================================
# Note
# =============================================================================


def test_note_defaults_to_the_canonical_other_kind() -> None:
    """Test an omitted kind is the canonical `OTHER_NOTE_KIND` object."""
    assert Note("message").kind is OTHER_NOTE_KIND


def test_note_returns_the_canonical_object_of_a_custom_kind() -> None:
    """Test a note's kind is the single Python object of its canonical kind."""
    kind = NoteKind(Identifier("binding-note-kind"), "desc")

    note = Note(message="message", kind=kind)

    assert note.kind is kind
    assert note.message == "message"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param((1,), "Note message must be a str, got int.", id="message"),
        pytest.param(
            ("m", None), "Note kind must be a NoteKind, got NoneType.", id="kind_none"
        ),
        pytest.param(
            ("m", "other"), "Note kind must be a NoteKind, got str.", id="kind_str"
        ),
    ],
)
def test_note_rejects_arguments_of_the_wrong_type(
    arguments: tuple[Any, ...], message: str
) -> None:
    """Test a note checks its argument types, an explicit `None` kind included."""
    with pytest.raises(TypeError) as info:
        Note(*arguments)

    assert str(info.value) == message


def test_note_repr_and_str_match_the_dataclass() -> None:
    """Test a note renders in the dataclass `repr` form."""
    note = Note("it's here", OTHER_NOTE_KIND)

    assert repr(note) == f'Note(message="it\'s here", kind={_OTHER_KIND_REPR})'
    assert str(note) == "other: it's here"


def test_notes_compare_and_hash_by_message_and_kind() -> None:
    """Test notes are equal, and hash equally, exactly when their fields are."""
    note = Note("message", RATIONALE_NOTE_KIND)

    assert note == Note("message", RATIONALE_NOTE_KIND)
    assert hash(note) == hash(Note("message", RATIONALE_NOTE_KIND))
    assert note != Note("message", SUGGESTION_NOTE_KIND)
    assert note != Note("other message", RATIONALE_NOTE_KIND)


def test_note_is_unequal_to_an_instance_of_another_class() -> None:
    """Test equality requires the same class, as a dataclass's does."""
    assert Note("message") != fhy_core._rs.Note("message")
    assert Note("message") != "other: message"


def test_note_payload_keeps_the_python_shape() -> None:
    """Test a note's payload nests its kind's payload."""
    note = Note("message", RATIONALE_NOTE_KIND)

    assert note.serialize_to_dict() == {
        "message": "message",
        "kind": RATIONALE_NOTE_KIND.serialize_to_dict(),
    }


@pytest.mark.parametrize("fmt", list(SerializationFormat))
def test_note_round_trips_to_the_canonical_kind(fmt: SerializationFormat) -> None:
    """Test a note round-trips in every format and keeps the canonical kind."""
    note = Note("message", SUGGESTION_NOTE_KIND)

    restored = Note.deserialize(note.serialize(fmt), fmt)

    assert restored == note
    assert type(restored) is Note
    assert restored.kind is SUGGESTION_NOTE_KIND


def test_note_payload_registers_a_new_kind() -> None:
    """Test decoding a note whose kind is unregistered registers the kind."""
    name = Identifier("binding-decoded-kind")
    payload: SerializedDict = {
        "message": "message",
        "kind": {"name": name.serialize_to_dict(), "description": "decoded"},
    }

    note = Note.deserialize_from_dict(payload)

    assert NoteKind.get_interned(name) is note.kind
    assert note.kind.description == "decoded"


@pytest.mark.parametrize(
    ("data", "owner"),
    [
        pytest.param({"message": "x"}, "Note", id="missing_kind"),
        pytest.param(
            {"message": 1, "kind": OTHER_NOTE_KIND.serialize_to_dict()},
            "Note",
            id="message_not_str",
        ),
        pytest.param(
            {"message": "x", "kind": {"name": 1, "description": "d"}},
            "NoteKind",
            id="malformed_kind",
        ),
    ],
)
def test_malformed_note_payload_raises_the_python_structure_error(
    data: SerializedDict, owner: str
) -> None:
    """Test a malformed payload raises the framework's structure error."""
    with pytest.raises(DeserializationDictStructureError) as info:
        Note.deserialize_from_dict(data)

    assert str(info.value).startswith(
        f'Invalid dictionary structure for deserializing to "{owner}".'
    )


# =============================================================================
# Diagnostic
# =============================================================================


def test_diagnostic_holds_its_fields() -> None:
    """Test a diagnostic returns its fields, and the very note it was given."""
    note = Note("bound is negative")

    diagnostic = Diagnostic(
        level=DiagnosticLevel.WARNING, message=note, source="bounds.check"
    )

    assert diagnostic.level is DiagnosticLevel.WARNING
    assert diagnostic.message is note
    assert diagnostic.source == "bounds.check"
    assert diagnostic.detail is None
    assert diagnostic.message_text == "bound is negative"


def test_diagnostic_converts_a_level_value_to_the_enum_member() -> None:
    """Test a level is converted as `DiagnosticLevel(level)` converts it."""
    diagnostic = Diagnostic("info", Note("m"), "source")

    assert diagnostic.level is DiagnosticLevel.INFO


def test_diagnostic_rejects_a_value_that_names_no_level() -> None:
    """Test an unknown level raises the enum's own `ValueError`."""
    with pytest.raises(ValueError) as info:
        Diagnostic("fatal", Note("m"), "source")

    assert str(info.value) == "'fatal' is not a valid DiagnosticLevel"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        pytest.param(
            (DiagnosticLevel.ERROR, "m", "s"),
            "Diagnostic message must be a Note, got str.",
            id="message",
        ),
        pytest.param(
            (DiagnosticLevel.ERROR, Note("m"), 1),
            "Diagnostic source must be a str, got int.",
            id="source",
        ),
        pytest.param(
            (DiagnosticLevel.ERROR, Note("m"), "s", 1),
            "Diagnostic detail must be a str, got int.",
            id="detail",
        ),
    ],
)
def test_diagnostic_rejects_arguments_of_the_wrong_type(
    arguments: tuple[Any, ...], message: str
) -> None:
    """Test a diagnostic checks its argument types."""
    with pytest.raises(TypeError) as info:
        Diagnostic(*arguments)

    assert str(info.value) == message


def test_diagnostic_repr_matches_the_dataclass() -> None:
    """Test a diagnostic renders in the dataclass `repr` form."""
    diagnostic = Diagnostic(DiagnosticLevel.ERROR, Note("m"), "s", "d")

    assert repr(diagnostic) == (
        "Diagnostic(level=<DiagnosticLevel.ERROR: 'error'>, "
        f"message=Note(message='m', kind={_OTHER_KIND_REPR}), "
        "source='s', detail='d')"
    )
    assert str(diagnostic) == repr(diagnostic)


def test_diagnostics_compare_and_hash_by_every_field() -> None:
    """Test diagnostics are equal, and hash equally, exactly when their fields are."""
    diagnostic = _build_error()

    assert diagnostic == _build_error()
    assert hash(diagnostic) == hash(_build_error())
    assert diagnostic != _build_error("another message")
    assert diagnostic != Diagnostic(
        DiagnosticLevel.WARNING, Note("bound is negative"), "bounds.check"
    )


def test_diagnostic_has_partial_but_not_total_equality() -> None:
    """Test a diagnostic reports partial equality and has no `supports_equality`."""
    diagnostic = _build_error()

    assert diagnostic.supports_partial_equality is True
    assert not hasattr(diagnostic, "supports_equality")


# =============================================================================
# ValidationReport
# =============================================================================


def test_report_defaults_to_no_diagnostics_and_no_records() -> None:
    """Test an argument-less report holds two empty tuples."""
    report: ValidationReport[Any] = ValidationReport()

    assert report.diagnostics == ()
    assert report.records == ()
    assert not report.has_errors()


def test_report_holds_the_given_tuples() -> None:
    """Test a report returns the very tuples, and diagnostics, it was given."""
    diagnostics = (_build_error(),)
    records = ("record",)

    report = ValidationReport(diagnostics=diagnostics, records=records)

    assert report.diagnostics is diagnostics
    assert report.records is records


def test_report_stores_any_iterable_as_a_tuple() -> None:
    """Test a report accepts any iterable and stores it as a tuple."""
    diagnostic = _build_error()

    report: ValidationReport[str] = ValidationReport(
        [diagnostic],
        iter(["record"]),
    )

    assert report.diagnostics == (diagnostic,)
    assert report.diagnostics[0] is diagnostic
    assert report.records == ("record",)


def test_report_rejects_a_diagnostic_that_is_not_a_diagnostic() -> None:
    """Test every diagnostic of a report must be a `Diagnostic`."""
    with pytest.raises(TypeError) as info:
        ValidationReport((Note("m"),))  # type: ignore[arg-type]  # test: wrong type

    assert str(info.value) == (
        "ValidationReport diagnostics must be Diagnostic instances, got Note."
    )


def test_report_filters_return_tuples_of_the_diagnostics() -> None:
    """Test `errors`, `warnings` and `infos` return the report's own objects."""
    report = _build_mixed_report()
    error, warning, info = report.diagnostics

    assert report.errors() == (error,)
    assert report.errors()[0] is error
    assert report.warnings()[0] is warning
    assert report.infos()[0] is info
    assert report.has_errors()


def test_report_format_matches_the_python_text() -> None:
    """Test `format()` renders one line per diagnostic, skipping empty detail."""
    assert _build_mixed_report().format() == (
        "[ERROR] bounds.check: bound is negative\n"
        "    detail: bound -1 in loop i\n"
        "[WARNING] perf: slow loop\n"
        "[INFO] info.source: fyi"
    )
    assert ValidationReport().format() == "No validation diagnostics."


def test_raise_if_failed_raises_the_python_error_with_the_report() -> None:
    """Test a report with errors raises `ValidationFailedError` carrying it."""
    report = _build_mixed_report()

    with pytest.raises(ValidationFailedError) as info:
        report.raise_if_failed()

    assert type(info.value) is ValidationFailedError
    assert info.value.report is report
    assert str(info.value) == report.format()


def test_raise_if_failed_does_nothing_without_errors() -> None:
    """Test a report without errors does not raise."""
    report: ValidationReport[Any] = ValidationReport(
        (Diagnostic(DiagnosticLevel.WARNING, Note("m"), "s"),)
    )

    report.raise_if_failed()


def test_reports_compare_by_diagnostics_and_records() -> None:
    """Test reports are equal exactly when their diagnostics and records are."""
    report = _build_mixed_report()

    assert report == _build_mixed_report()
    assert hash(report) == hash(_build_mixed_report())
    assert report != ValidationReport(report.diagnostics, ("record", 2))
    assert report != ValidationReport(report.diagnostics[:2], report.records)


def test_report_with_an_unhashable_record_is_unhashable() -> None:
    """Test hashing a report hashes its records, as a dataclass's hash does."""
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        hash(ValidationReport((), ([],)))


def test_report_repr_matches_the_dataclass() -> None:
    """Test a report renders in the dataclass `repr` form."""
    report = ValidationReport((), ("record",))

    assert repr(report) == "ValidationReport(diagnostics=(), records=('record',))"
    assert str(report) == repr(report)


def test_report_built_through_a_subscripted_class_records_it() -> None:
    """Test `ValidationReport[int](...)` works and keeps `__orig_class__`."""
    report = ValidationReport[int]((), (1,))

    assert report.records == (1,)
    assert report.__orig_class__ == ValidationReport[int]  # type: ignore[attr-defined]


# =============================================================================
# Frozen instances
# =============================================================================


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(Note("m"), id="Note"),
        pytest.param(_build_error(), id="Diagnostic"),
        pytest.param(ValidationReport(), id="ValidationReport"),
    ],
)
def test_mutation_raises_the_frozen_mixin_error(instance: Any) -> None:
    """Test assigning or deleting an attribute raises `FrozenMixin`'s error."""
    with pytest.raises(FrozenMutationError) as set_info:
        instance.message = "rewritten"
    with pytest.raises(FrozenMutationError) as delete_info:
        del instance.records

    frozen = f"on frozen {type(instance).__name__}."
    assert str(set_info.value) == f'Cannot modify "message" {frozen}'
    assert str(delete_info.value) == f'Cannot delete "records" {frozen}'
    instance.freeze()
    instance.assert_frozen()
    assert instance.is_frozen


# =============================================================================
# Pickles
# =============================================================================


class _GlobalRecordingUnpickler(pickle.Unpickler):
    """Unpickler that records every global a pickle refers to."""

    found_globals: list[tuple[str, str]]

    def __init__(self, data: bytes) -> None:
        super().__init__(io.BytesIO(data))
        self.found_globals = []

    @override
    def find_class(self, module_name: str, global_name: str) -> Any:
        self.found_globals.append((module_name, global_name))
        return super().find_class(module_name, global_name)


def test_pickle_refers_only_to_the_public_classes() -> None:
    """Test a report pickles as calls of the public classes, never of `_rs`."""
    report = _build_mixed_report()
    unpickler = _GlobalRecordingUnpickler(pickle.dumps(report))

    restored = unpickler.load()

    assert restored == report
    assert type(restored) is ValidationReport
    assert type(restored.diagnostics[0].message) is Note
    assert restored.diagnostics[0].message.kind is OTHER_NOTE_KIND
    assert {module for module, _ in unpickler.found_globals} <= {
        "builtins",
        "fhy_core.diagnostic",
    }
    assert ("fhy_core.diagnostic", "Diagnostic") in unpickler.found_globals


def test_copies_are_equal_new_objects() -> None:
    """Test `copy` and `deepcopy` build equal objects through the constructor."""
    report = _build_mixed_report()

    assert copy.copy(report) == report
    assert copy.deepcopy(report) == report
    assert copy.deepcopy(report) is not report


def _run_python_in_a_fresh_process(source: str, *, stdin: str) -> str:
    """Run a program in a fresh interpreter and return its output."""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.mark.slow
@pytest.mark.subprocess
def test_pickles_load_across_processes() -> None:
    """Test a report pickled in one process loads in another one."""
    report = _build_mixed_report()
    this_process_pickle = base64.b64encode(pickle.dumps(report)).decode("ascii")

    fresh_process_pickle = _run_python_in_a_fresh_process(
        "import base64, pickle, sys\n"
        "from fhy_core.diagnostic import OTHER_NOTE_KIND\n"
        "report = pickle.loads(base64.b64decode(sys.stdin.read()))\n"
        "assert report.diagnostics[0].message.kind is OTHER_NOTE_KIND\n"
        "assert type(report.diagnostics).__name__ == 'tuple'\n"
        "print(base64.b64encode(pickle.dumps(report)).decode('ascii'))",
        stdin=this_process_pickle,
    )

    assert pickle.loads(base64.b64decode(fresh_process_pickle)) == report
