"""Tests diagnostic message types."""

from fhy_core.diagnostic import Note, NoteKind
from fhy_core.traits import Equal, PartialEqual


def test_note_constructs_with_default_kind() -> None:
    """Test `Note` defaults `kind` to `NoteKind.OTHER`."""
    note = Note("lowered from ast")

    assert note.message == "lowered from ast"
    assert note.kind == NoteKind.OTHER


def test_note_constructs_with_explicit_kind() -> None:
    """Test `Note` carries the supplied kind."""
    note = Note("lowered from ast", NoteKind.OTHER)

    assert note.kind == NoteKind.OTHER


def test_note_string_representation() -> None:
    """Test `Note` renders as ``kind: message``."""
    note = Note("lowered from ast", NoteKind.OTHER)

    assert str(note) == "other: lowered from ast"


def test_note_satisfies_equal_traits() -> None:
    """Test `Note` implements equality trait protocols."""
    note = Note("hello")

    assert isinstance(note, PartialEqual)
    assert isinstance(note, Equal)
    assert note.supports_partial_equality is True
    assert note.supports_equality is True
