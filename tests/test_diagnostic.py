"""Tests diagnostic message types."""

import pytest

from fhy_core.diagnostic import (
    OTHER_NOTE_KIND,
    RATIONALE_NOTE_KIND,
    REMARK_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    Note,
    NoteKind,
)
from fhy_core.identifier import HasIdentifier, Identifier
from fhy_core.traits import Equal, Interned, PartialEqual

_DEFAULT_NOTE_KINDS = (
    RATIONALE_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    REMARK_NOTE_KIND,
    OTHER_NOTE_KIND,
)


# =============================================================================
# Note
# =============================================================================


def test_note_constructs_with_default_kind() -> None:
    """Test `Note` defaults `kind` to the OTHER note kind."""
    note = Note("lowered from ast")

    assert note.message == "lowered from ast"
    assert note.kind is OTHER_NOTE_KIND


def test_note_constructs_with_explicit_kind() -> None:
    """Test `Note` carries the supplied kind."""
    note = Note("tiled the loop for cache locality", RATIONALE_NOTE_KIND)

    assert note.kind is RATIONALE_NOTE_KIND


def test_note_string_representation() -> None:
    """Test `Note` renders as ``kind: message``."""
    note = Note("lowered from ast", OTHER_NOTE_KIND)

    assert str(note) == "other: lowered from ast"


def test_note_satisfies_equal_traits() -> None:
    """Test `Note` implements equality trait protocols."""
    note = Note("hello")

    assert isinstance(note, PartialEqual)
    assert isinstance(note, Equal)
    assert note.supports_partial_equality is True
    assert note.supports_equality is True


# =============================================================================
# Canonical note kinds
# =============================================================================


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        (RATIONALE_NOTE_KIND, "rationale"),
        (SUGGESTION_NOTE_KIND, "suggestion"),
        (REMARK_NOTE_KIND, "remark"),
        (OTHER_NOTE_KIND, "other"),
    ],
)
def test_canonical_note_kind_has_stable_name(kind: NoteKind, name: str) -> None:
    """Test each canonical note kind keeps its documented identifier name."""
    assert kind.name.name_hint == name


def test_note_kind_satisfies_documented_protocols() -> None:
    """Test a note kind satisfies the `HasIdentifier` and `Interned` protocols."""
    assert isinstance(OTHER_NOTE_KIND, HasIdentifier)
    assert isinstance(OTHER_NOTE_KIND, Interned)


def test_canonical_note_kind_is_registered() -> None:
    """Test a canonical note kind is the interned entry for its name."""
    assert NoteKind.get_interned(OTHER_NOTE_KIND.name) is OTHER_NOTE_KIND


# =============================================================================
# Open registry
# =============================================================================


def test_note_kind_registry_is_open_to_new_kinds() -> None:
    """Test a caller can register a new note kind without modifying core."""
    custom = NoteKind(Identifier("performance"), "An optimization remark.")

    assert NoteKind.get_interned(custom.name) is custom


def test_note_kind_equality_ignores_description() -> None:
    """Test note-kind equality is keyed on the identifier, not the description."""
    name = Identifier("shared")
    first = NoteKind(name, "first description")
    second = NoteKind(name, "second description")

    assert first == second


# =============================================================================
# Serialization
# =============================================================================


@pytest.mark.parametrize("kind", _DEFAULT_NOTE_KINDS, ids=lambda k: k.name.name_hint)
def test_note_round_trips_each_default_kind(kind: NoteKind) -> None:
    """Test a `Note` round-trips through the dict form for every default kind."""
    note = Note("a message", kind)

    assert Note.deserialize_from_dict(note.serialize_to_dict()) == note


def test_note_with_custom_kind_round_trips() -> None:
    """Test a `Note` tagged with a custom kind round-trips through the dict form."""
    custom = NoteKind(Identifier("deprecation"), "A deprecated-usage remark.")
    note = Note("uses a deprecated builtin", custom)

    assert Note.deserialize_from_dict(note.serialize_to_dict()) == note
