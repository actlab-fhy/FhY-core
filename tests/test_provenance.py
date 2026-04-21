"""Tests provenance tracking utilities."""

from pathlib import Path

import pytest

from fhy_core.provenance import Note, NoteKind, Position, Provenance, Span
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
)
from fhy_core.trait import Equal, Orderable, PartialEqual, PartialOrderable


def test_position_dict_serialization() -> None:
    """Test `Position` can be serialized/deserialized via a dictionary."""
    position = Position(2, 8)
    restored = Position.deserialize_from_dict(position.serialize_to_dict())

    assert restored == position


def test_position_supports_equal_and_orderable_traits() -> None:
    """Test `Position` satisfies equality and ordering trait protocols."""
    position = Position(2, 8)
    assert isinstance(position, PartialEqual)
    assert isinstance(position, Equal)
    assert isinstance(position, PartialOrderable)
    assert isinstance(position, Orderable)
    assert position.supports_partial_equality is True
    assert position.supports_equality is True
    assert position.supports_partial_ordering is True
    assert position.supports_ordering is True


def test_position_dict_deserialization_invalid_value_rejected() -> None:
    """Test invalid `Position` values are rejected during deserialization."""
    with pytest.raises(DeserializationValueError):
        Position.deserialize_from_dict({"line": 1, "column": 0})


def test_span_dict_serialization() -> None:
    """Test `Span` can be serialized/deserialized via a dictionary."""
    span = Span(Path("example.fhy"), 4, 12, Position(1, 5), Position(1, 13))
    restored = Span.deserialize_from_dict(span.serialize_to_dict())

    assert restored == span


def test_span_supports_equal_traits() -> None:
    """Test `Span` satisfies equality trait protocols."""
    span = Span(Path("example.fhy"), 4, 12, Position(1, 5), Position(1, 13))
    assert isinstance(span, PartialEqual)
    assert isinstance(span, Equal)
    assert span.supports_partial_equality is True
    assert span.supports_equality is True


def test_span_deserialization_structure_rejected() -> None:
    with pytest.raises(DeserializationDictStructureError):
        Span.deserialize_from_dict({"file_path": "x.fhy"})


def test_note_dict_serialization() -> None:
    """Test `Note` can be serialized/deserialized via a dictionary."""
    note = Note("lowered from ast", NoteKind.OTHER)
    restored = Note.deserialize_from_dict(note.serialize_to_dict())

    assert restored == note


def test_note_supports_equal_traits() -> None:
    """Test `Note` satisfies equality trait protocols."""
    note = Note("lowered from ast", NoteKind.OTHER)
    assert isinstance(note, PartialEqual)
    assert isinstance(note, Equal)
    assert note.supports_partial_equality is True
    assert note.supports_equality is True


def test_note_string_representation() -> None:
    """Test `Note` string formatting."""
    note = Note("lowered from ast", NoteKind.OTHER)

    assert str(note) == "other: lowered from ast"


def test_note_deserialization_invalid_kind_rejected() -> None:
    """Test invalid `Note` kind values are rejected during deserialization."""
    with pytest.raises(DeserializationValueError):
        Note.deserialize_from_dict({"message": "x", "kind": "bad"})


def test_provenance_unknown_span_is_none() -> None:
    """Test unknown provenance has no span."""
    assert Provenance.unknown().span is None


def test_provenance_with_span_sets_span() -> None:
    """Test `with_span` sets the current span."""
    span = Span(Path("a.fhy"), start_offset=0, end_offset=3)
    updated = Provenance.unknown().with_span(span)

    assert updated.span == span


def test_provenance_with_span_keeps_original_immutable() -> None:
    """Test `with_span` does not mutate the original provenance."""
    original = Provenance.unknown()
    original.with_span(Span(Path("a.fhy"), start_offset=0, end_offset=3))

    assert original.span is None


def test_provenance_add_origin_appends_origin() -> None:
    """Test `add_origin` appends one origin."""
    updated = Provenance.unknown().add_origin(
        Span(Path("b.fhy"), start_offset=2, end_offset=4)
    )

    assert len(updated.origins) == 1


def test_provenance_add_note_appends_note() -> None:
    """Test `add_note` appends one note."""
    note = Note("initial")
    updated = Provenance.unknown().add_note(note)

    assert updated.notes == (note,)


def test_provenance_add_source_id_appends_source_id() -> None:
    """Test `add_source_id` appends one source ID."""
    updated = Provenance.unknown().add_source_id("id::1")

    assert updated.source_ids == ("id::1",)


def test_provenance_merge_prefers_left_span() -> None:
    """Test `merge` keeps the left span when present."""
    left = (
        Provenance.unknown()
        .with_span(Span(Path("left.fhy"), start_offset=1, end_offset=2))
        .add_origin(Span(Path("left-origin.fhy"), start_offset=10, end_offset=11))
        .add_note(Note("left note"))
        .add_source_id("left-id")
    )
    right = (
        Provenance.unknown()
        .with_span(Span(Path("right.fhy"), start_offset=4, end_offset=7))
        .add_origin(Span(Path("right-origin.fhy"), start_offset=20, end_offset=21))
        .add_note(Note("right note"))
        .add_source_id("right-id")
    )

    merged = left.merge(right)

    assert merged.span == left.span


def test_provenance_merge_appends_origins() -> None:
    """Test `merge` concatenates origins in-order."""
    left = Provenance.unknown().add_origin(
        Span(Path("left-origin.fhy"), start_offset=10, end_offset=11)
    )
    right = Provenance.unknown().add_origin(
        Span(Path("right-origin.fhy"), start_offset=20, end_offset=21)
    )

    merged = left.merge(right)

    assert merged.origins == left.origins + right.origins


def test_provenance_merge_appends_notes() -> None:
    """Test `merge` concatenates notes in-order."""
    left = Provenance.unknown().add_note(Note("left note"))
    right = Provenance.unknown().add_note(Note("right note"))

    merged = left.merge(right)

    assert merged.notes == left.notes + right.notes


def test_provenance_merge_appends_source_ids() -> None:
    """Test `merge` concatenates source IDs in-order."""
    left = Provenance.unknown().add_source_id("left-id")
    right = Provenance.unknown().add_source_id("right-id")

    merged = left.merge(right)

    assert merged.source_ids == left.source_ids + right.source_ids


def test_provenance_dict_serialization() -> None:
    """Test `Provenance` can be serialized/deserialized via a dictionary."""
    provenance = (
        Provenance.unknown()
        .with_span(Span(Path("left.fhy"), start_offset=1, end_offset=2))
        .add_origin(Span(Path("left-origin.fhy"), start_offset=10, end_offset=11))
        .add_note(Note("left note"))
        .add_source_id("left-id")
    )
    restored = Provenance.deserialize_from_dict(provenance.serialize_to_dict())

    assert restored == provenance


def test_provenance_supports_equal_traits() -> None:
    """Test `Provenance` satisfies equality trait protocols."""
    provenance = Provenance.unknown()
    assert isinstance(provenance, PartialEqual)
    assert isinstance(provenance, Equal)
    assert provenance.supports_partial_equality is True
    assert provenance.supports_equality is True


def test_provenance_deserialization_structure_rejected() -> None:
    with pytest.raises(DeserializationDictStructureError):
        Provenance.deserialize_from_dict({"span": None, "origins": [], "notes": []})
