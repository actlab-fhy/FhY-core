"""Tests provenance tracking for compiler objects."""

from pathlib import Path

import pytest

from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Position,
    Provenance,
    Span,
    UnknownProvenance,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializationError,
)
from fhy_core.trait import (
    Equal,
    HasProvenance,
    Orderable,
    PartialEqual,
    PartialOrderable,
)

# ============================================================================
# Position
# ============================================================================


def test_position_constructs_with_valid_values() -> None:
    """Test `Position` is constructible with line and column >= 1."""
    position = Position(2, 8)

    assert position.line == 2
    assert position.column == 8


@pytest.mark.parametrize(
    "line,column",
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
    ids=["line-zero", "line-negative", "column-zero", "column-negative"],
)
def test_position_rejects_invalid_values(line: int, column: int) -> None:
    """Test `Position` raises `ValueError` for line or column < 1."""
    with pytest.raises(ValueError):
        Position(line, column)


def test_position_equality_is_by_value() -> None:
    """Test `Position` instances compare equal by line and column."""
    assert Position(1, 1) == Position(1, 1)
    assert Position(1, 1) != Position(1, 2)
    assert Position(1, 1) != Position(2, 1)


def test_position_orders_lexicographically_by_line_then_column() -> None:
    """Test `Position` ordering compares line first, then column."""
    assert Position(1, 1) < Position(1, 2)
    assert Position(1, 99) < Position(2, 1)


def test_position_dict_round_trip() -> None:
    """Test `Position` survives dict serialize/deserialize."""
    position = Position(2, 8)

    restored = Position.deserialize_from_dict(position.serialize_to_dict())

    assert restored == position


def test_position_dict_deserialization_invalid_value_rejected() -> None:
    """Test invalid `Position` values are rejected during deserialization."""
    with pytest.raises(DeserializationValueError):
        Position.deserialize_from_dict({"line": 1, "column": 0})


def test_position_dict_deserialization_structure_rejected() -> None:
    """Test malformed `Position` dicts are rejected during deserialization."""
    with pytest.raises(DeserializationDictStructureError):
        Position.deserialize_from_dict({"line": 1})


def test_position_satisfies_equal_and_orderable_traits() -> None:
    """Test `Position` implements equality and ordering trait protocols."""
    position = Position(2, 8)

    assert isinstance(position, PartialEqual)
    assert isinstance(position, Equal)
    assert isinstance(position, PartialOrderable)
    assert isinstance(position, Orderable)
    assert position.supports_partial_equality is True
    assert position.supports_equality is True
    assert position.supports_partial_ordering is True
    assert position.supports_ordering is True


def test_position_string_representation() -> None:
    """Test `Position` renders as ``line:column``."""
    assert str(Position(2, 8)) == "2:8"


@pytest.mark.parametrize(
    "line,column",
    [(True, 1), (1, True), (False, 1), (1, False)],
    ids=["bool-line-true", "bool-column-true", "bool-line-false", "bool-column-false"],
)
def test_position_rejects_bool_values(line: object, column: object) -> None:
    """Test `Position` rejects `bool` values for line and column."""
    with pytest.raises(TypeError):
        Position(line, column)  # type: ignore[arg-type]


def test_position_dict_deserialization_rejects_bool_values() -> None:
    """Test `Position.deserialize_from_dict` rejects `bool` values."""
    with pytest.raises(DeserializationDictStructureError):
        Position.deserialize_from_dict({"line": True, "column": 1})


# ============================================================================
# Span
# ============================================================================


def test_span_default_construction_is_unknown() -> None:
    """Test default-constructed `Span` reports unknown."""
    assert Span().is_unknown()


def test_span_with_offsets_is_not_unknown() -> None:
    """Test `Span` with offsets is not unknown."""
    assert not Span(start_offset=0, end_offset=3).is_unknown()


def test_span_with_positions_is_not_unknown() -> None:
    """Test `Span` with positions is not unknown."""
    span = Span(start_position=Position(1, 1), end_position=Position(1, 2))

    assert not span.is_unknown()


def test_span_does_not_carry_a_file_path() -> None:
    """Test `Span` exposes only offsets and positions."""
    span = Span(start_offset=0, end_offset=3)

    assert not hasattr(span, "file_path")


@pytest.mark.parametrize(
    "start_offset,end_offset",
    [(-1, None), (None, -1), (5, 3), (-3, -1)],
    ids=["negative-start", "negative-end", "end-before-start", "both-negative"],
)
def test_span_rejects_invalid_offsets(
    start_offset: int | None, end_offset: int | None
) -> None:
    """Test `Span` raises `ValueError` on invalid offset combinations."""
    with pytest.raises(ValueError):
        Span(start_offset=start_offset, end_offset=end_offset)


def test_span_rejects_end_position_before_start_position() -> None:
    """Test `Span` raises `ValueError` when end_position < start_position."""
    with pytest.raises(ValueError):
        Span(start_position=Position(2, 1), end_position=Position(1, 1))


def test_span_allows_equal_start_and_end_offset() -> None:
    """Test `Span` permits zero-width offset ranges."""
    span = Span(start_offset=4, end_offset=4)

    assert span.start_offset == 4
    assert span.end_offset == 4


def test_span_allows_equal_start_and_end_position() -> None:
    """Test `Span` permits zero-width position ranges."""
    span = Span(start_position=Position(1, 1), end_position=Position(1, 1))

    assert span.start_position == Position(1, 1)
    assert span.end_position == Position(1, 1)


def test_span_dict_round_trip_with_all_fields() -> None:
    """Test `Span` survives dict round-trip with offsets and positions."""
    span = Span(0, 3, Position(1, 1), Position(1, 4))

    restored = Span.deserialize_from_dict(span.serialize_to_dict())

    assert restored == span


def test_span_dict_round_trip_unknown() -> None:
    """Test unknown `Span` survives dict round-trip."""
    span = Span()

    restored = Span.deserialize_from_dict(span.serialize_to_dict())

    assert restored == span


def test_span_dict_deserialization_structure_rejected() -> None:
    """Test malformed `Span` dicts are rejected during deserialization."""
    with pytest.raises(DeserializationDictStructureError):
        Span.deserialize_from_dict({"start_offset": 0})


@pytest.mark.parametrize(
    "start_offset,end_offset",
    [(True, None), (None, False), (True, True)],
    ids=["bool-start", "bool-end", "both-bool"],
)
def test_span_rejects_bool_offsets(start_offset: object, end_offset: object) -> None:
    """Test `Span` rejects `bool` values for offsets."""
    with pytest.raises(TypeError):
        Span(start_offset=start_offset, end_offset=end_offset)  # type: ignore[arg-type]


def test_span_dict_deserialization_rejects_bool_offsets() -> None:
    """Test `Span.deserialize_from_dict` rejects `bool` offset values."""
    with pytest.raises(DeserializationDictStructureError):
        Span.deserialize_from_dict(
            {
                "start_offset": True,
                "end_offset": None,
                "start_position": None,
                "end_position": None,
            }
        )


def test_span_string_representation_for_unknown() -> None:
    """Test `Span.__str__` returns ``<unknown>`` for an unknown span."""
    assert str(Span()) == "<unknown>"


def test_span_string_representation_with_both_positions() -> None:
    """Test `Span.__str__` renders ``start-end`` when both positions are set."""
    span = Span(start_position=Position(1, 1), end_position=Position(1, 4))

    assert str(span) == "1:1-1:4"


def test_span_string_representation_with_both_offsets() -> None:
    """Test `Span.__str__` renders ``@start-end`` when only offsets are set."""
    assert str(Span(start_offset=0, end_offset=3)) == "@0-3"


def test_span_string_representation_partial_offsets_uses_placeholder() -> None:
    """Test `Span.__str__` renders a partial offset span using a placeholder."""
    assert str(Span(start_offset=5)) == "@5-?"


def test_span_string_representation_partial_positions_uses_placeholder() -> None:
    """Test `Span.__str__` renders a partial position span using a placeholder."""
    span = Span(start_position=Position(1, 1))

    assert str(span) == "1:1-?"


def test_span_satisfies_equal_traits() -> None:
    """Test `Span` implements equality trait protocols."""
    span = Span(0, 3)

    assert isinstance(span, PartialEqual)
    assert isinstance(span, Equal)
    assert span.supports_partial_equality is True
    assert span.supports_equality is True


# ============================================================================
# Provenance.unknown
# ============================================================================


def test_provenance_unknown_returns_unknown_variant() -> None:
    """Test `Provenance.unknown()` returns an `UnknownProvenance` instance."""
    assert isinstance(Provenance.unknown(), UnknownProvenance)


def test_provenance_unknown_instances_compare_equal() -> None:
    """Test all `UnknownProvenance` instances compare equal."""
    assert Provenance.unknown() == Provenance.unknown()
    assert UnknownProvenance() == UnknownProvenance()


def test_unknown_provenance_deserialization_rejects_non_empty_data() -> None:
    """Test `UnknownProvenance` rejects payloads with unexpected keys."""
    with pytest.raises(DeserializationDictStructureError):
        UnknownProvenance.deserialize_data_from_dict({"unexpected": 1})


# ============================================================================
# Variant construction
# ============================================================================


def test_file_provenance_constructs_without_span() -> None:
    """Test `FileProvenance` defaults `span` to `None`."""
    prov = FileProvenance(Path("a.fhy"))

    assert prov.file_path == Path("a.fhy")
    assert prov.span is None


def test_file_provenance_constructs_with_span() -> None:
    """Test `FileProvenance` carries the supplied span."""
    span = Span(0, 3)

    prov = FileProvenance(Path("a.fhy"), span)

    assert prov.span == span


def test_named_provenance_with_unknown_child_encodes_builtin_pattern() -> None:
    """Test `NamedProvenance` over `UnknownProvenance` encodes a builtin."""
    prov = NamedProvenance("fhy.add", UnknownProvenance())

    assert prov.name == "fhy.add"
    assert prov.child == UnknownProvenance()


def test_named_provenance_with_file_child_encodes_library_symbol_pattern() -> None:
    """Test `NamedProvenance` over `FileProvenance` encodes a library symbol."""
    child = FileProvenance(Path("mylib.fhyobj"))

    prov = NamedProvenance("mylib::matmul", child)

    assert prov.name == "mylib::matmul"
    assert prov.child == child


def test_named_provenance_rejects_empty_name() -> None:
    """Test `NamedProvenance` raises `ValueError` when name is empty."""
    with pytest.raises(ValueError):
        NamedProvenance("", UnknownProvenance())


def test_call_site_provenance_constructs_with_callee_and_caller() -> None:
    """Test `CallSiteProvenance` carries both arms."""
    callee = FileProvenance(Path("callee.fhy"))
    caller = FileProvenance(Path("caller.fhy"))

    prov = CallSiteProvenance(callee=callee, caller=caller)

    assert prov.callee == callee
    assert prov.caller == caller


def test_fused_provenance_constructs_with_sources() -> None:
    """Test `FusedProvenance` preserves sources in the order given."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))

    prov = FusedProvenance(sources=(a, b))

    assert prov.sources == (a, b)
    assert prov.metadata is None


def test_fused_provenance_constructs_with_metadata() -> None:
    """Test `FusedProvenance` preserves the metadata label."""
    a = FileProvenance(Path("a.fhy"))

    prov = FusedProvenance(sources=(a,), metadata="loop-fusion")

    assert prov.metadata == "loop-fusion"


# ============================================================================
# Provenance.fuse: smart-constructor reduction rules
# ============================================================================


def test_fuse_with_no_inputs_returns_unknown() -> None:
    """Test `Provenance.fuse()` returns `UnknownProvenance`."""
    assert Provenance.fuse() == UnknownProvenance()


def test_fuse_with_only_unknown_inputs_returns_unknown() -> None:
    """Test `Provenance.fuse(unknown, unknown)` returns `UnknownProvenance`."""
    fused = Provenance.fuse(UnknownProvenance(), UnknownProvenance())

    assert fused == UnknownProvenance()


def test_fuse_single_input_without_metadata_returns_input_unchanged() -> None:
    """Test `Provenance.fuse(p)` returns `p` when no metadata is given."""
    p = FileProvenance(Path("a.fhy"))

    assert Provenance.fuse(p) == p


def test_fuse_single_input_with_metadata_wraps_in_fused() -> None:
    """Test `Provenance.fuse(p, metadata=m)` wraps even one source."""
    p = FileProvenance(Path("a.fhy"))

    fused = Provenance.fuse(p, metadata="cse")

    assert fused == FusedProvenance(sources=(p,), metadata="cse")


def test_fuse_drops_unknown_inputs_from_mixed_input() -> None:
    """Test `Provenance.fuse` strips `UnknownProvenance` from inputs."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))

    fused = Provenance.fuse(a, UnknownProvenance(), b)

    assert fused == FusedProvenance(sources=(a, b))


def test_fuse_flattens_nested_fused_when_inner_metadata_is_none() -> None:
    """Test `Provenance.fuse` splices metadata-less nested fusions."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))
    inner = FusedProvenance(sources=(a, b))

    fused = Provenance.fuse(inner, c)

    assert fused == FusedProvenance(sources=(a, b, c))


def test_fuse_preserves_nested_fused_when_inner_metadata_is_set() -> None:
    """Test `Provenance.fuse` keeps labeled nested fusions intact."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))
    inner = FusedProvenance(sources=(a, b), metadata="cse")

    fused = Provenance.fuse(inner, c)

    assert fused == FusedProvenance(sources=(inner, c))


def test_fuse_preserves_input_order() -> None:
    """Test `Provenance.fuse` preserves the order of input sources."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))

    fused = Provenance.fuse(a, b, c)

    assert fused == FusedProvenance(sources=(a, b, c))


def test_fuse_preserves_duplicate_sources() -> None:
    """Test `Provenance.fuse` does not deduplicate equal sources."""
    a = FileProvenance(Path("a.fhy"))

    fused = Provenance.fuse(a, a)

    assert fused == FusedProvenance(sources=(a, a))


def test_fuse_attaches_metadata_to_result() -> None:
    """Test `Provenance.fuse` carries the supplied metadata into the result."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))

    fused = Provenance.fuse(a, b, metadata="loop-fusion")

    assert fused == FusedProvenance(sources=(a, b), metadata="loop-fusion")


def test_fuse_drops_unknown_inside_directly_constructed_nested_fused() -> None:
    """Test `Provenance.fuse` strips `UnknownProvenance` after splicing."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    non_canonical_inner = FusedProvenance(sources=(UnknownProvenance(), a))

    fused = Provenance.fuse(non_canonical_inner, b)

    assert fused == FusedProvenance(sources=(a, b))


def test_fuse_flattens_nested_metadata_less_fused_transitively() -> None:
    """Test `Provenance.fuse` flattens metadata-less fused at any depth."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))
    deeply_nested = FusedProvenance(sources=(a, FusedProvenance(sources=(b, c))))

    fused = Provenance.fuse(deeply_nested)

    assert fused == FusedProvenance(sources=(a, b, c))


def test_fuse_does_not_flatten_through_metadata_bearing_fused() -> None:
    """Test `Provenance.fuse` keeps labeled fused opaque even when nested."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))
    labeled_inner = FusedProvenance(sources=(b, c), metadata="cse")
    outer_metadata_less = FusedProvenance(sources=(a, labeled_inner))

    fused = Provenance.fuse(outer_metadata_less)

    assert fused == FusedProvenance(sources=(a, labeled_inner))


def test_fuse_combines_all_reduction_rules() -> None:
    """Test `Provenance.fuse` composes drop-unknown, flatten, and order rules."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    c = FileProvenance(Path("c.fhy"))
    inner_no_meta = FusedProvenance(sources=(a, b))
    inner_with_meta = FusedProvenance(sources=(b, c), metadata="x")

    fused = Provenance.fuse(
        UnknownProvenance(),
        inner_no_meta,
        UnknownProvenance(),
        inner_with_meta,
        c,
    )

    assert fused == FusedProvenance(sources=(a, b, inner_with_meta, c))


# ============================================================================
# Polymorphic dict round-trip
# ============================================================================


def _build_one_of_each_variant() -> list[Provenance]:
    """Return one instance of each provenance variant for parametrization."""
    src = FileProvenance(Path("a.fhy"), Span(0, 10))
    other = FileProvenance(Path("b.fhy"))
    return [
        UnknownProvenance(),
        FileProvenance(Path("c.fhy")),
        FileProvenance(
            Path("a.fhy"),
            Span(0, 3, Position(1, 1), Position(1, 4)),
        ),
        NamedProvenance("fhy.add", UnknownProvenance()),
        NamedProvenance("mylib::matmul", FileProvenance(Path("mylib.fhyobj"))),
        CallSiteProvenance(callee=src, caller=other),
        CallSiteProvenance(
            callee=NamedProvenance("inlined", src),
            caller=other,
        ),
        FusedProvenance(sources=(src, other)),
        FusedProvenance(sources=(src,), metadata="loop-fusion"),
    ]


@pytest.mark.parametrize(
    "provenance",
    _build_one_of_each_variant(),
    ids=lambda p: type(p).__name__,
)
def test_provenance_dict_round_trip_via_base(provenance: Provenance) -> None:
    """Test each variant round-trips via `Provenance.deserialize_from_dict`."""
    restored = Provenance.deserialize_from_dict(provenance.serialize_to_dict())

    assert restored == provenance


@pytest.mark.parametrize(
    "provenance",
    _build_one_of_each_variant(),
    ids=lambda p: type(p).__name__,
)
def test_provenance_binary_round_trip(provenance: Provenance) -> None:
    """Test each variant round-trips through the binary envelope."""
    restored = Provenance.from_bytes(provenance.to_bytes())

    assert restored == provenance


@pytest.mark.parametrize(
    "provenance",
    _build_one_of_each_variant(),
    ids=lambda p: type(p).__name__,
)
def test_provenance_satisfies_equal_traits(provenance: Provenance) -> None:
    """Test each variant implements the equality trait protocols."""
    assert isinstance(provenance, PartialEqual)
    assert isinstance(provenance, Equal)
    assert provenance.supports_partial_equality is True
    assert provenance.supports_equality is True


@pytest.mark.parametrize(
    "provenance",
    _build_one_of_each_variant(),
    ids=lambda p: type(p).__name__,
)
def test_provenance_is_hashable(provenance: Provenance) -> None:
    """Test each variant is hashable so it can be used in sets and dict keys."""
    assert hash(provenance) == hash(provenance)


def test_provenance_deserialization_with_unknown_type_id_raises() -> None:
    """Test deserializing an unknown `__type__` raises `SerializationError`."""
    with pytest.raises(SerializationError):
        Provenance.deserialize_from_dict(
            {"__type__": "provenance.does_not_exist", "__data__": {}}
        )


# ============================================================================
# HasProvenance trait integration
# ============================================================================


class _NodeWithProvenance:
    """Minimal `HasProvenance` implementation for trait-conformance tests."""

    def __init__(self, provenance: Provenance) -> None:
        self._provenance = provenance

    def get_provenance(self) -> Provenance:
        return self._provenance


@pytest.mark.parametrize(
    "provenance",
    _build_one_of_each_variant(),
    ids=lambda p: type(p).__name__,
)
def test_class_returning_each_variant_satisfies_has_provenance(
    provenance: Provenance,
) -> None:
    """Test a class returning any variant satisfies `HasProvenance`."""
    node = _NodeWithProvenance(provenance)

    assert isinstance(node, HasProvenance)
    assert node.get_provenance() == provenance


# ============================================================================
# User-story flows
# ============================================================================


def test_user_story_source_file_provenance_round_trips_via_binary() -> None:
    """Test source-file provenance survives a binary round-trip."""
    provenance = FileProvenance(
        Path("hello.fhy"),
        Span(
            start_offset=0,
            end_offset=12,
            start_position=Position(1, 1),
            end_position=Position(1, 13),
        ),
    )

    restored = Provenance.from_bytes(provenance.to_bytes())

    assert restored == provenance


def test_user_story_library_symbol_provenance_round_trips() -> None:
    """Test the library-symbol composition survives a dict round-trip."""
    provenance = NamedProvenance(
        "mylib::matmul",
        FileProvenance(Path("mylib.fhyobj")),
    )

    restored = Provenance.deserialize_from_dict(provenance.serialize_to_dict())

    assert restored == provenance


def test_user_story_builtin_provenance_round_trips() -> None:
    """Test the builtin composition (`Named` over `Unknown`) round-trips."""
    provenance = NamedProvenance("fhy.add", UnknownProvenance())

    restored = Provenance.deserialize_from_dict(provenance.serialize_to_dict())

    assert restored == provenance


def test_user_story_loop_fusion_pass_combines_two_provenances() -> None:
    """Test a fusion pass produces a `FusedProvenance` carrying both sources."""
    op_a = FileProvenance(Path("a.fhy"), Span(start_offset=0, end_offset=3))
    op_b = FileProvenance(Path("b.fhy"), Span(start_offset=10, end_offset=13))

    fused = Provenance.fuse(op_a, op_b, metadata="loop-fusion")

    assert isinstance(fused, FusedProvenance)
    assert fused.sources == (op_a, op_b)
    assert fused.metadata == "loop-fusion"


# ============================================================================
# Provenance.__str__ (human-readable rendering)
# ============================================================================


def test_provenance_str_is_abstract_on_base_class() -> None:
    """Test `Provenance.__str__` is declared abstract on the base class."""
    assert "__str__" in Provenance.__abstractmethods__


def test_unknown_provenance_string_representation() -> None:
    """Test `UnknownProvenance.__str__` renders as ``<unknown>``."""
    assert str(UnknownProvenance()) == "<unknown>"


def test_file_provenance_string_representation_without_span() -> None:
    """Test `FileProvenance.__str__` renders just the file path when no span."""
    assert str(FileProvenance(Path("a.fhy"))) == "a.fhy"


def test_file_provenance_string_representation_with_unknown_span() -> None:
    """Test `FileProvenance.__str__` omits an unknown span."""
    assert str(FileProvenance(Path("a.fhy"), Span())) == "a.fhy"


def test_file_provenance_string_representation_with_offset_span() -> None:
    """Test `FileProvenance.__str__` appends an offset-only span."""
    assert str(FileProvenance(Path("a.fhy"), Span(0, 3))) == "a.fhy:@0-3"


def test_file_provenance_string_representation_with_position_span() -> None:
    """Test `FileProvenance.__str__` appends a position-bearing span."""
    span = Span(start_position=Position(1, 1), end_position=Position(1, 4))

    assert str(FileProvenance(Path("a.fhy"), span)) == "a.fhy:1:1-1:4"


def test_named_provenance_string_representation_with_unknown_child() -> None:
    """Test `NamedProvenance.__str__` shows just the name for unknown child."""
    assert str(NamedProvenance("fhy.add", UnknownProvenance())) == "fhy.add"


def test_named_provenance_string_representation_with_known_child() -> None:
    """Test `NamedProvenance.__str__` shows the child in parentheses."""
    prov = NamedProvenance("mylib::matmul", FileProvenance(Path("mylib.fhyobj")))

    assert str(prov) == "mylib::matmul (mylib.fhyobj)"


def test_call_site_provenance_string_representation() -> None:
    """Test `CallSiteProvenance.__str__` renders ``callee at caller``."""
    prov = CallSiteProvenance(
        callee=FileProvenance(Path("callee.fhy")),
        caller=FileProvenance(Path("caller.fhy")),
    )

    assert str(prov) == "callee.fhy at caller.fhy"


def test_fused_provenance_string_representation_without_metadata() -> None:
    """Test `FusedProvenance.__str__` uses ``fused`` label when unlabeled."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))

    assert str(FusedProvenance(sources=(a, b))) == "fused[a.fhy, b.fhy]"


def test_fused_provenance_string_representation_with_metadata() -> None:
    """Test `FusedProvenance.__str__` uses metadata as the label."""
    a = FileProvenance(Path("a.fhy"))

    prov = FusedProvenance(sources=(a,), metadata="loop-fusion")

    assert str(prov) == "loop-fusion[a.fhy]"


def test_fused_provenance_string_representation_recurses_into_sources() -> None:
    """Test `FusedProvenance.__str__` renders nested provenances recursively."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    inner = FusedProvenance(sources=(a, b), metadata="cse")
    outer = FusedProvenance(sources=(inner, FileProvenance(Path("c.fhy"))))

    assert str(outer) == "fused[cse[a.fhy, b.fhy], c.fhy]"


def test_user_story_inliner_call_site_chain_is_walkable() -> None:
    """Test a chained `CallSiteProvenance` exposes both arms recursively."""
    inner = FileProvenance(Path("inner.fhy"))
    middle = FileProvenance(Path("middle.fhy"))
    outer = FileProvenance(Path("outer.fhy"))

    chain = CallSiteProvenance(
        callee=CallSiteProvenance(callee=inner, caller=middle),
        caller=outer,
    )

    assert chain.caller == outer
    assert isinstance(chain.callee, CallSiteProvenance)
    assert chain.callee.callee == inner
    assert chain.callee.caller == middle
