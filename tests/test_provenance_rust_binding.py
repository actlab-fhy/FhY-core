"""Tests the Python interface of the Rust-backed provenance classes.

``Position``, ``Span``, ``Provenance`` and its five variant classes are thin
Python subclasses of the ``fhy_core._rs`` classes over the Rust values (slice
S3b of ``docs/design/python-switch.md``). Their behavioral suites cover the
provenance semantics; this suite covers what the binding adds: the class
hierarchy and the registration of the public classes, construction and its
argument checks, path normalization, the dataclass reprs, equality and
ordering, the pinned payload text, pickles across processes, frozen errors,
and the recursion guard.
"""

import base64
import copy
import io
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

import fhy_core
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
    Serializable,
    SerializationFormat,
    SerializedDict,
    WrappedFamilySerializable,
)
from fhy_core.traits import EqualMixin, FrozenMixin, FrozenMutationError
from fhy_core.utils.override import override

_VARIANT_CLASSES = (
    UnknownProvenance,
    FileProvenance,
    NamedProvenance,
    CallSiteProvenance,
    FusedProvenance,
)
_PUBLIC_CLASSES = pytest.mark.parametrize(
    "cls",
    [Position, Span, Provenance, *_VARIANT_CLASSES],
    ids=lambda cls: cls.__name__,
)

# Builds one provenance of each shape, the same source text in every process.
_BUILD_PROVENANCES_SOURCE = """
from pathlib import Path
from fhy_core.provenance import *

source = FileProvenance(Path("a.fhy"), Span(0, 3, Position(1, 1), Position(1, 4)))
other = FileProvenance(Path("b/c.fhy"), Span(start_offset=5))
provenances = [
    UnknownProvenance(),
    source,
    other,
    FileProvenance(Path("d.fhy"), Span()),
    NamedProvenance("fhy.add", UnknownProvenance()),
    NamedProvenance("lib", source),
    CallSiteProvenance(NamedProvenance("inlined", source), other),
    FusedProvenance((source, other)),
    FusedProvenance((FusedProvenance((source,), "cse"), UnknownProvenance()), ""),
]
"""


def _build_provenances() -> list[Provenance]:
    """Return one provenance of each shape, from the shared source text."""
    namespace: dict[str, Any] = {}
    exec(_BUILD_PROVENANCES_SOURCE, namespace)
    provenances: list[Provenance] = namespace["provenances"]
    return provenances


def _call(cls: Any, *arguments: Any, **keywords: Any) -> Any:
    """Return ``cls(*arguments, **keywords)``, untyped, to pass wrong types."""
    return cls(*arguments, **keywords)


def _run_python_in_a_fresh_process(source: str, *, stdin: str = "") -> str:
    """Run a program in a fresh interpreter and return its output."""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


# =============================================================================
# Class structure and registration
# =============================================================================


@_PUBLIC_CLASSES
def test_public_class_is_a_thin_subclass_of_the_rust_class(cls: type[Any]) -> None:
    """Test the public class subclasses its `_rs` class and mixes in protocols."""
    rust_class = getattr(fhy_core._rs, cls.__name__)

    assert cls.__bases__[0] is rust_class
    assert issubclass(cls, Serializable)
    assert issubclass(cls, EqualMixin)
    assert issubclass(cls, FrozenMixin)
    assert FrozenMixin not in cls.__mro__


@pytest.mark.parametrize("cls", _VARIANT_CLASSES, ids=lambda cls: cls.__name__)
def test_variant_classes_form_a_hierarchy_over_provenance(cls: type[Any]) -> None:
    """Test each variant subclasses `Provenance` in Python and in Rust."""
    assert issubclass(cls, Provenance)
    assert issubclass(cls, fhy_core._rs.Provenance)
    assert issubclass(getattr(fhy_core._rs, cls.__name__), fhy_core._rs.Provenance)
    assert issubclass(cls, WrappedFamilySerializable)
    assert not cls.__abstractmethods__


def test_provenance_base_stays_abstract_and_unconstructible() -> None:
    """Test `Provenance` keeps its abstract `__str__` and cannot be built."""
    assert Provenance.__abstractmethods__ == frozenset({"__str__"})

    with pytest.raises(TypeError):
        Provenance()  # type: ignore[abstract]  # test: the base is abstract


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


def test_provenances_built_in_rust_are_instances_of_the_public_classes() -> None:
    """Test `unknown()` and `fuse` return the registered public classes."""
    source = FileProvenance(Path("a.fhy"))

    assert type(Provenance.unknown()) is UnknownProvenance
    assert type(Provenance.fuse()) is UnknownProvenance
    assert type(Provenance.fuse(source, metadata="cse")) is FusedProvenance
    assert type(FileProvenance.fuse(source, source)) is FusedProvenance


# =============================================================================
# Construction and fields
# =============================================================================


def test_fields_return_the_objects_the_value_was_built_from() -> None:
    """Test every field returns the very object it was given."""
    start, end = Position(1, 1), Position(1, 4)
    span = Span(0, 3, start, end)
    source = FileProvenance(Path("a.fhy"), span)
    name = "".join(["lib", "::sym"])
    named = NamedProvenance(name, source)
    call_site = CallSiteProvenance(named, source)
    sources = (source, named)
    fused = FusedProvenance(sources, "".join(["loop", "-fusion"]))

    assert span.start_position is start
    assert span.end_position is end
    assert source.span is span
    assert named.name is name
    assert named.child is source
    assert call_site.callee is named
    assert call_site.caller is source
    assert fused.sources is sources


def test_fused_provenance_stores_any_iterable_of_sources_as_a_tuple() -> None:
    """Test `FusedProvenance` accepts any iterable and stores a tuple."""
    source = FileProvenance(Path("a.fhy"))

    fused = _call(FusedProvenance, iter([source, source]))

    assert fused.sources == (source, source)
    assert fused.sources[0] is source
    assert fused.metadata is None


def test_fuse_returns_the_single_survivor_and_keeps_source_objects() -> None:
    """Test `fuse` returns its inputs' objects, as the Python loop does."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))

    fused = Provenance.fuse(UnknownProvenance(), FusedProvenance((a, b)))

    assert Provenance.fuse(a, UnknownProvenance()) is a
    assert isinstance(fused, FusedProvenance)
    assert fused.sources[0] is a
    assert fused.sources[1] is b


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        pytest.param((0, 1), ValueError, '"line" must be >= 1, got 0', id="zero"),
        pytest.param(
            (0, -1), ValueError, '"line" must be >= 1, got 0', id="line_first"
        ),
        pytest.param(
            (1, -(2**70)),
            ValueError,
            f'"column" must be >= 1, got {-(2**70)}',
            id="huge",
        ),
        pytest.param(
            (True, 1), TypeError, '"line" must be a strict int, got bool', id="bool"
        ),
        pytest.param(
            (2**64, 1),
            OverflowError,
            f'"line" must be at most {2**64 - 1}, got {2**64}',
            id="overflow",
        ),
    ],
)
def test_position_rejects_invalid_coordinates(
    arguments: tuple[Any, ...], error: type[Exception], message: str
) -> None:
    """Test `Position` raises the Python messages, and overflows past `u64`."""
    with pytest.raises(error) as info:
        Position(*arguments)

    assert str(info.value) == message


def test_position_accepts_the_largest_unsigned_coordinate() -> None:
    """Test a coordinate of `2**64 - 1` fits, and is returned as given."""
    position = Position(2**64 - 1, 1)

    assert position.line == 2**64 - 1
    assert str(position) == f"{2**64 - 1}:1"


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        pytest.param(
            {"start_offset": 5, "end_offset": 3},
            ValueError,
            '"end_offset" must be >= "start_offset", got 3 < 5',
            id="offsets_out_of_order",
        ),
        pytest.param(
            {"start_position": Position(2, 1), "end_position": Position(1, 9)},
            ValueError,
            '"end_position" must be >= "start_position", got 1:9 < 2:1',
            id="positions_out_of_order",
        ),
        pytest.param(
            {"start_offset": -3, "end_offset": -1},
            ValueError,
            '"start_offset" must be >= 0, got -3',
            id="negative",
        ),
        pytest.param(
            {"end_offset": 1.5},
            TypeError,
            '"end_offset" must be a strict int or None, got float',
            id="float",
        ),
        pytest.param(
            {"start_position": (1, 1)},
            TypeError,
            "Span start_position must be a Position or None, got tuple.",
            id="position_type",
        ),
        pytest.param(
            {"end_offset": 2**64},
            OverflowError,
            f'"end_offset" must be at most {2**64 - 1}, got {2**64}',
            id="overflow",
        ),
    ],
)
def test_span_rejects_invalid_bounds(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    """Test `Span` raises the Python messages, and checks position types."""
    with pytest.raises(error) as info:
        Span(**arguments)

    assert str(info.value) == message


def test_span_checks_offsets_before_positions() -> None:
    """Test offsets out of order are reported before positions out of order."""
    with pytest.raises(ValueError, match="end_offset"):
        Span(5, 3, Position(2, 1), Position(1, 1))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda: _call(FileProvenance, 3),
            "FileProvenance file_path must be a str or os.PathLike, got int.",
            id="file_path",
        ),
        pytest.param(
            lambda: _call(FileProvenance, Path("a"), (0, 3)),
            "FileProvenance span must be a Span or None, got tuple.",
            id="span",
        ),
        pytest.param(
            lambda: _call(NamedProvenance, b"n", UnknownProvenance()),
            "NamedProvenance name must be a str, got bytes.",
            id="name",
        ),
        pytest.param(
            lambda: _call(NamedProvenance, "n", None),
            "NamedProvenance child must be a Provenance, got NoneType.",
            id="child",
        ),
        pytest.param(
            lambda: _call(CallSiteProvenance, UnknownProvenance(), "caller"),
            "CallSiteProvenance caller must be a Provenance, got str.",
            id="caller",
        ),
        pytest.param(
            lambda: _call(FusedProvenance, (UnknownProvenance(), 1)),
            "FusedProvenance sources must be Provenance instances, got int.",
            id="sources",
        ),
        pytest.param(
            lambda: _call(FusedProvenance, (), 1),
            "FusedProvenance metadata must be a str or None, got int.",
            id="metadata",
        ),
        pytest.param(
            lambda: _call(Provenance.fuse, UnknownProvenance(), Position(1, 1)),
            "Provenance.fuse provenances must be Provenance instances, got Position.",
            id="fuse_argument",
        ),
        pytest.param(
            lambda: _call(Provenance.fuse, metadata=b"m"),
            "Provenance.fuse metadata must be a str or None, got bytes.",
            id="fuse_metadata",
        ),
    ],
)
def test_arguments_of_the_wrong_type_raise_type_error(build: Any, message: str) -> None:
    """Test the stricter argument checks raise `TypeError` in the S2 style."""
    with pytest.raises(TypeError) as info:
        build()

    assert str(info.value) == message


def test_named_provenance_rejects_an_empty_name_with_the_python_message() -> None:
    """Test an empty name raises the dataclass's `ValueError`."""
    with pytest.raises(ValueError) as info:
        NamedProvenance("", UnknownProvenance())

    assert str(info.value) == '"name" must be non-empty'


# =============================================================================
# File paths
# =============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "",
        ".",
        "./a",
        "a/./b",
        "a//b/",
        "a/..",
        "/",
        "/.",
        "//",
        "//a",
        "///a",
        ".//a/.",
        "~/a b",
        "C:\\src\\a.fhy",
    ],
)
def test_file_path_is_normalized_as_pure_posix_path_does(text: str) -> None:
    """Test a path, given as `str` or `Path`, normalizes as `PurePosixPath`."""
    for file_path in (text, Path(text)):
        provenance = _call(FileProvenance, file_path)

        assert type(provenance.file_path) is type(Path(text))
        assert str(provenance.file_path) == str(PurePosixPath(text))
        assert str(provenance) == str(PurePosixPath(text))


def test_a_path_in_normal_form_is_kept_as_given() -> None:
    """Test a `PurePath` in normal form is returned as the same object."""
    path = Path("src/a.fhy")
    pure_path = PurePosixPath("src/b.fhy")

    assert FileProvenance(path).file_path is path
    assert _call(FileProvenance, pure_path).file_path is pure_path


def test_a_str_path_is_returned_as_a_path() -> None:
    """Test a `str` path comes back as a normalized `pathlib.Path`."""
    provenance = _call(FileProvenance, "./src//a.fhy")

    assert provenance.file_path == Path("src/a.fhy")
    assert provenance == FileProvenance(Path("src/a.fhy"))


class _PathLike(os.PathLike[str]):
    """Path-like object that is not a `PurePath`."""

    @override
    def __fspath__(self) -> str:
        return "./src//a.fhy"


def test_an_os_path_like_is_read_through_fspath() -> None:
    """Test an `os.PathLike` of a `str` is accepted and normalized."""
    assert _call(FileProvenance, _PathLike()).file_path == Path("src/a.fhy")


# =============================================================================
# Text, equality and ordering
# =============================================================================


def test_reprs_match_the_dataclasses() -> None:
    """Test each class renders in the dataclass `repr` form."""
    span = Span(0, 3, Position(1, 1), None)
    source = FileProvenance(Path("a.fhy"), span)

    assert repr(span) == (
        "Span(start_offset=0, end_offset=3, "
        "start_position=Position(line=1, column=1), end_position=None)"
    )
    assert repr(UnknownProvenance()) == "UnknownProvenance()"
    assert repr(source) == f"FileProvenance(file_path={Path('a.fhy')!r}, span={span!r})"
    assert repr(NamedProvenance("n", source)) == (
        f"NamedProvenance(name='n', child={source!r})"
    )
    assert repr(CallSiteProvenance(source, source)) == (
        f"CallSiteProvenance(callee={source!r}, caller={source!r})"
    )
    assert repr(FusedProvenance((source,), "m")) == (
        f"FusedProvenance(sources=({source!r},), metadata='m')"
    )


def test_equality_requires_the_same_class() -> None:
    """Test equality needs the same class, as a dataclass's does."""
    assert UnknownProvenance() != fhy_core._rs.UnknownProvenance()
    assert Position(1, 1) != fhy_core._rs.Position(1, 1)
    assert Position(1, 1) != (1, 1)
    assert FileProvenance(Path("a")) != "a"


def test_equal_provenances_hash_equally() -> None:
    """Test equal provenances built separately are equal and hash equally."""
    for left, right in zip(_build_provenances(), _build_provenances(), strict=True):
        assert left == right
        assert hash(left) == hash(right)
        assert left != NamedProvenance("other", left)


def test_position_orders_only_positions() -> None:
    """Test all four orderings, and `TypeError` against another class."""
    first, second = Position(1, 9), Position(2, 1)

    assert first < second
    assert first <= Position(1, 9)
    assert second > first
    assert second >= Position(2, 1)
    assert not second < first
    with pytest.raises(TypeError):
        _ = first < _call(tuple, (2, 1))


def test_pattern_matching_uses_the_dataclass_fields() -> None:
    """Test `__match_args__` lists each class's fields, as a dataclass's does."""
    provenance: Provenance = NamedProvenance("n", FileProvenance(Path("a.fhy")))

    match provenance:
        case NamedProvenance(name, FileProvenance(file_path, span)):
            assert (name, file_path, span) == ("n", Path("a.fhy"), None)
        case _:
            pytest.fail("the named provenance did not match its fields")


# =============================================================================
# Payloads
# =============================================================================


def test_payloads_keep_the_python_shapes() -> None:
    """Test the payloads nest the envelope and the dataclass field shapes."""
    source = FileProvenance(Path("a.fhy"), Span(0, 3, None, Position(1, 4)))
    unknown_payload = {"__type__": "provenance.unknown", "__data__": {}}
    source_payload = {
        "__type__": "provenance.file",
        "__data__": {
            "file_path": "a.fhy",
            "span": {
                "start_offset": 0,
                "end_offset": 3,
                "start_position": None,
                "end_position": {"line": 1, "column": 4},
            },
        },
    }

    assert UnknownProvenance().serialize_to_dict() == unknown_payload
    assert source.serialize_to_dict() == source_payload
    assert NamedProvenance("n", UnknownProvenance()).serialize_to_dict() == {
        "__type__": "provenance.named",
        "__data__": {"name": "n", "child": unknown_payload},
    }
    assert CallSiteProvenance(source, UnknownProvenance()).serialize_to_dict() == {
        "__type__": "provenance.call_site",
        "__data__": {"callee": source_payload, "caller": unknown_payload},
    }
    assert FusedProvenance((source,)).serialize_to_dict() == {
        "__type__": "provenance.fused",
        "__data__": {"sources": [source_payload], "metadata": None},
    }


@pytest.mark.parametrize("fmt", list(SerializationFormat))
def test_provenances_round_trip_in_every_format(fmt: SerializationFormat) -> None:
    """Test every shape round-trips in every format to its public class."""
    for provenance in _build_provenances():
        restored = Provenance.deserialize(provenance.serialize(fmt), fmt)

        assert restored == provenance
        assert type(restored) is type(provenance)


@pytest.mark.parametrize(
    ("cls", "data", "owner"),
    [
        pytest.param(Position, {"line": 1, "column": 2.0}, "Position", id="position"),
        pytest.param(Span, {"start_offset": 0}, "Span", id="span"),
        pytest.param(
            Provenance,
            {"__type__": "provenance.unknown", "__data__": {"extra": 1}},
            "UnknownProvenance",
            id="unknown",
        ),
        pytest.param(
            Provenance,
            {"__type__": "provenance.file", "__data__": {"file_path": 1, "span": None}},
            "FileProvenance",
            id="file",
        ),
        pytest.param(
            Provenance,
            {"__type__": "provenance.fused", "__data__": {"sources": [1]}},
            "FusedProvenance",
            id="fused",
        ),
    ],
)
def test_malformed_payload_raises_the_python_structure_error(
    cls: type[Serializable], data: SerializedDict, owner: str
) -> None:
    """Test a malformed payload raises the framework's structure error."""
    with pytest.raises(DeserializationDictStructureError) as info:
        cls.deserialize_from_dict(data)

    assert str(info.value).startswith(
        f'Invalid dictionary structure for deserializing to "{owner}".'
    )


def test_invalid_payload_value_raises_the_deserialization_value_error() -> None:
    """Test a constructor `ValueError` becomes `DeserializationValueError`."""
    data: SerializedDict = {
        "__type__": "provenance.named",
        "__data__": {"name": "", "child": UnknownProvenance().serialize_to_dict()},
    }

    with pytest.raises(DeserializationValueError) as info:
        Provenance.deserialize_from_dict(data)

    assert str(info.value) == '"name" must be non-empty'
    assert isinstance(info.value.__cause__, ValueError)


def test_decoding_a_file_payload_normalizes_its_path() -> None:
    """Test a payload's path is normalized as the constructor normalizes it."""
    data: SerializedDict = {"file_path": "./a//b/", "span": None}

    provenance = FileProvenance.deserialize_data_from_dict(data)

    assert provenance.file_path == Path("a/b")


def _build_file_payload(file_path: str, span: SerializedDict) -> SerializedDict:
    """Return the payload of a file provenance."""
    return {
        "__type__": "provenance.file",
        "__data__": {"file_path": file_path, "span": span},
    }


def _build_fused_payload(
    sources: list[SerializedDict], metadata: str | None
) -> SerializedDict:
    """Return the payload of a fused provenance."""
    return {
        "__type__": "provenance.fused",
        "__data__": {"metadata": metadata, "sources": sources},
    }


_UNKNOWN_PAYLOAD: SerializedDict = {"__type__": "provenance.unknown", "__data__": {}}
_SOURCE_PAYLOAD = _build_file_payload(
    "a.fhy",
    {
        "start_offset": 0,
        "end_offset": 3,
        "start_position": {"line": 1, "column": 1},
        "end_position": {"line": 1, "column": 4},
    },
)
_OTHER_PAYLOAD = _build_file_payload(
    "b/c.fhy",
    {
        "start_offset": 5,
        "end_offset": None,
        "start_position": None,
        "end_position": None,
    },
)
# The payload of each provenance `_build_provenances` returns, in order: the
# wire format the pure-Python classes wrote, which the Rust-backed classes
# keep (decision 4 of docs/design/python-switch.md).
_EXPECTED_PAYLOADS: list[SerializedDict] = [
    _UNKNOWN_PAYLOAD,
    _SOURCE_PAYLOAD,
    _OTHER_PAYLOAD,
    _build_file_payload(
        "d.fhy",
        {
            "start_offset": None,
            "end_offset": None,
            "start_position": None,
            "end_position": None,
        },
    ),
    {
        "__type__": "provenance.named",
        "__data__": {"name": "fhy.add", "child": _UNKNOWN_PAYLOAD},
    },
    {
        "__type__": "provenance.named",
        "__data__": {"name": "lib", "child": _SOURCE_PAYLOAD},
    },
    {
        "__type__": "provenance.call_site",
        "__data__": {
            "callee": {
                "__type__": "provenance.named",
                "__data__": {"name": "inlined", "child": _SOURCE_PAYLOAD},
            },
            "caller": _OTHER_PAYLOAD,
        },
    },
    _build_fused_payload([_SOURCE_PAYLOAD, _OTHER_PAYLOAD], None),
    _build_fused_payload(
        [_build_fused_payload([_SOURCE_PAYLOAD], "cse"), _UNKNOWN_PAYLOAD], ""
    ),
]


def test_payloads_keep_the_pinned_json_text() -> None:
    """Test every shape serializes to its pinned JSON text, keys sorted."""
    payloads = [provenance.to_json() for provenance in _build_provenances()]

    assert payloads == [
        json.dumps(payload, sort_keys=True) for payload in _EXPECTED_PAYLOADS
    ]


# =============================================================================
# fuse
# =============================================================================


def test_fuse_logs_its_reductions_as_python_does(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test `fuse` logs the dropped unknowns and collapsed fusions at debug."""
    a = FileProvenance(Path("a.fhy"))
    b = FileProvenance(Path("b.fhy"))
    inputs = (UnknownProvenance(), FusedProvenance((a, UnknownProvenance())), b)

    with caplog.at_level(logging.DEBUG, logger="fhy_core.provenance"):
        Provenance.fuse(*inputs)
        Provenance.fuse(a, b)

    assert [record.getMessage() for record in caplog.records] == [
        "reduced input (inputs=3, unknowns_dropped=2, fused_collapsed=1, "
        "final_sources=2)"
    ]


def test_fuse_flattens_deeply_nested_fusions_without_recursing() -> None:
    """Test `fuse` walks unlabelled fusions far deeper than the recursion limit."""
    source = FileProvenance(Path("a.fhy"))
    nested: Provenance = source
    for _ in range(sys.getrecursionlimit() * 3):
        nested = FusedProvenance((nested, UnknownProvenance()))

    assert Provenance.fuse(nested) is source


# =============================================================================
# Deep trees
# =============================================================================


def test_operations_on_a_tree_deeper_than_the_recursion_limit_raise() -> None:
    """Test `==`, `hash` and `str` raise `RecursionError` instead of crashing."""
    depth = sys.getrecursionlimit() * 3
    left: Provenance = UnknownProvenance()
    right: Provenance = UnknownProvenance()
    for _ in range(depth):
        left = NamedProvenance("n", left)
        right = NamedProvenance("n", right)

    with pytest.raises(RecursionError):
        _ = left == right
    with pytest.raises(RecursionError):
        hash(left)
    with pytest.raises(RecursionError):
        str(left)
    assert isinstance(left, NamedProvenance)
    assert left.name == "n"


def test_operations_on_a_tree_within_the_recursion_limit_succeed() -> None:
    """Test a tree somewhat deeper than the shallow fast path still works."""
    left: Provenance = UnknownProvenance()
    right: Provenance = UnknownProvenance()
    for _ in range(200):
        left = CallSiteProvenance(left, UnknownProvenance())
        right = CallSiteProvenance(right, UnknownProvenance())

    assert left == right
    assert hash(left) == hash(right)
    assert str(left).count(" at ") == 200


# =============================================================================
# Frozen instances
# =============================================================================


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(Position(1, 1), id="Position"),
        pytest.param(Span(), id="Span"),
        *(
            pytest.param(provenance, id=type(provenance).__name__)
            for provenance in _build_provenances()[:2]
        ),
    ],
)
def test_mutation_raises_the_frozen_mixin_error(instance: Any) -> None:
    """Test assigning or deleting an attribute raises `FrozenMixin`'s error."""
    with pytest.raises(FrozenMutationError) as set_info:
        instance.span = None
    with pytest.raises(FrozenMutationError) as delete_info:
        del instance.line

    frozen = f"on frozen {type(instance).__name__}."
    assert str(set_info.value) == f'Cannot modify "span" {frozen}'
    assert str(delete_info.value) == f'Cannot delete "line" {frozen}'
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
    """Test a provenance pickles as calls of the public classes, never of `_rs`."""
    provenances = _build_provenances()
    unpickler = _GlobalRecordingUnpickler(pickle.dumps(provenances))

    restored = unpickler.load()

    assert restored == provenances
    assert [type(value) for value in restored] == [type(p) for p in provenances]
    assert {module for module, _ in unpickler.found_globals} <= {
        "builtins",
        "pathlib",
        "fhy_core.provenance",
    }
    assert ("fhy_core.provenance", "Span") in unpickler.found_globals


def test_copies_are_equal_new_objects() -> None:
    """Test `copy` and `deepcopy` build equal objects through the constructor."""
    provenance = _build_provenances()[-1]

    assert copy.copy(provenance) == provenance
    assert copy.deepcopy(provenance) == provenance
    assert copy.deepcopy(provenance) is not provenance


@pytest.mark.slow
@pytest.mark.subprocess
def test_pickles_load_across_processes() -> None:
    """Test provenances pickled in one process load in another one."""
    provenances = _build_provenances()
    this_process_pickle = base64.b64encode(pickle.dumps(provenances)).decode("ascii")

    fresh_process_pickle = _run_python_in_a_fresh_process(
        _BUILD_PROVENANCES_SOURCE
        + "import base64, pickle, sys\n"
        + "restored = pickle.loads(base64.b64decode(sys.stdin.read()))\n"
        + "assert restored == provenances\n"
        + "print(base64.b64encode(pickle.dumps(restored)).decode('ascii'))",
        stdin=this_process_pickle,
    )

    assert pickle.loads(base64.b64decode(fresh_process_pickle)) == provenances
