"""Generate golden cases from the `provenance` and `diagnostic` oracles.

Drives the real `fhy_core.provenance` and `fhy_core.diagnostic` modules
through hand-picked and randomly generated operation scripts, recording the
observed outcome of each operation. The Rust equivalence test
(`rust/fhy-core/tests/provenance_diagnostic_equivalence.rs`) replays these
scripts against `fhy_core::provenance` and `fhy_core::diagnostic` and
compares every observation.

A `provenance` script constructs positions, spans and provenances, fuses
provenance trees, and decodes payloads. Every provenance an operation takes
as input is given in its wire (dict) form, and every result is recorded in
its wire form together with its `str()` text.

A `diagnostic` script constructs note kinds and notes, decodes note payloads,
and formats validation reports. Note kinds are named by slot rather than by
identifier id, since concrete ids are process-local in both runtimes: each
script mints one `Identifier` per fresh slot the first time the script refers
to it, and the slots `rationale`, `suggestion`, `remark` and `other` refer to
the shipped note kinds instead.

Every integer this generator feeds the oracle fits in an unsigned 64-bit
integer: the Rust port stores lines, columns and offsets as `u64`, so larger
values are a documented divergence tested on the Rust side only.

Run from the repository root:

    uv run --no-sync python \
        rust/fhy-core/tests/golden/generate_provenance_diagnostic_cases.py

This overwrites
`rust/fhy-core/tests/golden/provenance_diagnostic_cases.json`. Options select
a larger random corpus written elsewhere, which the ignored expanded-corpus
equivalence test replays from the file named in
`FHY_PROVENANCE_DIAGNOSTIC_CORPUS`.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _golden_support import add_corpus_arguments, build_provenance, write_document

from fhy_core.diagnostic import (
    OTHER_NOTE_KIND,
    RATIONALE_NOTE_KIND,
    REMARK_NOTE_KIND,
    SUGGESTION_NOTE_KIND,
    Diagnostic,
    DiagnosticLevel,
    Note,
    NoteKind,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.provenance import (
    CallSiteProvenance,
    FileProvenance,
    FusedProvenance,
    NamedProvenance,
    Position,
    Provenance,
    Span,
)

GENERATOR_COMMAND = (
    "uv run --no-sync python "
    "rust/fhy-core/tests/golden/generate_provenance_diagnostic_cases.py"
)

_NOTE_KIND_DEFAULTS: dict[str, NoteKind] = {
    "rationale": RATIONALE_NOTE_KIND,
    "suggestion": SUGGESTION_NOTE_KIND,
    "remark": REMARK_NOTE_KIND,
    "other": OTHER_NOTE_KIND,
}

_RANDOM_SEED = 20260922
_RANDOM_SCRIPT_COUNT = 30
_RANDOM_MAX_OPS = 12
_RANDOM_TREE_DEPTH = 3
# Chance that a random tree node below the depth limit is a leaf.
_RANDOM_LEAF_PROBABILITY = 0.35
# Chance that a random `equals` op compares a tree with a copy of itself.
_RANDOM_EQUAL_PAIR_PROBABILITY = 0.5
# Chance that a random diagnostic op formats a report rather than a note.
_RANDOM_REPORT_PROBABILITY = 0.8

_U64_MAX = 2**64 - 1

_PROVENANCE_KIND = "provenance"
_DIAGNOSTIC_KIND = "diagnostic"

_RANDOM_FILE_PATHS = (
    "a.fhy",
    "b.fhy",
    "c.fhy",
    "./a.fhy",
    "dir//b.fhy",
    "dir/./c.fhy",
    "dir/../a.fhy",
    "trailing/",
    "/abs/path.fhy",
    "",
    "héllo/wörld.fhy",
    "with space.fhy",
)
_RANDOM_NAMES = ("n", "fhy.add", "mylib::matmul", " ", "inlined", "é")
_RANDOM_METADATA = (None, None, "cse", "loop-fusion", "")
_RANDOM_TEXTS = (
    "",
    "m",
    "missing return",
    "line one\nline two",
    "{x} braces",
    "café",
    "  padded  ",
    "tab\there",
)
_RANDOM_SOURCES = ("v", "shape.check", "scope.check", "a.b.c", "é")
_LEVELS = (DiagnosticLevel.ERROR, DiagnosticLevel.WARNING, DiagnosticLevel.INFO)


# =============================================================================
# Recording helpers
# =============================================================================


def _record_error(error: Exception) -> dict[str, Any]:
    return {"error": {"type": type(error).__name__, "message": str(error)}}


def _record_provenance(provenance: Provenance) -> dict[str, Any]:
    return {"value": provenance.serialize_to_dict(), "str": str(provenance)}


def _decode_provenance(payload: dict[str, Any]) -> Provenance:
    return Provenance.deserialize_from_dict(payload)


def _decode_span(payload: dict[str, Any] | None) -> Span | None:
    return None if payload is None else Span.deserialize_from_dict(payload)


def _build_position_pair(pair: list[int] | None) -> Position | None:
    return None if pair is None else Position(pair[0], pair[1])


# =============================================================================
# Wire-form builders (the inputs of provenance operations)
# =============================================================================


def _wrap(type_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"__type__": type_id, "__data__": data}


def _unknown() -> dict[str, Any]:
    return _wrap("provenance.unknown", {})


def _file(path: str, span: dict[str, Any] | None = None) -> dict[str, Any]:
    return _wrap("provenance.file", {"file_path": path, "span": span})


def _named(name: str, child: dict[str, Any]) -> dict[str, Any]:
    return _wrap("provenance.named", {"name": name, "child": child})


def _call_site(callee: dict[str, Any], caller: dict[str, Any]) -> dict[str, Any]:
    return _wrap("provenance.call_site", {"callee": callee, "caller": caller})


def _fused(
    sources: list[dict[str, Any]], metadata: str | None = None
) -> dict[str, Any]:
    return _wrap("provenance.fused", {"sources": sources, "metadata": metadata})


def _span(
    start_offset: int | None = None,
    end_offset: int | None = None,
    start_position: tuple[int, int] | None = None,
    end_position: tuple[int, int] | None = None,
) -> dict[str, Any]:
    def position(pair: tuple[int, int] | None) -> dict[str, int] | None:
        return None if pair is None else {"line": pair[0], "column": pair[1]}

    return {
        "start_offset": start_offset,
        "end_offset": end_offset,
        "start_position": position(start_position),
        "end_position": position(end_position),
    }


# =============================================================================
# Provenance operations
# =============================================================================


def _run_position_new(op: dict[str, Any]) -> dict[str, Any]:
    try:
        position = Position(op["line"], op["column"])
    except (TypeError, ValueError) as error:
        return _record_error(error)
    return {"value": position.serialize_to_dict(), "str": str(position)}


def _run_span_new(op: dict[str, Any]) -> dict[str, Any]:
    try:
        span = Span(
            op["start_offset"],
            op["end_offset"],
            _build_position_pair(op["start_position"]),
            _build_position_pair(op["end_position"]),
        )
    except (TypeError, ValueError) as error:
        return _record_error(error)
    return {
        "value": span.serialize_to_dict(),
        "str": str(span),
        "is_unknown": span.is_unknown(),
    }


def _run_file_new(op: dict[str, Any]) -> dict[str, Any]:
    provenance = FileProvenance(Path(op["file_path"]), _decode_span(op["span"]))
    return _record_provenance(provenance)


def _run_named_new(op: dict[str, Any]) -> dict[str, Any]:
    try:
        provenance = NamedProvenance(op["name"], _decode_provenance(op["child"]))
    except ValueError as error:
        return _record_error(error)
    return _record_provenance(provenance)


def _run_call_site_new(op: dict[str, Any]) -> dict[str, Any]:
    provenance = CallSiteProvenance(
        _decode_provenance(op["callee"]), _decode_provenance(op["caller"])
    )
    return _record_provenance(provenance)


def _run_fused_new(op: dict[str, Any]) -> dict[str, Any]:
    sources = tuple(_decode_provenance(source) for source in op["sources"])
    return _record_provenance(FusedProvenance(sources, op["metadata"]))


def _run_fuse(op: dict[str, Any]) -> dict[str, Any]:
    inputs = [_decode_provenance(value) for value in op["inputs"]]
    return _record_provenance(Provenance.fuse(*inputs, metadata=op["metadata"]))


def _run_equals(op: dict[str, Any]) -> dict[str, Any]:
    left = _decode_provenance(op["left"])
    right = _decode_provenance(op["right"])
    return {"result": left == right}


_DECODERS: dict[str, Callable[[Any], Any]] = {
    "position": Position.deserialize_from_dict,
    "span": Span.deserialize_from_dict,
    "provenance": Provenance.deserialize_from_dict,
}


def _run_decode(op: dict[str, Any]) -> dict[str, Any]:
    decoder = _DECODERS[op["family"]]
    try:
        value = decoder(copy.deepcopy(op["payload"]))
    except Exception as error:
        return _record_error(error)
    return {"value": value.serialize_to_dict(), "str": str(value)}


_PROVENANCE_OPS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "position_new": _run_position_new,
    "span_new": _run_span_new,
    "file_new": _run_file_new,
    "named_new": _run_named_new,
    "call_site_new": _run_call_site_new,
    "fused_new": _run_fused_new,
    "fuse": _run_fuse,
    "equals": _run_equals,
    "decode": _run_decode,
}


# =============================================================================
# Diagnostic operations (note kinds named by slot)
# =============================================================================


class _SlotContext:
    """Map one script's note-kind slots to identifiers.

    Default slots name the shipped note kinds. A fresh slot mints one
    `Identifier` the first time the script refers to it.
    """

    def __init__(self) -> None:
        self.identifiers: dict[str, Identifier] = {
            slot: kind.name for slot, kind in _NOTE_KIND_DEFAULTS.items()
        }
        self.id_to_slot: dict[int, str] = {
            identifier.id: slot for slot, identifier in self.identifiers.items()
        }

    def resolve_identifier(self, slot: str) -> Identifier:
        if slot not in self.identifiers:
            identifier = Identifier(slot)
            self.identifiers[slot] = identifier
            self.id_to_slot[identifier.id] = slot
        return self.identifiers[slot]

    def resolve_kind(self, slot: str) -> NoteKind:
        kind = NoteKind.get_interned(self.resolve_identifier(slot))
        if kind is None:
            raise KeyError(f"note-kind slot {slot!r} is not registered")
        return kind

    def normalize_note_dict(self, raw: dict[str, Any]) -> dict[str, Any]:
        kind = raw["kind"]
        name = kind["name"]
        return {
            **raw,
            "kind": {**kind, "name": {**name, "id": self.id_to_slot[name["id"]]}},
        }

    def denormalize_note_dict(self, normalized: dict[str, Any]) -> dict[str, Any]:
        """Return a note payload with each slot-valued id replaced by its id.

        Only a string `id` is a slot; anything else is left as written, so a
        payload can carry a malformed id on purpose.
        """
        payload = copy.deepcopy(normalized)
        kind = payload.get("kind")
        if isinstance(kind, dict):
            name = kind.get("name")
            if isinstance(name, dict) and isinstance(name.get("id"), str):
                name["id"] = self.resolve_identifier(name["id"]).id
        return payload


def _run_note_kind_new(ctx: _SlotContext, op: dict[str, Any]) -> dict[str, Any]:
    identifier = ctx.resolve_identifier(op["slot"])
    before = NoteKind.get_interned(identifier)
    NoteKind(identifier, op["description"])
    return {
        "registered": before is None,
        "canonical_description": ctx.resolve_kind(op["slot"]).description,
    }


def _run_note_new(ctx: _SlotContext, op: dict[str, Any]) -> dict[str, Any]:
    note = Note(op["message"], ctx.resolve_kind(op["kind_slot"]))
    return {
        "value": ctx.normalize_note_dict(note.serialize_to_dict()),
        "str": str(note),
    }


def _run_note_decode(ctx: _SlotContext, op: dict[str, Any]) -> dict[str, Any]:
    payload = ctx.denormalize_note_dict(op["payload"])
    try:
        note = Note.deserialize_from_dict(payload)
    except Exception as error:
        return _record_error(error)
    kind_slot = ctx.id_to_slot[note.kind.name.id]
    return {
        "value": ctx.normalize_note_dict(note.serialize_to_dict()),
        "str": str(note),
        "kind_slot": kind_slot,
        "kind_is_canonical": NoteKind.get_interned(note.kind.name) is note.kind,
    }


def _build_diagnostic(ctx: _SlotContext, spec: dict[str, Any]) -> Diagnostic:
    note = Note(spec["message"], ctx.resolve_kind(spec["kind_slot"]))
    return Diagnostic(
        DiagnosticLevel(spec["level"]), note, spec["source"], spec["detail"]
    )


def _run_report(ctx: _SlotContext, op: dict[str, Any]) -> dict[str, Any]:
    diagnostics = tuple(_build_diagnostic(ctx, spec) for spec in op["diagnostics"])
    report: ValidationReport[Any] = ValidationReport(diagnostics=diagnostics)

    def indices_of(selected: tuple[Diagnostic, ...]) -> list[int]:
        identities = {id(diagnostic) for diagnostic in selected}
        return [
            index
            for index, diagnostic in enumerate(diagnostics)
            if id(diagnostic) in identities
        ]

    try:
        report.raise_if_failed()
        failure_text = None
    except ValidationFailedError as error:
        failure_text = str(error)
    return {
        "format": report.format(),
        "has_errors": report.has_errors(),
        "errors": indices_of(report.errors()),
        "warnings": indices_of(report.warnings()),
        "infos": indices_of(report.infos()),
        "failure_text": failure_text,
    }


_DIAGNOSTIC_OPS: dict[str, Callable[[_SlotContext, dict[str, Any]], dict[str, Any]]] = {
    "note_kind_new": _run_note_kind_new,
    "note_new": _run_note_new,
    "note_decode": _run_note_decode,
    "report": _run_report,
}


# =============================================================================
# Script runners
# =============================================================================


def _run_provenance_script(name: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    recorded = [{**op, "expected": _PROVENANCE_OPS[op["op"]](op)} for op in ops]
    return {"name": name, "type": _PROVENANCE_KIND, "ops": recorded}


def _run_diagnostic_script(name: str, ops: list[dict[str, Any]]) -> dict[str, Any]:
    ctx = _SlotContext()
    recorded = [{**op, "expected": _DIAGNOSTIC_OPS[op["op"]](ctx, op)} for op in ops]
    return {"name": name, "type": _DIAGNOSTIC_KIND, "ops": recorded}


# =============================================================================
# Hand-picked provenance scripts
# =============================================================================


def _position_op(line: int, column: int) -> dict[str, Any]:
    return {"op": "position_new", "line": line, "column": column}


def _span_op(
    start_offset: int | None = None,
    end_offset: int | None = None,
    start_position: list[int] | None = None,
    end_position: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "op": "span_new",
        "start_offset": start_offset,
        "end_offset": end_offset,
        "start_position": start_position,
        "end_position": end_position,
    }


def _file_op(path: str, span: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"op": "file_new", "file_path": path, "span": span}


def _fuse_op(
    inputs: list[dict[str, Any]], metadata: str | None = None
) -> dict[str, Any]:
    return {"op": "fuse", "inputs": inputs, "metadata": metadata}


def _decode_op(family: str, payload: Any) -> dict[str, Any]:
    return {"op": "decode", "family": family, "payload": payload}


def _list_position_and_span_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "position_construction",
            [
                _position_op(1, 1),
                _position_op(2, 8),
                _position_op(0, 1),
                _position_op(1, 0),
                _position_op(0, 0),
                _position_op(_U64_MAX, _U64_MAX),
            ],
        ),
        (
            "span_construction_and_rendering",
            [
                _span_op(),
                _span_op(0, 3),
                _span_op(5, None),
                _span_op(None, 3),
                _span_op(4, 4),
                _span_op(5, 3),
                _span_op(0, _U64_MAX),
                _span_op(None, None, [1, 1], [1, 4]),
                _span_op(None, None, [1, 1], None),
                _span_op(None, None, None, [1, 4]),
                _span_op(None, None, [1, 1], [1, 1]),
                _span_op(None, None, [2, 1], [1, 1]),
                _span_op(None, None, [1, 99], [2, 1]),
                _span_op(0, 3, [1, 1], [1, 4]),
                _span_op(0, 3, [1, 1], None),
                _span_op(9, 12, [3, 1], [1, 1]),
                _span_op(5, 3, [2, 1], [1, 1]),
                _span_op(7, None, None, [4, 2]),
            ],
        ),
    ]


def _list_file_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    paths = [
        "a.fhy",
        "./a",
        "a/",
        "a//b",
        "a/./b",
        "///a",
        "//a",
        "",
        ".",
        "./",
        "a/.",
        "/./a",
        "a/../b",
        "../a",
        "./../a",
        "a/..",
        "~/x",
        ".a",
        "/",
        "//",
        "////a//b/",
        "héllo/wörld.fhy",
        "a b/c d.fhy",
    ]
    offsets = _span(0, 3)
    positions = _span(None, None, (1, 1), (1, 4))
    mixed = _span(0, 3, (1, 1), (1, 4))
    return [
        ("file_path_normalization", [_file_op(path) for path in paths]),
        (
            "file_rendering_with_spans",
            [
                _file_op("a.fhy", _span()),
                _file_op("a.fhy", offsets),
                _file_op("a.fhy", positions),
                _file_op("a.fhy", mixed),
                _file_op("./x//y.fhy", offsets),
                _file_op("a.fhy", _span(5)),
            ],
        ),
        (
            "file_equality_follows_normalization",
            [
                {"op": "equals", "left": _file("./a.fhy"), "right": _file("a.fhy")},
                {"op": "equals", "left": _file("a//b/"), "right": _file("a/b")},
                {"op": "equals", "left": _file("a/../b"), "right": _file("b")},
                {"op": "equals", "left": _file("a"), "right": _file("a", _span())},
                {"op": "equals", "left": _file("//a"), "right": _file("/a")},
            ],
        ),
    ]


def _list_variant_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    a = _file("a.fhy")
    b = _file("b.fhy")
    c = _file("c.fhy")
    return [
        (
            "named_construction_and_rendering",
            [
                {"op": "named_new", "name": "fhy.add", "child": _unknown()},
                {
                    "op": "named_new",
                    "name": "mylib::matmul",
                    "child": _file("mylib.fhyobj"),
                },
                {"op": "named_new", "name": "", "child": _unknown()},
                {"op": "named_new", "name": " ", "child": _unknown()},
                {"op": "named_new", "name": "outer", "child": _named("inner", a)},
                {"op": "named_new", "name": "empty-fusion", "child": _fused([])},
                {"op": "named_new", "name": "over-fused", "child": _fused([a, b])},
            ],
        ),
        (
            "call_site_construction_and_rendering",
            [
                {
                    "op": "call_site_new",
                    "callee": _file("callee.fhy"),
                    "caller": _file("caller.fhy"),
                },
                {"op": "call_site_new", "callee": _call_site(a, b), "caller": c},
                {"op": "call_site_new", "callee": _unknown(), "caller": _unknown()},
                {"op": "call_site_new", "callee": _named("inlined", a), "caller": b},
            ],
        ),
        (
            "fused_direct_construction",
            [
                {"op": "fused_new", "sources": [a, b], "metadata": None},
                {"op": "fused_new", "sources": [a], "metadata": "loop-fusion"},
                {"op": "fused_new", "sources": [], "metadata": None},
                {"op": "fused_new", "sources": [a, b], "metadata": ""},
                {"op": "fused_new", "sources": [a], "metadata": None},
                {"op": "fused_new", "sources": [_unknown(), a], "metadata": None},
                {
                    "op": "fused_new",
                    "sources": [_fused([a, b], "cse"), c],
                    "metadata": None,
                },
                {"op": "equals", "left": _fused([]), "right": _unknown()},
                {"op": "equals", "left": _fused([a]), "right": a},
                {"op": "equals", "left": _unknown(), "right": _unknown()},
            ],
        ),
    ]


def _list_fuse_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    a = _file("a.fhy")
    b = _file("b.fhy")
    c = _file("c.fhy")
    offsets_a = _file("a.fhy", _span(0, 3))
    offsets_b = _file("b.fhy", _span(10, 13))
    return [
        (
            "fuse_reduction_rules",
            [
                _fuse_op([]),
                _fuse_op([_unknown(), _unknown()]),
                _fuse_op([a]),
                _fuse_op([a], "cse"),
                _fuse_op([a, _unknown(), b]),
                _fuse_op([_fused([a, b]), c]),
                _fuse_op([_fused([a, b], "cse"), c]),
                _fuse_op([a, b, c]),
                _fuse_op([a, a]),
                _fuse_op([a, b], "loop-fusion"),
                _fuse_op([_fused([_unknown(), a]), b]),
                _fuse_op([_fused([a, _fused([b, c])])]),
                _fuse_op([_fused([a, _fused([b, c], "cse")])]),
                _fuse_op(
                    [_unknown(), _fused([a, b]), _unknown(), _fused([b, c], "x"), c]
                ),
                _fuse_op([offsets_a, offsets_b], "loop-fusion"),
            ],
        ),
        (
            "fuse_edge_cases",
            [
                _fuse_op([], "m"),
                _fuse_op([_unknown()], "m"),
                _fuse_op([_fused([])]),
                _fuse_op([_fused([]), a]),
                _fuse_op([_fused([a])]),
                _fuse_op([_fused([a], "")]),
                _fuse_op([a, b], ""),
                _fuse_op([_fused([a, b], "")]),
                _fuse_op([_fused([a, b], "cse")]),
                _fuse_op([_fused([a, b], "cse")], "cse"),
                _fuse_op([b, a]),
                _fuse_op([_named("n", _fused([_unknown(), a]))]),
                _fuse_op([_call_site(_fused([a, b]), _unknown())]),
                _fuse_op([_fused([_fused([_fused([a])])])]),
                _fuse_op([_fused([_unknown(), _fused([_unknown()])])], "m"),
            ],
        ),
        (
            "fuse_associativity_without_metadata",
            [
                _fuse_op([_fused([a, b]), c]),
                _fuse_op([a, _fused([b, c])]),
                _fuse_op([a, b, c]),
                _fuse_op([_fused([a, b], "m"), c]),
                _fuse_op([a, b, c], "m"),
            ],
        ),
    ]


def _list_decode_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    a = _file("a.fhy")
    full_span = _span(0, 3, (1, 1), (1, 4))
    return [
        (
            "decode_position",
            [
                _decode_op("position", {"line": 2, "column": 8}),
                _decode_op("position", {"line": 1}),
                _decode_op("position", {"line": True, "column": 1}),
                _decode_op("position", {"line": 1, "column": 2, "z": 3}),
                _decode_op("position", {"line": 0, "column": 1}),
                _decode_op("position", {"line": 1, "column": 0}),
                _decode_op("position", {"line": 1.0, "column": 1}),
                _decode_op("position", {"line": "1", "column": 1}),
                _decode_op("position", {"line": -1, "column": 1}),
                _decode_op("position", {"line": _U64_MAX, "column": 1}),
                _decode_op("position", {"line": None, "column": 1}),
            ],
        ),
        (
            "decode_span",
            [
                _decode_op("span", full_span),
                _decode_op("span", _span()),
                _decode_op("span", {"start_offset": 0}),
                _decode_op("span", {**_span(0, 3), "extra": 1}),
                _decode_op("span", _span(5, 3)),
                _decode_op("span", _span(None, None, (2, 1), (1, 1))),
                _decode_op("span", _span(-1)),
                _decode_op("span", {**_span(), "start_offset": True}),
                _decode_op(
                    "span", {**_span(), "start_position": {"line": 0, "column": 1}}
                ),
                _decode_op("span", {**_span(), "start_position": [1, 1]}),
            ],
        ),
        (
            "decode_provenance_variants",
            [
                _decode_op("provenance", _unknown()),
                _decode_op("provenance", _file("c.fhy")),
                _decode_op("provenance", _file("a.fhy", full_span)),
                _decode_op("provenance", _file("./x//y.fhy/", None)),
                _decode_op("provenance", _named("fhy.add", _unknown())),
                _decode_op(
                    "provenance", _named("mylib::matmul", _file("mylib.fhyobj"))
                ),
                _decode_op(
                    "provenance",
                    _call_site(_file("a.fhy", _span(0, 10)), _file("b.fhy")),
                ),
                _decode_op("provenance", _fused([a, _file("b.fhy")])),
                _decode_op("provenance", _fused([a], "loop-fusion")),
                _decode_op("provenance", _fused([])),
                _decode_op("provenance", _fused([_unknown(), a])),
            ],
        ),
        (
            "decode_provenance_rejections",
            [
                _decode_op(
                    "provenance",
                    {"__type__": "provenance.does_not_exist", "__data__": {}},
                ),
                _decode_op("provenance", {"__type__": "provenance.unknown"}),
                _decode_op(
                    "provenance",
                    {"__type__": "provenance.unknown", "__data__": {"x": 1}},
                ),
                _decode_op(
                    "provenance", {"__type__": "provenance.unknown", "__data__": None}
                ),
                _decode_op("provenance", {**_unknown(), "z": 1}),
                _decode_op("provenance", {"__data__": {}}),
                _decode_op(
                    "provenance",
                    {"__type__": "position", "__data__": {"line": 1, "column": 1}},
                ),
                _decode_op(
                    "provenance",
                    {"__type__": "provenance.file", "__data__": {"file_path": "a"}},
                ),
                _decode_op(
                    "provenance",
                    {
                        "__type__": "provenance.file",
                        "__data__": {"file_path": 3, "span": None},
                    },
                ),
                _decode_op("provenance", _file("a", {"start_offset": 0})),
                _decode_op("provenance", _named("", _unknown())),
                _decode_op("provenance", _named("n", {"line": 1, "column": 1})),
                _decode_op("provenance", _call_site(_unknown(), None)),  # type: ignore[arg-type]
                _decode_op(
                    "provenance",
                    {
                        "__type__": "provenance.fused",
                        "__data__": {"sources": [], "metadata": 5},
                    },
                ),
                _decode_op(
                    "provenance",
                    {
                        "__type__": "provenance.fused",
                        "__data__": {"sources": None, "metadata": None},
                    },
                ),
                _decode_op(
                    "provenance",
                    {
                        "__type__": "provenance.fused",
                        "__data__": {"sources": [_unknown()]},
                    },
                ),
                _decode_op("provenance", _fused([_named("", _unknown())])),
                _decode_op("provenance", [_unknown()]),
            ],
        ),
    ]


# =============================================================================
# Hand-picked diagnostic scripts
# =============================================================================


def _diagnostic_spec(
    level: str,
    message: str,
    source: str,
    detail: str | None = None,
    kind_slot: str = "other",
) -> dict[str, Any]:
    return {
        "level": level,
        "message": message,
        "kind_slot": kind_slot,
        "source": source,
        "detail": detail,
    }


def _report_op(*specs: dict[str, Any]) -> dict[str, Any]:
    return {"op": "report", "diagnostics": list(specs)}


def _note_payload(
    slot: str, name_hint: str, description: str, message: str
) -> dict[str, Any]:
    return {
        "message": message,
        "kind": {
            "name": {"id": slot, "name_hint": name_hint},
            "description": description,
        },
    }


def _list_note_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    other_description = OTHER_NOTE_KIND.description
    return [
        (
            "notes_with_default_kinds",
            [
                {"op": "note_new", "message": "lowered from ast", "kind_slot": "other"},
                {
                    "op": "note_new",
                    "message": "tiled for locality",
                    "kind_slot": "rationale",
                },
                {
                    "op": "note_new",
                    "message": "use a smaller tile",
                    "kind_slot": "suggestion",
                },
                {"op": "note_new", "message": "", "kind_slot": "remark"},
                {"op": "note_new", "message": "two\nlines", "kind_slot": "other"},
            ],
        ),
        (
            "notes_with_a_custom_kind",
            [
                {
                    "op": "note_kind_new",
                    "slot": "performance",
                    "description": "An optimization remark.",
                },
                {
                    "op": "note_kind_new",
                    "slot": "performance",
                    "description": "a different description",
                },
                {"op": "note_new", "message": "vectorized", "kind_slot": "performance"},
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "performance",
                        "performance",
                        "An optimization remark.",
                        "vectorized",
                    ),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "performance", "performance", "divergent", "vectorized"
                    ),
                },
            ],
        ),
        (
            "note_decoding",
            [
                {
                    "op": "note_decode",
                    "payload": _note_payload("other", "other", other_description, "hi"),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "rationale", "rationale", RATIONALE_NOTE_KIND.description, "why"
                    ),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "other", "renamed", other_description, "hint mismatch"
                    ),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "other", "other", "changed", "description mismatch"
                    ),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "fresh", "fresh", "decoded first", "new kind"
                    ),
                },
                {
                    "op": "note_decode",
                    "payload": _note_payload(
                        "fresh", "fresh", "decoded second", "again"
                    ),
                },
                {"op": "note_decode", "payload": {"message": "x"}},
                {
                    "op": "note_decode",
                    "payload": {
                        **_note_payload("other", "other", other_description, "x"),
                        "extra": 1,
                    },
                },
                {
                    "op": "note_decode",
                    "payload": {
                        "message": 5,
                        "kind": _note_payload("other", "other", other_description, "")[
                            "kind"
                        ],
                    },
                },
                {
                    "op": "note_decode",
                    "payload": {
                        "message": "x",
                        "kind": {"name": {"id": "other", "name_hint": "other"}},
                    },
                },
                {
                    "op": "note_decode",
                    "payload": {
                        "message": "x",
                        "kind": {
                            "name": {"id": -1, "name_hint": "other"},
                            "description": "",
                        },
                    },
                },
                {"op": "note_decode", "payload": {"message": "x", "kind": None}},
            ],
        ),
    ]


def _list_report_scripts() -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (
            "report_formatting",
            [
                _report_op(),
                _report_op(
                    _diagnostic_spec("error", "bad", "v1"),
                    _diagnostic_spec("warning", "meh", "v2"),
                    _diagnostic_spec("info", "fyi", "v3"),
                    _diagnostic_spec("error", "worse", "v4"),
                ),
                _report_op(
                    _diagnostic_spec(
                        "error",
                        "missing return",
                        "shape.check",
                        "function foo() has no return statement",
                    ),
                    _diagnostic_spec("warning", "unused", "scope.check"),
                ),
                _report_op(_diagnostic_spec("warning", "ok-ish", "v")),
                _report_op(_diagnostic_spec("info", "fyi", "v")),
                _report_op(_diagnostic_spec("error", "boom", "v.explode")),
                _report_op(_diagnostic_spec("error", "m", "s", "")),
                _report_op(
                    _diagnostic_spec("error", "m", "s"),
                    _diagnostic_spec("info", "i\nj", "t", "x\ny"),
                ),
                _report_op(
                    _diagnostic_spec("info", "kinds are hidden", "s", None, "rationale")
                ),
                _report_op(_diagnostic_spec("warning", "", "", " ")),
                _report_op(_diagnostic_spec("error", "dict {x} missing", "s", "{y}")),
            ],
        ),
    ]


# =============================================================================
# Random scripts
# =============================================================================


def _choose_random_offset_span(rng: random.Random) -> dict[str, Any]:
    start = rng.randrange(0, 50)
    end = rng.choice([None, start + rng.randrange(0, 20)])
    return _span(rng.choice([None, start]), end)


def _choose_random_position_span(rng: random.Random) -> dict[str, Any]:
    start = (rng.randrange(1, 9), rng.randrange(1, 9))
    end = max(start, (start[0] + rng.randrange(0, 3), rng.randrange(1, 9)))
    return _span(
        rng.choice([None, rng.randrange(0, 50)]),
        None,
        rng.choice([None, start]),
        rng.choice([None, end]),
    )


def _choose_random_span(rng: random.Random) -> dict[str, Any] | None:
    chooser = rng.choice(
        [
            lambda _: None,
            lambda _: _span(),
            _choose_random_offset_span,
            _choose_random_position_span,
        ]
    )
    return chooser(rng)


def _choose_random_tree(rng: random.Random, depth: int) -> dict[str, Any]:
    leaf_only = depth <= 0 or rng.random() < _RANDOM_LEAF_PROBABILITY
    if leaf_only:
        choice = rng.randrange(3)
        if choice == 0:
            return _unknown()
        return _file(rng.choice(_RANDOM_FILE_PATHS), _choose_random_span(rng))
    choice = rng.randrange(3)
    if choice == 0:
        return _named(rng.choice(_RANDOM_NAMES), _choose_random_tree(rng, depth - 1))
    if choice == 1:
        return _call_site(
            _choose_random_tree(rng, depth - 1), _choose_random_tree(rng, depth - 1)
        )
    sources = [_choose_random_tree(rng, depth - 1) for _ in range(rng.randrange(0, 4))]
    return _fused(sources, rng.choice(_RANDOM_METADATA))


def _choose_random_provenance_op(rng: random.Random) -> dict[str, Any]:
    kind = rng.choices(
        ["fuse", "decode", "equals", "fused_new", "named_new"], weights=[6, 3, 2, 1, 1]
    )[0]
    if kind == "fuse":
        inputs = [
            _choose_random_tree(rng, _RANDOM_TREE_DEPTH)
            for _ in range(rng.randrange(0, 5))
        ]
        return _fuse_op(inputs, rng.choice(_RANDOM_METADATA))
    if kind == "decode":
        return _decode_op("provenance", _choose_random_tree(rng, _RANDOM_TREE_DEPTH))
    if kind == "equals":
        left = _choose_random_tree(rng, 2)
        right = (
            copy.deepcopy(left)
            if rng.random() < _RANDOM_EQUAL_PAIR_PROBABILITY
            else _choose_random_tree(rng, 2)
        )
        return {"op": "equals", "left": left, "right": right}
    if kind == "fused_new":
        sources = [_choose_random_tree(rng, 2) for _ in range(rng.randrange(0, 4))]
        return {
            "op": "fused_new",
            "sources": sources,
            "metadata": rng.choice(_RANDOM_METADATA),
        }
    return {
        "op": "named_new",
        "name": rng.choice(("", *_RANDOM_NAMES)),
        "child": _choose_random_tree(rng, 2),
    }


def _choose_random_diagnostic_spec(rng: random.Random) -> dict[str, Any]:
    return _diagnostic_spec(
        rng.choice(_LEVELS).value,
        rng.choice(_RANDOM_TEXTS),
        rng.choice(_RANDOM_SOURCES),
        rng.choice((None, *_RANDOM_TEXTS)),
        rng.choice(tuple(_NOTE_KIND_DEFAULTS)),
    )


def _choose_random_diagnostic_op(rng: random.Random) -> dict[str, Any]:
    if rng.random() < _RANDOM_REPORT_PROBABILITY:
        specs = [
            _choose_random_diagnostic_spec(rng) for _ in range(rng.randrange(0, 7))
        ]
        return _report_op(*specs)
    return {
        "op": "note_new",
        "message": rng.choice(_RANDOM_TEXTS),
        "kind_slot": rng.choice(tuple(_NOTE_KIND_DEFAULTS)),
    }


def _build_random_script(
    rng: random.Random, index: int, max_ops: int
) -> tuple[str, str, list[dict[str, Any]]]:
    kind = _PROVENANCE_KIND if index % 2 == 0 else _DIAGNOSTIC_KIND
    choose = (
        _choose_random_provenance_op
        if kind == _PROVENANCE_KIND
        else _choose_random_diagnostic_op
    )
    ops = [choose(rng) for _ in range(rng.randint(1, max_ops))]
    return f"random_{kind}_{index:03d}", kind, ops


# =============================================================================
# Document assembly
# =============================================================================


def _parse_arguments(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    add_corpus_arguments(
        parser,
        seed=_RANDOM_SEED,
        random_count=_RANDOM_SCRIPT_COUNT,
        max_ops=_RANDOM_MAX_OPS,
        default_output=default_output,
    )
    return parser.parse_args()


def main() -> None:
    """Run every script through the oracle and write the golden document."""
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[3]
    arguments = _parse_arguments(
        script_path.parent / "provenance_diagnostic_cases.json"
    )

    provenance_scripts = [
        *_list_position_and_span_scripts(),
        *_list_file_scripts(),
        *_list_variant_scripts(),
        *_list_fuse_scripts(),
        *_list_decode_scripts(),
    ]
    diagnostic_scripts = [*_list_note_scripts(), *_list_report_scripts()]

    cases = [_run_provenance_script(name, ops) for name, ops in provenance_scripts]
    cases += [_run_diagnostic_script(name, ops) for name, ops in diagnostic_scripts]

    rng = random.Random(arguments.seed)
    for index in range(arguments.random_count):
        name, kind, ops = _build_random_script(rng, index, arguments.max_ops)
        if kind == _PROVENANCE_KIND:
            cases.append(_run_provenance_script(name, ops))
        else:
            cases.append(_run_diagnostic_script(name, ops))

    document = {
        "provenance": build_provenance(repository_root, GENERATOR_COMMAND),
        "default_note_kind_slots": list(_NOTE_KIND_DEFAULTS),
        "cases": cases,
    }
    write_document(arguments.output, document)


if __name__ == "__main__":
    main()
