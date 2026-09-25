"""Benchmarks of `Provenance`'s hot paths."""

import operator
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

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="provenance")

_PROVENANCE_KINDS = pytest.mark.parametrize(
    "kind", ["unknown", "file", "named", "call_site", "fused"]
)


def _build_span() -> Span:
    return Span(12, 40, Position(2, 5), Position(3, 9))


def test_span_construction(benchmark: Benchmark) -> None:
    """Benchmark constructing a span with offsets and positions."""
    benchmark(_build_span)


def test_unknown_provenance_construction(benchmark: Benchmark) -> None:
    """Benchmark constructing an unknown provenance."""
    benchmark(UnknownProvenance)


def test_file_provenance_construction(
    benchmark: Benchmark, source_path: Path, span: Span
) -> None:
    """Benchmark constructing a file provenance from a path and a span."""
    benchmark(FileProvenance, source_path, span)


def test_named_provenance_construction(
    benchmark: Benchmark, provenances: dict[str, Provenance]
) -> None:
    """Benchmark constructing a named provenance around a file provenance."""
    benchmark(NamedProvenance, "matmul", provenances["file"])


def test_call_site_provenance_construction(
    benchmark: Benchmark, provenances: dict[str, Provenance]
) -> None:
    """Benchmark constructing a call-site provenance."""
    file_provenance = provenances["file"]
    benchmark(CallSiteProvenance, file_provenance, file_provenance)


def test_fused_provenance_construction(
    benchmark: Benchmark, provenances: dict[str, Provenance]
) -> None:
    """Benchmark constructing a fused provenance of two sources."""
    sources = (provenances["file"], provenances["named"])
    benchmark(FusedProvenance, sources, "inline")


@_PROVENANCE_KINDS
def test_provenance_eq(
    benchmark: Benchmark,
    provenances: dict[str, Provenance],
    provenance_copies: dict[str, Provenance],
    kind: str,
) -> None:
    """Benchmark comparing two distinct, equal provenances of one kind."""
    assert benchmark(operator.eq, provenances[kind], provenance_copies[kind])


@_PROVENANCE_KINDS
def test_provenance_hash(
    benchmark: Benchmark, provenances: dict[str, Provenance], kind: str
) -> None:
    """Benchmark hashing a provenance of one kind."""
    benchmark(hash, provenances[kind])


def test_provenance_fuse_of_two(
    benchmark: Benchmark, provenances: dict[str, Provenance]
) -> None:
    """Benchmark fusing two provenances, which builds a fused provenance."""
    fused = benchmark(Provenance.fuse, provenances["file"], provenances["named"])
    assert isinstance(fused, FusedProvenance)


def test_provenance_fuse_with_reductions(
    benchmark: Benchmark, provenances: dict[str, Provenance]
) -> None:
    """Benchmark fusing that drops unknowns and flattens an unlabeled fusion."""
    unlabeled = FusedProvenance((provenances["file"], provenances["call_site"]))
    fused = benchmark(
        Provenance.fuse,
        provenances["unknown"],
        unlabeled,
        provenances["named"],
        UnknownProvenance(),
    )
    assert isinstance(fused, FusedProvenance)


_PROVENANCE_FIELDS = {
    "file": ("file_path", "span"),
    "named": ("name", "child"),
    "call_site": ("callee", "caller"),
    "fused": ("sources", "metadata"),
}


def test_position_construction(benchmark: Benchmark) -> None:
    """Benchmark constructing a position."""
    benchmark(Position, 2, 5)


def test_position_attribute_access(benchmark: Benchmark, span: Span) -> None:
    """Benchmark reading a position's two fields."""
    benchmark(operator.attrgetter("line", "column"), span.start_position)


def test_position_lt(benchmark: Benchmark) -> None:
    """Benchmark ordering two positions on the same line."""
    assert benchmark(operator.lt, Position(2, 5), Position(2, 9))


def test_span_attribute_access(benchmark: Benchmark, span: Span) -> None:
    """Benchmark reading a span's four fields."""
    benchmark(
        operator.attrgetter(
            "start_offset", "end_offset", "start_position", "end_position"
        ),
        span,
    )


def test_span_str(benchmark: Benchmark, span: Span) -> None:
    """Benchmark rendering a span with positions."""
    benchmark(str, span)


@pytest.mark.parametrize("kind", list(_PROVENANCE_FIELDS))
def test_provenance_attribute_access(
    benchmark: Benchmark, provenances: dict[str, Provenance], kind: str
) -> None:
    """Benchmark reading the two fields of a provenance of one kind."""
    benchmark(operator.attrgetter(*_PROVENANCE_FIELDS[kind]), provenances[kind])


@_PROVENANCE_KINDS
def test_provenance_str(
    benchmark: Benchmark, provenances: dict[str, Provenance], kind: str
) -> None:
    """Benchmark rendering a provenance of one kind."""
    benchmark(str, provenances[kind])


@_PROVENANCE_KINDS
def test_provenance_repr(
    benchmark: Benchmark, provenances: dict[str, Provenance], kind: str
) -> None:
    """Benchmark the repr of a provenance of one kind."""
    benchmark(repr, provenances[kind])


@_PROVENANCE_KINDS
def test_provenance_dict_round_trip(
    benchmark: Benchmark, provenances: dict[str, Provenance], kind: str
) -> None:
    """Benchmark serializing a provenance of one kind to a dict and back."""
    provenance = provenances[kind]

    def round_trip() -> Provenance:
        return Provenance.deserialize_from_dict(provenance.serialize_to_dict())

    assert benchmark(round_trip) == provenance
