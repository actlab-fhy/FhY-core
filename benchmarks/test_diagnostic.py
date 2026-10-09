"""Benchmarks of `Note`, `Diagnostic` and `ValidationReport`'s hot paths."""

import operator
from typing import Any

import pytest

from fhy_core.diagnostic import (
    REMARK_NOTE_KIND,
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationReport,
)

from .conftest import REPORT_DIAGNOSTIC_COUNT, Benchmark, build_diagnostics

pytestmark = pytest.mark.benchmark(group="diagnostic")


def _build_report(count: int) -> ValidationReport[Any]:
    return ValidationReport(diagnostics=build_diagnostics(count))


def test_note_construction(benchmark: Benchmark) -> None:
    """Benchmark constructing a note with a kind."""
    benchmark(Note, "value is out of range", REMARK_NOTE_KIND)


def test_note_eq(benchmark: Benchmark, note: Note) -> None:
    """Benchmark comparing two distinct, equal notes."""
    other = Note(note.message, note.kind)
    assert benchmark(operator.eq, note, other)


def test_note_str(benchmark: Benchmark, note: Note) -> None:
    """Benchmark rendering a note."""
    benchmark(str, note)


def test_note_hash(benchmark: Benchmark, note: Note) -> None:
    """Benchmark hashing a note."""
    benchmark(hash, note)


def test_note_attribute_access(benchmark: Benchmark, note: Note) -> None:
    """Benchmark reading a note's two fields."""
    benchmark(operator.attrgetter("message", "kind"), note)


def test_diagnostic_construction(benchmark: Benchmark, note: Note) -> None:
    """Benchmark constructing a diagnostic with detail."""
    benchmark(
        Diagnostic,
        level=DiagnosticLevel.ERROR,
        message=note,
        source="benchmarks.source",
        detail="expected a value below 10",
    )


def test_diagnostic_eq(benchmark: Benchmark, diagnostic: Diagnostic) -> None:
    """Benchmark comparing two distinct, equal diagnostics."""
    other = Diagnostic(
        level=diagnostic.level,
        message=Note(diagnostic.message.message, diagnostic.message.kind),
        source=diagnostic.source,
        detail=diagnostic.detail,
    )
    assert benchmark(operator.eq, diagnostic, other)


def test_diagnostic_hash(benchmark: Benchmark, diagnostic: Diagnostic) -> None:
    """Benchmark hashing a diagnostic."""
    benchmark(hash, diagnostic)


def test_diagnostic_attribute_access(
    benchmark: Benchmark, diagnostic: Diagnostic
) -> None:
    """Benchmark reading a diagnostic's four fields."""
    benchmark(operator.attrgetter("level", "message", "source", "detail"), diagnostic)


def test_validation_report_construction(
    benchmark: Benchmark, report_diagnostics: tuple[Diagnostic, ...]
) -> None:
    """Benchmark constructing a report from 100 existing diagnostics."""
    benchmark(ValidationReport, diagnostics=report_diagnostics)


def test_validation_report_build_of_100_diagnostics(benchmark: Benchmark) -> None:
    """Benchmark building 100 notes and diagnostics and a report of them."""
    report = benchmark(_build_report, REPORT_DIAGNOSTIC_COUNT)
    assert len(report.diagnostics) == REPORT_DIAGNOSTIC_COUNT


def test_validation_report_eq(
    benchmark: Benchmark, report: ValidationReport[Any]
) -> None:
    """Benchmark comparing two equal reports of 100 distinct diagnostics each."""
    other = _build_report(REPORT_DIAGNOSTIC_COUNT)
    assert benchmark(operator.eq, report, other)


def test_validation_report_format(
    benchmark: Benchmark, report: ValidationReport[Any]
) -> None:
    """Benchmark rendering a report of 100 diagnostics."""
    benchmark(report.format)


def test_validation_report_errors(
    benchmark: Benchmark, report: ValidationReport[Any]
) -> None:
    """Benchmark selecting the errors of a report of 100 diagnostics."""
    benchmark(report.errors)


def test_validation_report_has_errors(
    benchmark: Benchmark, report: ValidationReport[Any]
) -> None:
    """Benchmark checking a report of 100 diagnostics for errors."""
    benchmark(report.has_errors)
