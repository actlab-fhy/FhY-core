"""Shared fixtures for the benchmarks.

The benchmarks use the public API only, so the same benchmark measures a
concept before and after it switches to the Rust implementation. The
``benchmark`` fixture comes from ``pytest-benchmark``; :class:`Benchmark`
types it without importing the plugin, so the lint and type checks do not
need the ``bench`` dependency group.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, Protocol, TypeVar

import pytest

import fhy_core
from fhy_core.diagnostic import (
    REMARK_NOTE_KIND,
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    Analysis,
    AnalysisManager,
    CompilerPass,
    PassManager,
    register_pass,
)
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
from fhy_core.traits import FrozenMixin
from fhy_core.utils.override import override
from fhy_core.value_domain import ValueDomain

__all__ = [
    "REPORT_DIAGNOSTIC_COUNT",
    "Benchmark",
    "Box",
    "BoxValueAnalysis",
    "IdentityPass",
    "build_diagnostics",
]

_P = ParamSpec("_P")
_T = TypeVar("_T")

# How many diagnostics the benchmarked validation reports hold.
REPORT_DIAGNOSTIC_COUNT = 100
# How many identifiers the benchmarked identifier-keyed dict holds.
_IDENTIFIER_TABLE_SIZE = 100
# How many domains the benchmarked value-domain chain holds, root included.
_VALUE_DOMAIN_CHAIN_LENGTH = 4


class Benchmark(Protocol):
    """The part of ``pytest-benchmark``'s ``benchmark`` fixture used here."""

    def __call__(
        self, function: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Time repeated calls of ``function`` and return one call's result."""


def pytest_benchmark_update_machine_info(
    config: pytest.Config, machine_info: dict[str, Any]
) -> None:
    """Record the backend the run measured in the saved benchmark data."""
    _ = config
    machine_info["fhy_core_backend"] = (
        "rust" if fhy_core.RUST_BACKEND_SELECTED else "python"
    )


# ---------------------------------------------------------------------------
# Identifier
# ---------------------------------------------------------------------------


@pytest.fixture()
def identifier() -> Identifier:
    """Return a fresh identifier."""
    return Identifier("x")


@pytest.fixture()
def identifier_copy(identifier: Identifier) -> Identifier:
    """Return a distinct identifier object equal to ``identifier``."""
    return Identifier.deserialize_from_dict(identifier.serialize_to_dict())


@pytest.fixture()
def identifier_table(identifier: Identifier) -> dict[Identifier, int]:
    """Return an identifier-keyed dict that holds ``identifier``."""
    table = {Identifier(f"x{index}"): index for index in range(_IDENTIFIER_TABLE_SIZE)}
    table[identifier] = _IDENTIFIER_TABLE_SIZE
    return table


# ---------------------------------------------------------------------------
# Interned tags
# ---------------------------------------------------------------------------


@pytest.fixture()
def value_domain_chain() -> list[ValueDomain]:
    """Return a fresh chain of value domains, root first."""
    chain = [ValueDomain(Identifier("domain0"), "Root domain.")]
    for index in range(1, _VALUE_DOMAIN_CHAIN_LENGTH):
        chain.append(
            ValueDomain(Identifier(f"domain{index}"), "Subdomain.", parent=chain[-1])
        )
    return chain


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def build_diagnostics(count: int) -> tuple[Diagnostic, ...]:
    """Return ``count`` diagnostics that cycle through the levels."""
    levels = (DiagnosticLevel.ERROR, DiagnosticLevel.WARNING, DiagnosticLevel.INFO)
    return tuple(
        Diagnostic(
            level=levels[index % len(levels)],
            message=Note(f"message {index}", REMARK_NOTE_KIND),
            source="benchmarks.source",
            detail=f"detail {index}" if index % 2 else None,
        )
        for index in range(count)
    )


@pytest.fixture()
def note() -> Note:
    """Return a note with a kind."""
    return Note("value is out of range", REMARK_NOTE_KIND)


@pytest.fixture()
def diagnostic(note: Note) -> Diagnostic:
    """Return an error diagnostic with detail."""
    return Diagnostic(
        level=DiagnosticLevel.ERROR,
        message=note,
        source="benchmarks.source",
        detail="expected a value below 10",
    )


@pytest.fixture()
def report_diagnostics() -> tuple[Diagnostic, ...]:
    """Return the diagnostics of a benchmarked validation report."""
    return build_diagnostics(REPORT_DIAGNOSTIC_COUNT)


@pytest.fixture()
def report(report_diagnostics: tuple[Diagnostic, ...]) -> ValidationReport[Any]:
    """Return a validation report of ``REPORT_DIAGNOSTIC_COUNT`` diagnostics."""
    return ValidationReport(diagnostics=report_diagnostics)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@pytest.fixture()
def source_path() -> Path:
    """Return the path of a source file."""
    return Path("src/kernel.fhy")


@pytest.fixture()
def span() -> Span:
    """Return a span with offsets and positions."""
    return Span(12, 40, Position(2, 5), Position(3, 9))


def _build_provenances(source_path: Path, span: Span) -> dict[str, Provenance]:
    file_provenance = FileProvenance(source_path, span)
    other_file_provenance = FileProvenance(source_path, Span(50, 60))
    return {
        "unknown": UnknownProvenance(),
        "file": file_provenance,
        "named": NamedProvenance("matmul", file_provenance),
        "call_site": CallSiteProvenance(file_provenance, other_file_provenance),
        "fused": FusedProvenance((file_provenance, other_file_provenance), "inline"),
    }


@pytest.fixture()
def provenances(source_path: Path, span: Span) -> dict[str, Provenance]:
    """Return one provenance of each kind, keyed by kind."""
    return _build_provenances(source_path, span)


@pytest.fixture()
def provenance_copies(source_path: Path, span: Span) -> dict[str, Provenance]:
    """Return distinct provenance objects equal to ``provenances``."""
    return _build_provenances(source_path, span)


# ---------------------------------------------------------------------------
# Pass infrastructure
# ---------------------------------------------------------------------------


@dataclass
class Box(FrozenMixin):
    """Frozen single-value IR, so the analysis manager caches its analyses."""

    value: int

    def __post_init__(self) -> None:
        self.freeze()


class BoxValueAnalysis(Analysis[Box, int]):
    """Analysis that reads the box's value."""

    @override
    def run(self, ir: Box) -> int:
        return ir.value


@register_pass("benchmarks.identity", "Return the box unchanged.")
class IdentityPass(CompilerPass[Box, Box]):
    """Pass that returns its input unchanged."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        return ir


@register_pass("benchmarks.increment", "Return a box holding the next value.")
class IncrementPass(CompilerPass[Box, Box]):
    """Pass that returns a new box holding the next value."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        return Box(ir.value + 1)


@register_pass("benchmarks.read_analysis", "Read an analysis of the box.")
class ReadAnalysisPass(CompilerPass[Box, Box]):
    """Pass that requests an analysis and returns its input unchanged."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        self.get_analysis(BoxValueAnalysis, ir)
        return ir


@pytest.fixture()
def box() -> Box:
    """Return a small IR."""
    return Box(0)


@pytest.fixture()
def pass_manager() -> PassManager[Box]:
    """Return a pipeline of five passes that change, keep, and analyze the IR."""
    manager = PassManager[Box]()
    for compiler_pass in (
        IncrementPass(),
        ReadAnalysisPass(),
        IdentityPass(),
        ReadAnalysisPass(),
        IncrementPass(),
    ):
        manager.add_pass(compiler_pass)
    return manager


@pytest.fixture()
def warm_analysis_manager(box: Box) -> AnalysisManager[Box]:
    """Return an analysis manager that has cached ``BoxValueAnalysis`` of ``box``."""
    manager = AnalysisManager[Box]()
    manager.get(BoxValueAnalysis, box)
    return manager
