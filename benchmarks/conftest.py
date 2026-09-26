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
from typing import Any, ClassVar, ParamSpec, Protocol, TypeVar

import pytest

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
    CompilerPass,
    PassManager,
    PreservedAnalyses,
    register_pass,
    register_verification,
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
from fhy_core.traits import FrozenMixin, VerifiableMixin
from fhy_core.utils.override import override
from fhy_core.value_domain import ValueDomain

__all__ = [
    "PIPELINE_INCREMENT",
    "REPORT_DIAGNOSTIC_COUNT",
    "SATURATION_VALUE",
    "Benchmark",
    "Box",
    "BoxValueAnalysis",
    "CopyReadingPass",
    "CountingBoxValueAnalysis",
    "EveryHookPass",
    "FailingPass",
    "IdentityPass",
    "IncrementPass",
    "ReadAnalysisPass",
    "ReportingPass",
    "ReportingValidator",
    "SaturatingIncrementPass",
    "SkippedPass",
    "TwiceVerifiedBox",
    "VerifiedBox",
    "build_diagnostics",
    "build_pipeline",
]

_P = ParamSpec("_P")
_T = TypeVar("_T")

# How many diagnostics the benchmarked validation reports hold.
REPORT_DIAGNOSTIC_COUNT = 100
# How many identifiers the benchmarked identifier-keyed dict holds.
_IDENTIFIER_TABLE_SIZE = 100
# How many domains the benchmarked value-domain chain holds, root included.
_VALUE_DOMAIN_CHAIN_LENGTH = 4
# The value the benchmarked pipelines' two increment passes per five add.
PIPELINE_INCREMENT = 2
# The value at which `SaturatingIncrementPass` stops changing a box.
SATURATION_VALUE = 9
# The largest value `BoundedBoxVerifier` accepts.
_BOX_BOUND = 1_000_000


class Benchmark(Protocol):
    """The part of ``pytest-benchmark``'s ``benchmark`` fixture used here."""

    def __call__(
        self, function: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Time repeated calls of ``function`` and return one call's result."""

    def pedantic(
        self,
        target: Callable[..., _T],
        *,
        setup: Callable[[], object] | None = None,
        rounds: int = 1,
        iterations: int = 1,
        warmup_rounds: int = 0,
    ) -> _T:
        """Time ``rounds`` calls of ``target``, each after a call of ``setup``."""


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


@dataclass
class VerifiedBox(Box):
    """A box with one registered verification pass."""


@dataclass
class TwiceVerifiedBox(VerifiedBox):
    """A box with two registered verification passes: its own and its base's."""


class BoxValueAnalysis(Analysis[Box, int]):
    """Analysis that reads the box's value."""

    @override
    def run(self, ir: Box) -> int:
        return ir.value


class CountingBoxValueAnalysis(Analysis[Box, int]):
    """Analysis that reads the box's value and counts its runs."""

    runs: ClassVar[int] = 0

    @override
    def run(self, ir: Box) -> int:
        CountingBoxValueAnalysis.runs += 1
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
    """Pass that returns a new box of the input's type holding the next value."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        return type(ir)(ir.value + 1)


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


class CopyReadingPass(CompilerPass[Box, Box]):
    """Pass that reads a counting analysis and returns an equal, new box.

    The copy compares equal to its input, so the run is unchanged and
    preserves every analysis, which the pipeline carries over to the copy.
    """

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        self.get_analysis(CountingBoxValueAnalysis, ir)
        return type(ir)(ir.value)


class EveryHookPass(CompilerPass[Box, Box]):
    """Identity pass that overrides all seven lifecycle hooks in Python."""

    @override
    def validate_input(self, ir: Box) -> None:
        _ = ir

    @override
    def should_run(self, ir: Box) -> bool:
        _ = ir
        return True

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        return ir

    @override
    def validate_output(self, input_ir: Box, output: Box) -> None:
        _ = (input_ir, output)

    @override
    def did_change(self, input_ir: Box, output: Box) -> bool:
        return output is not input_ir

    @override
    def get_preserved_analyses(
        self, input_ir: Box, output: Box, *, changed: bool
    ) -> PreservedAnalyses:
        _ = (input_ir, output)
        return PreservedAnalyses.none() if changed else PreservedAnalyses.all()


class SkippedPass(CompilerPass[Box, Box]):
    """Pass that always skips, returning its input as the no-op output."""

    @override
    def should_run(self, ir: Box) -> bool:
        _ = ir
        return False

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        raise AssertionError("a skipped pass never runs")


class FailingPass(CompilerPass[Box, Box]):
    """Pass whose run always raises ``ValueError``."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        raise ValueError("boom")


class ReportingPass(CompilerPass[Box, Box]):
    """Identity pass that reports `count` diagnostics, cycling the levels."""

    _count: int

    def __init__(self, count: int) -> None:
        super().__init__()
        self._count = count

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        levels = (DiagnosticLevel.ERROR, DiagnosticLevel.WARNING, DiagnosticLevel.INFO)
        for index in range(self._count):
            self.report(levels[index % len(levels)], f"message {index}")
        return ir


class SaturatingIncrementPass(CompilerPass[Box, Box]):
    """Pass that increments a box until it holds `SATURATION_VALUE`."""

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        if ir.value >= SATURATION_VALUE:
            return ir
        return type(ir)(ir.value + 1)


class ReportingValidator(CompilerPass[Box, None]):
    """Validator pass that reports one warning when `is_reporting` is set."""

    _is_reporting: bool

    def __init__(self, *, is_reporting: bool) -> None:
        super().__init__()
        self._is_reporting = is_reporting

    @override
    def get_noop_output(self, ir: Box) -> None:
        _ = ir

    @override
    def run_pass(self, ir: Box) -> None:
        if self._is_reporting:
            self.report(DiagnosticLevel.WARNING, f"box holds {ir.value}")


@register_verification(
    VerifiedBox, "benchmarks.verify_non_negative", "Reject a negative box."
)
class NonNegativeBoxVerifier(CompilerPass[Box, None]):
    """Verification pass that reports an error for a negative box."""

    @override
    def get_noop_output(self, ir: Box) -> None:
        _ = ir

    @override
    def run_pass(self, ir: Box) -> None:
        if ir.value < 0:
            self.report(DiagnosticLevel.ERROR, "the box is negative")


@register_verification(
    TwiceVerifiedBox, "benchmarks.verify_bounded", "Reject an unbounded box."
)
class BoundedBoxVerifier(CompilerPass[Box, None]):
    """Verification pass that reports an error for a box above `_BOX_BOUND`."""

    @override
    def get_noop_output(self, ir: Box) -> None:
        _ = ir

    @override
    def run_pass(self, ir: Box) -> None:
        if ir.value > _BOX_BOUND:
            self.report(DiagnosticLevel.ERROR, "the box is too large")


class VerifiedNode(VerifiableMixin):
    """A `VerifiableMixin` whose default `verify` runs one registered pass."""

    value: int

    def __init__(self, value: int) -> None:
        self.value = value


@register_verification(
    VerifiedNode, "benchmarks.verify_node", "Reject a negative verified node."
)
class NonNegativeNodeVerifier(CompilerPass[VerifiedNode, None]):
    """Verification pass that reports an error for a negative node."""

    @override
    def get_noop_output(self, ir: VerifiedNode) -> None:
        _ = ir

    @override
    def run_pass(self, ir: VerifiedNode) -> None:
        if ir.value < 0:
            self.report(DiagnosticLevel.ERROR, "the node is negative")


def build_pipeline(repetitions: int) -> PassManager[Box]:
    """Return `repetitions` copies of five passes that change, keep and analyze."""
    manager = PassManager[Box]()
    for _ in range(repetitions):
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
def box() -> Box:
    """Return a small IR."""
    return Box(0)


@pytest.fixture()
def pass_manager() -> PassManager[Box]:
    """Return a pipeline of five passes that change, keep, and analyze the IR."""
    return build_pipeline(1)
