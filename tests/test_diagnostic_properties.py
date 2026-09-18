"""Hypothesis property tests for `ValidationReport`.

Draws reports with a random mix of `DiagnosticLevel`s and checks three
laws, each against an oracle computed independently of the method under
test: `has_errors()` equals whether any diagnostic is ERROR-level;
`errors()`/`warnings()`/`infos()` partition the diagnostics by level while
preserving order; and `raise_if_failed()` raises `ValidationFailedError`
exactly when `has_errors()` is true, with the raised error carrying the
same report instance.
"""

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationFailedError,
    ValidationReport,
)

pytestmark = pytest.mark.property


@st.composite
def draw_diagnostic(draw: st.DrawFn) -> Diagnostic:
    """Draw one Diagnostic with a random level, message, source, and detail."""
    level = draw(st.sampled_from(list(DiagnosticLevel)))
    message = draw(st.text(min_size=0, max_size=20))
    source = draw(st.text(min_size=1, max_size=10))
    detail = draw(st.one_of(st.none(), st.text(min_size=0, max_size=20)))
    return Diagnostic(level=level, message=Note(message), source=source, detail=detail)


@st.composite
def draw_validation_report(draw: st.DrawFn) -> ValidationReport[Any]:
    """Draw a ValidationReport with 0 to 8 diagnostics of random levels."""
    diagnostics = draw(st.lists(draw_diagnostic(), min_size=0, max_size=8))
    return ValidationReport(diagnostics=tuple(diagnostics))


@given(report=draw_validation_report())
def test_has_errors_matches_any_error_level_diagnostic(
    report: ValidationReport[Any],
) -> None:
    """Test has_errors() equals whether any diagnostic is ERROR-level.

    Oracle: any(...) over the report's own diagnostics tuple, independent
    of has_errors's internal implementation.
    """
    expected = any(d.level == DiagnosticLevel.ERROR for d in report.diagnostics)
    assert report.has_errors() == expected


@given(report=draw_validation_report())
def test_errors_warnings_infos_partition_diagnostics_preserving_order(
    report: ValidationReport[Any],
) -> None:
    """Test errors()/warnings()/infos() partition diagnostics and preserve order.

    Oracle: filtering report.diagnostics by level with a plain generator
    expression, independent of the methods under test.
    """
    errors = report.errors()
    warnings = report.warnings()
    infos = report.infos()

    assert errors == tuple(
        d for d in report.diagnostics if d.level == DiagnosticLevel.ERROR
    )
    assert warnings == tuple(
        d for d in report.diagnostics if d.level == DiagnosticLevel.WARNING
    )
    assert infos == tuple(
        d for d in report.diagnostics if d.level == DiagnosticLevel.INFO
    )
    assert len(errors) + len(warnings) + len(infos) == len(report.diagnostics)


@given(report=draw_validation_report())
def test_raise_if_failed_raises_iff_has_errors_and_carries_report(
    report: ValidationReport[Any],
) -> None:
    """Test raise_if_failed raises ValidationFailedError exactly when has_errors().

    Also checks the raised error's `.report` is the same report instance.

    Oracle: has_errors(), independently computed above, gates whether an
    exception is expected; the identity check follows
    ValidationFailedError's own documented contract.
    """
    if report.has_errors():
        with pytest.raises(ValidationFailedError) as exc_info:
            report.raise_if_failed()
        assert exc_info.value.report is report
    else:
        report.raise_if_failed()
