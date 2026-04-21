"""Tests the validation pipeline infrastructure."""

from dataclasses import dataclass
from typing import Any

import pytest
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    AnalysisVisitablePass,
    CompilerPass,
    DiagnosticLevel,
    PassDiagnostic,
    PassExecutionError,
    PassRunRecord,
    PassValidationError,
    ValidationFailedError,
    ValidationManager,
    ValidationReport,
    register_pass,
)
from fhy_core.provenance import Note
from fhy_core.trait import FrozenMixin, PartialEqual, Visitable


@dataclass
class ValueBox(FrozenMixin, Visitable):
    """Simple immutable IR node."""

    value: int

    def __post_init__(self) -> None:
        self.freeze()


# ---------------------------------------------------------------------------
# Helpers for building test validators.
# ---------------------------------------------------------------------------


def _single_error_validator(
    pass_name: str, message: str, *, detail: str | None = None
) -> CompilerPass[ValueBox, None]:
    """Build a validator that emits one ERROR diagnostic and returns."""

    @register_pass(pass_name, f"Emits a single ERROR: {message}")
    class _Validator(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, message, detail=detail)

    return _Validator()


def _single_warning_validator(
    pass_name: str, message: str
) -> CompilerPass[ValueBox, None]:
    @register_pass(pass_name, f"Emits a single WARNING: {message}")
    class _Validator(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.WARNING, message)

    return _Validator()


def _single_info_validator(
    pass_name: str, message: str
) -> CompilerPass[ValueBox, None]:
    @register_pass(pass_name, f"Emits a single INFO: {message}")
    class _Validator(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.INFO, message)

    return _Validator()


def _clean_validator(pass_name: str) -> CompilerPass[ValueBox, None]:
    @register_pass(pass_name, "Emits nothing; always succeeds.")
    class _Validator(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node

    return _Validator()


# ---------------------------------------------------------------------------
# ValidationReport: helpers and raise_if_failed.
# ---------------------------------------------------------------------------


def test_empty_report_has_no_errors_and_formats_placeholder_text() -> None:
    """Test that an empty report has no errors and formats a placeholder string."""
    report = ValidationReport()
    assert report.diagnostics == ()
    assert report.records == ()
    assert report.has_errors() is False
    assert report.errors() == ()
    assert report.warnings() == ()
    assert report.infos() == ()
    assert report.format() == "No validation diagnostics."


def test_report_filters_diagnostics_by_level() -> None:
    """Test that errors(), warnings(), and infos() partition diagnostics by level."""
    diagnostics = (
        PassDiagnostic(DiagnosticLevel.ERROR, Note("bad"), "v1"),
        PassDiagnostic(DiagnosticLevel.WARNING, Note("meh"), "v2"),
        PassDiagnostic(DiagnosticLevel.INFO, Note("fyi"), "v3"),
        PassDiagnostic(DiagnosticLevel.ERROR, Note("worse"), "v4"),
    )
    report = ValidationReport(diagnostics=diagnostics)

    assert report.has_errors() is True
    assert tuple(d.pass_name for d in report.errors()) == ("v1", "v4")
    assert tuple(d.pass_name for d in report.warnings()) == ("v2",)
    assert tuple(d.pass_name for d in report.infos()) == ("v3",)


def test_report_format_renders_level_pass_and_detail() -> None:
    """Test that format() includes level, pass name, message, and indented detail."""
    report = ValidationReport(
        diagnostics=(
            PassDiagnostic(
                DiagnosticLevel.ERROR,
                Note("missing return"),
                "shape.check",
                detail="function foo() has no return statement",
            ),
            PassDiagnostic(DiagnosticLevel.WARNING, Note("unused"), "scope.check"),
        )
    )

    rendered = report.format()

    assert "[ERROR] shape.check: missing return" in rendered
    assert "    detail: function foo() has no return statement" in rendered
    assert "[WARNING] scope.check: unused" in rendered


def test_raise_if_failed_is_noop_when_only_warnings_present() -> None:
    """Test that raise_if_failed does not raise when no ERROR diagnostics exist."""
    report = ValidationReport(
        diagnostics=(PassDiagnostic(DiagnosticLevel.WARNING, Note("ok-ish"), "v"),)
    )
    report.raise_if_failed()


def test_raise_if_failed_is_noop_when_only_infos_present() -> None:
    """Test that raise_if_failed does not raise on INFO-only reports."""
    report = ValidationReport(
        diagnostics=(PassDiagnostic(DiagnosticLevel.INFO, Note("fyi"), "v"),)
    )
    report.raise_if_failed()


def test_raise_if_failed_raises_validation_failed_with_formatted_message() -> None:
    """Test that raise_if_failed raises ValidationFailedError on errors."""
    report = ValidationReport(
        diagnostics=(PassDiagnostic(DiagnosticLevel.ERROR, Note("boom"), "v.explode"),)
    )

    with pytest.raises(ValidationFailedError) as excinfo:
        report.raise_if_failed()

    assert excinfo.value.report is report
    assert "[ERROR] v.explode: boom" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ValidationManager: orchestration semantics.
# ---------------------------------------------------------------------------


def test_validation_manager_runs_every_validator_even_after_errors() -> None:
    """Test that all registered validators run even when earlier ones report errors."""
    first = _single_error_validator("tests.vm.first_error", "first-error-msg")
    second = _single_error_validator("tests.vm.second_error", "second-error-msg")
    third = _clean_validator("tests.vm.third_clean")

    manager = ValidationManager[ValueBox]()
    manager.add(first)
    manager.add(second)
    manager.add(third)

    report = manager.validate(ValueBox(0))

    assert len(report.records) == 3
    assert [r.pass_name for r in report.records] == [
        "tests.vm.first_error",
        "tests.vm.second_error",
        "tests.vm.third_clean",
    ]
    assert tuple(d.message_text for d in report.errors()) == (
        "first-error-msg",
        "second-error-msg",
    )


def test_validation_manager_aggregates_diagnostics_in_pipeline_order() -> None:
    """Test that diagnostics appear in validator registration order."""
    warning = _single_warning_validator("tests.vm.warn", "warn-msg")
    error = _single_error_validator("tests.vm.err", "err-msg")

    manager = ValidationManager[ValueBox]()
    manager.add(warning)
    manager.add(error)
    report = manager.validate(ValueBox(42))

    assert [d.message_text for d in report.diagnostics] == ["warn-msg", "err-msg"]
    assert [d.level for d in report.diagnostics] == [
        DiagnosticLevel.WARNING,
        DiagnosticLevel.ERROR,
    ]


def test_validation_manager_returns_clean_report_when_all_validators_clean() -> None:
    """Test that a clean pipeline produces a report with no diagnostics."""
    manager = ValidationManager[ValueBox]()
    manager.add(_clean_validator("tests.vm.clean_a"))
    manager.add(_clean_validator("tests.vm.clean_b"))

    report = manager.validate(ValueBox(0))

    assert report.has_errors() is False
    assert report.diagnostics == ()
    assert [r.pass_name for r in report.records] == [
        "tests.vm.clean_a",
        "tests.vm.clean_b",
    ]


def test_validation_manager_wraps_validator_crash_as_error_diagnostic() -> None:
    """Test that unexpected crashes become ERROR diagnostics, not halts."""

    @register_pass("tests.vm.crasher", "Raises a ValueError in visit.")
    class _Crasher(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            raise ValueError("internal boom")

    follow_up = _single_error_validator("tests.vm.after_crasher", "still-runs")

    manager = ValidationManager[ValueBox]()
    manager.add(_Crasher())
    manager.add(follow_up)
    report = manager.validate(ValueBox(0))

    assert report.has_errors() is True
    assert len(report.records) == 2
    crasher_errors = [d for d in report.errors() if d.pass_name == "tests.vm.crasher"]
    assert crasher_errors, "Crasher should have produced an ERROR diagnostic"
    assert "internal boom" in crasher_errors[-1].message_text
    assert any(d.message_text == "still-runs" for d in report.errors())


def test_validation_manager_captures_diagnostics_emitted_before_crash() -> None:
    """Test that pre-crash diagnostics are kept with the synthesized error."""

    @register_pass("tests.vm.report_then_crash", "Reports then crashes.")
    class _ReportThenCrash(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.WARNING, "heads-up")
            raise RuntimeError("kaboom")

    manager = ValidationManager[ValueBox]()
    manager.add(_ReportThenCrash())
    report = manager.validate(ValueBox(0))

    messages = [d.message_text for d in report.diagnostics]
    assert "heads-up" in messages
    assert any("kaboom" in m for m in messages)


def test_validation_manager_continues_after_pass_execution_error() -> None:
    """Test that a validator raising PassExecutionError does not stop the pipeline."""

    @register_pass("tests.vm.raises_execution_error", "Emits then raises")
    class _RaisesExecution(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, "real-problem")
            raise PassExecutionError("internal wrap")

    follow_up = _single_warning_validator("tests.vm.after_exec_err", "still-here")

    manager = ValidationManager[ValueBox]()
    manager.add(_RaisesExecution())
    manager.add(follow_up)
    report = manager.validate(ValueBox(0))

    assert any(d.message_text == "real-problem" for d in report.errors())
    assert any(d.message_text == "still-here" for d in report.warnings())
    assert [r.pass_name for r in report.records] == [
        "tests.vm.raises_execution_error",
        "tests.vm.after_exec_err",
    ]


def test_validation_manager_synthesizes_error_when_pass_raises_without_reporting() -> (
    None
):
    """Test that an ERROR is synthesized when infra raises without reporting."""

    @register_pass("tests.vm.rejects_input", "Rejects None via validate_input.")
    class _RejectsInput(AnalysisVisitablePass[Any]):
        def visit_unknown(self, node: Any) -> None:
            _ = node

    manager = ValidationManager[Any]()
    manager.add(_RejectsInput())
    report = manager.validate(None)  # validate_input raises PassValidationError

    assert report.has_errors() is True
    errors = report.errors()
    assert any(
        "does not accept None" in d.message_text for d in errors
    ), f"expected infra-emitted diagnostic, got {[d.message_text for d in errors]}"


def test_validation_manager_returns_report_that_satisfies_partial_equal() -> None:
    """Test that report and records satisfy the PartialEqual protocol."""
    manager = ValidationManager[ValueBox]()
    manager.add(_clean_validator("tests.vm.partial_equal_clean"))
    report = manager.validate(ValueBox(0))

    assert isinstance(report, PartialEqual)
    assert report.supports_partial_equality is True
    assert isinstance(report.records[0], PartialEqual)


def test_validation_manager_end_to_end_raise_and_print_flow() -> None:
    """Test the end-to-end run, format, and raise-on-failure workflow."""
    manager = ValidationManager[ValueBox](name=Identifier("tests.vm.e2e"))
    manager.add(
        _single_error_validator(
            "tests.vm.e2e.missing_return",
            "function foo() has no return",
            detail="foo @ line 17",
        )
    )
    manager.add(_single_warning_validator("tests.vm.e2e.unused", "x is unused"))
    manager.add(_single_error_validator("tests.vm.e2e.shape", "shape mismatch"))

    report = manager.validate(ValueBox(0))

    # User prints the errors for the developer.
    rendered = report.format()
    assert (
        "[ERROR] tests.vm.e2e.missing_return: function foo() has no return" in rendered
    )
    assert "    detail: foo @ line 17" in rendered
    assert "[WARNING] tests.vm.e2e.unused: x is unused" in rendered
    assert "[ERROR] tests.vm.e2e.shape: shape mismatch" in rendered

    # Then bails out.
    with pytest.raises(ValidationFailedError) as excinfo:
        report.raise_if_failed()
    assert excinfo.value.report is report
    assert len(excinfo.value.report.errors()) == 2


def test_validation_manager_add_returns_none_and_exposes_validators_in_order() -> None:
    """Test that add() returns None and validators are exposed in registration order."""
    first = _clean_validator("tests.vm.add_first")
    second = _clean_validator("tests.vm.add_second")

    manager = ValidationManager[ValueBox]()
    assert manager.add(first) is None
    assert manager.add(second) is None

    assert manager.validators == (first, second)


def test_validation_manager_default_and_explicit_identifier() -> None:
    """Test that the identifier defaults but accepts an override."""
    default = ValidationManager[ValueBox]()
    named = ValidationManager[ValueBox](name=Identifier("tests.vm.named"))

    assert default.name.name_hint == "validation-pipeline"
    assert default.get_identifier() is default.name
    assert named.name.name_hint == "tests.vm.named"


def test_validation_manager_record_reports_no_changes_and_preserves_all() -> None:
    """Test that each record reports changed=False and preserves all."""
    manager = ValidationManager[ValueBox]()
    manager.add(_single_warning_validator("tests.vm.record_warn", "msg"))
    report = manager.validate(ValueBox(0))

    assert len(report.records) == 1
    record = report.records[0]
    assert isinstance(record, PassRunRecord)
    assert record.changed is False
    assert record.preserves_all_analyses is True
    assert record.preserved_analyses == ()


def test_validation_manager_does_not_suppress_pass_validation_error_diagnostics() -> (
    None
):
    """Test that a reported ERROR is kept without a synthetic fallback."""

    @register_pass(
        "tests.vm.report_then_pass_validation_error",
        "Reports an error then raises PassValidationError.",
    )
    class _ReportThenValidate(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, "real-error")
            raise PassValidationError("follow-up raise")

    manager = ValidationManager[ValueBox]()
    manager.add(_ReportThenValidate())
    report = manager.validate(ValueBox(0))

    messages = [d.message_text for d in report.errors()]
    # Only the real, explicitly-reported ERROR should be present —
    # no synthetic "raised without reporting" fallback.
    assert "real-error" in messages
    assert not any("raised without reporting" in m for m in messages)


def test_validation_manager_aggregates_mixed_severity_levels_through_pipeline() -> None:
    """Test that INFO/WARNING/ERROR all propagate through the manager in order."""
    manager = ValidationManager[ValueBox]()
    manager.add(_single_info_validator("tests.vm.mixed.info", "note-me"))
    manager.add(_single_warning_validator("tests.vm.mixed.warn", "watch-me"))
    manager.add(_single_error_validator("tests.vm.mixed.err", "fix-me"))

    report = manager.validate(ValueBox(0))

    assert [d.message_text for d in report.diagnostics] == [
        "note-me",
        "watch-me",
        "fix-me",
    ]
    assert tuple(d.message_text for d in report.infos()) == ("note-me",)
    assert tuple(d.message_text for d in report.warnings()) == ("watch-me",)
    assert tuple(d.message_text for d in report.errors()) == ("fix-me",)


def test_validation_manager_attributes_each_diagnostic_to_emitting_validator() -> None:
    """Test that `pass_name` on each diagnostic identifies its emitter."""
    manager = ValidationManager[ValueBox]()
    manager.add(_single_warning_validator("tests.vm.attr.first", "first-msg"))
    manager.add(_single_error_validator("tests.vm.attr.second", "second-msg"))

    report = manager.validate(ValueBox(0))

    assert [(d.pass_name, d.message_text) for d in report.diagnostics] == [
        ("tests.vm.attr.first", "first-msg"),
        ("tests.vm.attr.second", "second-msg"),
    ]


def test_validation_manager_preserves_multiple_diagnostics_from_single_validator() -> (
    None
):
    """Test that all diagnostics from one validator are kept in emission order."""

    @register_pass("tests.vm.multi_emit", "Reports multiple diagnostics in one visit.")
    class _MultiEmit(AnalysisVisitablePass[ValueBox]):
        def visit_value_box(self, node: ValueBox) -> None:
            _ = node
            self.report(DiagnosticLevel.INFO, "info-1")
            self.report(DiagnosticLevel.WARNING, "warn-1")
            self.report(DiagnosticLevel.ERROR, "err-1")
            self.report(DiagnosticLevel.WARNING, "warn-2")

    manager = ValidationManager[ValueBox]()
    manager.add(_MultiEmit())
    report = manager.validate(ValueBox(0))

    assert [d.message_text for d in report.diagnostics] == [
        "info-1",
        "warn-1",
        "err-1",
        "warn-2",
    ]
    assert len(report.records) == 1
    assert len(report.records[0].diagnostics) == 4
