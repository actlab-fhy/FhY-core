"""Tests for generic compiler pass infrastructure."""

import logging

import pytest

from fhy_core.diagnostic import Note
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    DiagnosticLevel,
    PassExecutionError,
    PassInfo,
    PassRegistrationError,
    PassResult,
    PassValidationError,
    PreservedAnalyses,
    register_pass,
)
from fhy_core.trait import PartialEqual

_PASS_INFRA_LOGGER_PREFIX = "fhy_core.pass_infrastructure.core"


def test_compiler_pass_executes_and_tracks_stats() -> None:
    """Test that a registered pass executes and tracks run statistics."""

    @register_pass("tests.append_pass", "Append sentinel value to a list.")
    class AppendPass(CompilerPass[list[int], list[int]]):
        def get_noop_output(self, ir: list[int]) -> list[int]:
            return ir

        def run_pass(self, ir: list[int]) -> list[int]:
            return [*ir, 9]

    total_before = CompilerPass.get_total_run_count()
    per_before = AppendPass.get_run_count()

    result = AppendPass().execute([1, 2])

    assert result.output == [1, 2, 9]
    assert result.changed is True
    assert result.diagnostics == ()
    assert AppendPass.get_run_count() == per_before + 1
    assert CompilerPass.get_total_run_count() == total_before + 1


def test_compiler_pass_wraps_internal_exceptions() -> None:
    """Test that internal exceptions are wrapped as PassExecutionError."""

    @register_pass("tests.exploding_pass", "Always raises during execution.")
    class ExplodingPass(CompilerPass[list[int], list[int]]):
        def get_noop_output(self, ir: list[int]) -> list[int]:
            return ir

        def run_pass(self, ir: list[int]) -> list[int]:
            raise ValueError("boom")

    compiler_pass = ExplodingPass()
    with pytest.raises(PassExecutionError):
        compiler_pass.execute([1])

    assert compiler_pass.diagnostics
    assert compiler_pass.diagnostics[0].level == DiagnosticLevel.ERROR
    assert isinstance(compiler_pass.diagnostics[0].message, Note)
    assert "boom" in compiler_pass.diagnostics[0].message_text


def test_compiler_pass_rejects_none_input() -> None:
    """Test that None input is rejected by default validation."""

    @register_pass("tests.identity_pass", "Identity pass for object IR.")
    class IdentityPass(CompilerPass[object, object]):
        def get_noop_output(self, ir: object) -> object:
            return ir

        def run_pass(self, ir: object) -> object:
            return ir

    compiler_pass = IdentityPass()
    with pytest.raises(PassValidationError):
        compiler_pass.execute(None)

    assert compiler_pass.diagnostics
    assert compiler_pass.diagnostics[0].level == DiagnosticLevel.ERROR
    assert isinstance(compiler_pass.diagnostics[0].message, Note)
    assert "does not accept None" in compiler_pass.diagnostics[0].message_text


def test_compiler_pass_registry_create_and_collision() -> None:
    """Test pass creation from registry and duplicate-name rejection."""

    @register_pass("tests.creatable_pass", "Increment an integer by one.")
    class CreatablePass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

    created = CompilerPass.create("tests.creatable_pass")
    assert isinstance(created, CreatablePass)
    assert created(2) == 3
    registered = CompilerPass.get_registered_passes()["tests.creatable_pass"]
    assert isinstance(registered, PassInfo)
    assert registered.description == "Increment an integer by one."

    with pytest.raises(PassRegistrationError):

        @register_pass("tests.creatable_pass", "Duplicate named pass.")
        class DuplicatePass(CompilerPass[int, int]):
            def get_noop_output(self, ir: int) -> int:
                return ir

            def run_pass(self, ir: int) -> int:
                return ir


def test_compiler_pass_skip_path_uses_noop_output() -> None:
    """Test the skipped execution path returns noop output, preserves all
    analyses, and does not increment the run counter."""

    @register_pass("tests.skipped_pass", "Skip execution and return noop output.")
    class SkippedPass(CompilerPass[int, int]):
        def should_run(self, ir: int) -> bool:
            _ = ir
            self.report(DiagnosticLevel.INFO, "skip requested")
            return False

        def get_noop_output(self, ir: int) -> int:
            return ir + 100

        def run_pass(self, ir: int) -> int:
            return ir + 1

    total_before = CompilerPass.get_total_run_count()
    per_before = SkippedPass.get_run_count()

    result = SkippedPass().execute(2)

    assert result.output == 102
    assert result.changed is False
    assert result.preserved_analyses == PreservedAnalyses.all()
    assert result.diagnostics
    assert result.diagnostics[0].level == DiagnosticLevel.INFO
    assert result.diagnostics[0].message_text == "skip requested"

    # Skipped runs do not count as executions.
    assert SkippedPass.get_run_count() == per_before
    assert CompilerPass.get_total_run_count() == total_before


def test_preserved_analyses_is_immutable() -> None:
    """Test preserved analyses updates produce new immutable values."""
    analysis_name = Identifier("tests.analysis")
    original = PreservedAnalyses.none()
    updated = original.preserve(analysis_name)

    assert original.is_preserved(analysis_name) is False
    assert updated.is_preserved(analysis_name) is True


def test_pass_result_preserved_analyses_cannot_be_mutated_in_place() -> None:
    """Test `PassResult` preserved analyses cannot be mutated in place."""
    analysis_name = Identifier("tests.analysis")
    result = PassResult(output=0, changed=False)
    updated = result.preserved_analyses.preserve(analysis_name)

    assert result.preserved_analyses.is_preserved(analysis_name) is False
    assert updated.is_preserved(analysis_name) is True


def test_pass_core_records_support_partial_equal_traits() -> None:
    """Test pass-core records satisfy `PartialEqual` protocol."""
    info = PassInfo("tests.info", "desc", CompilerPass)  # type: ignore[type-abstract]  # test: abstract class used as placeholder
    result = PassResult(
        output=0,
        changed=False,
        diagnostics=(),
        preserved_analyses=PreservedAnalyses.none(),
    )
    preserved = PreservedAnalyses.none()

    assert isinstance(info, PartialEqual)
    assert info.supports_partial_equality is True
    assert isinstance(result, PartialEqual)
    assert result.supports_partial_equality is True
    assert isinstance(preserved, PartialEqual)
    assert preserved.supports_partial_equality is True


# ---------------------------------------------------------------------------
# Lifecycle exceptions are wrapped to PassValidationError /
# PassExecutionError. The validate_* methods wrap to PassValidationError;
# the others wrap to PassExecutionError. PassValidationError /
# PassExecutionError raised by user code pass through unchanged.
# ---------------------------------------------------------------------------


def _identity_pass_methods() -> dict[str, object]:
    """Return the minimum-set of method overrides for a working int identity pass."""
    return {
        "get_noop_output": lambda self, ir: ir,
        "run_pass": lambda self, ir: ir,
    }


def test_validate_input_unexpected_exception_wraps_to_pass_validation_error() -> None:
    """Test unexpected exceptions in `validate_input` wrap to PassValidationError."""

    @register_pass(
        "tests.wrap.validate_input_raises", "validate_input raises KeyError."
    )
    class WrappedValidateInputPass(CompilerPass[int, int]):
        def validate_input(self, ir: int) -> None:
            raise KeyError("missing-key")

        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir

    compiler_pass = WrappedValidateInputPass()
    with pytest.raises(PassValidationError, match="missing-key"):
        compiler_pass.execute(1)

    assert compiler_pass.diagnostics
    assert compiler_pass.diagnostics[-1].level == DiagnosticLevel.ERROR
    assert "missing-key" in compiler_pass.diagnostics[-1].message_text


def test_validate_output_unexpected_exception_wraps_to_pass_validation_error() -> None:
    """Test unexpected exceptions in `validate_output` wrap to PassValidationError."""

    @register_pass(
        "tests.wrap.validate_output_raises", "validate_output raises KeyError."
    )
    class WrappedValidateOutputPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

        def validate_output(self, input_ir: int, output: int) -> None:
            raise KeyError("output-issue")

    compiler_pass = WrappedValidateOutputPass()
    with pytest.raises(PassValidationError, match="output-issue"):
        compiler_pass.execute(1)

    assert any(
        "output-issue" in d.message_text and d.level == DiagnosticLevel.ERROR
        for d in compiler_pass.diagnostics
    )


def test_should_run_unexpected_exception_wraps_to_pass_execution_error() -> None:
    """Test unexpected exceptions in `should_run` wrap to PassExecutionError."""

    @register_pass("tests.wrap.should_run_raises", "should_run raises RuntimeError.")
    class WrappedShouldRunPass(CompilerPass[int, int]):
        def should_run(self, ir: int) -> bool:
            raise RuntimeError("predicate-broken")

        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir

    compiler_pass = WrappedShouldRunPass()
    with pytest.raises(PassExecutionError, match="predicate-broken"):
        compiler_pass.execute(1)

    assert any(
        "predicate-broken" in d.message_text and d.level == DiagnosticLevel.ERROR
        for d in compiler_pass.diagnostics
    )


def test_get_noop_output_unexpected_exception_wraps_to_pass_execution_error() -> None:
    """Test unexpected exceptions in `get_noop_output` (skip path) wrap to
    PassExecutionError."""

    @register_pass(
        "tests.wrap.get_noop_output_raises",
        "should_run False, get_noop_output raises RuntimeError.",
    )
    class WrappedNoopPass(CompilerPass[int, int]):
        def should_run(self, ir: int) -> bool:
            return False

        def get_noop_output(self, ir: int) -> int:
            raise RuntimeError("noop-broken")

        def run_pass(self, ir: int) -> int:
            return ir

    with pytest.raises(PassExecutionError, match="noop-broken"):
        WrappedNoopPass().execute(1)


def test_did_change_unexpected_exception_wraps_to_pass_execution_error() -> None:
    """Test unexpected exceptions in `did_change` wrap to PassExecutionError."""

    @register_pass("tests.wrap.did_change_raises", "did_change raises RuntimeError.")
    class WrappedDidChangePass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

        def did_change(self, input_ir: int, output: int) -> bool:
            raise RuntimeError("did-change-broken")

    with pytest.raises(PassExecutionError, match="did-change-broken"):
        WrappedDidChangePass().execute(1)


def test_get_preserved_analyses_wraps_unexpected_exception() -> None:
    """Test that exceptions in `get_preserved_analyses` wrap to `PassExecutionError`."""

    @register_pass(
        "tests.wrap.get_preserved_raises",
        "get_preserved_analyses raises RuntimeError.",
    )
    class WrappedPreservedPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

        def get_preserved_analyses(
            self, input_ir: int, output: int, *, changed: bool
        ) -> PreservedAnalyses:
            raise RuntimeError("preserved-broken")

    with pytest.raises(PassExecutionError, match="preserved-broken"):
        WrappedPreservedPass().execute(1)


def test_explicit_pass_validation_error_passes_through_unchanged() -> None:
    """Test PassValidationError raised by user code is not double-wrapped."""

    @register_pass(
        "tests.wrap.explicit_validation_error",
        "validate_input raises PassValidationError directly.",
    )
    class ExplicitValidationPass(CompilerPass[int, int]):
        def validate_input(self, ir: int) -> None:
            raise PassValidationError("explicitly-invalid")

        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir

    with pytest.raises(PassValidationError, match="explicitly-invalid"):
        ExplicitValidationPass().execute(1)


def test_explicit_pass_execution_error_passes_through_unchanged() -> None:
    """Test PassExecutionError raised by user code is not double-wrapped."""

    @register_pass(
        "tests.wrap.explicit_execution_error",
        "run_pass raises PassExecutionError directly.",
    )
    class ExplicitExecutionPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            raise PassExecutionError("explicitly-broken")

    with pytest.raises(PassExecutionError, match="explicitly-broken"):
        ExplicitExecutionPass().execute(1)


# ---------------------------------------------------------------------------
# Run counters track real executions, not invocations.
# ---------------------------------------------------------------------------


def test_run_counter_counts_only_real_executions() -> None:
    """Test that `_record_run` only counts passes that actually executed."""

    @register_pass(
        "tests.counter.gated", "Pass that runs only when its input is positive."
    )
    class GatedPass(CompilerPass[int, int]):
        def should_run(self, ir: int) -> bool:
            return ir > 0

        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

    per_before = GatedPass.get_run_count()
    total_before = CompilerPass.get_total_run_count()
    compiler_pass = GatedPass()

    compiler_pass.execute(0)  # skipped - should NOT count
    compiler_pass.execute(0)  # skipped - should NOT count

    assert GatedPass.get_run_count() == per_before
    assert CompilerPass.get_total_run_count() == total_before

    compiler_pass.execute(1)  # executed
    compiler_pass.execute(2)  # executed

    assert GatedPass.get_run_count() == per_before + 2
    assert CompilerPass.get_total_run_count() == total_before + 2


def test_run_counter_counts_attempts_even_when_run_pass_raises() -> None:
    """Test run counters increment for a pass that crashes inside ``run_pass``."""

    @register_pass(
        "tests.counter.crashes_in_run_pass",
        "Pass that always raises inside run_pass.",
    )
    class CrashingRunPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            raise RuntimeError("intentional crash in run_pass")

    per_before = CrashingRunPass.get_run_count()
    total_before = CompilerPass.get_total_run_count()
    compiler_pass = CrashingRunPass()

    with pytest.raises(PassExecutionError):
        compiler_pass.execute(1)

    assert CrashingRunPass.get_run_count() == per_before + 1
    assert CompilerPass.get_total_run_count() == total_before + 1

    # A second crashed attempt still counts.
    with pytest.raises(PassExecutionError):
        compiler_pass.execute(2)

    assert CrashingRunPass.get_run_count() == per_before + 2
    assert CompilerPass.get_total_run_count() == total_before + 2


# ---------------------------------------------------------------------------
# register_pass input validation.
# ---------------------------------------------------------------------------


def test_register_pass_rejects_empty_name() -> None:
    """Test that `register_pass` rejects an empty name."""
    with pytest.raises(PassRegistrationError, match="name"):

        @register_pass("", "non-empty description")
        class _EmptyNamePass(CompilerPass[int, int]):
            def get_noop_output(self, ir: int) -> int:
                return ir

            def run_pass(self, ir: int) -> int:
                return ir


def test_register_pass_rejects_empty_description() -> None:
    """Test that `register_pass` rejects an empty description."""
    with pytest.raises(PassRegistrationError, match="description"):

        @register_pass("tests.register.empty_description", "")
        class _EmptyDescriptionPass(CompilerPass[int, int]):
            def get_noop_output(self, ir: int) -> int:
                return ir

            def run_pass(self, ir: int) -> int:
                return ir


def test_register_pass_rejects_non_compiler_pass_class() -> None:
    """Test that `register_pass` rejects non-`CompilerPass` subclasses."""
    with pytest.raises(PassRegistrationError, match="CompilerPass"):

        @register_pass("tests.register.bad_class", "Not a CompilerPass.")
        class _NotAPass:  # type: ignore[type-var]  # test: deliberately invalid input
            pass


def test_register_pass_rejects_same_class_with_different_description() -> None:
    """Test re-registering the same class with a different description raises."""

    @register_pass("tests.register.mismatch", "Original description.")
    class MismatchPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir

    with pytest.raises(PassRegistrationError, match="refusing to overwrite"):
        register_pass("tests.register.mismatch", "A different description.")(
            MismatchPass
        )

    # The original description must remain intact.
    registered = CompilerPass.get_registered_passes()["tests.register.mismatch"]
    assert registered.description == "Original description."


def test_register_pass_is_idempotent_for_same_class() -> None:
    """Test that re-registering the same class with the same name is a no-op."""

    @register_pass("tests.register.idempotent", "Idempotent registration.")
    class IdempotentPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir

    # Re-applying the same decorator with the same args to the same class
    # must not raise.
    register_pass("tests.register.idempotent", "Idempotent registration.")(
        IdempotentPass
    )

    info = CompilerPass.get_registered_passes()["tests.register.idempotent"]
    assert info.pass_type is IdempotentPass


# ---------------------------------------------------------------------------
# PreservedAnalyses contracts.
# ---------------------------------------------------------------------------


def test_preserved_analyses_all_preserves_arbitrary_name() -> None:
    """Test `PreservedAnalyses.all().is_preserved(any_name)` returns True."""
    assert PreservedAnalyses.all().is_preserved(Identifier("any.thing")) is True
    assert PreservedAnalyses.all().is_preserved(Identifier("other.thing")) is True


def test_preserve_on_preserve_all_returns_self() -> None:
    """Test that `preserve` is a no-op when `preserve_all=True` is set."""
    preserved = PreservedAnalyses.all()

    result = preserved.preserve(Identifier("ignored.name"))

    assert result is preserved


def test_preserve_when_already_preserved_returns_self() -> None:
    """Test that re-preserving an already-present name returns the same instance."""
    name = Identifier("dup.name")
    preserved = PreservedAnalyses.none().preserve(name)

    result = preserved.preserve(name)

    assert result is preserved


# ---------------------------------------------------------------------------
# report() emits to a per-pass-class FhY-core logger.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("diagnostic_level", "logging_level"),
    [
        (DiagnosticLevel.ERROR, logging.ERROR),
        (DiagnosticLevel.WARNING, logging.WARNING),
        (DiagnosticLevel.INFO, logging.INFO),
    ],
    ids=["error", "warning", "info"],
)
def test_report_emits_log_record_at_matching_level(
    caplog: pytest.LogCaptureFixture,
    diagnostic_level: DiagnosticLevel,
    logging_level: int,
) -> None:
    """Test that `report` emits a log record at the level matching the diagnostic."""

    @register_pass(
        f"tests.log.level.{diagnostic_level.value}",
        f"Emits one {diagnostic_level.value} report.",
    )
    class LoggingLevelPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            self.report(diagnostic_level, "level-test-message")
            return ir

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        LoggingLevelPass().execute(0)

    matching = [
        record
        for record in caplog.records
        if record.getMessage() == "level-test-message"
    ]
    assert len(matching) == 1
    assert matching[0].levelno == logging_level


def test_report_appends_detail_to_log_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that `detail` is appended to the log message in the documented format."""

    @register_pass("tests.log.detail", "Emits a report with a detail string.")
    class DetailPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            self.report(
                DiagnosticLevel.WARNING, "primary-message", detail="extra-context"
            )
            return ir

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        DetailPass().execute(0)

    matching = [
        record for record in caplog.records if "primary-message" in record.getMessage()
    ]
    assert len(matching) == 1
    assert matching[0].getMessage() == "primary-message | detail: extra-context"


def test_report_uses_per_pass_class_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that two distinct passes emit on distinct logger names."""

    @register_pass("tests.log.a", "First logging pass.")
    class LoggerA(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.INFO, "from-A")
            return ir

    @register_pass("tests.log.b", "Second logging pass.")
    class LoggerB(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.INFO, "from-B")
            return ir

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        LoggerA().execute(0)
        LoggerB().execute(0)

    a_records = [r for r in caplog.records if r.getMessage() == "from-A"]
    b_records = [r for r in caplog.records if r.getMessage() == "from-B"]
    assert len(a_records) == 1
    assert len(b_records) == 1
    assert a_records[0].name != b_records[0].name
    assert a_records[0].name.endswith("tests.log.a")
    assert b_records[0].name.endswith("tests.log.b")


def test_report_logging_does_not_disturb_diagnostics_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that logging is purely additive: in-memory diagnostics stay
    populated identically whether or not a log handler is attached."""

    @register_pass("tests.log.additive", "Verifies logging is additive.")
    class AdditivePass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.INFO, "additive-msg")
            return ir

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        result = AdditivePass().execute(0)

    assert any(d.message_text == "additive-msg" for d in result.diagnostics)


def test_guarded_exception_attaches_exc_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that the synthesized ERROR log from a crashing run_pass carries exc_info."""

    @register_pass(
        "tests.log.guarded_exc_info", "Verifies exc_info travels with guarded errors."
    )
    class CrashingPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            raise ValueError("boom")

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        with pytest.raises(PassExecutionError):
            CrashingPass().execute(0)

    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "boom" in record.getMessage()
    ]
    assert error_records, "expected ERROR record for crash"
    assert any(record.exc_info is not None for record in error_records), (
        "expected exc_info on at least one ERROR record"
    )


def test_execute_emits_lifecycle_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test CompilerPass.execute emits DEBUG entry/exit records."""

    @register_pass("tests.log.lifecycle", "Verifies execute lifecycle DEBUGs.")
    class LifecyclePass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

        def run_pass(self, ir: int) -> int:
            return ir + 1

    with caplog.at_level(logging.DEBUG, logger=_PASS_INFRA_LOGGER_PREFIX):
        LifecyclePass().execute(0)

    entry = [
        record
        for record in caplog.records
        if "entering" in record.getMessage()
        and record.name.endswith("tests.log.lifecycle")
    ]
    finished = [
        record
        for record in caplog.records
        if "finished" in record.getMessage()
        and record.name.endswith("tests.log.lifecycle")
    ]
    assert entry, "expected entry DEBUG record"
    assert finished, "expected finished DEBUG record"
