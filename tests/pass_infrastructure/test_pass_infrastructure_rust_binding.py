"""Interface tests for the pass infrastructure's Rust binding.

The pass infrastructure runs on the Rust core (``fhy_core._rs``, S6 of
``docs/design/python-switch.md``). The other suites in this directory test
the Python API's behavior; this one covers what the binding adds over the
core: the class structure, the mapping of the Python hooks onto the core's
lifecycle, the owned hook context, the Python errors, the analysis cache
seen from Python, pipelines and validation over Python objects, logging,
and pickles.
"""

import gc
import logging
import pickle
import weakref
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import Diagnostic, DiagnosticLevel, Note, ValidationReport
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    Analysis,
    AnalysisManager,
    CompilerPass,
    FixpointGroupRecord,
    FixpointIterationRecord,
    FixpointPassGroup,
    PassExecutionError,
    PassManager,
    PassManagerResult,
    PassResult,
    PassRunRecord,
    PassValidationError,
    PreservedAnalyses,
    ValidationManager,
    Validator,
    ValidatorRecord,
    register_verification,
)
from fhy_core.pass_infrastructure.core import AnalysisVisitablePass
from fhy_core.symbolic.expression import (
    Capture,
    CapturePattern,
    IdentifierExpression,
    LiteralExpression,
    LiteralPattern,
    RewriteRule,
    RewriteRuleApplier,
)
from fhy_core.symbolic.expression.core import BinaryExpression, BinaryOperation
from fhy_core.symbolic.expression.pattern.core import BinaryExpressionPattern
from fhy_core.traits import FrozenMixin, Visitable, VisitableMixin
from fhy_core.utils.override import override

_CORE_LOGGER = "fhy_core.pass_infrastructure.core"
_MANAGER_LOGGER = "fhy_core.pass_infrastructure.manager"
_VALIDATION_LOGGER = "fhy_core.pass_infrastructure.validation"


@dataclass
class Box(FrozenMixin, VisitableMixin):
    """A frozen IR node."""

    value: int

    def __post_init__(self) -> None:
        self.freeze()


class CountingAnalysis(Analysis[Box, int]):
    """Analysis that reads the box's value and counts its runs."""

    runs: ClassVar[int] = 0

    @override
    def run(self, ir: Box) -> int:
        type(self).runs += 1
        return ir.value


@pytest.fixture(autouse=True)
def _reset_analysis_runs() -> None:
    CountingAnalysis.runs = 0


class IdentityPass(CompilerPass[Any, Any]):
    """Pass that returns its input."""

    @override
    def run_pass(self, ir: Any) -> Any:
        return ir


class IncrementPass(CompilerPass[Box, Box]):
    """Pass that returns a new box holding the next value."""

    @override
    def run_pass(self, ir: Box) -> Box:
        return Box(ir.value + 1)


def _run_one(compiler_pass: CompilerPass[Any, Any], ir: Any) -> PassManagerResult[Any]:
    """Run ``compiler_pass`` over ``ir`` in a pipeline without a verifier."""
    manager = PassManager[Any]()
    manager.add_pass(compiler_pass)
    manager.set_verifier(None)
    return manager.run(ir)


# ---------------------------------------------------------------------------
# Class structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("public_class", "rust_class"),
    [
        (CompilerPass, _rs.CompilerPassBase),
        (Analysis, _rs.AnalysisBase),
        (Validator, _rs.ValidatorBase),
        (PreservedAnalyses, _rs.PreservedAnalyses),
        (PassResult, _rs.PassResult),
        (PassRunRecord, _rs.PassRunRecord),
        (FixpointIterationRecord, _rs.FixpointIterationRecord),
        (FixpointGroupRecord, _rs.FixpointGroupRecord),
        (PassManagerResult, _rs.PassManagerResult),
        (ValidatorRecord, _rs.ValidatorRecord),
        (PassManager, _rs.PassManager),
        (FixpointPassGroup, _rs.FixpointPassGroup),
        (ValidationManager, _rs.ValidationManager),
    ],
)
def test_public_class_extends_its_rust_class(
    public_class: type, rust_class: type
) -> None:
    """Test each public class is a Python subclass of its `_rs` class."""
    assert issubclass(public_class, rust_class)


def test_analysis_manager_is_the_rust_view_without_a_constructor() -> None:
    """Test `AnalysisManager` is the `_rs` view, which cannot be constructed."""
    assert AnalysisManager is _rs.AnalysisManager
    assert AnalysisManager[Box] is AnalysisManager
    with pytest.raises(TypeError):
        AnalysisManager()


@pytest.mark.parametrize(
    "value",
    [
        PreservedAnalyses.all(),
        PassResult(0, changed=False),
        PassRunRecord("p", False, (), PreservedAnalyses.all()),
        FixpointIterationRecord(1, False, ()),
        FixpointGroupRecord(Identifier("g"), (), True),
        PassManagerResult(0, ()),
        ValidatorRecord("v", False, ()),
    ],
    ids=lambda value: type(value).__name__,
)
def test_values_are_frozen_mixins_that_refuse_mutation(value: Any) -> None:
    """Test the Rust-backed values are virtual `FrozenMixin`s and immutable."""
    assert isinstance(value, FrozenMixin)
    assert value.is_frozen is True
    with pytest.raises(AttributeError, match="frozen"):
        setattr(value, "anything", 1)  # noqa: B010
    with pytest.raises(AttributeError, match="frozen"):
        delattr(value, "anything")


def test_abstract_hooks_are_enforced() -> None:
    """Test `run_pass`, `Analysis.run` and `Validator.validate` are abstract."""

    class NoRunPass(CompilerPass[int, int]):
        pass

    class NoRunAnalysis(Analysis[int, int]):
        pass

    class NoValidateValidator(Validator[int]):
        pass

    abstract_classes: tuple[Any, ...] = (
        CompilerPass,
        NoRunPass,
        NoRunAnalysis,
        NoValidateValidator,
    )
    for abstract_class in abstract_classes:
        with pytest.raises(TypeError, match="abstract"):
            abstract_class()


def test_get_noop_output_is_not_abstract() -> None:
    """Test a pass that never skips needs no `get_noop_output` (D-S6-2)."""
    assert IdentityPass().execute(3).output == 3


@pytest.mark.parametrize("calls_super", [True, False])
def test_subclass_with_its_own_init_constructs(calls_super: bool) -> None:
    """Test a subclass's `__init__`, with or without `super().__init__()`, works."""

    class ScaledPass(CompilerPass[int, int]):
        def __init__(self, scale: int) -> None:
            if calls_super:
                super().__init__()
            self.scale = scale

        @override
        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.INFO, "scaled")
            return ir * self.scale

    compiler_pass = ScaledPass(3)
    result = compiler_pass.execute(2)

    assert result.output == 6
    assert [d.message_text for d in compiler_pass.diagnostics] == ["scaled"]


def test_constructor_refuses_arguments_it_does_not_take() -> None:
    """Test `CompilerPass()` takes no arguments, as before the switch."""
    with pytest.raises(TypeError):
        IdentityPass(1)


# ---------------------------------------------------------------------------
# The hook mapping
# ---------------------------------------------------------------------------


class RecordingPass(CompilerPass[int, int]):
    """Pass recording every hook the lifecycle calls, in order."""

    def __init__(self, *, skip: bool = False) -> None:
        super().__init__()
        self.skip = skip
        self.calls: list[str] = []

    @override
    def validate_input(self, ir: int) -> None:
        self.calls.append("validate_input")

    @override
    def should_run(self, ir: int) -> bool:
        self.calls.append("should_run")
        return not self.skip

    @override
    def get_noop_output(self, ir: int) -> int:
        self.calls.append("get_noop_output")
        return ir

    @override
    def run_pass(self, ir: int) -> int:
        self.calls.append("run_pass")
        return ir + 1

    @override
    def validate_output(self, input_ir: int, output: int) -> None:
        self.calls.append("validate_output")

    @override
    def did_change(self, input_ir: int, output: int) -> bool:
        self.calls.append("did_change")
        return True

    @override
    def get_preserved_analyses(
        self, input_ir: int, output: int, *, changed: bool
    ) -> PreservedAnalyses:
        self.calls.append(f"get_preserved_analyses(changed={changed})")
        return PreservedAnalyses.none()


def test_hooks_run_in_the_lifecycle_order() -> None:
    """Test a run calls the Python hooks in the core lifecycle's order."""
    compiler_pass = RecordingPass()

    result = compiler_pass.execute(1)

    assert result.output == 2
    assert compiler_pass.calls == [
        "validate_input",
        "should_run",
        "run_pass",
        "validate_output",
        "did_change",
        "get_preserved_analyses(changed=True)",
    ]


def test_false_should_run_calls_get_noop_output_and_no_run_pass() -> None:
    """Test `should_run` and `get_noop_output` together are the core's `skip`."""
    compiler_pass = RecordingPass(skip=True)

    result = compiler_pass.execute(1)

    assert result.output == 1
    assert result.skipped is True
    assert result.changed is False
    assert compiler_pass.calls == [
        "validate_input",
        "should_run",
        "get_noop_output",
        "get_preserved_analyses(changed=False)",
    ]


def test_class_records_which_hooks_with_defaults_it_overrides() -> None:
    """Test `_python_hooks` has a bit per overridden hook, none for the defaults."""
    assert IdentityPass._python_hooks == 0
    assert RecordingPass._python_hooks == 0b011111

    class NamedPass(IdentityPass):
        @classmethod
        @override
        def get_pass_name(cls) -> str:
            return "named"

    assert NamedPass._python_hooks == 0b100000
    assert NamedPass().execute(0).output == 0


@pytest.mark.parametrize(
    "hook",
    [
        "validate_input",
        "should_run",
        "validate_output",
        "did_change",
        "get_preserved_analyses",
    ],
)
def test_hooks_a_class_does_not_override_run_in_rust(
    monkeypatch: pytest.MonkeyPatch, hook: str
) -> None:
    """Test the default of a hook the class does not override calls no Python."""

    class PlainPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            return ir + 1

    calls: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(hook)
        raise AssertionError("the Python default ran")

    monkeypatch.setattr(CompilerPass, hook, spy)

    result = PlainPass().execute(1)

    assert calls == []
    assert result.output == 2
    assert result.changed is True
    assert result.preserved_analyses == PreservedAnalyses.none()


def test_default_hooks_accept_none_and_compare_by_value() -> None:
    """Test the D-S6-3 defaults: `None` is accepted, and `!=` decides the change."""
    assert IdentityPass().execute(None).output is None
    unchanged = _NewEqualBoxPass().execute(Box(1))
    assert unchanged.changed is False
    assert unchanged.preserved_analyses == PreservedAnalyses.all()


class _NewEqualBoxPass(CompilerPass[Box, Box]):
    """Pass returning a new box equal to its input."""

    @override
    def run_pass(self, ir: Box) -> Box:
        return Box(ir.value)


class _Incomparable:
    """IR whose `!=` raises."""

    @override
    def __ne__(self, other: object) -> bool:
        raise TypeError("cannot compare")


def test_default_did_change_falls_back_to_identity_when_ne_raises() -> None:
    """Test the default `did_change` uses `is not` when `!=` raises."""

    class ReplacingPass(CompilerPass[_Incomparable, _Incomparable]):
        @override
        def run_pass(self, ir: _Incomparable) -> _Incomparable:
            return _Incomparable()

    ir = _Incomparable()

    assert IdentityPass().execute(ir).changed is False
    assert ReplacingPass().execute(ir).changed is True


def test_skipping_pass_without_get_noop_output_fails_in_get_noop_output() -> None:
    """Test a false `should_run` without a no-op output fails in that hook."""

    class SkippingPass(IdentityPass):
        @override
        def should_run(self, ir: Any) -> bool:
            return False

    with pytest.raises(PassExecutionError) as excinfo:
        SkippingPass().execute(0)

    assert excinfo.value.hook == "get_noop_output"
    assert isinstance(excinfo.value.__cause__, NotImplementedError)
    assert "get_noop_output" in str(excinfo.value.__cause__)


class _Falsy:
    """Object whose truth value is false."""

    def __bool__(self) -> bool:
        return False


class _Undecidable:
    """Object whose truth value raises."""

    def __bool__(self) -> bool:
        raise ValueError("no truth value")


def test_should_run_and_did_change_results_are_read_by_truthiness() -> None:
    """Test D-S6-15: `should_run` and `did_change` results are read by `bool()`."""

    class TruthyPass(CompilerPass[int, int]):
        @override
        def should_run(self, ir: int) -> Any:
            return ir

        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

        @override
        def did_change(self, input_ir: int, output: int) -> Any:
            return _Falsy() if input_ir == 1 else [1]

    assert TruthyPass().execute(0).skipped is True
    assert TruthyPass().execute(1).changed is False
    assert TruthyPass().execute(2).changed is True


def test_raising_truth_value_fails_the_hook() -> None:
    """Test an exception from `__bool__` is the hook's failure."""

    class UndecidablePass(IdentityPass):
        @override
        def should_run(self, ir: Any) -> Any:
            return _Undecidable()

    with pytest.raises(PassExecutionError) as excinfo:
        UndecidablePass().execute(0)

    assert excinfo.value.hook == "should_run"
    assert str(excinfo.value.__cause__) == "no truth value"


def test_preserved_analyses_of_the_wrong_type_fail_the_hook() -> None:
    """Test D-S6-15: `get_preserved_analyses` must return a `PreservedAnalyses`."""

    class WrongPreservedPass(IdentityPass):
        @override
        def get_preserved_analyses(
            self, input_ir: Any, output: Any, *, changed: bool
        ) -> PreservedAnalyses:
            return "all"  # type: ignore[return-value]

    with pytest.raises(PassExecutionError) as excinfo:
        WrongPreservedPass().execute(0)

    assert excinfo.value.hook == "get_preserved_analyses"
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert str(excinfo.value.__cause__) == (
        "test_preserved_analyses_of_the_wrong_type_fail_the_hook.<locals>."
        "WrongPreservedPass.get_preserved_analyses must return a "
        "PreservedAnalyses, got str."
    )


def test_pass_name_that_is_not_a_str_fails_the_run() -> None:
    """Test a `get_pass_name` returning something else raises `TypeError`."""

    class BadNamePass(IdentityPass):
        @classmethod
        @override
        def get_pass_name(cls) -> str:
            return 3  # type: ignore[return-value]

    with pytest.raises(TypeError, match="get_pass_name must return a str, got int"):
        BadNamePass().execute(0)


# ---------------------------------------------------------------------------
# The hook context
# ---------------------------------------------------------------------------


class ReportingEverywherePass(CompilerPass[int, int]):
    """Pass reporting one diagnostic in every hook that has a context."""

    def __init__(self, *, skip: bool = False) -> None:
        super().__init__()
        self.skip = skip

    @override
    def validate_input(self, ir: int) -> None:
        self.report(DiagnosticLevel.INFO, "validate_input")

    @override
    def should_run(self, ir: int) -> bool:
        self.report(DiagnosticLevel.INFO, "should_run")
        return not self.skip

    @override
    def get_noop_output(self, ir: int) -> int:
        self.report(DiagnosticLevel.INFO, "get_noop_output")
        return ir

    @override
    def run_pass(self, ir: int) -> int:
        self.report(DiagnosticLevel.WARNING, "run_pass")
        return ir

    @override
    def validate_output(self, input_ir: int, output: int) -> None:
        self.report(DiagnosticLevel.INFO, "validate_output")


@pytest.mark.parametrize(
    ("skip", "expected"),
    [
        (False, ["validate_input", "should_run", "run_pass", "validate_output"]),
        (True, ["validate_input", "should_run", "get_noop_output"]),
    ],
)
def test_report_records_in_every_hook_with_a_context(
    skip: bool, expected: list[str]
) -> None:
    """Test `report` records into the run in each hook the core gives a context."""
    compiler_pass = ReportingEverywherePass(skip=skip)

    result = compiler_pass.execute(0)

    assert [d.message_text for d in result.diagnostics] == expected
    assert compiler_pass.diagnostics == result.diagnostics
    assert all(d.source == "ReportingEverywherePass" for d in result.diagnostics)


def test_report_outside_a_run_records_on_the_pass_until_the_next_run() -> None:
    """Test `report` outside a run records on the pass, and a run clears it."""
    compiler_pass = IdentityPass()

    compiler_pass.report(DiagnosticLevel.WARNING, "outside", detail="context")

    (diagnostic,) = compiler_pass.diagnostics
    assert diagnostic.message_text == "outside"
    assert diagnostic.detail == "context"
    assert compiler_pass.execute(0).diagnostics == ()
    assert compiler_pass.diagnostics == ()


def test_diagnostics_during_a_hook_are_the_current_runs() -> None:
    """Test `diagnostics` read in a hook lists the run's diagnostics so far."""
    seen: list[tuple[str, ...]] = []

    class ReadingPass(CompilerPass[int, int]):
        @override
        def validate_input(self, ir: int) -> None:
            self.report(DiagnosticLevel.INFO, "first")

        @override
        def run_pass(self, ir: int) -> int:
            seen.append(tuple(d.message_text for d in self.diagnostics))
            self.report(DiagnosticLevel.INFO, "second")
            seen.append(tuple(d.message_text for d in self.diagnostics))
            return ir

    compiler_pass = ReadingPass()
    compiler_pass.report(DiagnosticLevel.INFO, "stale")
    compiler_pass.execute(0)

    assert seen == [("first",), ("first", "second")]


@pytest.mark.parametrize("hook", ["did_change", "get_preserved_analyses"])
@pytest.mark.parametrize("operation", ["report", "get_analysis"])
def test_hooks_without_a_context_refuse_report_and_get_analysis(
    hook: str, operation: str
) -> None:
    """Test D-S6-4: `report` and `get_analysis` raise in the context-free hooks."""

    def use_context(compiler_pass: CompilerPass[Box, Box]) -> None:
        if operation == "report":
            compiler_pass.report(DiagnosticLevel.INFO, "refused")
        else:
            compiler_pass.get_analysis(CountingAnalysis, Box(0))

    class ContextFreePass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

        @override
        def did_change(self, input_ir: Box, output: Box) -> bool:
            if hook == "did_change":
                use_context(self)
            return False

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            if hook == "get_preserved_analyses":
                use_context(self)
            return PreservedAnalyses.all()

    with pytest.raises(PassExecutionError) as excinfo:
        ContextFreePass().execute(Box(0))

    assert excinfo.value.hook == hook
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == (
        f"{operation} is not available in {hook}, which runs without a pass context"
    )


def test_reported_diagnostics_come_back_as_the_same_objects() -> None:
    """Test D-S6-18: a reported diagnostic is one object everywhere it appears."""
    reported: list[Diagnostic] = []

    class ReportingPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.INFO, "same object")
            reported.append(self.diagnostics[-1])
            return ir + 1

    compiler_pass = ReportingPass()
    result = compiler_pass.execute(0)
    pipeline = _run_one(compiler_pass, 0)

    assert result.diagnostics[0] is reported[0]
    (record,) = pipeline.records
    assert isinstance(record, PassRunRecord)
    assert record.diagnostics[0] is reported[1]
    assert compiler_pass.diagnostics[0] is reported[1]


def test_failure_diagnostic_is_one_object_on_the_error_and_the_pass() -> None:
    """Test the core-made failure diagnostic is built once."""

    class FailingPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.WARNING, "before failing")
            raise ValueError("boom")

    compiler_pass = FailingPass()
    with pytest.raises(PassExecutionError) as excinfo:
        compiler_pass.execute(0)

    assert excinfo.value.diagnostics == compiler_pass.diagnostics
    assert all(
        error_diagnostic is pass_diagnostic
        for error_diagnostic, pass_diagnostic in zip(
            excinfo.value.diagnostics, compiler_pass.diagnostics, strict=True
        )
    )
    assert [d.message_text for d in excinfo.value.diagnostics] == [
        "before failing",
        'pass "FailingPass" failed in run_pass: ValueError: boom',
    ]


def test_report_refuses_a_value_that_is_not_a_diagnostic() -> None:
    """Test the binding's recording entry point type-checks its argument."""
    with pytest.raises(TypeError, match="report diagnostic must be a Diagnostic"):
        IdentityPass()._record_diagnostic("text")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hook", "error_class"),
    [
        ("validate_input", PassValidationError),
        ("should_run", PassExecutionError),
        ("get_noop_output", PassExecutionError),
        ("run_pass", PassExecutionError),
        ("validate_output", PassValidationError),
        ("did_change", PassExecutionError),
        ("get_preserved_analyses", PassExecutionError),
    ],
)
def test_each_hook_failure_raises_the_cores_error(
    hook: str, error_class: type[Exception]
) -> None:
    """Test the class, message, attributes and cause of each hook's failure."""
    raised = ValueError(f"{hook} broke")

    class FailingHookPass(CompilerPass[int, int]):
        @override
        def validate_input(self, ir: int) -> None:
            if hook == "validate_input":
                raise raised

        @override
        def should_run(self, ir: int) -> bool:
            if hook == "should_run":
                raise raised
            return hook != "get_noop_output"

        @override
        def get_noop_output(self, ir: int) -> int:
            raise raised

        @override
        def run_pass(self, ir: int) -> int:
            if hook == "run_pass":
                raise raised
            return ir

        @override
        def validate_output(self, input_ir: int, output: int) -> None:
            if hook == "validate_output":
                raise raised

        @override
        def did_change(self, input_ir: int, output: int) -> bool:
            if hook == "did_change":
                raise raised
            return False

        @override
        def get_preserved_analyses(
            self, input_ir: int, output: int, *, changed: bool
        ) -> PreservedAnalyses:
            if hook == "get_preserved_analyses":
                raise raised
            return PreservedAnalyses.all()

    with pytest.raises(error_class) as excinfo:
        FailingHookPass().execute(0)

    error = excinfo.value
    assert type(error) is error_class
    assert str(error) == f'pass "FailingHookPass" failed in {hook}'
    assert error.hook == hook  # type: ignore[attr-defined]
    assert error.pass_name == "FailingHookPass"  # type: ignore[attr-defined]
    assert error.__cause__ is raised
    assert error.records == ()  # type: ignore[attr-defined]
    (diagnostic,) = error.diagnostics  # type: ignore[attr-defined]
    assert diagnostic.level == DiagnosticLevel.ERROR
    assert diagnostic.message_text == (
        f'pass "FailingHookPass" failed in {hook}: ValueError: {hook} broke'
    )


@pytest.mark.parametrize("in_pipeline", [False, True])
def test_keyboard_interrupt_propagates_unwrapped(in_pipeline: bool) -> None:
    """Test an exception that is not an `Exception` is raised as itself."""
    interrupt = KeyboardInterrupt()

    class InterruptedPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            raise interrupt

    with pytest.raises(KeyboardInterrupt) as excinfo:
        if in_pipeline:
            _run_one(InterruptedPass(), 0)
        else:
            InterruptedPass().execute(0)

    assert excinfo.value is interrupt


def test_keyboard_interrupt_stops_a_validation() -> None:
    """Test an interrupt in one validator skips the rest and propagates."""
    later_runs: list[int] = []

    class InterruptingValidator(Validator[int]):
        @override
        def validate(self, ir: int) -> None:
            raise KeyboardInterrupt

    class LaterValidator(Validator[int]):
        @override
        def validate(self, ir: int) -> None:
            later_runs.append(ir)

    manager = ValidationManager[int]()
    manager.add(InterruptingValidator())
    manager.add(LaterValidator())

    with pytest.raises(KeyboardInterrupt):
        manager.validate(0)

    assert later_runs == []


class _InnerFailingPass(CompilerPass[int, int]):
    """Pass whose run fails with a `ValueError`."""

    @override
    def run_pass(self, ir: int) -> int:
        raise ValueError("inner boom")


def test_error_of_a_nested_run_nests_in_the_outer_error() -> None:
    """Test N-S6-2: a nested run's error is the core's `Nested`.

    The outer error names the outer pass and hook, keeps the outer run's
    diagnostics, and has the exception the nested run raised as its
    `__cause__`.
    """
    caught: list[BaseException] = []

    class OuterPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            self.report(DiagnosticLevel.WARNING, "before the nested run")
            try:
                return _InnerFailingPass().execute(ir).output
            except PassExecutionError as error:
                caught.append(error)
                raise

    with pytest.raises(PassExecutionError) as excinfo:
        OuterPass().execute(0)

    error = excinfo.value
    assert str(error) == 'pass "OuterPass" failed in run_pass'
    assert error.pass_name == "OuterPass"
    assert error.hook == "run_pass"
    assert error.__cause__ is caught[0]
    assert str(caught[0]) == 'pass "_InnerFailingPass" failed in run_pass'
    assert isinstance(caught[0].__cause__, ValueError)
    assert [d.message_text for d in error.diagnostics] == [
        "before the nested run",
        'pass "OuterPass" failed in run_pass: '
        'pass "_InnerFailingPass" failed in run_pass: ValueError: inner boom',
    ]


def test_nested_error_under_run_pass_takes_the_inner_class() -> None:
    """Test a nested validation failure under `run_pass` stays a validation error."""

    class RefusingPass(IdentityPass):
        @override
        def validate_input(self, ir: Any) -> None:
            raise ValueError("refused")

    class OuterPass(CompilerPass[Any, Any]):
        @override
        def run_pass(self, ir: Any) -> Any:
            return RefusingPass().execute(ir).output

    class OuterShouldRunPass(IdentityPass):
        @override
        def should_run(self, ir: Any) -> bool:
            RefusingPass().execute(ir)
            return True

    with pytest.raises(PassValidationError) as under_run:
        OuterPass().execute(0)
    with pytest.raises(PassExecutionError) as under_should_run:
        OuterShouldRunPass().execute(0)

    assert isinstance(under_run.value.__cause__, PassValidationError)
    assert under_should_run.value.hook == "should_run"
    assert isinstance(under_should_run.value.__cause__, PassValidationError)


def test_nested_pipeline_error_nests_too() -> None:
    """Test the error of a pipeline run inside a hook nests in the outer run."""
    inner = PassManager[int](name=Identifier("inner"))
    inner.add_pass(_InnerFailingPass())

    class OuterPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            return inner.run(ir).output

    with pytest.raises(PassExecutionError) as excinfo:
        _run_one(OuterPass(), 0)

    assert excinfo.value.pass_name == "OuterPass"
    assert isinstance(excinfo.value.__cause__, PassExecutionError)
    assert excinfo.value.__cause__.pass_name == "_InnerFailingPass"


def test_verification_failure_carries_the_verifiers_report() -> None:
    """Test a verification failure's `report` is the verifier's report."""

    @dataclass
    class CheckedBox(Box):
        pass

    @register_verification(
        CheckedBox, "tests.binding.verify.negative", "Refuses negative boxes."
    )
    class NegativeCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            if isinstance(node, CheckedBox) and node.value < 0:
                self.report(DiagnosticLevel.ERROR, "negative")

    class NegatingPass(CompilerPass[CheckedBox, CheckedBox]):
        @override
        def run_pass(self, ir: CheckedBox) -> CheckedBox:
            return CheckedBox(-1)

    manager = PassManager[CheckedBox]()
    manager.add_pass(NegatingPass())

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(CheckedBox(1))

    error = excinfo.value
    assert error.hook is None
    assert error.pass_name == "NegatingPass"
    assert error.__cause__ is None
    report = error.report
    assert isinstance(report, ValidationReport)
    assert [d.message_text for d in report.diagnostics] == ["negative"]
    (record,) = report.records
    assert isinstance(record, ValidatorRecord)
    assert record.validator_name == "verification"
    assert record.diagnostics[0] is report.diagnostics[0]
    assert error.diagnostics[-1].message_text == str(error)


def test_non_convergence_carries_the_records_of_the_completed_work() -> None:
    """Test a non-converging group's error carries every record, its own too."""

    class FlipPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            return 1 - ir

    group = FixpointPassGroup[int](Identifier("flip"), max_iterations=2)
    group.add_pass(FlipPass())
    manager = PassManager[int]()
    manager.add_pass(IdentityPass())
    manager.add_fixpoint_group(group)

    with pytest.raises(PassExecutionError) as excinfo:
        manager.run(0)

    first, group_record = excinfo.value.records
    assert isinstance(first, PassRunRecord)
    assert first.pass_name == "IdentityPass"
    assert isinstance(group_record, FixpointGroupRecord)
    assert group_record.iterations == 2
    assert excinfo.value.diagnostics == ()


def test_failure_inside_a_group_ends_with_the_partial_group_record() -> None:
    """Test a failure in a group records the iterations begun, the last partial."""
    runs: list[int] = []

    class FailOnSecondPass(CompilerPass[int, int]):
        @override
        def run_pass(self, ir: int) -> int:
            runs.append(ir)
            if len(runs) == 2:
                raise ValueError("second run")
            return ir + 1

    group = FixpointPassGroup[int](Identifier("partial"))
    group.add_pass(IncrementPassInt())
    group.add_pass(FailOnSecondPass())
    manager = PassManager[int]()
    manager.add_fixpoint_group(group)

    with pytest.raises(PassExecutionError) as excinfo:
        manager.run(0)

    (record,) = excinfo.value.records
    assert isinstance(record, FixpointGroupRecord)
    assert [len(iteration.pass_runs) for iteration in record.iteration_records] == [
        2,
        1,
    ]
    assert record.converged is False


class IncrementPassInt(CompilerPass[int, int]):
    """Pass returning the next integer."""

    @override
    def run_pass(self, ir: int) -> int:
        return ir + 1


def test_errors_pickle_with_their_attributes_and_without_the_rust_error() -> None:
    """Test a binding-made error pickles without its Rust error."""
    with pytest.raises(PassExecutionError) as excinfo:
        _InnerFailingPass().execute(0)

    restored = pickle.loads(pickle.dumps(excinfo.value))

    assert type(restored) is PassExecutionError
    assert str(restored) == str(excinfo.value)
    assert restored.hook == "run_pass"
    assert restored.pass_name == "_InnerFailingPass"
    assert restored.diagnostics == excinfo.value.diagnostics
    assert not hasattr(restored, "_rust_error")


def test_user_raised_errors_keep_their_constructors() -> None:
    """Test code can still raise the error classes, with the new keywords."""
    report: ValidationReport[object] = ValidationReport()
    validation = PassValidationError("message", report=report, hook="validate_input")
    execution = PassExecutionError("message", pass_name="p")

    assert validation.report is report
    assert validation.hook == "validate_input"
    assert validation.pass_name is None
    assert execution.pass_name == "p"
    assert execution.diagnostics == ()
    assert execution.records == ()
    assert isinstance(validation, RuntimeError)
    assert isinstance(execution, RuntimeError)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def test_cache_is_keyed_by_identity_not_equality() -> None:
    """Test two equal but distinct frozen boxes are analyzed separately."""

    class ReadBothPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            twin = Box(ir.value)
            self.get_analysis(CountingAnalysis, ir)
            self.get_analysis(CountingAnalysis, twin)
            self.get_analysis(CountingAnalysis, ir)
            self.get_analysis(CountingAnalysis, twin)
            return ir

    _run_one(ReadBothPass(), Box(1))

    assert CountingAnalysis.runs == 2


def test_merge_only_transfer_keeps_the_outputs_own_result() -> None:
    """Test W-12: a preserving pass's output keeps a result of its own."""
    replacement = Box(10)
    observed: list[int] = []

    class SeedPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(CountingAnalysis, ir)
            return ir

    class ReplacePreservingPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(CountingAnalysis, replacement)
            return replacement

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            return PreservedAnalyses.all()

    class ReadPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            observed.append(self.get_analysis(CountingAnalysis, ir))
            return ir

    manager = PassManager[Box]()
    manager.add_pass(SeedPass())
    manager.add_pass(ReplacePreservingPass())
    manager.add_pass(ReadPass())
    manager.run(Box(1))

    assert observed == [10]
    assert CountingAnalysis.runs == 2


def test_cache_pins_its_nodes_during_the_run_and_releases_them_after() -> None:
    """Test a cached node lives as long as the run, and no longer."""
    references: list[weakref.ref[Box]] = []
    alive_during_run: list[bool] = []

    class AnalyzeTemporaryPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            temporary = Box(ir.value + 100)
            self.get_analysis(CountingAnalysis, temporary)
            references.append(weakref.ref(temporary))
            return Box(ir.value + 1)

    class CheckPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            gc.collect()
            alive_during_run.append(references[0]() is not None)
            return ir

    manager = PassManager[Box]()
    manager.add_pass(AnalyzeTemporaryPass())
    manager.add_pass(CheckPass())
    manager.run(Box(0))
    gc.collect()

    assert alive_during_run == [True]
    assert references[0]() is None


def test_raising_analysis_caches_nothing() -> None:
    """Test an analysis that raises is computed again on the next request."""
    attempts: list[int] = []

    class FlakyAnalysis(Analysis[Box, int]):
        @override
        def run(self, ir: Box) -> int:
            attempts.append(ir.value)
            if len(attempts) == 1:
                raise ValueError("first attempt")
            return ir.value

    class RetryingPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            with pytest.raises(ValueError, match="first attempt"):
                self.get_analysis(FlakyAnalysis, ir)
            assert self.get_analysis(FlakyAnalysis, ir) == ir.value
            assert self.get_analysis(FlakyAnalysis, ir) == ir.value
            return ir

    _run_one(RetryingPass(), Box(4))

    assert attempts == [4, 4]


def test_get_analysis_refuses_a_type_that_is_not_an_analysis() -> None:
    """Test D-S6-15: `get_analysis` type-checks the analysis type."""
    with pytest.raises(
        TypeError, match="get_analysis analysis_type must be an Analysis subclass"
    ):
        IdentityPass().get_analysis(int, Box(0))  # type: ignore[arg-type]


def test_analysis_manager_view_reads_the_same_cache() -> None:
    """Test the view and `get_analysis` share the run's cache."""

    class ViewPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            view = self.get_analysis_manager()
            assert view is not None
            assert view.get(CountingAnalysis, ir) == ir.value
            assert self.get_analysis(CountingAnalysis, ir) == ir.value
            return ir

    _run_one(ViewPass(), Box(5))

    assert CountingAnalysis.runs == 1


def test_preserved_analyses_are_keyed_by_identifier() -> None:
    """Test D-S6-9: `PreservedAnalyses` names analyses by `Identifier`."""
    name = CountingAnalysis.get_analysis_name()
    other = Identifier("other")
    preserved = PreservedAnalyses.none().preserve(name)

    assert preserved.is_preserved(name) is True
    assert preserved.is_preserved(other) is False
    assert preserved.analysis_names == frozenset({name})
    assert preserved.preserve(name) is preserved
    assert PreservedAnalyses.all().preserve(name) == PreservedAnalyses.all()
    assert PreservedAnalyses.all().analysis_names == frozenset()
    assert PreservedAnalyses(analysis_names=[name]) == preserved
    assert hash(PreservedAnalyses(analysis_names=[name])) == hash(preserved)
    assert preserved != PreservedAnalyses.none()


def test_preserved_analyses_check_their_arguments() -> None:
    """Test the `ValueError` for both fields and the `TypeError`s."""
    with pytest.raises(ValueError, match="analysis_names must be empty"):
        PreservedAnalyses(preserve_all=True, analysis_names=[Identifier("a")])
    with pytest.raises(TypeError, match="preserve_all must be a bool"):
        PreservedAnalyses(preserve_all=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an Identifier"):
        PreservedAnalyses(analysis_names=["name"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="must be an Identifier"):
        PreservedAnalyses.none().is_preserved("name")  # type: ignore[arg-type]


def test_preserved_set_from_a_record_names_the_python_analyses() -> None:
    """Test a set that crossed the core rebuilds its Python identifiers."""
    name = CountingAnalysis.get_analysis_name()

    class PreservingPass(IncrementPass):
        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            return PreservedAnalyses.none().preserve(name)

    (record,) = _run_one(PreservingPass(), Box(0)).records
    assert isinstance(record, PassRunRecord)

    assert record.preserved_analyses.analysis_names == frozenset({name})
    assert record.preserved_analyses.is_preserved(name) is True


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_group_changed_after_it_was_added_runs_as_it_is_now() -> None:
    """Test D-S6-11: a run reads the current passes of each group."""
    group = FixpointPassGroup[int](
        Identifier("late"), max_iterations=1, fail_on_non_convergence=False
    )
    manager = PassManager[int]()
    manager.add_fixpoint_group(group)
    group.add_pass(IncrementPassInt())

    result = manager.run(0)

    assert result.output == 1
    assert [run.pass_name for run in result.pass_runs()] == ["IncrementPassInt"]


def test_one_pass_added_twice_runs_twice() -> None:
    """Test the same pass object added twice runs once per addition."""
    compiler_pass = IncrementPassInt()
    manager = PassManager[int]()
    manager.add_pass(compiler_pass)
    manager.add_pass(compiler_pass)

    result = manager.run(0)

    assert result.output == 2
    assert result.run_count() == 2


def test_reentrant_run_of_the_same_pipeline_inside_a_hook() -> None:
    """Test a hook can run the pipeline it runs in, with its own cache."""
    manager = PassManager[Box]()

    class ReentrantPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(CountingAnalysis, ir)
            if ir.value < 2:
                return manager.run(Box(ir.value + 1)).output
            return ir

    manager.add_pass(ReentrantPass())
    manager.set_verifier(None)

    assert manager.run(Box(0)).output == Box(2)
    assert CountingAnalysis.runs == 3


def test_skipped_runs_are_recorded_and_not_counted() -> None:
    """Test `skipped`, `pass_runs()` and `run_count()` of a pipeline result."""

    class SkippedPass(IdentityPass):
        @override
        def should_run(self, ir: Any) -> bool:
            return False

        @override
        def get_noop_output(self, ir: Any) -> Any:
            return ir

    group = FixpointPassGroup[int](Identifier("group"))
    group.add_pass(SkippedPass())
    manager = PassManager[int]()
    manager.add_pass(IdentityPass())
    manager.add_fixpoint_group(group)
    manager.add_pass(SkippedPass())

    result = manager.run(0)

    assert [run.skipped for run in result.pass_runs()] == [False, True, True]
    assert result.run_count() == 1


def test_pipelines_type_check_their_arguments() -> None:
    """Test D-S6-15: the pipelines refuse arguments of the wrong type."""
    manager = PassManager[int]()
    group = FixpointPassGroup[int](Identifier("g"))

    with pytest.raises(
        TypeError,
        match=(
            r"^PassManager\.add_pass compiler_pass must be a CompilerPass, "
            r"got int\.$"
        ),
    ):
        manager.add_pass(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a FixpointPassGroup, got str"):
        manager.add_fixpoint_group("g")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a ValidationManager or None"):
        manager.set_verifier(IdentityPass())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a CompilerPass, got int"):
        group.add_pass(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name must be an Identifier, got str"):
        PassManager[int](name="pipeline")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="name must be an Identifier, got str"):
        FixpointPassGroup[int]("group")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r'"max_iterations" must be >= 1\.'):
        FixpointPassGroup[int](Identifier("g"), max_iterations=0)
    with pytest.raises(TypeError, match="max_iterations must be an int"):
        FixpointPassGroup[int](Identifier("g"), max_iterations=1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fail_on_non_convergence must be a bool"):
        FixpointPassGroup[int](Identifier("g"), fail_on_non_convergence=1)  # type: ignore[arg-type]


def test_pipeline_defaults_and_group_configuration() -> None:
    """Test the pipelines' default names and the group's configuration."""
    manager = PassManager[int]()
    group = FixpointPassGroup[int](Identifier("g"))

    assert manager.name.name_hint == "pipeline"
    assert manager.get_identifier() is manager.name
    assert group.max_iterations == 10
    assert group.fail_on_non_convergence is True
    assert group.passes == ()
    assert group.get_identifier() is group.name


def test_group_that_does_not_fail_hands_on_its_last_output() -> None:
    """Test a non-converging group with `fail_on_non_convergence=False`."""
    group = FixpointPassGroup[int](
        Identifier("lenient"), max_iterations=3, fail_on_non_convergence=False
    )
    group.add_pass(IncrementPassInt())
    manager = PassManager[int]()
    manager.add_fixpoint_group(group)

    result = manager.run(0)

    assert result.output == 3
    (record,) = result.records
    assert isinstance(record, FixpointGroupRecord)
    assert record.converged is False
    assert record.group_name is group.name


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_set_verifier_with_a_validation_manager_verifies_with_it() -> None:
    """Test a custom verifier runs, sharing the run's analysis cache."""
    verified: list[int] = []

    class ReadingValidator(Validator[Box]):
        @override
        def validate(self, ir: Box) -> None:
            verified.append(self.get_analysis(CountingAnalysis, ir))
            if ir.value > 1:
                self.report(DiagnosticLevel.ERROR, "too large")

    verifier = ValidationManager[Box]()
    verifier.add(ReadingValidator())
    manager = PassManager[Box]()
    manager.add_pass(IncrementPass())
    manager.add_pass(IncrementPass())
    manager.set_verifier(verifier)

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(Box(0))

    assert verified == [0, 1, 2]
    assert CountingAnalysis.runs == 3
    assert excinfo.value.report is not None
    assert excinfo.value.report.records[0].validator_name == "ReadingValidator"
    assert str(excinfo.value) == (
        'verification rejected the output of pass "IncrementPass" (errors: 1)'
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class _WarningValidator(Validator[int]):
    """Validator reporting one warning."""

    @override
    def validate(self, ir: int) -> None:
        self.report(DiagnosticLevel.WARNING, f"warning for {ir}")


class _ErrorCheckPass(CompilerPass[int, None]):
    """Pass run as a check, reporting one error."""

    @override
    def run_pass(self, ir: int) -> None:
        self.report(DiagnosticLevel.ERROR, f"error for {ir}")


def test_validation_runs_validators_and_passes_collect_all() -> None:
    """Test a `Validator` and a pass validator both run into one report."""

    class NamedValidator(_WarningValidator):
        @property
        @override
        def name(self) -> str:
            return "named"

    manager = ValidationManager[int]()
    manager.add(_WarningValidator())
    manager.add(_ErrorCheckPass())
    manager.add(NamedValidator())

    report = manager.validate(7)

    assert [(d.source, d.message_text) for d in report.diagnostics] == [
        ("_WarningValidator", "warning for 7"),
        ("_ErrorCheckPass", "error for 7"),
        ("named", "warning for 7"),
    ]
    assert [record.validator_name for record in report.records] == [
        "_WarningValidator",
        "_ErrorCheckPass",
        "named",
    ]
    for record, diagnostic in zip(report.records, report.diagnostics, strict=True):
        assert record.diagnostics == (diagnostic,)
        assert record.diagnostics[0] is diagnostic


def test_pass_validator_runs_the_check_part_of_the_lifecycle() -> None:
    """Test a pass runs as a check: no `did_change`, no `get_preserved_analyses`."""
    compiler_pass = RecordingPass()
    manager = ValidationManager[int]()
    manager.add(compiler_pass)

    manager.validate(1)

    assert compiler_pass.calls == [
        "validate_input",
        "should_run",
        "run_pass",
        "validate_output",
    ]


def test_validation_manager_refuses_what_is_not_a_validator() -> None:
    """Test `add` type-checks, and a validator's name must be a `str`."""

    class BadNameValidator(_WarningValidator):
        @property
        @override
        def name(self) -> str:
            return 1  # type: ignore[return-value]

    manager = ValidationManager[int]()
    with pytest.raises(
        TypeError,
        match=r"ValidationManager\.add validator must be a Validator or a CompilerPass",
    ):
        manager.add(1)  # type: ignore[arg-type]

    manager.add(BadNameValidator())
    with pytest.raises(TypeError, match="name must be a str, got int"):
        manager.validate(0)


def test_validator_report_outside_a_check_is_only_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a validator's `report` outside a validation records nothing."""
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        _WarningValidator().report(DiagnosticLevel.WARNING, "outside a check")

    assert [record.getMessage() for record in caplog.records] == ["outside a check"]


# ---------------------------------------------------------------------------
# Passes over expressions
# ---------------------------------------------------------------------------


def test_rewrite_rule_applier_in_a_pipeline_keeps_the_input_object() -> None:
    """Test the applier runs in a pipeline, and nothing firing keeps the input."""
    x = Capture("x")
    rule = RewriteRule(
        BinaryExpressionPattern(
            BinaryOperation.ADD, CapturePattern(x), LiteralPattern(0)
        ),
        lambda bindings: bindings[x],
        name="add-zero",
    )
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(Identifier("v")),
        LiteralExpression(0),
    )
    manager = PassManager[Any]()
    manager.add_pass(RewriteRuleApplier([rule]))

    rewritten = manager.run(expression)
    unchanged = manager.run(rewritten.output)

    assert isinstance(rewritten.output, IdentifierExpression)
    (record,) = rewritten.records
    assert isinstance(record, PassRunRecord)
    assert [d.message_text for d in record.diagnostics] == [
        'applied rewrite rule "add-zero"'
    ]
    assert unchanged.output is rewritten.output
    assert unchanged.run_count() == 1


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_failure_diagnostic_is_logged_with_the_hooks_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test D-S6-5: a hook's failure is logged at ERROR with its exception."""
    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        with pytest.raises(PassExecutionError):
            _InnerFailingPass().execute(0)

    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert record.name == f"{_CORE_LOGGER}._InnerFailingPass"
    assert record.getMessage() == (
        'pass "_InnerFailingPass" failed in run_pass: ValueError: inner boom'
    )
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


def test_lifecycle_lines_are_logged_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the entering, skipped and finished lines of a pass run."""

    class SkippedPass(IdentityPass):
        @override
        def should_run(self, ir: Any) -> bool:
            return bool(ir > 0)

        @override
        def get_noop_output(self, ir: Any) -> Any:
            return ir

    with caplog.at_level(logging.DEBUG, logger=_CORE_LOGGER):
        SkippedPass().execute(0)
        SkippedPass().execute(1)

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == f"{_CORE_LOGGER}.SkippedPass"
    ] == [
        "entering (input type=int)",
        "skipped: should_run returned False",
        "entering (input type=int)",
        "finished (changed=False, output type=int, diagnostics=0)",
    ]


def test_pipeline_and_validation_lines_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the pipeline's INFO and DEBUG lines and the validation counts."""
    manager = PassManager[int](name=Identifier("logged"))
    manager.add_pass(IdentityPass())
    validation = ValidationManager[int](name=Identifier("checks"))
    validation.add(_WarningValidator())

    with caplog.at_level(logging.DEBUG):
        manager.run(0)
        validation.validate(0)

    manager_messages = [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == _MANAGER_LOGGER
    ]
    validation_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == _VALIDATION_LOGGER
    ]
    assert manager_messages[0][0] == logging.INFO
    assert manager_messages[0][1].startswith("logged starting (items=1, input id=")
    assert manager_messages[1] == (
        logging.DEBUG,
        "pass IdentityPass finished (changed=False, skipped=False, diagnostics=0, "
        "preserve_all=True, preserved=0)",
    )
    assert manager_messages[2][1].startswith("logged finished (output id=")
    assert validation_messages == [
        "checks starting (validators=1)",
        "checks finished (errors=0, warnings=1, infos=0)",
    ]


# ---------------------------------------------------------------------------
# Pickles and reprs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        PreservedAnalyses.all(),
        PreservedAnalyses.none().preserve(Identifier("kept")),
        PassResult(
            1,
            changed=True,
            diagnostics=(Diagnostic(DiagnosticLevel.INFO, Note("n"), "s"),),
            preserved_analyses=PreservedAnalyses.all(),
            skipped=False,
        ),
        PassRunRecord("p", True, (), PreservedAnalyses.none(), skipped=True),
        FixpointIterationRecord(
            1, False, (PassRunRecord("p", False, (), PreservedAnalyses.all()),)
        ),
        FixpointGroupRecord(
            Identifier("g"), (FixpointIterationRecord(1, False, ()),), True
        ),
        PassManagerResult(2, (PassRunRecord("p", False, (), PreservedAnalyses.all()),)),
        ValidatorRecord("v", True, ()),
    ],
    ids=lambda value: type(value).__name__,
)
def test_values_pickle_as_a_call_of_their_class(value: Any) -> None:
    """Test D-S6-16: the P2 values pickle as a call of their class with fields."""
    restored = pickle.loads(pickle.dumps(value))

    assert type(restored) is type(value)
    assert restored == value
    assert hash(restored) == hash(value)


def test_records_print_as_dataclasses() -> None:
    """Test the records' reprs are the dataclass ones."""
    record = PassRunRecord("p", False, (), PreservedAnalyses.all())

    assert repr(record) == (
        "PassRunRecord(pass_name='p', changed=False, diagnostics=(), "
        "preserved_analyses=PreservedAnalyses(preserve_all=True, "
        "analysis_names=frozenset()), skipped=False)"
    )
    assert repr(ValidatorRecord("v", False, ())) == (
        "ValidatorRecord(validator_name='v', failed=False, diagnostics=())"
    )


def test_records_compare_by_class_and_fields() -> None:
    """Test equality needs the same class, and an unhashable field fails `hash`."""
    record = PassRunRecord("p", False, (), PreservedAnalyses.all())

    assert record == PassRunRecord("p", False, (), PreservedAnalyses.all())
    assert record != PassRunRecord(
        "p", False, (), PreservedAnalyses.all(), skipped=True
    )
    assert record != ValidatorRecord("p", False, ())
    with pytest.raises(TypeError):
        hash(PassResult([1], changed=False))


def test_records_type_check_their_fields() -> None:
    """Test D-S6-15: the records refuse fields of the wrong type."""
    with pytest.raises(TypeError, match="PassResult changed must be a bool, got int"):
        PassResult(0, changed=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="diagnostics must be Diagnostic instances"):
        PassRunRecord("p", False, ("text",), PreservedAnalyses.all())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a PreservedAnalyses"):
        PassRunRecord("p", False, (), "all")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pass_runs must be PassRunRecord instances"):
        FixpointIterationRecord(1, False, (1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="group_name must be an Identifier"):
        FixpointGroupRecord("g", (), True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PassRunRecord or FixpointGroupRecord"):
        PassManagerResult(0, (1,))  # type: ignore[arg-type]


def test_pass_instance_pickles_through_its_dict() -> None:
    """Test a pass pickles its `__dict__`, and the base keeps no state then."""
    compiler_pass = RecordingPass(skip=True)
    compiler_pass.execute(1)

    restored = pickle.loads(pickle.dumps(compiler_pass))

    assert type(restored) is RecordingPass
    assert restored.skip is True
    assert restored.calls == compiler_pass.calls
    assert restored.diagnostics == ()


def test_managers_do_not_pickle() -> None:
    """Test D-S6-16: the managers and the analysis view refuse pickling."""
    views: list[AnalysisManager[Any]] = []

    class ViewPass(IdentityPass):
        @override
        def run_pass(self, ir: Any) -> Any:
            view = self.get_analysis_manager()
            assert view is not None
            views.append(view)
            return ir

    _run_one(ViewPass(), 0)

    for value in (
        PassManager[int](),
        FixpointPassGroup[int](Identifier("g")),
        ValidationManager[int](),
        views[0],
    ):
        with pytest.raises(TypeError, match="pickle"):
            pickle.dumps(value)
