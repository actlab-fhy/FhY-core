"""Tests for the verification registry, the analysis, and pipeline verification.

A pipeline verifies its input once and every changed output, blaming the
producing pass, with the passes registered for the IR's type; a standalone
``execute`` never verifies, and ``set_verifier`` replaces the verifier or
turns it off.

Test isolation strategy: every test that touches the verification registry
uses an IR class that is unique to that test, either defined inline or
produced by the ``fresh_box_ir`` fixture. Because the registry is keyed by
IR type and each test's type is distinct, registrations from one test
cannot pollute another.
"""

import gc
import weakref
from dataclasses import dataclass

import pytest

import fhy_core.pass_infrastructure as pass_infra
from fhy_core import _rs
from fhy_core.diagnostic import (
    DiagnosticLevel,
    Note,
    ValidationFailedError,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    Analysis,
    AnalysisVisitablePass,
    CompilerPass,
    PassManager,
    PassRegistrationError,
    PassRunRecord,
    PassValidationError,
    ValidationManager,
    VerificationAnalysis,
    VerificationRegistry,
    register_pass,
    register_verification,
    run_verification,
)
from fhy_core.traits import FrozenMixin, VerifiableMixin, Visitable, VisitableMixin
from fhy_core.utils.override import override

# ---------------------------------------------------------------------------
# Shared test scaffolding.
# ---------------------------------------------------------------------------


@dataclass
class _BoxIR(FrozenMixin, VisitableMixin):
    """Base IR class used by per-test fixtures.

    Each test's fixture produces a fresh subclass for registry isolation;
    test code annotates against this base type to access ``value``.
    """

    value: int = 0

    def __post_init__(self) -> None:
        self.freeze()


@pytest.fixture
def fresh_box_ir() -> type[_BoxIR]:
    """Return a fresh frozen, visitable IR class scoped to this test.

    Using a per-test class makes the verification registry naturally
    isolated: registrations live under this class, and other tests'
    classes do not see them.
    """

    @dataclass
    class _FreshBoxIR(_BoxIR):
        pass

    return _FreshBoxIR


@pytest.fixture
def other_box_ir() -> type[_BoxIR]:
    """Return a second frozen IR class disjoint from ``fresh_box_ir``."""

    @dataclass
    class _OtherBoxIR(_BoxIR):
        pass

    return _OtherBoxIR


def build_clean_pass(
    pass_name: str, ir_type: type
) -> type[AnalysisVisitablePass[Visitable]]:
    """Return a verification pass class that reports nothing."""

    @register_verification(ir_type, pass_name, f"Clean verification pass: {pass_name}")
    class _CleanPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    return _CleanPass


def build_error_pass(
    pass_name: str, message: str, ir_type: type
) -> type[AnalysisVisitablePass[Visitable]]:
    """Return a verification pass that emits one ERROR diagnostic."""

    @register_verification(
        ir_type, pass_name, f"Error-emitting verification pass: {pass_name}"
    )
    class _ErrorPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, message)

    return _ErrorPass


def build_warning_pass(
    pass_name: str, message: str, ir_type: type
) -> type[AnalysisVisitablePass[Visitable]]:
    """Return a verification pass that emits one WARNING diagnostic."""

    @register_verification(
        ir_type, pass_name, f"Warning-emitting verification pass: {pass_name}"
    )
    class _WarningPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.WARNING, message)

    return _WarningPass


def build_crashing_pass(
    pass_name: str, exception_message: str, ir_type: type
) -> type[AnalysisVisitablePass[Visitable]]:
    """Return a verification pass that crashes inside its visitor."""

    @register_verification(
        ir_type, pass_name, f"Crashing verification pass: {pass_name}"
    )
    class _CrashingPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            raise RuntimeError(exception_message)

    return _CrashingPass


# ---------------------------------------------------------------------------
# VerificationRegistry: register / lookup / MRO walk / idempotency.
# ---------------------------------------------------------------------------


def test_registry_register_stores_pass_class_for_type(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that register places a pass class under its IR type key."""

    @register_pass("tests.vr.register_basic", "Pass for register_basic test.")
    class _Pass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    VerificationRegistry.register(fresh_box_ir, _Pass)

    assert VerificationRegistry.get_passes_for(fresh_box_ir) == (_Pass,)


def test_registry_register_returns_none(fresh_box_ir: type[_BoxIR]) -> None:
    """Test that register returns None (no fluent chaining)."""

    @register_pass(
        "tests.vr.register_returns_none", "Pass for register_returns_none test."
    )
    class _Pass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    assert VerificationRegistry.register(fresh_box_ir, _Pass) is None


def test_registry_get_passes_for_unknown_type_returns_empty() -> None:
    """Test that lookup for a type with no registrations returns an empty tuple."""

    class _Unknown:
        pass

    assert VerificationRegistry.get_passes_for(_Unknown) == ()


def test_registry_register_is_idempotent_for_same_pair(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that re-registering the same (type, pass) pair is a no-op."""

    @register_pass("tests.vr.idempotent", "Pass for idempotent registration test.")
    class _Pass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    VerificationRegistry.register(fresh_box_ir, _Pass)
    VerificationRegistry.register(fresh_box_ir, _Pass)

    assert VerificationRegistry.get_passes_for(fresh_box_ir) == (_Pass,)


def test_registry_register_preserves_order_of_multiple_passes(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that multiple registrations preserve insertion order under one type."""

    @register_pass("tests.vr.order_a", "First pass for ordering test.")
    class _PassA(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    @register_pass("tests.vr.order_b", "Second pass for ordering test.")
    class _PassB(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    VerificationRegistry.register(fresh_box_ir, _PassA)
    VerificationRegistry.register(fresh_box_ir, _PassB)

    assert VerificationRegistry.get_passes_for(fresh_box_ir) == (_PassA, _PassB)


def test_registry_get_passes_for_walks_mro_base_first(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that lookup walks the MRO with base-class passes first."""

    @register_pass("tests.vr.mro_base", "Base-class pass for MRO test.")
    class _BasePass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    @register_pass("tests.vr.mro_sub", "Subclass pass for MRO test.")
    class _SubPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    @dataclass
    class _SubBoxIR(fresh_box_ir):  # type: ignore[misc,valid-type]
        pass

    VerificationRegistry.register(fresh_box_ir, _BasePass)
    VerificationRegistry.register(_SubBoxIR, _SubPass)

    assert VerificationRegistry.get_passes_for(_SubBoxIR) == (_BasePass, _SubPass)


def test_registry_get_passes_for_deduplicates_across_mro(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a pass registered against both base and subclass appears once."""

    @register_pass("tests.vr.dedup_shared", "Shared pass registered twice.")
    class _SharedPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    @dataclass
    class _SubBoxIR(fresh_box_ir):  # type: ignore[misc,valid-type]
        pass

    VerificationRegistry.register(fresh_box_ir, _SharedPass)
    VerificationRegistry.register(_SubBoxIR, _SharedPass)

    assert VerificationRegistry.get_passes_for(_SubBoxIR) == (_SharedPass,)


def test_registry_rejects_non_compiler_pass_class(fresh_box_ir: type[_BoxIR]) -> None:
    """Test that register rejects a non-CompilerPass class."""

    class _NotAPass:
        pass

    with pytest.raises(PassRegistrationError, match="CompilerPass"):
        VerificationRegistry.register(fresh_box_ir, _NotAPass)  # type: ignore[arg-type]


def test_registry_isolates_types_in_separate_keys(
    fresh_box_ir: type[_BoxIR], other_box_ir: type
) -> None:
    """Test that registering for one type does not affect lookup of another."""

    @register_pass("tests.vr.isolate_a", "Pass for isolation test.")
    class _BoxPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    VerificationRegistry.register(fresh_box_ir, _BoxPass)

    assert VerificationRegistry.get_passes_for(other_box_ir) == ()


# ---------------------------------------------------------------------------
# VerificationAnalysis: no-arg constructor, stable name, empty / non-empty.
# ---------------------------------------------------------------------------


def test_verification_analysis_constructs_with_no_args() -> None:
    """Test that VerificationAnalysis supports no-arg construction."""
    analysis = VerificationAnalysis()

    assert isinstance(analysis, Analysis)


def test_verification_analysis_name_is_stable_across_instances() -> None:
    """Test that the analysis name is stable and class-derived."""
    name = VerificationAnalysis.get_analysis_name()

    assert isinstance(name, Identifier)
    assert name == VerificationAnalysis.get_analysis_name()
    assert "VerificationAnalysis" in name.name_hint


def test_verification_analysis_returns_empty_report_when_no_passes_registered(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that the analysis returns an empty report when no passes match."""
    report = VerificationAnalysis().run(fresh_box_ir(0))

    assert isinstance(report, ValidationReport)
    assert report.diagnostics == ()
    assert report.has_errors() is False


def test_verification_analysis_aggregates_diagnostics_in_registration_order(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that diagnostics aggregate in pipeline order across registered passes."""
    build_error_pass("tests.va.order.first", "first-error", fresh_box_ir)
    build_warning_pass("tests.va.order.second", "second-warning", fresh_box_ir)
    build_error_pass("tests.va.order.third", "third-error", fresh_box_ir)

    report = VerificationAnalysis().run(fresh_box_ir(1))

    messages = [diagnostic.message_text for diagnostic in report.diagnostics]
    assert messages == ["first-error", "second-warning", "third-error"]


def test_verification_analysis_returns_error_carrying_report_for_failing_ir(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a failing verification surfaces in the returned report."""
    build_error_pass("tests.va.failure", "structural-failure", fresh_box_ir)

    report = VerificationAnalysis().run(fresh_box_ir(2))

    assert report.has_errors() is True
    assert any(
        diagnostic.message_text == "structural-failure"
        for diagnostic in report.errors()
    )


def test_verification_analysis_dispatches_by_concrete_ir_type(
    fresh_box_ir: type[_BoxIR], other_box_ir: type
) -> None:
    """Test that the analysis uses ``type(ir)`` for registry dispatch."""
    build_error_pass("tests.va.dispatch.box", "box-only-error", fresh_box_ir)

    report_box = VerificationAnalysis().run(fresh_box_ir(0))
    report_other = VerificationAnalysis().run(other_box_ir(0))

    assert report_box.has_errors() is True
    assert report_other.has_errors() is False


def test_verification_analysis_wraps_crashed_pass_as_synthetic_error(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a verification pass that raises is captured as an ERROR diagnostic."""
    build_crashing_pass("tests.va.crash", "boom-from-verifier", fresh_box_ir)
    build_error_pass("tests.va.after_crash", "still-runs", fresh_box_ir)

    report = VerificationAnalysis().run(fresh_box_ir(0))

    error_messages = [diagnostic.message_text for diagnostic in report.errors()]
    assert any("boom-from-verifier" in message for message in error_messages)
    assert "still-runs" in error_messages


# ---------------------------------------------------------------------------
# register_verification decorator semantics.
# ---------------------------------------------------------------------------


def test_register_verification_adds_class_to_pass_registry(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that the decorator registers with the global CompilerPass registry."""

    @register_verification(
        fresh_box_ir,
        "tests.rv.in_pass_registry",
        "Pass registered via register_verification.",
    )
    class _Pass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    assert "tests.rv.in_pass_registry" in CompilerPass.get_registered_passes()
    assert (
        CompilerPass.get_registered_passes()["tests.rv.in_pass_registry"].pass_type
        is _Pass
    )


def test_register_verification_adds_class_to_verification_registry(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that the decorator registers with the verification registry."""

    @register_verification(
        fresh_box_ir,
        "tests.rv.in_verification_registry",
        "Pass for registry-add test.",
    )
    class _Pass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    assert _Pass in VerificationRegistry.get_passes_for(fresh_box_ir)


def test_register_verification_leaves_the_class_a_pass_that_never_verifies(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a verification pass runs standalone without verifying its IR.

    A standalone run never verifies, so a verification pass cannot recurse
    into verification.
    """

    @register_verification(
        fresh_box_ir,
        "tests.rv.auto_verify_off",
        "Pass for the standalone-run test.",
    )
    class _Pass(AnalysisVisitablePass[Visitable]):
        invocations = 0

        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            type(self).invocations += 1

    result = _Pass().execute(fresh_box_ir(0))

    assert result.output is None
    assert _Pass.invocations == 1
    assert not hasattr(_Pass, "_auto_verify")


def test_register_verification_rejects_non_compiler_pass(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that non-CompilerPass targets are rejected."""
    with pytest.raises(PassRegistrationError, match="CompilerPass"):

        @register_verification(
            fresh_box_ir, "tests.rv.bad_target", "Decorator applied to non-pass."
        )
        class _NotAPass:  # type: ignore[type-var]  # test: deliberately invalid
            pass


def test_register_verification_rejects_empty_name(fresh_box_ir: type[_BoxIR]) -> None:
    """Test that an empty pass name is rejected."""
    with pytest.raises(PassRegistrationError, match="name"):

        @register_verification(fresh_box_ir, "", "non-empty description")
        class _Pass(AnalysisVisitablePass[Visitable]):
            @override
            def visit_unknown(self, node: Visitable) -> None:
                _ = node


def test_register_verification_rejects_empty_description(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that an empty description is rejected."""
    with pytest.raises(PassRegistrationError, match="description"):

        @register_verification(fresh_box_ir, "tests.rv.empty_desc", "")
        class _Pass(AnalysisVisitablePass[Visitable]):
            @override
            def visit_unknown(self, node: Visitable) -> None:
                _ = node


def test_register_verification_rejects_duplicate_name(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a name already used by a different class is rejected."""

    @register_verification(
        fresh_box_ir, "tests.rv.dup_name", "Original holder of the duplicate name."
    )
    class _Original(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _Original

    with pytest.raises(PassRegistrationError, match="already registered"):

        @register_verification(
            fresh_box_ir, "tests.rv.dup_name", "Different class, same name."
        )
        class _Duplicate(AnalysisVisitablePass[Visitable]):
            @override
            def visit_unknown(self, node: Visitable) -> None:
                _ = node


# ---------------------------------------------------------------------------
# run_verification helper.
# ---------------------------------------------------------------------------


def test_run_verification_returns_report_without_raising_on_errors(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that run_verification returns a report regardless of errors."""
    build_error_pass("tests.rv_helper.error", "irrecoverable", fresh_box_ir)

    report = run_verification(fresh_box_ir(0))

    assert isinstance(report, ValidationReport)
    assert report.has_errors() is True
    assert any(
        diagnostic.message_text == "irrecoverable" for diagnostic in report.errors()
    )


def test_run_verification_returns_empty_report_when_no_passes_registered(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that run_verification returns an empty report when no passes match."""
    report = run_verification(fresh_box_ir(0))

    assert isinstance(report, ValidationReport)
    assert report.diagnostics == ()


# ---------------------------------------------------------------------------
# VerifiableMixin.__new__ abstractness check.
# ---------------------------------------------------------------------------


def test_verifiable_subclass_without_override_or_passes_cannot_be_instantiated() -> (
    None
):
    """Test that instantiation fails when no override and no registered passes exist."""

    class _Unverifiable(VerifiableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.raises(TypeError, match="_Unverifiable"):
        _Unverifiable(0)


def test_verifiable_subclass_with_override_can_be_instantiated_without_passes() -> None:
    """Test that overriding verify bypasses the registry-passes requirement."""

    class _OverridesVerify(VerifiableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

        @override
        def verify(self) -> ValidationReport[object]:
            return ValidationReport()

    instance = _OverridesVerify(7)

    assert instance.value == 7


def test_verifiable_subclass_with_registered_passes_can_be_instantiated() -> None:
    """Test that registering passes makes a subclass instantiable."""

    class _NeedsRegisteredPass(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(
        _NeedsRegisteredPass,
        "tests.vm_new.permits_instantiation",
        "Pass that makes _NeedsRegisteredPass instantiable.",
    )
    class _CheckPass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _CheckPass
    instance = _NeedsRegisteredPass(3)

    assert instance.value == 3


def test_verifiable_subclass_becomes_instantiable_after_late_registration() -> None:
    """Test that a failed instantiation can be retried after registration."""

    class _Late(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.raises(TypeError, match="_Late"):
        _Late(0)

    @register_verification(
        _Late,
        "tests.vm_new.late_registration",
        "Pass registered after first failure.",
    )
    class _LateCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _LateCheck

    instance = _Late(1)

    assert instance.value == 1


def test_verifiable_subclass_caches_positive_instantiation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that a class flagged ``ok`` is not re-checked on later instantiations."""

    class _Cached(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(
        _Cached, "tests.vm_new.positive_cache", "Pass for positive-cache test."
    )
    class _CachedCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _CachedCheck
    _Cached(0)  # primes the cache

    # If subsequent instantiations consulted the registry again, breaking the
    # registry lookup `VerifiableMixin` calls (`_rs`) would cause the test to
    # fail. The positive-result cache means the lookup is never invoked again.
    def _unreachable(_ir_type: type) -> tuple[type[CompilerPass[object, object]], ...]:
        raise AssertionError(
            "_rs.get_verification_passes_for should not be called after the "
            "positive-result cache is primed."
        )

    monkeypatch.setattr(_rs, "get_verification_passes_for", _unreachable)

    follow_up = _Cached(1)

    assert follow_up.value == 1


def test_verifiable_mixin_itself_cannot_be_instantiated() -> None:
    """Test that ``VerifiableMixin`` directly is rejected by the abstractness check."""
    with pytest.raises(TypeError, match="VerifiableMixin"):
        VerifiableMixin()


# ---------------------------------------------------------------------------
# VerifiableMixin.verify default body.
# ---------------------------------------------------------------------------


def test_verify_default_returns_report_from_registered_passes() -> None:
    """Test that verify returns the report produced by registered passes."""

    class _IR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(_IR, "tests.vm_verify.error", "Pass that reports an error.")
    class _ErrorCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, "verify-failure")

    _ = _ErrorCheck
    report = _IR(0).verify()

    assert isinstance(report, ValidationReport)
    assert any(
        diagnostic.message_text == "verify-failure" for diagnostic in report.errors()
    )


def test_verify_default_aggregates_multiple_registered_passes() -> None:
    """Test that diagnostics from multiple passes aggregate in registration order."""

    class _IR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(_IR, "tests.vm_verify.first", "First check.")
    class _First(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.INFO, "info-one")

    @register_verification(_IR, "tests.vm_verify.second", "Second check.")
    class _Second(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.WARNING, "warn-two")

    _ = (_First, _Second)

    report = _IR(0).verify()

    assert [diagnostic.message_text for diagnostic in report.diagnostics] == [
        "info-one",
        "warn-two",
    ]


def test_verify_override_takes_precedence_over_registry() -> None:
    """Test that a subclass override of verify bypasses the registry."""
    sentinel_report: ValidationReport[object] = ValidationReport()

    class _IR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

        @override
        def verify(self) -> ValidationReport[object]:
            return sentinel_report

    @register_verification(
        _IR, "tests.vm_verify.bypassed", "Pass that should not be invoked."
    )
    class _UnusedCheck(AnalysisVisitablePass[Visitable]):
        invoked = False

        @override
        def visit_unknown(self, node: Visitable) -> None:
            type(self).invoked = True
            self.report(DiagnosticLevel.ERROR, "would-fail-if-invoked")

    _ = _UnusedCheck
    report = _IR(0).verify()

    assert report is sentinel_report
    assert _UnusedCheck.invoked is False


def test_verify_default_report_can_be_raised_as_validation_failed_error() -> None:
    """Test that callers can convert the returned report to ValidationFailedError."""

    class _IR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(_IR, "tests.vm_verify.raise", "Verify that raise works.")
    class _RaiseCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, "structurally-broken")

    _ = _RaiseCheck
    report = _IR(0).verify()

    with pytest.raises(ValidationFailedError) as excinfo:
        report.raise_if_failed()

    assert "structurally-broken" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Pipeline verification: the input once, and every changed output.
# ---------------------------------------------------------------------------


def test_standalone_execute_never_verifies(fresh_box_ir: type[_BoxIR]) -> None:
    """Test that a pass executed on its own does not verify its IR.

    Only a pipeline verifies.
    """
    build_error_pass("tests.av.default.fail", "would-fail-in-a-pipeline", fresh_box_ir)

    @register_pass("tests.av.default.identity", "Identity pass run standalone.")
    class _IdentityPass(CompilerPass[object, object]):
        @override
        def run_pass(self, ir: object) -> object:
            return ir

    input_ir = fresh_box_ir(0)

    assert _IdentityPass().execute(input_ir).output is input_ir


def test_pipeline_raises_pass_validation_error_when_its_input_is_malformed(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a pipeline rejects an input that fails verification.

    The input is verified once, blaming the first pass.
    """
    build_error_pass("tests.av.pre.fail", "pre-failure", fresh_box_ir)

    @register_pass("tests.av.pre.identity", "Identity pass for pre-verify test.")
    class _IdentityPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    manager = PassManager[object]()
    manager.add_pass(_IdentityPass())

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(fresh_box_ir(0))

    assert str(excinfo.value) == (
        'verification rejected the input of pass "tests.av.pre.identity" (errors: 1)'
    )
    assert excinfo.value.pass_name == "tests.av.pre.identity"
    assert excinfo.value.records == ()
    assert isinstance(excinfo.value.report, ValidationReport)
    assert excinfo.value.report.has_errors() is True
    assert any(
        diagnostic.message_text == "pre-failure"
        for diagnostic in excinfo.value.report.errors()
    )


def test_pipeline_raises_when_a_pass_produces_malformed_output(
    fresh_box_ir: type[_BoxIR], other_box_ir: type
) -> None:
    """Test that a pipeline rejects a changed output that fails verification."""
    build_error_pass("tests.av.post.check", "post-corruption", other_box_ir)

    @register_pass(
        "tests.av.post.corrupting",
        "Output is well-formed for input verification but fails output verification.",
    )
    class _CorruptingPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return other_box_ir(0)

        @override
        def run_pass(self, ir: object) -> object:
            assert isinstance(ir, fresh_box_ir)
            return other_box_ir(ir.value)

    manager = PassManager[object]()
    manager.add_pass(_CorruptingPass())

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(fresh_box_ir(0))

    assert str(excinfo.value) == (
        "verification rejected the output of pass "
        '"tests.av.post.corrupting" (errors: 1)'
    )
    assert excinfo.value.report is not None
    assert any(
        diagnostic.message_text == "post-corruption"
        for diagnostic in excinfo.value.report.errors()
    )


def test_set_verifier_none_disables_input_and_output_verification(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a pipeline without a verifier skips both checks.

    `set_verifier(None)` turns verification off for the pipeline.
    """
    build_error_pass("tests.av.disabled.fail", "would-fail-if-enabled", fresh_box_ir)

    @register_pass("tests.av.disabled.identity", "Identity pass with auto-verify off.")
    class _NoAutoVerifyPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            assert isinstance(ir, fresh_box_ir)
            return fresh_box_ir(ir.value + 1)

    manager = PassManager[object]()
    manager.add_pass(_NoAutoVerifyPass())
    manager.set_verifier(None)
    input_ir = fresh_box_ir(0)
    result = manager.run(input_ir)

    assert result.output == fresh_box_ir(1)


def test_verification_pass_does_not_recurse_into_verification(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that verification passes run as checks, which never verify."""

    @register_verification(
        fresh_box_ir,
        "tests.av.recurse.check",
        "Verification pass that must not recurse.",
    )
    class _Check(AnalysisVisitablePass[Visitable]):
        invocations = 0

        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            type(self).invocations += 1

    @register_pass(
        "tests.av.recurse.trigger", "Pass that triggers verification on the IR."
    )
    class _TriggerPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    manager = PassManager[object]()
    manager.add_pass(_TriggerPass())
    manager.run(fresh_box_ir(0))

    # The input is verified once; the unchanged output is not.
    assert _Check.invocations == 1


# ---------------------------------------------------------------------------
# Verification is uncached: a changed output is verified again.
# ---------------------------------------------------------------------------


def test_pipeline_verifies_its_input_once_and_no_unchanged_output(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that an unchanged IR is verified once across two pipeline passes."""

    @register_verification(
        fresh_box_ir, "tests.cache.count", "Counting verification pass."
    )
    class _CountingCheck(AnalysisVisitablePass[Visitable]):
        invocations = 0

        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            type(self).invocations += 1

    @register_pass("tests.cache.identity_a", "Identity pass A.")
    class _IdentityA(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    @register_pass("tests.cache.identity_b", "Identity pass B.")
    class _IdentityB(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    manager = PassManager[object]()
    manager.add_pass(_IdentityA())
    manager.add_pass(_IdentityB())
    manager.run(fresh_box_ir(0))

    # The input is verified once; neither pass changes it, so neither
    # output is verified. Total: 1.
    assert _CountingCheck.invocations == 1


def test_pipeline_verifies_a_changed_output_again(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that verification is not cached: a node is verified every time
    a pass returns it as a changed output."""

    @register_verification(
        fresh_box_ir, "tests.cache.again.count", "Counting verification pass."
    )
    class _CountingCheck(AnalysisVisitablePass[Visitable]):
        invocations = 0

        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            type(self).invocations += 1

    original = fresh_box_ir(0)
    replacement = fresh_box_ir(1)

    @register_pass("tests.cache.again.forth", "Returns the replacement.")
    class _ForthPass(CompilerPass[object, object]):
        @override
        def run_pass(self, ir: object) -> object:
            return replacement

    @register_pass("tests.cache.again.back", "Returns the original.")
    class _BackPass(CompilerPass[object, object]):
        @override
        def run_pass(self, ir: object) -> object:
            return original

    manager = PassManager[object]()
    manager.add_pass(_ForthPass())
    manager.add_pass(_BackPass())
    manager.run(original)

    # The input, the replacement, and the original again: 3 checks.
    assert _CountingCheck.invocations == 3


def test_pipeline_verifies_the_output_of_a_changing_pass(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that mutating the IR forces a fresh verification run on the new IR."""

    @register_verification(
        fresh_box_ir,
        "tests.cache.mutate.count",
        "Counting verification pass for mutation.",
    )
    class _CountingCheck(AnalysisVisitablePass[Visitable]):
        invocations = 0

        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            type(self).invocations += 1

    @register_pass("tests.cache.mutate.inc", "Increment the IR value.")
    class _IncrementPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            assert isinstance(ir, fresh_box_ir)
            return fresh_box_ir(ir.value + 1)

    manager = PassManager[object]()
    manager.add_pass(_IncrementPass())
    manager.run(fresh_box_ir(0))

    # Pre-verify of FreshBox(0): 1 run.
    # Post-verify of FreshBox(1): 1 run (different IR identity, fresh entry).
    assert _CountingCheck.invocations == 2


# ---------------------------------------------------------------------------
# User-story: pipeline with a corrupting pass.
# ---------------------------------------------------------------------------


def test_user_story_pipeline_blames_pass_that_produced_invalid_ir(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a corrupting pass surfaces in the report attached to the error."""

    @register_verification(
        fresh_box_ir,
        "tests.story.positive_only",
        "IR values must be non-negative.",
    )
    class _PositiveOnly(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            if isinstance(node, fresh_box_ir) and node.value < 0:
                self.report(DiagnosticLevel.ERROR, f"negative value: {node.value}")

    _ = _PositiveOnly

    @register_pass("tests.story.first_clean", "First pass: clean.")
    class _FirstClean(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            assert isinstance(ir, fresh_box_ir)
            return fresh_box_ir(ir.value + 1)

    @register_pass("tests.story.corrupt", "Middle pass: produces negative output.")
    class _CorruptingPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            _ = ir
            return fresh_box_ir(-100)

    @register_pass("tests.story.last_clean", "Last pass: clean.")
    class _LastClean(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            assert isinstance(ir, fresh_box_ir)
            return fresh_box_ir(ir.value + 1)

    manager = PassManager[object]()
    manager.add_pass(_FirstClean())
    manager.add_pass(_CorruptingPass())
    manager.add_pass(_LastClean())

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(fresh_box_ir(0))

    assert str(excinfo.value) == (
        'verification rejected the output of pass "tests.story.corrupt" (errors: 1)'
    )
    assert excinfo.value.pass_name == "tests.story.corrupt"
    (record,) = excinfo.value.records
    assert isinstance(record, PassRunRecord)
    assert record.pass_name == "tests.story.first_clean"
    assert excinfo.value.report is not None
    error_messages = [
        diagnostic.message_text for diagnostic in excinfo.value.report.errors()
    ]
    assert any("negative value: -100" in message for message in error_messages)


# ---------------------------------------------------------------------------
# Adversarial / chaos cases.
# ---------------------------------------------------------------------------


def test_registering_verification_pass_with_global_pass_name_collision(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that a verification name colliding with an existing pass is rejected."""

    @register_pass("tests.adv.name_collision", "General pass that owns the name first.")
    class _Original(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    _ = _Original

    with pytest.raises(PassRegistrationError, match="already registered"):

        @register_verification(
            fresh_box_ir,
            "tests.adv.name_collision",
            "Verification reusing the name.",
        )
        class _Verifier(AnalysisVisitablePass[Visitable]):
            @override
            def visit_unknown(self, node: Visitable) -> None:
                _ = node


def test_verification_pass_crashing_mid_run_does_not_stop_pipeline(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that one crashing verifier does not prevent the rest from running."""
    build_crashing_pass("tests.adv.mid_crash", "mid-crash-boom", fresh_box_ir)
    build_error_pass("tests.adv.after_crash", "follow-up-error", fresh_box_ir)

    report = VerificationAnalysis().run(fresh_box_ir(0))

    error_messages = [diagnostic.message_text for diagnostic in report.errors()]
    assert any("mid-crash-boom" in message for message in error_messages)
    assert "follow-up-error" in error_messages


def test_registry_mutation_between_passes_does_not_invalidate_cached_results(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test the documented "registration is module-load-time" caching contract.

    Once a run cached a `VerificationAnalysis` result for an IR node,
    appending to the registry does not invalidate it for the rest of the
    run. The next run starts with an empty cache, so the new pass runs.
    """

    @register_verification(
        fresh_box_ir, "tests.adv.registry_mutation.initial", "Initial pass."
    )
    class _Initial(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _Initial
    reports: list[ValidationReport[object]] = []

    def register_late_pass() -> None:
        @register_verification(
            fresh_box_ir,
            "tests.adv.registry_mutation.late",
            "Late-added failing pass.",
        )
        class _Late(AnalysisVisitablePass[Visitable]):
            @override
            def visit_unknown(self, node: Visitable) -> None:
                _ = node
                self.report(DiagnosticLevel.ERROR, "late-failure")

    @register_pass(
        "tests.adv.registry_mutation.reader", "Reads the verification analysis."
    )
    class _ReaderPass(CompilerPass[object, object]):
        registers_late_pass = True

        @override
        def run_pass(self, ir: object) -> object:
            reports.append(self.get_analysis(VerificationAnalysis, ir))
            if type(self).registers_late_pass:
                type(self).registers_late_pass = False
                register_late_pass()
            reports.append(self.get_analysis(VerificationAnalysis, ir))
            return ir

    manager = PassManager[object]()
    manager.add_pass(_ReaderPass())
    manager.set_verifier(None)
    ir = fresh_box_ir(0)

    manager.run(ir)
    # The same IR returns the cached (stale) result; no errors reported.
    assert [report.has_errors() for report in reports] == [False, False]

    # The next run's cache is empty, so the new pass runs.
    manager.run(ir)
    assert reports[2].has_errors() is True


def test_verification_report_format_uses_validation_report_protocol(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that the returned report formats via the standard ValidationReport API."""
    build_error_pass("tests.adv.format", "format-me", fresh_box_ir)

    report = run_verification(fresh_box_ir(0))

    assert "format-me" in report.format()
    assert "[ERROR]" in report.format()


def test_verify_report_supports_partial_equal_like_other_reports() -> None:
    """Test that the report returned by verify still satisfies the report contract."""

    class _IR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(
        _IR, "tests.adv.partial_equal_report", "Verify report PartialEqual contract."
    )
    class _Check(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _Check
    report = _IR(0).verify()

    assert isinstance(report, ValidationReport)


# ---------------------------------------------------------------------------
# Edge cases for the verifiable __new__ check.
# ---------------------------------------------------------------------------


def test_verify_override_inherited_through_intermediate_class_satisfies_check() -> None:
    """Test that an override on a parent class lets a leaf class instantiate."""
    sentinel: ValidationReport[object] = ValidationReport()

    class _ParentWithOverride(VerifiableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

        @override
        def verify(self) -> ValidationReport[object]:
            return sentinel

    class _Leaf(_ParentWithOverride):
        pass

    instance = _Leaf(0)

    assert instance.value == 0
    assert instance.verify() is sentinel


def test_verifiable_subclass_with_passes_from_base_satisfies_check() -> None:
    """Test that passes registered on a base class let a subclass instantiate."""

    class _BaseIR(VerifiableMixin, VisitableMixin):
        def __init__(self, value: int) -> None:
            self.value = value

    @register_verification(
        _BaseIR, "tests.vm_new.base_passes_inherited", "Pass on the base class."
    )
    class _BaseCheck(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node

    _ = _BaseCheck

    class _SubIR(_BaseIR):
        pass

    instance = _SubIR(2)

    assert instance.value == 2


# ---------------------------------------------------------------------------
# Pipeline verification: an empty registry is a silent no-op.
# ---------------------------------------------------------------------------


def test_pipeline_verification_is_silent_when_no_passes_registered_for_ir_type(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that verification is a no-op when the registry has no entries for the IR."""

    @register_pass(
        "tests.av.empty_registry", "Identity pass with no registered verifiers."
    )
    class _IdentityPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    input_ir = fresh_box_ir(7)
    manager = PassManager[object]()
    manager.add_pass(_IdentityPass())
    result = manager.run(input_ir)

    assert result.output == input_ir
    (record,) = result.records
    assert isinstance(record, PassRunRecord)
    assert record.changed is False


# ---------------------------------------------------------------------------
# Cache GC interaction: VerificationAnalysis cached for a run.
# ---------------------------------------------------------------------------


def test_verification_analysis_cache_does_not_pin_ir(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that caching a verification report does not pin the IR after the run."""
    build_clean_pass("tests.cache.gc.clean", fresh_box_ir)

    @register_pass("tests.cache.gc.reader", "Reads the verification analysis.")
    class _ReaderPass(CompilerPass[object, object]):
        @override
        def run_pass(self, ir: object) -> object:
            self.get_analysis(VerificationAnalysis, ir)
            return fresh_box_ir(12)

    manager = PassManager[object]()
    manager.add_pass(_ReaderPass())
    ir = fresh_box_ir(11)
    weak_ir = weakref.ref(ir)

    manager.run(ir)
    assert weak_ir() is not None

    del ir
    gc.collect()

    assert weak_ir() is None


# ---------------------------------------------------------------------------
# PassValidationError carrying an optional report.
# ---------------------------------------------------------------------------


def test_pass_validation_error_carries_optional_report() -> None:
    """Test that PassValidationError can be constructed with or without a report."""
    bare = PassValidationError("bare-message")
    assert bare.report is None

    diagnostics_report: ValidationReport[object] = ValidationReport()
    with_report = PassValidationError("with-message", report=diagnostics_report)
    assert with_report.report is diagnostics_report


def test_pass_validation_error_remains_runtime_error_subclass() -> None:
    """Test that PassValidationError still inherits from RuntimeError."""
    assert issubclass(PassValidationError, RuntimeError)


# ---------------------------------------------------------------------------
# Pipeline-level integration: verification failures interrupt the pipeline.
# ---------------------------------------------------------------------------


def test_pass_manager_surfaces_verification_failure_to_caller(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that verification failures propagate out of PassManager.run."""
    build_error_pass("tests.pm_av.failure", "auto-verify-blocked", fresh_box_ir)

    @register_pass("tests.pm_av.identity", "Identity pass under auto-verify.")
    class _IdentityPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    manager = PassManager[object]()
    manager.add_pass(_IdentityPass())

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(fresh_box_ir(0))

    assert excinfo.value.report is not None
    assert any(
        diagnostic.message_text == "auto-verify-blocked"
        for diagnostic in excinfo.value.report.errors()
    )


def test_pass_manager_continues_with_a_verifier_that_finds_nothing(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that replacing the verifier lets the pipeline complete.

    `set_verifier` takes a `ValidationManager` in place of the registry
    verifier, whose verification pass would block the run.
    """
    build_error_pass(
        "tests.pm_av.continues_failure", "would-block-if-enabled", fresh_box_ir
    )

    @register_pass(
        "tests.pm_av.identity_no_verify", "Identity pass with auto-verify disabled."
    )
    class _IdentityPass(CompilerPass[object, object]):
        @override
        def get_noop_output(self, ir: object) -> object:
            return ir

        @override
        def run_pass(self, ir: object) -> object:
            return ir

    manager = PassManager[object]()
    manager.add_pass(_IdentityPass())
    manager.set_verifier(ValidationManager[object]())
    input_ir = fresh_box_ir(0)
    result = manager.run(input_ir)

    assert result.output == input_ir


# ---------------------------------------------------------------------------
# Sanity: imports and exports are wired correctly.
# ---------------------------------------------------------------------------


def test_verification_module_exports_expected_public_symbols() -> None:
    """Test that the new public API is re-exported from pass_infrastructure."""
    assert pass_infra.VerificationRegistry is VerificationRegistry
    assert pass_infra.VerificationAnalysis is VerificationAnalysis
    assert pass_infra.register_verification is register_verification
    assert pass_infra.run_verification is run_verification


def test_diagnostic_note_construction_round_trips_through_report(
    fresh_box_ir: type[_BoxIR],
) -> None:
    """Test that the Note message attached by a verifier surfaces in the report."""

    @register_verification(
        fresh_box_ir,
        "tests.sanity.note_round_trip",
        "Emit a structured Note diagnostic.",
    )
    class _NotePass(AnalysisVisitablePass[Visitable]):
        @override
        def visit_unknown(self, node: Visitable) -> None:
            _ = node
            self.report(DiagnosticLevel.ERROR, Note("structured-message"))

    _ = _NotePass
    report = run_verification(fresh_box_ir(0))

    assert any(
        diagnostic.message_text == "structured-message"
        for diagnostic in report.errors()
    )
