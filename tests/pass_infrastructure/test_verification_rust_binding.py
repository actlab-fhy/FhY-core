"""Interface tests for the verification registry's Rust binding.

The verification registry runs on the Rust core's
``fhy_core::pass::VerificationRegistry``, held in the extension's module
state. ``test_verification.py`` tests the Python API's behavior; this
suite covers what the binding adds over the core: the ``_rs`` functions
and the module state, the typed arguments, the lineage read from
``__mro__``, pass classes whose construction fails, interrupts, the
references the registry keeps, threads, logging, and
``VerifiableMixin``'s calls.
"""

import gc
import logging
import threading
import weakref
from dataclasses import dataclass
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel, ValidationReport
from fhy_core.pass_infrastructure import (
    CompilerPass,
    PassManager,
    PassRegistrationError,
    PassValidationError,
    ValidatorRecord,
    VerificationRegistry,
    register_pass,
    register_verification,
    run_verification,
)
from fhy_core.traits import FrozenMixin, VerifiableMixin
from fhy_core.utils.override import override

_THREAD_COUNT = 8
_PASSES_PER_THREAD = 20


@dataclass
class _Ir(FrozenMixin):
    """A frozen IR whose test subclasses key the registrations."""

    value: int = 0

    def __post_init__(self) -> None:
        self.freeze()


def _build_ir_type() -> type[_Ir]:
    """Return a new IR class, so each test's registrations are its own."""

    @dataclass
    class _FreshIr(_Ir):
        pass

    return _FreshIr


def _build_check(
    name: str, *, message: str | None = None
) -> type[CompilerPass[Any, None]]:
    """Return an unregistered check named `name` that reports `message`."""

    class _Check(CompilerPass[Any, None]):
        @classmethod
        @override
        def get_pass_name(cls) -> str:
            return name

        @override
        def run_pass(self, ir: Any) -> None:
            _ = ir
            if message is not None:
                self.report(DiagnosticLevel.ERROR, message)

    return _Check


def _build_raising_check(
    name: str, error: BaseException
) -> type[CompilerPass[Any, None]]:
    """Return an unregistered check named `name` whose constructor raises."""

    class _RaisingCheck(CompilerPass[Any, None]):
        def __init__(self) -> None:
            raise error

        @classmethod
        @override
        def get_pass_name(cls) -> str:
            return name

        @override
        def run_pass(self, ir: Any) -> None:
            _ = ir

    return _RaisingCheck


def _build_identity_pipeline(name: str) -> PassManager[Any]:
    """Return a pipeline of one pass named `name` that returns its input."""

    @register_pass(name, "Identity pass of the verification binding tests.")
    class _Identity(CompilerPass[Any, Any]):
        @override
        def run_pass(self, ir: Any) -> Any:
            return ir

    manager = PassManager[Any]()
    manager.add_pass(_Identity())
    return manager


def _record_names(report: ValidationReport[ValidatorRecord]) -> list[str]:
    """Return the names of `report`'s records, in order."""
    return [record.validator_name for record in report.records]


# ---------------------------------------------------------------------------
# The functions and the module state
# ---------------------------------------------------------------------------


def test_register_verification_pass_returns_whether_the_registration_is_new() -> None:
    """Test `_rs.register_verification_pass` reports new and repeated pairs."""
    ir_type = _build_ir_type()
    check = _build_check("tests.vrb.new")

    assert _rs.register_verification_pass(ir_type, check) is True
    assert _rs.register_verification_pass(ir_type, check) is False
    assert _rs.get_verification_passes_for(ir_type) == (check,)


def test_classmethods_and_functions_share_one_registry() -> None:
    """Test a registration through either spelling is seen through the other."""
    ir_type = _build_ir_type()
    first = _build_check("tests.vrb.shared.first")
    second = _build_check("tests.vrb.shared.second")

    _rs.register_verification_pass(ir_type, first)
    VerificationRegistry.register(ir_type, second)

    assert VerificationRegistry.get_passes_for(ir_type) == (first, second)
    assert _rs.get_verification_passes_for(ir_type) == (first, second)


def test_the_registry_lives_in_the_extensions_module_state() -> None:
    """Test the one registry is a private attribute of `fhy_core._rs`."""
    state = _rs._verification_registry  # type: ignore[attr-defined]  # test: module state

    assert type(state).__name__ == "VerificationRegistryState"
    assert type(state).__module__ == "fhy_core._rs"
    assert "VerificationRegistryState" not in dir(_rs)


def test_lookups_return_the_registered_class_objects() -> None:
    """Test `get_passes_for` returns the classes themselves, in a tuple."""
    ir_type = _build_ir_type()
    check = _build_check("tests.vrb.identity")
    VerificationRegistry.register(ir_type, check)

    (found,) = VerificationRegistry.get_passes_for(ir_type)

    assert found is check


# ---------------------------------------------------------------------------
# Typed arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ir_type", [3, "Box", None, _Ir()])
def test_a_non_type_ir_type_is_refused(ir_type: object) -> None:
    """Test a non-type `ir_type` raises `TypeError` from each entry point."""
    check = _build_check("tests.vrb.non_type")

    with pytest.raises(
        TypeError, match=r"^register_verification_pass ir_type must be a type"
    ):
        VerificationRegistry.register(ir_type, check)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match=r"^get_verification_passes_for ir_type must be a type"
    ):
        VerificationRegistry.get_passes_for(ir_type)  # type: ignore[arg-type]


def test_a_refused_registration_stores_nothing() -> None:
    """Test a registration refused for its pass class leaves the type empty."""
    ir_type = _build_ir_type()

    class _NotAPass:
        pass

    with pytest.raises(PassRegistrationError):
        VerificationRegistry.register(ir_type, _NotAPass)  # type: ignore[arg-type]

    assert VerificationRegistry.get_passes_for(ir_type) == ()


def test_a_non_compiler_pass_class_keeps_the_registration_error() -> None:
    """Test a class that is not a `CompilerPass` keeps today's error and text."""
    ir_type = _build_ir_type()

    class _NotAPass:
        pass

    with pytest.raises(PassRegistrationError) as excinfo:
        _rs.register_verification_pass(ir_type, _NotAPass)  # type: ignore[arg-type]

    assert str(excinfo.value) == (
        "Cannot register non-CompilerPass type as a verification pass: "
        f"{_NotAPass.__qualname__}."
    )


def test_a_non_class_pass_is_refused_with_its_repr() -> None:
    """Test an object that is not a class is named by its `repr`."""
    with pytest.raises(PassRegistrationError) as excinfo:
        _rs.register_verification_pass(_build_ir_type(), 42)  # type: ignore[arg-type]

    assert str(excinfo.value) == (
        "Cannot register non-CompilerPass type as a verification pass: 42."
    )


# ---------------------------------------------------------------------------
# The lineage
# ---------------------------------------------------------------------------


def test_the_lineage_is_the_reversed_mro_of_a_diamond() -> None:
    """Test a diamond hierarchy is looked up in reversed `__mro__` order."""
    root = _build_ir_type()

    class _Left(root):  # type: ignore[misc,valid-type]
        pass

    class _Right(root):  # type: ignore[misc,valid-type]
        pass

    class _Joined(_Left, _Right):
        pass

    checks = {
        ir_type: _build_check(f"tests.vrb.diamond.{ir_type.__name__}")
        for ir_type in (root, _Left, _Right, _Joined)
    }
    for ir_type, check in checks.items():
        VerificationRegistry.register(ir_type, check)

    expected = tuple(
        checks[ir_type] for ir_type in reversed(_Joined.__mro__) if ir_type in checks
    )
    assert VerificationRegistry.get_passes_for(_Joined) == expected
    assert expected == (checks[root], checks[_Right], checks[_Left], checks[_Joined])


def test_run_verification_of_an_unregistered_type_is_an_empty_report() -> None:
    """Test verifying an IR with no registrations gives no records."""
    report = run_verification(_build_ir_type()())

    assert isinstance(report, ValidationReport)
    assert report.diagnostics == ()
    assert report.records == ()


def test_run_verification_has_one_record_per_pass() -> None:
    """Test each registered pass is a record, named after it, in order."""
    ir_type = _build_ir_type()
    VerificationRegistry.register(
        ir_type, _build_check("tests.vrb.records.a", message="a")
    )
    VerificationRegistry.register(ir_type, _build_check("tests.vrb.records.b"))

    report = run_verification(ir_type())

    assert _record_names(report) == ["tests.vrb.records.a", "tests.vrb.records.b"]
    assert [d.message_text for d in report.records[0].diagnostics] == ["a"]
    assert report.records[1].diagnostics == ()


# ---------------------------------------------------------------------------
# Pass classes whose construction fails
# ---------------------------------------------------------------------------


def test_a_raising_constructor_fails_only_its_check() -> None:
    """Test a pass class that raises when built is a failed record."""
    ir_type = _build_ir_type()
    VerificationRegistry.register(
        ir_type, _build_raising_check("tests.vrb.raising", ValueError("boom"))
    )
    VerificationRegistry.register(
        ir_type, _build_check("tests.vrb.after_raising", message="still runs")
    )

    report = run_verification(ir_type())

    assert _record_names(report) == ["tests.vrb.raising", "tests.vrb.after_raising"]
    failed, after = report.records
    assert failed.failed is True
    assert after.failed is False
    (synthesized,) = failed.diagnostics
    assert synthesized.level == DiagnosticLevel.ERROR
    assert synthesized.message_text.startswith(
        'validator "tests.vrb.raising" failed without reporting an error: '
    )
    assert "boom" in synthesized.message_text
    assert [d.message_text for d in after.diagnostics] == ["still runs"]


def test_a_raising_constructor_fails_a_pipelines_verification() -> None:
    """Test a pipeline's `verification` record fails for a raising constructor."""
    ir_type = _build_ir_type()
    VerificationRegistry.register(
        ir_type, _build_raising_check("tests.vrb.pipeline_raising", ValueError("boom"))
    )
    manager = _build_identity_pipeline("tests.vrb.pipeline_raising.identity")

    with pytest.raises(PassValidationError) as excinfo:
        manager.run(ir_type())

    report = excinfo.value.report
    assert report is not None
    (record,) = report.records
    assert record.validator_name == "verification"
    assert record.failed is True
    assert any("boom" in d.message_text for d in report.errors())


def test_keyboard_interrupt_from_a_constructor_propagates_from_run_verification() -> (
    None
):
    """Test an interrupt raised while building a check is raised unchanged."""
    ir_type = _build_ir_type()
    interrupt = KeyboardInterrupt()
    VerificationRegistry.register(
        ir_type, _build_raising_check("tests.vrb.interrupt", interrupt)
    )

    with pytest.raises(KeyboardInterrupt) as excinfo:
        run_verification(ir_type())

    assert excinfo.value is interrupt


def test_keyboard_interrupt_from_a_constructor_propagates_from_a_pipeline() -> None:
    """Test an interrupt raised while a pipeline verifies is raised unchanged."""
    ir_type = _build_ir_type()
    VerificationRegistry.register(
        ir_type,
        _build_raising_check("tests.vrb.pipeline_interrupt", KeyboardInterrupt()),
    )
    manager = _build_identity_pipeline("tests.vrb.pipeline_interrupt.identity")

    with pytest.raises(KeyboardInterrupt):
        manager.run(ir_type())


# ---------------------------------------------------------------------------
# References, threads and logging
# ---------------------------------------------------------------------------


def test_the_registry_keeps_registered_types_and_classes_alive() -> None:
    """Test registered objects outlive their other references, as with a dict."""
    ir_type = _build_ir_type()
    check = _build_check("tests.vrb.alive")
    VerificationRegistry.register(ir_type, check)
    weak_ir_type = weakref.ref(ir_type)
    weak_check = weakref.ref(check)

    del ir_type, check
    gc.collect()

    assert weak_ir_type() is not None
    assert weak_check() is not None


def test_concurrent_registrations_keep_every_registration_in_order() -> None:
    """Test threads registering at once lose nothing, with a verification running."""
    ir_type = _build_ir_type()
    checks = [
        [
            _build_check(f"tests.vrb.thread.{thread}.{index}")
            for index in range(_PASSES_PER_THREAD)
        ]
        for thread in range(_THREAD_COUNT)
    ]
    barrier = threading.Barrier(_THREAD_COUNT + 1)
    reports: list[ValidationReport[ValidatorRecord]] = []

    def register_all(thread_checks: list[type[CompilerPass[Any, None]]]) -> None:
        barrier.wait()
        for check in thread_checks:
            VerificationRegistry.register(ir_type, check)

    def verify_repeatedly() -> None:
        barrier.wait()
        for _ in range(_PASSES_PER_THREAD):
            reports.append(run_verification(ir_type()))

    threads = [threading.Thread(target=register_all, args=(c,)) for c in checks]
    threads.append(threading.Thread(target=verify_repeatedly))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    found = VerificationRegistry.get_passes_for(ir_type)
    assert len(found) == _THREAD_COUNT * _PASSES_PER_THREAD
    for thread_checks in checks:
        positions = [found.index(check) for check in thread_checks]
        assert positions == sorted(positions)
    assert len(reports) == _PASSES_PER_THREAD


def test_register_logs_new_and_repeated_registrations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the DEBUG lines of a new and of an idempotent registration."""
    ir_type = _build_ir_type()
    check = _build_check("tests.vrb.logging")

    with caplog.at_level(logging.DEBUG):
        VerificationRegistry.register(ir_type, check)
        VerificationRegistry.register(ir_type, check)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "verification pass" in record.getMessage()
    ]
    assert messages == [
        f"registered verification pass {check.__qualname__} for {ir_type.__qualname__}",
        f"verification pass {check.__qualname__} already registered for "
        f"{ir_type.__qualname__} (idempotent)",
    ]


# ---------------------------------------------------------------------------
# VerifiableMixin over `_rs`
# ---------------------------------------------------------------------------


def test_verifiable_mixin_asks_rs_on_its_first_instantiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the abstractness check calls `_rs.get_verification_passes_for`."""
    asked: list[type] = []

    class _Node(VerifiableMixin):
        pass

    def _report_one_pass(ir_type: type) -> tuple[type, ...]:
        asked.append(ir_type)
        return (object,)

    monkeypatch.setattr(_rs, "get_verification_passes_for", _report_one_pass)

    _Node()
    _Node()

    assert asked == [_Node]


def test_verifiable_mixin_verify_returns_the_rs_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the default `verify` is `_rs.run_verification` of the object."""
    sentinel: ValidationReport[object] = ValidationReport()
    verified: list[object] = []

    class _Node(VerifiableMixin):
        pass

    @register_verification(_Node, "tests.vrb.mixin_verify", "Makes _Node instantiable.")
    class _NodeCheck(CompilerPass[Any, None]):
        @override
        def run_pass(self, ir: Any) -> None:
            _ = ir

    _ = _NodeCheck
    node = _Node()

    def _run(ir: object) -> ValidationReport[object]:
        verified.append(ir)
        return sentinel

    monkeypatch.setattr(_rs, "run_verification", _run)

    assert node.verify() is sentinel
    assert verified == [node]
