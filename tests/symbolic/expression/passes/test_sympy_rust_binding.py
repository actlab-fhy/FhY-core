"""Interface tests of the SymPy backend over the Rust core.

The behavior of the lowering, the simplification and its workarounds, and
the lifting is specified by the Rust stories in
``rust/fhy-core/tests/it/solver/sympy_*``. This suite covers what the binding
adds: the class and its registration, the native path through a ``Solver``,
the SymPy-level methods behind the bridge's functions, the Python exceptions
of the core's errors, pickles, the adapter behind ``SolverBackend.SYMPY``,
and threads.
"""

import logging
import pickle
import re
import subprocess
import sys
import textwrap
import threading
from types import FrameType
from typing import Any

import pytest

pytest.importorskip("sympy")

import sympy  # type: ignore[import-untyped]

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic import solver as solver_module
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    ComplexInfinityLiftError,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    NativeConstantBindingError,
    NonBooleanLogicalOperandError,
    PartialPiecewiseError,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    get_native_constant_identifier,
    piecewise,
    substitute_sympy_expression_variables,
)
from fhy_core.symbolic.expression.passes import sympy as sympy_bridge
from fhy_core.symbolic.expression.passes.sympy import SympySimplifier
from fhy_core.symbolic.solver import (
    Simplifier,
    Solver,
    SolverBackend,
    get_default_solver,
    is_backend_available,
    simplify_expression,
)

pytestmark = pytest.mark.sympy


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.fixture()
def x() -> Identifier:
    """Return an identifier."""
    return Identifier("x")


# =============================================================================
# The class
# =============================================================================


def test_sympy_simplifier_is_a_native_registered_simplifier() -> None:
    """Test the backend extends the base, is a `Simplifier`, and is named sympy."""
    backend = SympySimplifier()

    assert SympySimplifier is _rs.SympySimplifier
    assert isinstance(backend, _rs.SimplifierBase)
    assert isinstance(backend, Simplifier)
    assert backend.name == "sympy"
    assert repr(backend) == "SympySimplifier()"


def test_sympy_simplifier_is_frozen_and_final() -> None:
    """Test the backend refuses attributes and subclasses."""
    backend = SympySimplifier()

    with pytest.raises(AttributeError):
        backend.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        type("Sub", (SympySimplifier,), {})


def test_sympy_simplifier_pickles_as_a_new_backend() -> None:
    """Test a pickled backend loads as another backend of the class."""
    restored = pickle.loads(pickle.dumps(SympySimplifier()))

    assert isinstance(restored, SympySimplifier)


@pytest.mark.subprocess
def test_constructing_the_backend_imports_nothing_and_its_question_reports_sympy() -> (
    None
):
    """Test a backend constructs without SymPy, and its question names the extra."""
    completed = _run_python(
        """
        import sys
        sys.modules["sympy"] = None
        from fhy_core import _rs
        from fhy_core.identifier import Identifier
        from fhy_core.symbolic.expression import IdentifierExpression
        from fhy_core.symbolic.solver import SolverBackendUnavailableError
        backend = _rs.SympySimplifier()
        try:
            backend.simplify(IdentifierExpression(Identifier("x")) < 1)
        except SolverBackendUnavailableError as error:
            print(isinstance(error.__cause__, ImportError))
            print(error)
        """
    )

    assert completed.returncode == 0, completed.stderr
    cause, message = completed.stdout.splitlines()
    assert cause == "True"
    assert "fhy_core[sympy]" in message


@pytest.mark.subprocess
def test_missing_sympy_reports_unavailable() -> None:
    """Test a missing SymPy fails ``load``, and a later load succeeds.

    This needs a fresh process, so it runs here rather than in the Rust
    tests: the backend keeps no failed load, so once SymPy imports, the
    same backend loads and simplifies.
    """
    completed = _run_python(
        """
        import sys
        sys.modules["sympy"] = None
        from fhy_core import _rs
        from fhy_core.symbolic.expression import LiteralExpression
        from fhy_core.symbolic.solver import SolverBackendUnavailableError
        backend = _rs.SympySimplifier()
        try:
            backend.load()
        except SolverBackendUnavailableError as error:
            print(isinstance(error.__cause__, ImportError))
        del sys.modules["sympy"]
        backend.load()
        simplified = backend.simplify(LiteralExpression(1))
        print(simplified.is_structurally_equivalent(LiteralExpression(1)))
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["True", "True"]


# =============================================================================
# The native path
# =============================================================================


def test_solver_holding_the_backend_runs_no_python_backend(x: Identifier) -> None:
    """Test a `Solver` simplifies through the backend with no bridge frame."""
    solver = Solver(simplifier=SympySimplifier())
    bridge_frames: list[str] = []

    def profile(frame: FrameType, event: str, _: Any) -> None:
        if event == "call" and frame.f_code.co_filename == sympy_bridge.__file__:
            bridge_frames.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        result = solver.simplify_expression(
            IdentifierExpression(x) >= 0, {x: LiteralExpression(3)}
        )
    finally:
        sys.setprofile(None)

    assert result.is_structurally_equivalent(LiteralExpression(True))
    assert bridge_frames == []


def test_simplify_returns_the_input_object_when_nothing_is_simpler(
    x: Identifier,
) -> None:
    """Test a result equal to the input's node is the input's object."""
    expression = IdentifierExpression(x)

    assert SympySimplifier().simplify(expression).is_structurally_equivalent(expression)


def test_simplify_reads_user_constants_from_the_registry(
    function_registry_snapshot: None,
) -> None:
    """Test a user constant lowers to its value through the registry."""
    from fhy_core.symbolic.expression import (  # noqa: PLC0415
        FunctionSort,
        register_native_constant,
    )

    register_native_constant("sympy_answer", FunctionSort.INT, 42)
    answer = IdentifierExpression(get_native_constant_identifier("sympy_answer"))

    result = SympySimplifier().simplify(answer > 41)

    assert result.is_structurally_equivalent(LiteralExpression(True))


# =============================================================================
# The methods behind the bridge's functions
# =============================================================================


def test_methods_agree_with_the_bridge_functions(x: Identifier) -> None:
    """Test `lower`, `lift` and `substitute` agree with the public functions."""
    backend = SympySimplifier()
    expression = IdentifierExpression(x) + 1
    environment = {x: LiteralExpression(2)}

    lowered = backend.lower(expression)

    assert lowered == convert_expression_to_sympy_expression(expression)
    assert backend.lift(lowered).is_structurally_equivalent(
        convert_sympy_expression_to_expression(lowered)
    )
    assert backend.substitute(lowered, environment) == (
        substitute_sympy_expression_variables(lowered, environment)
    )
    assert backend.simplify_object(sympy.Symbol("y_1") * 2 / 2) == sympy.Symbol("y_1")


def test_substitute_symbols_applies_a_mapping_of_symbols() -> None:
    """Test `substitute_symbols` replaces SymPy symbols simultaneously."""
    x_symbol, y_symbol = sympy.Symbol("x_1"), sympy.Symbol("y_2")

    result = SympySimplifier().substitute_symbols(
        x_symbol < y_symbol, {x_symbol: y_symbol, y_symbol: sympy.Integer(5)}
    )

    assert result == (y_symbol < 5)


# =============================================================================
# Errors
# =============================================================================


def test_lowering_failure_of_a_simplification_is_the_lowering_pass_error() -> None:
    """Test a call SymPy has no function for raises the lowering pass's error."""
    expression = call("max", LiteralExpression(1), LiteralExpression(2))

    with pytest.raises(PassExecutionError) as exc_info:
        simplify_expression(expression)

    assert exc_info.value.pass_name == "fhy_core.symbolic.expression.to_sympy"
    assert exc_info.value.hook == "run_pass"
    assert isinstance(exc_info.value.__cause__, TypeError)
    assert "inline_functions" in str(exc_info.value.__cause__)


def test_lifting_failure_of_a_simplification_is_the_lifting_pass_error() -> None:
    """Test a quotient by zero raises the lifting pass's error."""
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(1), LiteralExpression(0)
    )

    with pytest.raises(PassExecutionError) as exc_info:
        simplify_expression(expression)

    assert exc_info.value.pass_name == "fhy_core.symbolic.expression.from_sympy"
    assert isinstance(exc_info.value.__cause__, ComplexInfinityLiftError)


def test_screen_and_bound_constant_errors_are_never_wrapped() -> None:
    """Test the screen's refusal and a bound constant raise their own classes."""
    backend = SympySimplifier()
    ill_typed = LogicalExpression(
        LogicalOperation.AND, (LiteralExpression(1), LiteralExpression(True))
    )
    pi = get_native_constant_identifier("pi")

    with pytest.raises(NonBooleanLogicalOperandError):
        backend.lower(ill_typed)
    with pytest.raises(NativeConstantBindingError, match="pi"):
        backend.substitute(sympy.Symbol(f"pi_{pi.id}") > 3, {pi: LiteralExpression(1)})


def test_lifting_errors_are_the_mapped_exceptions_themselves() -> None:
    """Test `lift` raises the mapped exception, which the lifting pass wraps."""
    backend = SympySimplifier()
    partial = sympy.Piecewise((1, sympy.Symbol("flag_0")))

    with pytest.raises(ComplexInfinityLiftError):
        backend.lift(sympy.zoo)
    with pytest.raises(PartialPiecewiseError, match="flag_0"):
        backend.lift(partial)
    with pytest.raises(TypeError, match="unsupported node type"):
        backend.lift(42)
    with pytest.raises(RuntimeError, match="cannot read an identifier"):
        backend.lift(sympy.Symbol("no_identifier"))


def test_lifting_an_implies_logs_the_bridges_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test an `Implies` raises `NotImplementedError` and logs at WARNING."""
    implies = sympy.Implies(sympy.Symbol("a_0"), sympy.Symbol("b_1"))

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(NotImplementedError, match="implies is not supported"),
    ):
        SympySimplifier().lift(implies)

    assert any(
        record.name == "fhy_core.symbolic.expression.passes.sympy"
        and "Implies" in record.getMessage()
        for record in caplog.records
    )


def test_simplify_failure_propagates_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test an exception of `sympy.simplify` itself is raised unwrapped."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(sympy, "simplify", fail)

    with pytest.raises(ZeroDivisionError, match="boom"):
        SympySimplifier().simplify(IdentifierExpression(Identifier("x")) < 1)


def test_keyboard_interrupt_in_simplify_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a `KeyboardInterrupt` in `sympy.simplify` is never wrapped."""

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(sympy, "simplify", interrupt)

    with pytest.raises(KeyboardInterrupt):
        simplify_expression(IdentifierExpression(Identifier("x")) < 1)


# =============================================================================
# Pickles
# =============================================================================


def test_lowered_round_and_piecewise_pickle_within_the_process(x: Identifier) -> None:
    """Test lowered nodes of the prelude's classes pickle and load."""
    reference = IdentifierExpression(x)
    rounded = convert_expression_to_sympy_expression(call("round", reference))
    choice = convert_expression_to_sympy_expression(
        piecewise((reference > 0, LiteralExpression(1)), otherwise=LiteralExpression(2))
    )

    assert pickle.loads(pickle.dumps(rounded)) == rounded
    assert pickle.loads(pickle.dumps(choice)) == choice
    prelude = sys.modules[type(choice).__module__]
    assert type(choice).__module__.startswith("_fhy_core_sympy_0_")
    assert re.fullmatch(r"[0-9a-f]{16}", prelude.__fhy_core_prelude__)
    assert type(choice).__module__.endswith(prelude.__fhy_core_prelude__)


@pytest.mark.subprocess
def test_lowered_piecewise_pickle_loads_where_the_bridge_is_imported(
    x: Identifier,
) -> None:
    """Test a pickle names the prelude module, which importing the bridge loads."""
    choice = convert_expression_to_sympy_expression(
        piecewise(
            (IdentifierExpression(x) > 0, LiteralExpression(1)),
            otherwise=LiteralExpression(2),
        )
    )
    payload = pickle.dumps(choice).hex()

    completed = _run_python(
        f"""
        import pickle
        import fhy_core.symbolic.expression.passes.sympy
        loaded = pickle.loads(bytes.fromhex({payload!r}))
        print(type(loaded).__module__, type(loaded).__name__)
        """
    )

    assert completed.returncode == 0, completed.stderr
    module_name = type(choice).__module__
    assert completed.stdout.split() == [module_name, "ParityOpaquePiecewise"]


# =============================================================================
# Resolution and threads
# =============================================================================


def test_sympy_backend_resolves_to_one_native_backend() -> None:
    """Test `SolverBackend.SYMPY` resolves to one native backend."""
    adapter = solver_module._resolve_adapter(SolverBackend.SYMPY)

    assert isinstance(adapter, SympySimplifier)
    assert solver_module._resolve_adapter(SolverBackend.SYMPY) is adapter
    assert is_backend_available(SolverBackend.SYMPY)


def test_default_solver_holds_a_native_sympy_backend() -> None:
    """Test the default solver's simplifier is the Rust backend."""
    assert isinstance(get_default_solver().simplifier, SympySimplifier)


def test_threads_share_one_backend() -> None:
    """Test concurrent simplifications through one backend agree."""
    solver = Solver(simplifier=SympySimplifier())
    results: dict[int, bool] = {}

    def simplify(index: int) -> None:
        x = Identifier("x")
        result = solver.simplify_expression(
            IdentifierExpression(x) > 3, {x: LiteralExpression(index)}
        )
        assert isinstance(result, LiteralExpression)
        results[index] = bool(result.value)

    threads = [threading.Thread(target=simplify, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {index: index > 3 for index in range(8)}
