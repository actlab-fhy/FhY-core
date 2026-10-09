"""Interface tests of the solver over the Rust core.

The behavior of the screens, the lowering and the encodings is specified by
the Rust tests in ``rust/fhy-core/tests/it/solver/``. This suite covers what
the binding adds: the backend classes and how a Python backend is driven,
``SatResult`` and ``SmtScript``, the adapters behind ``SolverBackend``, the
lazy imports, the errors and warnings, the default solver, and threads.
"""

import logging
import pathlib
import pickle
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.symbolic import solver as solver_module
from fhy_core.symbolic.constraint import EquationConstraint, create_constraint_system
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NativeConstantBindingError,
    NativeConstantLoweringError,
    NonBooleanLogicalOperandError,
    get_native_constant_identifier,
    logical_and,
)
from fhy_core.symbolic.expression.errors import UndecidableError
from fhy_core.symbolic.param import create_integer_param_between, create_natural_param
from fhy_core.symbolic.solver import (
    SatResult,
    SatStatus,
    Simplifier,
    SmtLib2ProcessSolver,
    SmtScript,
    SmtSolver,
    Solver,
    SolverBackend,
    SolverBackendError,
    SolverCapabilityError,
    SolverQueryKind,
    check_expression_satisfiability,
    convert_expression_to_smtlib2,
    get_backend_capabilities,
    get_default_solver,
    is_backend_available,
    set_default_solver,
    simplify_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.utils.override import override

from .conftest import mock_identifier

_SOLVER_LOGGER_NAME = "fhy_core.symbolic.solver"

# A fake SMT-LIB2 executable, run with the interpreter, that answers `sat`
# when the script declares an Int and `unsat` otherwise, reports the reason
# `incomplete` when asked, and exits on `(exit)`.
_FAKE_SOLVER_PROGRAM = textwrap.dedent(
    """
    import sys
    declares_an_int = False
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("(declare-const") and line.endswith(" Int)"):
            declares_an_int = True
        elif line == "(check-sat)":
            print("sat" if declares_an_int else "unsat", flush=True)
        elif line == "(get-info :reason-unknown)":
            print('(:reason-unknown "incomplete")', flush=True)
        elif line == "(exit)":
            break
    """
)


def _build_fake_process_solver(program: str = _FAKE_SOLVER_PROGRAM) -> Any:
    return SmtLib2ProcessSolver(sys.executable, ["-c", program])


class _RecordingSmtSolver(SmtSolver):
    """A Python backend answering every check the same way and recording it."""

    def __init__(self, answer: Any) -> None:
        super().__init__()
        self.answer = answer
        self.checks: list[tuple[str, int | None]] = []

    @override
    def check(
        self, script: SmtScript, *, timeout_milliseconds: int | None
    ) -> SatResult:
        self.checks.append((script.text, timeout_milliseconds))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer  # type: ignore[no-any-return]


class _RecordingSimplifier(Simplifier):
    """A Python simplifier returning its input, or a fixed result, and recording."""

    def __init__(self, result: Any = None) -> None:
        super().__init__()
        self.result = result
        self.inputs: list[Expression] = []

    @override
    def simplify(self, expression: Expression) -> Expression:
        self.inputs.append(expression)
        if isinstance(self.result, BaseException):
            raise self.result
        return expression if self.result is None else self.result


@pytest.fixture
def x() -> Identifier:
    """Return an integer variable."""
    return mock_identifier("x", 0)


@pytest.fixture
def restore_default_solver() -> Iterator[None]:
    """Restore the default solver after a test replaces it."""
    original = get_default_solver()
    yield
    set_default_solver(original)


# =============================================================================
# Class structure
# =============================================================================


def test_backend_classes_enforce_their_abstract_hooks() -> None:
    """Test a subclass without its hook cannot be instantiated."""

    class Incomplete(SmtSolver):
        pass

    class IncompleteSimplifier(Simplifier):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]
    with pytest.raises(TypeError, match="abstract"):
        IncompleteSimplifier()  # type: ignore[abstract]


def test_backend_subclass_with_its_own_init_constructs() -> None:
    """Test a backend subclass defining ``__init__`` takes its own arguments."""
    backend = _RecordingSmtSolver(SatResult.SAT)

    assert backend.answer == SatResult.SAT
    assert backend.name == "_RecordingSmtSolver"


def test_backend_subclass_without_an_init_refuses_arguments() -> None:
    """Test the Rust base refuses arguments a subclass does not take."""

    class NoInit(SmtSolver):
        @override
        def check(
            self, script: SmtScript, *, timeout_milliseconds: int | None
        ) -> SatResult:
            return SatResult.SAT

    with pytest.raises(TypeError, match="takes no arguments"):
        NoInit(1)


def test_process_solver_is_a_native_smt_solver() -> None:
    """Test ``SmtLib2ProcessSolver`` extends the base and is a virtual ``SmtSolver``."""
    process = SmtLib2ProcessSolver("/usr/bin/z3", ["-in"])

    assert process.name == "z3"
    assert process.program == "/usr/bin/z3"
    assert process.args == ("-in",)
    assert repr(process) == "SmtLib2ProcessSolver('/usr/bin/z3', ('-in',))"
    assert isinstance(process, _rs.SmtSolverBase)
    assert isinstance(process, SmtSolver)


@pytest.mark.parametrize(
    "arguments",
    [pytest.param((1,), id="program"), pytest.param(("z3", [1]), id="argument")],
)
def test_process_solver_refuses_a_bad_program_or_argument(
    arguments: tuple[Any, ...],
) -> None:
    """Test the program must be a path and every argument a ``str``."""
    with pytest.raises(TypeError, match="SmtLib2ProcessSolver"):
        SmtLib2ProcessSolver(*arguments)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(Solver, id="solver"),
        pytest.param(lambda: SatResult.SAT, id="sat_result"),
        pytest.param(lambda: SmtScript.lower(LiteralExpression(True)), id="script"),
    ],
)
def test_values_are_frozen(value: Any) -> None:
    """Test the solver's values take no new attributes."""
    with pytest.raises(AttributeError):
        value().extra = 1


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"smt_solver": object()}, id="smt_solver"),
        pytest.param(
            {"simplifier": _RecordingSmtSolver(SatResult.SAT)}, id="simplifier"
        ),
    ],
)
def test_solver_refuses_a_backend_of_another_type(arguments: dict[str, Any]) -> None:
    """Test ``Solver`` takes an ``SmtSolver`` and a ``Simplifier`` only."""
    with pytest.raises(TypeError, match=r"must be an? (SmtSolver|Simplifier)"):
        Solver(**arguments)


def test_solver_reports_its_backends_and_capabilities() -> None:
    """Test ``Solver`` exposes its backends and what it can answer."""
    smt = _RecordingSmtSolver(SatResult.SAT)
    simplifier = _RecordingSimplifier()
    solver = Solver(smt_solver=smt, simplifier=simplifier)

    assert solver.smt_solver is smt
    assert solver.simplifier is simplifier
    assert Solver().smt_solver is None
    assert [solver.can_answer(kind) for kind in SolverQueryKind] == [True] * 4
    assert [Solver(smt_solver=smt).can_answer(kind) for kind in SolverQueryKind] == [
        False,
        True,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="not a SolverQueryKind"):
        solver.can_answer("guessing")
    assert repr(Solver()) == "Solver(smt_solver=None, simplifier=None)"


# =============================================================================
# Driving a Python SmtSolver
# =============================================================================


def test_python_smt_solver_receives_the_script_and_timeout_once_per_query(
    x: Identifier,
) -> None:
    """Test the hook is called once, with the script's text and the timeout."""
    backend = _RecordingSmtSolver(SatResult.SAT)
    solver = Solver(smt_solver=backend)

    result = solver.check_expression_satisfiability(
        IdentifierExpression(x) > 0, {x: SymbolType.INT}, timeout_milliseconds=250
    )

    assert result is True
    assert backend.checks == [
        (
            "(set-logic QF_LIA)\n(declare-const |x_0| Int)\n(assert (> |x_0| 0))\n",
            250,
        )
    ]


@pytest.mark.parametrize(
    "answer, expected",
    [
        pytest.param(SatResult.SAT, True, id="sat"),
        pytest.param(SatResult.UNSAT, False, id="unsat"),
        pytest.param(SatResult.unknown("timeout"), None, id="unknown"),
    ],
)
def test_python_smt_solver_result_becomes_the_answer(
    x: Identifier, answer: SatResult, expected: bool | None
) -> None:
    """Test each ``SatResult`` becomes the satisfiability answer."""
    solver = Solver(smt_solver=_RecordingSmtSolver(answer))

    result = solver.check_expression_satisfiability(
        IdentifierExpression(x) > 0, {x: SymbolType.INT}
    )

    assert result is expected


def test_python_smt_solver_of_the_wrong_result_type_raises_type_error(
    x: Identifier,
) -> None:
    """Test a hook returning something other than a ``SatResult`` is refused."""
    solver = Solver(smt_solver=_RecordingSmtSolver(True))

    with pytest.raises(
        TypeError, match=r"_RecordingSmtSolver\.check must return a SatResult, got bool"
    ):
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0, {x: SymbolType.INT}
        )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("backend failure"), id="exception"),
        pytest.param(KeyboardInterrupt(), id="keyboard_interrupt"),
    ],
)
def test_python_smt_solver_exception_propagates_as_the_same_object(
    x: Identifier, error: BaseException
) -> None:
    """Test the hook's exception, a ``KeyboardInterrupt`` included, is raised as is."""
    solver = Solver(smt_solver=_RecordingSmtSolver(error))

    with pytest.raises(type(error)) as exception_info:
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0, {x: SymbolType.INT}
        )

    assert exception_info.value is error


def test_python_smt_solver_can_ask_a_nested_question(x: Identifier) -> None:
    """Test a hook asking another solver a question inside its own check works."""
    inner = Solver(smt_solver=_RecordingSmtSolver(SatResult.UNSAT))

    class Nesting(SmtSolver):
        @override
        def check(
            self, script: SmtScript, *, timeout_milliseconds: int | None
        ) -> SatResult:
            decided = inner.check_expression_satisfiability(
                IdentifierExpression(x) > 1, {x: SymbolType.INT}
            )
            return SatResult.SAT if decided is False else SatResult.UNSAT

    result = Solver(smt_solver=Nesting()).check_expression_satisfiability(
        IdentifierExpression(x) > 0, {x: SymbolType.INT}
    )

    assert result is True


def test_strict_companions_raise_undecidable_error_with_the_backend_reason(
    x: Identifier,
) -> None:
    """Test ``unknown`` raises ``UndecidableError`` naming the backend and reason."""
    solver = Solver(smt_solver=_RecordingSmtSolver(SatResult.unknown("timeout")))

    with pytest.raises(UndecidableError, match="_RecordingSmtSolver") as exception_info:
        solver.assert_expression_implies(
            IdentifierExpression(x) > 0,
            IdentifierExpression(x) > -1,
            {x: SymbolType.INT},
        )

    assert exception_info.value.reason == "timeout"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param(SatResult.UNSAT, True, id="no_counterexample"),
        pytest.param(SatResult.SAT, False, id="a_counterexample"),
        pytest.param(SatResult.unknown("timeout"), None, id="unknown"),
    ],
)
def test_holds_for_all_free_assignments_asks_its_backend_once(
    x: Identifier, answer: SatResult, expected: bool | None
) -> None:
    """Test the universal question is one check for a counterexample.

    With nothing considered, the question is universal validity, so an
    unsatisfiable negation holds. The answer is the very ``True``, ``False``
    or ``None`` object.
    """
    backend = _RecordingSmtSolver(answer)
    solver = Solver(smt_solver=backend)

    result = solver.holds_for_all_free_assignments(
        [], IdentifierExpression(x) * IdentifierExpression(x) >= 0, {x: SymbolType.INT}
    )

    assert result is expected
    assert len(backend.checks) == 1


def test_assert_holds_for_all_free_assignments_raises_undecidable_error(
    x: Identifier,
) -> None:
    """Test the strict companion raises on ``unknown`` and answers otherwise."""
    expression = IdentifierExpression(x) * IdentifierExpression(x) >= 0
    undecided = Solver(smt_solver=_RecordingSmtSolver(SatResult.unknown("timeout")))
    decided = Solver(smt_solver=_RecordingSmtSolver(SatResult.UNSAT))

    with pytest.raises(UndecidableError, match="_RecordingSmtSolver") as exception_info:
        undecided.assert_holds_for_all_free_assignments(
            [], expression, {x: SymbolType.INT}
        )

    assert exception_info.value.reason == "timeout"
    assert (
        decided.assert_holds_for_all_free_assignments(
            [], expression, {x: SymbolType.INT}
        )
        is True
    )


@pytest.mark.parametrize(
    "method",
    ["holds_for_all_free_assignments", "assert_holds_for_all_free_assignments"],
)
def test_holds_for_all_free_assignments_raises_the_backend_s_keyboard_interrupt(
    x: Identifier, method: str
) -> None:
    """Test a backend's ``KeyboardInterrupt`` is raised as the same object."""
    interrupt = KeyboardInterrupt()
    solver = Solver(smt_solver=_RecordingSmtSolver(interrupt))

    with pytest.raises(KeyboardInterrupt) as exception_info:
        getattr(solver, method)([], IdentifierExpression(x) >= 0, {x: SymbolType.INT})

    assert exception_info.value is interrupt


def test_solver_method_refuses_a_bad_timeout(x: Identifier) -> None:
    """Test a ``Solver`` method checks the timeout as the module functions do."""
    solver = Solver(smt_solver=_RecordingSmtSolver(SatResult.SAT))

    with pytest.raises(ValueError, match="positive integer"):
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0,
            {x: SymbolType.INT},
            timeout_milliseconds=True,
        )


def test_solver_method_refuses_a_timeout_of_2_to_the_64_milliseconds(
    x: Identifier,
) -> None:
    """Test a positive timeout of ``2**64`` ms or more names the bound it breaks."""
    backend = _RecordingSmtSolver(SatResult.SAT)
    solver = Solver(smt_solver=backend)

    with pytest.raises(
        ValueError,
        match=r"^timeout_milliseconds must be below 2\*\*64 milliseconds, but got "
        rf"{2**64}\.$",
    ):
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0,
            {x: SymbolType.INT},
            timeout_milliseconds=2**64,
        )
    assert backend.checks == []


# =============================================================================
# Driving a Python Simplifier
# =============================================================================


def test_python_simplifier_receives_the_input_object_when_nothing_is_bound(
    x: Identifier,
) -> None:
    """Test the hook receives, and the caller gets back, the very input object."""
    simplifier = _RecordingSimplifier()
    expression = IdentifierExpression(x) + 0

    result = Solver(simplifier=simplifier).simplify_expression(expression)

    assert simplifier.inputs == [expression]
    assert simplifier.inputs[0] is expression
    assert result is expression


def test_python_simplifier_receives_the_substituted_expression_object(
    x: Identifier,
) -> None:
    """Test the hook's input reuses the environment's value objects."""
    simplifier = _RecordingSimplifier()
    three = LiteralExpression(3)
    expression = IdentifierExpression(x) + 2

    result = Solver(simplifier=simplifier).simplify_expression(expression, {x: three})

    (received,) = simplifier.inputs
    assert isinstance(received, BinaryExpression)
    assert received.left is three
    assert result is received


def test_python_simplifier_result_object_is_returned(x: Identifier) -> None:
    """Test the object the hook returns is the object the caller gets."""
    five = LiteralExpression(5)

    result = Solver(simplifier=_RecordingSimplifier(five)).simplify_expression(
        IdentifierExpression(x) + 2, {x: LiteralExpression(3)}
    )

    assert result is five


def test_python_simplifier_of_the_wrong_result_type_raises_type_error() -> None:
    """Test a hook returning something other than an ``Expression`` is refused."""
    solver = Solver(simplifier=_RecordingSimplifier(5))

    with pytest.raises(TypeError, match=r"simplify must return an Expression, got int"):
        solver.simplify_expression(LiteralExpression(1))


def test_python_simplifier_exception_propagates_as_the_same_object() -> None:
    """Test the hook's exception is raised as is."""
    error = RuntimeError("cannot simplify")

    with pytest.raises(RuntimeError) as exception_info:
        Solver(simplifier=_RecordingSimplifier(error)).simplify_expression(
            LiteralExpression(1)
        )

    assert exception_info.value is error


def test_python_simplifier_can_ask_a_nested_simplification(x: Identifier) -> None:
    """Test a hook simplifying another expression inside its own call works."""
    inner = Solver(simplifier=_RecordingSimplifier())
    seven = LiteralExpression(7)

    class Nesting(Simplifier):
        @override
        def simplify(self, expression: Expression) -> Expression:
            return inner.simplify_expression(IdentifierExpression(x), {x: seven})

    result = Solver(simplifier=Nesting()).simplify_expression(LiteralExpression(1))

    assert result is seven


class _TimeoutReadingSimplifier(Simplifier):
    """A Python simplifier recording the timeouts its context gives."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[tuple[float | None, int | None]] = []

    @override
    def simplify(self, expression: Expression) -> Expression:
        self.timeouts.append((self.context.timeout, self.context.timeout_milliseconds))
        return expression


def test_python_simplifier_reads_the_timeout_from_its_context() -> None:
    """Test a subclass reads the simplification's timeout from ``context``."""
    simplifier = _TimeoutReadingSimplifier()
    solver = Solver(simplifier=simplifier)

    solver.simplify_expression(LiteralExpression(1), timeout_milliseconds=1500)
    solver.simplify_expression(LiteralExpression(1))

    assert simplifier.timeouts == [(1.5, 1500), (None, None)]


def test_python_simplifier_context_is_unbounded_outside_a_simplification() -> None:
    """Test ``context`` has no timeout when ``simplify`` is called directly."""
    simplifier = _TimeoutReadingSimplifier()

    simplifier.simplify(LiteralExpression(1))

    assert simplifier.timeouts == [(None, None)]
    assert repr(simplifier.context) == "SimplifyContext(timeout_milliseconds=None)"


def test_nested_simplification_has_its_own_context(x: Identifier) -> None:
    """Test an inner simplification's timeout does not leak to the outer one."""
    reader = _TimeoutReadingSimplifier()
    inner = Solver(simplifier=reader)
    seen: list[float | None] = []

    class Nesting(Simplifier):
        @override
        def simplify(self, expression: Expression) -> Expression:
            inner.simplify_expression(expression, timeout_milliseconds=10)
            seen.append(self.context.timeout)
            return expression

    Solver(simplifier=Nesting()).simplify_expression(
        LiteralExpression(1), timeout_milliseconds=2000
    )

    assert reader.timeouts == [(0.01, 10)]
    assert seen == [2.0]


@pytest.mark.parametrize("timeout_milliseconds", [0, -1, 1.5, True])
def test_simplification_refuses_a_bad_timeout(timeout_milliseconds: Any) -> None:
    """Test ``simplify_expression`` validates ``timeout_milliseconds``."""
    with pytest.raises(ValueError, match="timeout_milliseconds"):
        Solver(simplifier=_RecordingSimplifier()).simplify_expression(
            LiteralExpression(1), timeout_milliseconds=timeout_milliseconds
        )


def test_simplification_refuses_a_value_that_is_no_expression(x: Identifier) -> None:
    """Test an environment value must be an ``Expression``."""
    with pytest.raises(TypeError, match="environment values must be Expressions"):
        Solver(simplifier=_RecordingSimplifier()).simplify_expression(
            IdentifierExpression(x),
            {x: 3},  # type: ignore[dict-item]
        )


# =============================================================================
# SatResult and SmtScript
# =============================================================================


def test_sat_result_values_compare_hash_print_and_pickle() -> None:
    """Test ``SatResult`` is a value: its members, status, reason, repr and pickles."""
    unknown = SatResult.unknown("timeout")

    assert SatResult.SAT == SatResult.SAT
    assert SatResult.SAT != SatResult.UNSAT
    assert unknown == SatResult.unknown("timeout")
    assert unknown != SatResult.unknown("incomplete")
    assert hash(unknown) == hash(SatResult.unknown("timeout"))
    assert [result.status for result in (SatResult.SAT, SatResult.UNSAT, unknown)] == [
        SatStatus.SAT,
        SatStatus.UNSAT,
        SatStatus.UNKNOWN,
    ]
    assert isinstance(unknown.status, SatStatus)
    assert (SatResult.SAT.reason, unknown.reason) == (None, "timeout")
    assert repr(SatResult.UNSAT) == "SatResult.UNSAT"
    assert repr(unknown) == "SatResult.unknown('timeout')"
    for value in (SatResult.SAT, SatResult.UNSAT, unknown):
        assert pickle.loads(pickle.dumps(value)) == value
    with pytest.raises(TypeError, match="reason must be a str"):
        SatResult.unknown(1)  # type: ignore[arg-type]


def test_smt_script_exposes_its_text_logic_declarations_and_value_sort(
    x: Identifier,
) -> None:
    """Test a script's parts, for a predicate and for a named value."""
    predicate = SmtScript.lower(IdentifierExpression(x) > 0, {x: SymbolType.INT})
    value = SmtScript.lower(IdentifierExpression(x) + 1, {x: SymbolType.REAL})

    assert predicate.logic == "QF_LIA"
    assert predicate.declarations == ((x, "x_0", SymbolType.INT),)
    assert predicate.value_sort is None
    assert str(predicate) == predicate.text
    assert value.value_sort is SymbolType.REAL
    assert value.text == (
        "(set-logic QF_LRA)\n(declare-const |x_0| Real)\n(declare-const value Real)\n"
        "(assert (= value (+ |x_0| 1.0)))\n"
    )
    assert repr(predicate).startswith("SmtScript(logic='QF_LIA', text='(set-logic")


@pytest.mark.parametrize(
    "expression, text",
    [
        pytest.param(
            lambda x: BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(x),
                LiteralExpression(-3),
            ).equals(LiteralExpression(1)),
            "(assert (= (div (- |x_0|) 3) 1))",
            id="floor_division_by_a_negative",
        ),
        pytest.param(
            lambda x: IdentifierExpression(x) ** 4 > 1,
            "(assert (let ((t!1 (* |x_0| |x_0|))) (> (* t!1 t!1) 1)))",
            id="power_by_squaring",
        ),
        pytest.param(
            lambda x: LiteralExpression(0.5) < IdentifierExpression(x),
            "(assert (< (/ 1.0 2.0) (to_real |x_0|)))",
            id="exact_rational_and_to_real",
        ),
    ],
)
def test_convert_expression_to_smtlib2_writes_the_script(
    x: Identifier, expression: Any, text: str
) -> None:
    """Test a few scripts, pinned whole (the Rust stories pin the rest)."""
    script = convert_expression_to_smtlib2(expression(x), {x: SymbolType.INT})

    assert script.splitlines()[-1] == text


def test_convert_expression_to_z3_expression_maps_identifiers_to_declared_constants(
    x: Identifier,
) -> None:
    """Test the z3 conversion's map holds the script's declared constants."""
    z3 = pytest.importorskip("z3")
    from fhy_core.symbolic.expression import (  # noqa: PLC0415
        convert_expression_to_z3_expression,
    )

    y = mock_identifier("y", 1)
    term, mapping = convert_expression_to_z3_expression(
        IdentifierExpression(x) < IdentifierExpression(y),
        {x: SymbolType.INT, y: SymbolType.REAL},
    )

    assert set(mapping) == {x, y}
    assert mapping[x].eq(z3.Int("x_0"))
    assert mapping[y].eq(z3.Real("y_1"))
    assert z3.is_app_of(term, z3.Z3_OP_LT)


# =============================================================================
# Backends and their packages
# =============================================================================


@pytest.mark.z3
@pytest.mark.sympy
def test_solver_backends_resolve_to_one_adapter_each() -> None:
    """Test each ``SolverBackend`` member resolves to its adapter, created once."""
    from fhy_core.symbolic.expression.passes.sympy import (  # noqa: PLC0415
        SympySimplifier,
    )
    from fhy_core.symbolic.expression.passes.z3 import Z3Solver  # noqa: PLC0415

    z3_adapter = solver_module._resolve_adapter(SolverBackend.Z3)
    sympy_adapter = solver_module._resolve_adapter(SolverBackend.SYMPY)

    assert isinstance(z3_adapter, Z3Solver)
    assert isinstance(sympy_adapter, SympySimplifier)
    assert solver_module._resolve_adapter(SolverBackend.Z3) is z3_adapter
    assert solver_module._resolve_adapter(SolverBackend.SYMPY) is sympy_adapter
    assert (z3_adapter.name, sympy_adapter.name) == ("z3", "sympy")


@pytest.mark.z3
@pytest.mark.sympy
def test_backends_are_available_when_their_packages_are_installed() -> None:
    """Test ``is_backend_available`` for both shipped backends in this environment."""
    assert is_backend_available(SolverBackend.Z3) is True
    assert is_backend_available(SolverBackend.SYMPY) is True


def test_backend_capabilities_are_unchanged() -> None:
    """Test ``get_backend_capabilities`` keeps its table."""
    assert get_backend_capabilities(SolverBackend.SYMPY) == {
        SolverQueryKind.SIMPLIFICATION
    }
    assert get_backend_capabilities(SolverBackend.Z3) == {
        SolverQueryKind.SATISFIABILITY,
        SolverQueryKind.IMPLICATION,
        SolverQueryKind.UNIVERSAL_VALIDITY,
    }


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "package, backend, extra",
    [
        pytest.param("z3", "Z3", "z3", id="z3"),
        pytest.param("sympy", "SYMPY", "sympy", id="sympy"),
    ],
)
def test_missing_package_raises_solver_backend_unavailable_error(
    package: str, backend: str, extra: str
) -> None:
    """Test a missing package is reported when a question needs it, never at import."""
    completed = _run_python(
        f"""
        import sys
        sys.modules[{package!r}] = None
        import fhy_core
        from fhy_core.identifier import Identifier
        from fhy_core.symbolic.expression import IdentifierExpression
        from fhy_core.symbolic.solver import (
            SolverBackend,
            SolverBackendUnavailableError,
            check_expression_satisfiability,
            is_backend_available,
            simplify_expression,
        )
        from fhy_core.symbolic.symbol_type import SymbolType

        assert not is_backend_available(SolverBackend.{backend})
        x = Identifier("x")
        try:
            if {backend!r} == "Z3":
                check_expression_satisfiability(
                    IdentifierExpression(x) > 0, {{x: SymbolType.INT}}
                )
            else:
                simplify_expression(IdentifierExpression(x) + 0)
        except SolverBackendUnavailableError as error:
            assert isinstance(error, ImportError)
            print(error)
        else:
            raise AssertionError("no error")
        try:
            import fhy_core.symbolic.expression.passes.{package}
        except SolverBackendUnavailableError:
            print("the bridge refuses to import")
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert f"pip install fhy_core[{extra}]" in completed.stdout
    assert "the bridge refuses to import" in completed.stdout


@pytest.mark.subprocess
def test_missing_sympy_leaves_the_ground_backend_available() -> None:
    """Test ``GROUND`` folds without SymPy, and ``GROUND_THEN_SYMPY`` needs it."""
    completed = _run_python(
        """
        import sys
        sys.modules["sympy"] = None
        import fhy_core
        from fhy_core.symbolic.expression import LiteralExpression
        from fhy_core.symbolic.solver import (
            SolverBackend,
            is_backend_available,
            simplify_expression,
        )

        assert not is_backend_available(SolverBackend.GROUND_THEN_SYMPY)
        assert is_backend_available(SolverBackend.GROUND)
        folded = simplify_expression(
            LiteralExpression(2) + LiteralExpression(3), backend=SolverBackend.GROUND
        )
        assert folded == LiteralExpression(5), folded
        print("folded")
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["folded"]


@pytest.mark.subprocess
def test_importing_fhy_core_imports_neither_sympy_nor_z3() -> None:
    """Test a fresh ``import fhy_core`` leaves both packages unimported."""
    completed = _run_python(
        """
        import sys
        import fhy_core
        import fhy_core.symbolic.solver
        print(sorted(name for name in ("sympy", "z3") if name in sys.modules))
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["[]"]


@pytest.mark.z3
@pytest.mark.subprocess
def test_lazy_bridge_export_imports_its_package_on_first_access() -> None:
    """Test the first access of a lazy re-export imports the bridge's package."""
    completed = _run_python(
        """
        import sys
        from fhy_core.symbolic.expression import convert_expression_to_z3_expression
        print("z3" in sys.modules)
        """
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["True"]


@pytest.mark.z3
@pytest.mark.sympy
def test_lazy_bridge_exports_resolve_to_the_bridge_functions() -> None:
    """Test the package's lazy re-exports are the bridges' functions."""
    import fhy_core.symbolic.expression as expression_package  # noqa: PLC0415
    from fhy_core.symbolic.expression.passes import (  # noqa: PLC0415
        sympy as sympy_bridge,
    )
    from fhy_core.symbolic.expression.passes import z3 as z3_bridge  # noqa: PLC0415

    assert (
        expression_package.convert_expression_to_z3_expression
        is z3_bridge.convert_expression_to_z3_expression
    )
    assert (
        expression_package.convert_expression_to_sympy_expression
        is sympy_bridge.convert_expression_to_sympy_expression
    )
    with pytest.raises(AttributeError, match="no attribute 'missing_bridge'"):
        _ = expression_package.missing_bridge


# =============================================================================
# Errors and warnings
# =============================================================================


def test_missing_backend_raises_solver_capability_error_with_the_core_text(
    x: Identifier,
) -> None:
    """Test a solver without a capable backend raises the core's text."""
    with pytest.raises(
        SolverCapabilityError,
        match="the backends of this solver cannot answer satisfiability queries",
    ):
        Solver().check_expression_satisfiability(IdentifierExpression(x) > 0, {})
    with pytest.raises(SolverCapabilityError, match="cannot answer simplification"):
        Solver().simplify_expression(LiteralExpression(1))


def test_errors_map_to_the_python_classes(x: Identifier) -> None:
    """Test each core error raises its Python class."""
    solver = Solver(
        smt_solver=_RecordingSmtSolver(SatResult.SAT), simplifier=_RecordingSimplifier()
    )
    pi = get_native_constant_identifier("pi")

    with pytest.raises(
        KeyError, match=f"symbol_types is missing entries for identifiers: {x!r}"
    ):
        solver.check_expression_satisfiability(IdentifierExpression(x) > 0, {})
    with pytest.raises(NonBooleanLogicalOperandError):
        solver.check_expression_satisfiability(
            logical_and(LiteralExpression(2), LiteralExpression(4)), {}
        )
    with pytest.raises(NativeConstantBindingError, match=repr(pi)):
        solver.simplify_expression(
            IdentifierExpression(pi) > 3, {pi: LiteralExpression(1)}
        )
    with pytest.raises(TypeError, match="inline it first"):
        solver.check_expression_satisfiability(
            CallExpression("relu", (LiteralExpression(1),)) > 0, {}
        )
    with pytest.raises(NativeConstantLoweringError, match=repr(pi)):
        convert_expression_to_smtlib2(IdentifierExpression(pi) > 3)


def test_rust_backend_failure_raises_solver_backend_error(x: Identifier) -> None:
    """Test a Rust backend's failure raises ``SolverBackendError`` naming it."""
    solver = Solver(smt_solver=SmtLib2ProcessSolver("/nonexistent/fhy-smt-solver"))

    with pytest.raises(SolverBackendError, match="fhy-smt-solver") as exception_info:
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0, {x: SymbolType.INT}
        )

    assert isinstance(exception_info.value, RuntimeError)
    assert "cannot start the solver" in str(exception_info.value)


def test_refused_question_is_logged_naming_the_entry_point_node_and_sorts(
    caplog: pytest.LogCaptureFixture, x: Identifier
) -> None:
    """Test the hazard warning's logger, level, entry point, node repr and sorts."""
    hazard = BinaryExpression(
        BinaryOperation.FLOOR_DIVIDE, IdentifierExpression(x), IdentifierExpression(x)
    )
    question = hazard.equals(LiteralExpression(1))
    solver = Solver(smt_solver=_RecordingSmtSolver(SatResult.SAT))

    with caplog.at_level(logging.WARNING, logger=_SOLVER_LOGGER_NAME):
        result = solver.check_expression_satisfiability(question, {x: SymbolType.INT})

    assert result is None
    (record,) = [
        record for record in caplog.records if record.name == _SOLVER_LOGGER_NAME
    ]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert message.startswith(
        "check_expression_satisfiability: the expression applies a partial operation"
    )
    assert repr(hazard) in message
    assert f"identifier sorts at that node: {x!r}: INT" in message
    assert message.endswith("bounding timeout_milliseconds cannot change this outcome.")


def test_unknown_answer_is_logged_naming_the_backend_and_reason(
    caplog: pytest.LogCaptureFixture, x: Identifier
) -> None:
    """Test a backend's ``unknown`` is logged with its name and reason."""
    solver = Solver(smt_solver=_RecordingSmtSolver(SatResult.unknown("incomplete")))

    with caplog.at_level(logging.WARNING, logger=_SOLVER_LOGGER_NAME):
        solver.does_expression_imply(
            IdentifierExpression(x) > 0,
            IdentifierExpression(x) > 1,
            {x: SymbolType.INT},
        )

    assert [record.getMessage() for record in caplog.records] == [
        "does_expression_imply: the backend _RecordingSmtSolver answered unknown "
        "(incomplete)"
    ]


# =============================================================================
# The process backend from Python
# =============================================================================


def test_process_solver_answers_questions_through_a_solver(x: Identifier) -> None:
    """Test the process backend runs its program and reads its answer."""
    solver = Solver(smt_solver=_build_fake_process_solver())

    assert (
        solver.check_expression_satisfiability(
            IdentifierExpression(x) > 0, {x: SymbolType.INT}
        )
        is True
    )
    assert solver.check_expression_satisfiability(LiteralExpression(True), {}) is False


def test_process_solver_checks_a_script_directly(x: Identifier) -> None:
    """Test ``SmtLib2ProcessSolver.check`` decides a script called from Python."""
    script = SmtScript.lower(IdentifierExpression(x) > 0, {x: SymbolType.INT})

    assert _build_fake_process_solver().check(script) == SatResult.SAT


# A fake SMT-LIB2 executable that answers `sat` only when every line of the
# script it reads is one whole, printable command.
_PRINTABLE_SCRIPT_PROGRAM = textwrap.dedent(
    """
    import sys
    printable = True
    for line in sys.stdin:
        line = line.rstrip("\\n")
        if line == "(check-sat)":
            print("sat" if printable else "unsat", flush=True)
        elif line == "(exit)":
            break
        elif line and (
            not line.startswith("(")
            or any(not character.isprintable() for character in line)
        ):
            printable = False
    """
)


def _installed_z3_program() -> pathlib.Path | None:
    """Return the ``z3`` executable beside this interpreter, if one is."""
    program = pathlib.Path(sys.executable).parent / "z3"
    return program if program.is_file() else None


@pytest.mark.parametrize(
    "name_hint",
    ["nul\x00x", "ctl\x01x", "tab\tx", "nl\nx", "del\x7fx"],
    ids=["nul", "soh", "tab", "newline", "delete"],
)
def test_a_control_character_name_hint_answers_the_same_on_every_backend(
    name_hint: str,
) -> None:
    """Test a name hint's control characters never reach a solver's script."""
    identifier = Identifier(name_hint)
    reference = IdentifierExpression(identifier)
    expression = logical_and(reference > 0, reference < 2)
    symbol_types = {identifier: SymbolType.INT}
    backends: list[Any] = [_build_fake_process_solver(_PRINTABLE_SCRIPT_PROGRAM)]
    z3_program = _installed_z3_program()
    if z3_program is not None:
        backends.append(SmtLib2ProcessSolver(str(z3_program), ["-in"]))

    answers = [
        Solver(smt_solver=backend).check_expression_satisfiability(
            expression, symbol_types
        )
        for backend in backends
    ]
    if is_backend_available(SolverBackend.Z3):
        answers.append(
            check_expression_satisfiability(
                expression, symbol_types, backend=SolverBackend.Z3
            )
        )

    assert answers == [True] * len(answers)


# =============================================================================
# The default solver
# =============================================================================


def test_default_solver_holds_the_z3_and_sympy_adapters() -> None:
    """Test the default solver is one object holding the two shipped backends."""
    default = get_default_solver()

    assert get_default_solver() is default
    assert default.smt_solver is not None and default.smt_solver.name == "z3"
    assert default.simplifier is not None and default.simplifier.name == "sympy"


def test_default_smt_solver_refuses_an_adapter_that_is_no_smt_solver(
    x: Identifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the deferred z3 adapter raises ``TypeError`` for a non-SMT adapter."""
    default_smt_solver = get_default_solver().smt_solver
    assert default_smt_solver is not None
    script = SmtScript.lower(IdentifierExpression(x) > 0, {x: SymbolType.INT})
    monkeypatch.setattr(
        solver_module, "_resolve_adapter", lambda backend: _RecordingSimplifier()
    )

    with pytest.raises(TypeError, match="z3 adapter must be an SmtSolver"):
        default_smt_solver.check(script, timeout_milliseconds=None)


@pytest.mark.usefixtures("restore_default_solver")
def test_replacing_the_default_solver_reaches_the_functions_constraints_and_params(
    x: Identifier,
) -> None:
    """Test a plugged backend answers the module functions, constraints and params.

    The param's bound ``>= 0`` is decided exactly without the solver, so the
    param also holds a non-bound constraint, ``% 2 == 0``, that only the
    plugged simplifier decides: it answers False even for the even value 4.
    """
    natural = create_natural_param()
    param = natural.add_constraint(
        EquationConstraint((natural.variable_expression % 2).equals(0))
    )
    backend = _RecordingSmtSolver(SatResult.UNSAT)
    simplifier = _RecordingSimplifier(LiteralExpression(False))
    set_default_solver(Solver(smt_solver=backend, simplifier=simplifier))

    satisfiable = check_expression_satisfiability(
        IdentifierExpression(x) > 0, {x: SymbolType.INT}
    )
    simplified = simplify_expression(
        IdentifierExpression(x) >= 0, {x: LiteralExpression(3)}
    )
    implied = create_constraint_system(
        EquationConstraint(IdentifierExpression(x) >= 1)
    ).check_implication(
        create_constraint_system(EquationConstraint(IdentifierExpression(x) >= 0)),
        {x: SymbolType.INT},
    )
    is_valid = param.is_value_valid(4)

    assert satisfiable is False
    assert simplified == LiteralExpression(False)
    assert implied.name == "SATISFIED"
    assert is_valid is False
    assert len(backend.checks) == 2
    assert len(simplifier.inputs) >= 2


@pytest.mark.usefixtures("restore_default_solver")
def test_named_backend_ignores_the_default_solver(x: Identifier) -> None:
    """Test ``backend=SolverBackend.Z3`` asks z3, whatever the default is."""
    pytest.importorskip("z3")
    set_default_solver(Solver(smt_solver=_RecordingSmtSolver(SatResult.UNSAT)))

    result = check_expression_satisfiability(
        IdentifierExpression(x) > 0, {x: SymbolType.INT}, backend=SolverBackend.Z3
    )

    assert result is True


def test_set_default_solver_refuses_another_value() -> None:
    """Test only a ``Solver`` can become the default."""
    with pytest.raises(TypeError, match="must be a Solver"):
        set_default_solver(object())  # type: ignore[arg-type]


# =============================================================================
# Threads
# =============================================================================


@pytest.mark.parametrize(
    "build_backend",
    [
        pytest.param(lambda: _RecordingSmtSolver(SatResult.SAT), id="python_backend"),
        pytest.param(_build_fake_process_solver, id="process_backend"),
    ],
)
def test_concurrent_questions_from_several_threads(build_backend: Any) -> None:
    """Test threads asking one solver at once each get their own answer."""
    solver = Solver(smt_solver=build_backend())
    answers: dict[int, bool | None] = {}
    errors: list[BaseException] = []

    def ask(index: int) -> None:
        try:
            identifier = Identifier(f"t{index}")
            answers[index] = solver.check_expression_satisfiability(
                IdentifierExpression(identifier) > index, {identifier: SymbolType.INT}
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=ask, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert answers == dict.fromkeys(range(8), True)


@pytest.mark.sympy
@pytest.mark.z3
def test_param_checks_ask_the_default_solver() -> None:
    """Test a param's feasibility and value checks reach the default backends."""
    left = create_integer_param_between(0, 10)
    right = create_integer_param_between(5, 20)

    assert left.check_feasibility().name == "SATISFIED"
    assert (left & right).is_value_valid(7)
