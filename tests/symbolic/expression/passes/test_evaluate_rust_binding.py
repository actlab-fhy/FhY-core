"""Tests for the Python API of the evaluators over the Rust core (S9).

The Rust tests in ``rust/fhy-core/tests/it/expression/`` specify the
evaluation itself; these cover what the binding adds: NumPy as an
optional import, the conversion of bindings and results, the Python error
classes, the two passes, native user functions called from Rust, the
built-ins' implementations, the literal helpers, and threads.
"""

import math
import pickle
import subprocess
import sys
import threading
import time
from decimal import Decimal
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassExecutionError
from fhy_core.symbolic.expression import (
    BUILTIN_FUNCTIONS,
    CallExpression,
    EntryLookupError,
    Expression,
    FunctionArityError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeConstantBindingError,
    NativeFunction,
    NativeResultSortError,
    NonBooleanLogicalOperandError,
    NonFiniteCastError,
    StringLiteralPrecisionError,
    UnboundVariableError,
    UnsupportedNumpyLoweringError,
    call,
    evaluate_expression,
    evaluate_expression_with_numpy,
    get_native_constant_identifier,
    logical_and,
    piecewise,
    register_function,
    register_native_function,
)
from fhy_core.symbolic.expression.passes.evaluate import ExpressionEvaluator
from fhy_core.symbolic.expression.passes.native_lowering import (
    coerce_literal_value,
    is_decimal_text_exactly_binary,
)
from fhy_core.symbolic.expression.passes.numpy import NumpyExpressionEvaluator

np = pytest.importorskip("numpy")


def _reference(name: str) -> tuple[Identifier, IdentifierExpression]:
    identifier = Identifier(name)
    return identifier, IdentifierExpression(identifier)


def _run(program: str) -> str:
    return subprocess.check_output(
        [sys.executable, "-c", program], text=True, stderr=subprocess.STDOUT
    ).strip()


# =============================================================================
# NumPy is optional (D-S9-11)
# =============================================================================


@pytest.mark.subprocess
def test_fhy_core_imports_and_folds_without_numpy() -> None:
    """Test ``import fhy_core`` and a fold work, and never import NumPy."""
    output = _run(
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "from fhy_core.symbolic.expression import CallExpression, LiteralExpression,"
        " evaluate_expression\n"
        "print(evaluate_expression(CallExpression('exp', (LiteralExpression(0.0),))))"
    )

    assert output == "1"


@pytest.mark.subprocess
def test_a_fresh_import_and_a_fold_leave_numpy_unimported() -> None:
    """Test neither ``import fhy_core`` nor a fold imports NumPy."""
    output = _run(
        "import sys\n"
        "import fhy_core.symbolic.expression as expression\n"
        "expression.evaluate_expression(expression.call('floor', 2.5))\n"
        "print('numpy' in sys.modules)"
    )

    assert output == "False"


@pytest.mark.subprocess
def test_the_numpy_evaluator_without_numpy_raises_the_guiding_import_error() -> None:
    """Test the NumPy evaluator names the extra to install, caused by the failure."""
    output = _run(
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "from fhy_core.identifier import Identifier\n"
        "from fhy_core.symbolic.expression import IdentifierExpression,"
        " evaluate_expression_with_numpy\n"
        "x = Identifier('x')\n"
        "try:\n"
        "    evaluate_expression_with_numpy(IdentifierExpression(x), {x: 1.0})\n"
        "except ImportError as error:\n"
        "    print(type(error).__name__, '|', error, '|',"
        " type(error.__cause__).__name__)\n"
    )

    assert "fhy_core[numpy]" in output
    assert output.startswith("ImportError |")
    cause = output.rsplit(" | ", 1)[-1]
    assert cause in {"ImportError", "ModuleNotFoundError"}


# =============================================================================
# Bindings (D-S9-4, D-S9-12)
# =============================================================================


@pytest.mark.parametrize(
    "dtype, domain",
    [
        ("bool", np.bool_),
        ("int8", np.int64),
        ("int16", np.int64),
        ("int32", np.int64),
        ("int64", np.int64),
        ("uint8", np.int64),
        ("uint16", np.int64),
        ("uint32", np.int64),
        ("uint64", np.int64),
        ("float16", np.float64),
        ("float32", np.float64),
        ("float64", np.float64),
    ],
)
def test_an_admitted_dtype_is_read_in_its_domain(dtype: str, domain: type) -> None:
    """Test each admitted dtype is read as a Boolean, ``int64`` or ``float64``."""
    x, reference = _reference("x")
    values = np.array([0, 1, 1], dtype=dtype)

    result = evaluate_expression_with_numpy(reference, {x: values})

    assert result.dtype == domain
    assert np.array_equal(result, values.astype(domain))


@pytest.mark.parametrize(
    "values",
    [
        np.array([1 + 2j]),
        np.array([1, 2], dtype=object),
        np.array(["a"]),
        np.array([b"a"]),
        np.array(["2026-09-26"], dtype="datetime64[D]"),
    ],
    ids=["complex", "object", "str", "bytes", "datetime"],
)
def test_an_unsupported_dtype_is_refused_naming_the_identifier(values: Any) -> None:
    """Test a dtype outside the three domains raises ``TypeError``."""
    x, reference = _reference("x")

    with pytest.raises(TypeError, match=r'bound to "x" has dtype'):
        evaluate_expression_with_numpy(reference, {x: values})


def test_a_uint64_above_the_signed_range_raises_overflow_error() -> None:
    """Test a ``uint64`` array with a value above ``2**63 - 1`` is refused."""
    x, reference = _reference("x")
    fits = np.array([0, 2**63 - 1], dtype=np.uint64)
    too_big = np.array([0, 2**63], dtype=np.uint64)

    assert np.asarray(evaluate_expression_with_numpy(reference, {x: fits}))[1] == (
        2**63 - 1
    )
    with pytest.raises(OverflowError, match="uint64"):
        evaluate_expression_with_numpy(reference, {x: too_big})


def test_a_python_int_beyond_int64_raises_overflow_error() -> None:
    """Test a Python ``int`` outside the 64-bit range is refused."""
    x, reference = _reference("x")

    with pytest.raises(OverflowError, match="outside the 64-bit range"):
        evaluate_expression_with_numpy(reference + 1, {x: 2**64})


@pytest.mark.parametrize(
    "values",
    [
        np.arange(6.0).reshape(2, 3).astype(">f8"),
        np.asfortranarray(np.arange(6.0).reshape(2, 3)),
        np.arange(12.0).reshape(2, 6)[:, ::2],
        np.arange(6.0).reshape(2, 3)[:, ::-1],
        np.broadcast_to(np.arange(3.0), (2, 3)),
    ],
    ids=["swapped-byte-order", "fortran", "strided", "negative-strides", "zero-stride"],
)
def test_bindings_of_any_layout_evaluate_as_their_values(values: Any) -> None:
    """Test a binding's byte order and strides do not change its lanes."""
    x, reference = _reference("x")

    result = evaluate_expression_with_numpy(reference * 2.0 + 1.0, {x: values})

    assert np.array_equal(result, np.asarray(values, dtype=np.float64) * 2.0 + 1.0)
    assert result.flags.c_contiguous


def test_python_scalars_and_nested_lists_are_bindings() -> None:
    """Test ``bool``, ``int``, ``float`` and nested lists bind as NumPy reads them."""
    x, x_reference = _reference("x")
    y, y_reference = _reference("y")
    p, p_reference = _reference("p")
    tree = piecewise((p_reference, x_reference * y_reference), otherwise=-1)

    scalar = evaluate_expression_with_numpy(tree, {x: 2, y: 1.5, p: True})
    array = evaluate_expression_with_numpy(
        tree, {x: [[1], [2]], y: [1.0, 2.0], p: True}
    )

    assert scalar == 3.0
    assert np.array_equal(array, [[1.0, 2.0], [2.0, 4.0]])


def test_bindings_the_expression_does_not_refer_to_are_never_read() -> None:
    """Test an unreferenced binding is ignored, even one NumPy cannot read."""
    x, reference = _reference("x")
    unused = Identifier("unused")

    result = evaluate_expression_with_numpy(
        reference + 1,
        {x: np.array([1, 2]), unused: object()},  # type: ignore[dict-item]
    )

    assert np.array_equal(result, [2, 3])


def test_empty_and_zero_dimensional_bindings() -> None:
    """Test an empty binding gives an empty result and a 0-d one a scalar."""
    x, reference = _reference("x")

    empty = evaluate_expression_with_numpy(reference * 2.0, {x: np.zeros((0, 3))})
    scalar = evaluate_expression_with_numpy(reference * 2.0, {x: np.array(3.0)})

    assert empty.shape == (0, 3)
    assert isinstance(scalar, np.float64)
    assert scalar == 6.0


# =============================================================================
# Results (D-S9-13)
# =============================================================================


@pytest.mark.parametrize(
    "tree_of, scalar_type",
    [
        (lambda x: x > 0, np.bool_),
        (lambda x: call("floor", x), np.int64),
        (lambda x: x * 2.0, np.float64),
        (lambda x: x, np.float64),
        (lambda x: piecewise((x > 0, x), otherwise=0.0), np.float64),
    ],
    ids=["bool", "int", "real", "identifier-root", "piecewise-root"],
)
def test_a_zero_dimensional_result_is_a_numpy_scalar(
    tree_of: Any, scalar_type: type
) -> None:
    """Test every 0-d result is a NumPy scalar of its domain, whatever its root."""
    x, reference = _reference("x")

    result = evaluate_expression_with_numpy(tree_of(reference), {x: 1.5})

    assert type(result) is scalar_type


def test_an_array_result_is_a_new_writeable_c_contiguous_array() -> None:
    """Test a result never is, nor shares memory with, a binding."""
    x, reference = _reference("x")
    values = np.arange(4.0)

    result = np.asarray(evaluate_expression_with_numpy(reference, {x: values}))

    assert result is not values
    assert not np.shares_memory(result, values)
    assert result.flags.writeable
    assert result.flags.c_contiguous
    result[0] = 99.0
    assert values[0] == 0.0


# =============================================================================
# Errors (D-S9-14), raised directly (D-S9-10)
# =============================================================================


def _raise_case(case: str) -> tuple[type[BaseException], Expression, dict[Any, Any]]:
    x, reference = _reference("x")
    ints = np.array([1, 2])
    pi = get_native_constant_identifier("pi")
    cases: dict[str, tuple[type[BaseException], Expression, dict[Any, Any]]] = {
        "unknown": (EntryLookupError, call("s9_nowhere", reference), {x: ints}),
        "constant-called": (FunctionArityError, CallExpression("pi", ()), {}),
        "ill-typed": (
            NonBooleanLogicalOperandError,
            logical_and(reference, LiteralExpression(True)),
            {x: ints},
        ),
        "bound-constant": (
            NativeConstantBindingError,
            IdentifierExpression(pi) + 1,
            {pi: 3.0},
        ),
        "unbound": (UnboundVariableError, reference + 1, {}),
        "inexact-decimal": (
            StringLiteralPrecisionError,
            reference + LiteralExpression(Decimal("0.1")),
            {x: ints},
        ),
        "overflow": (OverflowError, reference * 2**62, {x: np.array([4])}),
        "division-by-zero": (ZeroDivisionError, reference // 0, {x: ints}),
        "negative-exponent": (ValueError, reference ** (-1), {x: ints}),
        "non-finite-cast": (
            NonFiniteCastError,
            call("floor", reference / 0.0),
            {x: np.array([1.0])},
        ),
        "boolean-arithmetic": (
            TypeError,
            reference + LiteralExpression(True),
            {x: ints},
        ),
        "shape": (
            ValueError,
            reference + IdentifierExpression(Identifier("y")),
            {x: ints},
        ),
    }
    exception, tree, environment = cases[case]
    if case == "shape":
        y = next(iter(tree.get_free_identifiers() - {x}))
        environment = {x: ints, y: np.array([1, 2, 3])}
    return exception, tree, environment


_RAISE_CASES = [
    "unknown",
    "constant-called",
    "ill-typed",
    "bound-constant",
    "unbound",
    "inexact-decimal",
    "overflow",
    "division-by-zero",
    "negative-exponent",
    "non-finite-cast",
    "boolean-arithmetic",
    "shape",
]


@pytest.mark.parametrize("case", _RAISE_CASES)
def test_the_function_raises_each_error_directly(case: str) -> None:
    """Test ``evaluate_expression_with_numpy`` raises each error as its class."""
    exception, tree, environment = _raise_case(case)

    with pytest.raises(exception):
        evaluate_expression_with_numpy(tree, environment)


@pytest.mark.parametrize("case", _RAISE_CASES)
def test_a_pass_run_wraps_each_error(case: str) -> None:
    """Test ``NumpyExpressionEvaluator`` wraps the error, as every pass does."""
    exception, tree, environment = _raise_case(case)

    with pytest.raises(PassExecutionError) as exception_info:
        NumpyExpressionEvaluator(environment)(tree)

    assert isinstance(exception_info.value.__cause__, exception)


def test_an_unsupported_native_user_function_is_refused(
    function_registry_snapshot: None,
) -> None:
    """Test a registered native function is refused by the NumPy evaluator."""
    register_native_function(
        "s9_softplus", [FunctionSort.REAL], FunctionSort.REAL, math.exp
    )
    x, reference = _reference("x")

    with pytest.raises(UnsupportedNumpyLoweringError, match="s9_softplus"):
        evaluate_expression_with_numpy(call("s9_softplus", reference), {x: 1.0})


def test_the_screen_runs_before_the_bound_constant_refusal() -> None:
    """Test the Boolean screen comes before the bound-constant refusal."""
    x, reference = _reference("x")
    pi = get_native_constant_identifier("pi")
    tree = logical_and(reference, IdentifierExpression(pi) > 0)

    with pytest.raises(NonBooleanLogicalOperandError):
        evaluate_expression_with_numpy(tree, {x: np.array([1]), pi: 3.0})


# =============================================================================
# The passes (D-S9-10, D-S9-15)
# =============================================================================


def test_both_evaluator_passes_are_registered() -> None:
    """Test the two passes keep their registry names."""
    passes = CompilerPass.get_registered_passes()

    assert passes["fhy_core.symbolic.expression.evaluate"].pass_type is (
        ExpressionEvaluator
    )
    assert passes["fhy_core.symbolic.expression.evaluate_with_numpy"].pass_type is (
        NumpyExpressionEvaluator
    )


def test_the_numpy_pass_inlines_and_reads_its_snapshot() -> None:
    """Test the NumPy pass inlines, and reads the environment given at construction."""
    x, reference = _reference("x")
    environment = {x: np.array([-1.0, 2.0])}
    evaluator = NumpyExpressionEvaluator(environment)
    environment[x] = np.array([5.0, 5.0])

    result = evaluator(call("relu", reference))

    assert np.array_equal(result, [0.0, 2.0])


def test_the_fold_reports_each_function_with_a_body_once(
    function_registry_snapshot: None,
) -> None:
    """Test the fold reports a WARNING per kept function with a body, once each."""
    x = Identifier("x")
    register_function(
        "s9_double",
        [x],
        [FunctionSort.REAL],
        FunctionSort.REAL,
        IdentifierExpression(x) * 2,
    )
    tree = call("s9_double", 1.0) + call("s9_double", 2.0) + call("relu", 3.0)

    result = ExpressionEvaluator().execute(tree)

    assert result.output is tree
    assert not result.changed
    messages = [diagnostic.message for diagnostic in result.diagnostics]
    assert [diagnostic.level for diagnostic in result.diagnostics] == [
        DiagnosticLevel.WARNING,
        DiagnosticLevel.WARNING,
    ]
    assert "'s9_double'" in str(messages[0])
    assert "'relu'" in str(messages[1])


def test_the_fold_changes_the_ir_exactly_when_it_folds() -> None:
    """Test ``did_change`` is by identity, and the output shares what it keeps."""
    _, reference = _reference("x")
    kept = reference * 2
    tree = kept + call("exp", 0.0)

    result = ExpressionEvaluator().execute(tree)

    assert result.changed
    assert str(result.output) == "((x * 2) + 1)"
    assert result.output.left is kept  # type: ignore[attr-defined]


def test_the_fold_checks_a_folded_calls_arity() -> None:
    """Test a folded native call's arity is checked (Z-8).

    The Python fold called ``math.sin(1.0, 2.0)`` and passed on its
    ``TypeError``; the core checks the arity first.
    """
    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression(CallExpression("sin", (LiteralExpression(1.0),) * 2))

    assert isinstance(exception_info.value.__cause__, FunctionArityError)
    assert '"sin" takes 1 argument but the call passes 2' in str(
        exception_info.value.__cause__
    )


# =============================================================================
# Native user functions called from Rust
# =============================================================================


def test_a_native_implementation_receives_python_values(
    function_registry_snapshot: None,
) -> None:
    """Test the implementation receives ``float``, ``int`` and ``bool`` values."""
    received: list[tuple[type, ...]] = []

    def record(a: object, b: object, c: object) -> float:
        received.append((type(a), type(b), type(c)))
        return 1.0

    register_native_function(
        "s9_record",
        [FunctionSort.REAL, FunctionSort.INT, FunctionSort.BOOL],
        FunctionSort.REAL,
        record,
    )

    result = evaluate_expression(
        call(
            "s9_record",
            LiteralExpression(Decimal("0.5")),
            2**70,
            LiteralExpression(True),
        )
    )

    assert isinstance(result, LiteralExpression)
    assert received == [(float, int, bool)]


def test_a_native_implementations_exception_is_the_cause_itself(
    function_registry_snapshot: None,
) -> None:
    """Test the exception an implementation raises propagates as the same object."""
    raised = ValueError("s9 boom")

    def fail(_value: float) -> float:
        raise raised

    register_native_function("s9_fail", [FunctionSort.REAL], FunctionSort.REAL, fail)

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression(call("s9_fail", 1.0))

    assert exception_info.value.__cause__ is raised


def test_a_keyboard_interrupt_passes_through_a_native_implementation(
    function_registry_snapshot: None,
) -> None:
    """Test a ``KeyboardInterrupt`` from an implementation is not wrapped."""

    def interrupt(_value: float) -> float:
        raise KeyboardInterrupt

    register_native_function(
        "s9_interrupt", [FunctionSort.REAL], FunctionSort.REAL, interrupt
    )

    with pytest.raises(KeyboardInterrupt):
        evaluate_expression(call("s9_interrupt", 1.0))


def test_a_native_result_of_no_numeric_type_is_refused(
    function_registry_snapshot: None,
) -> None:
    """Test a result that is no ``bool``, ``int`` or ``float`` is refused."""
    register_native_function(
        "s9_text",
        [FunctionSort.REAL],
        FunctionSort.REAL,
        lambda _value: "one",  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression(call("s9_text", 1.0))

    assert isinstance(exception_info.value.__cause__, NativeResultSortError)
    assert "'one'" in str(exception_info.value.__cause__)


# =============================================================================
# The built-ins' implementations (D-S9-9)
# =============================================================================


@pytest.mark.parametrize(
    "name",
    [
        name
        for name, entry in BUILTIN_FUNCTIONS.items()
        if isinstance(entry, NativeFunction)
    ],
)
def test_a_builtin_implementation_agrees_with_the_fold(name: str) -> None:
    """Test a native built-in's implementation computes what the fold folds to."""
    entry = BUILTIN_FUNCTIONS[name]  # type: ignore[literal-required]
    assert isinstance(entry, NativeFunction)
    argument = 0.625

    folded = evaluate_expression(call(name, argument))

    assert isinstance(folded, LiteralExpression)
    value = entry.implementation(argument)
    assert value == folded.value
    assert type(value) is type(folded.value)


def test_builtin_implementations_are_one_object_each_and_pickle_as_themselves() -> None:
    """Test each implementation is one object, prints its name, and pickles."""
    sqrt = BUILTIN_FUNCTIONS["sqrt"].implementation
    floor = BUILTIN_FUNCTIONS["floor"].implementation

    assert _rs.BuiltinNativeImplementation._of("sqrt") is sqrt
    assert pickle.loads(pickle.dumps(sqrt)) is sqrt
    assert repr(sqrt) == "<built-in native implementation of sqrt>"
    assert sqrt.__name__ == "sqrt"
    assert math.isnan(sqrt(-1.0))
    assert sqrt(16) == 4.0
    assert floor(2.5) == 2
    assert type(floor(1e300)) is int
    with pytest.raises(TypeError, match=r"takes a real, got bool"):
        sqrt(True)
    with pytest.raises(ValueError, match="not a native built-in"):
        _rs.BuiltinNativeImplementation._of("relu")


# =============================================================================
# The literal helpers (D-S9-16)
# =============================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("0.5", True),
        ("0.1", False),
        ("-0.25", True),
        ("5", True),
        ("9007199254740993", False),
        (Decimal("1.5"), True),
        (Decimal("-0.1"), False),
    ],
)
def test_is_decimal_text_exactly_binary(text: str | Decimal, expected: bool) -> None:
    """Test the exact-binary test over text and decimals, signed or not."""
    assert is_decimal_text_exactly_binary(text) is expected


def test_coerce_literal_value() -> None:
    """Test the coercion of literal values to Python numbers."""
    three = 3
    half = 0.5

    assert coerce_literal_value(three) is three
    assert coerce_literal_value(half) is half
    assert coerce_literal_value(True) is True
    assert coerce_literal_value("12") == 12
    assert type(coerce_literal_value("12")) is int
    assert coerce_literal_value("0.5") == 0.5
    assert coerce_literal_value(Decimal("-2.5")) == -2.5
    with pytest.raises(StringLiteralPrecisionError, match=r"0\.1"):
        coerce_literal_value(Decimal("0.1"))
    with pytest.raises(ValueError):
        coerce_literal_value("1e5")


# =============================================================================
# Threads (D-S9-12)
# =============================================================================


def test_another_thread_runs_during_a_large_evaluation() -> None:
    """Test the interpreter is released while the array backend walks."""
    x, reference = _reference("x")
    values = np.random.default_rng(0).standard_normal(4_000_000)
    tree = call("sqrt", reference * reference + 1.0) * 2.0 - reference
    ticks = 0
    done = threading.Event()

    def tick() -> None:
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0)

    ticker = threading.Thread(target=tick)
    ticker.start()
    try:
        before = ticks
        evaluate_expression_with_numpy(tree, {x: values})
        during = ticks - before
    finally:
        done.set()
        ticker.join()

    assert during > 0


def test_concurrent_evaluations_agree() -> None:
    """Test evaluations from several threads agree with one another."""
    x, reference = _reference("x")
    values = np.linspace(-3.0, 3.0, 200_001)
    tree = call("tanh", reference) + call("sigmoid", reference)
    expected = evaluate_expression_with_numpy(tree, {x: values})
    results: list[Any] = []

    def evaluate() -> None:
        results.append(evaluate_expression_with_numpy(tree, {x: values}))

    threads = [threading.Thread(target=evaluate) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 4
    for result in results:
        assert np.array_equal(result, expected)


# =============================================================================
# NumPy kernels (N-S9-2 (b))
# =============================================================================


@pytest.mark.parametrize(
    "values",
    [
        np.linspace(-2.0, 2.0, 7),
        np.asfortranarray(np.linspace(-2.0, 2.0, 12).reshape(3, 4)),
        np.linspace(-2.0, 2.0, 200_003),
    ],
    ids=["contiguous", "fortran", "chunked"],
)
@pytest.mark.parametrize("name", ["exp", "tanh", "arcsin"])
def test_a_transcendental_native_is_numpys_ufunc(name: str, values: Any) -> None:
    """Test the 14 transcendental natives compute NumPy's ufunc over arrays.

    A lone call over a binding, a Fortran-ordered binding and one evaluated
    in chunks agree exactly with the ufunc, and the result is a new
    C-contiguous array; NumPy's warnings are silenced.
    """
    x, reference = _reference("x")

    with np.errstate(all="raise"):
        result = evaluate_expression_with_numpy(call(name, reference), {x: values})
        compound = evaluate_expression_with_numpy(
            call(name, reference) + 0.0, {x: values}
        )

    with np.errstate(all="ignore"):
        expected = getattr(np, name)(values)
    assert np.array_equal(result, expected, equal_nan=True)
    assert np.array_equal(compound, expected, equal_nan=True)
    assert result.flags.c_contiguous
    assert not np.shares_memory(result, values)


def test_a_lone_native_over_a_bound_constant_is_still_refused() -> None:
    """Test binding a constant's identifier is refused whatever the tree's shape."""
    pi = get_native_constant_identifier("pi")

    with pytest.raises(NativeConstantBindingError):
        evaluate_expression_with_numpy(
            call("exp", IdentifierExpression(pi)), {pi: np.array([1.0, 2.0])}
        )
