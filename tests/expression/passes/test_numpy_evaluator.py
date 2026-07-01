"""Tests for `fhy_core.expression.passes.numpy`."""

import math
import sys
import time
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    EntryLookupError,
    FunctionArityError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeFunction,
    StringLiteralPrecisionError,
    UnaryExpression,
    UnaryOperation,
    UnboundVariableError,
    UnsupportedNumpyLoweringError,
    call,
    evaluate_expression_with_numpy,
    get_registered_entries,
    register_function,
    register_native_function,
    ternary,
)
from fhy_core.expression.passes.numpy import _NATIVE_FUNCTION_UFUNC_NAMES
from fhy_core.pass_infrastructure import PassExecutionError

from ..conftest import mock_identifier

np = pytest.importorskip("numpy")


# =============================================================================
# Binary arithmetic operators
# =============================================================================

ARITHMETIC_CASES = [
    (BinaryOperation.ADD, lambda a, b: a + b),
    (BinaryOperation.SUBTRACT, lambda a, b: a - b),
    (BinaryOperation.MULTIPLY, lambda a, b: a * b),
    (BinaryOperation.DIVIDE, lambda a, b: a / b),
    (BinaryOperation.FLOOR_DIVIDE, lambda a, b: a // b),
    (BinaryOperation.MODULO, lambda a, b: a % b),
    (BinaryOperation.POWER, lambda a, b: a**b),
]


@pytest.mark.parametrize(
    "operation, reference",
    ARITHMETIC_CASES,
    ids=[operation.value for operation, _ in ARITHMETIC_CASES],
)
def test_evaluates_binary_arithmetic_elementwise(
    operation: BinaryOperation, reference: Callable[[Any, Any], Any]
) -> None:
    """Test each arithmetic operator matches the hand-written NumPy result."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([6.0, 4.0, 9.0])
    right = np.array([2.0, 2.0, 3.0])
    expression = BinaryExpression(
        operation, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = evaluate_expression_with_numpy(expression, {x: left, y: right})

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, reference(left, right))


# =============================================================================
# Binary comparison operators
# =============================================================================

COMPARISON_CASES = [
    (BinaryOperation.EQUAL, lambda a, b: a == b),
    (BinaryOperation.NOT_EQUAL, lambda a, b: a != b),
    (BinaryOperation.LESS, lambda a, b: a < b),
    (BinaryOperation.LESS_EQUAL, lambda a, b: a <= b),
    (BinaryOperation.GREATER, lambda a, b: a > b),
    (BinaryOperation.GREATER_EQUAL, lambda a, b: a >= b),
]


@pytest.mark.parametrize(
    "operation, reference",
    COMPARISON_CASES,
    ids=[operation.value for operation, _ in COMPARISON_CASES],
)
def test_evaluates_comparison_to_boolean_array(
    operation: BinaryOperation, reference: Callable[[Any, Any], Any]
) -> None:
    """Test each comparison operator yields an elementwise boolean array."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([2.0, 2.0, 2.0])
    expression = BinaryExpression(
        operation, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = evaluate_expression_with_numpy(expression, {x: left, y: right})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, reference(left, right))


# =============================================================================
# Logical operators
# =============================================================================

LOGICAL_BINARY_CASES = [
    (BinaryOperation.LOGICAL_AND, np.logical_and),
    (BinaryOperation.LOGICAL_OR, np.logical_or),
]


@pytest.mark.parametrize(
    "operation, reference",
    LOGICAL_BINARY_CASES,
    ids=[operation.value for operation, _ in LOGICAL_BINARY_CASES],
)
def test_evaluates_logical_binary_elementwise(
    operation: BinaryOperation, reference: Callable[[Any, Any], Any]
) -> None:
    """Test logical and/or evaluate elementwise over boolean arrays."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    expression = BinaryExpression(
        operation, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = evaluate_expression_with_numpy(expression, {x: left, y: right})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, reference(left, right))


def test_evaluates_logical_not_elementwise() -> None:
    """Test logical negation inverts each element of a boolean array."""
    x = mock_identifier("x", 0)
    values = np.array([True, False, True])
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(x))

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, np.logical_not(values))


def test_evaluates_chained_comparison_with_logical_and() -> None:
    """Test a range check built from comparisons and logical-and."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = (x_expression > 0.0).logical_and(x_expression < 10.0)
    values = np.array([-1.0, 5.0, 15.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, (values > 0.0) & (values < 10.0))


# =============================================================================
# Unary numeric operators
# =============================================================================

UNARY_NUMERIC_CASES = [
    (UnaryOperation.NEGATE, lambda a: -a),
    (UnaryOperation.POSITIVE, lambda a: +a),
]


@pytest.mark.parametrize(
    "operation, reference",
    UNARY_NUMERIC_CASES,
    ids=[operation.value for operation, _ in UNARY_NUMERIC_CASES],
)
def test_evaluates_unary_numeric_elementwise(
    operation: UnaryOperation, reference: Callable[[Any], Any]
) -> None:
    """Test unary negate/positive match the hand-written NumPy result."""
    x = mock_identifier("x", 0)
    values = np.array([-2.0, 0.0, 3.5])
    expression = UnaryExpression(operation, IdentifierExpression(x))

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, reference(values))


# =============================================================================
# Native math functions
# =============================================================================

NATIVE_FUNCTION_CASES = [
    ("exp", np.exp),
    ("exp2", np.exp2),
    ("log", np.log),
    ("log2", np.log2),
    ("log10", np.log10),
    ("sqrt", np.sqrt),
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("arcsin", np.arcsin),
    ("arccos", np.arccos),
    ("arctan", np.arctan),
    ("sinh", np.sinh),
    ("cosh", np.cosh),
    ("tanh", np.tanh),
    ("round", np.round),
    ("floor", np.floor),
    ("ceil", np.ceil),
]


@pytest.mark.parametrize(
    "function_name, reference",
    NATIVE_FUNCTION_CASES,
    ids=[name for name, _ in NATIVE_FUNCTION_CASES],
)
def test_evaluates_native_function_over_array(
    function_name: str, reference: Callable[[Any], Any]
) -> None:
    """Test each mapped native math function matches its NumPy reference."""
    x = mock_identifier("x", 0)
    values = np.array([0.1, 0.5, 0.9])
    expression = call(function_name, x)

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, reference(values))


NATIVE_FUNCTION_HAND_VALUES = [
    ("exp", 0.0, 1.0),
    ("sqrt", 4.0, 2.0),
    ("log", 1.0, 0.0),
    ("log2", 8.0, 3.0),
    ("log10", 1000.0, 3.0),
    ("sin", 0.0, 0.0),
    ("cos", 0.0, 1.0),
    ("floor", 2.7, 2.0),
    ("ceil", 2.1, 3.0),
    ("round", 2.4, 2.0),
    ("round", 2.6, 3.0),
]


@pytest.mark.parametrize(
    "function_name, input_value, expected",
    NATIVE_FUNCTION_HAND_VALUES,
)
def test_native_function_matches_hand_computed_value(
    function_name: str, input_value: float, expected: float
) -> None:
    """Test native math functions against independently computed values."""
    x = mock_identifier("x", 0)
    expression = call(function_name, x)

    result = evaluate_expression_with_numpy(expression, {x: np.array([input_value])})

    assert np.allclose(result, [expected])


# =============================================================================
# Result-sort conformance
# =============================================================================

_INTEGER_SORT_FUNCTION_NAMES = ["round", "floor", "ceil"]


@pytest.mark.parametrize("function_name", _INTEGER_SORT_FUNCTION_NAMES)
def test_integer_sort_native_returns_integer_dtype(function_name: str) -> None:
    """Test an ``INT``-sorted native casts its result to an integer dtype.

    ``numpy.floor``/``round``/``ceil`` return floating-point by default;
    the evaluator casts to the declared ``INT`` result sort so the NumPy
    path agrees with ``evaluate_expression`` and the declared sort.
    """
    x = mock_identifier("x", 0)
    values = np.array([2.7, -1.2, 3.5])
    expression = call(function_name, x)

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.integer)


def test_real_sort_native_preserves_float_width() -> None:
    """Test a ``REAL``-sorted native leaves a ``float32`` result as ``float32``.

    Real-sorted results are already floating-point, so the evaluator casts
    nothing and the input's narrower width survives the call.
    """
    x = mock_identifier("x", 0)
    values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    expression = call("sqrt", x)

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


# =============================================================================
# Literal leaves
# =============================================================================

LITERAL_CASES = [
    (LiteralExpression(4), 4),
    (LiteralExpression(1.5), 1.5),
    (LiteralExpression(True), True),
    (LiteralExpression("4"), 4),
]


@pytest.mark.parametrize(
    "literal, expected",
    LITERAL_CASES,
    ids=["int", "float", "bool", "int-string"],
)
def test_evaluates_literal_leaf(
    literal: LiteralExpression, expected: bool | int | float
) -> None:
    """Test literal leaves evaluate to their value, coercing exact string forms."""
    result = evaluate_expression_with_numpy(literal, {})

    assert result == expected


def test_raises_for_float_grammar_string_literal() -> None:
    """Test a float-grammar string literal is refused to avoid precision loss."""
    expression = LiteralExpression("1.5")

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {})

    assert isinstance(exception_info.value.__cause__, StringLiteralPrecisionError)


# =============================================================================
# Native constants
# =============================================================================

CONSTANT_CASES = [
    ("pi", math.pi),
    ("e", math.e),
    ("inf", math.inf),
]


@pytest.mark.parametrize(
    "constant_name, expected",
    CONSTANT_CASES,
    ids=[name for name, _ in CONSTANT_CASES],
)
def test_resolves_native_constant_without_binding(
    constant_name: str, expected: float
) -> None:
    """Test a native-constant reference resolves without an environment entry."""
    constant = mock_identifier(constant_name, 0)
    expression = IdentifierExpression(constant)

    result = evaluate_expression_with_numpy(expression, {})

    assert result == expected


def test_resolves_nan_constant_without_binding() -> None:
    """Test the ``nan`` constant resolves to a NaN value."""
    constant = mock_identifier("nan", 0)
    expression = IdentifierExpression(constant)

    result = evaluate_expression_with_numpy(expression, {})

    assert np.isnan(result)


def test_resolves_native_constant_within_expression() -> None:
    """Test a native constant is usable as an operand alongside a bound array."""
    x = mock_identifier("x", 0)
    pi = mock_identifier("pi", 1)
    expression = IdentifierExpression(x) / IdentifierExpression(pi)
    values = np.array([math.pi, 2.0 * math.pi])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [1.0, 2.0])


# =============================================================================
# Ternary selection
# =============================================================================


def test_evaluates_ternary_as_elementwise_select() -> None:
    """Test a ternary computes absolute value elementwise via select."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = ternary(x_expression > 0.0, x_expression, -x_expression)
    values = np.array([-2.0, 3.0, -4.0, 0.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [2.0, 3.0, 4.0, 0.0])


def test_ternary_selects_between_two_arrays() -> None:
    """Test a ternary selects elementwise between two bound arrays."""
    condition = mock_identifier("c", 0)
    true_values = mock_identifier("t", 1)
    false_values = mock_identifier("f", 2)
    expression = ternary(IdentifierExpression(condition), true_values, false_values)
    mask = np.array([True, False, True])
    on_true = np.array([10.0, 20.0, 30.0])
    on_false = np.array([1.0, 2.0, 3.0])

    result = evaluate_expression_with_numpy(
        expression,
        {condition: mask, true_values: on_true, false_values: on_false},
    )

    assert np.allclose(result, [10.0, 2.0, 30.0])


# =============================================================================
# Auto-inlined expression-bodied built-ins
# =============================================================================


def test_auto_inlines_relu_over_array() -> None:
    """Test ``relu`` is inlined and applied elementwise without pre-inlining."""
    x = mock_identifier("x", 0)
    expression = call("relu", x)
    values = np.array([-1.0, 2.0, -3.0, 4.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [0.0, 2.0, 0.0, 4.0])


def test_auto_inlines_sigmoid_matches_logistic_reference() -> None:
    """Test ``sigmoid`` matches the closed-form logistic function."""
    x = mock_identifier("x", 0)
    expression = call("sigmoid", x)
    values = np.array([-2.0, 0.0, 1.0, 3.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, 1.0 / (1.0 + np.exp(-values)))


def test_auto_inlines_silu_matches_reference() -> None:
    """Test ``silu`` matches ``x * sigmoid(x)``."""
    x = mock_identifier("x", 0)
    expression = call("silu", x)
    values = np.array([-2.0, 0.0, 1.5])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, values * (1.0 / (1.0 + np.exp(-values))))


def test_auto_inlines_clamp_bounds_values() -> None:
    """Test ``clamp`` restricts values to the given range."""
    x = mock_identifier("x", 0)
    expression = call("clamp", x, 0.0, 1.0)
    values = np.array([-1.0, 0.5, 2.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [0.0, 0.5, 1.0])


def test_auto_inlines_leaky_relu_matches_reference() -> None:
    """Test ``leaky_relu`` matches the piecewise reference."""
    x = mock_identifier("x", 0)
    expression = call("leaky_relu", x, 0.1)
    values = np.array([-3.0, -1.0, 2.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, np.where(values > 0.0, values, values * 0.1))


@pytest.mark.parametrize(
    "builtin_name, reference",
    [("max", np.maximum), ("min", np.minimum)],
    ids=["max", "min"],
)
def test_auto_inlines_binary_extremum_builtin(
    builtin_name: str, reference: Callable[[Any, Any], Any]
) -> None:
    """Test ``max``/``min`` select the extreme of two arrays elementwise."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([1.0, 5.0, 3.0])
    right = np.array([4.0, 2.0, 3.0])

    result = evaluate_expression_with_numpy(
        call(builtin_name, x, y), {x: left, y: right}
    )

    assert np.allclose(result, reference(left, right))


@pytest.mark.parametrize(
    "builtin_name, reference",
    [("abs", np.abs), ("sign", np.sign)],
    ids=["abs", "sign"],
)
def test_auto_inlines_unary_math_builtin(
    builtin_name: str, reference: Callable[[Any], Any]
) -> None:
    """Test ``abs``/``sign`` map each element to its NumPy equivalent."""
    x = mock_identifier("x", 0)
    values = np.array([-2.0, 0.0, 5.0])

    result = evaluate_expression_with_numpy(call(builtin_name, x), {x: values})

    assert np.allclose(result, reference(values))


def test_auto_inlines_xor_over_boolean_arrays() -> None:
    """Test the boolean ``xor`` built-in matches exclusive-or elementwise."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    expression = call("xor", a, b)
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])

    result = evaluate_expression_with_numpy(expression, {a: left, b: right})

    assert np.array_equal(result, np.logical_xor(left, right))


def test_evaluates_nested_builtin_composition() -> None:
    """Test a composition of built-ins matches the closed-form reference."""
    x = mock_identifier("x", 0)
    w = mock_identifier("w", 1)
    b = mock_identifier("b", 2)
    inner = call("relu", x) * IdentifierExpression(w) + IdentifierExpression(b)
    expression = call("sigmoid", inner)
    x_values = np.array([-1.0, 0.5, 2.0])
    w_values = np.array([2.0, 2.0, 2.0])
    b_values = np.array([0.1, 0.1, 0.1])

    result = evaluate_expression_with_numpy(
        expression, {x: x_values, w: w_values, b: b_values}
    )

    reference = 1.0 / (1.0 + np.exp(-(np.maximum(x_values, 0.0) * w_values + b_values)))
    assert np.allclose(result, reference)


# =============================================================================
# User-story: author once, apply to a large array
# =============================================================================


def test_author_function_once_and_apply_to_large_array() -> None:
    """Test an authored element-wise function evaluates over a large array."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = call("relu", x_expression * 2.0 - 1.0)
    values = np.linspace(-5.0, 5.0, 1_000_000)

    result = evaluate_expression_with_numpy(expression, {x: values})

    reference = np.maximum(values * 2.0 - 1.0, 0.0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1_000_000,)
    assert np.allclose(result, reference)


def test_large_array_evaluation_stays_near_native_speed() -> None:
    """Test large-array evaluation stays within a constant factor of NumPy."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = x_expression * x_expression + x_expression * 2.0 + 1.0
    values = np.random.default_rng(0).standard_normal(2_000_000)

    def compute_reference(sample: Any) -> Any:
        return sample * sample + sample * 2.0 + 1.0

    compute_reference(values)  # warm up allocation paths
    native_start = time.perf_counter()
    reference = compute_reference(values)
    native_seconds = time.perf_counter() - native_start

    evaluated_start = time.perf_counter()
    result = evaluate_expression_with_numpy(expression, {x: values})
    evaluated_seconds = time.perf_counter() - evaluated_start

    assert np.allclose(result, reference)
    assert evaluated_seconds < max(native_seconds * 100.0, 0.05)


# =============================================================================
# Edge cases
# =============================================================================


def test_empty_array_input_returns_empty_array() -> None:
    """Test an empty array input yields an empty array of the same rank."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) * 2.0 + 1.0

    result = evaluate_expression_with_numpy(expression, {x: np.array([])})

    assert isinstance(result, np.ndarray)
    assert result.shape == (0,)


def test_broadcasts_scalar_literal_over_array() -> None:
    """Test a scalar literal broadcasts across a bound array."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) + 10

    values = np.array([1.0, 2.0, 3.0])
    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, values + 10)


def test_scalar_environment_returns_scalar_value() -> None:
    """Test a purely scalar environment yields a rank-zero value."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) * 2.0

    result = evaluate_expression_with_numpy(expression, {x: 3.0})

    assert not isinstance(result, np.ndarray)
    assert np.ndim(result) == 0
    assert float(result) == 6.0


def test_array_environment_returns_ndarray() -> None:
    """Test an array-valued environment yields an ndarray."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) * 2.0

    result = evaluate_expression_with_numpy(expression, {x: np.array([1.0, 2.0])})

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)


def test_integer_array_true_division_promotes_to_float() -> None:
    """Test integer division follows NumPy promotion to a float result."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) / 2

    result = evaluate_expression_with_numpy(expression, {x: np.array([1, 3, 5])})

    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.floating)
    assert np.allclose(result, [0.5, 1.5, 2.5])


def test_evaluates_over_multidimensional_array() -> None:
    """Test evaluation is shape-agnostic across a two-dimensional array."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x) * 2.0 + 1.0
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.allclose(result, values * 2.0 + 1.0)


def test_ignores_environment_bindings_not_free_in_expression() -> None:
    """Test bindings for identifiers absent from the expression are ignored."""
    x = mock_identifier("x", 0)
    unused = mock_identifier("unused", 1)
    expression = IdentifierExpression(x) + 1.0
    values = np.array([1.0, 2.0])

    result = evaluate_expression_with_numpy(
        expression, {x: values, unused: np.array([99.0, 99.0])}
    )

    assert np.allclose(result, values + 1.0)


# =============================================================================
# Adversarial cases
# =============================================================================


def test_raises_for_unbound_variable() -> None:
    """Test an unbound, non-constant variable raises ``UnboundVariableError``."""
    x = mock_identifier("unbound", 0)
    expression = IdentifierExpression(x) + 1.0

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {})

    assert isinstance(exception_info.value.__cause__, UnboundVariableError)


def test_raises_for_unbound_identifier_matching_a_native_function_name() -> None:
    """Test an unbound identifier named like a native function is not resolved."""
    exp_reference = mock_identifier("exp", 0)
    expression = IdentifierExpression(exp_reference) + 1.0

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {})

    assert isinstance(exception_info.value.__cause__, UnboundVariableError)


def test_raises_for_unsupported_erf() -> None:
    """Test ``erf`` raises ``UnsupportedNumpyLoweringError``."""
    x = mock_identifier("x", 0)
    expression = call("erf", x)

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {x: np.array([0.0])})

    assert isinstance(exception_info.value.__cause__, UnsupportedNumpyLoweringError)


def test_raises_for_gelu_due_to_unsupported_erf() -> None:
    """Test ``gelu`` raises because it inlines to an ``erf`` call."""
    x = mock_identifier("x", 0)
    expression = call("gelu", x)

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {x: np.array([0.0])})

    assert isinstance(exception_info.value.__cause__, UnsupportedNumpyLoweringError)


def test_raises_for_unregistered_function_name() -> None:
    """Test a call to an unregistered name surfaces ``EntryLookupError``."""
    x = mock_identifier("x", 0)
    expression = call("np_eval_never_registered", x)

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {x: np.array([0.0])})

    assert isinstance(exception_info.value.__cause__, EntryLookupError)


def test_raises_when_calling_native_constant() -> None:
    """Test calling a native constant surfaces ``FunctionArityError``."""
    expression = CallExpression("pi", ())

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {})

    assert isinstance(exception_info.value.__cause__, FunctionArityError)


def test_raises_for_native_function_without_numpy_mapping(
    function_registry_snapshot: None,
) -> None:
    """Test a non-built-in native function has no NumPy lowering."""
    register_native_function(
        "np_eval_atan2",
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.atan2,
    )
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = call("np_eval_atan2", x, y)

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(
            expression, {x: np.array([1.0]), y: np.array([1.0])}
        )

    assert isinstance(exception_info.value.__cause__, UnsupportedNumpyLoweringError)


def test_raises_for_recursive_function(
    function_registry_snapshot: None,
) -> None:
    """Test a transitively-recursive function surfaces ``RecursionError``.

    Inlining runs before the walk; a self-recursive registration cannot be
    inlined, so the inliner's recursion guard raises ``RecursionError``,
    wrapped as ``PassExecutionError``.
    """
    x = mock_identifier("x", 0)
    register_function(
        "np_eval_recursive",
        parameters=[x],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=call("np_eval_recursive", x),
    )
    expression = call("np_eval_recursive", IdentifierExpression(x))

    with pytest.raises(PassExecutionError) as exception_info:
        evaluate_expression_with_numpy(expression, {x: np.array([1.0])})

    assert isinstance(exception_info.value.__cause__, RecursionError)


def test_raises_import_error_when_numpy_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a missing NumPy raises ``ImportError`` with install guidance."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x)

    with pytest.raises(ImportError, match=r"fhy_core\[numpy\]"):
        evaluate_expression_with_numpy(expression, {x: [1.0, 2.0]})


def test_does_not_mutate_input_expression() -> None:
    """Test the input expression tree is unchanged after evaluation."""
    x = mock_identifier("x", 0)
    expression = call("relu", IdentifierExpression(x) * 2.0)
    reference = call("relu", IdentifierExpression(x) * 2.0)

    evaluate_expression_with_numpy(expression, {x: np.array([1.0, -1.0])})

    assert expression.is_structurally_equivalent(reference)


# =============================================================================
# Lowering-table coverage
# =============================================================================

_UNSUPPORTED_NATIVE_FUNCTION_NAMES = frozenset({"erf"})


def test_every_lowered_native_name_resolves_to_a_numpy_callable() -> None:
    """Test every lowering-table value names a real, callable NumPy attribute.

    A typo'd ufunc name would otherwise surface only as an ``AttributeError``
    at call time; this makes the drift a fast, deterministic failure.
    """
    for ufunc_name in _NATIVE_FUNCTION_UFUNC_NAMES.values():
        assert callable(getattr(np, ufunc_name))


def test_every_builtin_native_function_has_a_lowering() -> None:
    """Test every built-in native (besides ``erf``) has a NumPy lowering.

    Guards against a newly-registered built-in native silently falling
    through to ``UnsupportedNumpyLoweringError`` because nobody added it to
    the lowering table.
    """
    native_function_names = {
        entry.name
        for entry in get_registered_entries().values()
        if isinstance(entry, NativeFunction)
    }
    unmapped = (
        native_function_names
        - set(_NATIVE_FUNCTION_UFUNC_NAMES)
        - _UNSUPPORTED_NATIVE_FUNCTION_NAMES
    )

    assert not unmapped
