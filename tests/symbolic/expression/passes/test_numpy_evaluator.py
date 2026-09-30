"""Tests for `fhy_core.symbolic.expression.passes.numpy`."""

import contextlib
import math
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    EntryLookupError,
    Expression,
    FunctionArityError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    LogicalExpression,
    LogicalOperation,
    NativeConstantBindingError,
    NativeFunction,
    NonBooleanLogicalOperandError,
    NonFiniteCastError,
    StringLiteralPrecisionError,
    UnaryExpression,
    UnaryOperation,
    UnboundVariableError,
    UnsupportedNumpyLoweringError,
    call,
    evaluate_expression_with_numpy,
    get_native_constant_identifier,
    get_registered_entries,
    piecewise,
    register_function,
    register_native_function,
)
from fhy_core.symbolic.expression.passes.numpy import NumpyExpressionEvaluator
from fhy_core.symbolic.solver import simplify_expression

from ..conftest import mock_identifier

pytestmark = pytest.mark.numpy

np = pytest.importorskip("numpy")


@contextlib.contextmanager
def _refuse_warnings() -> Iterator[None]:
    """Turn every warning into an error: the evaluator warns nothing.

    Real arithmetic is IEEE's, silently, and the NumPy ufuncs the
    evaluator calls for the transcendental natives run with NumPy's
    floating-point warnings silenced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


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

LOGICAL_CASES = [
    (LogicalOperation.AND, np.logical_and),
    (LogicalOperation.OR, np.logical_or),
]


@pytest.mark.parametrize(
    "operation, reference",
    LOGICAL_CASES,
    ids=[operation.value for operation, _ in LOGICAL_CASES],
)
def test_evaluates_logical_expression_elementwise(
    operation: LogicalOperation, reference: Callable[[Any, Any], Any]
) -> None:
    """Test logical and/or evaluate elementwise over boolean arrays."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    expression = LogicalExpression(
        operation, (IdentifierExpression(x), IdentifierExpression(y))
    )

    result = evaluate_expression_with_numpy(expression, {x: left, y: right})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, reference(left, right))


@pytest.mark.parametrize(
    "operation, reference",
    LOGICAL_CASES,
    ids=[operation.value for operation, _ in LOGICAL_CASES],
)
def test_evaluates_three_operand_logical_expression_elementwise(
    operation: LogicalOperation, reference: Callable[[Any, Any], Any]
) -> None:
    """Test an n-ary logical expression reduces all its operands elementwise."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    first = np.array([True, True, True, True, False, False, False, False])
    second = np.array([True, True, False, False, True, True, False, False])
    third = np.array([True, False, True, False, True, False, True, False])
    expression = LogicalExpression(
        operation,
        (IdentifierExpression(x), IdentifierExpression(y), IdentifierExpression(z)),
    )

    result = evaluate_expression_with_numpy(expression, {x: first, y: second, z: third})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_
    assert np.array_equal(result, reference(reference(first, second), third))


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
# A provably numeric connective operand is refused, not read by truthiness
# =============================================================================


def test_logical_and_of_two_int_literals_raises_directly() -> None:
    """Test `logical_and(2, 4)` raises rather than being read as `True`.

    `numpy.logical_and` treats any nonzero value as true, so an
    unscreened lowering would silently accept two ill-typed integer
    operands and hand back a `True`-valued answer that means nothing.
    """
    expression = LogicalExpression(
        LogicalOperation.AND, (LiteralExpression(2), LiteralExpression(4))
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        evaluate_expression_with_numpy(expression, {})


def test_logical_not_of_a_falsy_int_literal_raises_directly() -> None:
    """Test `logical_not(0)` raises rather than being read as `True`."""
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(0))

    with pytest.raises(NonBooleanLogicalOperandError):
        evaluate_expression_with_numpy(expression, {})


def test_logical_and_of_two_int_bound_identifiers_raises_directly() -> None:
    """Test an int-dtype binding is screened as `INT`, not treated as boolean.

    The static check reads a bound Python `int`'s dtype the same way it
    would read an int-dtype array's, so a scalar binding is screened
    exactly like an array one.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = LogicalExpression(
        LogicalOperation.AND, (IdentifierExpression(x), IdentifierExpression(y))
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        evaluate_expression_with_numpy(expression, {x: 2, y: 4})


def test_logical_and_of_two_float_arrays_raises_directly() -> None:
    """Test a float-dtype array binding under `logical_and` is refused."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = LogicalExpression(
        LogicalOperation.AND, (IdentifierExpression(x), IdentifierExpression(y))
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        evaluate_expression_with_numpy(
            expression, {x: np.array([1.0, 0.0]), y: np.array([1.0, 1.0])}
        )


def test_logical_not_of_an_object_dtype_array_is_refused_when_converted() -> None:
    """Test an object-dtype binding is refused before the evaluation.

    The evaluator computes in the Boolean, ``int64`` and ``float64``
    domains, so a binding of any other kind is refused with ``TypeError``
    when it is converted, and no operand is left to a runtime guard.
    """
    x = mock_identifier("x", 0)
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(x))

    with pytest.raises(TypeError, match="dtype object"):
        evaluate_expression_with_numpy(expression, {x: np.array([1, 2], dtype=object)})


def test_logical_connectives_still_evaluate_boolean_literals() -> None:
    """Test bare Python bool literals under `logical_and`/`logical_not` still work."""
    conjunction = LogicalExpression(
        LogicalOperation.AND, (LiteralExpression(True), LiteralExpression(False))
    )
    negation = UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True))

    assert bool(evaluate_expression_with_numpy(conjunction, {})) is False
    assert bool(evaluate_expression_with_numpy(negation, {})) is False


def test_arithmetic_is_unaffected_by_the_connective_dtype_screen() -> None:
    """Test plain arithmetic over an int-bound identifier is unaffected by the guard."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )

    result = evaluate_expression_with_numpy(expression, {x: 2})

    assert result == 3


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

    An integer-sorted result is an ``int64``, agreeing with
    ``evaluate_expression`` and the declared sort.
    """
    x = mock_identifier("x", 0)
    values = np.array([2.7, -1.2, 3.5])
    expression = call(function_name, x)

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.integer)


def test_real_sort_native_widens_a_float32_binding_to_float64() -> None:
    """Test a ``float32`` binding is read as ``float64``, the real domain.

    The evaluator computes reals in ``float64``, so a narrower binding is
    widened and a real-sorted result is ``float64``.
    """
    x = mock_identifier("x", 0)
    values = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    expression = call("sqrt", x)

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert np.array_equal(result, np.sqrt(values.astype(np.float64)))


def test_native_results_take_the_dtype_of_their_result_sort() -> None:
    """Test an ``INT``-sorted native gives ``int64`` and a ``REAL`` one ``float64``.

    The cast of each result sort is specified by the Rust evaluator's
    stories; no built-in native is ``NAT``- or ``BOOL``-sorted, so Python
    reaches the ``INT`` and ``REAL`` casts only.
    """
    x = mock_identifier("x", 0)
    values = np.array([1.5, 2.5], dtype=np.float32)

    rounded = evaluate_expression_with_numpy(call("round", x), {x: values})
    root = evaluate_expression_with_numpy(call("sqrt", x), {x: values})

    assert rounded.dtype == np.int64
    assert np.array_equal(rounded, [2, 2])
    assert root.dtype == np.float64


@pytest.mark.parametrize("function_name", _INTEGER_SORT_FUNCTION_NAMES)
def test_non_finite_integer_sort_result_raises_non_finite_cast_error(
    function_name: str,
) -> None:
    """Test ``nan``/``inf`` through an ``INT``-sorted native raises.

    A non-finite float has no faithful integer representation; casting
    it to ``int64`` would silently produce a platform-defined sentinel,
    so the evaluator raises ``NonFiniteCastError`` instead.
    """
    x = mock_identifier("x", 0)
    values = np.array([np.inf, -np.inf, np.nan])
    expression = call(function_name, x)

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: values})


def test_partially_non_finite_integer_sort_result_raises() -> None:
    """Test a single ``nan`` among otherwise-finite values still raises.

    The finiteness check applies elementwise across the whole result,
    not only when every element is non-finite.
    """
    x = mock_identifier("x", 0)
    values = np.array([2.7, np.nan, 3.5])
    expression = call("round", x)

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: values})


def test_floor_of_sqrt_of_negative_raises_non_finite_cast_error() -> None:
    """Test ``floor(sqrt(-1))`` raises: ``sqrt`` yields ``nan``, then the cast fails."""
    x = mock_identifier("x", 0)
    expression = call("floor", call("sqrt", x))

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: np.array([-1.0])})


def test_round_of_reciprocal_at_zero_raises_non_finite_cast_error() -> None:
    """Test ``round(1 / x)`` at ``x = 0`` raises: the division yields ``inf``."""
    x = mock_identifier("x", 0)
    expression = call("round", LiteralExpression(1.0) / IdentifierExpression(x))

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: np.array([0.0])})


def test_piecewise_condition_guards_the_non_finite_cast_of_its_own_branch() -> None:
    """Test a condition guarding a domain error suppresses the non-finite cast.

    ``floor(sqrt(x))`` is ``nan`` for negative ``x`` and integer-sorted,
    so evaluating it unguarded raises. Guarding it with ``x >= 0`` is the
    idiomatic way to express the partial function, and must yield the
    fallback for the negative elements rather than raising: the branch
    ``numpy.where`` discards, cannot poison the result.
    """
    x = mock_identifier("x", 0)
    expression = piecewise(
        (
            IdentifierExpression(x) >= LiteralExpression(0.0),
            call("floor", call("sqrt", x)),
        ),
        otherwise=LiteralExpression(0),
    )

    with _refuse_warnings():
        result = evaluate_expression_with_numpy(
            expression, {x: np.array([-1.0, 4.0, 9.0])}
        )

    np.testing.assert_array_equal(result, np.array([0, 2, 3]))


def test_piecewise_raises_when_the_selected_branch_is_non_finite() -> None:
    """Test the non-finite guard still fires for an element the branch does return.

    The companion to the guarded case: here the condition selects the
    ``floor(sqrt(x))`` branch for a negative element, so the ``nan`` is
    not discarded and casting it to ``int64`` would produce a platform
    sentinel. Deferring the check must not weaken it.
    """
    x = mock_identifier("x", 0)
    expression = piecewise(
        (
            IdentifierExpression(x) < LiteralExpression(100.0),
            call("floor", call("sqrt", x)),
        ),
        otherwise=LiteralExpression(0),
    )

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: np.array([-1.0, 4.0])})


def _make_unguarded_sqrt_piecewise(x: Identifier) -> Expression:
    """Build a piecewise that returns ``floor(sqrt(x))`` for every input.

    Its own condition excludes nothing in the ranges used below, so the
    node's selected value is non-finite wherever ``x`` is negative.
    """
    return piecewise(
        (
            IdentifierExpression(x) < LiteralExpression(100.0),
            call("floor", call("sqrt", x)),
        ),
        otherwise=LiteralExpression(0),
    )


def test_outer_piecewise_condition_guards_a_nested_piecewise() -> None:
    """Test an enclosing condition suppresses a nested piecewise's non-finite value.

    The inner node's own selection is poisoned for the negative element,
    but the outer condition discards that element entirely, so nothing
    should raise. Deciding at the inner node would report an error about
    a value the caller never receives.
    """
    x = mock_identifier("x", 0)
    expression = piecewise(
        (
            IdentifierExpression(x) >= LiteralExpression(0.0),
            _make_unguarded_sqrt_piecewise(x),
        ),
        otherwise=LiteralExpression(-1),
    )

    with _refuse_warnings():
        result = evaluate_expression_with_numpy(expression, {x: np.array([-1.0, 4.0])})

    np.testing.assert_array_equal(result, np.array([-1, 2]))


def test_nested_piecewise_still_raises_when_the_outer_branch_selects_it() -> None:
    """Test propagating the nested mask upward does not lose the error.

    Companion to the guarded nesting case: here the outer condition
    selects the inner node for the negative element too, so the
    non-finite value does reach the caller and must raise.
    """
    x = mock_identifier("x", 0)
    expression = piecewise(
        (
            IdentifierExpression(x) < LiteralExpression(1000.0),
            _make_unguarded_sqrt_piecewise(x),
        ),
        otherwise=LiteralExpression(-1),
    )

    with _refuse_warnings():
        with pytest.raises(NonFiniteCastError):
            evaluate_expression_with_numpy(expression, {x: np.array([-1.0, 4.0])})


# =============================================================================
# Floating-point domain conditions (nan/inf, not exceptions)
# =============================================================================


def test_sqrt_of_negative_returns_nan_without_raising() -> None:
    """Test ``sqrt`` of a negative yields ``nan`` rather than raising.

    This is the defining divergence from ``evaluate_expression``, whose
    scalar native implementations raise on the same input.
    """
    x = mock_identifier("x", 0)
    expression = call("sqrt", x)

    with _refuse_warnings():
        result = evaluate_expression_with_numpy(expression, {x: np.array([-1.0])})

    assert np.isnan(result).all()


def test_log_of_zero_returns_negative_infinity_without_raising() -> None:
    """Test ``log`` of zero yields ``-inf`` rather than raising."""
    x = mock_identifier("x", 0)
    expression = call("log", x)

    with _refuse_warnings():
        result = evaluate_expression_with_numpy(expression, {x: np.array([0.0])})

    assert np.isneginf(result).all()


def test_division_by_zero_returns_infinity_without_raising() -> None:
    """Test division by zero yields ``inf`` rather than raising."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = IdentifierExpression(x) / IdentifierExpression(y)

    with _refuse_warnings():
        result = evaluate_expression_with_numpy(
            expression, {x: np.array([1.0]), y: np.array([0.0])}
        )

    assert np.isposinf(result).all()


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


EXACT_BINARY_FLOAT_STRING_CASES = [
    ("0.5", 0.5),
    ("0.25", 0.25),
]


@pytest.mark.parametrize(
    "value, expected",
    EXACT_BINARY_FLOAT_STRING_CASES,
    ids=[value for value, _ in EXACT_BINARY_FLOAT_STRING_CASES],
)
def test_evaluates_float_grammar_string_literal_with_exact_binary_value(
    value: str, expected: float
) -> None:
    """Test a float-grammar string literal that is exactly a binary float."""
    result = evaluate_expression_with_numpy(LiteralExpression(value), {})

    assert result == expected


def test_evaluates_negated_float_grammar_string_literal_with_exact_binary_value() -> (
    None
):
    """Test negating an exact-binary-value string literal evaluates to its negative."""
    expression = -LiteralExpression("0.5")

    result = evaluate_expression_with_numpy(expression, {})

    assert result == -0.5


@pytest.mark.sympy
def test_evaluates_simplified_half_division_of_a_bound_variable() -> None:
    """Test a simplified division-by-two literal evaluates without precision loss."""
    x = mock_identifier("x", 0)
    expression = simplify_expression(IdentifierExpression(x) / LiteralExpression(2))

    result = evaluate_expression_with_numpy(expression, {x: 3.0})

    assert result == 1.5


LOSSY_FLOAT_GRAMMAR_STRING_CASES = [
    "0.1",
    "0.3",
    "0.1000000000000000055511151231257827",
]


@pytest.mark.parametrize(
    "value",
    LOSSY_FLOAT_GRAMMAR_STRING_CASES,
    ids=["repeating-tenth", "repeating-third", "long-inexact-decimal"],
)
def test_raises_for_float_grammar_string_literal_with_no_exact_binary_value(
    value: str,
) -> None:
    """Test a float-grammar string literal with no exact binary value is refused."""
    expression = LiteralExpression(value)

    with pytest.raises(StringLiteralPrecisionError):
        evaluate_expression_with_numpy(expression, {})


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
    constant = get_native_constant_identifier(constant_name)
    expression = IdentifierExpression(constant)

    result = evaluate_expression_with_numpy(expression, {})

    assert result == expected


def test_resolves_nan_constant_without_binding() -> None:
    """Test the ``nan`` constant resolves to a NaN value."""
    constant = get_native_constant_identifier("nan")
    expression = IdentifierExpression(constant)

    result = evaluate_expression_with_numpy(expression, {})

    assert np.isnan(result)


def test_resolves_native_constant_within_expression() -> None:
    """Test a native constant is usable as an operand alongside a bound array."""
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(x) / IdentifierExpression(pi)
    values = np.array([math.pi, 2.0 * math.pi])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [1.0, 2.0])


def test_binds_an_identifier_merely_named_like_a_constant_from_environment() -> None:
    """Test an identifier that only shares ``pi``'s name takes the bound value.

    Constant resolution keys on the canonical identifier, so this
    identifier is an ordinary variable and the environment binding is
    what decides its value.
    """
    pi_lookalike = mock_identifier("pi", 832)
    expression = IdentifierExpression(pi_lookalike) * 2.0

    result = evaluate_expression_with_numpy(
        expression, {pi_lookalike: np.array([1.0, 3.0])}
    )

    assert np.allclose(result, [2.0, 6.0])


def test_raises_for_unbound_identifier_merely_named_like_a_native_constant() -> None:
    """Test an unbound identifier that shares a constant's name is not resolved.

    The message says the identifier is distinct from the constant it is
    named after, rather than claiming the name is a registered function.
    """
    pi_lookalike = mock_identifier("pi", 833)
    expression = IdentifierExpression(pi_lookalike) + 1.0

    with pytest.raises(UnboundVariableError, match="shares its name with the constant"):
        evaluate_expression_with_numpy(expression, {})


def test_raises_for_a_binding_that_shadows_a_referenced_native_constant() -> None:
    """Test binding pi's canonical identifier is refused when pi is referenced.

    Without the refusal, the evaluator would read the environment before
    checking for a constant and silently prefer the caller's value over
    the constant's, unlike the SymPy bridge, which resolves the constant
    by identity and never consults the environment for it.
    """
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(pi) + 1.0

    with pytest.raises(NativeConstantBindingError, match="pi"):
        evaluate_expression_with_numpy(expression, {pi: 3.0})


def test_ignores_a_binding_for_an_unreferenced_native_constant() -> None:
    """Test a binding for pi is ignored when the expression does not reference pi."""
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(x) + 1.0

    result = evaluate_expression_with_numpy(expression, {x: 2.0, pi: 3.0})

    assert result == 3.0


def test_raises_for_a_binding_shadowing_a_constant_referenced_inside_an_inlined_body(
    function_registry_snapshot: None,
) -> None:
    """Test the refusal accounts for a constant referenced only inside an inlined body.

    The evaluator inlines expression-bodied function calls before
    walking the tree, so a constant reference hidden inside a called
    function's body must still be caught even though it is absent from
    the un-inlined call expression.
    """
    pi = get_native_constant_identifier("pi")
    register_function(
        "np_eval_scaled_by_pi",
        parameters=[],
        parameter_sorts=[],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(pi),
    )
    expression = call("np_eval_scaled_by_pi")

    with pytest.raises(NativeConstantBindingError, match="pi"):
        evaluate_expression_with_numpy(expression, {pi: 3.0})


# =============================================================================
# Piecewise selection
# =============================================================================


def test_evaluates_single_case_piecewise_as_elementwise_select() -> None:
    """Test a one-case piecewise computes absolute value elementwise via select."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = piecewise((x_expression > 0.0, x_expression), otherwise=-x_expression)
    values = np.array([-2.0, 3.0, -4.0, 0.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [2.0, 3.0, 4.0, 0.0])


def test_single_case_piecewise_selects_between_two_arrays() -> None:
    """Test a one-case piecewise selects elementwise between two bound arrays."""
    condition = mock_identifier("c", 0)
    true_values = mock_identifier("t", 1)
    false_values = mock_identifier("f", 2)
    expression = piecewise(
        (IdentifierExpression(condition), true_values), otherwise=false_values
    )
    mask = np.array([True, False, True])
    on_true = np.array([10.0, 20.0, 30.0])
    on_false = np.array([1.0, 2.0, 3.0])

    result = evaluate_expression_with_numpy(
        expression,
        {condition: mask, true_values: on_true, false_values: on_false},
    )

    assert np.allclose(result, [10.0, 2.0, 30.0])


def test_multi_case_piecewise_selects_first_matching_case_per_element() -> None:
    """Test the numpy lowering is first-match-wins under overlapping conditions.

    Every element satisfies both ``x > -100.0`` and ``x > 0.0``; the first
    case in declaration order must win, not the last.
    """
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = piecewise(
        (x_expression > -100.0, LiteralExpression(1)),
        (x_expression > 0.0, LiteralExpression(2)),
        otherwise=LiteralExpression(3),
    )
    values = np.array([-5.0, 5.0, 50.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.array_equal(result, [1, 1, 1])


def test_multi_case_piecewise_falls_through_to_second_case() -> None:
    """Test an element failing the first case is selected by the second case."""
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = piecewise(
        (x_expression > 0.0, LiteralExpression(1)),
        (x_expression < 0.0, LiteralExpression(-1)),
        otherwise=LiteralExpression(0),
    )
    values = np.array([5.0, -5.0, 0.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.array_equal(result, [1, -1, 0])


def test_piecewise_with_duplicate_identical_conditions_selects_first() -> None:
    """Test two cases with the exact same condition still resolve first-match-wins.

    Distinct from overlapping-but-different conditions: both cases here
    test the identical predicate, so only declaration order distinguishes
    them.
    """
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    expression = piecewise(
        (x_expression > 0.0, LiteralExpression(1)),
        (x_expression > 0.0, LiteralExpression(2)),
        otherwise=LiteralExpression(0),
    )
    values = np.array([-1.0, 1.0])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.array_equal(result, [0, 1])


def test_piecewise_with_over_one_hundred_cases_selects_correctly() -> None:
    """Test a piecewise with well over one hundred cases selects the right value.

    Guards the right-folded ``numpy.where`` chain against case counts far
    beyond the two- and three-case examples used elsewhere in this file.
    """
    NUM_CASES = 150
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    cases = tuple(
        (x_expression.equals(float(i)), LiteralExpression(i)) for i in range(NUM_CASES)
    )
    expression = piecewise(*cases, otherwise=LiteralExpression(-1))
    values = np.array([0.0, 37.0, 149.0, 150.0, 75.5])

    result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.array_equal(result, [0, 37, 149, -1, -1])


def test_unselected_case_domain_error_is_discarded_without_a_warning() -> None:
    """Test a domain error in an unselected case is discarded, and warns nothing.

    Every case is evaluated for every element, so ``log(0)`` in a case
    whose condition is always false still computes ``-inf`` there. The
    element is discarded, and real arithmetic is IEEE's, with no NumPy
    ``RuntimeWarning``.
    """
    x = mock_identifier("x", 0)
    x_expression = IdentifierExpression(x)
    never_true = x_expression > 1000.0
    expression = piecewise(
        (never_true, call("log", x_expression)), otherwise=LiteralExpression(0.0)
    )
    values = np.array([0.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_expression_with_numpy(expression, {x: values})

    assert np.allclose(result, [0.0])


def test_piecewise_broadcasts_scalar_case_value_against_array_otherwise() -> None:
    """Test a scalar case value broadcasts against an array-valued ``otherwise``."""
    condition = mock_identifier("c", 0)
    otherwise_values = mock_identifier("o", 1)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(9.0)),
        otherwise=IdentifierExpression(otherwise_values),
    )
    mask = np.array([True, False, True])
    otherwise_array = np.array([1.0, 2.0, 3.0])

    result = evaluate_expression_with_numpy(
        expression, {condition: mask, otherwise_values: otherwise_array}
    )

    assert np.allclose(result, [9.0, 2.0, 9.0])


def test_piecewise_broadcasts_array_case_value_against_scalar_otherwise() -> None:
    """Test an array-valued case value broadcasts against a scalar ``otherwise``."""
    condition = mock_identifier("c", 0)
    case_values = mock_identifier("v", 1)
    expression = piecewise(
        (IdentifierExpression(condition), IdentifierExpression(case_values)),
        otherwise=LiteralExpression(-1.0),
    )
    mask = np.array([True, False, True])
    values = np.array([10.0, 20.0, 30.0])

    result = evaluate_expression_with_numpy(
        expression, {condition: mask, case_values: values}
    )

    assert np.allclose(result, [10.0, -1.0, 30.0])


def test_piecewise_with_non_boolean_condition_array_raises() -> None:
    """Test a non-boolean-dtyped condition raises rather than being silently truthy.

    ``numpy.where`` treats any nonzero value as true, so an int-typed
    condition bound directly (not built from a comparison) would
    otherwise silently "work" while hiding a condition the type checker
    would reject as non-boolean.
    """
    condition = mock_identifier("c", 0)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )
    values = np.array([0, 1, 2])

    with pytest.raises(
        NonBooleanLogicalOperandError, match="condition of piecewise case 0"
    ):
        evaluate_expression_with_numpy(expression, {condition: values})


def test_piecewise_with_object_dtype_condition_array_is_refused_when_converted() -> (
    None
):
    """Test a condition bound to an object-dtype array is refused as a binding."""
    condition = mock_identifier("c", 0)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )
    values = np.array([0, 1, 2], dtype=object)

    with pytest.raises(TypeError, match="dtype object"):
        evaluate_expression_with_numpy(expression, {condition: values})


def test_piecewise_with_boolean_condition_array_is_unaffected() -> None:
    """Test a genuinely boolean-dtyped condition is unaffected by the dtype guard."""
    condition = mock_identifier("c", 0)
    expression = piecewise(
        (IdentifierExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )
    mask = np.array([True, False, True])

    result = evaluate_expression_with_numpy(expression, {condition: mask})

    assert np.array_equal(result, [1, 0, 1])


@pytest.mark.parametrize(
    "condition, expected",
    [(True, 1), (False, 0)],
    ids=["true_selects_case", "false_selects_otherwise"],
)
def test_piecewise_accepts_a_boolean_literal_condition(
    condition: bool, expected: int
) -> None:
    """Test a boolean ``LiteralExpression`` condition is accepted by the dtype guard.

    Such a condition lowers to a raw Python ``bool``, which has no
    ``dtype`` at all, so the boolean-dtype guard has to admit it
    explicitly. Every other piecewise test supplies a condition derived
    from a comparison or a bound array, leaving this branch unexercised
    even though ``PiecewiseExpression`` accepts a boolean literal.
    """
    expression = piecewise(
        (LiteralExpression(condition), LiteralExpression(1)),
        otherwise=LiteralExpression(0),
    )

    result = evaluate_expression_with_numpy(expression, {})

    assert result == expected


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


def test_ignores_a_ragged_binding_not_free_in_expression() -> None:
    """Test a ragged, non-array-convertible binding is ignored when unreferenced.

    An unreferenced binding is documented to be ignored, so it must not be
    coerced with ``numpy.asarray`` at all: a ragged nested sequence raises
    ``ValueError`` from that coercion alone, regardless of whether anything
    in the expression ever reads it.
    """
    x = mock_identifier("x", 0)
    unused = mock_identifier("unused", 1)
    expression = IdentifierExpression(x) + LiteralExpression(1)
    values = np.array([1, 2])

    result = evaluate_expression_with_numpy(
        expression, {x: values, unused: [[1, 2], [3]]}
    )

    assert np.array_equal(result, np.array([2, 3]))


# =============================================================================
# Adversarial cases
# =============================================================================


def test_raises_for_unbound_variable() -> None:
    """Test an unbound, non-constant variable raises ``UnboundVariableError``."""
    x = mock_identifier("unbound", 0)
    expression = IdentifierExpression(x) + 1.0

    with pytest.raises(UnboundVariableError):
        evaluate_expression_with_numpy(expression, {})


def test_raises_for_unbound_identifier_matching_a_native_function_name() -> None:
    """Test an unbound identifier named like a native function is not resolved.

    The error message distinguishes this from an unknown name: it says the
    name is a function's and points the caller at the call form.
    """
    exp_reference = mock_identifier("exp", 0)
    expression = IdentifierExpression(exp_reference) + 1.0

    with pytest.raises(UnboundVariableError, match=r"names a function.*exp\(\.\.\.\)"):
        evaluate_expression_with_numpy(expression, {})


def test_evaluates_erf_as_math_does() -> None:
    """Test ``erf`` is computed although NumPy has no ufunc for it.

    ``erf`` has no NumPy ufunc, so the core computes it.
    """
    x = mock_identifier("x", 0)
    values = np.array([-1.0, 0.0, 0.5, 2.0])

    result = evaluate_expression_with_numpy(call("erf", x), {x: values})

    assert np.allclose(result, [math.erf(value) for value in values], rtol=1e-15)


def test_evaluates_gelu_through_erf() -> None:
    """Test ``gelu`` inlines to its body and evaluates, ``erf`` included."""
    x = mock_identifier("x", 0)
    values = np.array([-1.0, 0.0, 1.5])

    result = evaluate_expression_with_numpy(call("gelu", x), {x: values})

    expected = [
        0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))) for value in values
    ]
    assert np.allclose(result, expected, rtol=1e-15)


def test_raises_for_unregistered_function_name() -> None:
    """Test a call to an unregistered name surfaces ``EntryLookupError``."""
    x = mock_identifier("x", 0)
    expression = call("np_eval_never_registered", x)

    with pytest.raises(EntryLookupError):
        evaluate_expression_with_numpy(expression, {x: np.array([0.0])})


def test_raises_when_calling_native_constant() -> None:
    """Test calling a native constant surfaces ``FunctionArityError``."""
    expression = CallExpression("pi", ())

    with pytest.raises(FunctionArityError):
        evaluate_expression_with_numpy(expression, {})


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

    with pytest.raises(UnsupportedNumpyLoweringError):
        evaluate_expression_with_numpy(
            expression, {x: np.array([1.0]), y: np.array([1.0])}
        )


def test_integer_base_to_negative_integer_power_raises_value_error() -> None:
    """Test an integer base to a negative integer power raises ``ValueError``.

    An integer power stays an integer, and no integer is ``2 ** -1``, so
    the element fails, and the error names the node.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(x), LiteralExpression(-1)
    )

    with pytest.raises(ValueError, match="negative integer power") as exception_info:
        evaluate_expression_with_numpy(expression, {x: np.array([2, 3])})

    assert type(exception_info.value) is ValueError


def test_raises_for_recursive_function(
    function_registry_snapshot: None,
) -> None:
    """Test a transitively-recursive function surfaces ``RecursionError``.

    Inlining runs before the walk; a self-recursive registration cannot be
    inlined, so the inliner's recursion guard raises ``RecursionError``.
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

    with pytest.raises(RecursionError):
        evaluate_expression_with_numpy(expression, {x: np.array([1.0])})


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
# Every operation and native built-in is evaluated
# =============================================================================


@pytest.mark.parametrize("operation", list(BinaryOperation))
def test_every_binary_operation_is_evaluated(operation: BinaryOperation) -> None:
    """Test every ``BinaryOperation`` evaluates over arrays, as NumPy does.

    The core defines each operation, so this compares every operation with
    NumPy on values where the two agree.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = np.array([6.0, -4.0, 9.0])
    right = np.array([2.0, 3.0, -3.0])
    reference = {
        BinaryOperation.ADD: np.add,
        BinaryOperation.SUBTRACT: np.subtract,
        BinaryOperation.MULTIPLY: np.multiply,
        BinaryOperation.DIVIDE: np.true_divide,
        BinaryOperation.FLOOR_DIVIDE: np.floor_divide,
        BinaryOperation.MODULO: np.mod,
        BinaryOperation.POWER: np.power,
        BinaryOperation.EQUAL: np.equal,
        BinaryOperation.NOT_EQUAL: np.not_equal,
        BinaryOperation.LESS: np.less,
        BinaryOperation.LESS_EQUAL: np.less_equal,
        BinaryOperation.GREATER: np.greater,
        BinaryOperation.GREATER_EQUAL: np.greater_equal,
    }[operation]
    expression = BinaryExpression(
        operation, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = evaluate_expression_with_numpy(expression, {x: left, y: right})

    assert np.array_equal(result, reference(left, right))


def test_every_unary_operation_is_evaluated() -> None:
    """Test every ``UnaryOperation`` evaluates over arrays."""
    x = mock_identifier("x", 0)
    p = mock_identifier("p", 1)
    values = np.array([1.5, -2.0])
    flags = np.array([True, False])
    results = {
        UnaryOperation.NEGATE: (IdentifierExpression(x), -values),
        UnaryOperation.POSITIVE: (IdentifierExpression(x), values),
        UnaryOperation.LOGICAL_NOT: (IdentifierExpression(p), ~flags),
    }

    assert set(results) == set(UnaryOperation)
    for operation, (operand, expected) in results.items():
        result = evaluate_expression_with_numpy(
            UnaryExpression(operation, operand), {x: values, p: flags}
        )
        assert np.array_equal(result, expected), operation


def test_every_builtin_native_function_is_evaluated() -> None:
    """Test every built-in native evaluates over arrays, as ``math`` computes it.

    Every native built-in has a kernel, ``erf`` included, so none raises
    ``UnsupportedNumpyLoweringError``.
    """
    x = mock_identifier("x", 0)
    values = np.array([0.25, 0.5, 0.75])
    references: dict[str, Callable[[float], float]] = {
        "exp": math.exp,
        "exp2": lambda value: 2.0**value,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "arcsin": math.asin,
        "arccos": math.acos,
        "arctan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "erf": math.erf,
        "round": round,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    native_function_names = {
        entry.name
        for entry in get_registered_entries().values()
        if isinstance(entry, NativeFunction)
    }

    assert native_function_names == set(references)
    for name, reference in references.items():
        result = evaluate_expression_with_numpy(call(name, x), {x: values})
        expected = [reference(float(value)) for value in values]
        assert np.allclose(result, expected, rtol=1e-14), name


# =============================================================================
# Environment is snapshotted at construction
# =============================================================================


def test_numpy_expression_evaluator_snapshots_environment_at_construction() -> None:
    """Test the evaluator's resolution is unaffected by mutating the caller's dict.

    Builds the evaluator from a plain, still-mutable ``dict`` binding
    ``x`` to ``1``, then rebinds ``x`` to ``2`` in that same dict after
    construction. Evaluating the bare identifier must still give ``1``,
    guarding against the evaluator aliasing the caller's dict instead of
    snapshotting it.
    """
    x = mock_identifier("x", 0)
    environment = {x: 1}
    evaluator = NumpyExpressionEvaluator(environment)

    environment[x] = 2

    result = evaluator(IdentifierExpression(x))

    assert result == 1
