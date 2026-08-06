"""Tests for `evaluate_expression`.

The evaluator is a bottom-up rewriter that performs two narrow
substitutions:

1. ``CallExpression`` whose registered target is a ``NativeFunction``
   and whose arguments are all ``LiteralExpression`` becomes
   ``LiteralExpression(implementation(*values))``.
2. ``IdentifierExpression`` whose identifier name matches a registered
   ``NativeConstant`` becomes ``LiteralExpression(constant.value)``.

Every other node kind (arithmetic, comparison, logical, ternary,
non-native calls, calls with at least one non-literal argument,
identifier references that do not match a constant) is reconstructed
with its evaluated children but otherwise left as-is. The evaluator
explicitly does **not** fold literal arithmetic — that remains a
``simplify_expression`` (sympy) job.
"""

import math
from collections.abc import Callable

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeFunction,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    evaluate_expression,
    register_function,
    register_native_constant,
    register_native_function,
)


def register_real_unary_native(
    name: str,
    implementation: Callable[..., bool | int | float],
    result_sort: FunctionSort = FunctionSort.REAL,
) -> NativeFunction:
    """Register a unary native function with a ``REAL`` parameter."""
    return register_native_function(
        name,
        parameter_sorts=[FunctionSort.REAL],
        result_sort=result_sort,
        implementation=implementation,
    )


# =============================================================================
# Native-call folding: happy path
# =============================================================================


def test_evaluate_folds_native_call_with_single_literal_argument(
    function_registry_snapshot: None,
) -> None:
    """Test a native call with a single literal argument folds to the result."""
    register_real_unary_native("test_eval_exp", math.exp)

    result = evaluate_expression(
        CallExpression("test_eval_exp", (LiteralExpression(0.0),))
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == math.exp(0.0)


def test_evaluate_folds_native_call_with_multiple_literal_arguments(
    function_registry_snapshot: None,
) -> None:
    """Test a native call with multiple literal arguments folds to the result."""
    register_native_function(
        "test_eval_atan2",
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.atan2,
    )

    result = evaluate_expression(
        CallExpression(
            "test_eval_atan2",
            (LiteralExpression(1.0), LiteralExpression(1.0)),
        )
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == math.atan2(1.0, 1.0)


def test_evaluate_folds_native_call_returning_int_to_int_literal(
    function_registry_snapshot: None,
) -> None:
    """Test a native call whose impl returns ``int`` produces an ``int`` literal.

    Pins down that the evaluator preserves the Python runtime type of
    the implementation's return value (``math.ceil`` returns ``int``).
    """
    register_real_unary_native(
        "test_eval_ceil", math.ceil, result_sort=FunctionSort.INT
    )

    result = evaluate_expression(
        CallExpression("test_eval_ceil", (LiteralExpression(2.5),))
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == 3
    assert type(result.value) is int


def test_evaluate_extracts_int_value_from_int_literal_argument(
    function_registry_snapshot: None,
) -> None:
    """Test the evaluator passes ``int`` to a native impl on int literals."""
    register_real_unary_native("test_eval_int_arg", math.sqrt)

    result = evaluate_expression(
        CallExpression("test_eval_int_arg", (LiteralExpression(4),))
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == 2.0


# =============================================================================
# Native-call folding: non-literal arguments leave the call as-is
# =============================================================================


def test_evaluate_leaves_native_call_with_identifier_argument_alone(
    function_registry_snapshot: None,
) -> None:
    """Test a native call whose argument is an identifier is not folded."""
    register_real_unary_native("test_eval_keep_symbolic", math.exp)
    x = Identifier("x")
    expression = call("test_eval_keep_symbolic", x)

    result = evaluate_expression(expression)

    assert isinstance(result, CallExpression)
    assert result.function_name == "test_eval_keep_symbolic"
    assert result.arguments[0].is_structurally_equivalent(IdentifierExpression(x))


def _add(a: float, b: float) -> float:
    return a + b


def test_evaluate_leaves_native_call_with_mixed_arguments_alone(
    function_registry_snapshot: None,
) -> None:
    """Test a native call with a mix of literal and non-literal args stays as-is."""
    register_native_function(
        "test_eval_mixed_args",
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=_add,
    )
    x = Identifier("x")
    expression = call("test_eval_mixed_args", LiteralExpression(1.0), x)

    result = evaluate_expression(expression)

    assert isinstance(result, CallExpression)
    assert result.function_name == "test_eval_mixed_args"


# =============================================================================
# Non-native (expression-bodied) calls are NOT folded
# =============================================================================


def test_evaluate_leaves_expression_bodied_call_alone_even_with_literal_args(
    function_registry_snapshot: None,
) -> None:
    """Test a ``RegisteredFunction`` call is not folded by the evaluator.

    Expression-bodied registrations are inlined by ``inline_functions``,
    not the evaluator.
    """
    parameter = Identifier("x")
    register_function(
        "test_eval_expression_bodied",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=parameter + LiteralExpression(1),
    )

    expression = call("test_eval_expression_bodied", LiteralExpression(2))
    result = evaluate_expression(expression)

    assert isinstance(result, CallExpression)
    assert result.function_name == "test_eval_expression_bodied"


# =============================================================================
# Native-constant substitution
# =============================================================================


def test_evaluate_substitutes_identifier_reference_with_native_constant_value(
    function_registry_snapshot: None,
) -> None:
    """Test an identifier whose name matches a registered constant is substituted."""
    register_native_constant("test_eval_const", sort=FunctionSort.REAL, value=2.5)
    expression = IdentifierExpression(Identifier("test_eval_const"))

    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 2.5


def test_evaluate_leaves_identifier_alone_when_name_not_a_constant(
    function_registry_snapshot: None,
) -> None:
    """Test an identifier whose name is not in the registry is unchanged."""
    x = Identifier("test_eval_unbound")
    expression = IdentifierExpression(x)

    result = evaluate_expression(expression)

    assert isinstance(result, IdentifierExpression)
    assert result.identifier is x


# =============================================================================
# Nested native calls fold bottom-up
# =============================================================================


def test_evaluate_folds_nested_native_calls_bottom_up(
    function_registry_snapshot: None,
) -> None:
    """Test a chain of native calls is folded innermost-first."""
    register_real_unary_native("test_eval_nest_exp", math.exp)
    register_real_unary_native("test_eval_nest_sqrt", math.sqrt)

    # sqrt(exp(0.0)) == sqrt(1.0) == 1.0
    expression = CallExpression(
        "test_eval_nest_sqrt",
        (CallExpression("test_eval_nest_exp", (LiteralExpression(0.0),)),),
    )
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 1.0


def test_evaluate_folds_native_call_with_constant_argument(
    function_registry_snapshot: None,
) -> None:
    """Test a native call whose only argument is a registered constant folds.

    The constant is substituted bottom-up, leaving the native call
    with a literal argument; the native call then folds.
    """
    register_native_constant("test_eval_const_arg", sort=FunctionSort.REAL, value=0.0)
    register_real_unary_native("test_eval_const_native", math.exp)

    expression = call("test_eval_const_native", Identifier("test_eval_const_arg"))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 1.0


# =============================================================================
# Literal arithmetic is NOT folded
# =============================================================================


def test_evaluate_does_not_fold_binary_addition_of_literals() -> None:
    """Test ``1.0 + 2.0`` is left intact (no literal-arithmetic folding)."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1.0), LiteralExpression(2.0)
    )

    result = evaluate_expression(expression)

    assert isinstance(result, BinaryExpression)
    assert result.operation == BinaryOperation.ADD


def test_evaluate_does_not_fold_unary_negation_of_literal() -> None:
    """Test ``-(2.5)`` is left intact."""
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(2.5))

    result = evaluate_expression(expression)

    assert isinstance(result, UnaryExpression)
    assert result.operation == UnaryOperation.NEGATE


def test_evaluate_does_not_fold_literal_only_ternary() -> None:
    """Test ``True ? 1 : 2`` is left intact."""
    expression = TernaryExpression(
        LiteralExpression(True),
        LiteralExpression(1),
        LiteralExpression(2),
    )

    result = evaluate_expression(expression)

    assert isinstance(result, TernaryExpression)


# =============================================================================
# Recursive walk through non-native nodes
# =============================================================================


def test_evaluate_recurses_into_binary_expression_children(
    function_registry_snapshot: None,
) -> None:
    """Test the evaluator visits inside binary expressions to fold a native call."""
    register_real_unary_native("test_eval_into_binary", math.exp)

    expression = BinaryExpression(
        BinaryOperation.ADD,
        CallExpression("test_eval_into_binary", (LiteralExpression(0.0),)),
        LiteralExpression(1),
    )
    result = evaluate_expression(expression)

    assert isinstance(result, BinaryExpression)
    assert isinstance(result.left, LiteralExpression)
    assert result.left.value == 1.0
    assert result.right.is_structurally_equivalent(LiteralExpression(1))


def test_evaluate_recurses_into_ternary_branches(
    function_registry_snapshot: None,
) -> None:
    """Test the evaluator folds native calls within the ternary's branches."""
    register_real_unary_native("test_eval_into_ternary", math.exp)
    x = Identifier("x")

    expression = TernaryExpression(
        IdentifierExpression(x) > 0,
        call("test_eval_into_ternary", LiteralExpression(0.0)),
        LiteralExpression(0),
    )
    result = evaluate_expression(expression)

    assert isinstance(result, TernaryExpression)
    assert isinstance(result.true_value, LiteralExpression)
    assert result.true_value.value == 1.0


# =============================================================================
# Native implementations that raise propagate via PassExecutionError
# =============================================================================


def test_evaluate_wraps_native_value_error_in_pass_execution_error(
    function_registry_snapshot: None,
) -> None:
    """Test ``math.sqrt(-1.0)``'s ``ValueError`` surfaces as ``PassExecutionError``."""
    register_real_unary_native("test_eval_sqrt_neg", math.sqrt)

    expression = CallExpression("test_eval_sqrt_neg", (LiteralExpression(-1.0),))

    with pytest.raises(PassExecutionError, match="ValueError"):
        evaluate_expression(expression)


def _reciprocal(value: float) -> float:
    return 1.0 / value


def test_evaluate_wraps_native_zero_division_error_in_pass_execution_error(
    function_registry_snapshot: None,
) -> None:
    """Test a native ``ZeroDivisionError`` surfaces as ``PassExecutionError``."""
    register_real_unary_native("test_eval_div_zero", _reciprocal)

    expression = CallExpression("test_eval_div_zero", (LiteralExpression(0.0),))

    with pytest.raises(PassExecutionError, match="ZeroDivisionError"):
        evaluate_expression(expression)


# =============================================================================
# Identity / preservation when there is nothing to do
# =============================================================================


def test_evaluate_returns_literal_unchanged() -> None:
    """Test a bare ``LiteralExpression`` is returned unchanged."""
    original = LiteralExpression(5)

    result = evaluate_expression(original)

    assert result.is_structurally_equivalent(original)


def test_evaluate_returns_identifier_unchanged_when_not_a_constant() -> None:
    """Test a bare ``IdentifierExpression`` (not a constant) is returned unchanged."""
    x = Identifier("test_eval_passthrough")
    original = IdentifierExpression(x)

    result = evaluate_expression(original)

    assert result.is_structurally_equivalent(original)


def test_evaluate_uses_call_helper_for_construction(
    function_registry_snapshot: None,
) -> None:
    """Test the public ``call`` helper builds a tree the evaluator can fold."""
    register_real_unary_native("test_eval_via_call_helper", math.exp)

    expression = call("test_eval_via_call_helper", LiteralExpression(0.0))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 1.0


# =============================================================================
# Source-input integrity
# =============================================================================


def test_evaluate_does_not_mutate_input_expression(
    function_registry_snapshot: None,
) -> None:
    """Test the input expression tree is not mutated by evaluation."""
    register_real_unary_native("test_eval_purity", math.sqrt)

    expression = CallExpression("test_eval_purity", (LiteralExpression(4),))
    original_name = expression.function_name
    original_arguments = expression.arguments

    _ = evaluate_expression(expression)

    assert expression.function_name == original_name
    assert expression.arguments == original_arguments


# =============================================================================
# String-form numeric literal coercion
# =============================================================================


def test_evaluate_rejects_string_form_float_literal_argument(
    function_registry_snapshot: None,
) -> None:
    """Test a native call with a float-grammar string argument is refused.

    A float-grammar string ``LiteralExpression`` preserves an exact
    decimal value; collapsing it to a binary float for a native call
    would lose that precision, so evaluation raises instead.
    """
    register_real_unary_native("test_eval_str_coerce", math.sqrt)

    expression = CallExpression("test_eval_str_coerce", (LiteralExpression("4.0"),))

    with pytest.raises(PassExecutionError, match="StringLiteralPrecisionError"):
        evaluate_expression(expression)


def test_evaluate_coerces_string_form_integer_literal_to_int(
    function_registry_snapshot: None,
) -> None:
    """Test ``LiteralExpression("4")`` is coerced to an int for native calls."""
    register_real_unary_native("test_eval_str_int_coerce", math.sqrt)

    expression = CallExpression("test_eval_str_int_coerce", (LiteralExpression("9"),))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 3.0


# =============================================================================
# Distinguished error cases for call targets
# =============================================================================


def test_evaluate_raises_for_unregistered_call_name() -> None:
    """Test a call to an unregistered name surfaces ``EntryLookupError``."""
    expression = CallExpression("test_eval_never_registered", (LiteralExpression(0),))

    with pytest.raises(PassExecutionError, match="EntryLookupError"):
        evaluate_expression(expression)


def test_evaluate_raises_for_call_to_native_constant(
    function_registry_snapshot: None,
) -> None:
    """Test a call to a registered native constant raises ``FunctionArityError``."""
    register_native_constant("test_eval_constant_called", FunctionSort.REAL, 2.5)

    expression = CallExpression("test_eval_constant_called", ())

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        evaluate_expression(expression)


def test_evaluate_preserves_call_to_registered_function_unchanged(
    function_registry_snapshot: None,
) -> None:
    """Test a call to an expression-bodied function is preserved (not inlined).

    The evaluator emits a diagnostic recommending ``inline_functions``
    and leaves the call node unchanged.
    """
    parameter = Identifier("x")
    register_function(
        "test_eval_unfolded_call",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression("test_eval_unfolded_call", (LiteralExpression(3),))
    result = evaluate_expression(expression)

    assert isinstance(result, CallExpression)
    assert result.function_name == "test_eval_unfolded_call"


def test_evaluate_raises_native_result_sort_error_for_wrong_return_type(
    function_registry_snapshot: None,
) -> None:
    """Test a native returning a wrong-sort value raises ``NativeResultSortError``."""
    register_native_function(
        "test_eval_wrong_return_sort",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.INT,
        implementation=lambda _value: 1.5,
    )

    expression = CallExpression(
        "test_eval_wrong_return_sort", (LiteralExpression(0.0),)
    )

    with pytest.raises(PassExecutionError, match="NativeResultSortError"):
        evaluate_expression(expression)
