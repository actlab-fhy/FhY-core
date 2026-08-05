"""Tests for ``inline_functions``."""

import pytest

from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    inline_functions,
    register_function,
    register_native_constant,
    register_native_function,
)

from ..conftest import mock_identifier

# =============================================================================
# inline_functions: leaves a tree without CallExpressions untouched
# =============================================================================


def test_inline_functions_returns_literal_unchanged() -> None:
    """Test ``inline_functions`` leaves a literal expression unchanged."""
    original = LiteralExpression(5)

    result = inline_functions(original)

    assert result.is_structurally_equivalent(original)


def test_inline_functions_returns_identifier_unchanged() -> None:
    """Test ``inline_functions`` leaves an identifier expression unchanged."""
    identifier = mock_identifier("x", 0)
    original = IdentifierExpression(identifier)

    result = inline_functions(original)

    assert result.is_structurally_equivalent(original)


def test_inline_functions_traverses_binary_expression_without_calls() -> None:
    """Test ``inline_functions`` preserves a tree with no ``CallExpression``."""
    original = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    result = inline_functions(original)

    assert result.is_structurally_equivalent(original)


def test_inline_functions_preserves_piecewise_expression_structurally() -> None:
    """Test a call-free ``PiecewiseExpression`` is returned unchanged."""
    original = PiecewiseExpression(
        (LiteralExpression(True),), (LiteralExpression(1),), LiteralExpression(2)
    )

    result = inline_functions(original)

    assert result.is_structurally_equivalent(original)


# =============================================================================
# inline_functions: substitutes a single CallExpression with the body
# =============================================================================


def test_inline_functions_substitutes_call_with_registered_body(
    function_registry_snapshot: None,
) -> None:
    """Test a single ``CallExpression`` is replaced by its inlined body."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_double",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter) + IdentifierExpression(parameter),
    )

    expression = CallExpression("test_double", (LiteralExpression(3),))
    expected = LiteralExpression(3) + LiteralExpression(3)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


def test_inline_functions_substitutes_parameter_in_piecewise_body(
    function_registry_snapshot: None,
) -> None:
    """Test a ``PiecewiseExpression`` body is inlined with arguments substituted."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    register_function(
        "test_piecewise_body",
        parameters=[a, b],
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=PiecewiseExpression(
            (IdentifierExpression(a) > IdentifierExpression(b),),
            (IdentifierExpression(a),),
            IdentifierExpression(b),
        ),
    )

    x = LiteralExpression(7)
    y = LiteralExpression(2)
    expression = CallExpression("test_piecewise_body", (x, y))
    expected = PiecewiseExpression((x > y,), (x,), y)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


def test_inline_functions_inlines_call_nested_inside_arithmetic(
    function_registry_snapshot: None,
) -> None:
    """Test a ``CallExpression`` nested inside an arithmetic tree is inlined."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_double_nested",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter) + IdentifierExpression(parameter),
    )

    expression = LiteralExpression(1) + CallExpression(
        "test_double_nested", (LiteralExpression(5),)
    )
    expected = LiteralExpression(1) + (LiteralExpression(5) + LiteralExpression(5))

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


# =============================================================================
# inline_functions: bottom-up inlining of nested calls
# =============================================================================


def test_inline_functions_recursively_inlines_nested_calls(
    function_registry_snapshot: None,
) -> None:
    """Test nested ``CallExpression``s are inlined bottom-up."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_double_recursive",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter) + IdentifierExpression(parameter),
    )

    inner_call = CallExpression("test_double_recursive", (LiteralExpression(4),))
    outer_call = CallExpression("test_double_recursive", (inner_call,))
    inlined_inner = LiteralExpression(4) + LiteralExpression(4)
    expected = inlined_inner + inlined_inner

    result = inline_functions(outer_call)

    assert result.is_structurally_equivalent(expected)


def test_inline_functions_inlines_call_that_references_another_registered_function(
    function_registry_snapshot: None,
) -> None:
    """Test a body that calls another registered function is also inlined."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_inner",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter) + LiteralExpression(1),
    )
    outer_parameter = mock_identifier("y", 1)
    register_function(
        "test_outer",
        parameters=[outer_parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=CallExpression("test_inner", (IdentifierExpression(outer_parameter),))
        + LiteralExpression(2),
    )

    expression = CallExpression("test_outer", (LiteralExpression(10),))
    expected = (LiteralExpression(10) + LiteralExpression(1)) + LiteralExpression(2)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


# =============================================================================
# inline_functions: error paths
# =============================================================================


def test_inline_functions_raises_for_unknown_function_name() -> None:
    """Test a call to an unregistered function surfaces ``EntryLookupError``."""
    expression = CallExpression("not_registered", (LiteralExpression(1),))

    with pytest.raises(PassExecutionError, match="EntryLookupError"):
        inline_functions(expression)


def test_inline_functions_raises_when_argument_count_exceeds_parameters(
    function_registry_snapshot: None,
) -> None:
    """Test inlining a call with too many arguments surfaces ``FunctionArityError``."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_arity_too_many",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression(
        "test_arity_too_many", (LiteralExpression(1), LiteralExpression(2))
    )

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        inline_functions(expression)


def test_inline_functions_raises_when_argument_count_is_too_few(
    function_registry_snapshot: None,
) -> None:
    """Test inlining a call with too few arguments surfaces ``FunctionArityError``."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    register_function(
        "test_arity_too_few",
        parameters=[a, b],
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(a) + IdentifierExpression(b),
    )

    expression = CallExpression("test_arity_too_few", (LiteralExpression(1),))

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        inline_functions(expression)


def test_inline_functions_raises_for_recursive_function(
    function_registry_snapshot: None,
) -> None:
    """Test a self-referential function call is detected and rejected."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_recursive_self",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=CallExpression("test_recursive_self", (IdentifierExpression(parameter),)),
    )

    expression = CallExpression("test_recursive_self", (LiteralExpression(0),))

    with pytest.raises(PassExecutionError, match="RecursionError"):
        inline_functions(expression)


def test_inline_functions_raises_for_mutually_recursive_functions(
    function_registry_snapshot: None,
) -> None:
    """Test a mutual-recursion cycle between two functions is detected."""
    parameter_a = mock_identifier("a", 0)
    register_function(
        "test_mutual_a",
        parameters=[parameter_a],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=CallExpression("test_mutual_b", (IdentifierExpression(parameter_a),)),
    )
    parameter_b = mock_identifier("b", 1)
    register_function(
        "test_mutual_b",
        parameters=[parameter_b],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=CallExpression("test_mutual_a", (IdentifierExpression(parameter_b),)),
    )

    expression = CallExpression("test_mutual_a", (LiteralExpression(0),))

    with pytest.raises(PassExecutionError, match="RecursionError"):
        inline_functions(expression)


# =============================================================================
# inline_functions: result contains no CallExpression
# =============================================================================


def _contains_call_expression(expression: Expression) -> bool:
    if isinstance(expression, CallExpression):
        return True
    return any(
        _contains_call_expression(child) for child in expression.get_visit_children()
    )


def test_inline_functions_result_contains_no_call_expression(
    function_registry_snapshot: None,
) -> None:
    """Test the inliner output is free of ``CallExpression`` nodes."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_no_residual_calls",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = LiteralExpression(1) + CallExpression(
        "test_no_residual_calls", (LiteralExpression(2),)
    )

    result = inline_functions(expression)

    assert not _contains_call_expression(result)


def test_inline_functions_does_not_modify_input_expression(
    function_registry_snapshot: None,
) -> None:
    """Test the input expression tree is not mutated by inlining."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_pure_inline",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression("test_pure_inline", (LiteralExpression(3),))
    original_function_name = expression.function_name
    original_arguments = expression.arguments

    _ = inline_functions(expression)

    assert expression.function_name == original_function_name
    assert expression.arguments == original_arguments


# =============================================================================
# inline_functions: NativeConstant and NativeFunction call targets
# =============================================================================


def test_inline_functions_rejects_call_to_native_constant(
    function_registry_snapshot: None,
) -> None:
    """Test calling a registered native constant raises ``FunctionArityError``."""
    register_native_constant("test_inline_const", FunctionSort.REAL, 3.14)

    expression = CallExpression("test_inline_const", ())

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        inline_functions(expression)


def test_inline_functions_passes_through_native_function_call_unchanged(
    function_registry_snapshot: None,
) -> None:
    """Test a call to a native function is preserved untouched.

    The inliner does not fold native calls; that is the evaluator's
    job. The call passes through with arguments rewritten (children
    are still recursively inlined).
    """
    register_native_function(
        "test_inline_native_passthrough",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=lambda value: value * 2,
    )

    expression = CallExpression(
        "test_inline_native_passthrough",
        (IdentifierExpression(mock_identifier("x", 0)),),
    )

    result = inline_functions(expression)

    assert isinstance(result, CallExpression)
    assert result.function_name == "test_inline_native_passthrough"
    assert len(result.arguments) == 1


def test_inline_functions_rejects_wrong_arity_to_native_function(
    function_registry_snapshot: None,
) -> None:
    """Test a native call with the wrong arity raises ``FunctionArityError``."""
    register_native_function(
        "test_inline_native_arity",
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=lambda a, b: a + b,
    )

    expression = CallExpression("test_inline_native_arity", (LiteralExpression(1),))

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        inline_functions(expression)
