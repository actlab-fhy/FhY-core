"""Tests for ``FunctionInliner`` and ``inline_functions``."""

from typing import cast

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    FunctionInliner,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    inline_functions,
    register_function,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

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
    identifier = Identifier("x")
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


def test_inline_functions_preserves_ternary_expression_structurally() -> None:
    """Test a call-free ``TernaryExpression`` is returned unchanged."""
    original = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
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
    parameter = Identifier("x")
    register_function(
        "test_double",
        parameters=[parameter],
        body=IdentifierExpression(parameter) + IdentifierExpression(parameter),
    )

    expression = CallExpression("test_double", (LiteralExpression(3),))
    expected = LiteralExpression(3) + LiteralExpression(3)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


def test_inline_functions_substitutes_parameter_in_ternary_body(
    function_registry_snapshot: None,
) -> None:
    """Test a ``TernaryExpression`` body is inlined with arguments substituted."""
    a = Identifier("a")
    b = Identifier("b")
    register_function(
        "test_ternary_body",
        parameters=[a, b],
        body=TernaryExpression(
            IdentifierExpression(a) > IdentifierExpression(b),
            IdentifierExpression(a),
            IdentifierExpression(b),
        ),
    )

    x = LiteralExpression(7)
    y = LiteralExpression(2)
    expression = CallExpression("test_ternary_body", (x, y))
    expected = TernaryExpression(x > y, x, y)

    result = inline_functions(expression)

    assert result.is_structurally_equivalent(expected)


def test_inline_functions_inlines_call_nested_inside_arithmetic(
    function_registry_snapshot: None,
) -> None:
    """Test a ``CallExpression`` nested inside an arithmetic tree is inlined."""
    parameter = Identifier("x")
    register_function(
        "test_double_nested",
        parameters=[parameter],
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
    parameter = Identifier("x")
    register_function(
        "test_double_recursive",
        parameters=[parameter],
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
    parameter = Identifier("x")
    register_function(
        "test_inner",
        parameters=[parameter],
        body=IdentifierExpression(parameter) + LiteralExpression(1),
    )
    outer_parameter = Identifier("y")
    register_function(
        "test_outer",
        parameters=[outer_parameter],
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
    """Test a call to an unregistered function surfaces ``FunctionLookupError``."""
    expression = CallExpression("not_registered", (LiteralExpression(1),))

    with pytest.raises(PassExecutionError, match="FunctionLookupError"):
        inline_functions(expression)


def test_inline_functions_raises_when_argument_count_exceeds_parameters(
    function_registry_snapshot: None,
) -> None:
    """Test inlining a call with too many arguments surfaces ``FunctionArityError``."""
    parameter = Identifier("x")
    register_function(
        "test_arity_too_many",
        parameters=[parameter],
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
    a = Identifier("a")
    b = Identifier("b")
    register_function(
        "test_arity_too_few",
        parameters=[a, b],
        body=IdentifierExpression(a) + IdentifierExpression(b),
    )

    expression = CallExpression("test_arity_too_few", (LiteralExpression(1),))

    with pytest.raises(PassExecutionError, match="FunctionArityError"):
        inline_functions(expression)


def test_inline_functions_raises_for_recursive_function(
    function_registry_snapshot: None,
) -> None:
    """Test a self-referential function call is detected and rejected."""
    parameter = Identifier("x")
    register_function(
        "test_recursive_self",
        parameters=[parameter],
        body=CallExpression("test_recursive_self", (IdentifierExpression(parameter),)),
    )

    expression = CallExpression("test_recursive_self", (LiteralExpression(0),))

    with pytest.raises(PassExecutionError, match="RecursionError"):
        inline_functions(expression)


def test_inline_functions_raises_for_mutually_recursive_functions(
    function_registry_snapshot: None,
) -> None:
    """Test a mutual-recursion cycle between two functions is detected."""
    parameter_a = Identifier("a")
    register_function(
        "test_mutual_a",
        parameters=[parameter_a],
        body=CallExpression("test_mutual_b", (IdentifierExpression(parameter_a),)),
    )
    parameter_b = Identifier("b")
    register_function(
        "test_mutual_b",
        parameters=[parameter_b],
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
        _contains_call_expression(cast(Expression, child))
        for child in expression.get_visit_children()
    )


def test_inline_functions_result_contains_no_call_expression(
    function_registry_snapshot: None,
) -> None:
    """Test the inliner output is free of ``CallExpression`` nodes."""
    parameter = Identifier("x")
    register_function(
        "test_no_residual_calls",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    expression = LiteralExpression(1) + CallExpression(
        "test_no_residual_calls", (LiteralExpression(2),)
    )

    result = inline_functions(expression)

    assert not _contains_call_expression(result)


# =============================================================================
# FunctionInliner: pass-style invocation
# =============================================================================


def test_function_inliner_callable_returns_inlined_expression(
    function_registry_snapshot: None,
) -> None:
    """Test ``FunctionInliner()`` as a callable yields the inlined expression."""
    parameter = Identifier("x")
    register_function(
        "test_inliner_callable",
        parameters=[parameter],
        body=IdentifierExpression(parameter) + LiteralExpression(1),
    )

    inliner = FunctionInliner()
    expression = CallExpression("test_inliner_callable", (LiteralExpression(5),))

    result = inliner(expression)

    assert result.is_structurally_equivalent(
        LiteralExpression(5) + LiteralExpression(1)
    )


def test_inline_functions_does_not_modify_input_expression(
    function_registry_snapshot: None,
) -> None:
    """Test the input expression tree is not mutated by inlining."""
    parameter = Identifier("x")
    register_function(
        "test_pure_inline",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    expression = CallExpression("test_pure_inline", (LiteralExpression(3),))
    original_function_name = expression.function_name
    original_arguments = expression.arguments

    _ = inline_functions(expression)

    assert expression.function_name == original_function_name
    assert expression.arguments == original_arguments
