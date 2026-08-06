"""End-to-end user-story and integration tests for expression functions.

These exercise the public surface (registry, inliner, type-checker,
pretty-printer) in shapes that mirror real callers. They are
coarser-grained than the per-module unit tests in this directory.
"""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    inline_functions,
    piecewise,
    register_function,
)
from fhy_core.symbolic.expression.passes.type_checker import synthesize_expression_type
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)

from .conftest import mock_identifier


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _no_identifiers(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise KeyError(identifier.name_hint)


# =============================================================================
# Integration: build -> inline -> type-check
# =============================================================================


pytestmark = pytest.mark.integration


def test_max_call_then_inline_yields_expected_piecewise_tree() -> None:
    """Test ``max(1, 2)`` inlines to a literal piecewise tree."""
    expression = call("max", LiteralExpression(1), LiteralExpression(2))

    inlined = inline_functions(expression)

    expected = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                LiteralExpression(1),
                LiteralExpression(2),
            ),
        ),
        (LiteralExpression(1),),
        LiteralExpression(2),
    )

    assert inlined.is_structurally_equivalent(expected)


def test_piecewise_synthesize_type_returns_value_and_otherwise_common_type() -> None:
    """Test ``{1 if True; 2 otherwise}`` synthesizes to the common branch type.

    Two weak unsigned-int literal operands stay weak ``UINT`` without an
    outer context.
    """
    expression = piecewise((LiteralExpression(True), LiteralExpression(1)), otherwise=2)

    result_type, _ = synthesize_expression_type(expression, _no_identifiers)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.UINT))


def test_inline_synthesize_for_max_min_clamp() -> None:
    """Test a clamp expression inlines and synthesizes a numeric type.

    ``max(low, min(value, high))`` is the canonical 3-argument clamp. With
    literal integer bounds and no outer context, the result type is the
    weak ``UINT``.
    """
    expression = call(
        "max",
        LiteralExpression(0),
        call("min", LiteralExpression(5), LiteralExpression(10)),
    )

    inlined = inline_functions(expression)
    result_type, _ = synthesize_expression_type(inlined, _no_identifiers)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.UINT))


# =============================================================================
# User story: clamp a parameter to a range
# =============================================================================


def test_user_story_clamp_a_value_between_low_and_high() -> None:
    """Test the standard ``max(low, min(value, high))`` clamp construction.

    Caller: a scheduling pass that wants to clamp a tunable parameter into
    a valid bound. They build the call via the public ``call`` helper and
    rely on inlining to substitute the body.
    """
    low = mock_identifier("low", 0)
    high = mock_identifier("high", 1)
    value = mock_identifier("value", 2)

    expression = call(
        "max",
        IdentifierExpression(low),
        call("min", IdentifierExpression(value), IdentifierExpression(high)),
    )

    inlined = inline_functions(expression)

    inner_min = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(value),
                IdentifierExpression(high),
            ),
        ),
        (IdentifierExpression(value),),
        IdentifierExpression(high),
    )
    expected = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(low),
                inner_min,
            ),
        ),
        (IdentifierExpression(low),),
        inner_min,
    )

    assert inlined.is_structurally_equivalent(expected)


# =============================================================================
# User story: predicate-guarded fall-back
# =============================================================================


def test_user_story_predicate_guarded_fallback_with_piecewise() -> None:
    """Test using ``piecewise`` to choose between a fast-path and a fallback expression.

    Caller: a lowering pass that wants to express
    ``{sqrt_path if x > 0; fallback_path otherwise}`` at the IR level.
    They build the expression directly via the public ``piecewise``
    helper.
    """
    x = mock_identifier("x", 0)
    sqrt_path = mock_identifier("sqrt_path", 1)
    fallback_path = mock_identifier("fallback_path", 2)

    expression = piecewise(
        (IdentifierExpression(x) > 0, IdentifierExpression(sqrt_path)),
        otherwise=IdentifierExpression(fallback_path),
    )

    assert isinstance(expression, PiecewiseExpression)
    assert expression.conditions[0].is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
        )
    )
    assert expression.values[0].is_structurally_equivalent(
        IdentifierExpression(sqrt_path)
    )
    assert expression.otherwise.is_structurally_equivalent(
        IdentifierExpression(fallback_path)
    )


# =============================================================================
# User story: define a custom function and inline it
# =============================================================================


def test_user_story_register_and_inline_custom_abs_function(
    function_registry_snapshot: None,
) -> None:
    """Test registering a custom ``abs_value`` function and inlining a call to it.

    Caller: a user defines absolute-value as a pure function over the IR
    and then references it. Inlining substitutes the body. The inlined
    expression should match an equivalent hand-built tree.
    """
    parameter = mock_identifier("x", 0)
    register_function(
        "test_user_story_abs",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=piecewise(
            (IdentifierExpression(parameter) < 0, -IdentifierExpression(parameter)),
            otherwise=IdentifierExpression(parameter),
        ),
    )

    y = mock_identifier("y", 1)
    expression = call("test_user_story_abs", IdentifierExpression(y))
    inlined = inline_functions(expression)
    expected = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(y),
                LiteralExpression(0),
            ),
        ),
        (UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(y)),),
        IdentifierExpression(y),
    )

    assert inlined.is_structurally_equivalent(expected)
