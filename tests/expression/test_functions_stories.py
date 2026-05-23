"""End-to-end user-story and integration tests for the expression-functions feature.

These exercise the public surface (parser, registry, inliner,
type-checker, pretty-printer) in shapes that mirror real callers. They
are coarser-grained than the per-module unit tests in this directory.
"""

from unittest.mock import patch

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    inline_functions,
    parse_expression,
    pformat_expression,
    register_function,
    ternary,
)
from fhy_core.expression.passes.type_checker import synthesize_expression_type
from fhy_core.identifier import Identifier
from fhy_core.types import (
    CoreDataType,
    NumericalType,
    PrimitiveDataType,
    Type,
    TypeQualifier,
)


def _scalar(core_data_type: CoreDataType) -> NumericalType:
    return NumericalType(PrimitiveDataType(core_data_type))


def _no_identifiers(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise AssertionError(f"unexpected lookup: {identifier}")


# =============================================================================
# Integration: parse -> inline -> type-check
# =============================================================================


pytestmark = pytest.mark.integration


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_max_call_then_inline_yields_expected_ternary_tree() -> None:
    """Test ``max(1, 2)`` parses, inlines, and equals a literal ternary tree."""
    parsed = parse_expression("max(1, 2)")

    inlined = inline_functions(parsed)

    expected = TernaryExpression(
        BinaryExpression(
            BinaryOperation.GREATER,
            LiteralExpression(1),
            LiteralExpression(2),
        ),
        LiteralExpression(1),
        LiteralExpression(2),
    )

    assert inlined.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_ternary_then_synthesize_type_returns_branch_type() -> None:
    """Test ``True ? 1 : 2`` parses and synthesizes to the branches' type.

    Two weak unsigned-int literal branches stay weak ``UINT`` without an
    outer context.
    """
    parsed = parse_expression("True ? 1 : 2")

    result_type, _ = synthesize_expression_type(parsed, _no_identifiers)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.UINT))


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_inline_synthesize_for_max_min_clamp() -> None:
    """Test a clamp expression parses, inlines, and synthesizes a numeric type.

    ``max(low, min(value, high))`` is the canonical 3-argument clamp. With
    literal integer bounds and no outer context, the result type is the
    weak ``UINT``.
    """
    parsed = parse_expression("max(0, min(5, 10))")

    inlined = inline_functions(parsed)
    result_type, _ = synthesize_expression_type(inlined, _no_identifiers)

    assert result_type.is_structurally_equivalent(_scalar(CoreDataType.UINT))


# =============================================================================
# Integration: pretty-printer round-trip
# =============================================================================


def test_pretty_print_and_parse_round_trip_for_ternary_and_call() -> None:
    """Test ``parse(pformat(ternary(call(...))))`` round-trips structurally."""
    expression = TernaryExpression(
        LiteralExpression(True),
        CallExpression("max", (LiteralExpression(1), LiteralExpression(2))),
        LiteralExpression(0),
    )

    formatted = pformat_expression(expression)
    reparsed = parse_expression(formatted)

    assert reparsed.is_structurally_equivalent(expression)


# =============================================================================
# Integration: serialization round-trip for mixed trees
# =============================================================================


def test_mixed_ternary_and_call_tree_round_trips_through_serialize_to_dict() -> None:
    """Test a mixed ``TernaryExpression`` / ``CallExpression`` tree round-trips."""
    expression = TernaryExpression(
        LiteralExpression(True),
        CallExpression(
            "max",
            (
                LiteralExpression(1),
                BinaryExpression(
                    BinaryOperation.ADD,
                    LiteralExpression(2),
                    LiteralExpression(3),
                ),
            ),
        ),
        UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(7)),
    )

    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())

    assert restored.is_structurally_equivalent(expression)


# =============================================================================
# User story: clamp a parameter to a range
# =============================================================================


def test_user_story_clamp_a_value_between_low_and_high() -> None:
    """Test the standard ``max(low, min(value, high))`` clamp construction.

    Caller: a scheduling pass that wants to clamp a tunable parameter into
    a valid bound. They build the call via the public ``call`` helper and
    rely on inlining to substitute the body.
    """
    low = Identifier("low")
    high = Identifier("high")
    value = Identifier("value")

    expression = call(
        "max",
        IdentifierExpression(low),
        call("min", IdentifierExpression(value), IdentifierExpression(high)),
    )

    inlined = inline_functions(expression)

    expected = TernaryExpression(
        BinaryExpression(
            BinaryOperation.GREATER,
            IdentifierExpression(low),
            TernaryExpression(
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(value),
                    IdentifierExpression(high),
                ),
                IdentifierExpression(value),
                IdentifierExpression(high),
            ),
        ),
        IdentifierExpression(low),
        TernaryExpression(
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(value),
                IdentifierExpression(high),
            ),
            IdentifierExpression(value),
            IdentifierExpression(high),
        ),
    )

    assert inlined.is_structurally_equivalent(expected)


# =============================================================================
# User story: predicate-guarded fall-back
# =============================================================================


def test_user_story_predicate_guarded_fallback_with_ternary() -> None:
    """Test using ``ternary`` to choose between a fast-path and a fallback expression.

    Caller: a lowering pass that wants to express
    ``x > 0 ? sqrt_path : fallback_path`` at the IR level. They build the
    expression directly via the public ``ternary`` helper.
    """
    x = Identifier("x")
    sqrt_path = Identifier("sqrt_path")
    fallback_path = Identifier("fallback_path")

    expression = ternary(
        IdentifierExpression(x) > 0,
        IdentifierExpression(sqrt_path),
        IdentifierExpression(fallback_path),
    )

    assert isinstance(expression, TernaryExpression)
    assert expression.condition.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(0)
        )
    )
    assert expression.true_value.is_structurally_equivalent(
        IdentifierExpression(sqrt_path)
    )
    assert expression.false_value.is_structurally_equivalent(
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
    parameter = Identifier("x")
    register_function(
        "test_user_story_abs",
        parameters=[parameter],
        body=ternary(
            IdentifierExpression(parameter) < 0,
            -IdentifierExpression(parameter),
            IdentifierExpression(parameter),
        ),
    )

    y = Identifier("y")
    expression = call("test_user_story_abs", IdentifierExpression(y))
    inlined = inline_functions(expression)
    expected = TernaryExpression(
        BinaryExpression(
            BinaryOperation.LESS,
            IdentifierExpression(y),
            LiteralExpression(0),
        ),
        UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(y)),
        IdentifierExpression(y),
    )

    assert inlined.is_structurally_equivalent(expected)
