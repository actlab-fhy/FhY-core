"""Tests for ``TernaryExpression``, ``CallExpression``, and their helpers."""

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    call,
    ternary,
)
from fhy_core.traits import FrozenMutationError, HasOperands

# =============================================================================
# TernaryExpression: construction and accessors
# =============================================================================


def test_ternary_expression_stores_three_operands_in_declared_order() -> None:
    """Test ``TernaryExpression`` exposes the three fields it was built with."""
    condition = LiteralExpression(True)
    true_value = LiteralExpression(1)
    false_value = LiteralExpression(2)

    expression = TernaryExpression(condition, true_value, false_value)

    assert expression.condition is condition
    assert expression.true_value is true_value
    assert expression.false_value is false_value


def test_ternary_expression_is_frozen_after_construction() -> None:
    """Test ``TernaryExpression`` instances reject attribute mutation."""
    expression = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )

    with pytest.raises(FrozenMutationError):
        expression._mutation = "denied"


def test_ternary_expression_satisfies_has_operands_protocol() -> None:
    """Test ``TernaryExpression`` satisfies the ``HasOperands`` runtime protocol."""
    expression = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    assert isinstance(expression, HasOperands)


def test_ternary_expression_get_operands_returns_condition_then_branches() -> None:
    """Test ``TernaryExpression.get_operands()`` returns ``(cond, true, false)``."""
    condition = LiteralExpression(True)
    true_value = LiteralExpression(1)
    false_value = LiteralExpression(2)
    expression = TernaryExpression(condition, true_value, false_value)

    assert expression.get_operands() == (condition, true_value, false_value)


def test_ternary_expression_get_visit_children_returns_three_operands() -> None:
    """Test ``TernaryExpression.get_visit_children()`` exposes the three operands."""
    condition = LiteralExpression(True)
    true_value = LiteralExpression(1)
    false_value = LiteralExpression(2)
    expression = TernaryExpression(condition, true_value, false_value)

    assert expression.get_visit_children() == (condition, true_value, false_value)


# =============================================================================
# ternary() constructor helper
# =============================================================================


def test_ternary_helper_wraps_three_expressions_directly() -> None:
    """Test ``ternary(...)`` returns expression operands unchanged."""
    condition = LiteralExpression(True)
    true_value = LiteralExpression(1)
    false_value = LiteralExpression(2)

    expression = ternary(condition, true_value, false_value)

    assert isinstance(expression, TernaryExpression)
    assert expression.condition is condition
    assert expression.true_value is true_value
    assert expression.false_value is false_value


def test_ternary_helper_coerces_identifier_operand_to_identifier_expression() -> None:
    """Test ``ternary(...)`` wraps an ``Identifier`` in ``IdentifierExpression``."""
    identifier = Identifier("x")

    expression = ternary(LiteralExpression(True), identifier, LiteralExpression(0))

    assert isinstance(expression.true_value, IdentifierExpression)
    assert expression.true_value.identifier is identifier


def test_ternary_helper_coerces_python_literal_operand_to_literal_expression() -> None:
    """Test ``ternary(...)`` wraps a Python literal in ``LiteralExpression``."""
    expression = ternary(LiteralExpression(True), 5, 10)

    assert isinstance(expression.true_value, LiteralExpression)
    assert expression.true_value.value == 5
    assert isinstance(expression.false_value, LiteralExpression)
    assert expression.false_value.value == 10


def test_ternary_helper_rejects_unsupported_operand_type() -> None:
    """Test ``ternary(...)`` raises ``ValueError`` for unsupported operand types."""
    with pytest.raises(ValueError):
        ternary(LiteralExpression(True), object(), LiteralExpression(0))  # type: ignore[arg-type]


# =============================================================================
# CallExpression: construction and accessors
# =============================================================================


def test_call_expression_stores_name_and_arguments() -> None:
    """Test ``CallExpression`` exposes the name and argument tuple it was built with."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)

    expression = CallExpression("max", (arg_a, arg_b))

    assert expression.function_name == "max"
    assert expression.arguments == (arg_a, arg_b)


def test_call_expression_supports_zero_arguments() -> None:
    """Test ``CallExpression`` accepts an empty arguments tuple structurally."""
    expression = CallExpression("nullary", ())

    assert expression.function_name == "nullary"
    assert expression.arguments == ()


def test_call_expression_rejects_empty_function_name() -> None:
    """Test ``CallExpression`` rejects an empty ``function_name`` at construction.

    This is a value constraint enforced by ``__post_init__``, independent of
    serialization; deserializing such a payload surfaces it through the generic
    engine as a value error.
    """
    with pytest.raises(ValueError, match="non-empty"):
        CallExpression("", (LiteralExpression(1),))


def test_call_expression_is_frozen_after_construction() -> None:
    """Test ``CallExpression`` instances reject attribute mutation."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    with pytest.raises(FrozenMutationError):
        expression._mutation = "denied"


def test_call_expression_satisfies_has_operands_protocol() -> None:
    """Test ``CallExpression`` satisfies the ``HasOperands`` runtime protocol."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    assert isinstance(expression, HasOperands)


def test_call_expression_get_operands_returns_arguments_in_order() -> None:
    """Test ``CallExpression.get_operands()`` returns the argument tuple in order."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)
    arg_c = LiteralExpression(3)
    expression = CallExpression("ternary", (arg_a, arg_b, arg_c))

    assert expression.get_operands() == (arg_a, arg_b, arg_c)


def test_call_expression_get_visit_children_returns_arguments_in_order() -> None:
    """Test ``CallExpression.get_visit_children()`` exposes its arguments."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)
    expression = CallExpression("max", (arg_a, arg_b))

    assert expression.get_visit_children() == (arg_a, arg_b)


# =============================================================================
# call() constructor helper
# =============================================================================


def test_call_helper_returns_call_expression_with_name_and_arguments() -> None:
    """Test ``call(...)`` produces a ``CallExpression`` carrying its name and args."""
    arg_a = LiteralExpression(1)
    arg_b = LiteralExpression(2)

    expression = call("max", arg_a, arg_b)

    assert isinstance(expression, CallExpression)
    assert expression.function_name == "max"
    assert expression.arguments == (arg_a, arg_b)


def test_call_helper_supports_zero_arguments() -> None:
    """Test ``call(name)`` produces a ``CallExpression`` with empty arguments."""
    expression = call("nullary")

    assert isinstance(expression, CallExpression)
    assert expression.arguments == ()


def test_call_helper_coerces_identifier_argument_to_identifier_expression() -> None:
    """Test ``call(...)`` wraps an ``Identifier`` argument in identifier expr."""
    identifier = Identifier("x")

    expression = call("f", identifier)

    arguments = expression.arguments
    assert len(arguments) == 1
    assert isinstance(arguments[0], IdentifierExpression)
    assert arguments[0].identifier is identifier


def test_call_helper_coerces_python_literal_to_literal_expression() -> None:
    """Test ``call(...)`` wraps a Python literal argument in ``LiteralExpression``."""
    expression = call("max", 1, 2)

    arguments = expression.arguments
    assert len(arguments) == 2
    assert isinstance(arguments[0], LiteralExpression)
    assert arguments[0].value == 1
    assert isinstance(arguments[1], LiteralExpression)
    assert arguments[1].value == 2


def test_call_helper_rejects_unsupported_argument_type() -> None:
    """Test ``call(...)`` raises ``ValueError`` for unsupported argument types."""
    with pytest.raises(ValueError):
        call("f", object())  # type: ignore[arg-type]
