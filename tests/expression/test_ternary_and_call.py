"""Tests for ``TernaryExpression``, ``CallExpression``, and their helpers."""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    call,
    ternary,
)
from fhy_core.identifier import Identifier
from fhy_core.trait import FrozenMutationError, HasOperands

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
# TernaryExpression: structural equivalence
# =============================================================================


def test_ternary_expression_equivalent_when_all_three_operands_equivalent() -> None:
    """Test two ``TernaryExpression``s with matching operands are equivalent."""
    left = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    right = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


@pytest.mark.parametrize(
    "differ_kind",
    ["condition", "true_value", "false_value"],
)
def test_ternary_expression_not_equivalent_when_one_operand_differs(
    differ_kind: str,
) -> None:
    """Test ``TernaryExpression`` equivalence is false if any operand differs."""
    base = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    overrides = {
        "condition": LiteralExpression(False),
        "true_value": LiteralExpression(99),
        "false_value": LiteralExpression(-1),
    }
    other = TernaryExpression(
        overrides["condition"] if differ_kind == "condition" else base.condition,
        overrides["true_value"] if differ_kind == "true_value" else base.true_value,
        (
            overrides["false_value"]
            if differ_kind == "false_value"
            else base.false_value
        ),
    )

    assert not base.is_structurally_equivalent(other)


def test_ternary_expression_not_equivalent_to_other_expression_kinds() -> None:
    """Test a ``TernaryExpression`` is not equivalent to a binary expression."""
    ternary_expression = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )
    binary_expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    assert not ternary_expression.is_structurally_equivalent(binary_expression)


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
# CallExpression: structural equivalence
# =============================================================================


def test_call_expression_equivalent_for_same_name_and_arguments() -> None:
    """Test ``CallExpression`` equivalence is true for matching name and arguments."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    assert left.is_structurally_equivalent(right)


def test_call_expression_not_equivalent_when_name_differs() -> None:
    """Test ``CallExpression`` equivalence is false when function names differ."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression("min", (LiteralExpression(1), LiteralExpression(2)))

    assert not left.is_structurally_equivalent(right)


def test_call_expression_not_equivalent_when_argument_count_differs() -> None:
    """Test ``CallExpression`` equivalence is false when argument counts differ."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression(
        "max", (LiteralExpression(1), LiteralExpression(2), LiteralExpression(3))
    )

    assert not left.is_structurally_equivalent(right)


def test_call_expression_not_equivalent_when_one_argument_differs() -> None:
    """Test ``CallExpression`` equivalence is false when an argument differs."""
    left = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))
    right = CallExpression("max", (LiteralExpression(1), LiteralExpression(99)))

    assert not left.is_structurally_equivalent(right)


def test_call_expression_not_equivalent_to_other_expression_kinds() -> None:
    """Test a ``CallExpression`` is not equivalent to a binary expression."""
    call_expression = CallExpression(
        "max", (LiteralExpression(1), LiteralExpression(2))
    )
    binary_expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )

    assert not call_expression.is_structurally_equivalent(binary_expression)


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
