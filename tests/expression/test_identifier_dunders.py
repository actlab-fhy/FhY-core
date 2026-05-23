"""Tests for the expression-building dunder methods on ``Identifier``.

The dunders let callers write ``a + b`` (where ``a`` and ``b`` are
``Identifier``\\s) and get a ``BinaryExpression`` back, instead of having
to wrap each operand in ``IdentifierExpression`` first. Behavior matches
the corresponding ``Expression`` dunders; the tests here verify that
each new dunder produces the right node, with the right operation, and
in the right operand order.
"""

from typing import Callable

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier

_BinaryDunder = Callable[[Identifier, Identifier], BinaryExpression]

# =============================================================================
# Unary dunders
# =============================================================================


def test_identifier_neg_produces_unary_negate_over_identifier_expression() -> None:
    """Test ``-a`` returns ``UnaryExpression(NEGATE, IdentifierExpression(a))``."""
    a = Identifier("a")

    expression = -a

    expected = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(a))
    assert isinstance(expression, UnaryExpression)
    assert expression.is_structurally_equivalent(expected)


def test_identifier_pos_produces_unary_positive_over_identifier_expression() -> None:
    """Test ``+a`` returns ``UnaryExpression(POSITIVE, IdentifierExpression(a))``."""
    a = Identifier("a")

    expression = +a

    expected = UnaryExpression(UnaryOperation.POSITIVE, IdentifierExpression(a))
    assert isinstance(expression, UnaryExpression)
    assert expression.is_structurally_equivalent(expected)


# =============================================================================
# Arithmetic dunders: identifier OP identifier
# =============================================================================


_ARITHMETIC_DUNDER_CASES = [
    pytest.param(lambda a, b: a + b, BinaryOperation.ADD, id="add"),
    pytest.param(lambda a, b: a - b, BinaryOperation.SUBTRACT, id="subtract"),
    pytest.param(lambda a, b: a * b, BinaryOperation.MULTIPLY, id="multiply"),
    pytest.param(lambda a, b: a / b, BinaryOperation.DIVIDE, id="divide"),
    pytest.param(lambda a, b: a // b, BinaryOperation.FLOOR_DIVIDE, id="floor_divide"),
    pytest.param(lambda a, b: a % b, BinaryOperation.MODULO, id="modulo"),
    pytest.param(lambda a, b: a**b, BinaryOperation.POWER, id="power"),
]


@pytest.mark.parametrize(("dunder", "expected_operation"), _ARITHMETIC_DUNDER_CASES)
def test_identifier_arithmetic_dunder_over_two_identifiers_wraps_each_side(
    dunder: _BinaryDunder, expected_operation: BinaryOperation
) -> None:
    """Test each arithmetic dunder builds a binary expression wrapping both sides."""
    a = Identifier("a")
    b = Identifier("b")

    expression = dunder(a, b)

    expected = BinaryExpression(
        expected_operation, IdentifierExpression(a), IdentifierExpression(b)
    )
    assert isinstance(expression, BinaryExpression)
    assert expression.is_structurally_equivalent(expected)


# =============================================================================
# Arithmetic dunders: identifier OP literal (forward + reflected forms)
# =============================================================================


def test_identifier_add_literal_coerces_literal_on_right() -> None:
    """Test ``a + 5`` wraps both operands and places the literal on the right."""
    a = Identifier("a")

    expression = a + 5

    expected = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(a), LiteralExpression(5)
    )
    assert expression.is_structurally_equivalent(expected)


def test_identifier_radd_literal_coerces_literal_on_left() -> None:
    """Test ``5 + a`` wraps both operands and places the literal on the left."""
    a = Identifier("a")

    expression = 5 + a

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(5), IdentifierExpression(a)
    )
    assert expression.is_structurally_equivalent(expected)


def test_identifier_rsub_literal_preserves_operand_order() -> None:
    """Test ``5 - a`` builds a SUBTRACT with the literal on the left."""
    a = Identifier("a")

    expression = 5 - a

    expected = BinaryExpression(
        BinaryOperation.SUBTRACT, LiteralExpression(5), IdentifierExpression(a)
    )
    assert expression.is_structurally_equivalent(expected)


def test_identifier_pow_literal_preserves_operand_order() -> None:
    """Test ``a ** 2`` builds a POWER with the identifier on the left."""
    a = Identifier("a")

    expression = a**2

    expected = BinaryExpression(
        BinaryOperation.POWER, IdentifierExpression(a), LiteralExpression(2)
    )
    assert expression.is_structurally_equivalent(expected)


# =============================================================================
# Comparison dunders
# =============================================================================


_COMPARISON_DUNDER_CASES = [
    pytest.param(lambda a, b: a < b, BinaryOperation.LESS, id="less"),
    pytest.param(lambda a, b: a <= b, BinaryOperation.LESS_EQUAL, id="less_equal"),
    pytest.param(lambda a, b: a > b, BinaryOperation.GREATER, id="greater"),
    pytest.param(
        lambda a, b: a >= b, BinaryOperation.GREATER_EQUAL, id="greater_equal"
    ),
]


@pytest.mark.parametrize(("dunder", "expected_operation"), _COMPARISON_DUNDER_CASES)
def test_identifier_comparison_dunder_over_two_identifiers_produces_binary_expression(
    dunder: _BinaryDunder, expected_operation: BinaryOperation
) -> None:
    """Test each comparison dunder builds a binary expression with the expected op."""
    a = Identifier("a")
    b = Identifier("b")

    expression = dunder(a, b)

    expected = BinaryExpression(
        expected_operation, IdentifierExpression(a), IdentifierExpression(b)
    )
    assert isinstance(expression, BinaryExpression)
    assert expression.is_structurally_equivalent(expected)


# =============================================================================
# Identity equality and hashing are unchanged
# =============================================================================


def test_identifier_equality_remains_identity_based() -> None:
    """Test ``==`` on identifiers still compares by ``id`` (not a binary node)."""
    a = Identifier("a")
    same = Identifier.deserialize_from_dict({"id": a.id, "name_hint": a.name_hint})
    other = Identifier("a")

    assert a == same
    assert a != other
    assert isinstance(a == same, bool)


def test_identifier_hash_remains_consistent_with_equality() -> None:
    """Test ``hash(a) == hash(same)`` after adding the new comparison dunders."""
    a = Identifier("a")
    same = Identifier.deserialize_from_dict({"id": a.id, "name_hint": a.name_hint})

    assert hash(a) == hash(same)


# =============================================================================
# Composition with other Identifier dunders
# =============================================================================


def test_identifier_chained_arithmetic_associates_left() -> None:
    """Test ``a + b - c`` parses left-to-right via consecutive dunder calls."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")

    expression = a + b - c

    expected = BinaryExpression(
        BinaryOperation.SUBTRACT,
        BinaryExpression(
            BinaryOperation.ADD, IdentifierExpression(a), IdentifierExpression(b)
        ),
        IdentifierExpression(c),
    )
    assert expression.is_structurally_equivalent(expected)


def test_identifier_negation_inside_arithmetic_wraps_operand() -> None:
    """Test ``a + -b`` nests a UnaryExpression inside the BinaryExpression."""
    a = Identifier("a")
    b = Identifier("b")

    expression = a + -b

    expected = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(a),
        UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(b)),
    )
    assert expression.is_structurally_equivalent(expected)


def test_identifier_arithmetic_composes_with_expression_operands() -> None:
    """Test ``a + (b * 2)`` accepts an Expression on either side of the dunder."""
    a = Identifier("a")
    b = Identifier("b")
    inner = b * LiteralExpression(2)

    expression = a + inner

    expected = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(a),
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            IdentifierExpression(b),
            LiteralExpression(2),
        ),
    )
    assert expression.is_structurally_equivalent(expected)
