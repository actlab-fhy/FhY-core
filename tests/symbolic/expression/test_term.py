"""Tests `Expression` participating in the `HasFreeIdentifiers` / `Term` traits.

The expression IR has no internal binders, so every referenced identifier is
free and substitution is always capture-free; these tests pin that behavior on
the trait methods exposed directly on expression nodes.
"""

import pytest

from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.term import HasFreeIdentifiers, Term

from .conftest import mock_identifier


def test_expression_is_a_term() -> None:
    """Test expression nodes satisfy the `Term` and `HasFreeIdentifiers` protocols."""
    node = IdentifierExpression(mock_identifier("x", 1))
    assert isinstance(node, Term)
    assert isinstance(node, HasFreeIdentifiers)
    assert isinstance(LiteralExpression(1), Term)


def test_identifier_expression_free_identifier_is_itself() -> None:
    """Test an identifier reference reports itself as free."""
    x = mock_identifier("x", 1)
    assert IdentifierExpression(x).get_free_identifiers() == frozenset({x})


def test_literal_expression_has_no_free_identifiers() -> None:
    """Test a literal has no free identifiers."""
    assert LiteralExpression(7).get_free_identifiers() == frozenset()


def test_composite_free_identifiers_union_over_children() -> None:
    """Test free identifiers of a composite union over its sub-expressions."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            IdentifierExpression(y),
        ),
    )
    assert expression.get_free_identifiers() == frozenset({x, y})


def test_unary_expression_free_identifiers_walk_into_operand() -> None:
    """Test free identifiers reach into a unary operand."""
    x = mock_identifier("x", 1)
    node = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))
    assert node.get_free_identifiers() == frozenset({x})


def test_call_expression_free_identifiers_union_over_arguments() -> None:
    """Test free identifiers of a call union over its argument expressions."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    call = CallExpression("f", (IdentifierExpression(x), IdentifierExpression(y)))
    assert call.get_free_identifiers() == frozenset({x, y})


def test_substitute_replaces_identifier_with_term() -> None:
    """Test substituting a mapped identifier yields the replacement expression."""
    x = mock_identifier("x", 1)
    result = IdentifierExpression(x).substitute({x: LiteralExpression(5)})
    assert result.is_structurally_equivalent(LiteralExpression(5))


def test_substitute_leaves_unmapped_identifier_untouched() -> None:
    """Test an identifier absent from the map is returned unchanged."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    node = IdentifierExpression(y)
    assert node.substitute({x: LiteralExpression(5)}) is node


def test_substitute_returns_literal_unchanged() -> None:
    """Test substituting into a literal returns the same instance."""
    literal = LiteralExpression(3)
    assert literal.substitute({}) is literal


def test_substitute_recurses_into_composite() -> None:
    """Test substitution rewrites mapped identifiers throughout a composite."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = expression.substitute({x: LiteralExpression(1)})

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), IdentifierExpression(y)
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_recurses_into_unary_expression() -> None:
    """Test substitution traverses into a unary operand."""
    x = mock_identifier("x", 1)
    node = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))

    result = node.substitute({x: LiteralExpression(7)})

    expected = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(7))
    assert result.is_structurally_equivalent(expected)


def test_substitute_preserves_unchanged_nested_literal_identity() -> None:
    """Test unchanged literal children keep their instance through substitution."""
    literal = LiteralExpression(42)
    tree = BinaryExpression(BinaryOperation.ADD, literal, literal)

    result = tree.substitute({})

    assert isinstance(result, BinaryExpression)
    assert result.left is literal
    assert result.right is literal


def test_substitute_renames_identifier_via_identifier_expression() -> None:
    """Test substituting an identifier with an identifier reference renames it."""
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(5)
    )

    result = expression.substitute({x: IdentifierExpression(y)})

    expected = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(5)
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_rejects_non_expression_replacement() -> None:
    """Test substituting an identifier with a non-Expression term raises `TypeError`."""
    x = mock_identifier("x", 1)
    node = IdentifierExpression(x)
    with pytest.raises(TypeError):
        node.substitute({x: object()})  # type: ignore[dict-item]
