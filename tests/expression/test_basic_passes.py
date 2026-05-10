"""Tests for `fhy_core.expression.passes.basic`."""

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    collect_identifiers,
    replace_identifiers,
    substitute_identifiers,
)
from fhy_core.expression.passes.basic import IdentifierCollector, IdentifierSubstituter
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import CompilerPass, PassInfo

# =============================================================================
# collect_identifiers
# =============================================================================


def test_collect_identifiers_returns_every_distinct_identifier_in_a_tree() -> None:
    """Test `collect_identifiers` returns exactly the set of identifiers referenced."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            IdentifierExpression(y),
        ),
    )
    assert collect_identifiers(expression) == {x, y}


def test_collect_identifiers_on_literal_only_tree_returns_empty_set() -> None:
    """Test `collect_identifiers` returns an empty set for a literal-only tree."""
    assert collect_identifiers(LiteralExpression(7)) == set()


# =============================================================================
# substitute_identifiers
# =============================================================================


def test_substitute_identifiers_replaces_mapped_identifiers_with_expressions() -> None:
    """Test `substitute_identifiers` rewrites each mapped identifier in the tree."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            IdentifierExpression(y),
        ),
    )
    substitutions: dict[Identifier, Expression] = {
        x: LiteralExpression(10),
        y: LiteralExpression(5),
    }

    result = substitute_identifiers(expression, substitutions)

    expected = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(10),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            LiteralExpression(5),
        ),
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_identifiers_leaves_unmapped_identifiers_untouched() -> None:
    """Test identifiers not present in the substitution map are unchanged."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(y)
    )

    result = substitute_identifiers(expression, {x: LiteralExpression(1)})

    expected = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), IdentifierExpression(y)
    )
    assert result.is_structurally_equivalent(expected)


def test_substitute_identifiers_recurses_into_unary_expression() -> None:
    """Test `substitute_identifiers` traverses into a `UnaryExpression` operand."""
    x = Identifier("x")
    expression = UnaryExpression(UnaryOperation.NEGATE, IdentifierExpression(x))

    result = substitute_identifiers(expression, {x: LiteralExpression(7)})

    expected = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(7))
    assert result.is_structurally_equivalent(expected)


def test_substitute_identifiers_rebuilds_literal_subexpressions_fresh() -> None:
    """Test `substitute_identifiers` rebuilds literal sub-expressions as new nodes."""
    literal = LiteralExpression(3)

    result = substitute_identifiers(literal, {})

    assert result.is_structurally_equivalent(literal)
    assert result is not literal


def test_identifier_substituter_get_noop_output_returns_input() -> None:
    """Test `IdentifierSubstituter.get_noop_output` returns its input unchanged."""
    expression = LiteralExpression(0)
    substituter = IdentifierSubstituter({})

    assert substituter.get_noop_output(expression) is expression


# =============================================================================
# replace_identifiers
# =============================================================================


def test_replace_identifiers_rewrites_each_mapped_identifier_by_name() -> None:
    """Test `replace_identifiers` rewrites identifier nodes to reference the new id."""
    x = Identifier("x")
    y = Identifier("y")
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(5)
    )

    result = replace_identifiers(expression, {x: y})

    expected = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(y), LiteralExpression(5)
    )
    assert result.is_structurally_equivalent(expected)


# =============================================================================
# Pass registry
# =============================================================================


@pytest.mark.parametrize(
    "pass_name, pass_class",
    [
        ("fhy_core.expression.collect_identifiers", IdentifierCollector),
        ("fhy_core.expression.substitute_identifiers", IdentifierSubstituter),
    ],
)
def test_basic_expression_passes_are_registered_under_expected_names(
    pass_name: str, pass_class: type
) -> None:
    """Test each basic expression pass is registered under its expected name."""
    registered = CompilerPass.get_registered_passes()
    assert pass_name in registered
    info = registered[pass_name]
    assert isinstance(info, PassInfo)
    assert info.pass_type is pass_class
    assert info.description.strip() != ""
