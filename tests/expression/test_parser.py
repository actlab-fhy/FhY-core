"""Tests for `fhy_core.expression.parser`."""

from unittest.mock import patch

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    parse_expression,
    tokenize_expression,
)

from .conftest import mock_identifier

# =============================================================================
# Tokenizer
# =============================================================================


@pytest.mark.parametrize(
    "expression_str, expected_tokens",
    [
        ("7", ["7"]),
        ("342.5", ["342.5"]),
        ("True", ["True"]),
        ("False", ["False"]),
        ("y", ["y"]),
        ("-5", ["-", "5"]),
        ("10.5 + _x_", ["10.5", "+", "_x_"]),
        (
            "((10+2) >= 2) > 5",
            ["(", "(", "10", "+", "2", ")", ">=", "2", ")", ">", "5"],
        ),
        ("x // 5 + 3", ["x", "//", "5", "+", "3"]),
        ("a ** b", ["a", "**", "b"]),
        ("a && b", ["a", "&&", "b"]),
        ("a || b", ["a", "||", "b"]),
        ("a == b", ["a", "==", "b"]),
        ("a != b", ["a", "!=", "b"]),
        ("a <= b", ["a", "<=", "b"]),
        ("a >= b", ["a", ">=", "b"]),
        ("!a", ["!", "a"]),
        ("  a   +\tb\n", ["a", "+", "b"]),
    ],
)
def test_tokenize_expression_covers_operator_set(
    expression_str: str, expected_tokens: list[str]
) -> None:
    """Test `tokenize_expression` recognizes every operator in the grammar."""
    assert tokenize_expression(expression_str) == expected_tokens


# =============================================================================
# Parser - happy paths from the grammar
# =============================================================================


@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        ("5", LiteralExpression("5")),
        ("3.2", LiteralExpression("3.2")),
        ("True", LiteralExpression(True)),
        ("False", LiteralExpression(False)),
        ("-5", UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("5"))),
        (
            "34 / 5.7",
            BinaryExpression(
                BinaryOperation.DIVIDE,
                LiteralExpression("34"),
                LiteralExpression("5.7"),
            ),
        ),
        (
            "10 + -2 * 5",
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression("10"),
                BinaryExpression(
                    BinaryOperation.MULTIPLY,
                    UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("2")),
                    LiteralExpression("5"),
                ),
            ),
        ),
        (
            "(2 + (5+6)) * -0",
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                BinaryExpression(
                    BinaryOperation.ADD,
                    LiteralExpression("2"),
                    BinaryExpression(
                        BinaryOperation.ADD,
                        LiteralExpression("5"),
                        LiteralExpression("6"),
                    ),
                ),
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("0")),
            ),
        ),
        (
            "x + y",
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
        ),
        (
            "i >= 1",
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("i", 0)),
                LiteralExpression("1"),
            ),
        ),
        (
            "a == b",
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("a", 0)),
                IdentifierExpression(mock_identifier("b", 1)),
            ),
        ),
        (
            "a != b",
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("a", 0)),
                IdentifierExpression(mock_identifier("b", 1)),
            ),
        ),
        (
            "a < b == c",
            BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("a", 0)),
                    IdentifierExpression(mock_identifier("b", 1)),
                ),
                IdentifierExpression(mock_identifier("c", 2)),
            ),
        ),
        (
            "a && b == c",
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("a", 0)),
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    IdentifierExpression(mock_identifier("b", 1)),
                    IdentifierExpression(mock_identifier("c", 2)),
                ),
            ),
        ),
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_expression_produces_expected_tree(
    expression_str: str, expected_tree: Expression
) -> None:
    """Test `parse_expression` builds the structurally-expected tree for each input."""
    assert parse_expression(expression_str).is_structurally_equivalent(expected_tree)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_expression_respects_full_precedence_ladder() -> None:
    """Test `!a && b == c < d + e * f ** g` parses per the documented grammar."""
    a = IdentifierExpression(mock_identifier("a", 0))
    b = IdentifierExpression(mock_identifier("b", 1))
    c = IdentifierExpression(mock_identifier("c", 2))
    d = IdentifierExpression(mock_identifier("d", 3))
    e = IdentifierExpression(mock_identifier("e", 4))
    f = IdentifierExpression(mock_identifier("f", 5))
    g = IdentifierExpression(mock_identifier("g", 6))
    expected = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        UnaryExpression(UnaryOperation.LOGICAL_NOT, a),
        BinaryExpression(
            BinaryOperation.EQUAL,
            b,
            BinaryExpression(
                BinaryOperation.LESS,
                c,
                BinaryExpression(
                    BinaryOperation.ADD,
                    d,
                    BinaryExpression(
                        BinaryOperation.MULTIPLY,
                        e,
                        BinaryExpression(BinaryOperation.POWER, f, g),
                    ),
                ),
            ),
        ),
    )
    result = parse_expression("!a && b == c < d + e * f ** g")
    assert result.is_structurally_equivalent(expected)


# =============================================================================
# Parser - boolean literals in compound expressions
# =============================================================================


@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        (
            "True && False",
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
        ),
        (
            "(True)",
            LiteralExpression(True),
        ),
        (
            "(False)",
            LiteralExpression(False),
        ),
    ],
)
def test_parse_expression_recognizes_boolean_keywords_in_context(
    expression_str: str, expected_tree: Expression
) -> None:
    """Test ``True``/``False`` are recognized as booleans even when surrounded by
    other tokens.

    This guards against regressions where a substring identity comparison
    (``token is "True"``) replaces value equality: under those conditions the
    runtime tokens produced by the tokenizer are not interned alongside the
    Python string literal, so the identity check would fail and the booleans
    would fall through to the identifier branch.
    """
    assert parse_expression(expression_str).is_structurally_equivalent(expected_tree)


# =============================================================================
# Parser - keyword vs identifier boundary
# =============================================================================


@pytest.mark.parametrize("name", ["Alpha", "Baseline", "Bool", "Check", "Ephemeral"])
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_capitalized_identifiers_before_keywords_stay_identifiers(name: str) -> None:
    """Test names sorting before ``True``/``False`` lex-wise stay identifiers."""
    result = parse_expression(name)
    expected = IdentifierExpression(mock_identifier(name, 0))
    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_capitalized_identifier_remains_identifier_in_compound_expression() -> None:
    """Test ``Baseline + 1`` yields an identifier-expression, not a boolean literal."""
    result = parse_expression("Baseline + 1")
    expected = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(mock_identifier("Baseline", 0)),
        LiteralExpression("1"),
    )
    assert result.is_structurally_equivalent(expected)


# =============================================================================
# Parser - error paths
# =============================================================================


def test_parse_expression_raises_on_unclosed_paren() -> None:
    """Test a missing closing paren raises `RuntimeError`."""
    with pytest.raises(RuntimeError):
        parse_expression("(1 + 2")


def test_parse_expression_raises_on_mismatched_open_paren() -> None:
    """Test a stray opening paren where `)` is expected raises `RuntimeError`."""
    with pytest.raises(RuntimeError):
        parse_expression("((1 + 2) (")


def test_parse_expression_raises_on_trailing_operator() -> None:
    """Test an expression ending with an operator raises `RuntimeError`."""
    with pytest.raises(RuntimeError):
        parse_expression("1 +")


def test_parse_expression_raises_on_empty_input() -> None:
    """Test an empty expression string raises `RuntimeError`."""
    with pytest.raises(RuntimeError):
        parse_expression("")


# =============================================================================
# Parser - large-input boundary
# =============================================================================


def _count_literal_leaves(expression: Expression, target_value: str) -> int:
    if isinstance(expression, LiteralExpression):
        return 1 if expression.value == target_value else 0
    elif isinstance(expression, BinaryExpression):
        return _count_literal_leaves(
            expression.left, target_value
        ) + _count_literal_leaves(expression.right, target_value)
    elif isinstance(expression, UnaryExpression):
        return _count_literal_leaves(expression.operand, target_value)
    else:
        return 0


def test_parse_expression_handles_token_count_past_cpython_small_int_cache() -> None:
    """Test parsing a 300-term sum yields a tree with 300 literal leaves of value 1."""
    num_terms = 300
    source = " + ".join(["1"] * num_terms)
    tree = parse_expression(source)
    assert _count_literal_leaves(tree, "1") == num_terms


def test_parse_expression_reuses_identifier_for_repeated_symbol() -> None:
    """Test repeated identifier symbols within one parse share the same `Identifier`."""
    tree = parse_expression("x + x")
    assert isinstance(tree, BinaryExpression)
    assert isinstance(tree.left, IdentifierExpression)
    assert isinstance(tree.right, IdentifierExpression)
    assert tree.left.identifier is tree.right.identifier
