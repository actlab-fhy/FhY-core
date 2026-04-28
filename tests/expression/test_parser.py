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
# Parser - precedence and associativity edge cases
# =============================================================================


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_chained_exponent_is_left_associative_per_grammar() -> None:
    """Test ``a ** b ** c`` parses left-to-right at the exponentiation level.

    The grammar docstring on `ExpressionParser` notes that ``**`` is parsed
    left-to-right despite being right-associative in typical usage; this test
    locks that documented behavior in.
    """
    a = IdentifierExpression(mock_identifier("a", 0))
    b = IdentifierExpression(mock_identifier("b", 1))
    c = IdentifierExpression(mock_identifier("c", 2))
    expected = BinaryExpression(
        BinaryOperation.POWER,
        BinaryExpression(BinaryOperation.POWER, a, b),
        c,
    )
    assert parse_expression("a ** b ** c").is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_exponent_binds_tighter_than_unary_minus() -> None:
    """Test ``-a ** b`` parses as ``-(a ** b)``."""
    a = IdentifierExpression(mock_identifier("a", 0))
    b = IdentifierExpression(mock_identifier("b", 1))
    expected = UnaryExpression(
        UnaryOperation.NEGATE, BinaryExpression(BinaryOperation.POWER, a, b)
    )
    assert parse_expression("-a ** b").is_structurally_equivalent(expected)


@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        (
            "--x",
            UnaryExpression(
                UnaryOperation.NEGATE,
                UnaryExpression(
                    UnaryOperation.NEGATE,
                    IdentifierExpression(mock_identifier("x", 0)),
                ),
            ),
        ),
        (
            "!!x",
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                UnaryExpression(
                    UnaryOperation.LOGICAL_NOT,
                    IdentifierExpression(mock_identifier("x", 0)),
                ),
            ),
        ),
        (
            "-+-x",
            UnaryExpression(
                UnaryOperation.NEGATE,
                UnaryExpression(
                    UnaryOperation.POSITIVE,
                    UnaryExpression(
                        UnaryOperation.NEGATE,
                        IdentifierExpression(mock_identifier("x", 0)),
                    ),
                ),
            ),
        ),
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_stacked_unary_operators_nest_in_token_order(
    expression_str: str, expected_tree: Expression
) -> None:
    """Test stacked unary operators produce nested ``UnaryExpression`` nodes."""
    assert parse_expression(expression_str).is_structurally_equivalent(expected_tree)


@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        (
            "1+2*3",
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression("1"),
                BinaryExpression(
                    BinaryOperation.MULTIPLY,
                    LiteralExpression("2"),
                    LiteralExpression("3"),
                ),
            ),
        ),
        (
            "a&&b||c",
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    IdentifierExpression(mock_identifier("a", 0)),
                    IdentifierExpression(mock_identifier("b", 1)),
                ),
                IdentifierExpression(mock_identifier("c", 2)),
            ),
        ),
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_expression_handles_operators_without_surrounding_whitespace(
    expression_str: str, expected_tree: Expression
) -> None:
    """Test the tokenizer separates adjacent operators without requiring spaces."""
    assert parse_expression(expression_str).is_structurally_equivalent(expected_tree)


# =============================================================================
# Tokenizer - numeric oddities
# =============================================================================


@pytest.mark.parametrize(
    "expression_str, expected_tokens",
    [
        ("1.", ["1"]),
        (".1", ["1"]),
        ("00", ["00"]),
    ],
)
def test_tokenize_expression_handles_malformed_numeric_fragments(
    expression_str: str, expected_tokens: list[str]
) -> None:
    """Test the tokenizer drops stray ``.`` and preserves leading-zero integer runs."""
    assert tokenize_expression(expression_str) == expected_tokens


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


@pytest.mark.parametrize("name", ["Truex", "TRUE", "_True", "True_"])
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_boolean_keyword_lookalikes_stay_identifiers(name: str) -> None:
    """Test names sharing a prefix or differing in case from bools stay identifiers."""
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


@pytest.mark.parametrize(
    "malformed_expression",
    [
        pytest.param("(1 + 2", id="unclosed_paren"),
        pytest.param("((1 + 2) (", id="mismatched_open_paren"),
        pytest.param("1 +", id="trailing_operator"),
        pytest.param("", id="empty_input"),
    ],
)
def test_parse_expression_raises_on_malformed_input(malformed_expression: str) -> None:
    """Test malformed inputs raise `RuntimeError` from the parser."""
    with pytest.raises(RuntimeError):
        parse_expression(malformed_expression)


def test_parse_expression_error_message_reports_offending_token_position() -> None:
    """Test the error message includes the token index of the unexpected token."""
    with pytest.raises(RuntimeError, match=r'Unexpected token "\)" at position 0'):
        parse_expression(") + 1")


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
