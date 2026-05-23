"""Parser tests for the ``?:`` ternary syntax and generic call syntax."""

from unittest.mock import patch

import pytest

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    IdentifierExpression,
    LiteralExpression,
    TernaryExpression,
    UnaryExpression,
    UnaryOperation,
    parse_expression,
    tokenize_expression,
)

from .conftest import mock_identifier

# =============================================================================
# Tokenizer additions
# =============================================================================


@pytest.mark.parametrize(
    "expression_str, expected_tokens",
    [
        pytest.param("max(a, b)", ["max", "(", "a", ",", "b", ")"], id="binary_call"),
        pytest.param(
            "c ? t : f",
            ["c", "?", "t", ":", "f"],
            id="ternary",
        ),
        pytest.param("f()", ["f", "(", ")"], id="empty_arg_list"),
        pytest.param(
            "max(a , b , c)",
            ["max", "(", "a", ",", "b", ",", "c", ")"],
            id="comma_with_whitespace",
        ),
    ],
)
def test_tokenize_expression_recognizes_call_and_ternary_tokens(
    expression_str: str, expected_tokens: list[str]
) -> None:
    """Test the tokenizer treats ``,``, ``?``, and ``:`` as discrete tokens."""
    assert tokenize_expression(expression_str) == expected_tokens


# =============================================================================
# Parser: ternary `?:` syntax
# =============================================================================


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_ternary_with_literal_operands_builds_ternary_expression() -> None:
    """Test ``True ? 1 : 2`` parses as a ``TernaryExpression``."""
    result = parse_expression("True ? 1 : 2")
    expected = TernaryExpression(
        LiteralExpression(True), LiteralExpression(1), LiteralExpression(2)
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_ternary_with_compound_condition_and_branches() -> None:
    """Test ``x > 0 ? x : -x`` parses with branches and a compound condition."""
    result = parse_expression("x > 0 ? x : -x")

    expected = TernaryExpression(
        BinaryExpression(
            BinaryOperation.GREATER,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(0),
        ),
        IdentifierExpression(mock_identifier("x", 0)),
        UnaryExpression(
            UnaryOperation.NEGATE,
            IdentifierExpression(mock_identifier("x", 0)),
        ),
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_ternary_right_associates_in_false_branch() -> None:
    """Test ``a ? b : c ? d : e`` parses as ``a ? b : (c ? d : e)``."""
    result = parse_expression("a ? b : c ? d : e")

    a, b, c, d, e = (mock_identifier(name, index) for index, name in enumerate("abcde"))
    expected = TernaryExpression(
        IdentifierExpression(a),
        IdentifierExpression(b),
        TernaryExpression(
            IdentifierExpression(c),
            IdentifierExpression(d),
            IdentifierExpression(e),
        ),
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_ternary_inside_parenthesised_subexpression() -> None:
    """Test ``(a ? b : c) + 1`` parses with the ternary on the left of ``+``."""
    result = parse_expression("(a ? b : c) + 1")

    a, b, c = (mock_identifier(name, index) for index, name in enumerate("abc"))
    expected = BinaryExpression(
        BinaryOperation.ADD,
        TernaryExpression(
            IdentifierExpression(a),
            IdentifierExpression(b),
            IdentifierExpression(c),
        ),
        LiteralExpression(1),
    )

    assert result.is_structurally_equivalent(expected)


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param("True ? 1", id="missing_colon_and_false_branch"),
        pytest.param("True ? 1 :", id="missing_false_branch"),
        pytest.param("True ?", id="missing_branches"),
        pytest.param("? 1 : 2", id="missing_condition"),
        pytest.param("True ? : 2", id="missing_true_branch"),
    ],
)
def test_parse_ternary_with_malformed_syntax_raises(malformed: str) -> None:
    """Test malformed ternary syntax raises ``ParseError`` from the parser."""
    with pytest.raises(RuntimeError):
        parse_expression(malformed)


# =============================================================================
# Parser: generic call syntax
# =============================================================================


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_to_named_function_with_two_arguments() -> None:
    """Test ``max(a, b)`` parses as a ``CallExpression`` with two arguments."""
    result = parse_expression("max(a, b)")

    expected = CallExpression(
        "max",
        (
            IdentifierExpression(mock_identifier("a", 0)),
            IdentifierExpression(mock_identifier("b", 1)),
        ),
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_with_empty_argument_list() -> None:
    """Test ``f()`` parses as a ``CallExpression`` with zero arguments."""
    result = parse_expression("f()")

    expected = CallExpression("f", ())

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_with_single_argument() -> None:
    """Test ``f(x)`` parses as a ``CallExpression`` with one argument."""
    result = parse_expression("f(x)")

    expected = CallExpression("f", (IdentifierExpression(mock_identifier("x", 0)),))

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_with_compound_argument_expression() -> None:
    """Test ``f(a + b, c * 2)`` parses each argument as an expression."""
    result = parse_expression("f(a + b, c * 2)")

    expected = CallExpression(
        "f",
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("a", 0)),
                IdentifierExpression(mock_identifier("b", 1)),
            ),
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("c", 2)),
                LiteralExpression(2),
            ),
        ),
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_nested_inside_arithmetic() -> None:
    """Test ``1 + max(a, b)`` parses with the call as the right operand of ``+``."""
    result = parse_expression("1 + max(a, b)")

    expected = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        CallExpression(
            "max",
            (
                IdentifierExpression(mock_identifier("a", 0)),
                IdentifierExpression(mock_identifier("b", 1)),
            ),
        ),
    )

    assert result.is_structurally_equivalent(expected)


@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_call_with_nested_call_argument() -> None:
    """Test ``max(min(a, b), c)`` parses with the inner call as an argument."""
    result = parse_expression("max(min(a, b), c)")

    expected = CallExpression(
        "max",
        (
            CallExpression(
                "min",
                (
                    IdentifierExpression(mock_identifier("a", 0)),
                    IdentifierExpression(mock_identifier("b", 1)),
                ),
            ),
            IdentifierExpression(mock_identifier("c", 2)),
        ),
    )

    assert result.is_structurally_equivalent(expected)


def test_parser_does_not_special_case_max_for_variadic_arguments() -> None:
    """Test ``max(a, b, c)`` parses with three arguments verbatim (no folding)."""
    result = parse_expression("max(a, b, c)")

    assert isinstance(result, CallExpression)
    assert result.function_name == "max"
    assert len(result.arguments) == 3


# =============================================================================
# Parser: malformed call syntax
# =============================================================================


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param("max(a,", id="trailing_comma_no_close"),
        pytest.param("max(a, b,)", id="trailing_comma_before_close"),
        pytest.param("max(,)", id="leading_comma_in_args"),
        pytest.param("max(a b)", id="missing_separator"),
        pytest.param("max(", id="missing_close_paren"),
    ],
)
def test_parse_call_raises_for_malformed_argument_list(malformed: str) -> None:
    """Test malformed call argument lists raise ``ParseError`` (``RuntimeError``)."""
    with pytest.raises(RuntimeError):
        parse_expression(malformed)
