"""Tests for `fhy_core.expression.pprint`."""

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
    pformat_expression,
)
from fhy_core.expression.pprint import ExpressionPrettyFormatter
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

from .conftest import mock_identifier

# =============================================================================
# Symbolic format (default)
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(4.5), "4.5"),
        (LiteralExpression("0.1"), "0.1"),
        (IdentifierExpression(Identifier("baz")), "baz"),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            "(!True)",
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression("3.14"),
                LiteralExpression(10.5),
            ),
            "(3.14 * 10.5)",
        ),
    ],
)
def test_pformat_expression_renders_symbolic_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression` emits the expected symbolic-notation string."""
    assert pformat_expression(expression) == expected_str


# =============================================================================
# Functional format
# =============================================================================


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (IdentifierExpression(Identifier("test_identifier")), "test_identifier"),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
            "(negate 5)",
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(5),
                LiteralExpression(10),
            ),
            "(add 5 10)",
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            "(divide (negate 5) 10)",
        ),
    ],
)
def test_pformat_expression_renders_functional_form(
    expression: Expression, expected_str: str
) -> None:
    """Test `pformat_expression(..., functional=True)` emits functional-notation."""
    assert pformat_expression(expression, functional=True) == expected_str


# =============================================================================
# Identifier-id visibility
# =============================================================================


def test_pformat_expression_with_show_id_includes_name_hint_and_id() -> None:
    """Test `pformat_expression(..., show_id=True)` renders both name hint and id."""
    identifier = Identifier("foo")
    result = pformat_expression(IdentifierExpression(identifier), show_id=True)
    assert identifier.name_hint in result
    assert str(identifier.id) in result


# =============================================================================
# ExpressionPrettyFormatter defaults
# =============================================================================


def test_pretty_formatter_default_does_not_show_identifier_id() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render identifiers without an id."""
    identifier = Identifier("name_only")
    result = ExpressionPrettyFormatter()(IdentifierExpression(identifier))
    assert result == identifier.name_hint


def test_pretty_formatter_default_uses_symbolic_notation() -> None:
    """Test `ExpressionPrettyFormatter()` defaults render binary ops symbolically."""
    expression = BinaryExpression(
        BinaryOperation.ADD, LiteralExpression(1), LiteralExpression(2)
    )
    assert ExpressionPrettyFormatter()(expression) == "(1 + 2)"


# =============================================================================
# Symbolic-format round-trip with the parser
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(LiteralExpression(5), id="native_int_literal"),
        pytest.param(LiteralExpression("3.5"), id="str_form_float_literal"),
        pytest.param(LiteralExpression(True), id="bool_literal"),
        pytest.param(IdentifierExpression(mock_identifier("x", 0)), id="identifier"),
        pytest.param(
            UnaryExpression(
                UnaryOperation.NEGATE,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            id="unary_negate",
        ),
        pytest.param(
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("flag", 0)),
            ),
            id="unary_logical_not",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(1),
                BinaryExpression(
                    BinaryOperation.MULTIPLY,
                    LiteralExpression(2),
                    LiteralExpression(3),
                ),
            ),
            id="add_times_precedence",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("a", 0)),
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    IdentifierExpression(mock_identifier("b", 1)),
                    IdentifierExpression(mock_identifier("c", 2)),
                ),
            ),
            id="and_or",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(2),
            ),
            id="power",
        ),
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_pformat_expression_round_trips_through_parse_expression(
    expression: Expression,
) -> None:
    """Test ``parse_expression(pformat_expression(e))`` round-trips back to ``e``.

    Round-trip is guaranteed for ``int`` / ``bool`` / str-form-float literals
    and for tree shapes the pretty-printer emits in the parser-recognized
    form. Native ``float`` literals are explicitly lossy and are exercised
    separately below.
    """
    formatted = pformat_expression(expression)
    reparsed = parse_expression(formatted)
    assert reparsed.is_structurally_equivalent(expression)


def test_pformat_native_float_literal_does_not_round_trip_through_parser() -> None:
    """Test ``parse(pformat(LiteralExpression(<float>)))`` is structurally non-equiv.

    The design treats native ``float`` as IEEE-754-imprecise and str-form
    floats as exact decimals. The pretty-printer emits the float's textual
    form; the parser stores that as a str literal, structurally unequal to
    the original native-float literal. This is the documented lossy case
    in the round-trip contract - locked in here so the asymmetry can't
    quietly disappear.
    """
    original = LiteralExpression(0.5)

    reparsed = parse_expression(pformat_expression(original))

    assert isinstance(reparsed, LiteralExpression)
    assert type(reparsed.value) is str
    assert reparsed.value == "0.5"
    assert not original.is_structurally_equivalent(reparsed)


def test_pformat_int_literal_round_trips_through_parser() -> None:
    """Test ``parse(pformat(LiteralExpression(<int>)))`` is structurally equivalent.

    Positive control for the native-float lossy case above.
    """
    original = LiteralExpression(42)

    reparsed = parse_expression(pformat_expression(original))

    assert original.is_structurally_equivalent(reparsed)
    assert isinstance(reparsed, LiteralExpression)
    assert type(reparsed.value) is int
    assert reparsed.value == 42


# =============================================================================
# Defensive guards
# =============================================================================


def test_pretty_formatter_call_rejects_non_string_formatted_result() -> None:
    """Test `__call__` raises `TypeError` when the visit result is not a `str`."""

    class _NonStringFormatter(ExpressionPrettyFormatter):
        def visit_literal_expression(
            self, literal_expression: LiteralExpression
        ) -> str:
            return 42  # type: ignore[return-value]  # intentional contract violation

    with pytest.raises(TypeError, match=r"Invalid formatted expression type"):
        _NonStringFormatter()(LiteralExpression(5))


def test_pretty_formatter_get_noop_output_raises() -> None:
    """Test `ExpressionPrettyFormatter.get_noop_output` raises `PassExecutionError`."""
    formatter = ExpressionPrettyFormatter()

    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        formatter.get_noop_output(LiteralExpression(0))
