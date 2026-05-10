"""Cross-cutting tests that span multiple modules in `fhy_core.expression`."""

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
from fhy_core.expression.passes.basic import IdentifierCollector, IdentifierSubstituter
from fhy_core.expression.passes.sympy import ExpressionToSympyConverter
from fhy_core.expression.passes.type_checker import ExpressionTypeChecker
from fhy_core.expression.passes.z3 import ExpressionToZ3Converter
from fhy_core.pass_infrastructure import CompilerPass, PassInfo

from .conftest import mock_identifier

# =============================================================================
# Pass registry - every expression pass must self-register
# =============================================================================

_EXPECTED_REGISTRATIONS: list[tuple[str, type]] = [
    ("fhy_core.expression.collect_identifiers", IdentifierCollector),
    ("fhy_core.expression.substitute_identifiers", IdentifierSubstituter),
    ("fhy_core.expression.type_check", ExpressionTypeChecker),
    ("fhy_core.expression.to_sympy", ExpressionToSympyConverter),
    ("fhy_core.expression.to_z3", ExpressionToZ3Converter),
]


@pytest.mark.parametrize("pass_name, pass_class", _EXPECTED_REGISTRATIONS)
def test_expression_pass_is_registered_under_expected_name(
    pass_name: str, pass_class: type
) -> None:
    """Test each expression pass registers under its expected name and class."""
    registered = CompilerPass.get_registered_passes()
    assert pass_name in registered
    info = registered[pass_name]
    assert isinstance(info, PassInfo)
    assert info.pass_type is pass_class
    assert info.description.strip() != ""


# =============================================================================
# Serialize -> deserialize round-trip integrity
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        LiteralExpression(42),
        IdentifierExpression(mock_identifier("x", 7)),
        UnaryExpression(
            UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("y", 8))
        ),
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(mock_identifier("a", 9)),
            LiteralExpression(5),
        ),
    ],
)
def test_serialize_round_trip_preserves_structure(expression: Expression) -> None:
    """Test `serialize_to_dict` -> `deserialize_from_dict` preserves structure."""
    restored = Expression.deserialize_from_dict(expression.serialize_to_dict())
    assert restored.is_structurally_equivalent(expression)


# =============================================================================
# Parse -> pformat -> parse round-trip
#
# These fixtures only cover expressions the pretty-printer emits in a form the
# parser can re-tokenize. The pretty-printer's docstring notes that full
# round-trip is not guaranteed - the fixture list is curated accordingly. The
# parser mints fresh `Identifier` objects on each call, so we check that the
# formatter's output is idempotent under a second parse-and-format cycle
# rather than comparing trees directly.
# =============================================================================


@pytest.mark.parametrize(
    "source",
    [
        "x + 1",
        "(a <= b) && (c != d)",
        "-x ** 2",
        "a * b + c",
        "(i + j) * 2",
        "!p || q",
    ],
)
def test_parse_then_format_is_idempotent_under_a_second_cycle(source: str) -> None:
    """Test `pformat(parse(s))` is stable under a further parse-and-format cycle."""
    once = pformat_expression(parse_expression(source))
    twice = pformat_expression(parse_expression(once))
    assert once == twice
