"""Behavioral tests for `EquationConstraint`."""

import pytest

from fhy_core.constraint import EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    pformat_expression,
)

from .conftest import mock_identifier


@pytest.mark.parametrize(
    "expression, value, expected_outcome",
    [
        (LiteralExpression(True), LiteralExpression(0), True),
        (LiteralExpression(False), LiteralExpression(0), False),
        (IdentifierExpression(mock_identifier("x", 0)), LiteralExpression(True), True),
        (
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(False),
            False,
        ),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            LiteralExpression(True),
            False,
        ),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(False)),
            LiteralExpression(True),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(True),
            ),
            LiteralExpression(True),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(True),
            False,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                LiteralExpression(False),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                LiteralExpression(True),
                LiteralExpression(True),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS, LiteralExpression(5), LiteralExpression(10)
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                LiteralExpression(10),
                LiteralExpression(10),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER, LiteralExpression(10), LiteralExpression(5)
            ),
            LiteralExpression(0),
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                LiteralExpression(10),
                LiteralExpression(10),
            ),
            LiteralExpression(0),
            True,
        ),
    ],
)
def test_equation_constraint_is_satisfied(
    expression: object, value: LiteralExpression, expected_outcome: bool
) -> None:
    """Test `is_satisfied` evaluates the expression against the substituted value."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, expression)  # type: ignore[arg-type]
    assert constraint.is_satisfied(value) is expected_outcome


@pytest.mark.parametrize("primitive", [0, 1, True, False, 1.5])
def test_equation_constraint_is_satisfied_accepts_literal_primitive(
    primitive: int | float | bool,
) -> None:
    """Test primitive scalars are auto-wrapped in a `LiteralExpression`."""
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x)
    constraint = EquationConstraint(x, expression)
    assert constraint.is_satisfied(primitive) == constraint.is_satisfied(
        LiteralExpression(primitive)
    )


def test_equation_constraint_call_delegates_to_is_satisfied() -> None:
    """Test ``constraint(value)`` matches ``constraint.is_satisfied(value)``."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, IdentifierExpression(x))
    value = LiteralExpression(True)
    assert constraint(value) == constraint.is_satisfied(value)


def test_equation_constraint_variable_property_returns_constructor_argument() -> None:
    """Test the ``variable`` property returns the identifier passed to ``__init__``."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(True))
    assert constraint.variable is x


def test_equation_constraint_convert_to_expression_returns_inner_expression() -> None:
    """Test ``convert_to_expression`` returns the wrapped expression unchanged."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )
    constraint = EquationConstraint(x, expression)
    assert constraint.convert_to_expression().is_structurally_equivalent(expression)


def test_equation_constraint_repr_matches_expression_repr() -> None:
    """Test ``repr(constraint)`` matches ``repr`` of the inner expression."""
    x = mock_identifier("x", 0)
    expression = LiteralExpression(True)
    constraint = EquationConstraint(x, expression)
    assert repr(constraint) == repr(expression)


def test_equation_constraint_str_matches_expression_pformat() -> None:
    """Test ``str(constraint)`` matches ``pformat_expression`` of the expression."""
    x = mock_identifier("x", 0)
    expression = LiteralExpression(True)
    constraint = EquationConstraint(x, expression)
    assert str(constraint) == pformat_expression(expression)


# =============================================================================
# Adversarial / edge cases
# =============================================================================


def test_equation_constraint_returns_false_for_non_bool_literal_result() -> None:
    """Test `is_satisfied` rejects numeric-truthy literals as the reduction.

    The check ``isinstance(result.value, bool)`` is strict
    (``isinstance(1, bool)`` is ``False``).
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(1))
    assert constraint.is_satisfied(LiteralExpression(0)) is False


def test_equation_constraint_returns_false_when_unable_to_reduce_to_literal() -> None:
    """Test `is_satisfied` returns ``False`` when the expression has a free variable.

    A free variable other than ``self.variable`` leaves the expression
    un-reducible; ``simplify_expression`` returns a non-literal, so
    ``is_satisfied`` returns ``False`` rather than raising.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, IdentifierExpression(y))
    assert constraint.is_satisfied(LiteralExpression(True)) is False


def test_equation_constraint_ignores_value_when_variable_absent_from_expression() -> (
    None
):
    """Test `is_satisfied` ignores the value when the variable is absent.

    If the variable doesn't appear in the expression, the substitution
    is a no-op — only the standalone expression's truth value matters.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(True))
    assert constraint.is_satisfied(LiteralExpression(0)) is True
    assert constraint.is_satisfied(LiteralExpression(False)) is True
