"""Behavioral tests for `EquationConstraint`."""

import logging

import pytest

from fhy_core.constraint import EquationConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    logical_and,
    make_binary_expression,
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
    expression = make_binary_expression(BinaryOperation.EQUAL, x, True)
    constraint = EquationConstraint(x, expression)

    assert constraint.convert_to_expression().is_structurally_equivalent(expression)


def test_equation_constraint_repr_includes_class_name_and_variable() -> None:
    """Test ``repr`` includes the class name and the constrained variable."""
    x = mock_identifier("x", 0)
    expression = LiteralExpression(True)
    constraint = EquationConstraint(x, expression)

    rendered = repr(constraint)

    assert "EquationConstraint" in rendered
    assert repr(x) in rendered


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
    """Test `is_satisfied` rejects numeric-truthy literals as the reduction."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(1))

    assert constraint.is_satisfied(LiteralExpression(0)) is False


def test_equation_constraint_returns_false_and_logs_when_free_variable_remains(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the indeterminate case returns False and logs a warning."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, IdentifierExpression(y))

    with caplog.at_level(logging.WARNING, logger="fhy_core.constraint"):
        result = constraint.is_satisfied(LiteralExpression(True))

    assert result is False
    assert any(record.levelno == logging.WARNING for record in caplog.records), (
        "expected a WARNING-level log record from fhy_core.constraint"
    )


def test_equation_constraint_returns_false_and_logs_when_simplifier_yields_non_literal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a non-reducible expression returns False and emits a single warning."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    expression = logical_and(y, z)
    constraint = EquationConstraint(x, expression)

    with caplog.at_level(logging.WARNING, logger="fhy_core.constraint"):
        result = constraint.is_satisfied(LiteralExpression(True))

    assert result is False
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected at least one WARNING log record"


def test_equation_constraint_returns_false_without_warning_for_non_bool_literal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a non-bool literal reduction returns False without warning."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(1))

    with caplog.at_level(logging.WARNING, logger="fhy_core.constraint"):
        result = constraint.is_satisfied(LiteralExpression(0))

    assert result is False
    assert not [r for r in caplog.records if r.levelno == logging.WARNING], (
        "non-bool literal reduction should not emit a warning"
    )


def test_equation_constraint_ignores_value_when_variable_absent_from_expression() -> (
    None
):
    """Test `is_satisfied` ignores the value when the variable is absent."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(True))

    assert constraint.is_satisfied(LiteralExpression(0)) is True
    assert constraint.is_satisfied(LiteralExpression(False)) is True
