"""Behavioral tests for `EquationConstraint`."""

import logging

import pytest

from fhy_core.symbolic.constraint import ConstraintOutcome, EquationConstraint
from fhy_core.symbolic.expression import (
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
        pytest.param(
            LiteralExpression(True),
            LiteralExpression(0),
            True,
            id="literal_true",
        ),
        pytest.param(
            LiteralExpression(False),
            LiteralExpression(0),
            False,
            id="literal_false",
        ),
        pytest.param(
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(True),
            True,
            id="variable_substituted_true",
        ),
        pytest.param(
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(False),
            False,
            id="variable_substituted_false",
        ),
        pytest.param(
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            LiteralExpression(True),
            False,
            id="not_true",
        ),
        pytest.param(
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(False)),
            LiteralExpression(True),
            True,
            id="not_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(True),
            ),
            LiteralExpression(True),
            True,
            id="and_true_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(True),
            False,
            id="and_true_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            True,
            id="or_true_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                LiteralExpression(False),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            False,
            id="or_false_false",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.EQUAL,
                LiteralExpression(True),
                LiteralExpression(True),
            ),
            LiteralExpression(0),
            True,
            id="equal_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                LiteralExpression(True),
                LiteralExpression(False),
            ),
            LiteralExpression(0),
            True,
            id="not_equal_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LESS, LiteralExpression(5), LiteralExpression(10)
            ),
            LiteralExpression(0),
            True,
            id="less_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                LiteralExpression(10),
                LiteralExpression(10),
            ),
            LiteralExpression(0),
            True,
            id="less_equal_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.GREATER, LiteralExpression(10), LiteralExpression(5)
            ),
            LiteralExpression(0),
            True,
            id="greater_true",
        ),
        pytest.param(
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                LiteralExpression(10),
                LiteralExpression(10),
            ),
            LiteralExpression(0),
            True,
            id="greater_equal_true",
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


@pytest.mark.parametrize(
    "primitive",
    [
        pytest.param(0, id="int_zero"),
        pytest.param(1, id="int_one"),
        pytest.param(True, id="bool_true"),
        pytest.param(False, id="bool_false"),
        pytest.param(1.5, id="float"),
    ],
)
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

    with caplog.at_level(logging.WARNING, logger="fhy_core.symbolic.constraint"):
        result = constraint.is_satisfied(LiteralExpression(True))

    assert result is False
    assert any(record.levelno == logging.WARNING for record in caplog.records), (
        "expected a WARNING-level log record from fhy_core.symbolic.constraint"
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

    with caplog.at_level(logging.WARNING, logger="fhy_core.symbolic.constraint"):
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

    with caplog.at_level(logging.WARNING, logger="fhy_core.symbolic.constraint"):
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


# =============================================================================
# Tri-state `evaluate` outcomes
# =============================================================================


def test_equation_constraint_evaluate_reports_satisfied_for_bool_true() -> None:
    """Test `evaluate` reports SATISFIED when the reduction is bool ``True``."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, IdentifierExpression(x))

    assert constraint.evaluate(LiteralExpression(True)) is ConstraintOutcome.SATISFIED


def test_equation_constraint_evaluate_reports_violated_for_bool_false() -> None:
    """Test `evaluate` reports VIOLATED when the reduction is bool ``False``."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, IdentifierExpression(x))

    assert constraint.evaluate(LiteralExpression(False)) is ConstraintOutcome.VIOLATED


def test_equation_constraint_evaluate_reports_violated_for_non_bool_literal() -> None:
    """Test `evaluate` reports VIOLATED when the reduction is a non-bool literal."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(x, LiteralExpression(1))

    assert constraint.evaluate(LiteralExpression(0)) is ConstraintOutcome.VIOLATED


def test_equation_constraint_evaluate_reports_undecided_when_free_variable_remains(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the indeterminate case reports UNDECIDED, logs, and rejects.

    The expression references a second free identifier the substitution
    cannot bind, so the simplifier cannot reduce it to a literal.
    `evaluate` must report UNDECIDED (and log a warning), while the
    conservative `is_satisfied` still returns ``False``.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    constraint = EquationConstraint(x, IdentifierExpression(y))

    with caplog.at_level(logging.WARNING, logger="fhy_core.symbolic.constraint"):
        outcome = constraint.evaluate(LiteralExpression(True))

    assert outcome is ConstraintOutcome.UNDECIDED
    assert constraint.is_satisfied(LiteralExpression(True)) is False
    assert any(record.levelno == logging.WARNING for record in caplog.records), (
        "expected a WARNING-level log record from fhy_core.symbolic.constraint"
    )
