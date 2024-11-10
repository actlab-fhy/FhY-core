"""Tests the constraint utility."""

import pytest
from fhy_core.constraint import EquationConstraint, InSetConstraint, NotInSetConstraint
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    LiteralType,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier

from .utils import assert_exact_expression_equality, mock_identifier


@pytest.mark.parametrize(
    "constraint, value, expected_outcome",
    [
        (
            EquationConstraint(mock_identifier("x", 0), LiteralExpression(True)),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(mock_identifier("x", 0), LiteralExpression(False)),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0), IdentifierExpression(mock_identifier("x", 0))
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0), IdentifierExpression(mock_identifier("x", 0))
            ),
            LiteralExpression(False),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            ),
            LiteralExpression(True),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(False)),
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(True),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(True),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(False),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(5), LiteralExpression(10)
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(10), LiteralExpression(5)
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(5),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(10), LiteralExpression(5)
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(5), LiteralExpression(10)
                ),
            ),
            LiteralExpression(0),
            False,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            True,
        ),
        (
            EquationConstraint(
                mock_identifier("x", 0),
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(5),
                    LiteralExpression(10),
                ),
            ),
            LiteralExpression(0),
            False,
        ),
    ],
)
def test_equation_constraint_checks_correctly(
    constraint: EquationConstraint,
    value: LiteralExpression,
    expected_outcome: bool,
):
    """Test the equation constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(value) == expected_outcome


@pytest.mark.parametrize(
    "constraint, value, expected_outcome",
    [
        (
            InSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            1,
            True,
        ),
        (
            InSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            4,
            False,
        ),
        (
            InSetConstraint(mock_identifier("x", 0), {"a", "b", "c"}),
            "a",
            True,
        ),
        (
            InSetConstraint(mock_identifier("y", 1), {"a", "b", "c"}),
            "d",
            False,
        ),
    ],
)
def test_in_set_constraint_checks_correctly(
    constraint: InSetConstraint,
    value: LiteralExpression,
    expected_outcome: bool,
):
    """Test the in-set constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(value) == expected_outcome


@pytest.mark.parametrize(
    "constraint, values, expected_outcome",
    [
        (
            NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            1,
            False,
        ),
        (
            NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3}),
            4,
            True,
        ),
        (
            NotInSetConstraint(mock_identifier("x", 0), {"a", "b", "c"}),
            "a",
            False,
        ),
        (
            NotInSetConstraint(mock_identifier("y", 1), {"a", "b", "c"}),
            "d",
            True,
        ),
    ],
)
def test_not_in_set_constraint_checks_correctly(
    constraint: NotInSetConstraint,
    values: dict[Identifier, LiteralType],
    expected_outcome: bool,
):
    """Test the not-in-set constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(values) == expected_outcome


# TODO: If there is ever an equality for constraints, use this instead of
#       accessing private attributes for the three following tests.
def test_copy_equation_constraint():
    """Test the equation constraint is copied correctly."""
    constraint = EquationConstraint(mock_identifier("x", 0), LiteralExpression(True))
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert_exact_expression_equality(constraint._expression, copy._expression)
    assert constraint is not copy


def test_copy_in_set_constraint():
    """Test the in-set constraint is copied correctly."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2, 3})
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert constraint._valid_values == copy._valid_values
    assert constraint is not copy


def test_copy_not_in_set_constraint():
    """Test the not-in-set constraint is copied correctly."""
    constraint = NotInSetConstraint(mock_identifier("x", 0), {1, 2, 3})
    copy = constraint.copy()
    assert constraint.variable == copy.variable
    assert constraint._invalid_values == copy._invalid_values
    assert constraint is not copy


def test_convert_equation_constraint_to_expression():
    """Test the equation constraint is converted to an expression correctly."""
    constraint_expression = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(mock_identifier("x", 0)),
        LiteralExpression(True),
    )
    constraint = EquationConstraint(mock_identifier("x", 0), constraint_expression)
    expression = constraint.convert_to_expression()
    assert_exact_expression_equality(constraint_expression, expression)


def test_convert_in_set_constraint_to_expression():
    """Test the in-set constraint is converted to an expression correctly."""
    constraint = InSetConstraint(mock_identifier("x", 0), {1, 2})
    expression = constraint.convert_to_expression()
    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_OR,
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(1),
        ),
        BinaryExpression(
            BinaryOperation.EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(2),
        ),
    )
    assert_exact_expression_equality(expected_expression, expression)


def test_convert_not_in_set_constraint_to_expression():
    """Test the not-in-set constraint is converted to an expression correctly."""
    constraint = NotInSetConstraint(mock_identifier("x", 0), {1, 2})
    expression = constraint.convert_to_expression()
    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(1),
        ),
        BinaryExpression(
            BinaryOperation.NOT_EQUAL,
            IdentifierExpression(mock_identifier("x", 0)),
            LiteralExpression(2),
        ),
    )
    assert_exact_expression_equality(expected_expression, expression)
