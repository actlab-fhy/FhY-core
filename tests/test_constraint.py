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

from .utils import mock_identifier


@pytest.mark.parametrize(
    "constraint, values, expected_outcome",
    [
        (EquationConstraint(LiteralExpression(True)), {}, True),
        (EquationConstraint(LiteralExpression(False)), {}, False),
        (
            EquationConstraint(IdentifierExpression(mock_identifier("x", 0))),
            {mock_identifier("x", 0): LiteralExpression(True)},
            True,
        ),
        (
            EquationConstraint(IdentifierExpression(mock_identifier("x", 0))),
            {mock_identifier("x", 0): LiteralExpression(False)},
            False,
        ),
        (
            EquationConstraint(
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True))
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(False))
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(True),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LOGICAL_AND,
                    LiteralExpression(True),
                    LiteralExpression(False),
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(True),
                    LiteralExpression(False),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    LiteralExpression(False),
                    LiteralExpression(False),
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(True),
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.NOT_EQUAL,
                    LiteralExpression(True),
                    LiteralExpression(False),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(5), LiteralExpression(10)
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LESS, LiteralExpression(10), LiteralExpression(5)
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.LESS_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(5),
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(10), LiteralExpression(5)
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER, LiteralExpression(5), LiteralExpression(10)
                )
            ),
            {},
            False,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(10),
                    LiteralExpression(10),
                )
            ),
            {},
            True,
        ),
        (
            EquationConstraint(
                BinaryExpression(
                    BinaryOperation.GREATER_EQUAL,
                    LiteralExpression(5),
                    LiteralExpression(10),
                )
            ),
            {},
            False,
        ),
    ],
)
def test_equation_constraint_checks_correctly(
    constraint: EquationConstraint,
    values: dict[Identifier, LiteralType],
    expected_outcome: bool,
):
    """Test the equation constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(values) == expected_outcome


@pytest.mark.parametrize(
    "constraint, values, expected_outcome",
    [
        (
            InSetConstraint({mock_identifier("x", 0)}, {1, 2, 3}),
            {mock_identifier("x", 0): 1},
            True,
        ),
        (
            InSetConstraint({mock_identifier("x", 0)}, {1, 2, 3}),
            {mock_identifier("x", 0): 4},
            False,
        ),
        (
            InSetConstraint(
                {mock_identifier("x", 0), mock_identifier("y", 1)}, {"a", "b", "c"}
            ),
            {mock_identifier("x", 0): "a", mock_identifier("y", 1): "b"},
            True,
        ),
        (
            InSetConstraint(
                {mock_identifier("x", 0), mock_identifier("y", 1)}, {"a", "b", "c"}
            ),
            {mock_identifier("x", 0): "a", mock_identifier("y", 1): "d"},
            False,
        ),
    ],
)
def test_in_set_constraint_checks_correctly(
    constraint: InSetConstraint,
    values: dict[Identifier, LiteralType],
    expected_outcome: bool,
):
    """Test the in-set constraint evaluates correctly when checked."""
    assert constraint.is_satisfied(values) == expected_outcome


@pytest.mark.parametrize(
    "constraint, values, expected_outcome",
    [
        (
            NotInSetConstraint({mock_identifier("x", 0)}, {1, 2, 3}),
            {mock_identifier("x", 0): 1},
            False,
        ),
        (
            NotInSetConstraint({mock_identifier("x", 0)}, {1, 2, 3}),
            {mock_identifier("x", 0): 4},
            True,
        ),
        (
            NotInSetConstraint(
                {mock_identifier("x", 0), mock_identifier("y", 1)}, {"a", "b", "c"}
            ),
            {mock_identifier("x", 0): "a", mock_identifier("y", 1): "b"},
            False,
        ),
        (
            NotInSetConstraint(
                {mock_identifier("x", 0), mock_identifier("y", 1)}, {"a", "b", "c"}
            ),
            {mock_identifier("x", 0): "a", mock_identifier("y", 1): "d"},
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
