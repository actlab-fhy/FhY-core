"""Tests the expression utility."""

import operator
from collections.abc import Callable
from typing import Any

import pytest
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import DeserializationValueError, SerializedDict
from fhy_core.trait import (
    HasOperands,
    StructuralEquivalence,
)

from .conftest import mock_identifier


def test_unary_expression() -> None:
    """Test that the unary expression is correctly initialized."""
    operand = LiteralExpression(5)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    assert expr.operation == UnaryOperation.NEGATE
    assert expr.operand is operand


def test_binary_expression() -> None:
    """Test that the binary expression is correctly initialized."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    assert expr.operation == BinaryOperation.ADD
    assert expr.left is left
    assert expr.right is right


def test_identifier_expression() -> None:
    """Test that the identifier expression is correctly initialized."""
    identifier = Identifier("test_identifier")
    expr = IdentifierExpression(identifier)
    assert expr.identifier == identifier


@pytest.mark.parametrize("value", [5, 3.14, True, "3.14"])
def test_literal_expression_valid_values(value: int | float | bool | str) -> None:
    """Test that the literal expression is correctly initialized with valid values."""
    expr = LiteralExpression(value)
    assert expr.value == value


def test_literal_expression_invalid_string() -> None:
    """Test that the literal expression raises an exception for invalid string
    values.
    """
    with pytest.raises(ValueError):
        LiteralExpression("invalid_literal")


def test_unary_expression_has_operands_runtime_protocol() -> None:
    """Test `UnaryExpression` satisfies the `HasOperands` runtime protocol."""
    expression = UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(1))
    assert isinstance(expression, HasOperands)


def test_unary_expression_operands_contract() -> None:
    """Test `UnaryExpression` operands method."""
    operand = LiteralExpression(1)
    expression = UnaryExpression(UnaryOperation.NEGATE, operand)
    assert expression.get_operands() == (operand,)


def test_binary_expression_has_operands_runtime_protocol() -> None:
    """Test `BinaryExpression` satisfies the `HasOperands` runtime protocol."""
    expression = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        LiteralExpression(2),
    )
    assert isinstance(expression, HasOperands)


def test_binary_expression_operands_contract() -> None:
    """Test `BinaryExpression` operands method."""
    left = LiteralExpression(1)
    right = LiteralExpression(2)
    expression = BinaryExpression(BinaryOperation.ADD, left, right)
    assert expression.get_operands() == (left, right)


def test_expression_is_structural_equivalence_runtime_protocol() -> None:
    """Test `Expression` satisfies `StructuralEquivalence` runtime protocol."""
    expression = LiteralExpression(7)
    assert isinstance(expression, StructuralEquivalence)


def test_expression_structural_equivalence_true_for_same_tree() -> None:
    """Test expression structural equivalence for identical trees."""
    left = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        LiteralExpression(2),
    )
    right = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        LiteralExpression(2),
    )
    assert left.is_structurally_equivalent(right)


def test_expression_structural_equivalence_false_for_different_tree() -> None:
    """Test expression structural equivalence for different trees."""
    left = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        LiteralExpression(2),
    )
    right = BinaryExpression(
        BinaryOperation.SUBTRACT,
        LiteralExpression(1),
        LiteralExpression(2),
    )
    assert not left.is_structurally_equivalent(right)


@pytest.mark.parametrize(
    "unary_operator, expected_operation",
    [
        (operator.neg, UnaryOperation.NEGATE),
        (operator.pos, UnaryOperation.POSITIVE),
        (lambda x: x.logical_not(), UnaryOperation.LOGICAL_NOT),
    ],
)
def test_unary_operator_dunder_methods(
    unary_operator: Callable[[Expression], Expression],
    expected_operation: UnaryOperation,
) -> None:
    """Test that the unary operation dunder methods correctly create unary
    expressions.
    """
    operand = LiteralExpression(5)
    expected_expr = UnaryExpression(expected_operation, operand)
    assert unary_operator(operand).is_structurally_equivalent(expected_expr)


_binary_operator_operations_pairs = pytest.mark.parametrize(
    "binary_operator, expected_operation",
    [
        (operator.add, BinaryOperation.ADD),
        (operator.sub, BinaryOperation.SUBTRACT),
        (operator.mul, BinaryOperation.MULTIPLY),
        (operator.truediv, BinaryOperation.DIVIDE),
        (operator.mod, BinaryOperation.MODULO),
        (operator.pow, BinaryOperation.POWER),
        (lambda x, y: x.equals(y), BinaryOperation.EQUAL),
        (lambda x, y: x.not_equals(y), BinaryOperation.NOT_EQUAL),
        (operator.lt, BinaryOperation.LESS),
        (operator.le, BinaryOperation.LESS_EQUAL),
        (operator.gt, BinaryOperation.GREATER),
        (operator.ge, BinaryOperation.GREATER_EQUAL),
    ],
)


@_binary_operator_operations_pairs
def test_binary_operation_dunder_methods(
    binary_operator: Callable[[Expression, Expression], Expression],
    expected_operation: BinaryOperation,
) -> None:
    """Test that the binary operation dunder methods correctly create binary"""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expected_expr = BinaryExpression(expected_operation, left, right)
    assert binary_operator(left, right).is_structurally_equivalent(expected_expr)


@_binary_operator_operations_pairs
@pytest.mark.parametrize(
    "left, right, expected_right_type",
    [
        (LiteralExpression(5), 10, LiteralExpression),
        (IdentifierExpression(mock_identifier("x", 0)), 10.23, LiteralExpression),
        (
            UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(10)),
            False,
            LiteralExpression,
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            "2.264",
            LiteralExpression,
        ),
        (LiteralExpression(5), mock_identifier("x", 2), IdentifierExpression),
    ],
)
def test_binary_operation_left_dunder_methods_for_literals(
    binary_operator: Callable[[Any, Any], Expression],
    expected_operation: BinaryOperation,
    left: Expression,
    right: Identifier | str | float | int | bool,
    expected_right_type: type[Expression],
) -> None:
    """Test that the binary operation left dunder methods correctly create
    literal expressions.
    """
    expected_expr = BinaryExpression(
        expected_operation,
        left,
        expected_right_type(right),  # type: ignore[call-arg]  # test: Expression subclass constructors vary
    )
    assert binary_operator(left, right).is_structurally_equivalent(expected_expr)


@pytest.mark.parametrize(
    "binary_operator, expected_operation",
    [
        (operator.add, BinaryOperation.ADD),
        (operator.sub, BinaryOperation.SUBTRACT),
        (operator.mul, BinaryOperation.MULTIPLY),
        (operator.truediv, BinaryOperation.DIVIDE),
        (operator.mod, BinaryOperation.MODULO),
        (operator.pow, BinaryOperation.POWER),
    ],
)
@pytest.mark.parametrize(
    "left, expected_left_type, right",
    [
        (6, LiteralExpression, LiteralExpression(10)),
        (10.3, LiteralExpression, IdentifierExpression(mock_identifier("y", 19))),
        (
            True,
            LiteralExpression,
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(15)),
        ),
        (
            "2.4",
            LiteralExpression,
            BinaryExpression(
                BinaryOperation.SUBTRACT, LiteralExpression(2), LiteralExpression(3)
            ),
        ),
        (mock_identifier("x", 1), IdentifierExpression, LiteralExpression(5)),
    ],
)
def test_binary_operation_right_dunder_methods_for_literals(
    binary_operator: Callable[[Any, Any], Expression],
    expected_operation: BinaryOperation,
    left: Identifier | str | float | int | bool,
    expected_left_type: type[Expression],
    right: Expression,
) -> None:
    """Test that the binary operation right dunder methods correctly create
    literal expressions.
    """
    if binary_operator == operator.mod and isinstance(left, str):
        pytest.skip(
            "Modulo operation with string on the left is reserved for formatting."
        )
    expected_expr = BinaryExpression(
        expected_operation,
        expected_left_type(left),  # type: ignore[call-arg]  # test: Expression subclass constructors vary
        right,
    )
    assert binary_operator(left, right).is_structurally_equivalent(expected_expr)


def test_binary_operation_dunder_fails_to_create_expression_with_unknown_type() -> None:
    """Test that the binary operation dunder methods fail to create an expression
    with an unknown type.
    """
    with pytest.raises(ValueError):
        operator.add(LiteralExpression(5), [])


def test_logical_and() -> None:
    """Test that the logical and static method creates an AND tree."""
    expression_1 = LiteralExpression(True)
    expression_2 = LiteralExpression(False)
    expression_3 = True

    result = Expression.logical_and(expression_1, expression_2, expression_3)  # type: ignore[arg-type]  # test: accepts coerceable literals

    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_AND,
        expression_1,
        BinaryExpression(
            BinaryOperation.LOGICAL_AND, expression_2, LiteralExpression(expression_3)
        ),
    )
    assert result.is_structurally_equivalent(expected_expression)


def test_logical_or() -> None:
    """Test that the logical or static method creates an OR tree."""
    expression_1 = LiteralExpression(True)
    expression_2 = False
    expression_3 = LiteralExpression(True)

    result = Expression.logical_or(expression_1, expression_2, expression_3)  # type: ignore[arg-type]  # test: accepts coerceable literals

    expected_expression = BinaryExpression(
        BinaryOperation.LOGICAL_OR,
        expression_1,
        BinaryExpression(
            BinaryOperation.LOGICAL_OR, LiteralExpression(expression_2), expression_3
        ),
    )
    assert result.is_structurally_equivalent(expected_expression)


@pytest.mark.parametrize(
    "expression, expected_dict",
    [
        (
            LiteralExpression(True),
            {
                "__type__": "literal_expression",
                "__data__": {"value": True},
            },
        ),
        (
            IdentifierExpression(mock_identifier("x", 1)),
            {
                "__type__": "identifier_expression",
                "__data__": {"identifier": {"id": 1, "name_hint": "x"}},
            },
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("y", 2))
            ),
            {
                "__type__": "unary_expression",
                "__data__": {
                    "operation": "negate",
                    "operand": {
                        "__type__": "identifier_expression",
                        "__data__": {
                            "identifier": {"id": 2, "name_hint": "y"},
                        },
                    },
                },
            },
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {
                "__type__": "binary_expression",
                "__data__": {
                    "operation": "add",
                    "left": {
                        "__type__": "identifier_expression",
                        "__data__": {
                            "identifier": {"id": 0, "name_hint": "x"},
                        },
                    },
                    "right": {
                        "__type__": "literal_expression",
                        "__data__": {
                            "value": 5,
                        },
                    },
                },
            },
        ),
    ],
)
def test_dict_serialization(
    expression: Expression, expected_dict: SerializedDict
) -> None:
    """Test that expressions can be serialized/deserialized via a dictionary."""
    assert expression.serialize_to_dict() == expected_dict
    assert Expression.deserialize_from_dict(expected_dict).is_structurally_equivalent(
        expression
    )


def test_dict_deserialization_fails_with_invalid_unary_operation() -> None:
    """Test dictionary deserialization fails with invalid unary operation name."""
    data = {
        "__type__": "unary_expression",
        "__data__": {"operation": "invalid_operation", "operand": {}},
    }
    with pytest.raises(DeserializationValueError):
        Expression.deserialize_from_dict(data)  # type: ignore[arg-type]  # test: invalid input


def test_dict_deserialization_fails_with_invalid_binary_operation() -> None:
    """Test dictionary deserialization fails with invalid binary operation name."""
    data = {
        "__type__": "binary_expression",
        "__data__": {
            "operation": "invalid_operation",
            "left": {},
            "right": {},
        },
    }
    with pytest.raises(DeserializationValueError):
        Expression.deserialize_from_dict(data)  # type: ignore[arg-type]  # test: invalid input


# TODO: Check serialization structure errors and value errors for all types.
