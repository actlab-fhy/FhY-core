"""Tests the expression utility."""

from unittest.mock import MagicMock, patch

import pytest
import sympy
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    collect_identifiers,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    copy_expression,
    parse_expression,
    pformat_expression,
    simplify_expression,
    substitute_sympy_expression_variables,
    tokenize_expression,
)
from fhy_core.expression.core import LiteralType
from fhy_core.expression.visitor import (
    ExpressionBasePass,
)
from fhy_core.identifier import Identifier

from .utils import mock_identifier


def _assert_exact_expression_equality(
    expression1: Expression, expression2: Expression
) -> None:
    if isinstance(expression1, LiteralExpression) and isinstance(
        expression2, LiteralExpression
    ):
        assert expression1.value == expression2.value
    elif isinstance(expression1, IdentifierExpression) and isinstance(
        expression2, IdentifierExpression
    ):
        assert expression1.identifier == expression2.identifier
    elif isinstance(expression1, UnaryExpression) and isinstance(
        expression2, UnaryExpression
    ):
        assert expression1.operation == expression2.operation
        _assert_exact_expression_equality(expression1.operand, expression2.operand)
    elif isinstance(expression1, BinaryExpression) and isinstance(
        expression2, BinaryExpression
    ):
        assert expression1.operation == expression2.operation
        _assert_exact_expression_equality(expression1.left, expression2.left)
        _assert_exact_expression_equality(expression1.right, expression2.right)
    else:
        assert False, "Expression trees did not have the same structure."


def test_unary_expression():
    """Test that the unary expression is correctly initialized."""
    operand = LiteralExpression(5)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    assert expr.operation == UnaryOperation.NEGATE
    assert expr.operand is operand


def test_binary_expression():
    """Test that the binary expression is correctly initialized."""
    left = LiteralExpression(5)
    right = LiteralExpression(10)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    assert expr.operation == BinaryOperation.ADD
    assert expr.left is left
    assert expr.right is right


def test_identifier_expression():
    """Test that the identifier expression is correctly initialized."""
    identifier = Identifier("test_identifier")
    expr = IdentifierExpression(identifier)
    assert expr.identifier == identifier


@pytest.mark.parametrize("value", [5, 3.14, True, complex(2, 3), "2+3j"])
def test_literal_expression_valid_values(value):
    """Test that the literal expression is correctly initialized with valid values."""
    expr = LiteralExpression(value)
    assert expr._value == value if not isinstance(value, str) else complex(value)


def test_literal_expression_invalid_string():
    """Test that the literal expression raises an exception for invalid string
    values.
    """
    with pytest.raises(ValueError, match="Invalid literal expression value:"):
        LiteralExpression("invalid_literal")


# TODO: More tests for tokenization!
@pytest.mark.parametrize(
    "expression_str, expected_tokens",
    [
        ("7", ["7"]),
        ("342.5", ["342.5"]),
        ("2j", ["2j"]),
        ("8.9j", ["8.9j"]),
        ("True", ["True"]),
        ("False", ["False"]),
        ("y", ["y"]),
        ("-5", ["-", "5"]),
        ("10.5 + _x_", ["10.5", "+", "_x_"]),
        ("(5 + 6j)", ["(", "5", "+", "6j", ")"]),
        (
            "((10j+2) >> 2) > 5",
            ["(", "(", "10j", "+", "2", ")", ">>", "2", ")", ">", "5"],
        ),
    ],
)
def test_tokenize_expression(expression_str: str, expected_tokens: list[str]):
    """Test that the expression is correctly tokenized."""
    assert tokenize_expression(expression_str) == expected_tokens


# TODO: More tests for parsing!
@pytest.mark.parametrize(
    "expression_str, expected_tree",
    [
        ("5", LiteralExpression("5")),
        ("3.2", LiteralExpression("3.2")),
        ("10j", LiteralExpression("10j")),
        ("3.6j", LiteralExpression("3.6j")),
        ("True", LiteralExpression(True)),
        ("False", LiteralExpression(False)),
        ("~5", UnaryExpression(UnaryOperation.BITWISE_NOT, LiteralExpression("5"))),
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
            "(2 + (5+6j)) * -0",
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                BinaryExpression(
                    BinaryOperation.ADD,
                    LiteralExpression("2"),
                    BinaryExpression(
                        BinaryOperation.ADD,
                        LiteralExpression("5"),
                        LiteralExpression("6j"),
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
    ],
)
@patch("fhy_core.identifier.Identifier._next_id", 0)
def test_parse_expression(expression_str: str, expected_tree: Expression):
    """Test that the expression is correctly parsed."""
    result = parse_expression(expression_str)
    _assert_exact_expression_equality(result, expected_tree)


# TODO: Refactor pformat tests to be together and just alter the parameters
@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(4.5), "4.5"),
        (
            IdentifierExpression(Identifier("baz")),
            "baz",
        ),
        (
            UnaryExpression(UnaryOperation.LOGICAL_NOT, LiteralExpression(True)),
            "(!True)",
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(5 + 6j),
                LiteralExpression(10.5),
            ),
            "((5+6j) * 10.5)",
        ),
    ],
)
def test_pformat_expression(expression: Expression, expected_str: str):
    """Test that the expression is correctly pretty-formatted."""
    assert pformat_expression(expression) == expected_str


@pytest.mark.parametrize(
    "expression, expected_str",
    [
        (LiteralExpression(5), "5"),
        (
            IdentifierExpression(Identifier("test_identifier")),
            "test_identifier",
        ),
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
def test_pformat_expressions_with_functional(expression: Expression, expected_str: str):
    """Test that the expression is correctly pretty-formatted in a functional
    format.
    """
    assert pformat_expression(expression, functional=True) == expected_str


def test_pformat_expressions_with_id():
    """Test that the expression is correctly pretty-formatted with the
    identifier ID.
    """
    identifier = Identifier("test_identifier")
    expression = IdentifierExpression(identifier)
    assert (
        pformat_expression(expression, show_id=True)
        == f"test_identifier_{identifier.id}"
    )


def test_collect_expression_identifiers():
    """Test that the identifiers are correctly collected from an expression."""
    x = Identifier("x")
    y = Identifier("y")
    expr = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(5),
            IdentifierExpression(y),
        ),
    )
    assert collect_identifiers(expr) == {x, y}


# TODO: Revisit the use of MagicMock here and in the following tests.
@pytest.fixture
def base_pass():
    class ConcreteBasePass(ExpressionBasePass):
        """Concrete base pass for testing"""

    base_pass = ConcreteBasePass()
    base_pass.visit_unary_expression = MagicMock()
    base_pass.visit_binary_expression = MagicMock()
    base_pass.visit_identifier_expression = MagicMock()
    base_pass.visit_literal_expression = MagicMock()
    return base_pass


def test_base_pass_call_calls_visit(base_pass: ExpressionBasePass):
    """Test that the visit method calls the correct visit method for
    Expression.
    """
    base_pass.visit = MagicMock()
    expr = MagicMock()
    base_pass(expr)
    base_pass.visit.assert_called_once_with(expr)


def test_base_pass_calls_unary_expression_visitor(base_pass: ExpressionBasePass):
    """Test that the visit method calls the correct visit method for
    UnaryExpression.
    """
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=MagicMock())
    base_pass.visit(expr)
    base_pass.visit_unary_expression.assert_called_once_with(expr)


def test_base_pass_calls_binary_expression_visitor(base_pass: ExpressionBasePass):
    """Test that the visit method calls the correct visit method for
    BinaryExpression.
    """
    expr = BinaryExpression(
        operation=BinaryOperation.ADD, left=MagicMock(), right=MagicMock()
    )
    base_pass.visit(expr)
    base_pass.visit_binary_expression.assert_called_once_with(expr)


def test_base_pass_calls_identifier_expression_visitor(base_pass: ExpressionBasePass):
    """Test that the visit method calls the correct visit method for
    IdentifierExpression.
    """
    expr = IdentifierExpression(identifier=Identifier("x"))
    base_pass.visit(expr)
    base_pass.visit_identifier_expression.assert_called_once_with(expr)


def test_base_pass_calls_literal_visitor(base_pass: ExpressionBasePass):
    """Test that the visit method calls the correct visit method for
    LiteralExpression.
    """
    expr = LiteralExpression(value=42)
    base_pass.visit(expr)
    base_pass.visit_literal_expression.assert_called_once_with(expr)


def test_base_pass_with_unsupported_expression(base_pass: ExpressionBasePass):
    """Test that the visit method raises an exception for unsupported
    expressions.
    """
    with pytest.raises(NotImplementedError, match="Unsupported expression type:"):
        base_pass.visit(MagicMock())


def test_copy_literal_expression():
    """Test that the literal expression is correctly copied."""
    expr = LiteralExpression(value=42)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.value == expr.value


def test_copy_identifier_expression():
    """Test that the identifier expression is correctly copied."""
    expr = IdentifierExpression(identifier=Identifier("x"))
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.identifier == expr.identifier


def test_copy_unary_expression():
    """Test that the unary expression is correctly copied."""
    operand = LiteralExpression(value=42)
    expr = UnaryExpression(operation=UnaryOperation.NEGATE, operand=operand)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.operation == expr.operation
    assert copy.operand is not expr.operand
    assert copy.operand.value == expr.operand.value


def test_copy_binary_expression():
    """Test that the binary expression is correctly copied."""
    left = LiteralExpression(value=42)
    right = LiteralExpression(value=24)
    expr = BinaryExpression(operation=BinaryOperation.ADD, left=left, right=right)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.operation == expr.operation
    assert copy.left is not expr.left
    assert copy.left.value == expr.left.value
    assert copy.right is not expr.right
    assert copy.right.value == expr.right.value


@pytest.mark.parametrize(
    "expression, expected_sympy_expression",
    [
        (LiteralExpression(5), sympy.Integer(5)),
        (
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
            -sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            sympy.Integer(5) + sympy.Integer(10),
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression(5)),
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            -sympy.Integer(5) * sympy.Symbol("x_0"),
        ),
    ],
)
def test_convert_expression_to_sympy_expression(
    expression: Expression, expected_sympy_expression: sympy.Expr
):
    """Test that the expression is correctly converted to a sympy expression."""
    result = convert_expression_to_sympy_expression(expression)
    assert result == expected_sympy_expression


def test_substitute_sympy_expression_variables():
    """Test that the sympy expression variables are correctly substituted."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = sympy.Symbol("x_0") + sympy.Symbol("y_1")
    substitutions = {x: 5, y: 10}
    result = substitute_sympy_expression_variables(sympy_expression, substitutions)
    assert result == 15


@pytest.mark.parametrize(
    "sympy_expression, expected_expression",
    [
        (sympy.Integer(5), LiteralExpression("5")),
        (
            -sympy.Integer(5),
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("5")),
        ),
        (
            sympy.Integer(5) - sympy.Symbol("x_0"),
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                LiteralExpression("5"),
                IdentifierExpression(mock_identifier("x", 0)),
            ),
        ),
        (
            -sympy.Integer(5) / sympy.Symbol("y_3"),
            BinaryExpression(
                BinaryOperation.DIVIDE,
                UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("5")),
                IdentifierExpression(mock_identifier("y", 3)),
            ),
        ),
    ],
)
def test_convert_sympy_expression_to_expression(
    sympy_expression: sympy.Expr, expected_expression: Expression
):
    """Test that the sympy expression is correctly converted to an expression."""
    result = convert_sympy_expression_to_expression(sympy_expression)
    _assert_exact_expression_equality(result, expected_expression)


def test_sympy_expression_conversion_fails_when_symbol_is_not_identifier():
    """Test that the sympy expression conversion fails when the symbol is not an
    identifier.
    """
    with pytest.raises(RuntimeError):
        convert_sympy_expression_to_expression(sympy.Symbol("x"))


# TODO: See if this and the next test can combined to just test simplify expr.
@pytest.mark.parametrize(
    "expression, expected_value",
    [
        (LiteralExpression(5), "5"),
        (UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(5)), "5"),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            "15",
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            "50",
        ),
    ],
)
def test_simplify_constant_expression(
    expression: Expression, expected_value: LiteralType
):
    """Test that a constant expression is correctly simplified."""
    result = simplify_expression(expression)
    assert isinstance(result, LiteralExpression)
    assert result.value == expected_value


def test_simplify_variable_expression():
    """Test that variables are correctly substituted and simplified in an
    expression.
    """
    x_1 = Identifier("x")
    x_2 = Identifier("x")
    expr = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x_1),
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(5),
            IdentifierExpression(x_2),
        ),
    )

    result = simplify_expression(expr, {x_1: 10, x_2: 5})

    assert isinstance(result, LiteralExpression)
    assert result.value == "35"
