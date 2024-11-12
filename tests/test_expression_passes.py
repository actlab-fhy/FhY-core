"""Tests the expression tree analysis and transformation passes."""

from unittest.mock import MagicMock

import pytest
import sympy
import z3
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    SymbolType,
    UnaryExpression,
    UnaryOperation,
    collect_identifiers,
    convert_expression_to_sympy_expression,
    convert_expression_to_z3_expression,
    convert_sympy_expression_to_expression,
    copy_expression,
    is_satisfiable,
    pformat_expression,
    replace_identifiers,
    simplify_expression,
    substitute_identifiers,
    substitute_sympy_expression_variables,
)
from fhy_core.expression.core import LiteralType
from fhy_core.expression.visitor import (
    ExpressionBasePass,
)
from fhy_core.identifier import Identifier

from .utils import assert_exact_expression_equality, mock_identifier


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
    result = pformat_expression(expression, show_id=True)
    assert identifier.name_hint in result
    assert str(identifier.id) in result


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
    expr = LiteralExpression(42)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.value == expr.value


def test_copy_identifier_expression():
    """Test that the identifier expression is correctly copied."""
    expr = IdentifierExpression(Identifier("x"))
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.identifier == expr.identifier


def test_copy_unary_expression():
    """Test that the unary expression is correctly copied."""
    operand = LiteralExpression(42)
    expr = UnaryExpression(UnaryOperation.NEGATE, operand)
    copy = copy_expression(expr)
    assert copy is not expr
    assert isinstance(copy, UnaryExpression)
    assert copy.operation == expr.operation
    assert copy.operand is not expr.operand
    assert copy.operand.value == operand.value


def test_copy_binary_expression():
    """Test that the binary expression is correctly copied."""
    left = LiteralExpression(2)
    right = LiteralExpression(24)
    expr = BinaryExpression(BinaryOperation.ADD, left, right)
    copy = copy_expression(expr)
    assert copy is not expr
    assert copy.operation == expr.operation
    assert copy.left is not left
    assert copy.left.value == left.value
    assert copy.right is not right
    assert copy.right.value == right.value


def test_substitute_identifiers():
    """Test that the identifiers are correctly substituted in an expression."""
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
    substitutions = {x: LiteralExpression(10), y: LiteralExpression(5)}
    result = substitute_identifiers(expr, substitutions)
    assert_exact_expression_equality(
        result,
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(10),
            BinaryExpression(
                BinaryOperation.DIVIDE,
                LiteralExpression(5),
                LiteralExpression(5),
            ),
        ),
    )


def test_replace_identifiers():
    """Test that the identifiers are correctly replaced in an expression."""
    x = Identifier("x")
    y = Identifier("y")
    expr = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        LiteralExpression(5),
    )
    replacements = {x: y}
    result = replace_identifiers(expr, replacements)
    assert_exact_expression_equality(
        result,
        BinaryExpression(
            BinaryOperation.ADD,
            IdentifierExpression(y),
            LiteralExpression(5),
        ),
    )


@pytest.mark.parametrize(
    "expression, expected_sympy_expression",
    [
        (LiteralExpression(5), sympy.Integer(5)),
        (LiteralExpression(5.5), sympy.Float(5.5)),
        (LiteralExpression(True), sympy.true),
        (LiteralExpression(False), sympy.false),
        (LiteralExpression("10.6"), sympy.Float(10.6)),
        (
            UnaryExpression(
                UnaryOperation.POSITIVE, IdentifierExpression(mock_identifier("x", 0))
            ),
            sympy.Symbol("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("x", 0))
            ),
            -sympy.Symbol("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            not sympy.Symbol("x_0"),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") + sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") - sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") * sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") / sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") // sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.MODULO,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") % sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") ** sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            sympy.Symbol("x_0") & sympy.Symbol("y_1"),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            sympy.Symbol("x_0") | sympy.Symbol("y_1"),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Eq(sympy.Symbol("x_0"), sympy.Integer(5)),
        ),
        (
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Ne(sympy.Symbol("x_0"), sympy.Integer(5)),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") < sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") <= sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") > sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            sympy.Symbol("x_0") >= sympy.Integer(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.MODULO,
                    IdentifierExpression(mock_identifier("x", 0)),
                    LiteralExpression(5),
                ),
                LiteralExpression(0),
            ),
            sympy.Eq(sympy.Symbol("x_0") % sympy.Integer(5), sympy.Integer(0)),
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
    substitutions = {x: LiteralExpression(5), y: LiteralExpression(10)}
    result = substitute_sympy_expression_variables(sympy_expression, substitutions)
    assert result == 15


@pytest.mark.parametrize(
    "sympy_expression, expected_expression",
    [
        (sympy.Integer(5), LiteralExpression(5)),
        (sympy.Float(5.5), LiteralExpression(5.5)),
        (sympy.true, LiteralExpression(True)),
        (sympy.false, LiteralExpression(False)),
        (sympy.Symbol("x_0"), IdentifierExpression(mock_identifier("x", 0))),
        (
            -sympy.Symbol("x_0"),
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(-1),
                IdentifierExpression(mock_identifier("x", 0)),
            ),
        ),
        (
            ~sympy.Symbol("x_0"),
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
        ),
        (
            sympy.Add(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Mul(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Pow(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Mod(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.MODULO,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.And(sympy.Symbol("x_0"), sympy.Symbol("y_1"), evaluate=False),
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
        ),
        (
            sympy.Or(sympy.Symbol("x_0"), sympy.Symbol("y_1"), evaluate=False),
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
        ),
        (
            sympy.Xor(sympy.Symbol("x_0"), sympy.Symbol("y_1"), evaluate=False),
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                BinaryExpression(
                    BinaryOperation.LOGICAL_OR,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("y", 1)),
                ),
                UnaryExpression(
                    UnaryOperation.LOGICAL_NOT,
                    BinaryExpression(
                        BinaryOperation.LOGICAL_AND,
                        IdentifierExpression(mock_identifier("x", 0)),
                        IdentifierExpression(mock_identifier("y", 1)),
                    ),
                ),
            ),
        ),
        (
            sympy.Eq(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Ne(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Lt(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Le(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Gt(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
        (
            sympy.Ge(sympy.Symbol("x_0"), sympy.Integer(5), evaluate=False),
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
        ),
    ],
)
def test_convert_sympy_expression_to_expression(
    sympy_expression: sympy.Expr, expected_expression: Expression
):
    """Test that the sympy expression is correctly converted to an expression."""
    result = convert_sympy_expression_to_expression(sympy_expression)
    assert_exact_expression_equality(result, expected_expression)


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
        (LiteralExpression(5), 5),
        (UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(5)), 5),
        (
            BinaryExpression(
                BinaryOperation.ADD, LiteralExpression(5), LiteralExpression(10)
            ),
            15,
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                UnaryExpression(UnaryOperation.POSITIVE, LiteralExpression(5)),
                LiteralExpression(10),
            ),
            50,
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.MODULO, LiteralExpression(15), LiteralExpression(5)
                ),
                LiteralExpression(0),
            ),
            True,
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

    result = simplify_expression(
        expr, {x_1: LiteralExpression(10), x_2: LiteralExpression(5)}
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == 35


@pytest.mark.parametrize(
    "expression, symbol_types, expected_z3_expression",
    [
        (LiteralExpression(5), {}, z3.IntVal(5)),
        (LiteralExpression(5.5), {}, z3.RealVal(5.5)),
        (LiteralExpression(True), {}, z3.BoolVal(True)),
        (LiteralExpression(False), {}, z3.BoolVal(False)),
        (LiteralExpression("10.6"), {}, z3.RealVal(10.6)),
        (
            UnaryExpression(
                UnaryOperation.POSITIVE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE, IdentifierExpression(mock_identifier("x", 0))
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            -z3.Real("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.LOGICAL_NOT,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            {mock_identifier("x", 0): SymbolType.BOOL},
            z3.Not(z3.Bool("x_0")),
        ),
        (
            BinaryExpression(
                BinaryOperation.ADD,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") + z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") - z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") * z3.RealVal(5.5),
        ),
        (
            BinaryExpression(
                BinaryOperation.DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5.5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") / z3.RealVal(5.5),
        ),
        (
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") / z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.FLOOR_DIVIDE,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.ToInt(z3.Real("x_0") / z3.IntVal(5)),
        ),
        (
            BinaryExpression(
                BinaryOperation.MODULO,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") % z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.POWER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") ** z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            {
                mock_identifier("x", 0): SymbolType.BOOL,
                mock_identifier("y", 1): SymbolType.BOOL,
            },
            z3.And(z3.Bool("x_0"), z3.Bool("y_1")),
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_OR,
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            {
                mock_identifier("x", 0): SymbolType.BOOL,
                mock_identifier("y", 1): SymbolType.BOOL,
            },
            z3.Or(z3.Bool("x_0"), z3.Bool("y_1")),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") == z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.REAL},
            z3.Real("x_0") != z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") < z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.LESS_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") <= z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") > z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.GREATER_EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") >= z3.IntVal(5),
        ),
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.MODULO,
                    IdentifierExpression(mock_identifier("x", 0)),
                    LiteralExpression(5),
                ),
                LiteralExpression(0),
            ),
            {mock_identifier("x", 0): SymbolType.INT},
            z3.Int("x_0") % z3.IntVal(5) == z3.IntVal(0),
        ),
    ],
)
def test_convert_expression_to_z3_expression(
    expression: Expression,
    symbol_types: dict[Identifier, SymbolType],
    expected_z3_expression: z3.ExprRef,
):
    """Test that the expression is correctly converted to a Z3 expression."""
    result, identifier_to_z3_symbol = convert_expression_to_z3_expression(
        expression, symbol_types
    )
    assert result == expected_z3_expression
    # TODO: Check correctness of identifier_to_z3_symbol


@pytest.mark.parametrize(
    "expression, considered_identifiers, symbol_types, expected_output",
    [
        (
            BinaryExpression(
                BinaryOperation.EQUAL,
                IdentifierExpression(mock_identifier("x", 0)),
                LiteralExpression(5),
            ),
            {mock_identifier("x", 0)},
            {mock_identifier("x", 0): SymbolType.INT},
            True,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
                BinaryExpression(
                    BinaryOperation.GREATER,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
            ),
            {mock_identifier("x", 0)},
            {
                mock_identifier("x", 0): SymbolType.INT,
                mock_identifier("N", 3): SymbolType.INT,
            },
            False,
        ),
        (
            BinaryExpression(
                BinaryOperation.LOGICAL_AND,
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    IdentifierExpression(mock_identifier("N", 3)),
                ),
                BinaryExpression(
                    BinaryOperation.LESS,
                    IdentifierExpression(mock_identifier("x", 0)),
                    BinaryExpression(
                        BinaryOperation.SUBTRACT,
                        IdentifierExpression(mock_identifier("N", 3)),
                        LiteralExpression(1),
                    ),
                ),
            ),
            {mock_identifier("x", 0)},
            {
                mock_identifier("x", 0): SymbolType.INT,
                mock_identifier("N", 3): SymbolType.INT,
            },
            True,
        ),
    ],
)
def test_find_satisfying_assignment(
    expression: Expression,
    considered_identifiers: set[Identifier],
    symbol_types: dict[Identifier, SymbolType],
    expected_output: dict[Identifier, LiteralType] | None,
):
    """Test that a satisfying assignment is correctly found for an expression."""
    result = is_satisfiable(considered_identifiers, expression, symbol_types)
    assert result == expected_output
