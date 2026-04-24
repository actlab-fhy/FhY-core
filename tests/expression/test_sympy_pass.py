"""Tests for `fhy_core.expression.passes.sympy`.

Notes on known-equivalent mutants not targeted by this file:

- The ``evaluate=False`` at :mod:`fhy_core.expression.passes.sympy` lines 388
  (``convert_Xor``) and line 535 for ``And``/``Or`` flavours is observably
  indistinguishable from ``evaluate=True`` when the recursive tail contains
  only symbolic arguments - SymPy yields identical node structure in both
  cases. Only the ``Add``/``Mul`` flavours at line 535 are killable here
  (arithmetic folding at the tail changes the tree shape).
- The empty-args / single-arg guards in ``convert_Add`` and ``convert_Mul``
  are unreachable through any public SymPy constructor and are therefore not
  targeted.
- ``Not.args[0]`` vs ``args[-1]`` is equivalent because SymPy's ``Not`` is
  always unary.
- The ``value == "True"`` / ``value == "False"`` string branches in
  ``visit_literal_expression`` are unreachable: ``LiteralExpression``'s
  ``__post_init__`` rejects any non-numeric string at construction. All
  comparison-operator mutants on those branches are therefore unkillable
  from the public surface.

"""

import pytest
import sympy  # type: ignore[import-untyped]

from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    UnaryExpression,
    UnaryOperation,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    simplify_expression,
    substitute_sympy_expression_variables,
)
from fhy_core.expression.core import LiteralType
from fhy_core.identifier import Identifier

from .conftest import mock_identifier

# =============================================================================
# Expression -> SymPy
# =============================================================================


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
                UnaryOperation.POSITIVE,
                IdentifierExpression(mock_identifier("x", 0)),
            ),
            sympy.Symbol("x_0"),
        ),
        (
            UnaryExpression(
                UnaryOperation.NEGATE,
                IdentifierExpression(mock_identifier("x", 0)),
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
) -> None:
    """Test `convert_expression_to_sympy_expression` maps each expression correctly."""
    assert (
        convert_expression_to_sympy_expression(expression) == expected_sympy_expression
    )


# =============================================================================
# substitute_sympy_expression_variables
# =============================================================================


def test_substitute_sympy_expression_variables_folds_to_concrete_value() -> None:
    """Test substituting every free variable produces a fully-folded scalar."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = sympy.Symbol("x_0") + sympy.Symbol("y_1")
    substitutions: dict[Identifier, Expression] = {
        x: LiteralExpression(5),
        y: LiteralExpression(10),
    }
    assert substitute_sympy_expression_variables(sympy_expression, substitutions) == 15


@pytest.mark.parametrize("value", [True, False])
def test_substitute_sympy_variables_on_raw_bool_is_identity(value: bool) -> None:
    """Test `substitute_sympy_expression_variables` short-circuits raw Python bools."""
    assert substitute_sympy_expression_variables(value, {}) is value


# =============================================================================
# SymPy -> Expression
# =============================================================================


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
) -> None:
    """Test `convert_sympy_expression_to_expression` lowers each node correctly."""
    result = convert_sympy_expression_to_expression(sympy_expression)
    assert result.is_structurally_equivalent(expected_expression)


def test_convert_sympy_expression_fails_when_symbol_has_no_underscore_suffix() -> None:
    """Test `convert_sympy_expression_to_expression` raises on unsuffixed symbols."""
    with pytest.raises(RuntimeError):
        convert_sympy_expression_to_expression(
            sympy.Symbol("no_trailing_id_suffix".replace("_", ""))
        )


@pytest.mark.parametrize(
    "symbol_name, expected_name_hint, expected_id",
    [
        ("x_42", "x", 42),
        ("xy_42", "xy", 42),
        ("a_10", "a", 10),
        ("long_name_7", "long_name", 7),
    ],
)
def test_convert_symbol_recovers_name_hint_and_suffix_as_id(
    symbol_name: str, expected_name_hint: str, expected_id: int
) -> None:
    """Test `convert_Symbol` splits the symbol name at the last underscore."""
    result = convert_sympy_expression_to_expression(sympy.Symbol(symbol_name))
    assert isinstance(result, IdentifierExpression)
    assert result.identifier.name_hint == expected_name_hint
    assert result.identifier.id == expected_id


def test_convert_add_of_literals_and_symbol_preserves_unevaluated_tail() -> None:
    """Test `convert_Add` preserves the multi-arg shape without folding the tail."""
    source = sympy.Add(
        sympy.Integer(1),
        sympy.Integer(2),
        sympy.Integer(3),
        sympy.Symbol("x_0"),
        evaluate=False,
    )

    result = convert_sympy_expression_to_expression(source)

    expected = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(1),
        BinaryExpression(
            BinaryOperation.ADD,
            LiteralExpression(2),
            BinaryExpression(
                BinaryOperation.ADD,
                LiteralExpression(3),
                IdentifierExpression(mock_identifier("x", 0)),
            ),
        ),
    )
    assert result.is_structurally_equivalent(expected)


def test_convert_mul_of_literals_and_symbol_preserves_unevaluated_tail() -> None:
    """Test `convert_Mul` preserves the multi-arg shape without folding the tail."""
    source = sympy.Mul(
        sympy.Integer(2),
        sympy.Integer(3),
        sympy.Integer(5),
        sympy.Symbol("x_0"),
        evaluate=False,
    )

    result = convert_sympy_expression_to_expression(source)

    expected = BinaryExpression(
        BinaryOperation.MULTIPLY,
        LiteralExpression(2),
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(3),
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                LiteralExpression(5),
                IdentifierExpression(mock_identifier("x", 0)),
            ),
        ),
    )
    assert result.is_structurally_equivalent(expected)


# =============================================================================
# simplify_expression
# =============================================================================


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
) -> None:
    """Test `simplify_expression` folds a constant expression to its scalar value."""
    result = simplify_expression(expression)
    assert isinstance(result, LiteralExpression)
    assert result.value == expected_value


def test_simplify_variable_expression_with_environment_folds_to_scalar() -> None:
    """Test `simplify_expression` with an environment folds identifiers plus ops."""
    x_1 = Identifier("x")
    x_2 = Identifier("x")
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x_1),
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression(5),
            IdentifierExpression(x_2),
        ),
    )

    result = simplify_expression(
        expression, {x_1: LiteralExpression(10), x_2: LiteralExpression(5)}
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == 35
