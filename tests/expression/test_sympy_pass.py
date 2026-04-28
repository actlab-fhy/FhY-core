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

from unittest.mock import Mock

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
from fhy_core.expression.passes.sympy import (
    ExpressionToSympyConverter,
    SymPyToExpressionConverter,
)
from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError

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


@pytest.mark.parametrize(
    "name_hint, identifier_id",
    [
        pytest.param("a_b_c", 5, id="underscores_in_name_hint"),
        pytest.param("nested_name", 0, id="single_underscore_in_name_hint"),
        pytest.param("x1", 0, id="digits_in_name_hint"),
        pytest.param("x_1", 0, id="digit_after_underscore_in_name_hint"),
    ],
)
def test_identifier_round_trips_through_sympy_for_tricky_name_hints(
    name_hint: str, identifier_id: int
) -> None:
    """Test name hints with underscores/digits round-trip through SymPy unchanged.

    The encoding glues ``name_hint + "_" + id`` and the decoder splits at the
    *last* underscore, so name hints that themselves contain underscores or
    end in digits are the interesting boundary cases.
    """
    original = IdentifierExpression(mock_identifier(name_hint, identifier_id))
    sympy_expression = convert_expression_to_sympy_expression(original)
    restored = convert_sympy_expression_to_expression(sympy_expression)
    assert isinstance(restored, IdentifierExpression)
    assert restored.identifier.name_hint == name_hint
    assert restored.identifier.id == identifier_id


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


# =============================================================================
# Defensive dispatch branches and noop output
# =============================================================================


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param("True", sympy.true, id="true_string"),
        pytest.param("False", sympy.false, id="false_string"),
    ],
)
def test_sympy_converter_visit_literal_bool_string_via_mock(
    value: str, expected: sympy.logic.boolalg.BooleanAtom
) -> None:
    """Test the boolean-string branches map to `sympy.true` / `sympy.false`."""
    converter = ExpressionToSympyConverter()
    literal = Mock(spec=LiteralExpression)
    literal.value = value
    assert converter.visit_literal_expression(literal) is expected


def test_sympy_converter_visit_literal_unsupported_value_raises() -> None:
    """Test `visit_literal_expression` raises on a wholly unsupported literal value."""
    converter = ExpressionToSympyConverter()
    literal = Mock(spec=LiteralExpression)
    literal.value = object()  # not int/float/bool/str

    with pytest.raises(TypeError, match=r"Unsupported literal type"):
        converter.visit_literal_expression(literal)


def test_sympy_converter_get_noop_output_raises() -> None:
    """Test `ExpressionToSympyConverter.get_noop_output` raises `PassExecutionError`."""
    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        ExpressionToSympyConverter().get_noop_output(LiteralExpression(0))


def test_sympy_to_expression_convert_rejects_unknown_node_type() -> None:
    """Test `convert` raises `TypeError` for a node that is neither Expr nor Boolean."""
    with pytest.raises(TypeError, match=r"Unsupported node type"):
        SymPyToExpressionConverter().convert(42)


def test_sympy_to_expression_convert_expr_rejects_unsupported_expr_subtype() -> None:
    """Test `convert_expr` raises `TypeError` for an unsupported `sympy.Expr`."""
    x = sympy.Symbol("x_0")
    with pytest.raises(TypeError, match=r"Unsupported expression type"):
        SymPyToExpressionConverter().convert_expr(sympy.exp(x))


def test_sympy_to_expression_convert_bool_rejects_unsupported_boolean_subtype() -> None:
    """Test `convert_bool` raises `TypeError` for an unrecognized boolean subtype."""
    fake = Mock(spec=sympy.logic.boolalg.Boolean)
    with pytest.raises(TypeError, match=r"Unsupported boolean expression type"):
        SymPyToExpressionConverter().convert_bool(fake)


def test_sympy_to_expression_convert_relational_rejects_unsupported_subtype() -> None:
    """Test `convert_relational` raises `TypeError` for an unrecognized relational."""
    fake = Mock(spec=sympy.core.relational.Relational)
    with pytest.raises(TypeError, match=r"Unsupported relational type"):
        SymPyToExpressionConverter().convert_relational(fake)


@pytest.mark.parametrize(
    "method_name, sympy_class, identity_value, sample_arg",
    [
        pytest.param("convert_Add", sympy.Add, 0, 7, id="add"),
        pytest.param("convert_Mul", sympy.Mul, 1, 5, id="mul"),
    ],
)
def test_sympy_to_expression_convert_commutative_op_zero_arg_returns_identity(
    method_name: str,
    sympy_class: type,
    identity_value: int,
    sample_arg: int,  # noqa: ARG001
) -> None:
    """Test `convert_Add`/`convert_Mul` return the identity literal on zero args."""
    fake = Mock(spec=sympy_class)
    fake.args = ()
    result = getattr(SymPyToExpressionConverter(), method_name)(fake)
    assert result.is_structurally_equivalent(LiteralExpression(identity_value))


@pytest.mark.parametrize(
    "method_name, sympy_class, sample_arg",
    [
        pytest.param("convert_Add", sympy.Add, 7, id="add"),
        pytest.param("convert_Mul", sympy.Mul, 5, id="mul"),
    ],
)
def test_sympy_to_expression_convert_commutative_op_one_arg_unwraps(
    method_name: str, sympy_class: type, sample_arg: int
) -> None:
    """Test `convert_Add`/`convert_Mul` unwrap a single-arg node to its argument."""
    fake = Mock(spec=sympy_class)
    fake.args = (sympy.Integer(sample_arg),)
    result = getattr(SymPyToExpressionConverter(), method_name)(fake)
    assert result.is_structurally_equivalent(LiteralExpression(sample_arg))


def test_sympy_to_expression_convert_nor_lowers_to_not_or() -> None:
    """Test the ``Nor`` dispatch path lowers to ``NOT(OR(x, y))``.

    SymPy normalizes ``sympy.Nor(x, y)`` (default ``evaluate=True``) to a
    ``Not(Or(...))`` node, so the only way to reach ``convert_Nor`` is via a
    node that declares itself as ``Nor``. We simulate that with a
    ``Mock(spec=Nor)`` whose ``.func`` resolves to ``sympy.Or`` during the
    internal recursive rebuild -- that keeps the converter's control flow
    realistic while actually exercising ``convert_Nor``'s body.
    """
    fake_nor = Mock(spec=sympy.logic.boolalg.Nor)
    fake_nor.args = (sympy.Symbol("x_0"), sympy.Symbol("y_1"))
    fake_nor.func = sympy.Or

    result = convert_sympy_expression_to_expression(fake_nor)

    expected = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        BinaryExpression(
            BinaryOperation.LOGICAL_OR,
            IdentifierExpression(mock_identifier("x", 0)),
            IdentifierExpression(mock_identifier("y", 1)),
        ),
    )
    assert result.is_structurally_equivalent(expected)


def test_sympy_to_expression_convert_nand_lowers_to_not_and() -> None:
    """Test the ``Nand`` dispatch path lowers to ``NOT(AND(x, y))``.

    As with ``Nor``, SymPy normalizes ``sympy.Nand`` to ``Not(And(...))`` by
    default, so ``convert_Nand`` is only reachable via a mock declaring the
    ``Nand`` spec. The ``.func`` override keeps the recursive rebuild intact
    by routing it through ``sympy.And``.
    """
    fake_nand = Mock(spec=sympy.logic.boolalg.Nand)
    fake_nand.args = (sympy.Symbol("x_0"), sympy.Symbol("y_1"))
    fake_nand.func = sympy.And

    result = convert_sympy_expression_to_expression(fake_nand)

    expected = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        BinaryExpression(
            BinaryOperation.LOGICAL_AND,
            IdentifierExpression(mock_identifier("x", 0)),
            IdentifierExpression(mock_identifier("y", 1)),
        ),
    )
    assert result.is_structurally_equivalent(expected)


def test_sympy_to_expression_convert_implies_raises_not_implemented() -> None:
    """Test `convert_Implies` raises `NotImplementedError`."""
    implies = sympy.Implies(sympy.Symbol("x_0"), sympy.Symbol("y_1"))

    with pytest.raises(NotImplementedError, match=r"Implies is not supported"):
        convert_sympy_expression_to_expression(implies)


def test_sympy_two_argument_helper_rejects_wrong_arg_count() -> None:
    """Test the two-argument binary helper rejects nodes with wrong arity."""
    fake = Mock(spec=sympy.Pow)
    fake.args = (sympy.Integer(1), sympy.Integer(2), sympy.Integer(3))

    with pytest.raises(
        ValueError, match=r"Expected a binary operation to have exactly two arguments"
    ):
        SymPyToExpressionConverter().convert_Pow(fake)
