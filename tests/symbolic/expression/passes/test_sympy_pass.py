"""Tests for `fhy_core.symbolic.expression.passes.sympy`."""

import logging
import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Any, cast
from unittest.mock import Mock

import pytest
import sympy  # type: ignore[import-untyped]
from immutabledict import immutabledict
from sympy.core import random as sympy_random  # type: ignore[import-untyped]
from sympy.core.cache import clear_cache  # type: ignore[import-untyped]
from sympy.core.evalf import PrecisionExhausted  # type: ignore[import-untyped]

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import PassExecutionError
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    ComplexInfinityLiftError,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NativeConstantBindingError,
    NonBooleanLogicalOperandError,
    PartialPiecewiseError,
    PiecewiseExpression,
    UnaryExpression,
    UnaryOperation,
    call,
    convert_expression_to_sympy_expression,
    convert_sympy_expression_to_expression,
    evaluate_expression_with_numpy,
    get_native_constant_identifier,
    inline_functions,
    logical_and,
    logical_not,
    logical_or,
    piecewise,
    substitute_sympy_expression_variables,
)
from fhy_core.symbolic.expression.core import LiteralType
from fhy_core.symbolic.expression.passes.sympy import (
    _NATIVE_CONSTANT_LIFT,
    _NATIVE_CONSTANT_LOWER,
    _NATIVE_FUNCTION_LOWER,
    ExpressionToSympyConverter,
    SymPyToExpressionConverter,
    SympyVariableSubstitutionPass,
)
from fhy_core.symbolic.expression.passes.sympy import (
    simplify_expression as sympy_simplify_expression,
)
from fhy_core.symbolic.solver import simplify_expression

from ..conftest import mock_identifier

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
        (LiteralExpression("10.6"), sympy.Rational("10.6")),
        (LiteralExpression("5"), sympy.Integer(5)),
        (LiteralExpression("05"), sympy.Integer(5)),
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
            sympy.Not(sympy.Symbol("x_0")),
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


def test_substitute_sympy_expression_variables_is_simultaneous_not_chained() -> None:
    """Test chained bindings substitute simultaneously rather than sequentially.

    A sequential ``dict``-based ``.subs`` would chain ``x_0 -> y_1 -> 5``,
    collapsing ``x_0 < 5`` to the literal ``False`` (``5 < 5``).
    Simultaneous substitution must instead leave the residual ``y_1 < 5``,
    since ``y_1``'s replacement value is never itself re-substituted by
    the ``y_1: 5`` binding.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = sympy.Symbol("x_0") < 5
    substitutions: dict[Identifier, Expression] = {
        x: IdentifierExpression(y),
        y: LiteralExpression(5),
    }

    result = substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert result == (sympy.Symbol("y_1") < 5)


def test_substitute_sympy_variables_replaces_piecewise_condition_symbol() -> None:
    """Test substituting a ``Piecewise`` condition symbol with a relational.

    The substitution must succeed and preserve the branch structure.
    """
    flag = mock_identifier("flag", 0)
    y = mock_identifier("y", 1)
    sympy_expression = sympy.Piecewise(
        (sympy.Integer(1), sympy.Symbol("flag_0")), (sympy.Integer(2), True)
    )
    substitutions: dict[Identifier, Expression] = {
        flag: IdentifierExpression(y) > 0,
    }

    result = substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert isinstance(result, sympy.Piecewise)
    assert result.args[0] == (sympy.Integer(1), sympy.Symbol("y_1") > 0)
    assert result.args[1] == (sympy.Integer(2), True)


def test_substitute_sympy_expression_variables_raises_for_a_non_real_comparison() -> (
    None
):
    """Test a substitution reducing a comparison to `zoo` raises `PassExecutionError`.

    Substituting a zero divisor rebuilds the comparison against SymPy's
    complex infinity, which auto-evaluation cannot compare and raises a
    raw `TypeError`; that has to surface as `PassExecutionError` rather
    than escaping the bridge unwrapped.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = (sympy.Symbol("x_0") / sympy.Symbol("y_1")) > 1
    substitutions: dict[Identifier, Expression] = {
        x: LiteralExpression(1),
        y: LiteralExpression(0),
    }

    with pytest.raises(PassExecutionError) as exc_info:
        substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_substitute_sympy_expression_variables_raises_for_a_nan_comparison() -> None:
    """Test a NaN substitution into a strict comparison raises the same error."""
    x = mock_identifier("x", 0)
    sympy_expression = sympy.Symbol("x_0") < 5
    substitutions: dict[Identifier, Expression] = {x: LiteralExpression(math.nan)}

    with pytest.raises(PassExecutionError) as exc_info:
        substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_substitute_sympy_variables_refuses_a_referenced_constant_binding() -> None:
    """Test a binding for a native constant's symbol free in the tree is refused."""
    pi = get_native_constant_identifier("pi")
    pi_symbol = sympy.Symbol(ExpressionToSympyConverter.format_identifier(pi))
    sympy_expression = pi_symbol + 1
    substitutions: dict[Identifier, Expression] = {pi: LiteralExpression(3)}

    with pytest.raises(NativeConstantBindingError, match="pi"):
        substitute_sympy_expression_variables(sympy_expression, substitutions)


def test_substitute_sympy_variables_ignores_an_unreferenced_constant_binding() -> None:
    """Test a native-constant binding absent from the sympy expression is ignored."""
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    sympy_expression = sympy.Symbol("x_0") + 1
    substitutions: dict[Identifier, Expression] = {
        x: LiteralExpression(2),
        pi: LiteralExpression(3),
    }

    result = substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert result == 3


def test_substitute_sympy_variables_still_decides_a_well_defined_comparison() -> None:
    """Test a substitution with a nonzero divisor still decides the comparison.

    Confirms the `PassExecutionError` wrapping above is specific to the
    unrepresentable shapes, not a regression that makes every comparison
    substitution fail.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = (sympy.Symbol("x_0") / sympy.Symbol("y_1")) > 1
    substitutions: dict[Identifier, Expression] = {
        x: LiteralExpression(4),
        y: LiteralExpression(2),
    }

    result = substitute_sympy_expression_variables(sympy_expression, substitutions)

    assert result is sympy.true


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


def test_convert_symbol_advances_identifier_id_counter_past_recovered_id() -> None:
    """Test recovering an `Identifier` from a SymPy symbol advances `_next_id`.

    `Identifier` documents that ids are never reused. The reverse converter
    must therefore advance the global counter past any id it materializes,
    so a subsequently constructed `Identifier` gets a strictly greater id.

    The starting counter is captured up front and restored on teardown so
    this test does not leak large id values into other tests in the same
    pytest worker.
    """
    starting_next_id = Identifier._next_id
    recovered_id = starting_next_id + 100
    try:
        recovered = convert_sympy_expression_to_expression(
            sympy.Symbol(f"x_{recovered_id}")
        )
        assert isinstance(recovered, IdentifierExpression)
        assert recovered.identifier.id == recovered_id

        fresh = Identifier("y")
        assert fresh.id > recovered_id
    finally:
        Identifier._next_id = starting_next_id


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


@pytest.mark.parametrize(
    "string_value",
    [
        pytest.param("5", id="single_digit"),
        pytest.param("05", id="leading_zero"),
        pytest.param("42", id="multi_digit"),
    ],
)
def test_simplify_preserves_integer_bucket_for_int_grammar_string(
    string_value: str,
) -> None:
    """Test ``simplify_expression`` preserves the integer bucket for str-int input.

    Integer-grammar string literals lower to ``sympy.Integer`` (not
    ``sympy.Float``) and lift back as Python ``int`` on the way out, so
    ``LiteralExpression("5")`` and ``LiteralExpression(5)`` remain
    bucket-equivalent across the SymPy round trip.
    """
    result = simplify_expression(LiteralExpression(string_value))

    assert isinstance(result, LiteralExpression)
    assert type(result.value) is int
    assert result.is_structurally_equivalent(LiteralExpression(string_value))


def test_simplify_variable_expression_with_environment_folds_to_scalar() -> None:
    """Test `simplify_expression` with an environment folds identifiers plus ops."""
    x_1 = mock_identifier("x", 0)
    x_2 = mock_identifier("x", 1)
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


def test_simplify_expression_chained_bindings_leave_a_residual_not_a_literal() -> None:
    """Test chained bindings `{x: y, y: 3}` on `x + y` yield the residual `y + 3`.

    Sequential substitution would chain ``x -> y -> 3``, folding the sum to
    the literal ``6``. Simultaneous substitution must instead leave ``y``
    unresolved past its own binding, yielding the residual ``y + 3``.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.ADD,
        IdentifierExpression(x),
        IdentifierExpression(y),
    )

    result = simplify_expression(
        expression, {x: IdentifierExpression(y), y: LiteralExpression(3)}
    )

    expected = BinaryExpression(
        BinaryOperation.ADD,
        LiteralExpression(3),
        IdentifierExpression(y),
    )
    assert result.is_structurally_equivalent(expected)


def test_simplify_expression_chained_bindings_on_inequality_leave_a_residual() -> None:
    """Test chained bindings `{x: y, y: 5}` on `x < 5` leave the residual `y < 5`.

    Sequential substitution would chain ``x -> y -> 5``, folding the
    comparison to the literal ``False`` (``5 < 5``). Simultaneous
    substitution must instead leave the residual ``y < 5``.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(x),
        LiteralExpression(5),
    )

    result = simplify_expression(
        expression, {x: IdentifierExpression(y), y: LiteralExpression(5)}
    )

    expected = BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(y),
        LiteralExpression(5),
    )
    assert result.is_structurally_equivalent(expected)


def test_simplify_expression_swap_bindings_on_equality_stays_undecided() -> None:
    """Test swap bindings `{x: y, y: x}` on `x - y == 0` do not fold to a literal.

    Sequential substitution would chain the first binding's replacement
    through the second, resolving `x - y == 0` to `y - y == 0` and folding
    it to the literal `True`. Simultaneous substitution instead swaps the
    identifiers, leaving an undecided residual comparing two distinct
    identifiers.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(
            BinaryOperation.SUBTRACT, IdentifierExpression(x), IdentifierExpression(y)
        ),
        LiteralExpression(0),
    )

    result = simplify_expression(
        expression, {x: IdentifierExpression(y), y: IdentifierExpression(x)}
    )

    assert not isinstance(result, LiteralExpression)


def test_simplify_expression_swap_bindings_on_inequality_stays_undecided() -> None:
    """Test swap bindings `{x: y, y: x}` on `x < y` do not fold to a literal.

    Sequential substitution would chain the first binding's replacement
    through the second, resolving `x < y` to `y < y` and folding it to the
    literal `False`. Simultaneous substitution instead swaps the
    identifiers, leaving an undecided residual comparing two distinct
    identifiers.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(x),
        IdentifierExpression(y),
    )

    result = simplify_expression(
        expression, {x: IdentifierExpression(y), y: IdentifierExpression(x)}
    )

    assert not isinstance(result, LiteralExpression)


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_simplify_expression_bool_binding_value_avoids_sympy_deprecation() -> None:
    """Test a bare `bool` binding value does not hit SymPy's simultaneous-subs warning.

    SymPy's own ``.subs(..., simultaneous=True)`` masks every replacement
    behind a synthetic ``Dummy() * Dummy()`` product before unmasking it
    with a final ``xreplace``. When the replacement is a SymPy ``Boolean``
    (here, ``sympy.true`` from a ``LiteralExpression(True)`` binding),
    that unmasking step reconstructs a ``Mul`` with a non-``Expr``
    argument, which SymPy has deprecated since 1.7
    (``SymPyDeprecationWarning``, escalated to an error by this test's
    marker). Without the ``xreplace``-based fix in
    ``substitute_sympy_expression_variables``, this test fails on that
    warning.
    """
    x = mock_identifier("x", 0)
    expression = IdentifierExpression(x)

    result = simplify_expression(expression, {x: LiteralExpression(True)})

    assert isinstance(result, LiteralExpression)
    assert result.value is True


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_simplify_expression_boolean_expression_binding_avoids_sympy_deprecation() -> (
    None
):
    """Test a boolean-*valued expression* binding does not raise or warn.

    Binding ``x`` to a comparison (``y > 0``, a SymPy ``Relational``,
    itself a ``Boolean``) hits the same masking path as a bare ``bool``
    binding value, but harder: SymPy's simultaneous ``.subs`` does not
    merely warn here, it raises ``TypeError: Relational cannot be used in
    Mul`` outright, because the mask-unmask trick literally tries to
    build ``Mul(<Relational>, 1)``. Without the ``xreplace``-based fix,
    this test fails with that ``TypeError`` regardless of any warnings
    filter.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = IdentifierExpression(x)
    binding = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(y), LiteralExpression(0)
    )

    result = simplify_expression(expression, {x: binding})

    expected = BinaryExpression(
        BinaryOperation.GREATER, IdentifierExpression(y), LiteralExpression(0)
    )
    assert result.is_structurally_equivalent(expected)


def test_simplify_expression_raises_for_a_zero_divisor_under_a_comparison() -> None:
    """Test a zero-divisor binding under a comparison raises `PassExecutionError`.

    Substituting ``y = 0`` rebuilds ``(x / y) > 1`` against SymPy's
    complex infinity, which raises a raw `TypeError` from inside SymPy's
    relational auto-evaluation; the bridge has to surface that as
    `PassExecutionError` rather than let it escape unwrapped.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.GREATER,
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(y)
        ),
        LiteralExpression(1),
    )

    with pytest.raises(PassExecutionError) as exc_info:
        simplify_expression(
            expression, {x: LiteralExpression(1), y: LiteralExpression(0)}
        )

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_simplify_expression_raises_when_a_nan_binding_reaches_a_comparison() -> None:
    """Test a NaN binding under a strict comparison raises `PassExecutionError`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.LESS, IdentifierExpression(x), LiteralExpression(5)
    )

    with pytest.raises(PassExecutionError) as exc_info:
        simplify_expression(expression, {x: LiteralExpression(math.nan)})

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_simplify_expression_still_decides_a_well_defined_divided_comparison() -> None:
    """Test a nonzero-divisor binding still decides the comparison as a literal.

    Confirms the `PassExecutionError` wrapping above is specific to the
    unrepresentable shapes, not a regression over ordinary division.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.GREATER,
        BinaryExpression(
            BinaryOperation.DIVIDE, IdentifierExpression(x), IdentifierExpression(y)
        ),
        LiteralExpression(1),
    )

    result = simplify_expression(
        expression, {x: LiteralExpression(4), y: LiteralExpression(2)}
    )

    assert result.is_structurally_equivalent(LiteralExpression(True))


# =============================================================================
# simplify_expression keeps its input when SymPy exhausts precision
# =============================================================================


@pytest.fixture
def restore_sympy_random_state() -> Iterator[None]:
    """Restore SymPy's global random state after a test reseeds it."""
    state = sympy_random.rng.getstate()
    yield
    sympy_random.rng.setstate(state)


def test_simplify_expression_keeps_the_substituted_form_when_sympy_exhausts_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test SymPy exhausting precision leaves the substituted form unsimplified.

    Simplification is best-effort: when ``sympy.simplify`` raises
    ``PrecisionExhausted``, the result is the lowering with ``environment``
    substituted, lifted back unsimplified. Unpatched, ``(x - y) * (x + y)
    == x * x - 9`` with ``y = 3`` simplifies to the literal ``True``.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            BinaryExpression(
                BinaryOperation.SUBTRACT,
                IdentifierExpression(x),
                IdentifierExpression(y),
            ),
            BinaryExpression(
                BinaryOperation.ADD, IdentifierExpression(x), IdentifierExpression(y)
            ),
        ),
        BinaryExpression(
            BinaryOperation.SUBTRACT,
            BinaryExpression(
                BinaryOperation.MULTIPLY,
                IdentifierExpression(x),
                IdentifierExpression(x),
            ),
            LiteralExpression(9),
        ),
    )
    environment = {y: LiteralExpression(3)}
    assert simplify_expression(expression, environment).is_structurally_equivalent(
        LiteralExpression(True)
    )
    unsimplified = convert_sympy_expression_to_expression(
        substitute_sympy_expression_variables(
            convert_expression_to_sympy_expression(expression), environment
        )
    )

    def raise_precision_exhausted(*args: Any, **kwargs: Any) -> Any:
        raise PrecisionExhausted

    monkeypatch.setattr(sympy, "simplify", raise_precision_exhausted)
    result = simplify_expression(expression, environment)

    assert result.is_structurally_equivalent(unsimplified)
    assert not isinstance(result, LiteralExpression)


def test_simplify_expression_propagates_other_arithmetic_errors_from_sympy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test only ``PrecisionExhausted`` falls back; other SymPy errors propagate.

    ``ZeroDivisionError`` shares ``PrecisionExhausted``'s ``ArithmeticError``
    base, so it guards against catching the base class.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.LESS, IdentifierExpression(x), LiteralExpression(5)
    )

    def raise_zero_division(*args: Any, **kwargs: Any) -> Any:
        raise ZeroDivisionError

    monkeypatch.setattr(sympy, "simplify", raise_zero_division)

    with pytest.raises(ZeroDivisionError):
        simplify_expression(expression)


@pytest.mark.usefixtures("restore_sympy_random_state")
@pytest.mark.parametrize(
    "build_expression",
    [
        pytest.param(
            lambda x, y: BinaryExpression(
                BinaryOperation.EQUAL,
                BinaryExpression(
                    BinaryOperation.FLOOR_DIVIDE,
                    call(
                        "floor",
                        BinaryExpression(
                            BinaryOperation.ADD,
                            IdentifierExpression(x),
                            LiteralExpression(1),
                        ),
                    ),
                    LiteralExpression(-8),
                ),
                LiteralExpression(0),
            ),
            id="floor_divide_of_floor",
        ),
        pytest.param(
            lambda x, y: BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                BinaryExpression(
                    BinaryOperation.FLOOR_DIVIDE,
                    BinaryExpression(
                        BinaryOperation.SUBTRACT,
                        call("floor", call("floor", IdentifierExpression(y))),
                        call("ceil", call("ceil", IdentifierExpression(x))),
                    ),
                    LiteralExpression(-1),
                ),
                LiteralExpression(-10),
            ),
            id="floor_divide_of_floor_minus_ceil",
        ),
    ],
)
def test_simplify_expression_survives_sympy_precision_exhaustion(
    build_expression: Callable[[Identifier, Identifier], Expression],
) -> None:
    """Test SymPy's ``PrecisionExhausted`` never escapes ``simplify_expression``.

    ``sympy.simplify`` checks a relational numerically at random points,
    and SymPy 1.14's integer-part evaluation raises ``PrecisionExhausted``
    where the argument of an outer ``floor`` is exactly an integer at such
    a point. Whether a call fails depends on SymPy's random state, so the
    test reseeds it (seeds 0 to 9 include failing ones for both shapes on
    SymPy 1.14) and clears SymPy's cache before each call. The result must
    keep the input's truth value at every grid point.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    expression = build_expression(x, y)
    expected = _tabulate_over_x_and_y(expression)

    for seed in range(10):
        clear_cache()
        sympy_random.seed(seed)
        result = simplify_expression(expression)

        assert _tabulate_over_x_and_y(result) == expected


# =============================================================================
# Native constants resolve by identity, not by name
# =============================================================================


@pytest.mark.parametrize("constant_name", ["pi", "e"])
def test_simplify_expression_does_not_decide_a_variable_named_after_a_constant(
    constant_name: str,
) -> None:
    """Test comparing a variable named ``pi`` to a literal stays undecided.

    Resolving the constant by ``name_hint`` made the bridge substitute
    the constant's value here and answer ``False``, fabricating a
    decision about a free variable it knew nothing about.
    """
    variable = mock_identifier(constant_name, 960)
    expression = IdentifierExpression(variable).equals(LiteralExpression(1))

    result = simplify_expression(expression)

    assert not isinstance(result, LiteralExpression)
    assert result.get_free_identifiers() == {variable}


@pytest.mark.parametrize("constant_name", ["pi", "e"])
def test_simplify_expression_honors_a_binding_for_a_variable_named_after_a_constant(
    constant_name: str,
) -> None:
    """Test a binding for a variable named ``pi`` is applied, not discarded.

    The identifier lowers to a substitutable symbol, so the caller's
    environment reaches it and the product folds to a literal.
    """
    variable = mock_identifier(constant_name, 961)
    expression = LiteralExpression(2) * IdentifierExpression(variable)

    result = simplify_expression(expression, {variable: LiteralExpression(3)})

    assert isinstance(result, LiteralExpression)
    assert result.value == 6


@pytest.mark.parametrize("constant_name", ["pi", "e"])
def test_simplify_expression_is_idempotent_over_a_native_constant(
    constant_name: str,
) -> None:
    """Test simplifying twice yields the same tree and keeps the constant's identifier.

    Lifting a sympy constant used to mint a brand-new ``Identifier`` per
    call, so two simplifications of one expression disagreed and the
    identifier the caller wrote disappeared from the result.
    """
    constant = get_native_constant_identifier(constant_name)
    expression = IdentifierExpression(constant) + LiteralExpression(0)

    first = simplify_expression(expression)
    second = simplify_expression(expression)

    assert first.is_structurally_equivalent(second)
    assert first.get_free_identifiers() == {constant}


# =============================================================================
# Binding a referenced native constant's canonical identifier is refused
# =============================================================================


def test_simplify_expression_refuses_a_binding_for_a_referenced_native_constant() -> (
    None
):
    """Test binding pi's canonical identifier is refused when pi is referenced.

    Without the refusal, the SymPy bridge lowers ``pi`` to ``sympy.pi``
    before substitution ever runs, so the binding's ``xreplace`` key
    never matches and is silently dropped instead of applied.
    """
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(pi) + LiteralExpression(1)

    with pytest.raises(NativeConstantBindingError, match="pi"):
        simplify_expression(expression, {pi: LiteralExpression(3)})


def test_simplify_expression_ignores_an_unreferenced_native_constant_binding() -> None:
    """Test a binding for pi is ignored when the expression does not reference pi."""
    x = mock_identifier("x", 0)
    pi = get_native_constant_identifier("pi")
    expression = IdentifierExpression(x) + LiteralExpression(1)

    result = simplify_expression(
        expression, {x: LiteralExpression(2), pi: LiteralExpression(3)}
    )

    assert isinstance(result, LiteralExpression)
    assert result.value == 3


# =============================================================================
# Defensive dispatch branches and noop output
# =============================================================================


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


def test_sympy_to_expression_converter_get_noop_output_raises() -> None:
    """Test `SymPyToExpressionConverter.get_noop_output` raises `PassExecutionError`."""
    with pytest.raises(PassExecutionError, match=r"does not define noop output"):
        SymPyToExpressionConverter().get_noop_output(sympy.Integer(0))


def test_sympy_to_expression_convert_rejects_unknown_node_type() -> None:
    """Test `convert` raises `TypeError` for a node that is neither Expr nor Boolean."""
    with pytest.raises(TypeError, match=r"Unsupported node type"):
        SymPyToExpressionConverter().convert(42)


def test_sympy_to_expression_convert_expr_rejects_unsupported_expr_subtype() -> None:
    """Test `convert_expr` raises `TypeError` for an unsupported `sympy.Expr`.

    ``sympy.Derivative`` is not in the per-name native lift mapping, the
    constant atom mapping, the ``Pow(x, 1/2)`` sqrt special case, or
    the arithmetic dispatch table, so it falls through to the typed
    rejection.
    """
    x = sympy.Symbol("x_0")
    with pytest.raises(TypeError, match=r"Unsupported expression type"):
        SymPyToExpressionConverter().convert_expr(sympy.Derivative(x, x))


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
    "method_name, sympy_class, identity_value",
    [
        pytest.param("_convert_add", sympy.Add, 0, id="add"),
        pytest.param("_convert_mul", sympy.Mul, 1, id="mul"),
    ],
)
def test_sympy_to_expression_convert_commutative_op_zero_arg_returns_identity(
    method_name: str, sympy_class: type, identity_value: int
) -> None:
    """Test `convert_Add`/`convert_Mul` return the identity literal on zero args."""
    fake = Mock(spec=sympy_class)
    fake.args = ()
    result = getattr(SymPyToExpressionConverter(), method_name)(fake)
    assert result.is_structurally_equivalent(LiteralExpression(identity_value))


@pytest.mark.parametrize(
    "method_name, sympy_class, sample_arg",
    [
        pytest.param("_convert_add", sympy.Add, 7, id="add"),
        pytest.param("_convert_mul", sympy.Mul, 5, id="mul"),
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


@pytest.mark.parametrize(
    "binding, expected",
    [(False, True), (True, False)],
    ids=["not_false_is_true", "not_true_is_false"],
)
def test_simplify_expression_negates_a_bound_boolean(
    binding: bool, expected: bool
) -> None:
    """Test ``!b`` under a binding evaluates to the negation of that binding.

    Lowering logical-not with Python's ``operator.not_`` would call
    ``bool()`` on the SymPy operand, which is truthy for a ``Symbol``, and
    decide the negation as ``False`` before substitution ran -- so both
    bindings would produce ``False`` and one of them would be wrong.
    """
    b = mock_identifier("b", 0)
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(b))

    result = simplify_expression(expression, {b: LiteralExpression(binding)})

    assert result.is_structurally_equivalent(LiteralExpression(expected))


def test_simplify_expression_keeps_an_unbound_negation_symbolic() -> None:
    """Test ``!b`` with nothing bound round-trips back to ``!b``.

    The negation has no decidable value, so it must survive lowering and
    lifting rather than collapsing to a literal.
    """
    b = mock_identifier("b", 0)
    expression = UnaryExpression(UnaryOperation.LOGICAL_NOT, IdentifierExpression(b))

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(expression)


def test_simplify_expression_negates_a_comparison_by_inverting_it() -> None:
    """Test ``!(x > 5)`` simplifies to the inverted comparison ``x <= 5``."""
    x = mock_identifier("x", 0)
    expression = UnaryExpression(
        UnaryOperation.LOGICAL_NOT,
        BinaryExpression(
            BinaryOperation.GREATER, IdentifierExpression(x), LiteralExpression(5)
        ),
    )

    result = simplify_expression(expression)

    expected = BinaryExpression(
        BinaryOperation.LESS_EQUAL, IdentifierExpression(x), LiteralExpression(5)
    )
    assert result.is_structurally_equivalent(expected)


def test_sympy_to_expression_convert_implies_raises_not_implemented() -> None:
    """Test `convert_Implies` surfaces a `NotImplementedError` cause through the pass.

    `SymPyToExpressionConverter` is a registered `CompilerPass`, so unexpected
    exceptions raised by ``run_pass`` are wrapped in `PassExecutionError`. The
    underlying `NotImplementedError` remains accessible via `__cause__`.
    """
    implies = sympy.Implies(sympy.Symbol("x_0"), sympy.Symbol("y_1"))

    with pytest.raises(PassExecutionError, match=r"NotImplementedError") as exc_info:
        convert_sympy_expression_to_expression(implies)

    assert isinstance(exc_info.value.__cause__, NotImplementedError)
    assert "Implies is not supported" in str(exc_info.value.__cause__)


def test_convert_implies_via_convert_method_raises_not_implemented() -> None:
    """Test calling `.convert(...)` directly preserves the original error type."""
    implies = sympy.Implies(sympy.Symbol("x_0"), sympy.Symbol("y_1"))

    with pytest.raises(NotImplementedError, match=r"Implies is not supported"):
        SymPyToExpressionConverter().convert(implies)


def test_sympy_two_argument_helper_rejects_wrong_arg_count() -> None:
    """Test the two-argument binary helper rejects nodes with wrong arity."""
    fake = Mock(spec=sympy.Pow)
    fake.args = (sympy.Integer(1), sympy.Integer(2), sympy.Integer(3))

    with pytest.raises(
        ValueError, match=r"Expected a binary operation to have exactly two arguments"
    ):
        SymPyToExpressionConverter()._convert_pow(fake)


# =============================================================================
# PiecewiseExpression <-> SymPy.Piecewise
# =============================================================================


def test_convert_single_case_piecewise_expression_to_sympy_piecewise() -> None:
    """Test a one-case ``PiecewiseExpression`` lowers to a two-branch ``Piecewise``.

    The condition is symbolic so sympy does not fold the ``Piecewise``
    at construction time.
    """
    x_identifier = mock_identifier("x", 0)
    expression = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(x_identifier),
                LiteralExpression(0),
            ),
        ),
        (LiteralExpression(1),),
        LiteralExpression(2),
    )

    result = convert_expression_to_sympy_expression(expression)

    assert isinstance(result, sympy.Piecewise)


def test_convert_piecewise_expression_emits_true_guarded_otherwise_branch() -> None:
    """Test the lowered ``sympy.Piecewise`` trailing branch is guarded by ``True``.

    Every case becomes a ``(value, condition)`` pair (SymPy's pair order
    is reversed from ours); the final branch pairs ``otherwise`` with an
    unconditional ``True`` guard so the result is total.
    """
    x_identifier = mock_identifier("x", 0)
    expression = PiecewiseExpression(
        (
            BinaryExpression(
                BinaryOperation.GREATER,
                IdentifierExpression(x_identifier),
                LiteralExpression(0),
            ),
        ),
        (LiteralExpression(1),),
        LiteralExpression(2),
    )

    result = convert_expression_to_sympy_expression(expression)

    assert isinstance(result, sympy.Piecewise)
    last_value, last_condition = result.args[-1]
    assert last_value == sympy.Integer(2)
    assert last_condition == sympy.true


def test_convert_multi_case_piecewise_expression_to_sympy_emits_every_case() -> None:
    """Test a multi-case ``PiecewiseExpression`` lowers with one branch per case."""
    x_identifier = mock_identifier("x", 0)
    x_symbol = IdentifierExpression(x_identifier)
    expression = PiecewiseExpression(
        (x_symbol > 0, x_symbol < 0),
        (LiteralExpression(1), LiteralExpression(-1)),
        LiteralExpression(0),
    )

    result = convert_expression_to_sympy_expression(expression)

    assert isinstance(result, sympy.Piecewise)
    NUM_EXPECTED_BRANCHES = 3
    assert len(result.args) == NUM_EXPECTED_BRANCHES


def test_convert_multi_case_piecewise_preserves_branch_order_and_content() -> None:
    """Test lowering a 4-case ``PiecewiseExpression`` pairs each branch correctly.

    Uses distinguishable literal values (10/20/30/99) so a bug that
    reordered branches, or paired a value with the wrong condition,
    would be caught -- unlike
    ``test_convert_multi_case_piecewise_expression_to_sympy_emits_every_case``
    above, which only checks the branch count.
    """
    x_identifier = mock_identifier("x", 0)
    x_symbol = IdentifierExpression(x_identifier)
    expression = PiecewiseExpression(
        (x_symbol > 0, x_symbol > 10, x_symbol > 20),
        (LiteralExpression(10), LiteralExpression(20), LiteralExpression(30)),
        LiteralExpression(99),
    )

    result = convert_expression_to_sympy_expression(expression)

    assert isinstance(result, sympy.Piecewise)
    x = sympy.Symbol("x_0")
    expected_branches = (
        (sympy.Integer(10), x > 0),
        (sympy.Integer(20), x > 10),
        (sympy.Integer(30), x > 20),
        (sympy.Integer(99), sympy.true),
    )
    NUM_EXPECTED_BRANCHES = 4
    assert len(result.args) == NUM_EXPECTED_BRANCHES
    for index, expected_branch in enumerate(expected_branches):
        assert result.args[index] == expected_branch, (
            f"branch {index}: expected {expected_branch}, got {result.args[index]}"
        )


def test_single_branch_sympy_piecewise_with_non_true_condition_raises() -> None:
    """Test a single-branch ``sympy.Piecewise`` with a non-``True`` condition raises.

    A ``True``-condition single branch collapses to a bare SymPy value
    before it ever reaches the lifter, so the only single-branch
    ``Piecewise`` object that survives construction has a symbolic
    condition. ``PiecewiseExpression`` is a total function, so this
    partial shape has no faithful IR representation: the lifter raises
    rather than degenerating to the branch's bare value.
    ``SymPyToExpressionConverter`` is a registered ``CompilerPass``, so
    the underlying ``PartialPiecewiseError`` is wrapped in
    ``PassExecutionError`` with the original error attached as
    ``__cause__``.
    """
    sympy_expression = sympy.Piecewise(
        (sympy.Integer(5), sympy.Symbol("flag_0")), evaluate=False
    )

    with pytest.raises(PassExecutionError, match=r"PartialPiecewiseError") as exc_info:
        convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(exc_info.value.__cause__, PartialPiecewiseError)
    assert "flag_0" in str(exc_info.value.__cause__)


def test_two_branch_sympy_piecewise_lifts_to_single_case_piecewise_expression() -> None:
    """Test a two-branch ``sympy.Piecewise`` lifts back to a one-case ``Piecewise``.

    The first branch's condition is symbolic so sympy does not fold the
    ``Piecewise`` at construction.
    """
    sympy_expression = sympy.Piecewise(
        (sympy.Integer(1), sympy.Symbol("flag_0")), (sympy.Integer(2), True)
    )

    result = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(result, PiecewiseExpression)
    assert len(result.conditions) == 1
    assert result.values[0].is_structurally_equivalent(LiteralExpression(1))
    assert result.otherwise.is_structurally_equivalent(LiteralExpression(2))


def test_sympy_ite_lifts_to_a_single_case_piecewise_expression() -> None:
    """Test a boolean ``sympy.ITE`` lifts to a one-case ``PiecewiseExpression``.

    ``ITE(condition, consequent, alternative)`` selects its consequent
    where the condition holds and its alternative everywhere else, so it
    lifts to the same total, one-case shape a two-branch
    ``sympy.Piecewise`` lifts to: the condition is the lifted first
    argument, the case's value is the lifted second argument, and
    ``otherwise`` is the lifted third argument.
    """
    x = sympy.Symbol("x_0")
    y = sympy.Symbol("y_1")
    sympy_expression = sympy.ITE(sympy.Eq(x, 0), sympy.Eq(y, 1), x < y)

    result = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(result, PiecewiseExpression)
    assert len(result.conditions) == 1
    expected_condition = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(mock_identifier("x", 0)),
        LiteralExpression(0),
    )
    expected_value = BinaryExpression(
        BinaryOperation.EQUAL,
        IdentifierExpression(mock_identifier("y", 1)),
        LiteralExpression(1),
    )
    expected_otherwise = BinaryExpression(
        BinaryOperation.LESS,
        IdentifierExpression(mock_identifier("x", 0)),
        IdentifierExpression(mock_identifier("y", 1)),
    )
    assert result.conditions[0].is_structurally_equivalent(expected_condition)
    assert result.values[0].is_structurally_equivalent(expected_value)
    assert result.otherwise.is_structurally_equivalent(expected_otherwise)


def test_multi_branch_sympy_piecewise_lifts_to_one_flat_piecewise_expression(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a multi-branch ``sympy.Piecewise`` lifts to a single flat node.

    A three-branch ``sympy.Piecewise`` produces exactly one
    ``PiecewiseExpression`` with two cases and one ``otherwise`` --
    not a tree of nested one-case nodes. The final condition here is
    ``True`` (a total ``Piecewise``), so this normal round-trip shape
    logs no warning, contrasting with a genuinely partial final
    condition (see
    ``test_multi_branch_sympy_piecewise_with_non_true_final_condition_raises``),
    which raises ``PartialPiecewiseError`` instead.
    """
    sympy_expression = sympy.Piecewise(
        (sympy.Integer(1), sympy.Symbol("flag_0")),
        (sympy.Integer(2), sympy.Symbol("flag_1")),
        (sympy.Integer(3), True),
    )

    with caplog.at_level(
        logging.WARNING, logger="fhy_core.symbolic.expression.passes.sympy"
    ):
        result = convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(result, PiecewiseExpression)
    NUM_EXPECTED_CASES = 2
    assert len(result.conditions) == NUM_EXPECTED_CASES
    assert not isinstance(result.otherwise, PiecewiseExpression)
    assert result.values[0].is_structurally_equivalent(LiteralExpression(1))
    assert result.values[1].is_structurally_equivalent(LiteralExpression(2))
    assert result.otherwise.is_structurally_equivalent(LiteralExpression(3))
    assert not any(record.levelno == logging.WARNING for record in caplog.records), (
        "a total (True-terminated) multi-branch lift must not warn"
    )


def test_multi_branch_sympy_piecewise_with_non_true_final_condition_raises() -> None:
    """Test a multi-branch ``sympy.Piecewise`` whose final condition is not ``True``.

    A final branch condition that is not ``sympy.true`` means the
    ``Piecewise`` does not cover its domain. Treating the final
    branch's value as ``otherwise`` would silently cover the region the
    original condition excluded, so the lifter raises
    ``PartialPiecewiseError`` naming the offending condition instead of
    totalizing the result, wrapped in ``PassExecutionError`` per
    ``SymPyToExpressionConverter``'s pass contract.
    """
    sympy_expression = sympy.Piecewise(
        (sympy.Integer(1), sympy.Symbol("flag_0")),
        (sympy.Integer(2), sympy.Symbol("flag_1")),
    )

    with pytest.raises(PassExecutionError, match=r"PartialPiecewiseError") as exc_info:
        convert_sympy_expression_to_expression(sympy_expression)

    assert isinstance(exc_info.value.__cause__, PartialPiecewiseError)
    assert "flag_1" in str(exc_info.value.__cause__)


# =============================================================================
# CallExpression interplay with the SymPy converter
# =============================================================================


def test_convert_call_expression_to_sympy_rejects_unresolved_call() -> None:
    """Test SymPy lowering rejects ``CallExpression`` (callers inline first)."""
    expression = CallExpression("max", (LiteralExpression(1), LiteralExpression(2)))

    with pytest.raises(PassExecutionError, match="TypeError"):
        convert_expression_to_sympy_expression(expression)


# =============================================================================
# Boolean connectives refuse a numeric operand rather than folding it bitwise
# =============================================================================


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(logical_and(LiteralExpression(2), LiteralExpression(4)), id="and"),
        pytest.param(logical_or(LiteralExpression(2), LiteralExpression(4)), id="or"),
        pytest.param(logical_not(LiteralExpression(2)), id="not"),
    ],
)
def test_convert_expression_to_sympy_refuses_a_numeric_logical_operand(
    expression: Expression,
) -> None:
    """Test lowering a Boolean connective over integers is refused, not computed.

    SymPy's ``&``/``|`` are *bitwise* on ``sympy.Integer`` (``2 & 4`` is
    ``0``, ``2 | 4`` is ``6``) and its ``Not`` coerces by truthiness, so
    lowering such a node would hand back a well-formed SymPy object
    carrying a numerically wrong answer. The bridge screens the shape out
    before SymPy sees it.
    """
    with pytest.raises(NonBooleanLogicalOperandError):
        convert_expression_to_sympy_expression(expression)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param(
            logical_and(LiteralExpression(2), LiteralExpression(4)),
            id="and_does_not_fold_to_0",
        ),
        pytest.param(
            logical_or(LiteralExpression(2), LiteralExpression(4)),
            id="or_does_not_fold_to_6",
        ),
    ],
)
def test_simplify_expression_refuses_a_numeric_connective_instead_of_folding_it(
    expression: Expression,
) -> None:
    """Test simplification refuses rather than returning a bitwise literal.

    A bitwise lowering makes ``2 && 4`` fold to ``LiteralExpression(0)``
    and ``2 || 4`` to ``LiteralExpression(6)`` -- numerically wrong
    answers presented as decided ones, which is worse than no answer.
    Simplification has to refuse the shape instead of reporting either.
    """
    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(expression)


def test_simplify_expression_refusal_is_this_packages_error_not_sympys() -> None:
    """Test the refusal is the package's typed error, not SymPy's own `TypeError`.

    ``sympy.And(Integer(2), Integer(4))`` raises a bare
    ``TypeError("expecting bool or Boolean, ...")``. A caller cannot tell
    that apart from any other type error, and it names SymPy's vocabulary
    rather than the expression that is wrong, so the bridge screens ahead
    of SymPy and raises its own registered error instead.
    """
    expression = logical_and(LiteralExpression(2), LiteralExpression(4))

    with pytest.raises(NonBooleanLogicalOperandError) as exc_info:
        simplify_expression(expression)

    assert type(exc_info.value) is NonBooleanLogicalOperandError
    assert not isinstance(exc_info.value, PassExecutionError)
    assert "expecting bool or Boolean" not in str(exc_info.value)


def test_simplify_expression_refuses_a_number_bound_into_a_connective() -> None:
    """Test a numeric binding under a connective is refused by the same error.

    The number reaches the connective one step later than a written
    literal does -- SymPy raises from inside ``xreplace`` -- so without
    the environment in the screen this path would leak SymPy's own
    ``TypeError`` from a public entry point.
    """
    p = mock_identifier("p", 0)
    q = mock_identifier("q", 1)
    expression = logical_and(IdentifierExpression(p), IdentifierExpression(q))

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(
            expression, {p: LiteralExpression(2), q: LiteralExpression(4)}
        )


@pytest.mark.parametrize(
    "build_expression",
    [
        pytest.param(
            lambda constant: logical_and(constant, LiteralExpression(True)), id="and"
        ),
        pytest.param(logical_not, id="not"),
        pytest.param(
            lambda constant: PiecewiseExpression(
                (constant,), (LiteralExpression(1),), LiteralExpression(2)
            ),
            id="case_condition",
        ),
    ],
)
def test_simplify_expression_refuses_a_native_constant_in_a_boolean_position(
    build_expression: Callable[[Expression], Expression],
) -> None:
    """Test a constant in a Boolean position is refused with the package's error.

    The bridge lowers the constant to its value, so ``pi & True`` and a
    ``pi`` case condition made SymPy raise its own ``TypeError``, wrapped
    as a pass failure, and ``~pi`` came back as a residual that is no
    truth value at all. Each is ill-typed, and is refused as such.
    """
    constant = IdentifierExpression(get_native_constant_identifier("pi"))

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(build_expression(constant))


def test_simplify_expression_refuses_a_numeric_result_call_under_a_connective() -> None:
    """Test a call with a registered numeric result sort is refused under `not`.

    ``sqrt`` is registered with a REAL result sort, so ``sqrt(4.0)`` is as
    ill-typed under ``LOGICAL_NOT`` as a bare numeric literal, even though
    the call's own node carries no sort -- the registry does.
    """
    numeric_call = call("sqrt", LiteralExpression(4.0))

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(logical_not(numeric_call))


def test_simplify_expression_refuses_a_mixed_branch_piecewise_under_not() -> None:
    """Test a piecewise operand with a single numeric branch is refused too.

    Without this, SymPy's own truthiness reading of ``Piecewise`` folds
    ``not {2 if x > 0 else True}`` to ``LiteralExpression(False)``: a
    numerically meaningless answer presented as a decided one.
    """
    x = mock_identifier("x", 0)
    mixed_piecewise = piecewise(
        (IdentifierExpression(x) > LiteralExpression(0), LiteralExpression(2)),
        otherwise=LiteralExpression(True),
    )

    with pytest.raises(NonBooleanLogicalOperandError):
        simplify_expression(logical_not(mixed_piecewise))


@pytest.mark.parametrize(
    "expression, expected",
    [
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(False)),
            False,
            id="and_true_false",
        ),
        pytest.param(
            logical_and(LiteralExpression(True), LiteralExpression(True)),
            True,
            id="and_true_true",
        ),
        pytest.param(
            logical_or(LiteralExpression(True), LiteralExpression(False)),
            True,
            id="or_true_false",
        ),
        pytest.param(
            logical_or(LiteralExpression(False), LiteralExpression(False)),
            False,
            id="or_false_false",
        ),
        pytest.param(logical_not(LiteralExpression(True)), False, id="not_true"),
    ],
)
def test_simplify_expression_still_folds_boolean_connectives(
    expression: Expression, expected: bool
) -> None:
    """Test Boolean operands still fold end to end through the SymPy bridge.

    The connectives lower through ``sympy.And``/``sympy.Or``/``sympy.Not``,
    which decide a ground Boolean conjunction; refusing a numeric operand
    must not cost the Boolean case its evaluation.
    """
    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(LiteralExpression(expected))


def test_simplify_expression_folds_a_conjunction_of_a_bound_and_a_literal_boolean() -> (
    None
):
    """Test a connective mixing an identifier with a Boolean literal still folds."""
    p = mock_identifier("p", 0)
    expression = logical_and(IdentifierExpression(p), LiteralExpression(True))

    result = simplify_expression(expression, {p: LiteralExpression(False)})

    assert result.is_structurally_equivalent(LiteralExpression(False))


@pytest.mark.parametrize(
    "expression, sympy_type",
    [
        pytest.param(
            logical_and(
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            sympy.logic.boolalg.And,
            id="and",
        ),
        pytest.param(
            logical_or(
                IdentifierExpression(mock_identifier("x", 0)),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            sympy.logic.boolalg.Or,
            id="or",
        ),
    ],
)
def test_convert_expression_to_sympy_lowers_a_connective_to_a_sympy_boolean(
    expression: Expression, sympy_type: type
) -> None:
    """Test a connective over symbols lowers to the SymPy Boolean node, not a number.

    Pins the operator table's target: a bitwise-operator lowering happens
    to produce the same node for two ``Symbol`` operands, so the symbolic
    case alone cannot distinguish the two mappings. Asserting the node
    class keeps the table honest about what it lowers to.
    """
    lowered = convert_expression_to_sympy_expression(expression)

    assert isinstance(lowered, sympy_type)


@pytest.mark.parametrize(
    "function_name, left, right, expected",
    [
        pytest.param("xor", True, False, True, id="xor_true_false"),
        pytest.param("xor", True, True, False, id="xor_true_true"),
        pytest.param("nand", True, False, True, id="nand_true_false"),
        pytest.param("nand", True, True, False, id="nand_true_true"),
        pytest.param("nor", False, False, True, id="nor_false_false"),
        pytest.param("nor", True, False, False, id="nor_true_false"),
        pytest.param("implies", True, False, False, id="implies_true_false"),
        pytest.param("implies", False, True, True, id="implies_false_true"),
        pytest.param("iff", True, True, True, id="iff_true_true"),
        pytest.param("iff", True, False, False, id="iff_true_false"),
    ],
)
def test_inlined_boolean_builtin_folds_through_the_sympy_bridge(
    function_name: str, left: bool, right: bool, expected: bool
) -> None:
    """Test the composed Boolean built-ins still evaluate through the SymPy bridge.

    ``xor``, ``nand``, ``nor``, ``implies``, and ``iff`` have bodies built
    out of ``LOGICAL_AND``/``LOGICAL_OR``/``LOGICAL_NOT``, so inlining one
    over Boolean arguments produces exactly the operand shape the numeric
    screen must leave alone. Each row pins the truth-table entry, so a
    lowering that merely fails to raise is not enough to pass.
    """
    inlined = inline_functions(
        call(function_name, LiteralExpression(left), LiteralExpression(right))
    )

    result = simplify_expression(inlined)

    assert result.is_structurally_equivalent(LiteralExpression(expected))


@pytest.mark.parametrize(
    "sympy_expression",
    [
        pytest.param(
            sympy.Xor(sympy.Symbol("x_0"), sympy.Symbol("y_1"), evaluate=False),
            id="xor",
        ),
        pytest.param(
            sympy.Not(sympy.Or(sympy.Symbol("x_0"), sympy.Symbol("y_1"))), id="nor"
        ),
        pytest.param(
            sympy.Not(sympy.And(sympy.Symbol("x_0"), sympy.Symbol("y_1"))), id="nand"
        ),
        pytest.param(
            sympy.And(sympy.Symbol("x_0"), sympy.Symbol("y_1"), sympy.Symbol("z_2")),
            id="n_ary_and",
        ),
        pytest.param(
            sympy.Or(sympy.Symbol("x_0"), sympy.Symbol("y_1"), sympy.Symbol("z_2")),
            id="n_ary_or",
        ),
    ],
)
def test_lifted_boolean_node_lowers_back_to_an_equivalent_sympy_boolean(
    sympy_expression: sympy.logic.boolalg.Boolean,
) -> None:
    """Test the lifters' output lowers back to a logically equivalent SymPy node.

    ``Xor``, ``Nor``, ``Nand``, and the n-ary ``And``/``Or`` rebuild all
    lift to IR trees made of ``LOGICAL_AND``/``LOGICAL_OR``/
    ``LOGICAL_NOT`` over Boolean operands. Lowering has to accept every
    shape the lifters can produce, or a round trip through the bridge
    would fail on the bridge's own output.
    """
    lifted = convert_sympy_expression_to_expression(sympy_expression)

    lowered = convert_expression_to_sympy_expression(lifted)

    assert isinstance(lowered, sympy.logic.boolalg.Boolean)
    assert sympy.simplify(sympy.Equivalent(sympy_expression, lowered)) is sympy.true


# =============================================================================
# A Boolean piecewise lowers as an operand of a Boolean connective
# =============================================================================


_X_Y_GRID = tuple((x, y) for x in range(-1, 4) for y in range(-1, 4))


def _build_boolean_piecewise(x: Expression, y: Expression) -> Expression:
    """Build the Boolean piecewise ``y == 1 if x == 0, otherwise x < y``."""
    return piecewise(
        (
            BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(0)),
            BinaryExpression(BinaryOperation.EQUAL, y, LiteralExpression(1)),
        ),
        otherwise=BinaryExpression(BinaryOperation.LESS, x, y),
    )


def _build_first_match_expansion(x: Expression, y: Expression) -> Expression:
    """Build the meaning of `_build_boolean_piecewise` from connectives alone.

    ``(x == 0 && y == 1) || (!(x == 0) && x < y)`` is the first-match
    reading of the piecewise and builds no piecewise node.
    """
    is_first_case = BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(0))
    return logical_or(
        logical_and(
            is_first_case,
            BinaryExpression(BinaryOperation.EQUAL, y, LiteralExpression(1)),
        ),
        logical_and(
            logical_not(is_first_case), BinaryExpression(BinaryOperation.LESS, x, y)
        ),
    )


def _tabulate_over_x_and_y(expression: Expression) -> tuple[bool, ...]:
    """Return the lowered expression's truth value at every grid point."""
    lowered = convert_expression_to_sympy_expression(expression)
    x_symbol, y_symbol = sympy.Symbol("x_0"), sympy.Symbol("y_1")
    return tuple(
        bool(lowered.subs({x_symbol: x_value, y_symbol: y_value}))
        for x_value, y_value in _X_Y_GRID
    )


@pytest.mark.parametrize(
    "build_connective",
    [
        pytest.param(logical_and, id="and_left"),
        pytest.param(
            lambda operand, other: logical_and(other, operand), id="and_right"
        ),
        pytest.param(logical_or, id="or_left"),
        pytest.param(lambda operand, other: logical_or(other, operand), id="or_right"),
    ],
)
def test_convert_expression_to_sympy_lowers_a_boolean_piecewise_connective_operand(
    build_connective: Callable[[Expression, Expression], Expression],
) -> None:
    """Test a Boolean piecewise under ``&&``/``||`` lowers to its first-match meaning.

    ``sympy.And`` and ``sympy.Or`` accept only SymPy Booleans, and a
    ``sympy.Piecewise`` is not one even when every branch is Boolean. The
    lowered connective must agree with the connective-only expansion of
    the piecewise at every grid point.
    """
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))
    other = BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2))

    table = _tabulate_over_x_and_y(
        build_connective(_build_boolean_piecewise(x, y), other)
    )

    assert table == _tabulate_over_x_and_y(
        build_connective(_build_first_match_expansion(x, y), other)
    )


def test_convert_expression_to_sympy_lowers_a_nested_boolean_piecewise_operand() -> (
    None
):
    """Test a Boolean piecewise nested in another's case value lowers under ``||``."""
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))
    is_outer_case = BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2))
    y_is_zero = BinaryExpression(BinaryOperation.EQUAL, y, LiteralExpression(0))
    y_is_negative = BinaryExpression(BinaryOperation.LESS, y, LiteralExpression(0))
    nested = piecewise(
        (is_outer_case, _build_boolean_piecewise(x, y)), otherwise=y_is_zero
    )
    expansion = logical_or(
        logical_and(is_outer_case, _build_first_match_expansion(x, y)),
        logical_and(logical_not(is_outer_case), y_is_zero),
    )

    table = _tabulate_over_x_and_y(logical_or(nested, y_is_negative))

    assert table == _tabulate_over_x_and_y(logical_or(expansion, y_is_negative))


def test_convert_expression_to_sympy_keeps_a_numeric_piecewise_in_a_condition() -> None:
    """Test a numeric piecewise inside a connective operand's condition keeps its value.

    The Boolean piecewise ``y < 3 if (1 if x == 0, otherwise 2) < y,
    otherwise x == 3`` sits under ``&&``; its condition compares a numeric
    piecewise, which has to stay numeric while the Boolean one around it
    is lowered.
    """
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))
    numeric = piecewise(
        (
            BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(0)),
            LiteralExpression(1),
        ),
        otherwise=LiteralExpression(2),
    )
    condition = BinaryExpression(BinaryOperation.LESS, numeric, y)
    value = BinaryExpression(BinaryOperation.LESS, y, LiteralExpression(3))
    fallback = BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(3))
    other = BinaryExpression(BinaryOperation.GREATER_EQUAL, y, LiteralExpression(0))
    expansion = logical_or(
        logical_and(condition, value), logical_and(logical_not(condition), fallback)
    )

    table = _tabulate_over_x_and_y(
        logical_and(piecewise((condition, value), otherwise=fallback), other)
    )

    assert table == _tabulate_over_x_and_y(logical_and(expansion, other))


def test_boolean_piecewise_connective_operand_round_trips_through_sympy() -> None:
    """Test lifting a lowered Boolean piecewise connective keeps its truth table."""
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))
    other = BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2))
    lowered = convert_expression_to_sympy_expression(
        logical_and(_build_boolean_piecewise(x, y), other)
    )

    lifted = convert_sympy_expression_to_expression(lowered)

    assert _tabulate_over_x_and_y(lifted) == _tabulate_over_x_and_y(
        logical_and(_build_first_match_expansion(x, y), other)
    )


def test_convert_expression_to_sympy_negates_a_boolean_piecewise_operand() -> None:
    """Test ``!`` over a Boolean piecewise lowers to its first-match meaning."""
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))

    table = _tabulate_over_x_and_y(logical_not(_build_boolean_piecewise(x, y)))

    assert table == _tabulate_over_x_and_y(
        logical_not(_build_first_match_expansion(x, y))
    )


def test_simplify_expression_negates_a_constant_boolean_piecewise() -> None:
    """Test ``!(False if 0 < x, otherwise False)`` simplifies to ``True``."""
    x = IdentifierExpression(mock_identifier("x", 0))
    expression = logical_not(
        piecewise(
            (
                BinaryExpression(BinaryOperation.LESS, LiteralExpression(0), x),
                LiteralExpression(False),
            ),
            otherwise=LiteralExpression(False),
        )
    )

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(LiteralExpression(True))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "SymPy 1.14's simplify drops the boundary point of a negated "
        "univariate range: simplify(Not(0 < x & x < 5)) returns "
        "(x >= 5) | (x < 0), which is False at x = 0, where the expression "
        "holds."
    ),
)
def test_simplify_expression_keeps_the_truth_table_of_a_negated_range() -> None:
    """Test simplifying ``!(0 < x && x < 5)`` keeps its truth value everywhere."""
    x = IdentifierExpression(mock_identifier("x", 0))
    expression = logical_not(
        logical_and(
            BinaryExpression(BinaryOperation.LESS, LiteralExpression(0), x),
            BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(5)),
        )
    )

    simplified = simplify_expression(expression)

    assert _tabulate_over_x_and_y(simplified) == _tabulate_over_x_and_y(expression)


# =============================================================================
# A Boolean piecewise compares as a Boolean
# =============================================================================


@pytest.mark.parametrize(
    "build_comparison",
    [
        pytest.param(
            lambda operand, _: BinaryExpression(
                BinaryOperation.EQUAL, operand, LiteralExpression(True)
            ),
            id="equal_to_true",
        ),
        pytest.param(
            lambda operand, _: BinaryExpression(
                BinaryOperation.EQUAL, LiteralExpression(True), operand
            ),
            id="true_equal_to",
        ),
        pytest.param(
            lambda operand, x: BinaryExpression(
                BinaryOperation.EQUAL,
                operand,
                BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2)),
            ),
            id="equal_to_relation",
        ),
        pytest.param(
            lambda operand, x: BinaryExpression(
                BinaryOperation.NOT_EQUAL,
                operand,
                BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2)),
            ),
            id="not_equal_to_relation",
        ),
    ],
)
def test_convert_expression_to_sympy_compares_a_boolean_piecewise_as_a_boolean(
    build_comparison: Callable[[Expression, Expression], Expression],
) -> None:
    """Test ``==``/``!=`` over a Boolean piecewise agree with its connective form.

    SymPy decides a comparison between a Boolean and a non-Boolean as
    unequal on sight, and a ``sympy.Piecewise`` is not a SymPy Boolean
    even when every branch is Boolean. The comparison must agree, at
    every grid point, with the same comparison over the piecewise's
    connective-only expansion.
    """
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))

    table = _tabulate_over_x_and_y(build_comparison(_build_boolean_piecewise(x, y), x))

    assert table == _tabulate_over_x_and_y(
        build_comparison(_build_first_match_expansion(x, y), x)
    )


@pytest.mark.parametrize(
    ("x_value", "y_value", "expected"),
    [(0, 1, True), (0, 2, False), (1, 2, True), (3, 2, False)],
)
def test_simplify_expression_decides_a_boolean_piecewise_equality_by_its_bindings(
    x_value: int, y_value: int, expected: bool
) -> None:
    """Test ``pw == True`` simplifies to the piecewise's own value under a binding.

    The piecewise is ``y == 1 if x == 0, otherwise x < y``; the rows reach
    each branch with each truth value.
    """
    x_identifier = mock_identifier("x", 0)
    y_identifier = mock_identifier("y", 1)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        _build_boolean_piecewise(
            IdentifierExpression(x_identifier), IdentifierExpression(y_identifier)
        ),
        LiteralExpression(True),
    )

    result = simplify_expression(
        expression,
        {
            x_identifier: LiteralExpression(x_value),
            y_identifier: LiteralExpression(y_value),
        },
    )

    assert result.is_structurally_equivalent(LiteralExpression(expected))


@pytest.mark.parametrize(
    ("x_value", "y_value", "b_value", "expected"),
    [(0, 1, True, True), (0, 1, False, False), (1, 2, False, True)],
)
def test_simplify_expression_decides_a_symbol_branch_piecewise_equality_by_its_bindings(
    x_value: int, y_value: int, b_value: bool, expected: bool
) -> None:
    """Test ``(b if x == 0, otherwise x < y) == True`` follows its bindings.

    A branch that is a bare identifier does not show that the piecewise is
    Boolean; the ``True`` on the other side of ``==`` does.
    """
    x_identifier = mock_identifier("x", 0)
    y_identifier = mock_identifier("y", 1)
    b_identifier = mock_identifier("b", 2)
    x = IdentifierExpression(x_identifier)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        piecewise(
            (
                BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(0)),
                IdentifierExpression(b_identifier),
            ),
            otherwise=BinaryExpression(
                BinaryOperation.LESS, x, IdentifierExpression(y_identifier)
            ),
        ),
        LiteralExpression(True),
    )

    result = simplify_expression(
        expression,
        {
            x_identifier: LiteralExpression(x_value),
            y_identifier: LiteralExpression(y_value),
            b_identifier: LiteralExpression(b_value),
        },
    )

    assert result.is_structurally_equivalent(LiteralExpression(expected))


def test_convert_expression_to_sympy_keeps_a_numeric_piecewise_comparison_numeric() -> (
    None
):
    """Test ``==`` over a numeric piecewise still compares numbers.

    ``(1 if x == 0, otherwise 2) == 1`` holds exactly where ``x == 0`` does.
    """
    x = IdentifierExpression(mock_identifier("x", 0))
    is_zero = BinaryExpression(BinaryOperation.EQUAL, x, LiteralExpression(0))
    numeric = piecewise((is_zero, LiteralExpression(1)), otherwise=LiteralExpression(2))

    table = _tabulate_over_x_and_y(
        BinaryExpression(BinaryOperation.EQUAL, numeric, LiteralExpression(1))
    )

    assert table == _tabulate_over_x_and_y(is_zero)


def test_boolean_piecewise_comparison_round_trips_through_sympy() -> None:
    """Test lifting a lowered ``pw != (x < 2)`` keeps its truth table."""
    x = IdentifierExpression(mock_identifier("x", 0))
    y = IdentifierExpression(mock_identifier("y", 1))
    relation = BinaryExpression(BinaryOperation.LESS, x, LiteralExpression(2))
    lowered = convert_expression_to_sympy_expression(
        BinaryExpression(
            BinaryOperation.NOT_EQUAL, _build_boolean_piecewise(x, y), relation
        )
    )

    lifted = convert_sympy_expression_to_expression(lowered)

    assert _tabulate_over_x_and_y(lifted) == _tabulate_over_x_and_y(
        BinaryExpression(
            BinaryOperation.NOT_EQUAL, _build_first_match_expansion(x, y), relation
        )
    )


def _evaluate_sympy_boolean(sympy_expression: Any, values: Mapping[str, Any]) -> bool:
    """Return the truth value of ``sympy_expression`` with named symbols bound."""
    return bool(
        sympy_expression.subs(
            {sympy.Symbol(name): value for name, value in values.items()}
        )
    )


_PARTIAL_BINDING_ENTRY_POINTS = [
    pytest.param(
        lambda expression, environment: substitute_sympy_expression_variables(
            convert_expression_to_sympy_expression(expression), environment
        ),
        id="substitute",
    ),
    pytest.param(
        lambda expression, environment: convert_expression_to_sympy_expression(
            simplify_expression(expression, environment)
        ),
        id="simplify",
    ),
]


@pytest.mark.parametrize("apply_partial_binding", _PARTIAL_BINDING_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("x_value", "b1_value", "b2_value", "expected"),
    [
        (0, True, False, True),
        (0, False, True, False),
        (1, True, False, False),
        (1, False, True, True),
    ],
)
def test_binding_only_the_boolean_side_keeps_a_piecewise_comparison_open(
    apply_partial_binding: Callable[[Expression, Mapping[Identifier, Expression]], Any],
    x_value: int,
    b1_value: bool,
    b2_value: bool,
    expected: bool,
) -> None:
    """Test a piecewise compared with ``b3`` still means itself once ``b3 = True``.

    The expression is ``(b1 if x == 0, otherwise b2) == b3``. Nothing in
    the tree shows the piecewise is Boolean until ``b3`` is bound, and the
    comparison must then mean ``b1 if x == 0, otherwise b2`` rather than a
    decided constant.
    """
    x_identifier = mock_identifier("x", 0)
    b3_identifier = mock_identifier("b3", 5)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        piecewise(
            (
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    IdentifierExpression(x_identifier),
                    LiteralExpression(0),
                ),
                IdentifierExpression(mock_identifier("b1", 3)),
            ),
            otherwise=IdentifierExpression(mock_identifier("b2", 4)),
        ),
        IdentifierExpression(b3_identifier),
    )

    partially_bound = apply_partial_binding(
        expression, {b3_identifier: LiteralExpression(True)}
    )

    assert (
        _evaluate_sympy_boolean(
            partially_bound, {"x_0": x_value, "b1_3": b1_value, "b2_4": b2_value}
        )
        is expected
    )


@pytest.mark.parametrize("apply_partial_binding", _PARTIAL_BINDING_ENTRY_POINTS)
@pytest.mark.parametrize(
    ("x_value", "y_value", "expected"), [(0, 1, True), (0, 2, False), (1, 5, True)]
)
def test_binding_only_the_numeric_side_keeps_a_piecewise_comparison_numeric(
    apply_partial_binding: Callable[[Expression, Mapping[Identifier, Expression]], Any],
    x_value: int,
    y_value: int,
    expected: bool,
) -> None:
    """Test a numeric piecewise compared with ``z`` stays numeric once ``z = 1``.

    The expression is ``(y if x == 0, otherwise 1) == z``.
    """
    x_identifier = mock_identifier("x", 0)
    z_identifier = mock_identifier("z", 2)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        piecewise(
            (
                BinaryExpression(
                    BinaryOperation.EQUAL,
                    IdentifierExpression(x_identifier),
                    LiteralExpression(0),
                ),
                IdentifierExpression(mock_identifier("y", 1)),
            ),
            otherwise=LiteralExpression(1),
        ),
        IdentifierExpression(z_identifier),
    )

    partially_bound = apply_partial_binding(
        expression, {z_identifier: LiteralExpression(1)}
    )

    assert (
        _evaluate_sympy_boolean(partially_bound, {"x_0": x_value, "y_1": y_value})
        is expected
    )


# =============================================================================
# Rational lifting
# =============================================================================


@pytest.mark.parametrize(
    "rational, expected_expression",
    [
        pytest.param(sympy.Rational(1, 2), LiteralExpression("0.5"), id="one_half"),
        pytest.param(
            sympy.Rational(7, 2), LiteralExpression("3.5"), id="three_and_a_half"
        ),
        pytest.param(
            sympy.Rational(1, 8), LiteralExpression("0.125"), id="powers_of_two_only"
        ),
        pytest.param(
            sympy.Rational(2**53 - 1, 2),
            LiteralExpression("4503599627370495.5"),
            id="largest_half_a_float_holds",
        ),
        pytest.param(
            sympy.Rational(-7, 2),
            UnaryExpression(UnaryOperation.NEGATE, LiteralExpression("3.5")),
            id="negative",
        ),
    ],
)
def test_binary_exact_rational_lifts_to_an_exact_decimal_string_literal(
    rational: sympy.Rational, expected_expression: Expression
) -> None:
    """Test a rational some binary float equals lifts to its exact decimal text.

    Such a rational has a power-of-two denominator, so its decimal
    expansion terminates, and the NumPy evaluator reads that text back as
    the same number rather than refusing it. The float grammar carries no
    sign, so a negative one lifts as a ``NEGATE`` of its magnitude.
    """
    result = convert_sympy_expression_to_expression(rational)

    assert result.is_structurally_equivalent(expected_expression)


@pytest.mark.parametrize(
    "rational, expected_numerator, expected_denominator",
    [
        pytest.param(sympy.Rational(1, 10), 1, 10, id="one_tenth"),
        pytest.param(sympy.Rational(1, 5), 1, 5, id="powers_of_five_only"),
        pytest.param(sympy.Rational(3, 40), 3, 40, id="mixed_two_and_five_factors"),
        pytest.param(sympy.Rational(7, 5), 7, 5, id="seven_fifths"),
        pytest.param(
            sympy.Rational(2**53 + 1, 2),
            2**53 + 1,
            2,
            id="half_beyond_float_precision",
        ),
        pytest.param(sympy.Rational(-1, 10), -1, 10, id="negative_one_tenth"),
    ],
)
def test_non_binary_terminating_rational_lifts_to_an_exact_divide(
    rational: sympy.Rational, expected_numerator: int, expected_denominator: int
) -> None:
    """Test a terminating rational no binary float equals lifts to a `DIVIDE`.

    One tenth has finite decimal text, but no binary ``float`` equals it,
    so the NumPy evaluator would refuse that text rather than round it.
    Half past ``2**52`` has a power-of-two denominator and still misses:
    its significand needs 54 bits. The quotient of the integer numerator
    and denominator is exact, and the sign rides on the numerator, as it
    does for a repeating rational.
    """
    result = convert_sympy_expression_to_expression(rational)

    assert result.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(expected_numerator),
            LiteralExpression(expected_denominator),
        )
    )


@pytest.mark.parametrize(
    "rational, expected_numerator, expected_denominator",
    [
        pytest.param(sympy.Rational(1, 3), 1, 3, id="one_third"),
        pytest.param(sympy.Rational(2, 7), 2, 7, id="two_sevenths"),
        pytest.param(sympy.Rational(-1, 3), -1, 3, id="negative_one_third"),
        pytest.param(
            sympy.Rational(1, 30), 1, 30, id="terminating_factors_plus_a_third"
        ),
    ],
)
def test_repeating_rational_lifts_to_an_exact_divide(
    rational: sympy.Rational, expected_numerator: int, expected_denominator: int
) -> None:
    """Test a rational with no finite decimal text lifts to a `DIVIDE`.

    One third has no exact decimal spelling at any precision, so lifting
    it as decimal text would have to round. The quotient of its numerator
    and denominator is exact instead, and keeps lifting total over every
    rational SymPy can produce.
    """
    result = convert_sympy_expression_to_expression(rational)

    assert result.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(expected_numerator),
            LiteralExpression(expected_denominator),
        )
    )


@pytest.mark.parametrize(
    "rational, expected_type",
    [
        pytest.param(
            sympy.Rational(1, 2**100), LiteralExpression, id="hundred_binary_places"
        ),
        pytest.param(
            sympy.Rational(1, 10**30), BinaryExpression, id="thirty_decimal_places"
        ),
        # Reduces to an odd 63-bit numerator over 2**79: a power-of-two
        # denominator, but a significand wider than a float's 53 bits.
        pytest.param(
            sympy.Rational(12345678901234567890, 2**80),
            BinaryExpression,
            id="wide_numerator_and_scale",
        ),
    ],
)
def test_terminating_rational_lift_is_exact_past_the_default_decimal_precision(
    rational: sympy.Rational, expected_type: type[Expression]
) -> None:
    """Test a rational wider than `decimal`'s default context lifts exactly.

    ``decimal`` rounds arithmetic to its context precision, 28 digits by
    default, so a lift that computed the digits by dividing would silently
    truncate here, and so would a binary-exactness check that rounded the
    text before comparing it. Lowering the lift has to recover the
    original rational in either form: decimal text for ``2**-100``, which
    a binary float equals, and a ``DIVIDE`` for the other two, which no
    binary float equals. The text has to stay in fixed-point form, since
    the float grammar refuses scientific notation at construction.
    """
    result = convert_sympy_expression_to_expression(rational)

    assert type(result) is expected_type
    assert convert_expression_to_sympy_expression(result) == rational


def test_integer_lifts_as_an_integer_and_not_as_a_rational() -> None:
    """Test `sympy.Integer` is not captured by the rational arm.

    ``sympy.Integer`` subclasses ``sympy.Rational``, and the lift
    dispatch is first-match-wins, so an ``Integer`` reaching the rational
    arm would come back as ``5 / 1`` rather than as the literal ``5``.
    """
    result = convert_sympy_expression_to_expression(sympy.Integer(5))

    assert result.is_structurally_equivalent(LiteralExpression(5))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("1.5", id="one_and_a_half"),
        pytest.param(".5", id="no_integer_part"),
    ],
)
def test_binary_exact_decimal_string_literal_round_trips_through_sympy_unchanged(
    text: str,
) -> None:
    """Test a decimal string literal a binary float equals survives the round trip.

    Lowering reads the text as an exact rational, and lifting writes that
    rational back as exact decimal text because a binary ``float`` equals
    it, so the round trip lands in the float-decimal bucket it started in
    rather than collapsing into the float-binary one.
    """
    literal = LiteralExpression(text)

    result = convert_sympy_expression_to_expression(
        convert_expression_to_sympy_expression(literal)
    )

    assert result.is_structurally_equivalent(literal)
    assert isinstance(result, LiteralExpression)
    assert type(result.value) is str


@pytest.mark.parametrize(
    "text, expected_numerator, expected_denominator",
    [
        pytest.param("0.1", 1, 10, id="one_tenth"),
        pytest.param("0.075", 3, 40, id="three_decimal_places"),
    ],
)
def test_non_binary_decimal_string_literal_round_trips_as_an_equal_divide(
    text: str, expected_numerator: int, expected_denominator: int
) -> None:
    """Test a decimal string literal no binary float equals comes back as a `DIVIDE`.

    Lowering reads ``"0.1"`` as exactly one tenth, and lifting writes a
    rational no binary ``float`` equals as the quotient of its numerator
    and denominator. The form changes but the value does not: the
    quotient lowers to the same rational the text did.
    """
    literal = LiteralExpression(text)

    result = convert_sympy_expression_to_expression(
        convert_expression_to_sympy_expression(literal)
    )

    assert result.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.DIVIDE,
            LiteralExpression(expected_numerator),
            LiteralExpression(expected_denominator),
        )
    )
    assert convert_expression_to_sympy_expression(
        result
    ) == convert_expression_to_sympy_expression(literal)


@pytest.mark.parametrize(
    "text, expected_integer",
    [
        pytest.param("2.0", 2, id="trailing_zero"),
        pytest.param("2.", 2, id="no_fractional_digits"),
        pytest.param("0.0", 0, id="zero"),
    ],
)
def test_whole_valued_decimal_string_literal_lifts_into_the_integer_bucket(
    text: str, expected_integer: int
) -> None:
    """Test a float-grammar string with no fractional part comes back as an integer.

    ``sympy.Rational`` normalizes a unit denominator away, so ``"2.0"``
    reaches SymPy as ``sympy.Integer(2)``, indistinguishable from the
    literal ``2``. The lifter sees only that value, so the round trip
    lands in the integer bucket. The value is preserved exactly; only the
    bucket moves.
    """
    literal = LiteralExpression(text)

    result = convert_sympy_expression_to_expression(
        convert_expression_to_sympy_expression(literal)
    )

    assert result.is_structurally_equivalent(LiteralExpression(expected_integer))
    assert not result.is_structurally_equivalent(literal)


@pytest.mark.parametrize(
    "rational",
    [
        pytest.param(sympy.Rational(1, 2), id="one_half"),
        pytest.param(sympy.Rational(-7, 2), id="negative_three_and_a_half"),
        pytest.param(sympy.Rational(2**53 - 1, 2), id="largest_half_a_float_holds"),
        pytest.param(sympy.Rational(1, 2**100), id="hundred_binary_places"),
        pytest.param(sympy.Rational(1, 2**1074), id="least_subnormal"),
        pytest.param(sympy.Rational(1, 10), id="one_tenth"),
        pytest.param(sympy.Rational(-1, 10), id="negative_one_tenth"),
        pytest.param(sympy.Rational(3, 40), id="mixed_two_and_five_factors"),
        pytest.param(sympy.Rational(7, 5), id="seven_fifths"),
        pytest.param(sympy.Rational(1, 3), id="one_third"),
    ],
)
def test_numpy_evaluator_reads_each_rational_lift_as_the_nearest_float(
    rational: sympy.Rational,
) -> None:
    """Test the NumPy evaluator reads each tabled rational's lift as the nearest float.

    The evaluator refuses a decimal string literal no binary ``float``
    equals rather than round it, and the lift emits decimal text only
    when one does, so every string literal the lift produces is one the
    evaluator reads exactly. Every other rational lifts as a ``DIVIDE`` of
    integers, which the evaluator computes by true division. A rational
    whose denominator no ``float`` can hold, such as ``1/2**1075``, is
    outside this table: true division overflows on it.
    """
    pytest.importorskip("numpy")
    lifted = convert_sympy_expression_to_expression(rational)

    value = evaluate_expression_with_numpy(lifted, {})

    assert value == float(rational)


def test_simplify_divide_by_a_literal_yields_the_exact_half() -> None:
    """Test `x / 2` simplifies rather than raising on SymPy's `Half`.

    SymPy folds the quotient to ``Half * x``, and lifting ``Half``
    exactly is what makes ordinary division by a literal survive
    simplification at all.
    """
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, IdentifierExpression(x), LiteralExpression(2)
    )

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.MULTIPLY,
            LiteralExpression("0.5"),
            IdentifierExpression(x),
        )
    )


def test_simplify_ground_quotient_yields_the_exact_decimal_value() -> None:
    """Test `7 / 2` simplifies to exactly three and a half."""
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(7), LiteralExpression(2)
    )

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(LiteralExpression("3.5"))


def test_simplify_repeating_ground_quotient_yields_a_divide() -> None:
    """Test `1 / 3` simplifies to a `DIVIDE` rather than raising.

    One third has no exact literal form, so the residual quotient is the
    exact answer; refusing to lift it would make simplification partial
    over ordinary division.
    """
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(1), LiteralExpression(3)
    )

    result = simplify_expression(expression)

    assert result.is_structurally_equivalent(
        BinaryExpression(
            BinaryOperation.DIVIDE, LiteralExpression(1), LiteralExpression(3)
        )
    )


def test_simplify_rational_coefficient_equation_evaluates_with_numpy() -> None:
    """Test simplifying `21 == 15 * x` leaves a form NumPy decides at `x = 0`.

    SymPy solves the equation to ``x == 7/5``. Seven fifths has finite
    decimal text, but no binary ``float`` equals it, so it lifts as a
    ``DIVIDE`` the NumPy evaluator computes rather than as a decimal
    string literal the evaluator would refuse.
    """
    pytest.importorskip("numpy")
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL,
        LiteralExpression(21),
        BinaryExpression(
            BinaryOperation.MULTIPLY, LiteralExpression(15), IdentifierExpression(x)
        ),
    )

    simplified = simplify_expression(expression)

    assert bool(evaluate_expression_with_numpy(simplified, {x: 0})) is False


def test_simplify_quotient_by_zero_raises_the_complex_infinity_error() -> None:
    """Test `1 / 0` is refused by name rather than as an unsupported node type.

    SymPy folds the quotient to ``zoo``, its directionless complex
    infinity, which no expression denotes. Reporting it as an unlearned
    node kind would leave a caller unable to tell an ill-defined quotient
    from a lifting arm nobody has written yet.
    """
    expression = BinaryExpression(
        BinaryOperation.DIVIDE, LiteralExpression(1), LiteralExpression(0)
    )

    with pytest.raises(PassExecutionError) as exception_info:
        simplify_expression(expression)

    cause = exception_info.value.__cause__
    assert isinstance(cause, ComplexInfinityLiftError)
    assert "Unsupported expression type" not in str(exception_info.value)
    assert "zoo" in str(cause)


def test_complex_infinity_is_refused_by_the_lifter_directly() -> None:
    """Test lifting `sympy.zoo` on its own raises the named error."""
    with pytest.raises(PassExecutionError) as exception_info:
        convert_sympy_expression_to_expression(sympy.zoo)

    assert isinstance(exception_info.value.__cause__, ComplexInfinityLiftError)


# =============================================================================
# Native lookup tables are read-only
# =============================================================================


@pytest.mark.parametrize(
    "table",
    [_NATIVE_FUNCTION_LOWER, _NATIVE_CONSTANT_LOWER, _NATIVE_CONSTANT_LIFT],
    ids=["function-lower", "constant-lower", "constant-lift"],
)
def test_native_lookup_table_item_assignment_raises_type_error(
    table: Mapping[Any, Any],
) -> None:
    """Test assigning to an existing key in a native lookup table raises TypeError."""
    mutable_table = cast(MutableMapping[Any, Any], table)
    existing_key = next(iter(table))

    with pytest.raises(TypeError):
        mutable_table[existing_key] = table[existing_key]


# =============================================================================
# `environment`/`replacements` accept any `Mapping`, not only `dict`
# =============================================================================


def test_sympy_simplify_expression_accepts_an_immutabledict_environment() -> None:
    """Test the bridge's own `simplify_expression` accepts an `immutabledict`."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.ADD, IdentifierExpression(x), LiteralExpression(1)
    )
    environment = immutabledict({x: LiteralExpression(2)})

    result = sympy_simplify_expression(expression, environment)

    assert isinstance(result, LiteralExpression)
    assert result.value == 3


def test_substitute_sympy_expression_variables_accepts_an_immutabledict() -> None:
    """Test substituting an `immutabledict` environment folds to a scalar."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    sympy_expression = sympy.Symbol("x_0") + sympy.Symbol("y_1")
    substitutions = immutabledict({x: LiteralExpression(5), y: LiteralExpression(10)})

    assert substitute_sympy_expression_variables(sympy_expression, substitutions) == 15


def test_sympy_variable_substitution_pass_snapshots_replacements_at_construction() -> (
    None
):
    """Test the pass's substitution is unaffected by mutating the caller's dict.

    Builds the pass from a plain, still-mutable ``dict`` mapping the
    SymPy symbol ``x`` to ``1``, then rebinds ``x`` to ``2`` in that same
    dict after construction. Running the pass on ``x`` must still give
    ``1``, guarding against it aliasing the caller's dict instead of
    snapshotting it.
    """
    x = sympy.Symbol("x")
    replacements = {x: sympy.Integer(1)}
    pass_instance = SympyVariableSubstitutionPass(replacements)

    replacements[x] = sympy.Integer(2)

    result = pass_instance(x)

    assert result == sympy.Integer(1)
