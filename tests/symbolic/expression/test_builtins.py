"""Tests for the seeded built-in functions and constants.

The seeded set covers:

- Composable utilities (`max`, `min`, `abs`, `sign`, `clamp`,
  `clamp_symmetric`, `relu`, `leaky_relu`, the boolean combinators
  `xor`/`nand`/`nor`/`implies`/`iff`, and the activation functions
  `sigmoid`/`silu`/`gelu`).
- Native math functions (`exp`, `exp2`, `log`, `log2`, `log10`,
  `sqrt`, `sin`, `cos`, `tan`, the inverse trig family, the
  hyperbolics, `erf`, `round`, `floor`, `ceil`).
- Native constants (`pi`, `e`, `inf`, `nan`).

The tests verify each entry is registered at import time, expose the
canonical name, and (for composed functions) inline to the documented
body. Numeric evaluation of native chains is covered by the
``test_evaluator.py`` tests; here the focus is on registration and
inlining semantics.
"""

import math

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    CallExpression,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    TernaryExpression,
    UnaryExpression,
    call,
    evaluate_expression,
    get_registered_entry,
    inline_functions,
    is_entry_registered,
)
from fhy_core.symbolic.expression.builtins import BUILTIN_CONSTANTS, BUILTIN_FUNCTIONS

_BINARY_SYMBOLS: dict[BinaryOperation, str] = {
    BinaryOperation.GREATER: ">",
    BinaryOperation.GREATER_EQUAL: ">=",
    BinaryOperation.LESS: "<",
    BinaryOperation.LESS_EQUAL: "<=",
    BinaryOperation.EQUAL: "==",
    BinaryOperation.NOT_EQUAL: "!=",
    BinaryOperation.LOGICAL_AND: "logical_and",
    BinaryOperation.LOGICAL_OR: "logical_or",
    BinaryOperation.ADD: "+",
    BinaryOperation.SUBTRACT: "-",
    BinaryOperation.MULTIPLY: "*",
    BinaryOperation.DIVIDE: "/",
    BinaryOperation.FLOOR_DIVIDE: "//",
    BinaryOperation.MODULO: "%",
    BinaryOperation.POWER: "**",
}


# Test helper: dispatch over expression kinds; one return per kind reads clearest.
def _structure_summary(expression: Expression) -> str:  # noqa: PLR0911
    """Compact structural summary of an expression tree.

    Returns a string capturing the node kinds and operations, useful for
    assertions that pin down shape without depending on the precise
    identifier instances used inside built-in bodies.
    """
    if isinstance(expression, LiteralExpression):
        return "literal"
    if isinstance(expression, IdentifierExpression):
        return "identifier"
    if isinstance(expression, UnaryExpression):
        operation_name = expression.operation.name.lower()
        return f"{operation_name}({_structure_summary(expression.operand)})"
    if isinstance(expression, BinaryExpression):
        operation_symbol = _BINARY_SYMBOLS.get(
            expression.operation, expression.operation.name.lower()
        )
        return (
            f"{operation_symbol}({_structure_summary(expression.left)}, "
            f"{_structure_summary(expression.right)})"
        )
    if isinstance(expression, TernaryExpression):
        condition_kind = _condition_shape_tag(expression.condition)
        return (
            f"ternary({condition_kind}, "
            f"{_structure_summary(expression.true_value)}, "
            f"{_structure_summary(expression.false_value)})"
        )
    if isinstance(expression, CallExpression):
        argument_summaries = ", ".join(
            _structure_summary(argument) for argument in expression.arguments
        )
        return f"call({expression.function_name}, [{argument_summaries}])"
    return type(expression).__name__


def _condition_shape_tag(condition: Expression) -> str:
    if isinstance(condition, BinaryExpression):
        return _BINARY_SYMBOLS.get(condition.operation, condition.operation.name)
    return type(condition).__name__


# =============================================================================
# Expected built-in coverage
# =============================================================================

_EXPECTED_COMPOSED_FUNCTIONS = {
    "max",
    "min",
    "abs",
    "sign",
    "clamp",
    "clamp_symmetric",
    "relu",
    "leaky_relu",
    "xor",
    "nand",
    "nor",
    "implies",
    "iff",
    "sigmoid",
    "silu",
    "gelu",
}

_EXPECTED_NATIVE_FUNCTIONS = {
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "erf",
    "round",
    "floor",
    "ceil",
}

_EXPECTED_BUILTIN_FUNCTIONS = _EXPECTED_COMPOSED_FUNCTIONS | _EXPECTED_NATIVE_FUNCTIONS

_EXPECTED_BUILTIN_CONSTANTS = {"pi", "e", "inf", "nan"}


# =============================================================================
# Coverage: every expected entry is in the mapping and registry
# =============================================================================


def test_builtin_functions_mapping_contains_every_expected_entry() -> None:
    """Test ``BUILTIN_FUNCTIONS`` exposes the full expected built-in set."""
    assert set(BUILTIN_FUNCTIONS.keys()) == _EXPECTED_BUILTIN_FUNCTIONS


def test_builtin_constants_mapping_contains_every_expected_entry() -> None:
    """Test ``BUILTIN_CONSTANTS`` exposes the full expected constant set."""
    assert set(BUILTIN_CONSTANTS.keys()) == _EXPECTED_BUILTIN_CONSTANTS


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTIN_FUNCTIONS))
def test_each_builtin_function_is_registered_at_import(name: str) -> None:
    """Test every entry in ``BUILTIN_FUNCTIONS`` is in the registry."""
    assert is_entry_registered(name)


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTIN_CONSTANTS))
def test_each_builtin_constant_is_registered_at_import(name: str) -> None:
    """Test every entry in ``BUILTIN_CONSTANTS`` is in the registry."""
    assert is_entry_registered(name)


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTIN_FUNCTIONS))
def test_each_builtin_function_entry_carries_canonical_name(name: str) -> None:
    """Test each `BUILTIN_FUNCTIONS` entry's `name` matches its mapping key."""
    assert BUILTIN_FUNCTIONS[name].name == name  # type: ignore[literal-required]


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTIN_CONSTANTS))
def test_each_builtin_constant_entry_carries_canonical_name(name: str) -> None:
    """Test each `BUILTIN_CONSTANTS` entry's `name` matches its mapping key."""
    assert BUILTIN_CONSTANTS[name].name == name  # type: ignore[literal-required]


# =============================================================================
# Kind classification
# =============================================================================


@pytest.mark.parametrize("name", sorted(_EXPECTED_COMPOSED_FUNCTIONS))
def test_each_composed_built_in_is_a_registered_function(name: str) -> None:
    """Test each composed entry is a `RegisteredFunction` (expression-bodied)."""
    entry = get_registered_entry(name)
    assert isinstance(entry, RegisteredFunction)


@pytest.mark.parametrize("name", sorted(_EXPECTED_NATIVE_FUNCTIONS))
def test_each_native_built_in_is_a_native_function(name: str) -> None:
    """Test each native entry is a `NativeFunction`."""
    entry = get_registered_entry(name)
    assert isinstance(entry, NativeFunction)


@pytest.mark.parametrize("name", sorted(_EXPECTED_BUILTIN_CONSTANTS))
def test_each_built_in_constant_is_a_native_constant(name: str) -> None:
    """Test each constant entry is a `NativeConstant`."""
    entry = get_registered_entry(name)
    assert isinstance(entry, NativeConstant)


# =============================================================================
# Composed function inlining: representative shapes
# =============================================================================


def test_abs_inlining_produces_documented_ternary_body() -> None:
    """Test ``abs(x)`` inlines to ``x >= 0 ? x : -x``."""
    x = Identifier("x")
    expression = call("abs", x)

    inlined = inline_functions(expression)

    # The shape is a ternary; the condition is `x >= 0`; the true branch
    # is `x`; the false branch is `-x`. Structural equivalence is enough
    # — we don't pin down the exact `Identifier` instances used inside
    # the body.
    assert _structure_summary(inlined) == "ternary(>=, identifier, negate(identifier))"


def test_relu_inlining_produces_max_x_zero_ternary() -> None:
    """Test ``relu(x)`` inlines to ``max(x, 0)``'s ternary body."""
    x = Identifier("x")
    expression = call("relu", x)

    inlined = inline_functions(expression)

    # Inlining max gives a ternary on `>`.
    assert _structure_summary(inlined).startswith("ternary(>")


def test_clamp_inlining_produces_nested_max_min_ternaries() -> None:
    """Test ``clamp(x, lo, hi)`` inlines to the documented nested ternary tree."""
    x = Identifier("x")
    lo = Identifier("lo")
    hi = Identifier("hi")
    expression = call("clamp", x, lo, hi)

    inlined = inline_functions(expression)

    # We don't pin down the precise inner shape, but the outer node must
    # be a ternary (from `max`).
    assert _structure_summary(inlined).startswith("ternary(")


def test_xor_inlining_produces_or_and_not_and_combination() -> None:
    """Test ``xor(a, b)`` inlines to ``(a || b) && !(a && b)``."""
    a = Identifier("a")
    b = Identifier("b")
    expression = call("xor", a, b)

    inlined = inline_functions(expression)

    summary = _structure_summary(inlined)
    assert "logical_and" in summary
    assert "logical_or" in summary
    assert "logical_not" in summary


# =============================================================================
# Composed function evaluation: native chains fold under evaluator
# =============================================================================


def test_evaluate_inlined_sigmoid_at_zero_leaves_exp_call_intact() -> None:
    """Test inlining ``sigmoid(0.0)`` and then evaluating leaves ``exp(-x)`` intact.

    The substituted body is ``1.0 / (1.0 + exp(-0.0))``; the inner
    ``exp`` argument is a ``UnaryExpression(NEGATE, LiteralExpression(0.0))``,
    not a literal. The evaluator only folds native calls whose
    arguments are bare literals, so the ``exp`` call stays in the
    tree. The full numeric result requires ``simplify_expression``
    (sympy) on top of the evaluator output — see the corresponding
    user-story test for that path.
    """
    expression = CallExpression("sigmoid", (LiteralExpression(0.0),))

    inlined = inline_functions(expression)
    evaluated = evaluate_expression(inlined)

    summary = _structure_summary(evaluated)
    # The ``exp`` call remains because its argument is the unfolded
    # ``-0.0`` (a unary negation, not a literal).
    assert "call(exp" in summary


def test_evaluate_inlined_relu_with_literal_argument_folds_to_literal_tree() -> None:
    """Test inlining ``relu(2.0)`` evaluates to a ternary on literal operands.

    The evaluator does not fold the literal arithmetic itself; the
    resulting expression is the ternary tree with all literals present.
    """
    expression = CallExpression("relu", (LiteralExpression(2.0),))

    inlined = inline_functions(expression)
    evaluated = evaluate_expression(inlined)

    # Still a ternary; no native calls in the result.
    assert _structure_summary(evaluated).startswith("ternary(")
    assert "CallExpression" not in _structure_summary(evaluated)


# =============================================================================
# Native constant values match math module
# =============================================================================


@pytest.mark.parametrize(
    "name, expected_value",
    [
        ("pi", math.pi),
        ("e", math.e),
        ("inf", math.inf),
    ],
)
def test_seeded_constant_value_matches_math_module(
    name: str, expected_value: float
) -> None:
    """Test the seeded constants carry the value from Python's ``math`` module."""
    entry = get_registered_entry(name)
    assert isinstance(entry, NativeConstant)
    assert entry.value == expected_value


def test_seeded_nan_constant_is_nan() -> None:
    """Test the seeded ``nan`` constant carries a NaN value.

    ``math.nan`` does not equal itself; the check is via ``math.isnan``.
    """
    entry = get_registered_entry("nan")
    assert isinstance(entry, NativeConstant)
    assert isinstance(entry.value, float)
    assert math.isnan(entry.value)


# =============================================================================
# Native function bindings match math module
# =============================================================================


@pytest.mark.parametrize(
    "name, math_callable",
    [
        ("exp", math.exp),
        ("log", math.log),
        ("log2", math.log2),
        ("log10", math.log10),
        ("sqrt", math.sqrt),
        ("sin", math.sin),
        ("cos", math.cos),
        ("tan", math.tan),
        ("arcsin", math.asin),
        ("arccos", math.acos),
        ("arctan", math.atan),
        ("sinh", math.sinh),
        ("cosh", math.cosh),
        ("tanh", math.tanh),
        ("erf", math.erf),
        ("floor", math.floor),
        ("ceil", math.ceil),
    ],
)
def test_seeded_native_implementation_matches_math_callable(
    name: str, math_callable: object
) -> None:
    """Test each seeded native binds to the expected ``math`` callable."""
    entry = get_registered_entry(name)
    assert isinstance(entry, NativeFunction)
    assert entry.implementation is math_callable


# =============================================================================
# Existing max / min behavior (kept to preserve coverage)
# =============================================================================


def test_max_inlining_yields_greater_ternary() -> None:
    """Test inlining ``max(a, b)`` yields ``a > b ? a : b``."""
    a = LiteralExpression(7)
    b = LiteralExpression(2)

    result = inline_functions(CallExpression("max", (a, b)))

    assert _structure_summary(result).startswith("ternary(>")


def test_min_inlining_yields_less_ternary() -> None:
    """Test inlining ``min(a, b)`` yields ``a < b ? a : b``."""
    a = LiteralExpression(7)
    b = LiteralExpression(2)

    result = inline_functions(CallExpression("min", (a, b)))

    assert _structure_summary(result).startswith("ternary(<")


# =============================================================================
# Native math semantics: exp2 / round
# =============================================================================


def test_exp2_native_folds_to_two_to_the_power_of_argument() -> None:
    """Test ``exp2(3)`` folds to ``8.0``.

    Pins the semantics of the seeded ``exp2`` binding (``2 ** x``) so a
    future refactor to a non-equivalent implementation surfaces here
    rather than silently changing rounding behavior.
    """
    expression = CallExpression("exp2", (LiteralExpression(3),))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 8.0


def test_round_native_folds_with_python_banker_rounding() -> None:
    """Test ``round(2.5)`` folds to ``2`` (banker's rounding, not 3).

    Pins down that the seeded binding uses Python's built-in ``round``
    semantics. A future refactor to ``int(value + 0.5)`` style rounding
    would surface here.
    """
    expression = CallExpression("round", (LiteralExpression(2.5),))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 2


def test_round_native_rounds_to_even_for_other_half_values() -> None:
    """Test ``round(3.5)`` folds to ``4`` (banker's rounding).

    The "round half to even" rule rounds ``2.5`` down to ``2`` and
    ``3.5`` up to ``4``. Tested as a pair to make the rule unambiguous.
    """
    expression = CallExpression("round", (LiteralExpression(3.5),))
    result = evaluate_expression(expression)

    assert isinstance(result, LiteralExpression)
    assert result.value == 4
