"""Hypothesis property tests for ``inline_functions`` (P8).

Covers the expression-bodied ``RegisteredFunction`` builtins registered in
``fhy_core.symbolic.expression.builtins`` (``relu``, ``clamp``,
``clamp_symmetric``, ``leaky_relu``, ``sigmoid``, ``silu``, ``gelu``,
``abs``, ``max``, ``min``, ``sign``, ``xor``, ``nand``, ``nor``,
``implies``, ``iff``): inlining removes every call to one of them (even
when nested two levels deep), inlining is idempotent, and the inlined
tree evaluates identically to a reference table transcribed from each
function's registered formula. ``gelu`` is excluded from the evaluation
law: it inlines to a call to the native ``erf``, which has no NumPy
lowering (``fhy_core.symbolic.expression.passes.numpy`` module
docstring), a documented limitation the NumPy evaluator's example tests
pin directly.
"""

import pytest

pytest.importorskip("hypothesis")

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.expression import (
    CallExpression,
    Expression,
    LiteralExpression,
    NativeFunction,
    RegisteredFunction,
    call,
    evaluate_expression_with_numpy,
    get_registered_entry,
    inline_functions,
)

pytestmark = pytest.mark.property

np = pytest.importorskip("numpy")

# =============================================================================
# Reference table: transcribed from each function's registered formula in
# src/fhy_core/symbolic/expression/builtins.py
# =============================================================================


def _reference_max(a: float, b: float) -> float:
    return a if a > b else b


def _reference_min(a: float, b: float) -> float:
    return a if a < b else b


def _reference_abs(x: float) -> float:
    return x if x >= 0.0 else -x


def _reference_sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


def _reference_clamp(x: float, lo: float, hi: float) -> float:
    return _reference_min(_reference_max(x, lo), hi)


def _reference_clamp_symmetric(x: float, bound: float) -> float:
    return _reference_clamp(x, -bound, bound)


def _reference_relu(x: float) -> float:
    return _reference_max(x, 0)


def _reference_leaky_relu(x: float, slope: float) -> float:
    return x if x > 0.0 else x * slope


def _reference_xor(a: bool, b: bool) -> bool:
    return (a or b) and not (a and b)


def _reference_nand(a: bool, b: bool) -> bool:
    return not (a and b)


def _reference_nor(a: bool, b: bool) -> bool:
    return not (a or b)


def _reference_implies(a: bool, b: bool) -> bool:
    return (not a) or b


def _reference_iff(a: bool, b: bool) -> bool:
    return a == b


def _reference_sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _reference_silu(x: float) -> float:
    return x * _reference_sigmoid(x)


def _reference_gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


_REFERENCE_FUNCTIONS: Final[dict[str, Callable[..., float | int | bool]]] = {
    "max": _reference_max,
    "min": _reference_min,
    "abs": _reference_abs,
    "sign": _reference_sign,
    "clamp": _reference_clamp,
    "clamp_symmetric": _reference_clamp_symmetric,
    "relu": _reference_relu,
    "leaky_relu": _reference_leaky_relu,
    "xor": _reference_xor,
    "nand": _reference_nand,
    "nor": _reference_nor,
    "implies": _reference_implies,
    "iff": _reference_iff,
    "sigmoid": _reference_sigmoid,
    "silu": _reference_silu,
    "gelu": _reference_gelu,
}

# Declared parameter arity for every expression-bodied builtin (matches
# `_REAL_PARAMS_*`/`_BOOL_PARAMS_2` in builtins.py).
_REAL_PARAM_ARITIES: Final[dict[str, int]] = {
    "max": 2,
    "min": 2,
    "abs": 1,
    "sign": 1,
    "clamp": 3,
    "clamp_symmetric": 2,
    "relu": 1,
    "leaky_relu": 2,
    "sigmoid": 1,
    "silu": 1,
    "gelu": 1,
}
_BOOL_PARAM_ARITIES: Final[dict[str, int]] = {
    "xor": 2,
    "nand": 2,
    "nor": 2,
    "implies": 2,
    "iff": 2,
}

_REAL_FUNCTION_NAMES: Final[tuple[str, ...]] = tuple(_REAL_PARAM_ARITIES)
_BOOL_FUNCTION_NAMES: Final[tuple[str, ...]] = tuple(_BOOL_PARAM_ARITIES)

# `gelu` inlines to a call to the native `erf`, which
# `evaluate_expression_with_numpy` cannot lower (module docstring); excluded
# from the evaluation law. The example tests of the NumPy evaluator pin
# that limitation.
_NUMPY_EVALUABLE_REAL_FUNCTION_NAMES: Final[tuple[str, ...]] = tuple(
    name for name in _REAL_FUNCTION_NAMES if name != "gelu"
)


def _evaluate_reference(expression: Expression) -> float | int | bool:
    """Evaluate a literal-and-calls-only tree with the reference table.

    Every expression this module draws is either a finite-float or
    boolean ``LiteralExpression`` leaf, or a ``CallExpression`` to one of
    ``_REFERENCE_FUNCTIONS`` (nested at most two levels deep), so this
    recursive walk needs nothing more.
    """
    if isinstance(expression, LiteralExpression):
        value = expression.value
        assert isinstance(value, (bool, float))
        return value
    assert isinstance(expression, CallExpression)
    arguments = [_evaluate_reference(argument) for argument in expression.arguments]
    return _REFERENCE_FUNCTIONS[expression.function_name](*arguments)


# =============================================================================
# Strategies: flat and two-level (nested) calls to the builtins
# =============================================================================


def _build_real_literal_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy for finite float literals in [-8, 8]."""
    return st.floats(
        min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False
    ).map(LiteralExpression)


def _build_bool_literal_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy for boolean literals."""
    return st.booleans().map(LiteralExpression)


@st.composite
def _draw_flat_call(
    draw: st.DrawFn,
    names: Sequence[str],
    arities: Mapping[str, int],
    leaf_strategy: st.SearchStrategy[Expression],
) -> Expression:
    """Draw a single call to one of ``names`` with literal-leaf arguments."""
    name = draw(st.sampled_from(names))
    arguments = tuple(draw(leaf_strategy) for _ in range(arities[name]))
    return call(name, *arguments)


@st.composite
def _draw_nested_call(
    draw: st.DrawFn,
    names: Sequence[str],
    arities: Mapping[str, int],
    leaf_strategy: st.SearchStrategy[Expression],
) -> Expression:
    """Draw a call to one of ``names`` with one argument itself such a call.

    Both the outer and the inner call are drawn from the same family
    (``names``/``arities``), so every argument position -- outer and
    inner -- stays sort-correct by construction.
    """
    outer_name = draw(st.sampled_from(names))
    arity = arities[outer_name]
    replaced_index = draw(st.integers(min_value=0, max_value=arity - 1))
    inner_call = draw(_draw_flat_call(names, arities, leaf_strategy))
    arguments = tuple(
        inner_call if index == replaced_index else draw(leaf_strategy)
        for index in range(arity)
    )
    return call(outer_name, *arguments)


def _build_builtin_call_strategy(
    names: Sequence[str],
    arities: Mapping[str, int],
    leaf_strategy: st.SearchStrategy[Expression],
) -> st.SearchStrategy[Expression]:
    """Return a strategy for a flat or two-level-nested call over ``names``."""
    return st.one_of(
        _draw_flat_call(names, arities, leaf_strategy),
        _draw_nested_call(names, arities, leaf_strategy),
    )


def _build_any_builtin_call_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy over every expression-bodied builtin, real or boolean."""
    return st.one_of(
        _build_builtin_call_strategy(
            _REAL_FUNCTION_NAMES, _REAL_PARAM_ARITIES, _build_real_literal_strategy()
        ),
        _build_builtin_call_strategy(
            _BOOL_FUNCTION_NAMES, _BOOL_PARAM_ARITIES, _build_bool_literal_strategy()
        ),
    )


def _build_evaluable_real_call_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy over the real-domain builtins NumPy can evaluate."""
    return _build_builtin_call_strategy(
        _NUMPY_EVALUABLE_REAL_FUNCTION_NAMES,
        _REAL_PARAM_ARITIES,
        _build_real_literal_strategy(),
    )


def _build_bool_call_strategy() -> st.SearchStrategy[Expression]:
    """Return a strategy over the boolean-domain builtins."""
    return _build_builtin_call_strategy(
        _BOOL_FUNCTION_NAMES, _BOOL_PARAM_ARITIES, _build_bool_literal_strategy()
    )


def _collect_call_expressions(expression: Expression) -> list[CallExpression]:
    """Return every ``CallExpression`` node in ``expression``, in visit order."""
    calls: list[CallExpression] = []
    if isinstance(expression, CallExpression):
        calls.append(expression)
    for child in expression.get_visit_children():
        calls.extend(_collect_call_expressions(child))
    return calls


# =============================================================================
# P8a: inlining removes every call to a RegisteredFunction
# =============================================================================


@given(_build_any_builtin_call_strategy())
def test_inline_functions_leaves_no_registered_function_call(
    expression: Expression,
) -> None:
    """Test the inlined tree holds no call whose target is a RegisteredFunction.

    Covers both a flat call and a two-level nested call (an
    expression-bodied call as an argument of another).
    """
    inlined = inline_functions(expression)

    for call_expression in _collect_call_expressions(inlined):
        entry = get_registered_entry(call_expression.function_name)
        assert not isinstance(entry, RegisteredFunction)
        assert isinstance(entry, NativeFunction)


# =============================================================================
# P8b: the inlined tree evaluates identically to the reference table
# =============================================================================


@given(_build_evaluable_real_call_strategy())
def test_inline_functions_evaluates_like_the_reference_table_for_real_builtins(
    expression: Expression,
) -> None:
    """Test NumPy evaluation of the inlined real-domain builtin matches ``math``.

    Oracle: ``_REFERENCE_FUNCTIONS``, transcribed from each function's
    registered formula in builtins.py.
    """
    expected = _evaluate_reference(expression)

    result = evaluate_expression_with_numpy(inline_functions(expression), {})

    assert math.isclose(float(result), float(expected), rel_tol=1e-9, abs_tol=1e-9)


@given(_build_bool_call_strategy())
def test_inline_functions_evaluates_like_the_reference_table_for_bool_builtins(
    expression: Expression,
) -> None:
    """Test NumPy evaluation of the inlined boolean builtin matches the reference table.

    Boolean results are compared exactly, not with a tolerance.
    """
    expected = _evaluate_reference(expression)

    result = evaluate_expression_with_numpy(inline_functions(expression), {})

    assert bool(result) == bool(expected)


# =============================================================================
# P8c: inline_functions is idempotent
# =============================================================================


@given(_build_any_builtin_call_strategy())
def test_inline_functions_is_idempotent(expression: Expression) -> None:
    """Test inlining an already-inlined tree changes nothing structurally."""
    inlined_once = inline_functions(expression)

    inlined_twice = inline_functions(inlined_once)

    assert inlined_twice.is_structurally_equivalent(inlined_once)
