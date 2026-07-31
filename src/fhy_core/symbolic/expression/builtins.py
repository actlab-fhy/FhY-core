"""Built-in functions and constants pre-registered at module import.

Importing this module is a side-effecting registration: every member of
:data:`BUILTIN_FUNCTIONS` and :data:`BUILTIN_CONSTANTS` is bound into
the process-wide registry. Adding a new built-in is one field on the
relevant ``TypedDict`` and one entry in the corresponding mapping.

Numerical native implementations are backed by Python's ``math``
module. Their results are reproducible within a single run but the
final bits may differ across operating systems and CPU families.
Callers requiring cross-platform exact reproducibility must not rely
on the low-order bits of these results.

Expression-construction convention: ``Identifier`` itself does not
overload arithmetic, comparison, or logical operators, so wrapping in
``IdentifierExpression(...)`` is required when the bare ``Identifier``
is the operand dispatching the Python operator (for example
``x * y``, ``x > y``, ``-x``). By contrast, expression-side operators
and helpers still auto-lift bare ``Identifier`` operands in
non-dispatching positions, so forms like ``LiteralExpression(0.5) * x``
and arguments passed directly to ``call(...)`` or ``piecewise(...)``
need no wrapping. Removing the asymmetry by also wrapping those
operands is harmless but adds noise; introducing it by using a bare
``Identifier`` as the dispatching operand is a ``TypeError`` at
construction time.
"""

__all__ = [
    "BUILTIN_CONSTANTS",
    "BUILTIN_FUNCTIONS",
    "BuiltinConstants",
    "BuiltinFunctions",
]

import math
from typing import TypedDict, cast

from fhy_core.identifier import Identifier

from .core import (
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
    UnaryOperation,
    call,
    logical_and,
    logical_or,
    make_binary_expression,
    make_unary_expression,
    piecewise,
)
from .registry import (
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    register_function,
    register_native_constant,
    register_native_function,
)
from .sort import FunctionSort


class BuiltinFunctions(TypedDict):
    """Mapping of built-in function names to their registry entries.

    Composed entries are expression-bodied :class:`RegisteredFunction`
    instances; native entries are :class:`NativeFunction` instances
    bound to ``math`` callables.
    """

    # Composable utilities.
    max: RegisteredFunction
    min: RegisteredFunction
    abs: RegisteredFunction
    sign: RegisteredFunction
    clamp: RegisteredFunction
    clamp_symmetric: RegisteredFunction
    relu: RegisteredFunction
    leaky_relu: RegisteredFunction
    xor: RegisteredFunction
    nand: RegisteredFunction
    nor: RegisteredFunction
    implies: RegisteredFunction
    iff: RegisteredFunction
    sigmoid: RegisteredFunction
    silu: RegisteredFunction
    gelu: RegisteredFunction

    # Native math functions.
    exp: NativeFunction
    exp2: NativeFunction
    log: NativeFunction
    log2: NativeFunction
    log10: NativeFunction
    sqrt: NativeFunction
    sin: NativeFunction
    cos: NativeFunction
    tan: NativeFunction
    arcsin: NativeFunction
    arccos: NativeFunction
    arctan: NativeFunction
    sinh: NativeFunction
    cosh: NativeFunction
    tanh: NativeFunction
    erf: NativeFunction
    round: NativeFunction
    floor: NativeFunction
    ceil: NativeFunction


class BuiltinConstants(TypedDict):
    """Mapping of built-in constant names to their registry entries."""

    pi: NativeConstant
    e: NativeConstant
    inf: NativeConstant
    nan: NativeConstant


_REAL_PARAMS_1: list[FunctionSort] = [FunctionSort.REAL]
_REAL_PARAMS_2: list[FunctionSort] = [FunctionSort.REAL, FunctionSort.REAL]
_REAL_PARAMS_3: list[FunctionSort] = [
    FunctionSort.REAL,
    FunctionSort.REAL,
    FunctionSort.REAL,
]
_BOOL_PARAMS_2: list[FunctionSort] = [FunctionSort.BOOL, FunctionSort.BOOL]


def _exp2(value: int | float) -> float:
    return math.pow(2, value)


_NATIVE_FUNCTION_SPECS: tuple[tuple[str, FunctionSort, object], ...] = (
    ("exp", FunctionSort.REAL, math.exp),
    ("exp2", FunctionSort.REAL, _exp2),
    ("log", FunctionSort.REAL, math.log),
    ("log2", FunctionSort.REAL, math.log2),
    ("log10", FunctionSort.REAL, math.log10),
    ("sqrt", FunctionSort.REAL, math.sqrt),
    ("sin", FunctionSort.REAL, math.sin),
    ("cos", FunctionSort.REAL, math.cos),
    ("tan", FunctionSort.REAL, math.tan),
    ("arcsin", FunctionSort.REAL, math.asin),
    ("arccos", FunctionSort.REAL, math.acos),
    ("arctan", FunctionSort.REAL, math.atan),
    ("sinh", FunctionSort.REAL, math.sinh),
    ("cosh", FunctionSort.REAL, math.cosh),
    ("tanh", FunctionSort.REAL, math.tanh),
    ("erf", FunctionSort.REAL, math.erf),
    ("round", FunctionSort.INT, round),
    ("floor", FunctionSort.INT, math.floor),
    ("ceil", FunctionSort.INT, math.ceil),
)


def _register_real_param_native(
    name: str, result_sort: FunctionSort, implementation: object
) -> NativeFunction:
    """Register a unary native function with a ``REAL``-sort parameter."""
    return register_native_function(
        name,
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=result_sort,
        implementation=implementation,  # type: ignore[arg-type]
    )


def _register_max() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "max",
        parameters=[a, b],
        parameter_sorts=_REAL_PARAMS_2,
        result_sort=FunctionSort.REAL,
        body=piecewise((IdentifierExpression(a) > b, a), otherwise=b),
    )


def _register_min() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "min",
        parameters=[a, b],
        parameter_sorts=_REAL_PARAMS_2,
        result_sort=FunctionSort.REAL,
        body=piecewise((IdentifierExpression(a) < b, a), otherwise=b),
    )


def _register_abs() -> RegisteredFunction:
    x = Identifier("x")
    x_expression = IdentifierExpression(x)
    return register_function(
        "abs",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.REAL,
        body=piecewise((x_expression >= 0.0, x), otherwise=-x_expression),
    )


def _register_sign() -> RegisteredFunction:
    x = Identifier("x")
    x_expression = IdentifierExpression(x)
    return register_function(
        "sign",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.INT,
        body=piecewise((x_expression > 0.0, 1), (x_expression < 0.0, -1), otherwise=0),
    )


def _register_clamp() -> RegisteredFunction:
    x = Identifier("x")
    lo = Identifier("lo")
    hi = Identifier("hi")
    return register_function(
        "clamp",
        parameters=[x, lo, hi],
        parameter_sorts=_REAL_PARAMS_3,
        result_sort=FunctionSort.REAL,
        body=call("min", call("max", x, lo), hi),
    )


def _register_clamp_symmetric() -> RegisteredFunction:
    x = Identifier("x")
    bound = Identifier("bound")
    return register_function(
        "clamp_symmetric",
        parameters=[x, bound],
        parameter_sorts=_REAL_PARAMS_2,
        result_sort=FunctionSort.REAL,
        body=call("clamp", x, -IdentifierExpression(bound), bound),
    )


def _register_relu() -> RegisteredFunction:
    x = Identifier("x")
    return register_function(
        "relu",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.REAL,
        body=call("max", x, 0),
    )


def _register_leaky_relu() -> RegisteredFunction:
    x = Identifier("x")
    slope = Identifier("slope")
    x_expression = IdentifierExpression(x)
    return register_function(
        "leaky_relu",
        parameters=[x, slope],
        parameter_sorts=_REAL_PARAMS_2,
        result_sort=FunctionSort.REAL,
        body=piecewise((x_expression > 0.0, x), otherwise=x_expression * slope),
    )


def _register_xor() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "xor",
        parameters=[a, b],
        parameter_sorts=_BOOL_PARAMS_2,
        result_sort=FunctionSort.BOOL,
        body=logical_and(
            logical_or(a, b),
            make_unary_expression(UnaryOperation.LOGICAL_NOT, logical_and(a, b)),
        ),
    )


def _register_nand() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "nand",
        parameters=[a, b],
        parameter_sorts=_BOOL_PARAMS_2,
        result_sort=FunctionSort.BOOL,
        body=make_unary_expression(UnaryOperation.LOGICAL_NOT, logical_and(a, b)),
    )


def _register_nor() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "nor",
        parameters=[a, b],
        parameter_sorts=_BOOL_PARAMS_2,
        result_sort=FunctionSort.BOOL,
        body=make_unary_expression(UnaryOperation.LOGICAL_NOT, logical_or(a, b)),
    )


def _register_implies() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "implies",
        parameters=[a, b],
        parameter_sorts=_BOOL_PARAMS_2,
        result_sort=FunctionSort.BOOL,
        body=logical_or(
            make_unary_expression(UnaryOperation.LOGICAL_NOT, a),
            b,
        ),
    )


def _register_iff() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    return register_function(
        "iff",
        parameters=[a, b],
        parameter_sorts=_BOOL_PARAMS_2,
        result_sort=FunctionSort.BOOL,
        body=make_binary_expression(BinaryOperation.EQUAL, a, b),
    )


def _register_sigmoid() -> RegisteredFunction:
    x = Identifier("x")
    return register_function(
        "sigmoid",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.REAL,
        body=LiteralExpression(1.0)
        / (LiteralExpression(1.0) + call("exp", -IdentifierExpression(x))),
    )


def _register_silu() -> RegisteredFunction:
    x = Identifier("x")
    return register_function(
        "silu",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(x) * call("sigmoid", x),
    )


def _register_gelu() -> RegisteredFunction:
    """Register ``gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))``."""
    x = Identifier("x")
    return register_function(
        "gelu",
        parameters=[x],
        parameter_sorts=_REAL_PARAMS_1,
        result_sort=FunctionSort.REAL,
        body=LiteralExpression(0.5)
        * x
        * (
            LiteralExpression(1.0)
            + call(
                "erf", IdentifierExpression(x) / call("sqrt", LiteralExpression(2.0))
            )
        ),
    )


def _register_all_natives() -> dict[str, NativeFunction]:
    return {
        name: _register_real_param_native(name, result_sort, implementation)
        for name, result_sort, implementation in _NATIVE_FUNCTION_SPECS
    }


def _register_all_constants() -> BuiltinConstants:
    return {
        "pi": register_native_constant("pi", sort=FunctionSort.REAL, value=math.pi),
        "e": register_native_constant("e", sort=FunctionSort.REAL, value=math.e),
        "inf": register_native_constant("inf", sort=FunctionSort.REAL, value=math.inf),
        "nan": register_native_constant("nan", sort=FunctionSort.REAL, value=math.nan),
    }


# Constants are registered first so that composed function bodies can
# reference them without triggering the captured-free-identifier check.
BUILTIN_CONSTANTS: BuiltinConstants = _register_all_constants()

_BUILTIN_NATIVE_FUNCTIONS: dict[str, NativeFunction] = _register_all_natives()


def _build_builtin_functions() -> BuiltinFunctions:
    """Assemble the ``BUILTIN_FUNCTIONS`` mapping from the registered entries."""
    composed = {
        "max": _register_max(),
        "min": _register_min(),
        "abs": _register_abs(),
        "sign": _register_sign(),
        "clamp": _register_clamp(),
        "clamp_symmetric": _register_clamp_symmetric(),
        "relu": _register_relu(),
        "leaky_relu": _register_leaky_relu(),
        "xor": _register_xor(),
        "nand": _register_nand(),
        "nor": _register_nor(),
        "implies": _register_implies(),
        "iff": _register_iff(),
        "sigmoid": _register_sigmoid(),
        "silu": _register_silu(),
        "gelu": _register_gelu(),
    }
    merged: dict[str, RegisteredFunction | NativeFunction] = {
        **composed,
        **_BUILTIN_NATIVE_FUNCTIONS,
    }
    return cast(BuiltinFunctions, merged)


BUILTIN_FUNCTIONS: BuiltinFunctions = _build_builtin_functions()
