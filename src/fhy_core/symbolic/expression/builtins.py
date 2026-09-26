"""Built-in functions and constants.

The built-ins are the Rust core's catalogue
(``fhy_core::expression::builtins``): 16 composed functions defined by an
expression over their parameters, 19 native functions, and the constants
``pi``, ``e``, ``inf`` and ``nan``. Their names are reserved, their
signatures and bodies are the core's, and they are no user registry
entries: the registry's lookups resolve them first, through one entry
object per built-in, which :data:`BUILTIN_FUNCTIONS` and
:data:`BUILTIN_CONSTANTS` hold too. A composed built-in's entry is a
:class:`RegisteredFunction` over the catalogue's parameters and body, a
native one's a :class:`NativeFunction`, and a constant's a
:class:`NativeConstant` whose identifier has a fixed reserved id (``pi``
48, ``e`` 49, ``inf`` 50, ``nan`` 51), the same in every process.

The core computes no native function, so this module keeps the Python
callables that compute them, and importing it installs them: the
extension builds the built-ins' entries then, once. That the composed
bodies type-check against their declared sorts is pinned by
``tests/types/checking/test_builtin_bodies.py`` rather than checked on
every import.

Numerical native implementations are backed by Python's ``math``
module. Their results are reproducible within a single run but the
final bits may differ across operating systems and CPU families.
Callers requiring cross-platform exact reproducibility must not rely
on the low-order bits of these results.
"""

__all__ = [
    "BUILTIN_CONSTANTS",
    "BUILTIN_FUNCTIONS",
    "BuiltinConstants",
    "BuiltinFunctions",
]

import math
from collections.abc import Callable
from typing import cast

from immutabledict import immutabledict

from fhy_core import _rs
from fhy_core.utils.typed_dict import ReadOnly, TypedDict

# The composed built-ins' bodies are built from the expression classes of
# `core`, which register themselves with the extension when it is imported.
from . import core  # noqa: F401
from .registry import (
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    get_registered_entry,
)


class BuiltinFunctions(TypedDict):
    """Mapping of built-in function names to their registry entries.

    Composed entries are expression-bodied :class:`RegisteredFunction`
    instances; native entries are :class:`NativeFunction` instances
    bound to ``math`` callables.
    """

    # Composable utilities.
    max: ReadOnly[RegisteredFunction]
    min: ReadOnly[RegisteredFunction]
    abs: ReadOnly[RegisteredFunction]
    sign: ReadOnly[RegisteredFunction]
    clamp: ReadOnly[RegisteredFunction]
    clamp_symmetric: ReadOnly[RegisteredFunction]
    relu: ReadOnly[RegisteredFunction]
    leaky_relu: ReadOnly[RegisteredFunction]
    xor: ReadOnly[RegisteredFunction]
    nand: ReadOnly[RegisteredFunction]
    nor: ReadOnly[RegisteredFunction]
    implies: ReadOnly[RegisteredFunction]
    iff: ReadOnly[RegisteredFunction]
    sigmoid: ReadOnly[RegisteredFunction]
    silu: ReadOnly[RegisteredFunction]
    gelu: ReadOnly[RegisteredFunction]

    # Native math functions.
    exp: ReadOnly[NativeFunction]
    exp2: ReadOnly[NativeFunction]
    log: ReadOnly[NativeFunction]
    log2: ReadOnly[NativeFunction]
    log10: ReadOnly[NativeFunction]
    sqrt: ReadOnly[NativeFunction]
    sin: ReadOnly[NativeFunction]
    cos: ReadOnly[NativeFunction]
    tan: ReadOnly[NativeFunction]
    arcsin: ReadOnly[NativeFunction]
    arccos: ReadOnly[NativeFunction]
    arctan: ReadOnly[NativeFunction]
    sinh: ReadOnly[NativeFunction]
    cosh: ReadOnly[NativeFunction]
    tanh: ReadOnly[NativeFunction]
    erf: ReadOnly[NativeFunction]
    round: ReadOnly[NativeFunction]
    floor: ReadOnly[NativeFunction]
    ceil: ReadOnly[NativeFunction]


class BuiltinConstants(TypedDict):
    """Mapping of built-in constant names to their registry entries."""

    pi: ReadOnly[NativeConstant]
    e: ReadOnly[NativeConstant]
    inf: ReadOnly[NativeConstant]
    nan: ReadOnly[NativeConstant]


def _exp2(value: int | float) -> float:
    return math.pow(2, value)


# The Python callables computing the native built-ins, by name. The
# core's catalogue declares their sorts; `round` rounds half to even.
_NATIVE_IMPLEMENTATIONS: immutabledict[str, Callable[..., bool | int | float]] = (
    immutabledict(
        {
            "exp": math.exp,
            "exp2": _exp2,
            "log": math.log,
            "log2": math.log2,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "arcsin": math.asin,
            "arccos": math.acos,
            "arctan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "erf": math.erf,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
        }
    )
)

_rs.NativeFunction._install_builtins(_NATIVE_IMPLEMENTATIONS)

BUILTIN_CONSTANTS: BuiltinConstants = cast(
    BuiltinConstants,
    immutabledict(
        {name: get_registered_entry(name) for name in BuiltinConstants.__annotations__}
    ),
)

BUILTIN_FUNCTIONS: BuiltinFunctions = cast(
    BuiltinFunctions,
    immutabledict(
        {name: get_registered_entry(name) for name in BuiltinFunctions.__annotations__}
    ),
)
