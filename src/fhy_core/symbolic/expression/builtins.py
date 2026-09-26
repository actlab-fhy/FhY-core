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

Importing this module has the extension build the built-ins' entries,
once. A native built-in's ``implementation`` is the core's kernel
(``BuiltinFunction::native_value``), the one the evaluators compute it
with: IEEE results, so ``sqrt(-1.0)`` is ``nan``, ``round`` rounding half
to even, and ``round``, ``floor`` and ``ceil`` returning the exact
``int``. The kernels call the platform's math library, except ``erf``
(``libm``), so the final bits may differ across operating systems and
CPU families. That the composed bodies type-check against their declared
sorts is pinned by ``tests/types/checking/test_builtin_bodies.py``
rather than checked on every import.
"""

__all__ = [
    "BUILTIN_CONSTANTS",
    "BUILTIN_FUNCTIONS",
    "BuiltinConstants",
    "BuiltinFunctions",
]

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
    instances; native entries are :class:`NativeFunction` instances whose
    ``implementation`` is the core's kernel.
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


_rs.NativeFunction._install_builtins()

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
