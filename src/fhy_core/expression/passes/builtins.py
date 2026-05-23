"""Built-in functions pre-registered at module import.

Importing this module is a side-effecting registration: ``max`` and
``min`` are bound into the process-wide function registry. The single
``BUILTIN_FUNCTIONS`` mapping exposes every built-in by name so callers
do not hard-code strings and so new built-ins can be added in one place.

``max`` and ``min`` are strictly binary; n-ary uses (``max(a, b, c)``
etc.) are expressed by explicit folding (``max(max(a, b), c)``).
"""

__all__ = ["BUILTIN_FUNCTIONS", "BuiltinFunctions"]

from typing import TypedDict

from fhy_core.identifier import Identifier

from ..core import IdentifierExpression, TernaryExpression
from ..registry import RegisteredFunction, register_function


class BuiltinFunctions(TypedDict):
    """Mapping of built-in function names to their ``RegisteredFunction``.

    Each field corresponds to one built-in. Adding a new built-in is one
    field on this ``TypedDict`` and one entry in ``BUILTIN_FUNCTIONS``.
    """

    max: RegisteredFunction
    min: RegisteredFunction


def _register_max() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    body = TernaryExpression(
        IdentifierExpression(a) > IdentifierExpression(b),
        IdentifierExpression(a),
        IdentifierExpression(b),
    )
    return register_function("max", parameters=[a, b], body=body)


def _register_min() -> RegisteredFunction:
    a = Identifier("a")
    b = Identifier("b")
    body = TernaryExpression(
        IdentifierExpression(a) < IdentifierExpression(b),
        IdentifierExpression(a),
        IdentifierExpression(b),
    )
    return register_function("min", parameters=[a, b], body=body)


BUILTIN_FUNCTIONS: BuiltinFunctions = {
    "max": _register_max(),
    "min": _register_min(),
}
