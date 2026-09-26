"""Lowering of expression literals and constants to Python values.

The two literal helpers run in the Rust core (``Decimal::to_f64_exact``,
S9 of ``docs/design/python-switch.md``), which the evaluators use too: a
decimal becomes a binary ``float`` only when the float equals it exactly,
so ``0.5`` converts and ``0.1`` is refused. The SymPy bridge
(:mod:`fhy_core.symbolic.expression.passes.sympy`) asks
:func:`is_decimal_text_exactly_binary` before it writes a rational as
decimal text, so every decimal literal it emits is one the evaluators
accept.
"""

__all__ = [
    "coerce_literal_value",
    "is_decimal_text_exactly_binary",
    "try_get_native_constant_value",
]

from fhy_core import _rs
from fhy_core.identifier import Identifier

from ..registry import try_get_native_constant_for_identifier

is_decimal_text_exactly_binary = _rs.is_decimal_text_exactly_binary
coerce_literal_value = _rs.coerce_literal_value


def try_get_native_constant_value(
    identifier: Identifier,
) -> bool | int | float | None:
    """Return the constant value ``identifier`` denotes, or ``None`` if absent.

    Resolution is by identifier identity: only the canonical identifier
    of a constant carries its value. Returns ``None`` for every other
    identifier, including one that merely shares a constant's
    ``name_hint``.
    """
    entry = try_get_native_constant_for_identifier(identifier)
    if entry is None:
        return None
    return entry.value
