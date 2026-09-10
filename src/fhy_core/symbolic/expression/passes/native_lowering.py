"""Shared lowering of expression literals and constants to Python values.

Both the native-folding evaluator (:mod:`fhy_core.symbolic.expression.passes.evaluate`)
and the NumPy evaluator (:mod:`fhy_core.symbolic.expression.passes.numpy`) turn
expression-level literals and native-constant references into concrete
Python numerics at the point they hand off to Python or NumPy. These
helpers centralize that lowering so the two passes share one contract --
in particular, the refusal to coerce a float-grammar string literal to a
lossy binary ``float``.

The SymPy bridge (:mod:`fhy_core.symbolic.expression.passes.sympy`) does
not route through these helpers, and does not need the refusal: it has an
exact target to convert into, so it lowers a float-grammar string to a
``sympy.Rational`` carrying the literal's exact decimal value. The
refusal here is about the destination, not about the string form -- a
Python ``float`` is the only real number Python and NumPy arithmetic can
hold, and no binary ``float`` equals ``0.1``.
"""

__all__ = [
    "coerce_literal_value",
    "try_get_native_constant_value",
]

from fhy_core.identifier import Identifier

from ..core import LiteralType
from ..errors import StringLiteralPrecisionError
from ..registry import try_get_native_constant_for_identifier


def coerce_literal_value(value: LiteralType) -> bool | int | float:
    """Coerce a literal value to a Python numeric, rejecting lossy strings.

    ``bool`` / ``int`` / ``float`` values pass through unchanged.
    Integer-grammar string literals convert exactly via ``int``.
    Float-grammar string literals are refused: collapsing their exact
    decimal form to a binary ``float`` would discard the precision the
    string form exists to preserve.

    Args:
        value: Literal value to coerce.

    Returns:
        The Python numeric value.

    Raises:
        StringLiteralPrecisionError: If ``value`` is a float-grammar
            string literal.

    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        raise StringLiteralPrecisionError(
            f"cannot coerce string-form float literal {value!r} to a numeric "
            f"value without precision loss; use a float literal instead."
        ) from None


def try_get_native_constant_value(
    identifier: Identifier,
) -> bool | int | float | None:
    """Return the constant value ``identifier`` denotes, or ``None`` if absent.

    Resolution is by identifier identity: only the canonical identifier
    the registry minted for a constant carries its value. Returns
    ``None`` for every other identifier, including one that merely
    shares a constant's ``name_hint``.
    """
    entry = try_get_native_constant_for_identifier(identifier)
    if entry is None:
        return None
    return entry.value
