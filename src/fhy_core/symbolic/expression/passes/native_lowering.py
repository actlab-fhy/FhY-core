"""Shared lowering of expression literals and constants to Python values.

Both the native-folding evaluator (:mod:`fhy_core.symbolic.expression.passes.evaluate`)
and the NumPy evaluator (:mod:`fhy_core.symbolic.expression.passes.numpy`) turn
expression-level literals and native-constant references into concrete
Python numerics at the point they hand off to Python or NumPy. These
helpers centralize that lowering so the two passes share one contract --
in particular, the refusal to coerce a float-grammar string literal to a
lossy binary ``float``.

The SymPy bridge (:mod:`fhy_core.symbolic.expression.passes.sympy`) is
exempt from this contract: SymPy operates on binary floats, so it converts
a float-grammar string with ``sympy.Float`` -- accepting the precision loss
-- rather than routing through these helpers.
"""

__all__ = [
    "coerce_literal_value",
    "try_get_native_constant_value",
]

from ..core import LiteralType
from ..errors import EntryLookupError, StringLiteralPrecisionError
from ..registry import NativeConstant, get_registered_entry


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


def try_get_native_constant_value(name: str) -> bool | int | float | None:
    """Return the constant value bound to ``name``, or ``None`` if absent.

    Returns ``None`` when ``name`` is unregistered or resolves to a
    non-constant entry (a registered function).
    """
    try:
        entry = get_registered_entry(name)
    except EntryLookupError:
        return None
    if isinstance(entry, NativeConstant):
        return entry.value
    return None
