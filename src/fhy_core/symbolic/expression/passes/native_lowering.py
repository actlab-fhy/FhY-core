"""Shared lowering of expression literals and constants to Python values.

Both the native-folding evaluator (:mod:`fhy_core.symbolic.expression.passes.evaluate`)
and the NumPy evaluator (:mod:`fhy_core.symbolic.expression.passes.numpy`) turn
expression-level literals and native-constant references into concrete
Python numerics at the point they hand off to Python or NumPy. These
helpers centralize that lowering so the two passes share one contract --
in particular, the refusal to coerce a float-grammar string literal to a
binary ``float`` that does not denote the same exact value.

The SymPy bridge (:mod:`fhy_core.symbolic.expression.passes.sympy`) does
not lower through these helpers, and does not need the refusal: it has an
exact target to convert into, so it lowers a float-grammar string to a
``sympy.Rational`` carrying the literal's exact decimal value. The
refusal here is about the destination, not about the string form -- a
Python ``float`` is the only real number Python and NumPy arithmetic can
hold, and no binary ``float`` equals ``0.1``, while ``0.5`` is one. The
bridge's lifter asks :func:`is_decimal_text_exactly_binary`, the test the
refusal applies, before it writes a rational as decimal text, so every
string literal it emits is one these helpers accept.
"""

__all__ = [
    "coerce_literal_value",
    "is_decimal_text_exactly_binary",
    "try_get_native_constant_value",
]

from decimal import Decimal

from fhy_core.identifier import Identifier

from ..core import LiteralType
from ..errors import StringLiteralPrecisionError
from ..registry import try_get_native_constant_for_identifier


def is_decimal_text_exactly_binary(text: str) -> bool:
    """Return whether some binary ``float`` equals decimal ``text`` exactly.

    ``float`` rounds the text to the nearest binary value, and both sides
    of the comparison are exact decimal expansions, so the test holds only
    when that rounding changes nothing: ``"0.5"`` passes and ``"0.1"``
    does not. Neither the ``Decimal`` string constructor nor a ``Decimal``
    comparison rounds to the context precision, so the answer is exact for
    text of any length.

    Args:
        text: Integer- or float-grammar decimal text.

    Returns:
        Whether converting ``text`` to a ``float`` loses nothing.

    """
    return Decimal(text) == Decimal(float(text))


def coerce_literal_value(value: LiteralType) -> bool | int | float:
    """Coerce a literal value to a Python numeric, rejecting lossy strings.

    ``bool`` / ``int`` / ``float`` values pass through unchanged.
    Integer-grammar string literals convert exactly via ``int``.
    A float-grammar string literal converts via ``float`` when that
    binary value's exact decimal expansion equals the literal's exact
    decimal value (for example ``"0.5"``); otherwise the conversion is
    refused, since it would discard the precision the string form
    exists to preserve (for example ``"0.1"``).

    Args:
        value: Literal value to coerce.

    Returns:
        The Python numeric value.

    Raises:
        StringLiteralPrecisionError: If ``value`` is a float-grammar
            string literal with no exact binary ``float`` equivalent.

    """
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    if is_decimal_text_exactly_binary(value):
        return float(value)
    raise StringLiteralPrecisionError(
        f"cannot coerce string-form float literal {value!r} to a numeric "
        f"value: no binary float equals its exact decimal value; use a "
        f"float literal instead if binary-float semantics are intended."
    )


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
