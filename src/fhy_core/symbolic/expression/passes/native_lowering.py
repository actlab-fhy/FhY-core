"""Shared lowering of expression literals and constants to Python values.

Both the native-folding evaluator (:mod:`fhy_core.symbolic.expression.passes.evaluate`)
and the NumPy evaluator (:mod:`fhy_core.symbolic.expression.passes.numpy`) turn
expression-level literals and native-constant references into concrete
Python numerics at the point they hand off to Python or NumPy. These
helpers centralize that lowering so the two passes share one contract --
in particular, the refusal to coerce a decimal literal (a
``decimal.Decimal`` value, which a float-grammar string literal normalizes
to) to a binary ``float`` that does not denote the same exact value.

The SymPy bridge (:mod:`fhy_core.symbolic.expression.passes.sympy`) does
not lower through these helpers, and does not need the refusal: it has an
exact target to convert into, so it lowers a decimal literal to a
``sympy.Rational`` carrying its exact value. The refusal here is about the
destination, not about the literal -- a Python ``float`` is the only real
number Python and NumPy arithmetic can hold, and no binary ``float``
equals ``0.1``, while ``0.5`` is one. The bridge's lifter asks
:func:`is_decimal_text_exactly_binary`, the test the refusal applies,
before it writes a rational as decimal text, so every decimal literal it
emits is one these helpers accept.
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


def is_decimal_text_exactly_binary(text: str | Decimal) -> bool:
    """Return whether some binary ``float`` equals decimal ``text`` exactly.

    ``float`` rounds the text to the nearest binary value, and both sides
    of the comparison are exact decimal expansions, so the test holds only
    when that rounding changes nothing: ``"0.5"`` passes and ``"0.1"``
    does not. Neither the ``Decimal`` string constructor nor a ``Decimal``
    comparison rounds to the context precision, so the answer is exact for
    text of any length.

    Args:
        text: Integer- or float-grammar decimal text, or a finite
            ``Decimal``.

    Returns:
        Whether converting ``text`` to a ``float`` loses nothing.

    """
    return Decimal(text) == Decimal(float(text))


def coerce_literal_value(value: LiteralType) -> bool | int | float:
    """Coerce a literal value to a Python numeric, rejecting lossy decimals.

    ``bool`` / ``int`` / ``float`` values pass through unchanged. A
    decimal -- a ``Decimal``, the value a float-grammar string literal
    normalizes to, or float-grammar text -- converts via ``float`` when
    that binary value's exact decimal expansion equals the decimal's
    exact value (for example ``Decimal("0.5")``); otherwise the
    conversion is refused, since it would discard the precision the
    decimal form exists to preserve (for example ``Decimal("0.1")``).
    Integer-grammar text converts exactly via ``int``.

    Args:
        value: Literal value to coerce.

    Returns:
        The Python numeric value.

    Raises:
        StringLiteralPrecisionError: If ``value`` is a decimal with no
            exact binary ``float`` equivalent.

    """
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    elif not isinstance(value, Decimal):
        return value
    if is_decimal_text_exactly_binary(value):
        return float(value)
    raise StringLiteralPrecisionError(
        f"cannot coerce decimal literal {_format_decimal_literal(value)} to a "
        f"numeric value: no binary float equals its exact decimal value; use "
        f"a float literal instead if binary-float semantics are intended."
    )


def _format_decimal_literal(value: str | Decimal) -> str:
    """Return a decimal literal's value as quoted positional text."""
    text = value if isinstance(value, str) else format(value, "f")
    return repr(text)


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
