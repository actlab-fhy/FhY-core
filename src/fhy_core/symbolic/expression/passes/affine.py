"""The exact affine form of an expression over its free identifiers.

:func:`affine_form` reads an expression as ``c_1 * x_1 + ... + c_n * x_n +
c_0``, with an exact :class:`fractions.Fraction` coefficient per free
identifier and an exact constant, or answers ``None`` when it cannot prove
the expression affine. It runs in the Rust core
(``Expression::affine_form``) and needs no solver backend: ``(3 * s + t) -
t`` gives the form ``3 * s`` whatever ``t`` is.

It reads integer and decimal literals, identifiers, negation, addition and
subtraction, multiplication by a constant form, true division by a non-zero
constant, and floor division, modulo and integer powers of constant
operands, which it folds exactly. It declines everything else: a float or
Boolean literal, a comparison, a logical operation, a piecewise expression,
a call, a product of two non-constant forms, a tree nested more than 256
levels, and a coefficient whose parts would exceed 4096 bits.
"""

__all__ = [
    "AffineForm",
    "affine_form",
]

from fhy_core import _rs

from ..core import Expression

AffineForm = _rs.AffineForm
"""An expression as exact multiples of its free identifiers plus a constant."""


def affine_form(expression: Expression) -> AffineForm | None:
    """Return the exact affine form of ``expression`` over its free identifiers.

    Args:
        expression: The expression to read.

    Returns:
        The form, or ``None`` when the analysis cannot prove the expression
        affine.

    Raises:
        TypeError: If ``expression`` is no :class:`Expression`.

    """
    return _rs.affine_form(expression)
