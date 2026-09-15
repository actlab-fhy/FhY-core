"""Hypothesis strategies for expression literal values and literal leaves.

Each value strategy draws a plain Python value; the matching literal
strategy wraps that value in a
:class:`~fhy_core.symbolic.expression.LiteralExpression`. The decimal
string strategy is unsigned by construction: ``LiteralExpression``'s
string grammar (``src/fhy_core/symbolic/expression/core.py``, around
line 914) accepts only ``\\d+\\.\\d*`` or ``\\.\\d+``, with no leading
sign, so every drawn string matches one of those forms exactly.
"""

from typing import Final

from hypothesis import strategies as st

from fhy_core.symbolic.expression import LiteralExpression

__all__ = [
    "build_any_literal_strategy",
    "build_boolean_literal_strategy",
    "build_boolean_value_strategy",
    "build_decimal_string_literal_strategy",
    "build_decimal_string_value_strategy",
    "build_finite_float_literal_strategy",
    "build_finite_float_value_strategy",
    "build_integer_literal_strategy",
    "build_integer_value_strategy",
]

_DIGITS: Final = "0123456789"
_MAX_DIGIT_RUN: Final = 4


def build_integer_value_strategy(
    min_value: int = -64, max_value: int = 64
) -> st.SearchStrategy[int]:
    """Return a strategy for plain Python integers in ``[min_value, max_value]``."""
    return st.integers(min_value=min_value, max_value=max_value)


def build_boolean_value_strategy() -> st.SearchStrategy[bool]:
    """Return a strategy for plain Python booleans."""
    return st.booleans()


def build_finite_float_value_strategy(
    min_value: float = -1000.0, max_value: float = 1000.0
) -> st.SearchStrategy[float]:
    """Return a strategy for finite, non-subnormal floats in [min_value, max_value]."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


def _build_digit_string_strategy(
    min_size: int, max_size: int
) -> st.SearchStrategy[str]:
    """Return a strategy for strings of ``[min_size, max_size]`` decimal digits."""
    return st.text(alphabet=_DIGITS, min_size=min_size, max_size=max_size)


def build_decimal_string_value_strategy() -> st.SearchStrategy[str]:
    """Return a strategy for exact-decimal strings ``LiteralExpression`` accepts.

    Every drawn string matches ``\\d+\\.\\d*`` (a whole part, a point, and
    zero or more fraction digits, e.g. ``"12.500"``) or ``\\.\\d+`` (a
    point and one or more fraction digits, e.g. ``".25"``) by
    construction, so the value is never rejected downstream.
    """
    whole_dot_fraction = st.tuples(
        _build_digit_string_strategy(1, _MAX_DIGIT_RUN),
        _build_digit_string_strategy(0, _MAX_DIGIT_RUN),
    ).map(lambda parts: f"{parts[0]}.{parts[1]}")
    dot_fraction_only = _build_digit_string_strategy(1, _MAX_DIGIT_RUN).map(
        lambda digits: f".{digits}"
    )
    return st.one_of(whole_dot_fraction, dot_fraction_only)


def build_integer_literal_strategy(
    min_value: int = -64, max_value: int = 64
) -> st.SearchStrategy[LiteralExpression]:
    """Return a strategy for integer-valued ``LiteralExpression`` leaves."""
    return build_integer_value_strategy(min_value, max_value).map(LiteralExpression)


def build_boolean_literal_strategy() -> st.SearchStrategy[LiteralExpression]:
    """Return a strategy for boolean-valued ``LiteralExpression`` leaves."""
    return build_boolean_value_strategy().map(LiteralExpression)


def build_finite_float_literal_strategy(
    min_value: float = -1000.0, max_value: float = 1000.0
) -> st.SearchStrategy[LiteralExpression]:
    """Return a strategy for finite-float-valued ``LiteralExpression`` leaves."""
    return build_finite_float_value_strategy(min_value, max_value).map(
        LiteralExpression
    )


def build_decimal_string_literal_strategy() -> st.SearchStrategy[LiteralExpression]:
    """Return a strategy for decimal-string-valued ``LiteralExpression`` leaves."""
    return build_decimal_string_value_strategy().map(LiteralExpression)


def build_any_literal_strategy() -> st.SearchStrategy[LiteralExpression]:
    """Return a strategy drawing from every literal kind this module covers."""
    return st.one_of(
        build_integer_literal_strategy(),
        build_boolean_literal_strategy(),
        build_finite_float_literal_strategy(),
        build_decimal_string_literal_strategy(),
    )
