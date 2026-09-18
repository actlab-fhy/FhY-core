"""Hypothesis property tests for the natural-number bound factories.

The bounded factories admit exactly the natural numbers between their
bounds, with each endpoint admitted iff its bound is inclusive, and reject
reversed or degenerate bounds. The oracle is integer arithmetic on the
drawn bounds.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.symbolic.param import (
    ParamError,
    create_natural_param_between,
    create_natural_param_with_lower_bound,
    create_natural_param_with_upper_bound,
)

pytestmark = pytest.mark.property

# Bounds start at 1 so every drawn pair satisfies the natural domain whether or
# not zero is included; the zero edge cases live in the example tests. A gap of
# at least one keeps equal bounds inclusive-only, which is the documented rule.
_MIN_BOUND = 1
_MAX_BOUND = 12
_MAX_GAP = 12
_CANDIDATES = st.integers(min_value=-2, max_value=_MAX_BOUND + _MAX_GAP + 2)


@st.composite
def draw_between_case(
    draw: st.DrawFn,
) -> tuple[int, int, bool, bool, bool, int]:
    """Draw valid between-factory arguments and a candidate value.

    Returns ``(lower, upper, is_lower_inclusive, is_upper_inclusive,
    zero_included, candidate)``.
    """
    is_lower_inclusive = draw(st.booleans())
    is_upper_inclusive = draw(st.booleans())
    minimum_gap = 0 if is_lower_inclusive and is_upper_inclusive else 1
    lower = draw(st.integers(min_value=_MIN_BOUND, max_value=_MAX_BOUND))
    upper = lower + draw(st.integers(min_value=minimum_gap, max_value=_MAX_GAP))
    zero_included = draw(st.booleans())
    candidate = draw(_CANDIDATES)
    return (
        lower,
        upper,
        is_lower_inclusive,
        is_upper_inclusive,
        zero_included,
        candidate,
    )


def is_within_bounds(
    value: int,
    lower: int,
    upper: int,
    is_lower_inclusive: bool,
    is_upper_inclusive: bool,
) -> bool:
    """Return whether ``value`` lies between the bounds under their inclusivity."""
    above_lower = value >= lower if is_lower_inclusive else value > lower
    below_upper = value <= upper if is_upper_inclusive else value < upper
    return above_lower and below_upper


@given(case=draw_between_case())
def test_between_factory_admits_exactly_the_values_within_its_bounds(
    case: tuple[int, int, bool, bool, bool, int],
) -> None:
    """Test membership in a between param is the arithmetic bound check."""
    lower, upper, is_lower_inclusive, is_upper_inclusive, zero_included, candidate = (
        case
    )
    param = create_natural_param_between(
        lower,
        upper,
        zero_included=zero_included,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )

    is_valid = param.is_value_valid(candidate)

    assert is_valid == is_within_bounds(
        candidate, lower, upper, is_lower_inclusive, is_upper_inclusive
    )


@given(
    lower=st.integers(min_value=_MIN_BOUND + 1, max_value=_MAX_BOUND),
    gap=st.integers(min_value=1, max_value=_MAX_BOUND),
)
def test_between_factory_rejects_reversed_bounds(lower: int, gap: int) -> None:
    """Test a lower bound above the upper bound raises `ParamError`."""
    with pytest.raises(ParamError, match="Lower bound"):
        create_natural_param_between(lower, lower - gap)


@given(
    bound=st.integers(min_value=_MIN_BOUND, max_value=_MAX_BOUND),
    inclusivity=st.sampled_from([(False, True), (True, False), (False, False)]),
)
def test_between_factory_rejects_equal_bounds_with_an_exclusive_side(
    bound: int, inclusivity: tuple[bool, bool]
) -> None:
    """Test equal bounds are only admitted when both sides are inclusive."""
    is_lower_inclusive, is_upper_inclusive = inclusivity

    with pytest.raises(ParamError, match="Lower bound"):
        create_natural_param_between(
            bound,
            bound,
            is_lower_inclusive=is_lower_inclusive,
            is_upper_inclusive=is_upper_inclusive,
        )


@given(
    lower=st.integers(min_value=_MIN_BOUND, max_value=_MAX_BOUND),
    is_inclusive=st.booleans(),
    zero_included=st.booleans(),
    candidate=_CANDIDATES,
)
def test_lower_bound_factory_admits_exactly_the_values_from_its_bound(
    lower: int, is_inclusive: bool, zero_included: bool, candidate: int
) -> None:
    """Test membership in a lower-bounded param is the arithmetic bound check."""
    param = create_natural_param_with_lower_bound(
        lower, zero_included=zero_included, is_inclusive=is_inclusive
    )

    is_valid = param.is_value_valid(candidate)

    assert is_valid == (candidate >= lower if is_inclusive else candidate > lower)


@given(
    upper=st.integers(min_value=_MIN_BOUND + 1, max_value=_MAX_BOUND),
    is_inclusive=st.booleans(),
    zero_included=st.booleans(),
    candidate=_CANDIDATES,
)
def test_upper_bound_factory_admits_exactly_the_naturals_up_to_its_bound(
    upper: int, is_inclusive: bool, zero_included: bool, candidate: int
) -> None:
    """Test membership in an upper-bounded param combines the bound and the domain."""
    param = create_natural_param_with_upper_bound(
        upper, zero_included=zero_included, is_inclusive=is_inclusive
    )

    is_valid = param.is_value_valid(candidate)

    domain_minimum = 0 if zero_included else 1
    below_upper = candidate <= upper if is_inclusive else candidate < upper
    assert is_valid == (candidate >= domain_minimum and below_upper)
