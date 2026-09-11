"""Hypothesis property tests for interval-integer parameter multiplication.

Split out of ``test_param_multiplication.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import math

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings
from hypothesis import strategies as st

from fhy_core.symbolic.param import create_interval_integer_param_between

from .conftest import build_interval_integer_param

pytestmark = pytest.mark.property

# Both properties run without a hypothesis deadline. Every example takes
# about a millisecond, no input is reliably slower than another, and no draw
# is filtered, so a deadline here would time only the scheduler: on a
# contended machine one example can be descheduled for hundreds of
# milliseconds, failing a property about which values a product admits for a
# reason unrelated to it.

# A finite stand-in for an unbounded end: far beyond any product of the
# bounded endpoints drawn below, so an unbounded result end must admit it.
_FAR_BEYOND_ANY_FINITE_HULL = 10**6

_optional_bound = st.none() | st.integers(min_value=-25, max_value=25)


def _order_optional_bounds(
    first: int | None, second: int | None
) -> tuple[int | None, int | None]:
    """Return ``(lower, upper)``, sorting two finite draws; ``None`` stays put."""
    if first is None or second is None:
        return first, second
    lower, upper = sorted((first, second))
    return lower, upper


def _convert_to_extended_interval(
    lower: int | None, upper: int | None
) -> tuple[float, float]:
    """Read ``None`` ends as the matching infinity over the extended reals."""
    return (
        -math.inf if lower is None else lower,
        math.inf if upper is None else upper,
    )


def _multiply_extended_reals(left: float, right: float) -> float:
    """Multiply over the extended reals, a zero factor pinning the product.

    IEEE arithmetic makes ``0 * inf`` NaN; interval-product set semantics
    make it ``0``, since ``{0}`` times any interval is ``{0}``.
    """
    if left == 0 or right == 0:
        return 0.0
    return left * right


def _compute_product_hull(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    """Return the ``(min, max)`` hull of the four endpoint products."""
    products = [
        _multiply_extended_reals(left_end, right_end)
        for left_end in left
        for right_end in right
    ]
    return min(products), max(products)


def _clamp_to_finite(value: float) -> int:
    """Return ``value`` as an ``int``, an infinite value clamped to the stand-in."""
    if math.isinf(value):
        return int(math.copysign(_FAR_BEYOND_ANY_FINITE_HULL, value))
    return int(value)


def _sample_hull_boundary(hull_min: float, hull_max: float) -> list[int]:
    """Return integers on and just outside each end of the hull.

    An unbounded end has no boundary to straddle, so it contributes the
    far-out stand-in instead, which must be admitted.
    """
    samples: list[int] = []
    if math.isinf(hull_min):
        samples.append(_clamp_to_finite(hull_min))
    else:
        samples.extend((int(hull_min) - 1, int(hull_min)))
    if math.isinf(hull_max):
        samples.append(_clamp_to_finite(hull_max))
    else:
        samples.extend((int(hull_max), int(hull_max) + 1))
    return samples


def _build_integer_strategy_within(
    ends: tuple[float, float],
) -> st.SearchStrategy[int]:
    """Return a strategy over the integers in ``ends``, infinite ends clamped."""
    lower, upper = ends
    return st.integers(
        min_value=_clamp_to_finite(lower), max_value=_clamp_to_finite(upper)
    )


# =============================================================================
# Property: soundness of interval multiplication
# =============================================================================


@settings(deadline=None)
@given(
    bound_1=st.integers(min_value=-25, max_value=25),
    bound_2=st.integers(min_value=-25, max_value=25),
    bound_3=st.integers(min_value=-25, max_value=25),
    bound_4=st.integers(min_value=-25, max_value=25),
    data=st.data(),
)
def test_multiplication_is_sound_for_every_concrete_pair_in_range(
    bound_1: int, bound_2: int, bound_3: int, bound_4: int, data: st.DataObject
) -> None:
    """Test that for any concrete ``x in [a,b]``, ``y in [c,d]``, ``x*y`` is valid.

    The interval-product result must admit every actual product of a
    concrete value drawn from each operand's interval -- this is the
    defining soundness property of interval arithmetic.
    """
    lower_1, upper_1 = sorted((bound_1, bound_2))
    lower_2, upper_2 = sorted((bound_3, bound_4))
    x = create_interval_integer_param_between(lower_1, upper_1)
    y = create_interval_integer_param_between(lower_2, upper_2)
    concrete_x = data.draw(st.integers(min_value=lower_1, max_value=upper_1))
    concrete_y = data.draw(st.integers(min_value=lower_2, max_value=upper_2))

    z = x * y

    assert z.is_constraints_satisfied(concrete_x * concrete_y)


# =============================================================================
# Property: tightness of interval multiplication over independent operands
# =============================================================================


@settings(deadline=None)
@given(
    left_bound_1=_optional_bound,
    left_bound_2=_optional_bound,
    right_bound_1=_optional_bound,
    right_bound_2=_optional_bound,
    data=st.data(),
)
def test_multiplication_admits_exactly_the_endpoint_product_hull(
    left_bound_1: int | None,
    left_bound_2: int | None,
    right_bound_1: int | None,
    right_bound_2: int | None,
    data: st.DataObject,
) -> None:
    """Test ``x * y`` admits an integer iff it lies in the four-corner product hull.

    The two operands are drawn independently, each end optionally
    unbounded. The expected hull is computed over the extended reals, and
    values on, just inside, and just outside each end of it -- plus one
    drawn from a window around it -- must be admitted exactly when they
    lie within it. A concrete product is checked alongside so a hole
    anywhere inside the hull is caught, not only a misplaced end.
    """
    left_lower, left_upper = _order_optional_bounds(left_bound_1, left_bound_2)
    right_lower, right_upper = _order_optional_bounds(right_bound_1, right_bound_2)
    left_ends = _convert_to_extended_interval(left_lower, left_upper)
    right_ends = _convert_to_extended_interval(right_lower, right_upper)
    hull_min, hull_max = _compute_product_hull(left_ends, right_ends)
    x = build_interval_integer_param(left_lower, left_upper)
    y = build_interval_integer_param(right_lower, right_upper)
    concrete_x = data.draw(_build_integer_strategy_within(left_ends), label="x")
    concrete_y = data.draw(_build_integer_strategy_within(right_ends), label="y")
    candidate = data.draw(
        st.integers(
            min_value=_clamp_to_finite(hull_min) - 3,
            max_value=_clamp_to_finite(hull_max) + 3,
        ),
        label="candidate",
    )

    z = x * y

    assert z.is_constraints_satisfied(concrete_x * concrete_y)
    for value in (*_sample_hull_boundary(hull_min, hull_max), candidate):
        assert z.is_constraints_satisfied(value) == (hull_min <= value <= hull_max)
