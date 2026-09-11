"""Hypothesis property tests for interval-integer parameter multiplication.

Split out of ``test_param_multiplication.py`` so a test environment without
``hypothesis`` installed (the CI ``tests`` lane syncs only the ``test``
dependency group) can still collect the ordinary unit tests there; this
module is skipped wholesale via the ``importorskip`` below.
"""

import math
from typing import NamedTuple

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.symbolic.param import Param, create_interval_integer_param_between

from ...strategies.params import draw_interval_integer_param_with_bounds
from .conftest import build_interval_integer_param

pytestmark = pytest.mark.property

# Both properties run without a hypothesis deadline. Every example takes
# about a millisecond, no input is reliably slower than another, and no draw
# is filtered, so a deadline here would time only the scheduler: on a
# contended machine one example can be descheduled for hundreds of
# milliseconds, failing a property about which values a product admits for a
# reason unrelated to it. The `dev`/`thorough` profiles already set
# `deadline=None`.

# A finite stand-in for an unbounded end: far beyond any product of the
# bounded endpoints drawn below, so an unbounded result end must admit it.
_FAR_BEYOND_ANY_FINITE_HULL = 10**6


class BoundedMultiplicationCase(NamedTuple):
    """Two bounded interval-integer params with one concrete point each."""

    x: Param[int]
    y: Param[int]
    concrete_x: int
    concrete_y: int


class MultiplicationHullCase(NamedTuple):
    """Two interval-integer params, sample points, a candidate, and the hull."""

    x: Param[int]
    y: Param[int]
    concrete_x: int
    concrete_y: int
    candidate: int
    hull_min: float
    hull_max: float


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


@st.composite
def draw_bounded_multiplication_case(draw: st.DrawFn) -> BoundedMultiplicationCase:
    """Draw two bounded interval-integer params with a concrete point in each."""
    bound_1 = draw(st.integers(min_value=-25, max_value=25))
    bound_2 = draw(st.integers(min_value=-25, max_value=25))
    bound_3 = draw(st.integers(min_value=-25, max_value=25))
    bound_4 = draw(st.integers(min_value=-25, max_value=25))
    lower_1, upper_1 = sorted((bound_1, bound_2))
    lower_2, upper_2 = sorted((bound_3, bound_4))
    x = create_interval_integer_param_between(lower_1, upper_1)
    y = create_interval_integer_param_between(lower_2, upper_2)
    concrete_x = draw(st.integers(min_value=lower_1, max_value=upper_1))
    concrete_y = draw(st.integers(min_value=lower_2, max_value=upper_2))
    return BoundedMultiplicationCase(x, y, concrete_x, concrete_y)


@given(case=draw_bounded_multiplication_case())
def test_multiplication_is_sound_for_every_concrete_pair_in_range(
    case: BoundedMultiplicationCase,
) -> None:
    """Test that for any concrete ``x in [a,b]``, ``y in [c,d]``, ``x*y`` is valid.

    The interval-product result must admit every actual product of a
    concrete value drawn from each operand's interval -- this is the
    defining soundness property of interval arithmetic.
    """
    z = case.x * case.y

    assert z.is_constraints_satisfied(case.concrete_x * case.concrete_y)


# =============================================================================
# Property: tightness of interval multiplication over independent operands
# =============================================================================


@st.composite
def draw_multiplication_hull_case(draw: st.DrawFn) -> MultiplicationHullCase:
    """Draw two interval-integer params, sample points, a candidate, and the hull.

    The two operands are drawn independently, each end optionally
    unbounded. The expected hull is computed over the extended reals from
    the operands' own bounds, so the property below can check it directly
    rather than reconstructing it from the params.
    """
    x, left_lower, left_upper = draw(draw_interval_integer_param_with_bounds())
    y, right_lower, right_upper = draw(draw_interval_integer_param_with_bounds())
    left_ends = _convert_to_extended_interval(left_lower, left_upper)
    right_ends = _convert_to_extended_interval(right_lower, right_upper)
    hull_min, hull_max = _compute_product_hull(left_ends, right_ends)
    concrete_x = draw(_build_integer_strategy_within(left_ends))
    concrete_y = draw(_build_integer_strategy_within(right_ends))
    candidate = draw(
        st.integers(
            min_value=_clamp_to_finite(hull_min) - 3,
            max_value=_clamp_to_finite(hull_max) + 3,
        )
    )
    return MultiplicationHullCase(
        x, y, concrete_x, concrete_y, candidate, hull_min, hull_max
    )


def _build_hull_example(
    left_lower: int, left_upper: int, right_lower: int, right_upper: int
) -> MultiplicationHullCase:
    """Build a hull case at module level, for pinning a hand-picked example."""
    left_ends = _convert_to_extended_interval(left_lower, left_upper)
    right_ends = _convert_to_extended_interval(right_lower, right_upper)
    hull_min, hull_max = _compute_product_hull(left_ends, right_ends)
    return MultiplicationHullCase(
        x=build_interval_integer_param(left_lower, left_upper),
        y=build_interval_integer_param(right_lower, right_upper),
        concrete_x=left_lower,
        concrete_y=right_lower,
        candidate=_clamp_to_finite(hull_min),
        hull_min=hull_min,
        hull_max=hull_max,
    )


# Each pinned case below is a bounded pair whose hull is not the set of
# actual pairwise products (e.g. ``[1,3] * [1,3]`` admits ``5``, which is
# nobody's product), so the tightness check -- against the corner-product
# hull, not brute-force multiplication -- is the property that matters here.
@example(case=_build_hull_example(0, 0, 0, 0))
@example(case=_build_hull_example(0, 3, 0, 3))
@example(case=_build_hull_example(-3, 3, -3, 3))
@example(case=_build_hull_example(-3, -1, -3, -1))
@example(case=_build_hull_example(1, 3, 1, 3))
@example(case=_build_hull_example(-3, -1, 1, 3))
@example(case=_build_hull_example(1, 3, -3, -1))
@example(case=_build_hull_example(-2, 3, -3, 1))
@example(case=_build_hull_example(0, 3, -3, -1))
@given(case=draw_multiplication_hull_case())
def test_multiplication_admits_exactly_the_endpoint_product_hull(
    case: MultiplicationHullCase,
) -> None:
    """Test ``x * y`` admits an integer iff it lies in the four-corner product hull.

    Values on, just inside, and just outside each end of the hull -- plus
    one drawn from a window around it -- must be admitted exactly when
    they lie within it. A concrete product is checked alongside so a hole
    anywhere inside the hull is caught, not only a misplaced end.
    """
    z = case.x * case.y

    assert z.is_constraints_satisfied(case.concrete_x * case.concrete_y)
    for value in (
        *_sample_hull_boundary(case.hull_min, case.hull_max),
        case.candidate,
    ):
        assert z.is_constraints_satisfied(value) == (
            case.hull_min <= value <= case.hull_max
        )
