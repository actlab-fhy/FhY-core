"""Hypothesis property tests for interval-integer addition, subtraction, negation.

Covers the soundness and tightness of ``Param.__add__``, ``Param.__sub__``,
and ``Param.__neg__`` over interval-integer operands drawn bounded,
half-bounded, and fully unbounded, with each bounded end inclusive or
exclusive and now and then far beyond the int64 range.
"""

from typing import Final, NamedTuple

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.symbolic.param import Param

from ...strategies.params import draw_interval_integer_param_with_bounds
from .conftest import build_interval_integer_param

pytestmark = pytest.mark.property

# Every example below takes about a millisecond and no draw is filtered, so a
# deadline would time only the scheduler: on a contended machine one example
# can be descheduled for hundreds of milliseconds, failing a soundness or
# tightness check for a reason unrelated to it. The `dev`/`thorough` profiles
# already set `deadline=None`.

# How far beyond both zero and the finite opposite end an unbounded side is
# sampled for a concrete point, and probed to show that an unbounded result
# admits arbitrarily large magnitudes.
_UNBOUNDED_SAMPLE_REACH: Final = 35
_FAR_BEYOND_ANY_FINITE_HULL: Final = 10**6


class BinaryIntervalCase(NamedTuple):
    """Two interval-integer params, their bounds, and one concrete point each."""

    x: Param[int]
    x_lower: int | None
    x_upper: int | None
    y: Param[int]
    y_lower: int | None
    y_upper: int | None
    concrete_x: int
    concrete_y: int


def _find_far_below(upper: int | None, distance: int) -> int:
    """Return a value ``distance`` below zero and below ``upper`` when it is set."""
    return min(0, 0 if upper is None else upper) - distance


def _find_far_above(lower: int | None, distance: int) -> int:
    """Return a value ``distance`` above zero and above ``lower`` when it is set."""
    return max(0, 0 if lower is None else lower) + distance


def _build_sample_strategy(
    lower: int | None, upper: int | None
) -> st.SearchStrategy[int]:
    """Return a strategy over a finite side as-is, an unbounded side clamped.

    An unbounded side is clamped 35 steps beyond both zero and the finite
    opposite end, per the property's own scope: soundness only needs one
    witness point, and the clamp keeps the range ordered however far from
    zero that opposite end was drawn.
    """
    low = _find_far_below(upper, _UNBOUNDED_SAMPLE_REACH) if lower is None else lower
    high = _find_far_above(lower, _UNBOUNDED_SAMPLE_REACH) if upper is None else upper
    return st.integers(min_value=low, max_value=high)


@st.composite
def draw_binary_interval_case(draw: st.DrawFn) -> BinaryIntervalCase:
    """Draw two interval-integer params with one concrete point sampled from each."""
    x, x_lower, x_upper = draw(draw_interval_integer_param_with_bounds())
    y, y_lower, y_upper = draw(draw_interval_integer_param_with_bounds())
    concrete_x = draw(_build_sample_strategy(x_lower, x_upper))
    concrete_y = draw(_build_sample_strategy(y_lower, y_upper))
    return BinaryIntervalCase(
        x, x_lower, x_upper, y, y_lower, y_upper, concrete_x, concrete_y
    )


def _build_case(
    x_lower: int | None, x_upper: int | None, y_lower: int | None, y_upper: int | None
) -> BinaryIntervalCase:
    """Build a case at module level, for pinning a hand-picked example."""
    x = build_interval_integer_param(x_lower, x_upper)
    y = build_interval_integer_param(y_lower, y_upper)
    concrete_x = x_lower if x_lower is not None else (x_upper or 0)
    concrete_y = y_lower if y_lower is not None else (y_upper or 0)
    return BinaryIntervalCase(
        x, x_lower, x_upper, y, y_lower, y_upper, concrete_x, concrete_y
    )


def _build_identical_operand_case(
    lower_bound: int,
    upper_bound: int,
    *,
    is_lower_inclusive: bool,
    is_upper_inclusive: bool,
) -> BinaryIntervalCase:
    """Build a case of two operands over the same raw bounds, for pinning an example.

    The case records each operand's least and greatest members, one step
    inside an exclusive raw bound.
    """
    lower = lower_bound if is_lower_inclusive else lower_bound + 1
    upper = upper_bound if is_upper_inclusive else upper_bound - 1
    x, y = (
        build_interval_integer_param(
            lower_bound,
            upper_bound,
            is_lower_inclusive=is_lower_inclusive,
            is_upper_inclusive=is_upper_inclusive,
        )
        for _ in range(2)
    )
    return BinaryIntervalCase(x, lower, upper, y, lower, upper, lower, lower)


def _assert_admits_exactly(
    result: Param[int], lower: int | None, upper: int | None
) -> None:
    """Assert ``result`` admits exactly the integers in ``[lower, upper]``.

    ``None`` reads as unbounded on that side: probed with a value far past
    both zero and the finite opposite end, which must be admitted. A
    finite side is probed at the endpoint (admitted) and one step beyond
    it (not admitted).
    """
    if lower is None:
        assert result.is_constraints_satisfied(
            _find_far_below(upper, _FAR_BEYOND_ANY_FINITE_HULL)
        )
    else:
        assert result.is_constraints_satisfied(lower)
        assert not result.is_constraints_satisfied(lower - 1)
    if upper is None:
        assert result.is_constraints_satisfied(
            _find_far_above(lower, _FAR_BEYOND_ANY_FINITE_HULL)
        )
    else:
        assert result.is_constraints_satisfied(upper)
        assert not result.is_constraints_satisfied(upper + 1)


# =============================================================================
# Property: soundness of interval addition, subtraction, and negation
# =============================================================================


@given(case=draw_binary_interval_case())
def test_interval_addition_subtraction_negation_are_sound(
    case: BinaryIntervalCase,
) -> None:
    """Test x+y valid for A+B, x-y valid for A-B, -x valid for -A, for sampled points.

    For any concrete ``x`` admissible in ``A`` and ``y`` admissible in
    ``B`` (sampled within their finite bounds, or within 35 steps beyond
    zero and the opposite end on an unbounded side), the
    interval-arithmetic result must admit the actual sum, difference, and
    negation -- the defining soundness property of interval arithmetic.
    """
    sum_result = case.x + case.y
    difference_result = case.x - case.y
    negation_result = -case.x

    assert sum_result.is_constraints_satisfied(case.concrete_x + case.concrete_y)
    assert difference_result.is_constraints_satisfied(case.concrete_x - case.concrete_y)
    assert negation_result.is_constraints_satisfied(-case.concrete_x)


# =============================================================================
# Property: tightness of interval addition, subtraction, and negation
# =============================================================================


@example(case=_build_case(0, 0, 0, 0))
@example(case=_build_case(0, 1, 0, 1))
@example(case=_build_case(-3, 3, -3, 3))
@example(case=_build_case(-2, 2, -2, 2))
@example(
    case=_build_identical_operand_case(
        0, 1, is_lower_inclusive=False, is_upper_inclusive=True
    )
)
@example(
    case=_build_identical_operand_case(
        0, 1, is_lower_inclusive=True, is_upper_inclusive=False
    )
)
@example(
    case=_build_identical_operand_case(
        0, 2, is_lower_inclusive=False, is_upper_inclusive=False
    )
)
@example(
    case=_build_identical_operand_case(
        -3, 3, is_lower_inclusive=False, is_upper_inclusive=False
    )
)
@example(
    case=_build_identical_operand_case(
        -2, 2, is_lower_inclusive=False, is_upper_inclusive=False
    )
)
@given(case=draw_binary_interval_case())
def test_interval_addition_subtraction_negation_admit_exactly_the_endpoint_hull(
    case: BinaryIntervalCase,
) -> None:
    """Test A+B, A-B, and -A admit exactly the endpoint hull, boundary probed.

    ``A + B`` admits exactly ``[lo_A + lo_B, hi_A + hi_B]``, ``A - B``
    admits exactly ``[lo_A - hi_B, hi_A - lo_B]``, and ``-A`` admits
    exactly ``[-hi_A, -lo_A]`` -- ``None`` on either side of an operand
    propagating to ``None`` on the corresponding side of the result. Both
    endpoints of each hull must be admitted and one step beyond each
    finite endpoint must not be, which subsumes checking a concrete pair.
    An operand's endpoints are its least and greatest members, so an
    exclusive raw bound is checked through the member one step inside it.
    """
    sum_result = case.x + case.y
    difference_result = case.x - case.y
    negation_result = -case.x

    sum_lower = (
        None
        if case.x_lower is None or case.y_lower is None
        else case.x_lower + case.y_lower
    )
    sum_upper = (
        None
        if case.x_upper is None or case.y_upper is None
        else case.x_upper + case.y_upper
    )
    _assert_admits_exactly(sum_result, sum_lower, sum_upper)

    difference_lower = (
        None
        if case.x_lower is None or case.y_upper is None
        else case.x_lower - case.y_upper
    )
    difference_upper = (
        None
        if case.x_upper is None or case.y_lower is None
        else case.x_upper - case.y_lower
    )
    _assert_admits_exactly(difference_result, difference_lower, difference_upper)

    negation_lower = None if case.x_upper is None else -case.x_upper
    negation_upper = None if case.x_lower is None else -case.x_lower
    _assert_admits_exactly(negation_result, negation_lower, negation_upper)

    assert sum_result.is_constraints_satisfied(case.concrete_x + case.concrete_y)
    assert difference_result.is_constraints_satisfied(case.concrete_x - case.concrete_y)
    assert negation_result.is_constraints_satisfied(-case.concrete_x)
