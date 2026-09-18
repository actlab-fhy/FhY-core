"""Hypothesis property tests for `Param` and `ParamAssignment` serialization.

Covers the round-trip inverse pair (``serialize``/``deserialize``) over every
domain kind, over union/intersection/interval-arithmetic results, and over a
concrete assignment.
"""

from typing import Any, Final

import pytest

pytest.importorskip("hypothesis")

from hypothesis import example, given
from hypothesis import strategies as st

from fhy_core.serialization import SerializationFormat
from fhy_core.symbolic.param import (
    Param,
    ParamAssignment,
    create_categorical_param,
    create_integer_param_between,
    create_intersection_param,
    create_interval_integer_param_between,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param_between,
    create_single_valid_value_param,
    create_union_param,
)

from ...strategies.params import (
    build_categorical_value_set_strategy,
    build_ordinal_value_set_strategy,
    build_permutation_member_set_strategy,
    draw_interval_integer_param_with_bounds,
    draw_overlapping_intersection_eligible_group,
    draw_overlapping_union_eligible_group,
    draw_param_over_any_domain,
)
from .conftest import assert_param_round_trips_in_all_formats

pytestmark = pytest.mark.property

# Every property runs without a hypothesis deadline; the `dev`/`thorough`
# profiles already set `deadline=None`.

_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")
_CANDIDATE_LIMIT: Final = 25


@st.composite
def draw_derived_param_result(draw: st.DrawFn) -> Param[Any]:
    """Draw a param built by union, intersection, interval addition, or multiplication.

    Union and intersection operands are drawn sharing a member, so the
    result is non-empty by construction; an empty result would make the
    factory raise rather than return a param.
    """
    kind = draw(st.sampled_from(("union", "intersection", "add", "mul")))
    if kind == "union":
        (left, right), _ = draw(draw_overlapping_union_eligible_group(size=2))
        return create_union_param(left, right)
    if kind == "intersection":
        (left, right), _ = draw(draw_overlapping_intersection_eligible_group(size=2))
        return create_intersection_param(left, right)
    x, _, _ = draw(draw_interval_integer_param_with_bounds())
    y, _, _ = draw(draw_interval_integer_param_with_bounds())
    if kind == "add":
        return x + y
    return x * y


@st.composite
def draw_param_with_a_valid_value(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw a param over some domain together with a value guaranteed valid for it.

    Unlike ``draw_param_with_candidate``, which draws from a superset so
    an invalid candidate occurs too, every value here is the domain's
    own first (or only, or lower-bound) member, so ``Param.assign``
    never raises.
    """
    kind = draw(
        st.sampled_from(
            (
                "ordinal",
                "categorical",
                "permutation",
                "bounded_integer",
                "natural",
                "bounded_real",
                "single_valid_value",
                "interval_integer",
            )
        )
    )
    result: tuple[Param[Any], Any]
    if kind == "ordinal":
        values = draw(build_ordinal_value_set_strategy())
        result = create_ordinal_param(values), values[0]
    elif kind == "categorical":
        categories = draw(build_categorical_value_set_strategy())
        result = create_categorical_param(categories), categories[0]
    elif kind == "permutation":
        members = draw(build_permutation_member_set_strategy())
        result = create_permutation_param(members), tuple(members)
    elif kind == "bounded_integer":
        lower = draw(
            st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT)
        )
        extra = draw(st.integers(min_value=0, max_value=2 * _CANDIDATE_LIMIT))
        result = create_integer_param_between(lower, lower + extra), lower
    elif kind == "natural":
        zero_included = draw(st.booleans())
        value = 0 if zero_included else 1
        result = create_natural_param(zero_included=zero_included), value
    elif kind == "bounded_real":
        lower = draw(
            st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT)
        )
        extra = draw(st.integers(min_value=0, max_value=2 * _CANDIDATE_LIMIT))
        result = (
            create_real_param_between(float(lower), float(lower + extra)),
            float(lower),
        )
    elif kind == "single_valid_value":
        letter = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
        result = create_single_valid_value_param(letter), letter
    else:
        interval_param, interval_lower, interval_upper = draw(
            draw_interval_integer_param_with_bounds()
        )
        interval_value = (
            interval_lower
            if interval_lower is not None
            else (interval_upper if interval_upper is not None else 0)
        )
        result = interval_param, interval_value
    return result


# =============================================================================
# Property: a param over any domain round-trips through every format
# =============================================================================


@example(
    param=create_interval_integer_param_between(
        3, 5, is_lower_inclusive=True, is_upper_inclusive=False
    )
)
@given(param=draw_param_over_any_domain(include_empty=True))
def test_param_over_any_domain_round_trips_through_every_format(
    param: Param[Any],
) -> None:
    """Test a param over any domain kind round-trips through DICT, JSON, and BINARY.

    Oracle: ``assert_param_round_trips_in_all_formats``, the same
    structural-equivalence-based round-trip helper the sibling example
    tests use. An empty param round-trips like any other, so empties are
    drawn too.
    """
    assert_param_round_trips_in_all_formats(param)


# =============================================================================
# Property: a union/intersection/interval-arithmetic result round-trips
# =============================================================================


@pytest.mark.z3
@example(
    param=create_union_param(
        create_categorical_param({"a", "b"}), create_categorical_param({"b", "c"})
    )
)
@example(
    param=create_intersection_param(
        create_interval_integer_param_between(0, 10),
        create_interval_integer_param_between(5, 20),
    )
)
@example(
    param=create_intersection_param(
        create_categorical_param({"a", "b", "c"}),
        create_categorical_param({"b", "c", "d"}),
    )
)
@example(
    param=create_interval_integer_param_between(2, 3)
    * create_interval_integer_param_between(4, 5)
)
@given(param=draw_derived_param_result())
def test_derived_param_result_round_trips_through_every_format(
    param: Param[Any],
) -> None:
    """Test a union/intersection/addition/multiplication result round-trips.

    Marked z3: an intersection of numeric operands checks emptiness
    through ``Param.check_feasibility``, which may reach the solver.
    """
    assert_param_round_trips_in_all_formats(param)


# =============================================================================
# Property: a valid assignment round-trips through DICT
# =============================================================================


@given(case=draw_param_with_a_valid_value())
def test_param_assignment_round_trips_through_dict(
    case: tuple[Param[Any], Any],
) -> None:
    """Test a valid ``ParamAssignment`` round-trips through DICT.

    The restored assignment must be structurally equivalent to the
    original: a structurally equivalent param and an equal value, per
    ``ParamAssignment.is_structurally_equivalent``'s own docstring.
    """
    param, value = case
    assignment = param.assign(value)

    restored: ParamAssignment[Any] = ParamAssignment.deserialize(
        assignment.serialize(SerializationFormat.DICT), SerializationFormat.DICT
    )

    assert restored.is_structurally_equivalent(assignment)
