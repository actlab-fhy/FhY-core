"""Hypothesis strategies for `Param` instances over every domain kind.

Bounds are built ordered by construction (never rejected): the lower
and upper bound of an interval, and the lower and upper bound of a
candidate range, are drawn independently and then reconciled (swapped
or widened) rather than filtered, so no draw is ever discarded for
being out of order.

An integer bound is usually small, in ``[-limit, limit]``, and now and
then offset by a wide anchor (``2**31``, ``2**53``, ``2**63``, ``2**64``,
or ``2**100``, either sign), so draws reach the int64 and float-precision
edges and far beyond them. A real bound is an integer multiple of a
scale that is usually ``1.0`` and now and then a wide power of two, so
every real bound is an exact ``float`` and no two distinct multiples
round together.

Each end of a bounded integer or real param is drawn inclusive or
exclusive. No strategy draws an empty param unless its caller passes
``include_empty=True``: a property whose oracle needs a member of the
param keeps the default, and one whose oracle also holds for an empty
param opts in.
"""

from typing import Any, Final

from hypothesis import strategies as st

from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param_between,
    create_interval_integer_param,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param_between,
    create_single_valid_value_param,
)

from ..symbolic.param.conftest import build_interval_integer_param

__all__ = [
    "WIDE_INTEGER_ANCHORS",
    "WIDE_REAL_SCALES",
    "build_categorical_value_set_strategy",
    "build_optional_bound_strategy",
    "build_ordinal_value_set_strategy",
    "build_permutation_member_set_strategy",
    "draw_bounded_integer_param",
    "draw_bounded_real_param",
    "draw_categorical_param",
    "draw_integer_bound",
    "draw_interval_integer_param",
    "draw_interval_integer_param_with_bounds",
    "draw_natural_param",
    "draw_ordered_optional_bounds",
    "draw_ordinal_param",
    "draw_overlapping_bounded_integer_group",
    "draw_overlapping_categorical_group",
    "draw_overlapping_intersection_eligible_group",
    "draw_overlapping_interval_integer_group",
    "draw_overlapping_natural_group",
    "draw_overlapping_ordinal_group",
    "draw_overlapping_permutation_group",
    "draw_overlapping_real_group",
    "draw_overlapping_union_eligible_group",
    "draw_param_over_any_domain",
    "draw_param_with_candidate",
    "draw_permutation_param",
    "draw_real_scale",
    "draw_same_kind_param_pair_with_candidate",
    "draw_single_valid_value_param",
]

_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")
_CANDIDATE_LIMIT: Final = 25

WIDE_INTEGER_ANCHORS: Final = (2**31, 2**53, 2**63, 2**64, 2**100)
"""Magnitudes a wide integer bound is offset from, each drawn with either sign."""

WIDE_REAL_SCALES: Final = (2.0**31, 2.0**53, 2.0**64, 2.0**100, 2.0**1000)
"""Wide powers of two a real bound is a small integer multiple of."""

# Shrinks toward False, so a failing example shrinks toward the common shape.
_RARELY_TRUE: Final = st.integers(min_value=1, max_value=4).map(lambda roll: roll == 4)


@st.composite
def draw_integer_bound(draw: st.DrawFn, limit: int = 25) -> int:
    """Draw an integer bound, usually in ``[-limit, limit]``, now and then far beyond.

    A wide draw offsets that small value by one of ``WIDE_INTEGER_ANCHORS``
    with either sign, so it lands within ``limit`` of that anchor.
    """
    anchor = 0
    if draw(_RARELY_TRUE):
        magnitude = draw(st.sampled_from(WIDE_INTEGER_ANCHORS))
        anchor = magnitude if draw(st.booleans()) else -magnitude
    return anchor + draw(st.integers(min_value=-limit, max_value=limit))


@st.composite
def draw_real_scale(draw: st.DrawFn) -> float:
    """Draw ``1.0`` usually, and now and then one of ``WIDE_REAL_SCALES``."""
    if draw(_RARELY_TRUE):
        scale: float = draw(st.sampled_from(WIDE_REAL_SCALES))
        return scale
    return 1.0


def build_optional_bound_strategy(limit: int = 25) -> st.SearchStrategy[int | None]:
    """Return a strategy for ``None`` or an integer bound from `draw_integer_bound`."""
    return st.one_of(st.none(), draw_integer_bound(limit))


@st.composite
def draw_ordered_optional_bounds(
    draw: st.DrawFn, limit: int = 25
) -> tuple[int | None, int | None]:
    """Draw a ``(lower, upper)`` pair with ``lower <= upper`` whenever both are present.

    Each bound is drawn independently, then swapped if both are present
    and out of order, so no draw is ever discarded for being unordered.
    """
    first = draw(build_optional_bound_strategy(limit))
    second = draw(build_optional_bound_strategy(limit))
    if first is not None and second is not None and first > second:
        first, second = second, first
    return first, second


@st.composite
def _draw_empty_interval_integer_param_near(
    draw: st.DrawFn, lower: int, upper: int
) -> tuple[Param[int], int, int]:
    """Draw an empty interval-integer param near ``[lower, upper]``, with its ends.

    Either consistent exclusive bounds one apart, ``(lower, lower + 1)``,
    which the between factory builds empty, or inclusive bounds crossed
    past each other, ``[upper + 1, lower]``, which only chained bound
    constraints build. The returned least and greatest members are
    crossed the same way, the least exceeding the greatest.
    """
    if draw(st.booleans()):
        exclusive_param = build_interval_integer_param(
            lower, lower + 1, is_lower_inclusive=False, is_upper_inclusive=False
        )
        return exclusive_param, lower + 1, lower
    crossed_param = (
        create_interval_integer_param()
        .add_lower_bound_constraint(upper + 1)
        .add_upper_bound_constraint(lower)
    )
    return crossed_param, upper + 1, lower


@st.composite
def draw_interval_integer_param_with_bounds(
    draw: st.DrawFn, limit: int = 25, *, include_empty: bool = False
) -> tuple[Param[int], int | None, int | None]:
    """Draw an interval-integer param with the least and greatest integers it admits.

    For a property that needs the param's integer endpoints to compute an
    expected result alongside the param itself, such as an
    interval-arithmetic hull. Each bounded end is drawn inclusive or
    exclusive; an exclusive end is written one step outside the integer
    it admits, so the returned ``(lower, upper)`` are the param's extreme
    members whichever form was drawn, ``None`` on an unbounded side.

    With ``include_empty``, a param bounded on both sides is sometimes
    built empty instead, and its returned ``lower`` exceeds its ``upper``.
    """
    lower, upper = draw(draw_ordered_optional_bounds(limit))
    if (
        include_empty
        and lower is not None
        and upper is not None
        and draw(st.booleans())
    ):
        empty: tuple[Param[int], int, int] = draw(
            _draw_empty_interval_integer_param_near(lower, upper)
        )
        return empty
    is_lower_inclusive = draw(st.booleans())
    is_upper_inclusive = draw(st.booleans())
    param = build_interval_integer_param(
        _shift_bound_outward(lower, -1, is_inclusive=is_lower_inclusive),
        _shift_bound_outward(upper, 1, is_inclusive=is_upper_inclusive),
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )
    return param, lower, upper


def _shift_bound_outward(
    member: int | None, direction: int, *, is_inclusive: bool
) -> int | None:
    """Return the bound that admits ``member`` as the extreme one, in the given form.

    An inclusive bound is ``member`` itself; an exclusive one sits one
    step past it in ``direction`` (``-1`` below a lower end, ``1`` above
    an upper end). ``None`` stays unbounded.
    """
    if member is None or is_inclusive:
        return member
    return member + direction


@st.composite
def draw_interval_integer_param(
    draw: st.DrawFn, limit: int = 25, *, include_empty: bool = False
) -> Param[int]:
    """Draw an interval-integer param over optional, ordered bounds.

    Drawn as `draw_interval_integer_param_with_bounds` draws it, which
    says what ``include_empty`` adds.
    """
    param, _, _ = draw(
        draw_interval_integer_param_with_bounds(limit, include_empty=include_empty)
    )
    return param


def build_ordinal_value_set_strategy(
    min_size: int = 1, max_size: int = 6, limit: int = 12
) -> st.SearchStrategy[list[int]]:
    """Return a strategy for a sorted, unique list of ints in ``[-limit, limit]``."""
    return st.lists(
        st.integers(min_value=-limit, max_value=limit),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    ).map(sorted)


def build_categorical_value_set_strategy(
    min_size: int = 1, max_size: int = 5
) -> st.SearchStrategy[list[str]]:
    """Return a strategy for a unique list of letters from a fixed alphabet."""
    return st.lists(
        st.sampled_from(_CATEGORICAL_ALPHABET),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


def build_permutation_member_set_strategy(
    min_size: int = 1, max_size: int = 4
) -> st.SearchStrategy[list[str]]:
    """Return a strategy for a unique list of letters from a fixed alphabet."""
    return st.lists(
        st.sampled_from(_CATEGORICAL_ALPHABET),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


@st.composite
def draw_ordinal_param(draw: st.DrawFn) -> Param[int]:
    """Draw an ordinal param over a finite, sorted set of ints."""
    values = draw(build_ordinal_value_set_strategy())
    return create_ordinal_param(values)


@st.composite
def draw_categorical_param(draw: st.DrawFn) -> Param[str]:
    """Draw a categorical param over a finite set of letters."""
    categories = draw(build_categorical_value_set_strategy())
    return create_categorical_param(categories)


@st.composite
def draw_permutation_param(draw: st.DrawFn) -> Param[tuple[str, ...]]:
    """Draw a permutation param over a fixed, ordered set of letters."""
    members = draw(build_permutation_member_set_strategy())
    return create_permutation_param(members)


@st.composite
def draw_bounded_integer_param(
    draw: st.DrawFn, limit: int = 25, *, include_empty: bool = False
) -> Param[int]:
    """Draw an integer param bounded on both sides, each end inclusive or exclusive.

    The least and greatest members are drawn ordered, and an exclusive end
    is written one step outside its member, so the param admits both.
    With ``include_empty``, the param is sometimes built empty instead,
    from consistent exclusive bounds one apart.
    """
    lower, upper = sorted(
        (draw(draw_integer_bound(limit)), draw(draw_integer_bound(limit)))
    )
    if include_empty and draw(_RARELY_TRUE):
        return create_integer_param_between(
            lower, lower + 1, is_lower_inclusive=False, is_upper_inclusive=False
        )
    is_lower_inclusive = draw(st.booleans())
    is_upper_inclusive = draw(st.booleans())
    return create_integer_param_between(
        lower if is_lower_inclusive else lower - 1,
        upper if is_upper_inclusive else upper + 1,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )


@st.composite
def draw_natural_param(draw: st.DrawFn) -> Param[int]:
    """Draw a natural-number param, with or without zero included."""
    zero_included = draw(st.booleans())
    return create_natural_param(zero_included=zero_included)


@st.composite
def draw_bounded_real_param(draw: st.DrawFn, limit: int = 25) -> Param[str | float]:
    """Draw a real param bounded on both sides, each end inclusive or exclusive.

    Both bounds are multiples in ``[-limit, limit]`` of one scale from
    `draw_real_scale`. Equal multiples with an exclusive side would enclose
    nothing, so the upper multiple is widened by one instead, and the
    param is never empty.
    """
    scale = draw(draw_real_scale())
    lower, upper = sorted(
        (
            draw(st.integers(min_value=-limit, max_value=limit)),
            draw(st.integers(min_value=-limit, max_value=limit)),
        )
    )
    is_lower_inclusive = draw(st.booleans())
    is_upper_inclusive = draw(st.booleans())
    if lower == upper and not (is_lower_inclusive and is_upper_inclusive):
        upper += 1
    return create_real_param_between(
        lower * scale,
        upper * scale,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )


@st.composite
def draw_single_valid_value_param(draw: st.DrawFn) -> Param[str]:
    """Draw a param admitting exactly one letter from a fixed alphabet."""
    value = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return create_single_valid_value_param(value)


@st.composite
def draw_param_over_any_domain(
    draw: st.DrawFn, *, include_empty: bool = False
) -> Param[Any]:
    """Draw a param over any of this module's domain kinds.

    ``include_empty`` is forwarded to the interval and bounded integer
    strategies, the kinds that can be built empty.
    """
    result: Param[Any] = draw(
        st.one_of(
            draw_interval_integer_param(include_empty=include_empty),
            draw_ordinal_param(),
            draw_categorical_param(),
            draw_permutation_param(),
            draw_bounded_integer_param(include_empty=include_empty),
            draw_natural_param(),
            draw_bounded_real_param(),
            draw_single_valid_value_param(),
        )
    )
    return result


@st.composite
def _draw_integer_family_param_with_candidate(
    draw: st.DrawFn, *, include_empty: bool
) -> tuple[Param[Any], Any]:
    """Draw an interval, bounded, or natural integer param with an int candidate.

    The candidate is drawn the way a bound is, so it lands near a wide
    bound as well as a small one.
    """
    param = draw(
        st.one_of(
            draw_interval_integer_param(include_empty=include_empty),
            draw_bounded_integer_param(include_empty=include_empty),
            draw_natural_param(),
        )
    )
    candidate = draw(draw_integer_bound(_CANDIDATE_LIMIT + 5))
    return param, candidate


@st.composite
def _draw_categorical_family_param_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[Any], Any]:
    """Draw a categorical or single-valid-value param with a letter candidate."""
    param = draw(st.one_of(draw_categorical_param(), draw_single_valid_value_param()))
    candidate = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return param, candidate


@st.composite
def _draw_ordinal_param_with_candidate(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw an ordinal param with an int candidate."""
    param = draw(draw_ordinal_param())
    candidate = draw(
        st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT)
    )
    return param, candidate


@st.composite
def _draw_real_param_with_candidate(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw a real param with a candidate that is a multiple of a real scale.

    The candidate's scale is drawn the way a bound's is, so it lands near
    a widely scaled bound as well as a unit-scaled one.
    """
    param = draw(draw_bounded_real_param())
    multiple = draw(
        st.integers(min_value=-_CANDIDATE_LIMIT - 5, max_value=_CANDIDATE_LIMIT + 5)
    )
    return param, multiple * draw(draw_real_scale())


@st.composite
def _draw_permutation_param_with_candidate(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw a permutation param with a candidate permutation of it, or one superset."""
    members = draw(build_permutation_member_set_strategy())
    param = create_permutation_param(members)
    extra_pool = [letter for letter in _CATEGORICAL_ALPHABET if letter not in members]
    candidate_members = list(members)
    if extra_pool:
        if draw(st.booleans()):
            candidate_members.append(draw(st.sampled_from(extra_pool)))
    shuffled = draw(st.permutations(candidate_members))
    return param, tuple(shuffled)


@st.composite
def draw_param_with_candidate(
    draw: st.DrawFn, *, include_empty: bool = False
) -> tuple[Param[Any], Any]:
    """Draw a param over some domain together with a candidate from a wider superset.

    The candidate's range is a superset of the domain's admissible
    values, so both valid and invalid candidates occur: ints for
    integer/interval/natural/ordinal params, alphabet letters for
    categorical/single-valid-value params, a float multiple of a real
    scale for real params, and a permutation of the member set (or of the
    member set plus one extra letter) for permutation params.
    ``include_empty`` is forwarded to the integer-family strategies.
    """
    result: tuple[Param[Any], Any] = draw(
        st.one_of(
            _draw_integer_family_param_with_candidate(include_empty=include_empty),
            _draw_categorical_family_param_with_candidate(),
            _draw_ordinal_param_with_candidate(),
            _draw_real_param_with_candidate(),
            _draw_permutation_param_with_candidate(),
        )
    )
    return result


@st.composite
def draw_same_kind_param_pair_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[Any], Param[Any], Any]:
    """Draw two params of one union-eligible kind with a shared-superset candidate.

    Both ordinal or both categorical: the two domain kinds
    ``create_union_param`` accepts.
    """
    if draw(st.booleans()):
        left: Param[Any] = draw(draw_ordinal_param())
        right: Param[Any] = draw(draw_ordinal_param())
        candidate: Any = draw(
            st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT)
        )
    else:
        left = draw(draw_categorical_param())
        right = draw(draw_categorical_param())
        candidate = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return left, right, candidate


# =============================================================================
# Overlapping groups: guaranteed non-empty intersection, for the set-algebra
# properties. `create_intersection_param` raises `ParamError` on an empty
# result, so every group below shares a pivot value (or, for permutation, a
# member set) across all of its params by construction, rather than drawing
# independently and discarding disjoint groups.
# =============================================================================

_OVERLAP_SPAN: Final = 10
_OVERLAP_CANDIDATE_REACH: Final = _OVERLAP_SPAN + 5


@st.composite
def _draw_distance_from_pivot(draw: st.DrawFn) -> tuple[int, bool]:
    """Draw how many steps a bound sits from a pivot, and whether it is inclusive.

    An inclusive bound lies up to ``_OVERLAP_SPAN`` steps away and an
    exclusive one between one and ``_OVERLAP_SPAN + 1`` steps, so the
    pivot is admitted either way.
    """
    is_inclusive = draw(st.booleans())
    distance = draw(st.integers(min_value=0, max_value=_OVERLAP_SPAN))
    return (distance if is_inclusive else distance + 1), is_inclusive


@st.composite
def draw_overlapping_ordinal_group(
    draw: st.DrawFn, size: int = 2, limit: int = 12
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` ordinal params sharing a pivot value, plus a candidate."""
    pivot = draw(st.integers(min_value=-limit, max_value=limit))
    members = tuple(
        create_ordinal_param(
            sorted({pivot} | draw(st.sets(st.integers(-limit, limit), max_size=5)))
        )
        for _ in range(size)
    )
    candidate = draw(st.integers(min_value=-limit - 5, max_value=limit + 5))
    return members, candidate


@st.composite
def draw_overlapping_categorical_group(
    draw: st.DrawFn, size: int = 2
) -> tuple[tuple[Param[str], ...], str]:
    """Draw ``size`` categorical params sharing a pivot category, plus a candidate."""
    pivot = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    members = tuple(
        create_categorical_param(
            {pivot}
            | set(draw(build_categorical_value_set_strategy(min_size=0, max_size=4)))
        )
        for _ in range(size)
    )
    candidate = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return members, candidate


@st.composite
def draw_overlapping_permutation_group(
    draw: st.DrawFn, size: int = 2
) -> tuple[tuple[Param[tuple[str, ...]], ...], tuple[str, ...]]:
    """Draw ``size`` permutation params over one member set, each reordered.

    Two permutation domains intersect only when they range over the same
    member set (order-independent), so every member here is built from
    one shared set shuffled independently, which keeps every pairing's
    intersection non-empty without making the params identical.
    """
    base_members = draw(build_permutation_member_set_strategy())
    members = tuple(
        create_permutation_param(draw(st.permutations(base_members)))
        for _ in range(size)
    )
    candidate = tuple(draw(st.permutations(base_members)))
    return members, candidate


@st.composite
def _draw_bounded_integer_param_around_pivot(draw: st.DrawFn, pivot: int) -> Param[int]:
    """Draw a bounded-integer param admitting ``pivot``, each end in either form."""
    lower_distance, is_lower_inclusive = draw(_draw_distance_from_pivot())
    upper_distance, is_upper_inclusive = draw(_draw_distance_from_pivot())
    return create_integer_param_between(
        pivot - lower_distance,
        pivot + upper_distance,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )


@st.composite
def draw_overlapping_bounded_integer_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` bounded-integer params whose ranges all contain a pivot.

    The pivot is drawn the way `draw_integer_bound` draws a bound, and the
    candidate within a few steps beyond the widest member around it.
    """
    pivot = draw(draw_integer_bound(limit))
    members = tuple(
        draw(_draw_bounded_integer_param_around_pivot(pivot)) for _ in range(size)
    )
    candidate = pivot + draw(
        st.integers(
            min_value=-_OVERLAP_CANDIDATE_REACH, max_value=_OVERLAP_CANDIDATE_REACH
        )
    )
    return members, candidate


@st.composite
def draw_overlapping_natural_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` natural-number params; any such group already overlaps.

    Every natural-number domain admits infinitely many shared values (at
    least every sufficiently large integer), so no pivot construction is
    needed to keep the intersection non-empty. The candidate is drawn the
    way `draw_integer_bound` draws a bound.
    """
    members = tuple(draw(draw_natural_param()) for _ in range(size))
    candidate = draw(draw_integer_bound(limit))
    return members, candidate


@st.composite
def draw_overlapping_real_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[str | float], ...], float]:
    """Draw ``size`` bounded-real params whose ranges all contain a pivot.

    Every bound and the candidate are integer multiples of one scale from
    `draw_real_scale`, so each is an exact ``float``.
    """
    scale = draw(draw_real_scale())
    pivot = draw(st.integers(min_value=-limit, max_value=limit))
    members: list[Param[str | float]] = []
    for _ in range(size):
        lower_distance, is_lower_inclusive = draw(_draw_distance_from_pivot())
        upper_distance, is_upper_inclusive = draw(_draw_distance_from_pivot())
        members.append(
            create_real_param_between(
                (pivot - lower_distance) * scale,
                (pivot + upper_distance) * scale,
                is_lower_inclusive=is_lower_inclusive,
                is_upper_inclusive=is_upper_inclusive,
            )
        )
    candidate = scale * (
        pivot
        + draw(
            st.integers(
                min_value=-_OVERLAP_CANDIDATE_REACH, max_value=_OVERLAP_CANDIDATE_REACH
            )
        )
    )
    return tuple(members), candidate


@st.composite
def _draw_interval_integer_param_around_pivot(
    draw: st.DrawFn, pivot: int
) -> Param[int]:
    """Draw an interval-integer param admitting ``pivot``, each end optional."""
    lower_distance, is_lower_inclusive = draw(_draw_distance_from_pivot())
    upper_distance, is_upper_inclusive = draw(_draw_distance_from_pivot())
    return build_interval_integer_param(
        pivot - lower_distance if draw(st.booleans()) else None,
        pivot + upper_distance if draw(st.booleans()) else None,
        is_lower_inclusive=is_lower_inclusive,
        is_upper_inclusive=is_upper_inclusive,
    )


@st.composite
def draw_overlapping_interval_integer_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` interval-integer params whose ranges all contain a pivot.

    The pivot is drawn the way `draw_integer_bound` draws a bound, and the
    candidate within a few steps beyond the widest member around it.
    """
    pivot = draw(draw_integer_bound(limit))
    members = tuple(
        draw(_draw_interval_integer_param_around_pivot(pivot)) for _ in range(size)
    )
    candidate = pivot + draw(
        st.integers(
            min_value=-_OVERLAP_CANDIDATE_REACH, max_value=_OVERLAP_CANDIDATE_REACH
        )
    )
    return members, candidate


@st.composite
def draw_overlapping_union_eligible_group(
    draw: st.DrawFn, size: int = 2
) -> tuple[tuple[Param[Any], ...], Any]:
    """Draw ``size`` overlapping params of a kind ``create_union_param`` accepts.

    Ordinal and categorical are the two domain kinds it accepts.
    """
    result: tuple[tuple[Param[Any], ...], Any] = draw(
        st.one_of(
            draw_overlapping_ordinal_group(size),
            draw_overlapping_categorical_group(size),
        )
    )
    return result


@st.composite
def draw_overlapping_intersection_eligible_group(
    draw: st.DrawFn, size: int = 2
) -> tuple[tuple[Param[Any], ...], Any]:
    """Draw ``size`` overlapping params of a kind ``create_intersection_param`` accepts.

    Every domain kind it accepts: ordinal, categorical, permutation,
    bounded integer, natural, bounded real, and interval-integer.
    """
    result: tuple[tuple[Param[Any], ...], Any] = draw(
        st.one_of(
            draw_overlapping_ordinal_group(size),
            draw_overlapping_categorical_group(size),
            draw_overlapping_permutation_group(size),
            draw_overlapping_bounded_integer_group(size),
            draw_overlapping_natural_group(size),
            draw_overlapping_real_group(size),
            draw_overlapping_interval_integer_group(size),
        )
    )
    return result
