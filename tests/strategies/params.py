"""Hypothesis strategies for `Param` instances over every domain kind.

Bounds are built ordered by construction (never rejected): the lower
and upper bound of an interval, and the lower and upper bound of a
candidate range, are drawn independently and then reconciled (swapped
or widened) rather than filtered, so no draw is ever discarded for
being out of order.
"""

from typing import Any, Final

from hypothesis import strategies as st

from fhy_core.symbolic.param import (
    Param,
    create_categorical_param,
    create_integer_param_between,
    create_natural_param,
    create_ordinal_param,
    create_permutation_param,
    create_real_param_between,
    create_single_valid_value_param,
)

from ..symbolic.param.conftest import build_interval_integer_param

__all__ = [
    "build_categorical_value_set_strategy",
    "build_optional_bound_strategy",
    "build_ordinal_value_set_strategy",
    "build_permutation_member_set_strategy",
    "draw_bounded_integer_param",
    "draw_bounded_real_param",
    "draw_categorical_param",
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
    "draw_same_kind_param_pair_with_candidate",
    "draw_single_valid_value_param",
]

_CATEGORICAL_ALPHABET: Final = tuple("abcdefgh")
_CANDIDATE_LIMIT: Final = 25


def build_optional_bound_strategy(limit: int = 25) -> st.SearchStrategy[int | None]:
    """Return a strategy for an optional bound in ``[-limit, limit]``."""
    return st.one_of(st.none(), st.integers(min_value=-limit, max_value=limit))


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
def draw_interval_integer_param_with_bounds(
    draw: st.DrawFn, limit: int = 25
) -> tuple[Param[int], int | None, int | None]:
    """Draw an interval-integer param together with the bounds it was built from.

    For a property that needs the raw ``(lower, upper)`` endpoints to
    compute an expected result alongside the param itself, such as an
    interval-arithmetic hull.
    """
    lower, upper = draw(draw_ordered_optional_bounds(limit))
    return build_interval_integer_param(lower, upper), lower, upper


@st.composite
def draw_interval_integer_param(draw: st.DrawFn, limit: int = 25) -> Param[int]:
    """Draw an interval-integer param over optional, ordered bounds."""
    param, _, _ = draw(draw_interval_integer_param_with_bounds(limit))
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
def draw_bounded_integer_param(draw: st.DrawFn, limit: int = 25) -> Param[int]:
    """Draw an integer param bounded to ``[lower, upper]``, ``lower <= upper``."""
    lower = draw(st.integers(min_value=-limit, max_value=limit))
    extra = draw(st.integers(min_value=0, max_value=2 * limit))
    upper = min(lower + extra, limit)
    return create_integer_param_between(lower, upper)


@st.composite
def draw_natural_param(draw: st.DrawFn) -> Param[int]:
    """Draw a natural-number param, with or without zero included."""
    zero_included = draw(st.booleans())
    return create_natural_param(zero_included=zero_included)


@st.composite
def draw_bounded_real_param(draw: st.DrawFn) -> Param[str | float]:
    """Draw a real param bounded to ``[lower, upper]`` with integer-valued bounds."""
    lower = draw(st.integers(min_value=-_CANDIDATE_LIMIT, max_value=_CANDIDATE_LIMIT))
    extra = draw(st.integers(min_value=0, max_value=2 * _CANDIDATE_LIMIT))
    upper = lower + extra
    return create_real_param_between(float(lower), float(upper))


@st.composite
def draw_single_valid_value_param(draw: st.DrawFn) -> Param[str]:
    """Draw a param admitting exactly one letter from a fixed alphabet."""
    value = draw(st.sampled_from(_CATEGORICAL_ALPHABET))
    return create_single_valid_value_param(value)


@st.composite
def draw_param_over_any_domain(draw: st.DrawFn) -> Param[Any]:
    """Draw a param over any of this module's domain kinds."""
    return draw(
        st.one_of(
            draw_interval_integer_param(),
            draw_ordinal_param(),
            draw_categorical_param(),
            draw_permutation_param(),
            draw_bounded_integer_param(),
            draw_natural_param(),
            draw_bounded_real_param(),
            draw_single_valid_value_param(),
        )
    )


@st.composite
def _draw_integer_family_param_with_candidate(
    draw: st.DrawFn,
) -> tuple[Param[Any], Any]:
    """Draw an interval, bounded, or natural integer param with an int candidate."""
    param = draw(
        st.one_of(
            draw_interval_integer_param(),
            draw_bounded_integer_param(),
            draw_natural_param(),
        )
    )
    candidate = draw(
        st.integers(min_value=-_CANDIDATE_LIMIT - 5, max_value=_CANDIDATE_LIMIT + 5)
    )
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
    """Draw a real param with an integer-valued float candidate."""
    param = draw(draw_bounded_real_param())
    candidate = float(
        draw(
            st.integers(min_value=-_CANDIDATE_LIMIT - 5, max_value=_CANDIDATE_LIMIT + 5)
        )
    )
    return param, candidate


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
def draw_param_with_candidate(draw: st.DrawFn) -> tuple[Param[Any], Any]:
    """Draw a param over some domain together with a candidate from a wider superset.

    The candidate's range is a superset of the domain's admissible
    values, so both valid and invalid candidates occur: ints for
    integer/interval/natural/ordinal params, alphabet letters for
    categorical/single-valid-value params, an integer-valued float for
    real params, and a permutation of the member set (or of the member
    set plus one extra letter) for permutation params.
    """
    result: tuple[Param[Any], Any] = draw(
        st.one_of(
            _draw_integer_family_param_with_candidate(),
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
def draw_overlapping_bounded_integer_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` bounded-integer params whose ranges all contain a pivot."""
    pivot = draw(st.integers(min_value=-limit, max_value=limit))
    members = tuple(
        create_integer_param_between(
            pivot - draw(st.integers(min_value=0, max_value=_OVERLAP_SPAN)),
            pivot + draw(st.integers(min_value=0, max_value=_OVERLAP_SPAN)),
        )
        for _ in range(size)
    )
    candidate = draw(
        st.integers(min_value=-limit - _OVERLAP_SPAN, max_value=limit + _OVERLAP_SPAN)
    )
    return members, candidate


@st.composite
def draw_overlapping_natural_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` natural-number params; any such group already overlaps.

    Every natural-number domain admits infinitely many shared values (at
    least every sufficiently large integer), so no pivot construction is
    needed to keep the intersection non-empty.
    """
    members = tuple(draw(draw_natural_param()) for _ in range(size))
    candidate = draw(st.integers(min_value=-5, max_value=limit))
    return members, candidate


@st.composite
def draw_overlapping_real_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[str | float], ...], float]:
    """Draw ``size`` bounded-real params whose ranges all contain a pivot."""
    pivot = draw(st.integers(min_value=-limit, max_value=limit))
    members = tuple(
        create_real_param_between(
            float(pivot - draw(st.integers(min_value=0, max_value=_OVERLAP_SPAN))),
            float(pivot + draw(st.integers(min_value=0, max_value=_OVERLAP_SPAN))),
        )
        for _ in range(size)
    )
    candidate = float(
        draw(
            st.integers(
                min_value=-limit - _OVERLAP_SPAN, max_value=limit + _OVERLAP_SPAN
            )
        )
    )
    return members, candidate


@st.composite
def draw_overlapping_interval_integer_group(
    draw: st.DrawFn, size: int = 2, limit: int = 20
) -> tuple[tuple[Param[int], ...], int]:
    """Draw ``size`` interval-integer params whose ranges all contain a pivot."""
    pivot = draw(st.integers(min_value=-limit, max_value=limit))
    lower_choice = st.one_of(
        st.none(), st.integers(min_value=pivot - _OVERLAP_SPAN, max_value=pivot)
    )
    upper_choice = st.one_of(
        st.none(), st.integers(min_value=pivot, max_value=pivot + _OVERLAP_SPAN)
    )
    members = tuple(
        build_interval_integer_param(draw(lower_choice), draw(upper_choice))
        for _ in range(size)
    )
    candidate = draw(
        st.integers(min_value=-limit - _OVERLAP_SPAN, max_value=limit + _OVERLAP_SPAN)
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
