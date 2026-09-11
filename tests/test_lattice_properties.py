"""Hypothesis property tests for `Lattice` (P32).

Generates finite structures that are lattices by construction --- powersets
under inclusion, divisor sets under divisibility, chains, and the product
of two chains --- each paired with an independent, family-specific
"less than or equal" oracle (subset, divides, integer `<=`, componentwise
`<=`) that never calls back into `Lattice`. That oracle drives a brute-force
search over the element list for meet/join and bounds, which is the
external check the algebraic laws (commutative, associative, idempotent,
absorbing) are verified against. A separate case wraps an arbitrary random
poset (mirroring the P31 generator in `test_poset_properties.py`, which may
or may not be a lattice) as a `Lattice` to check `verify` against
`is_lattice` directly.
"""

import pytest

pytest.importorskip("hypothesis")

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.lattice import Lattice

pytestmark = pytest.mark.property

_POWERSET_SIZES: Final = (1, 2, 3)
_DIVISOR_NUMBERS: Final = (6, 12, 30, 36)
_CHAIN_LENGTHS: Final = (1, 2, 3, 4, 5)
_PRODUCT_CHAIN_LENGTHS: Final = (1, 2, 3)

_MIN_POSET_NODES = 2
_MAX_POSET_NODES = 6


@dataclass(frozen=True)
class LatticeCase:
    """A generated `Lattice` together with its elements and a ground-truth order.

    Attributes:
        lattice: The lattice under test.
        elements: Every element `lattice` holds, for iterating pairs and
            triples.
        less_than_or_equal: An order predicate independent of `lattice`
            itself (e.g. subset, divides, integer or tuple comparison),
            used to brute-force a reference meet or join.
    """

    lattice: Lattice[Any]
    elements: tuple[Any, ...]
    less_than_or_equal: Callable[[Any, Any], bool]


def build_powerset_lattice(size: int) -> LatticeCase:
    """Build the powerset of `range(size)` as a `Lattice` ordered by inclusion.

    Only covering pairs (no set strictly between the two) are added with
    `add_order`; `is_less_than`'s transitive closure over the poset
    reconstructs every other inclusion.
    """
    universe = tuple(range(size))
    subsets = tuple(
        frozenset(combination)
        for length in range(size + 1)
        for combination in itertools.combinations(universe, length)
    )
    lattice: Lattice[frozenset[int]] = Lattice()
    for subset in subsets:
        lattice.add_element(subset)
    for lower in subsets:
        for upper in subsets:
            if lower < upper and not any(lower < middle < upper for middle in subsets):
                lattice.add_order(lower, upper)
    return LatticeCase(lattice, subsets, frozenset.issubset)


def build_divisor_lattice(number: int) -> LatticeCase:
    """Build the divisors of `number` as a `Lattice` ordered by divisibility."""
    divisors = tuple(
        divisor for divisor in range(1, number + 1) if number % divisor == 0
    )
    lattice: Lattice[int] = Lattice()
    for divisor in divisors:
        lattice.add_element(divisor)
    for lower in divisors:
        for upper in divisors:
            if lower != upper and upper % lower == 0:
                covers = any(
                    middle not in (lower, upper)
                    and middle % lower == 0
                    and upper % middle == 0
                    for middle in divisors
                )
                if not covers:
                    lattice.add_order(lower, upper)
    return LatticeCase(lattice, divisors, lambda left, right: right % left == 0)


def build_chain_lattice(length: int) -> LatticeCase:
    """Build a total order on `range(length)` as a `Lattice`."""
    elements = tuple(range(length))
    lattice: Lattice[int] = Lattice()
    for element in elements:
        lattice.add_element(element)
    for index in range(length - 1):
        lattice.add_order(index, index + 1)
    return LatticeCase(lattice, elements, lambda left, right: left <= right)


def build_product_of_chains_lattice(
    first_length: int, second_length: int
) -> LatticeCase:
    """Build the product of two chains as a `Lattice` ordered componentwise."""
    elements = tuple(itertools.product(range(first_length), range(second_length)))
    lattice: Lattice[tuple[int, int]] = Lattice()
    for element in elements:
        lattice.add_element(element)
    for row, column in elements:
        if row + 1 < first_length:
            lattice.add_order((row, column), (row + 1, column))
        if column + 1 < second_length:
            lattice.add_order((row, column), (row, column + 1))
    return LatticeCase(
        lattice,
        elements,
        lambda left, right: left[0] <= right[0] and left[1] <= right[1],
    )


def build_lattice_case_strategy() -> st.SearchStrategy[LatticeCase]:
    """Return a strategy over powerset, divisor, chain, and product-of-chains cases."""
    return st.one_of(
        st.sampled_from(_POWERSET_SIZES).map(build_powerset_lattice),
        st.sampled_from(_DIVISOR_NUMBERS).map(build_divisor_lattice),
        st.sampled_from(_CHAIN_LENGTHS).map(build_chain_lattice),
        st.tuples(
            st.sampled_from(_PRODUCT_CHAIN_LENGTHS),
            st.sampled_from(_PRODUCT_CHAIN_LENGTHS),
        ).map(lambda lengths: build_product_of_chains_lattice(*lengths)),
    )


@st.composite
def draw_lattice_case_with_element(draw: st.DrawFn) -> tuple[LatticeCase, Any]:
    """Draw a generated lattice case together with one of its elements."""
    case = draw(build_lattice_case_strategy())
    element = draw(st.sampled_from(case.elements))
    return case, element


@st.composite
def draw_lattice_case_with_pair(draw: st.DrawFn) -> tuple[LatticeCase, Any, Any]:
    """Draw a generated lattice case together with a pair of its elements."""
    case = draw(build_lattice_case_strategy())
    first = draw(st.sampled_from(case.elements))
    second = draw(st.sampled_from(case.elements))
    return case, first, second


@st.composite
def draw_lattice_case_with_triple(
    draw: st.DrawFn,
) -> tuple[LatticeCase, Any, Any, Any]:
    """Draw a generated lattice case together with a triple of its elements."""
    case = draw(build_lattice_case_strategy())
    first = draw(st.sampled_from(case.elements))
    second = draw(st.sampled_from(case.elements))
    third = draw(st.sampled_from(case.elements))
    return case, first, second, third


def find_brute_force_bound(
    elements: Sequence[Any],
    less_than_or_equal: Callable[[Any, Any], bool],
    left: Any,
    right: Any,
    *,
    upper: bool,
) -> Any | None:
    """Return the unique minimal upper (or maximal lower) bound of two elements.

    Searches `elements` directly using `less_than_or_equal` rather than any
    `Lattice` method, so it is an oracle independent of `get_meet` and
    `get_join`. Returns `None` when no bound exists or more than one
    incomparable minimal (or maximal) bound does.
    """
    if upper:
        candidates = [
            element
            for element in elements
            if less_than_or_equal(left, element) and less_than_or_equal(right, element)
        ]
        extremal = [
            candidate
            for candidate in candidates
            if not any(
                candidate != other and less_than_or_equal(other, candidate)
                for other in candidates
            )
        ]
    else:
        candidates = [
            element
            for element in elements
            if less_than_or_equal(element, left) and less_than_or_equal(element, right)
        ]
        extremal = [
            candidate
            for candidate in candidates
            if not any(
                candidate != other and less_than_or_equal(candidate, other)
                for other in candidates
            )
        ]
    if len(extremal) == 1:
        return extremal[0]
    return None


# =============================================================================
# Meet and join: commutative, associative, idempotent, absorbing
# =============================================================================


@given(draw_lattice_case_with_pair())
def test_meet_agrees_with_the_brute_force_greatest_lower_bound(
    case_and_pair: tuple[LatticeCase, Any, Any],
) -> None:
    """Test `get_meet(x, y)` equals the brute-force bound under the family oracle."""
    case, first, second = case_and_pair
    expected = find_brute_force_bound(
        case.elements, case.less_than_or_equal, first, second, upper=False
    )
    assert case.lattice.get_meet(first, second) == expected


@given(draw_lattice_case_with_pair())
def test_join_agrees_with_the_brute_force_least_upper_bound(
    case_and_pair: tuple[LatticeCase, Any, Any],
) -> None:
    """Test `get_join(x, y)` equals the brute-force bound under the family oracle."""
    case, first, second = case_and_pair
    expected = find_brute_force_bound(
        case.elements, case.less_than_or_equal, first, second, upper=True
    )
    assert case.lattice.get_join(first, second) == expected


@given(draw_lattice_case_with_pair())
def test_meet_is_commutative(case_and_pair: tuple[LatticeCase, Any, Any]) -> None:
    """Test `get_meet(x, y)` equals `get_meet(y, x)`."""
    case, first, second = case_and_pair
    assert case.lattice.get_meet(first, second) == case.lattice.get_meet(second, first)


@given(draw_lattice_case_with_pair())
def test_join_is_commutative(case_and_pair: tuple[LatticeCase, Any, Any]) -> None:
    """Test `get_join(x, y)` equals `get_join(y, x)`."""
    case, first, second = case_and_pair
    assert case.lattice.get_join(first, second) == case.lattice.get_join(second, first)


@given(draw_lattice_case_with_triple())
def test_meet_is_associative(
    case_and_triple: tuple[LatticeCase, Any, Any, Any],
) -> None:
    """Test `get_meet(get_meet(x, y), z)` equals `get_meet(x, get_meet(y, z))`."""
    case, first, second, third = case_and_triple
    left_associated = case.lattice.get_meet(case.lattice.get_meet(first, second), third)
    right_associated = case.lattice.get_meet(
        first, case.lattice.get_meet(second, third)
    )
    assert left_associated == right_associated


@given(draw_lattice_case_with_triple())
def test_join_is_associative(
    case_and_triple: tuple[LatticeCase, Any, Any, Any],
) -> None:
    """Test `get_join(get_join(x, y), z)` equals `get_join(x, get_join(y, z))`."""
    case, first, second, third = case_and_triple
    left_associated = case.lattice.get_join(case.lattice.get_join(first, second), third)
    right_associated = case.lattice.get_join(
        first, case.lattice.get_join(second, third)
    )
    assert left_associated == right_associated


@given(draw_lattice_case_with_element())
def test_meet_is_idempotent(case_and_element: tuple[LatticeCase, Any]) -> None:
    """Test `get_meet(x, x)` equals `x`."""
    case, element = case_and_element
    assert case.lattice.get_meet(element, element) == element


@given(draw_lattice_case_with_element())
def test_join_is_idempotent(case_and_element: tuple[LatticeCase, Any]) -> None:
    """Test `get_join(x, x)` equals `x`."""
    case, element = case_and_element
    assert case.lattice.get_join(element, element) == element


@given(draw_lattice_case_with_pair())
def test_join_and_meet_absorb_each_other(
    case_and_pair: tuple[LatticeCase, Any, Any],
) -> None:
    """Test `get_join(x, get_meet(x, y))` and `get_meet(x, get_join(x, y))` equal `x`.

    Both directions of the absorption law.
    """
    case, first, second = case_and_pair
    assert case.lattice.get_join(first, case.lattice.get_meet(first, second)) == first
    assert case.lattice.get_meet(first, case.lattice.get_join(first, second)) == first


@given(draw_lattice_case_with_pair())
def test_least_upper_bound_is_below_every_brute_force_upper_bound(
    case_and_pair: tuple[LatticeCase, Any, Any],
) -> None:
    """Test the join is an upper bound of both operands and below every other one.

    Oracle: the family's independent `less_than_or_equal` predicate, applied
    to every element that predicate itself certifies as an upper bound.
    """
    case, first, second = case_and_pair
    least_upper_bound = case.lattice.get_least_upper_bound(first, second)
    assert case.less_than_or_equal(first, least_upper_bound)
    assert case.less_than_or_equal(second, least_upper_bound)

    upper_bounds = [
        element
        for element in case.elements
        if case.less_than_or_equal(first, element)
        and case.less_than_or_equal(second, element)
    ]
    for upper_bound in upper_bounds:
        assert case.less_than_or_equal(least_upper_bound, upper_bound)


@given(draw_lattice_case_with_pair())
def test_has_meet_and_has_join_agree_with_get_meet_and_get_join(
    case_and_pair: tuple[LatticeCase, Any, Any],
) -> None:
    """Test `has_meet(x, y)` iff `get_meet(x, y) is not None`, and likewise for join."""
    case, first, second = case_and_pair
    assert case.lattice.has_meet(first, second) == (
        case.lattice.get_meet(first, second) is not None
    )
    assert case.lattice.has_join(first, second) == (
        case.lattice.get_join(first, second) is not None
    )


@given(build_lattice_case_strategy())
def test_verify_reports_no_errors_for_a_generated_lattice(case: LatticeCase) -> None:
    """Test `verify()` reports no errors for every generated (always-valid) lattice."""
    assert case.lattice.is_lattice() is True
    assert case.lattice.verify().has_errors() is False


# =============================================================================
# `verify` versus `is_lattice` on an arbitrary (not-necessarily-a-lattice) poset
# =============================================================================


@st.composite
def draw_arbitrary_poset_as_lattice(draw: st.DrawFn) -> Lattice[int]:
    """Draw a random DAG and build it directly as a `Lattice`, valid or not.

    Mirrors the random-DAG generator in `test_poset_properties.py` (P31):
    nodes `0..n-1`, edges only from a smaller node to a larger one (acyclic
    by construction), added in a drawn order. Unlike the P32 families
    above, nothing here guarantees the result is a valid lattice.
    """
    node_count = draw(
        st.integers(min_value=_MIN_POSET_NODES, max_value=_MAX_POSET_NODES)
    )
    candidate_edges = [
        (lower, upper)
        for lower in range(node_count)
        for upper in range(lower + 1, node_count)
    ]
    included_edges = [edge for edge in candidate_edges if draw(st.booleans())]
    ordered_edges = draw(st.permutations(included_edges))

    lattice: Lattice[int] = Lattice()
    for node in range(node_count):
        lattice.add_element(node)
    for lower, upper in ordered_edges:
        lattice.add_order(lower, upper)
    return lattice


@given(draw_arbitrary_poset_as_lattice())
def test_verify_has_errors_iff_not_is_lattice_for_a_random_poset(
    lattice: Lattice[int],
) -> None:
    """Test `verify().has_errors()` equals `not is_lattice()` for an arbitrary poset."""
    report = lattice.verify()
    assert report.has_errors() == (not lattice.is_lattice())
