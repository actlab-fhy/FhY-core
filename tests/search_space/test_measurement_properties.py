"""Hypothesis properties of `Measurement.dominates`.

Measurements of three objectives with drawn directions and small values, so
ties are common: dominance is a strict partial order (irreflexive,
asymmetric, transitive), reversing every direction reverses it, and over
one compared objective it is that direction's strict order. A guard checks
a fixed sample holds dominating and non-dominating pairs.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from fhy_core.search_space import (
    ConfigurationKey,
    Direction,
    Measurement,
    Objective,
    non_dominated,
)

from ..strategies.settings import cap_max_examples
from .conftest import build_complete_configuration, build_tiling_space

pytestmark = pytest.mark.property

_NAMES = ("a", "b", "c")
_KEY: ConfigurationKey = build_complete_configuration(build_tiling_space()).key()

_DIRECTIONS = st.tuples(*(st.sampled_from(tuple(Direction)) for _ in _NAMES))
_VALUES = st.tuples(*(st.integers(0, 3).map(float) for _ in _NAMES))

_REVERSED = {
    Direction.MINIMIZE: Direction.MAXIMIZE,
    Direction.MAXIMIZE: Direction.MINIMIZE,
    Direction.REPORT: Direction.REPORT,
}


def _measure(
    directions: tuple[Direction, ...], values: tuple[float, ...]
) -> Measurement:
    """Return the successful measurement of `values` over `_NAMES` in `directions`."""
    return Measurement.ok(
        _KEY,
        [
            (Objective(name, direction), value)
            for name, direction, value in zip(_NAMES, directions, values, strict=True)
        ],
    )


@cap_max_examples(200)
@given(_DIRECTIONS, _VALUES)
def test_dominance_is_irreflexive(
    directions: tuple[Direction, ...], values: tuple[float, ...]
) -> None:
    """Test no measurement dominates itself."""
    measurement = _measure(directions, values)

    assert not measurement.dominates(measurement)


@cap_max_examples(200)
@given(_DIRECTIONS, _VALUES, _VALUES)
def test_dominance_is_asymmetric(
    directions: tuple[Direction, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> None:
    """Test two measurements never dominate each other."""
    first, second = _measure(directions, left), _measure(directions, right)

    assert not (first.dominates(second) and second.dominates(first))


@cap_max_examples(200)
@given(_DIRECTIONS, _VALUES, _VALUES, _VALUES)
def test_dominance_is_transitive(
    directions: tuple[Direction, ...],
    first: tuple[float, ...],
    second: tuple[float, ...],
    third: tuple[float, ...],
) -> None:
    """Test `a > b` and `b > c` give `a > c`."""
    a, b, c = (_measure(directions, values) for values in (first, second, third))

    if a.dominates(b) and b.dominates(c):
        assert a.dominates(c)


@cap_max_examples(200)
@given(_DIRECTIONS, _VALUES, _VALUES)
def test_reversing_the_directions_reverses_dominance(
    directions: tuple[Direction, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> None:
    """Test dominance under the reversed directions is dominance the other way."""
    backward = tuple(_REVERSED[direction] for direction in directions)

    ahead = _measure(directions, left).dominates(_measure(directions, right))
    behind = _measure(backward, right).dominates(_measure(backward, left))

    assert ahead == behind


@cap_max_examples(200)
@given(st.booleans(), st.integers(0, 3), st.integers(0, 3))
def test_dominance_over_one_objective_is_the_directions_order(
    maximize: bool, left: int, right: int
) -> None:
    """Test one compared objective dominates by `<` when minimizing, `>` maximizing."""
    direction = Direction.MAXIMIZE if maximize else Direction.MINIMIZE
    directions = (direction, Direction.REPORT, Direction.REPORT)

    dominates = _measure(directions, (float(left), 0.0, 0.0)).dominates(
        _measure(directions, (float(right), 9.0, 9.0))
    )

    assert dominates == (left > right if maximize else left < right)


def test_the_drawn_pairs_dominate_often_enough() -> None:
    """Test a fixed sample holds dominating and non-dominating comparable pairs."""
    pairs: list[tuple[Measurement, Measurement]] = []

    @settings(
        max_examples=200,
        derandomize=True,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(_DIRECTIONS, _VALUES, _VALUES)
    def collect(
        directions: tuple[Direction, ...],
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> None:
        if any(direction is not Direction.REPORT for direction in directions):
            pairs.append((_measure(directions, left), _measure(directions, right)))

    collect()
    dominating = sum(left.dominates(right) for left, right in pairs)

    assert len(pairs) >= 100
    assert dominating >= 20
    assert len(pairs) - dominating >= 20


# A population: each member is either a failure or a success with values.
_POPULATIONS = st.lists(st.one_of(st.none(), _VALUES), max_size=8)


def _population(
    directions: tuple[Direction, ...], cells: list[tuple[float, ...] | None]
) -> list[Measurement]:
    """Return the measurements of `cells`: a failure for `None`, else a success."""
    return [
        Measurement.failed(_KEY, "crashed")
        if cell is None
        else _measure(directions, cell)
        for cell in cells
    ]


@cap_max_examples(200)
@given(_DIRECTIONS, _POPULATIONS)
def test_no_returned_measurement_is_dominated_by_the_input(
    directions: tuple[Direction, ...], cells: list[tuple[float, ...] | None]
) -> None:
    """Test nothing `non_dominated` returns is dominated by any successful input.

    Oracle: the pairwise `dominates` of the input.
    """
    population = _population(directions, cells)
    successes = [m for m in population if m.is_ok()]

    front = non_dominated(population)

    for kept in front:
        assert kept.is_ok()
        assert not any(other.dominates(kept) for other in successes)


@cap_max_examples(200)
@given(_DIRECTIONS, _POPULATIONS)
def test_every_omitted_success_is_dominated_by_a_returned_one(
    directions: tuple[Direction, ...], cells: list[tuple[float, ...] | None]
) -> None:
    """Test each successful input left out is dominated by a returned measurement.

    Oracle: the pairwise `dominates` of the input.
    """
    population = _population(directions, cells)

    front = non_dominated(population)

    kept = {id(m) for m in front}
    for measurement in population:
        if measurement.is_ok() and id(measurement) not in kept:
            assert any(member.dominates(measurement) for member in front)


@cap_max_examples(200)
@given(_DIRECTIONS, _POPULATIONS)
def test_the_front_is_a_subsequence_of_the_input_without_failures(
    directions: tuple[Direction, ...], cells: list[tuple[float, ...] | None]
) -> None:
    """Test the front holds input measurements only, once each, in input order.

    Oracle: filtering the input by position with the pairwise `dominates`.
    """
    population = _population(directions, cells)
    expected = [
        m
        for m in population
        if m.is_ok() and not any(o.dominates(m) for o in population if o.is_ok())
    ]

    front = non_dominated(population)

    assert [id(m) for m in front] == [id(m) for m in expected]


def test_the_drawn_populations_hold_kept_and_omitted_measurements() -> None:
    """Test a fixed sample has fronts that drop a success, a failure, or neither."""
    fronts: list[tuple[int, int, int]] = []

    @settings(
        max_examples=200,
        derandomize=True,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(_DIRECTIONS, _POPULATIONS)
    def collect(
        directions: tuple[Direction, ...], cells: list[tuple[float, ...] | None]
    ) -> None:
        population = _population(directions, cells)
        front = non_dominated(population)
        successes = sum(m.is_ok() for m in population)
        fronts.append((len(population), successes, len(front)))

    collect()

    assert any(successes > front for _, successes, front in fronts)
    assert any(total > successes for total, successes, _ in fronts)
    assert any(front >= 2 for _, _, front in fronts)
    assert any(total == 0 for total, _, _ in fronts)
