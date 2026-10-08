"""Hypothesis properties of traces, replay, enumeration and sampling.

Spaces and configurations are drawn as in `test_search_space_properties.py`:
a shape built with fresh names, and a configuration read from a list of
picks. The enumeration properties draw smaller spaces, so that the
configurations a space has can be listed. Each property that is vacuous on
spaces without a choice has a guard test: a fixed sample of its strategy
must hold enough interesting spaces.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from fhy_core.search_space import (
    Cardinality,
    CardinalityKind,
    Configuration,
    RandomOracle,
    Rng,
    Trace,
    TraceError,
)

from ..strategies.settings import cap_max_examples
from .test_search_space_properties import (
    _DOMAINS,
    _PICKS,
    _SPACE_SHAPES,
    _build_space,
    _entries,
    _SpaceShape,
)

pytestmark = pytest.mark.property

_SEEDS = st.integers(min_value=0, max_value=2**32)

_SMALL_SHAPES: st.SearchStrategy[_SpaceShape] = st.tuples(
    st.lists(_DOMAINS, max_size=1),
    st.lists(
        st.lists(
            st.tuples(st.lists(_DOMAINS, max_size=1), st.just([])),
            min_size=1,
            max_size=3,
        ),
        max_size=2,
    ),
)

_SAMPLE_SIZE = 60

_NOTHING_TO_MUTATE = "no decision of the configuration has another admissible value"


def _has_a_choice(shape: _SpaceShape) -> bool:
    return bool(shape[1])


def _has_alternatives_of_different_sizes(shape: _SpaceShape) -> bool:
    """Return whether a choice's alternatives hold different numbers of decisions."""
    return any(
        len({len(domains) + len(sub) for domains, sub in choice}) > 1
        for choice in shape[1]
    )


def _draw_cases(strategy: st.SearchStrategy[_SpaceShape]) -> list[_SpaceShape]:
    """Return a fixed, reproducible sample of `strategy`'s draws."""
    cases: list[_SpaceShape] = []

    @settings(
        max_examples=_SAMPLE_SIZE,
        derandomize=True,
        database=None,
        phases=[Phase.generate],
        suppress_health_check=list(HealthCheck),
    )
    @given(strategy)
    def collect(shape: _SpaceShape) -> None:
        cases.append(shape)

    collect()
    return cases


def test_the_space_strategy_draws_choices_often_enough() -> None:
    """Test at least a third of a fixed sample of spaces hold a choice."""
    cases = _draw_cases(_SPACE_SHAPES)

    assert sum(map(_has_a_choice, cases)) >= len(cases) // 3


def test_the_small_space_strategy_draws_conditional_decisions_often_enough() -> None:
    """Test a tenth of a fixed sample of small spaces have unequal alternatives."""
    cases = _draw_cases(_SMALL_SHAPES)

    assert sum(map(_has_alternatives_of_different_sizes, cases)) >= len(cases) // 10


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS)
def test_replaying_a_configurations_trace_gives_an_equal_key(
    shape: _SpaceShape, picks: list[int]
) -> None:
    """Test `replay(configuration.trace())` rebuilds the configuration."""
    space = _build_space(shape)
    configuration = Configuration(space, _entries(space, picks))

    replayed = space.replay(configuration.trace())

    assert replayed.key() == configuration.key()


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS)
def test_a_trace_replays_onto_a_relabeled_space_with_an_equal_key(
    shape: _SpaceShape, picks: list[int]
) -> None:
    """Test a trace recorded over a space replays into a fresh copy of it."""
    space, relabeled = _build_space(shape), _build_space(shape)
    configuration = Configuration(space, _entries(space, picks))

    replayed = relabeled.replay(configuration.trace())

    assert replayed.space is relabeled
    assert replayed.key() == configuration.key()


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS)
def test_a_configurations_trace_round_trips_through_its_text(
    shape: _SpaceShape, picks: list[int]
) -> None:
    """Test decoding a trace's text gives an equal trace of the same text."""
    space = _build_space(shape)
    trace = Configuration(space, _entries(space, picks)).trace()
    text = trace.to_json()

    decoded = Trace.from_json(text)

    assert decoded == trace
    assert decoded.to_json() == text


@cap_max_examples(60)
@given(_SPACE_SHAPES, _SEEDS)
def test_a_sampled_configuration_is_complete_and_replays(
    shape: _SpaceShape, seed: int
) -> None:
    """Test `sample` gives a complete configuration its own trace replays."""
    space = _build_space(shape)

    configuration, trace = space.sample(RandomOracle(seed=seed))

    assert configuration.is_complete()
    assert len(trace) == len(configuration.entries)
    assert space.replay(trace).key() == configuration.key()


@cap_max_examples(60)
@given(_SPACE_SHAPES, _SEEDS)
def test_a_uniformly_sampled_configuration_is_complete(
    shape: _SpaceShape, seed: int
) -> None:
    """Test `sample_uniform` gives a complete configuration of the space."""
    space = _build_space(shape)

    configuration, _ = space.sample_uniform(Rng(seed))

    assert configuration.is_complete()
    assert configuration.space is space


@cap_max_examples(60)
@given(_SPACE_SHAPES, _PICKS, _SEEDS)
def test_a_mutated_configuration_is_complete(
    shape: _SpaceShape, picks: list[int], seed: int
) -> None:
    """Test `mutate` gives another complete configuration of the space.

    A space with one complete configuration, the empty space included, has
    nothing to mutate: `mutate` refuses it, as the Rust property checks.
    """
    space = _build_space(shape)
    configuration = Configuration(space, _entries(space, picks))

    if space.cardinality() == Cardinality(CardinalityKind.EXACT, 1, None):
        with pytest.raises(TraceError, match=_NOTHING_TO_MUTATE):
            space.mutate(configuration, Rng(seed))
        return
    mutated, _ = space.mutate(configuration, Rng(seed))

    assert mutated.is_complete()
    assert mutated.space is space
    assert mutated.key() != configuration.key()


@cap_max_examples(30)
@given(_SMALL_SHAPES)
def test_enumeration_yields_as_many_configurations_as_the_cardinality(
    shape: _SpaceShape,
) -> None:
    """Test `len(list(enumerate()))` equals the exact cardinality."""
    space = _build_space(shape)

    cardinality = space.cardinality()

    assert cardinality.kind is CardinalityKind.EXACT
    assert len(list(space.enumerate())) == cardinality.count


@cap_max_examples(30)
@given(_SMALL_SHAPES)
def test_enumerated_configurations_are_distinct_and_complete(
    shape: _SpaceShape,
) -> None:
    """Test no configuration is enumerated twice and each is complete."""
    configurations = list(_build_space(shape).enumerate())

    assert all(configuration.is_complete() for configuration in configurations)
    assert len({configuration.key() for configuration in configurations}) == len(
        configurations
    )
