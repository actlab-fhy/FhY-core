"""The interface suite of `GuidedOracle` and `Space.crossover`.

A guided oracle answers a step from a recorded guide where the guide has an
admissible answer and asks a fallback oracle where it does not. Crossover
builds a complete configuration out of two parents, taking each decision's
value from one of them, repairing a combination the space forbids and
drawing only when neither parent has an admissible value.
"""

from collections.abc import Callable
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    ChoiceDomain,
    Configuration,
    Forbidden,
    GuidedOracle,
    RandomOracle,
    Recorder,
    Rng,
    SearchOracle,
    Space,
    StridedDomain,
    StridedRun,
    Trace,
    TraceError,
    Variable,
)
from fhy_core.symbolic.constraint import InSetConstraint

from ..conftest import is_cycle_collected
from .conftest import (
    FailingOracle,
    build_dynamic_trace,
    build_tiling_space,
    make_alternative,
    make_choice,
    make_variable,
)


class _Boom(Exception):
    """The exception the failing fallback raises."""


class _Counting:
    """A fallback oracle answering the first admissible coordinate, counting asks."""

    def __init__(self) -> None:
        self.subjects: list[Identifier] = []
        self.kinds: list[str] = []

    def decide(self, step: Any) -> int:
        self.subjects.append(step.subject)
        self.kinds.append(step.kind)
        return next(
            coordinate
            for coordinate in range(step.domain.cardinality)
            if step.admits(coordinate)
        )


def _tiled(tiling: Any, unroll: int, tile: int) -> Configuration:
    """Return the complete configuration of `tiling` choosing `tiled`."""
    return Configuration(
        tiling.space,
        {
            tiling.unroll.name: unroll,
            tiling.layout.name: tiling.tiled.name,
            tiling.tile.name: tile,
        },
    )


# ===========================================================================
# GuidedOracle
# ===========================================================================


def test_a_guided_oracle_keeps_its_guide_and_fallback() -> None:
    """Test `guide` and `fallback` return the very objects given."""
    guide, fallback = build_dynamic_trace(), _Counting()

    oracle = GuidedOracle(guide, fallback)

    assert oracle.guide is guide
    assert oracle.fallback is fallback


def test_a_guided_oracle_is_a_search_oracle() -> None:
    """Test the runtime protocol check accepts a guided oracle."""
    assert isinstance(GuidedOracle(Trace(), RandomOracle(seed=0)), SearchOracle)


def test_a_guided_oracle_holds_a_rust_fallback_as_given() -> None:
    """Test a `RandomOracle` fallback is returned as the same object."""
    fallback = RandomOracle(seed=4)

    assert GuidedOracle(Trace(), fallback).fallback is fallback


@pytest.mark.parametrize(
    "guide",
    [
        pytest.param(lambda: 3, id="int"),
        pytest.param(lambda: None, id="none"),
        pytest.param(lambda: build_dynamic_trace().steps, id="steps"),
        pytest.param(lambda: build_dynamic_trace().key(), id="trace_key"),
    ],
)
def test_a_guide_that_is_no_trace_is_a_type_error(guide: Callable[[], Any]) -> None:
    """Test a guide that is no `Trace` raises `TypeError`."""
    with pytest.raises(TypeError):
        GuidedOracle(guide(), RandomOracle(seed=0))


def test_a_full_guide_reproduces_its_configuration_without_asking() -> None:
    """Test sampling under the trace of `c` gives `c` and never asks the fallback."""
    tiling = build_tiling_space()
    configuration = _tiled(tiling, unroll=4, tile=8)
    fallback = _Counting()

    sampled, trace = tiling.space.sample(GuidedOracle(configuration.trace(), fallback))

    assert sampled.key() == configuration.key()
    assert sampled.entries == configuration.entries
    assert trace.coordinates == configuration.trace().coordinates
    assert fallback.subjects == []


def test_a_guide_of_one_build_guides_a_run_over_the_next() -> None:
    """Test the guide is followed by position into a relabeled space."""
    recorded, relabeled = build_tiling_space(), build_tiling_space()
    configuration = _tiled(recorded, unroll=4, tile=8)
    fallback = _Counting()

    sampled, _ = relabeled.space.sample(GuidedOracle(configuration.trace(), fallback))

    assert sampled.space is relabeled.space
    assert sampled.key() == configuration.key()
    assert fallback.subjects == []


def test_an_inadmissible_guide_step_goes_to_the_fallback() -> None:
    """Test a step the space now forbids is asked of the fallback, the rest guided."""
    recorded = build_tiling_space()
    forbidding = build_tiling_space(forbid_unroll_four=True)
    configuration = _tiled(recorded, unroll=4, tile=8)
    fallback = _Counting()

    sampled, _ = forbidding.space.sample(GuidedOracle(configuration.trace(), fallback))

    assert fallback.subjects == [forbidding.unroll.name]
    assert sampled.value(forbidding.unroll.name) in {1, 2}
    assert sampled.value(forbidding.layout.name) == forbidding.tiled.name
    assert sampled.value(forbidding.tile.name) == 8
    assert sampled.is_complete()


def test_an_empty_guide_is_the_fallback() -> None:
    """Test a guide with no step asks the fallback every step."""
    tiling = build_tiling_space()
    asked = _Counting()

    sampled, _ = tiling.space.sample(GuidedOracle(Trace(), asked))
    plain, _ = tiling.space.sample(_Counting())

    assert sampled.key() == plain.key()
    assert set(asked.subjects) == {
        tiling.unroll.name,
        tiling.layout.name,
        tiling.tile.name,
    }


def test_an_empty_guide_over_a_seeded_fallback_is_that_fallback() -> None:
    """Test an empty guide over `RandomOracle(s)` samples as `RandomOracle(s)` does."""
    tiling = build_tiling_space()

    guided, guided_trace = tiling.space.sample(
        GuidedOracle(Trace(), RandomOracle(seed=17))
    )
    plain, plain_trace = tiling.space.sample(RandomOracle(seed=17))

    assert guided.key() == plain.key()
    assert guided_trace == plain_trace


def test_a_guide_answers_dynamic_steps_per_kind_in_order() -> None:
    """Test dynamic steps take the guide's next step of their kind, not the fallback."""
    guide = build_dynamic_trace(option=2, address=17)
    fallback = _Counting()
    recorder = Recorder(GuidedOracle(guide, fallback))
    subject = Identifier("fresh")

    option = recorder.decide_dynamic(
        "tests.option", subject, ChoiceDomain(("a", "b", "c"))
    )
    address = recorder.decide_dynamic(
        "tests.address", subject, StridedDomain((StridedRun(0, 64),))
    )

    assert option == "c"
    assert address == 17
    assert fallback.subjects == []
    assert recorder.trace.key() == guide.key()


def test_a_dynamic_step_past_the_guide_goes_to_the_fallback() -> None:
    """Test a second step of a kind the guide holds once is asked of the fallback."""
    guide = build_dynamic_trace(option=2)
    fallback = _Counting()
    recorder = Recorder(GuidedOracle(guide, fallback))
    domain = ChoiceDomain(("a", "b", "c"))
    subject = Identifier("fresh")

    first = recorder.decide_dynamic("tests.option", subject, domain)
    second = recorder.decide_dynamic("tests.option", subject, domain)

    assert first == "c"
    assert second == "a"
    assert fallback.kinds == ["tests.option"]


def test_a_dynamic_step_over_another_domain_goes_to_the_fallback() -> None:
    """Test a guide step recorded over another domain is not followed."""
    guide = build_dynamic_trace(option=2)
    fallback = _Counting()
    recorder = Recorder(GuidedOracle(guide, fallback))

    value = recorder.decide_dynamic(
        "tests.option", Identifier("fresh"), ChoiceDomain(("x", "y"))
    )

    assert value == "x"
    assert fallback.kinds == ["tests.option"]


def test_an_exception_from_the_fallback_propagates_as_itself() -> None:
    """Test the error a Python fallback raises is the one the run raises."""
    error = _Boom("the fallback failed")
    tiling = build_tiling_space()

    with pytest.raises(_Boom, match="the fallback failed") as info:
        tiling.space.sample(GuidedOracle(Trace(), FailingOracle(error)))

    assert info.value is error


def test_a_guided_run_only_reaches_the_fallback_where_the_guide_fails() -> None:
    """Test a failing fallback is silent while the guide answers every step."""
    tiling = build_tiling_space()
    configuration = _tiled(tiling, unroll=2, tile=4)

    sampled, _ = tiling.space.sample(
        GuidedOracle(configuration.trace(), FailingOracle(_Boom("never")))
    )

    assert sampled.key() == configuration.key()


def test_a_fallback_holding_its_guided_oracle_is_collected() -> None:
    """Test a Python fallback that points back at the oracle holding it dies."""

    def build() -> object:
        fallback = _Counting()
        oracle = GuidedOracle(Trace(), fallback)
        fallback.owner = oracle  # type: ignore[attr-defined]
        return fallback

    assert is_cycle_collected(build)


# ===========================================================================
# Space.crossover
# ===========================================================================


def _forbidding_space() -> tuple[Space, Variable[Any], Variable[Any]]:
    """Return a space of `x` and `y` over {1, 2} that forbids `x = 1, y = 1`."""
    x, y = make_variable("x", 1, 2), make_variable("y", 1, 2)
    forbidden = Forbidden((InSetConstraint(x.name, {1}), InSetConstraint(y.name, {1})))
    return Space(variables=(x, y), forbidden=(forbidden,)), x, y


def test_crossover_returns_a_configuration_and_its_trace() -> None:
    """Test the result is a complete `Configuration` of the space and a `Trace`."""
    tiling = build_tiling_space()
    first = _tiled(tiling, unroll=1, tile=4)
    second = _tiled(tiling, unroll=4, tile=8)

    child, trace = tiling.space.crossover(first, second, Rng(3))

    assert isinstance(child, Configuration)
    assert isinstance(trace, Trace)
    assert child.space is tiling.space
    assert child.is_complete()
    assert tiling.space.replay(trace).key() == child.key()


@pytest.mark.parametrize("seed", range(8))
def test_crossover_of_identical_parents_is_the_parent(seed: int) -> None:
    """Test `crossover(a, a)` is `a`, whatever the seed."""
    tiling = build_tiling_space()
    parent = _tiled(tiling, unroll=2, tile=8)

    child, _ = tiling.space.crossover(parent, parent, Rng(seed))

    assert child.key() == parent.key()
    assert child.entries == parent.entries


def test_every_value_of_the_child_comes_from_a_parent() -> None:
    """Test each decision of a child holds one of its parents' values."""
    tiling = build_tiling_space()
    first = _tiled(tiling, unroll=1, tile=4)
    second = _tiled(tiling, unroll=4, tile=8)

    children = [
        tiling.space.crossover(first, second, Rng(seed))[0] for seed in range(20)
    ]

    assert {child.value(tiling.unroll.name) for child in children} <= {1, 4}
    assert {child.value(tiling.layout.name) for child in children} == {
        tiling.tiled.name
    }
    assert {child.value(tiling.tile.name) for child in children} <= {4, 8}


def test_crossover_takes_values_from_both_parents() -> None:
    """Test over several seeds a decision is taken from each parent."""
    tiling = build_tiling_space()
    first = _tiled(tiling, unroll=1, tile=4)
    second = _tiled(tiling, unroll=4, tile=8)

    children = [
        tiling.space.crossover(first, second, Rng(seed))[0] for seed in range(40)
    ]

    assert {child.value(tiling.unroll.name) for child in children} == {1, 4}
    assert {child.value(tiling.tile.name) for child in children} == {4, 8}


def test_crossover_is_deterministic_from_the_seed() -> None:
    """Test one seed over the same parents gives one child and trace."""
    tiling = build_tiling_space()
    first = _tiled(tiling, unroll=1, tile=4)
    second = _tiled(tiling, unroll=4, tile=8)

    left, left_trace = tiling.space.crossover(first, second, Rng(11))
    right, right_trace = tiling.space.crossover(first, second, Rng(11))

    assert left.key() == right.key()
    assert left_trace == right_trace


def test_crossover_repairs_a_forbidden_combination() -> None:
    """Test no child holds the combination the space forbids."""
    space, x, y = _forbidding_space()
    first = Configuration(space, {x.name: 1, y.name: 2})
    second = Configuration(space, {x.name: 2, y.name: 1})

    children = [space.crossover(first, second, Rng(seed))[0] for seed in range(128)]

    pairs = {(child.value(x.name), child.value(y.name)) for child in children}
    assert all(child.is_complete() for child in children)
    assert (1, 1) not in pairs
    assert pairs <= {(1, 2), (2, 1), (2, 2)}
    assert {(1, 2), (2, 1)} <= pairs


def test_crossover_accepts_parents_that_are_not_complete() -> None:
    """Test unassigned decisions of the parents are filled by an admissible draw."""
    tiling = build_tiling_space()
    first = Configuration(tiling.space, {tiling.unroll.name: 1})
    second = Configuration(tiling.space)

    child, _ = tiling.space.crossover(first, second, Rng(2))

    assert child.is_complete()
    assert child.value(tiling.unroll.name) in {1, 2, 4}


def test_crossover_draws_where_no_parent_has_a_value() -> None:
    """Test a decision neither parent assigns is drawn from the admissible values."""
    tiling = build_tiling_space(forbid_unroll_four=True)
    empty = Configuration(tiling.space)

    values = {
        tiling.space.crossover(empty, empty, Rng(seed))[0].value(tiling.unroll.name)
        for seed in range(64)
    }

    assert values == {1, 2}


def test_crossover_takes_the_attempts_keyword() -> None:
    """Test `attempts` is a keyword of the call and a result still comes back."""
    tiling = build_tiling_space()
    parent = _tiled(tiling, unroll=2, tile=8)

    child, _ = tiling.space.crossover(parent, parent, Rng(0), attempts=3)

    assert child.key() == parent.key()


def test_crossover_refuses_a_parent_of_another_space() -> None:
    """Test a parent built over another space raises `TraceError`."""
    tiling, other = build_tiling_space(), build_tiling_space()
    mine = _tiled(tiling, unroll=1, tile=4)
    foreign = _tiled(other, unroll=1, tile=4)

    with pytest.raises(TraceError):
        tiling.space.crossover(mine, foreign, Rng(0))
    with pytest.raises(TraceError):
        tiling.space.crossover(foreign, mine, Rng(0))


def test_crossover_of_alternatives_keeps_the_chosen_subtree_complete() -> None:
    """Test a child that chose an alternative holds its parent's value under it.

    Each parent chooses another alternative, so every child chooses one of
    the two and takes the value under it from the parent that chose it.
    """
    left = make_alternative("left", (make_variable("a", 1, 2),))
    right = make_alternative("right", (make_variable("b", 3, 4),))
    choice = make_choice("c", left, right)
    space = Space(choices=(choice,))
    a, b = left.variables[0].name, right.variables[0].name
    first = Configuration(space, {choice.name: left.name, a: 1})
    second = Configuration(space, {choice.name: right.name, b: 4})

    children = [space.crossover(first, second, Rng(seed))[0] for seed in range(32)]

    outcomes = {
        (child.value(choice.name), child.value(a), child.value(b)) for child in children
    }
    assert all(child.is_complete() for child in children)
    assert outcomes == {(left.name, 1, None), (right.name, None, 4)}


def test_crossover_respects_a_condition() -> None:
    """Test a child never assigns a decision whose condition fails."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)
    flat = Configuration(tiling.space, {tiling.layout.name: tiling.flat.name})
    tiled = Configuration(
        tiling.space,
        {
            tiling.layout.name: tiling.tiled.name,
            tiling.unroll.name: 2,
            tiling.tile.name: 4,
        },
    )

    children = [tiling.space.crossover(flat, tiled, Rng(seed))[0] for seed in range(32)]

    outcomes = {
        (
            child.value(tiling.layout.name),
            child.value(tiling.unroll.name),
            child.value(tiling.tile.name),
        )
        for child in children
    }
    assert all(child.is_complete() for child in children)
    assert outcomes == {(tiling.flat.name, None, None), (tiling.tiled.name, 2, 4)}
