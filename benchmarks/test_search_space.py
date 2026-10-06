"""Benchmarks of the search-space package.

They measure ``fhy_core.search_space``, which the Rust core backs, through
the public API only, on the shape MOGA-VM's audit baseline measured: one
choice of eight alternatives, each holding four variables over the
categories ``{1, 2, 3, 4}``. The before numbers are MOGA-VM's Python core
on fhy_core 0.2.0 (``docs/design/search-space.md``, "Benchmark plan").

The Python API lands in SS1.7, which adapts these rows to its final
signatures; until then the module is absent and every row skips.
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.constraint import InSetConstraint
from fhy_core.symbolic.param import create_categorical_param

from .conftest import Benchmark

pytestmark = pytest.mark.benchmark(group="search_space")

search_space = pytest.importorskip("fhy_core.search_space")

# The shape of the benchmarked space: alternatives of the one choice, and
# variables per alternative.
_ALTERNATIVE_COUNT = 8
_VARIABLE_COUNT = 4
_CATEGORIES = frozenset({1, 2, 3, 4})


def _build_space(
    tag: str, *, conditions: tuple[Any, ...] = (), forbidden: tuple[Any, ...] = ()
) -> Any:
    """Return the benchmarked space, every name minted fresh under ``tag``."""
    alternatives = tuple(
        search_space.Alternative(
            name=Identifier(f"{tag}_alternative_{index}"),
            variables=tuple(
                search_space.Variable(
                    name=Identifier(f"{tag}_variable_{index}_{position}"),
                    param=create_categorical_param(_CATEGORIES),
                )
                for position in range(_VARIABLE_COUNT)
            ),
        )
        for index in range(_ALTERNATIVE_COUNT)
    )
    choice = search_space.Choice(
        name=Identifier(f"{tag}_choice"), alternatives=alternatives
    )
    return search_space.Space(
        name=Identifier(f"{tag}_space"),
        choices=(choice,),
        conditions=conditions,
        forbidden=forbidden,
    )


def _first_entries(space: Any) -> dict[Identifier, Any]:
    """Return entries choosing the first alternative and assigning its variables."""
    (choice,) = space.choices
    first = choice.alternatives[0]
    entries: dict[Identifier, Any] = {choice.name: first.name}
    entries.update({variable.name: 1 for variable in first.variables})
    return entries


def _build_tagged_alternative(name: Identifier, tag: int) -> Any:
    """Return an alternative with data of its own, compared through hooks."""
    base: Any = search_space.Alternative

    class TaggedAlternative(base):  # type: ignore[misc]
        def __init__(self, *, tag: int, **fields: Any) -> None:
            super().__init__(**fields)
            self.tag = tag

        def extension_is_structurally_equivalent(self, other: Any) -> bool:
            return bool(self.tag == other.tag)

        def extension_is_alpha_equivalent_under(
            self, other: Any, renaming: Any
        ) -> bool:
            del renaming
            return bool(self.tag == other.tag)

    return TaggedAlternative(name=name, tag=tag)


def _build_hooked_space(tag: str) -> Any:
    """Return a space of one choice whose one alternative has hooks."""
    alternative = _build_tagged_alternative(Identifier(f"{tag}_alternative"), 1)
    choice = search_space.Choice(
        name=Identifier(f"{tag}_choice"), alternatives=(alternative,)
    )
    return search_space.Space(name=Identifier(f"{tag}_space"), choices=(choice,))


def test_space_construction(benchmark: Benchmark) -> None:
    """Benchmark building the space, its params and parts included."""
    benchmark(_build_space, "built")


def test_space_construction_with_conditions_and_forbidden(benchmark: Benchmark) -> None:
    """Benchmark building a space with a condition and a forbidden clause."""
    template = _build_space("template")
    (choice,) = template.choices
    first, second = choice.alternatives[0], choice.alternatives[1]

    def build() -> Any:
        return search_space.Space(
            name=Identifier("guarded_space"),
            choices=(choice,),
            conditions=(
                search_space.Condition(
                    target=second.variables[0].name,
                    when=(InSetConstraint(choice.name, frozenset({second.name})),),
                ),
            ),
            forbidden=(
                search_space.Forbidden(
                    when=(InSetConstraint(first.variables[0].name, frozenset({4})),)
                ),
            ),
        )

    benchmark(build)


def test_configuration_construction(benchmark: Benchmark) -> None:
    """Benchmark building and checking a complete configuration."""
    space = _build_space("validated")
    entries = _first_entries(space)
    benchmark(search_space.Configuration, space, entries)


def test_configuration_with_entry(benchmark: Benchmark) -> None:
    """Benchmark replacing one value of a configuration."""
    space = _build_space("assigned")
    configuration = search_space.Configuration(space, _first_entries(space))
    variable = space.choices[0].alternatives[0].variables[0]
    benchmark(configuration.with_entry, variable.name, 2)


def test_configuration_key(benchmark: Benchmark) -> None:
    """Benchmark a configuration's key."""
    space = _build_space("keyed")
    configuration = search_space.Configuration(space, _first_entries(space))
    benchmark(configuration.key)


def test_space_structural_self_equivalence(benchmark: Benchmark) -> None:
    """Benchmark structural equivalence of a space with itself."""
    space = _build_space("structural")
    benchmark(space.is_structurally_equivalent, space)


def test_space_alpha_equivalence_against_a_relabeled_copy(benchmark: Benchmark) -> None:
    """Benchmark alpha equivalence against a copy with fresh names."""
    left, right = _build_space("left"), _build_space("right")
    benchmark(left.is_alpha_equivalent, right)


def test_space_alpha_equivalence_through_python_hooks(benchmark: Benchmark) -> None:
    """Benchmark alpha equivalence calling a Python subclass's hooks."""
    left, right = (
        _build_hooked_space("hooked_left"),
        _build_hooked_space("hooked_right"),
    )
    benchmark(left.is_alpha_equivalent, right)


def test_configuration_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark serializing a configuration."""
    space = _build_space("serialized")
    configuration = search_space.Configuration(space, _first_entries(space))
    benchmark(configuration.serialize_to_dict)


def test_variable_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark serializing a variable."""
    variable = search_space.Variable(
        name=Identifier("serialized_variable"),
        param=create_categorical_param(_CATEGORIES),
    )
    benchmark(variable.serialize_to_dict)


# ---------------------------------------------------------------------------
# The search stream (SS2)
# ---------------------------------------------------------------------------

_STEP_KIND = "bench.step"
_STREAM_LENGTH = 100
_RUN_WIDTH = 2**16


def _stream_domains() -> tuple[Any, Any, Any]:
    """Return a choice of 8, a strided domain of 4 runs of 2^16, an order of 4."""
    choice = search_space.ChoiceDomain(tuple(range(8)))
    strided = search_space.StridedDomain(
        tuple(
            search_space.StridedRun(
                2 * index * _RUN_WIDTH, (2 * index + 1) * _RUN_WIDTH
            )
            for index in range(4)
        )
    )
    order = search_space.OrderDomain(
        tuple(Identifier(f"level_{index}") for index in range(4))
    )
    return choice, strided, order


def _record_stream(oracle: Any, domains: tuple[Any, ...]) -> Any:
    """Return the recorder of a stream of `_STREAM_LENGTH` steps over `domains`."""
    subject = Identifier("subject")
    recorder = search_space.Recorder(oracle)
    for position in range(_STREAM_LENGTH):
        recorder.decide_dynamic(_STEP_KIND, subject, domains[position % len(domains)])
    return recorder


@pytest.mark.parametrize("shape", ["choice", "strided", "order"])
def test_one_random_draw(benchmark: Benchmark, shape: str) -> None:
    """Benchmark one draw of a `RandomOracle` through a `Recorder`."""
    domain = dict(zip(("choice", "strided", "order"), _stream_domains(), strict=True))[
        shape
    ]
    recorder = search_space.Recorder(search_space.RandomOracle(seed=0))
    subject = Identifier("subject")
    benchmark(recorder.decide_dynamic, _STEP_KIND, subject, domain)


def test_record_a_dynamic_stream(benchmark: Benchmark) -> None:
    """Benchmark recording a 100-step stream with a `RandomOracle`."""
    domains = _stream_domains()
    benchmark(lambda: _record_stream(search_space.RandomOracle(seed=0), domains).trace)


def test_replay_a_dynamic_stream(benchmark: Benchmark) -> None:
    """Benchmark replaying a 100-step stream with a `ReplayOracle`."""
    domains = _stream_domains()
    trace = _record_stream(search_space.RandomOracle(seed=0), domains).trace

    def replay() -> None:
        oracle = search_space.ReplayOracle(trace)
        _record_stream(oracle, domains)
        oracle.finish()

    benchmark(replay)


def test_strided_value_at_and_coordinate_of(benchmark: Benchmark) -> None:
    """Benchmark `value_at` and `coordinate_of` over 64 strided runs."""
    domain = search_space.StridedDomain(
        tuple(
            search_space.StridedRun(index * 1024, index * 1024 + 512, 8)
            for index in range(64)
        )
    )
    middle = domain.cardinality // 2

    benchmark(lambda: domain.coordinate_of(domain.value_at(middle)))


def _build_option_space(tag: str) -> Any:
    """Return 8 choices of 4 alternatives, each holding 2 variables of 4 values."""
    choices = tuple(
        search_space.Choice(
            name=Identifier(f"{tag}_entry_{entry}"),
            alternatives=tuple(
                search_space.Alternative(
                    name=Identifier(f"{tag}_option_{entry}_{option}"),
                    variables=tuple(
                        search_space.Variable(
                            name=Identifier(f"{tag}_axis_{entry}_{option}_{axis}"),
                            param=create_categorical_param(_CATEGORIES),
                        )
                        for axis in range(2)
                    ),
                )
                for option in range(4)
            ),
        )
        for entry in range(8)
    )
    return search_space.Space(name=Identifier(f"{tag}_space"), choices=choices)


def test_option_space_cardinality(benchmark: Benchmark) -> None:
    """Benchmark counting 8 choices of 4 alternatives of 2 variables."""
    space = _build_option_space("counted")
    benchmark(space.cardinality)


def test_option_space_sample(benchmark: Benchmark) -> None:
    """Benchmark sampling one configuration of that space with a `RandomOracle`."""
    space = _build_option_space("sampled")
    oracle = search_space.RandomOracle(seed=0)
    benchmark(space.sample, oracle)


def test_option_space_sample_uniform(benchmark: Benchmark) -> None:
    """Benchmark drawing one configuration of that space uniformly."""
    space = _build_option_space("uniform")
    rng = search_space.Rng(0)
    benchmark(space.sample_uniform, rng)


def test_option_space_mutate(benchmark: Benchmark) -> None:
    """Benchmark mutating one configuration of that space."""
    space = _build_option_space("mutated")
    configuration, _ = space.sample(search_space.RandomOracle(seed=0))
    rng = search_space.Rng(0)
    benchmark(space.mutate, configuration, rng)


def test_enumerate_ten_thousand_configurations(benchmark: Benchmark) -> None:
    """Benchmark enumerating the 10 000 configurations of 4 variables of 10."""
    values = frozenset(range(10))
    space = search_space.Space(
        name=Identifier("enumerated_space"),
        variables=tuple(
            search_space.Variable(
                name=Identifier(f"enumerated_{index}"),
                param=create_categorical_param(values),
            )
            for index in range(4)
        ),
    )
    benchmark.pedantic(lambda: sum(1 for _ in space.enumerate()), rounds=3)


def test_cardinality_with_a_condition_and_a_clause(benchmark: Benchmark) -> None:
    """Benchmark counting a space with a condition and a forbidden clause."""
    template = _build_space("guarded_template")
    (choice,) = template.choices
    first, second = choice.alternatives[0], choice.alternatives[1]
    space = search_space.Space(
        name=Identifier("guarded_count_space"),
        choices=(choice,),
        conditions=(
            search_space.Condition(
                target=second.variables[0].name,
                when=(InSetConstraint(choice.name, frozenset({second.name})),),
            ),
        ),
        forbidden=(
            search_space.Forbidden(
                when=(InSetConstraint(first.variables[0].name, frozenset({4})),)
            ),
        ),
    )
    benchmark(space.cardinality)


def test_trace_json_round_trip(benchmark: Benchmark) -> None:
    """Benchmark writing a 100-step trace as JSON and reading it back."""
    trace = _record_stream(search_space.RandomOracle(seed=0), _stream_domains()).trace
    benchmark(lambda: search_space.Trace.from_json(trace.to_json()))


class _FirstOracle:
    """A Python oracle answering the first coordinate."""

    def decide(self, step: Any) -> int:
        del step
        return 0


def test_one_step_through_a_python_oracle(benchmark: Benchmark) -> None:
    """Benchmark one dynamic step answered by a Python oracle."""
    choice, _, _ = _stream_domains()
    recorder = search_space.Recorder(_FirstOracle())
    subject = Identifier("subject")
    benchmark(recorder.decide_dynamic, _STEP_KIND, subject, choice)


# ---------------------------------------------------------------------------
# Objectives and measurements (SS3)
# ---------------------------------------------------------------------------


def _measurement_parts() -> tuple[Any, dict[Any, float]]:
    """Return a configuration key and values for four objectives."""
    space = _build_space("measured")
    key = search_space.Configuration(space, _first_entries(space)).key()
    directions = ("minimize", "maximize", "minimize", "report")
    values = {
        search_space.Objective(f"objective_{index}", direction): float(index) + 0.5
        for index, direction in enumerate(directions)
    }
    return key, values


def test_measurement_construction(benchmark: Benchmark) -> None:
    """Benchmark building a measurement of four objectives."""
    key, values = _measurement_parts()
    benchmark(search_space.Measurement.ok, key, values)


def test_measurement_serialize_to_dict(benchmark: Benchmark) -> None:
    """Benchmark serializing a measurement of four objectives."""
    key, values = _measurement_parts()
    measurement = search_space.Measurement.ok(key, values)
    benchmark(measurement.serialize_to_dict)
