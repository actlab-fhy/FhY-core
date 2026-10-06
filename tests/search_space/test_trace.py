"""The interface suite of traces, oracles, enumeration and sampling.

Domains, the random-number generator, the recorder, the shipped oracles,
traces and their text, and the `Space` methods that sample, replay,
enumerate, count and mutate. Each step the oracles answer is built by
`Recorder.decide_dynamic`, over domains of Python objects; the tests ported
from MOGA-VM's `tests/cir/lowering/search/` carry the name of the test they
port in their docstrings.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Cardinality,
    CardinalityKind,
    ChoiceDomain,
    Configuration,
    ExhaustiveOracle,
    InadmissibleAnswerError,
    NotEnumerableError,
    OrderDomain,
    RandomOracle,
    Recorder,
    ReplayMismatchError,
    ReplayOracle,
    Rng,
    SearchOracle,
    Space,
    StepDomainError,
    StridedDomain,
    StridedRun,
    Trace,
    TraceError,
    Variable,
)
from fhy_core.serialization import Serializable
from fhy_core.symbolic.param import create_natural_param

from .conftest import (
    build_complete_configuration,
    build_tiling_space,
    make_alternative,
    make_choice,
    make_variable,
)

_SUBJECT = Identifier("subject")
_OPTION = "moga.cir.option"
_ADDRESS = "moga.cir.address"
_WALK_ORDER = "moga.cir.walk_order"
_MASK = 2**64


class _Opaque:
    """An object whose `==` is identity, as MOGA-VM's options are."""


class _Constant:
    """An oracle answering `answer` to every step."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    def decide(self, step: Any) -> Any:
        return self._answer


class _Scripted:
    """An oracle answering the given coordinates, one per step, in order."""

    def __init__(self, answers: Iterable[Any]) -> None:
        self._answers = iter(answers)

    def decide(self, step: Any) -> Any:
        return next(self._answers)


class _Forbidden:
    """An oracle that fails when asked."""

    def decide(self, step: Any) -> Any:
        raise AssertionError("the oracle was asked")


def _choice(*values: Any) -> ChoiceDomain:
    return ChoiceDomain(values)


def _addresses(*runs: tuple[int, int]) -> StridedDomain:
    return StridedDomain(tuple(StridedRun(*run) for run in runs))


def _order(*names: str) -> OrderDomain:
    return OrderDomain(tuple(Identifier(name) for name in names))


def _record(
    oracle: Any, steps: Iterable[tuple[str, Any]], subject: Identifier = _SUBJECT
) -> tuple[Trace, list[Any]]:
    """Return the trace and the values of `oracle` answering `steps`."""
    recorder = Recorder(oracle)
    values = [recorder.decide_dynamic(kind, subject, domain) for kind, domain in steps]
    return recorder.trace, values


# ===========================================================================
# Rng
# ===========================================================================


def _stream(rng: Rng, count: int) -> list[int]:
    return [rng.next_u64() for _ in range(count)]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (
            0,
            [
                0xE220A8397B1DCDAF,
                0x6E789E6AA1B965F4,
                0x06C45D188009454F,
                0xF88BB8A8724C81EC,
                0x1B39896A51A8749B,
                0x53CB9F0C747EA2EA,
                0x2C829ABE1F4532E1,
                0xC584133AC916AB3C,
            ],
        ),
        (1, [0x910A2DEC89025CC1, 0xBEEB8DA1658EEC67, 0xF893A2EEFB32555E]),
    ],
    ids=["seed_0", "seed_1"],
)
def test_rng_next_u64_follows_the_golden_stream(seed: int, expected: list[int]) -> None:
    """Test `Rng(seed).next_u64()` gives SplitMix64's reference outputs."""
    rng = Rng(seed)

    assert _stream(rng, len(expected)) == expected


@pytest.mark.parametrize(
    ("seed", "bound", "expected"),
    [
        (42, 6, [4, 0, 1, 2, 0, 5, 1, 4]),
        (42, 3, [2, 0, 0, 1, 0, 2, 0, 2]),
        (123, 1000, [706, 976, 859, 686, 686, 667, 999, 482, 619, 140]),
    ],
    ids=["below_6", "below_3", "below_1000"],
)
def test_rng_below_follows_the_golden_draws(
    seed: int, bound: int, expected: list[int]
) -> None:
    """Test `Rng.below(bound)` draws the reference values."""
    rng = Rng(seed)

    assert [rng.below(bound) for _ in expected] == expected


def test_rng_below_draws_below_a_bound_wider_than_a_word() -> None:
    """Test `Rng.below` takes any positive `int`, as the reference does."""
    rng = Rng(42)

    assert [rng.below(10**30) for _ in range(4)] == [
        292897267883935188527843339925,
        811006502546476498273989829618,
        646668284273496457662110203349,
        259944956782528464820904019686,
    ]


def test_rng_below_one_is_always_zero() -> None:
    """Test a bound of one leaves one value to draw."""
    rng = Rng(9)

    assert [rng.below(1) for _ in range(5)] == [0] * 5


@pytest.mark.parametrize("bound", [0, -3], ids=["zero", "negative"])
def test_rng_below_refuses_a_bound_that_is_not_positive(bound: int) -> None:
    """Test `Rng.below` raises `ValueError` for a bound below one."""
    with pytest.raises(ValueError):
        Rng(5).below(bound)


@pytest.mark.parametrize(
    ("seed", "items", "expected"),
    [
        (7, list(range(10)), [9, 5, 8, 6, 1, 2, 4, 7, 0, 3]),
        (3, list("abcdef"), ["b", "e", "f", "c", "d", "a"]),
    ],
    ids=["ints", "letters"],
)
def test_rng_shuffle_follows_the_golden_permutation(
    seed: int, items: list[Any], expected: list[Any]
) -> None:
    """Test `Rng.shuffle` permutes a list in place as the reference does."""
    rng = Rng(seed)

    result = rng.shuffle(items)  # type: ignore[func-returns-value]

    assert result is None
    assert items == expected


def test_rng_split_gives_an_independent_stream_and_advances_the_parent() -> None:
    """Test a child is seeded with the parent's next number."""
    parent = Rng(7)

    child = parent.split()

    assert _stream(child, 3) == [
        0xB8B4C2977EABCE45,
        0xA65305FD338EC8FE,
        0x8CA3CBB6CA63129B,
    ]
    assert _stream(parent, 2) == [0x044C3CD7F43C661C, 0xE6984080BAB12A02]


def test_rng_reports_its_seed_after_drawing() -> None:
    """Test `Rng.seed` is the seed it started from, not its state."""
    rng = Rng(77)
    rng.next_u64()

    assert rng.seed == 77


@pytest.mark.parametrize("seed", [-1, _MASK], ids=["negative", "2_to_the_64"])
def test_rng_refuses_a_seed_outside_the_word(seed: int) -> None:
    """Test a seed outside `[0, 2**64)` raises `ValueError`."""
    with pytest.raises(ValueError):
        Rng(seed)


def test_rng_refuses_a_boolean_seed() -> None:
    """Test a `bool` is no seed: `TypeError`."""
    with pytest.raises(TypeError):
        Rng(True)


# ===========================================================================
# Choice domains
# ===========================================================================


def test_choice_domain_refuses_no_choices() -> None:
    """Test an empty choice domain is refused.

    Ports MOGA-VM test_decisions.py::test_choice_domain_rejects_an_empty_candidate_set.
    """
    with pytest.raises(StepDomainError):
        ChoiceDomain(())


def test_choice_domain_admits_eq_less_objects_by_identity() -> None:
    """Test an object whose `==` is identity matches only itself.

    Ports MOGA-VM
    test_decisions.py::test_choice_domain_admits_by_identity_for_eq_less_values.
    """
    first, second, absent = _Opaque(), _Opaque(), _Opaque()
    domain = ChoiceDomain((first, second))

    assert domain.cardinality == 2
    assert domain.admits(first)
    assert domain.admits(second)
    assert not domain.admits(absent)


def test_choice_domain_returns_the_objects_it_was_given() -> None:
    """Test `value_at` returns the very object given, not a copy."""
    choices = (_Opaque(), "text", (1, 2))

    domain = ChoiceDomain(choices)

    assert domain.value_at(0) is choices[0]
    assert domain.value_at(1) is choices[1]
    assert domain.choices == choices


def test_choice_domain_refuses_a_repeated_choice() -> None:
    """Test two equal choices are refused."""
    with pytest.raises(StepDomainError):
        ChoiceDomain(("x", "x"))


def test_choice_domain_tells_a_boolean_from_the_integer_one() -> None:
    """Test values compare type-strictly: `True` is not `1`."""
    domain = ChoiceDomain((1, True))

    assert domain.cardinality == 2
    assert domain.coordinate_of(1) == 0
    assert domain.coordinate_of(True) == 1


@pytest.mark.parametrize("index", [0, 1, 2], ids=["first", "second", "third"])
def test_choice_coordinates_round_trip(index: int) -> None:
    """Test a coordinate gives a value that gives the coordinate back.

    Ports MOGA-VM
    test_decisions.py::test_choice_domain_round_trips_a_value_through_its_coordinate.
    """
    domain = _choice("a", "b", "c")

    assert domain.coordinate_of(domain.value_at(index)) == index


def test_choice_value_at_past_the_end_raises_index_error() -> None:
    """Test an index past the last choice raises `IndexError`."""
    with pytest.raises(IndexError):
        _choice("a", "b").value_at(2)


def test_choice_coordinate_of_an_absent_value_raises_value_error() -> None:
    """Test `coordinate_of` raises `ValueError` for a value not among the choices."""
    with pytest.raises(ValueError):
        _choice("a", "b").coordinate_of("c")


# ===========================================================================
# Order domains
# ===========================================================================


def test_order_domain_counts_and_admits_only_permutations() -> None:
    """Test the count is `n!` and a full reordering is admitted.

    Ports MOGA-VM
    test_decisions.py::test_order_domain_counts_permutations_and_admits_only_permutations.
    """
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    domain = OrderDomain((a, b, c))

    assert domain.cardinality == 6
    assert domain.admits((c, a, b))
    assert domain.admits((a, b, c))


@pytest.mark.parametrize("size", [1, 2, 3, 4], ids=["1", "2", "3", "4"])
def test_order_domain_cardinality_is_the_factorial(size: int) -> None:
    """Test an order domain over `n` elements holds `n!` orderings."""
    domain = OrderDomain(tuple(Identifier(f"e{i}") for i in range(size)))

    assert domain.cardinality == [1, 2, 6, 24][size - 1]


def test_order_domain_refuses_a_prefix_a_repeat_or_a_list() -> None:
    """Test only a tuple holding each element once is admitted.

    Ports MOGA-VM
    test_decisions.py::test_order_domain_counts_permutations_and_admits_only_permutations.
    """
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    domain = OrderDomain((a, b, c))

    assert not domain.admits((a, b))
    assert not domain.admits((a, b, b))
    assert not domain.admits([a, b, c])


def test_order_domain_refuses_a_repeated_element() -> None:
    """Test two equal elements are refused.

    Ports MOGA-VM test_decisions.py::test_order_domain_rejects_repeated_elements.
    """
    a = Identifier("a")

    with pytest.raises(StepDomainError):
        OrderDomain((a, a))


def test_order_domain_refuses_no_elements() -> None:
    """Test an order domain over no element is refused."""
    with pytest.raises(StepDomainError):
        OrderDomain(())


def test_order_value_at_returns_the_objects_given_in_the_positions_order() -> None:
    """Test `positions[k]` is the index of the element placed `k`-th."""
    elements = (_Opaque(), _Opaque(), _Opaque())
    domain = OrderDomain(elements)

    ordering = domain.value_at((2, 0, 1))

    assert isinstance(ordering, tuple)
    assert len(ordering) == 3
    assert ordering[0] is elements[2]
    assert ordering[1] is elements[0]
    assert ordering[2] is elements[1]


def test_order_coordinates_round_trip_onto_fresh_elements() -> None:
    """Test positions read off one domain rebuild the ordering over fresh elements.

    Ports MOGA-VM
    test_decisions.py::test_order_domain_round_trips_a_permutation_through_its_coordinate.
    """
    a, b, c = Identifier("a"), Identifier("b"), Identifier("c")
    positions = OrderDomain((a, b, c)).coordinate_of((c, a, b))
    fresh = tuple(Identifier(name) for name in "abc")

    rebuilt = OrderDomain(fresh).value_at(positions)

    assert positions == (2, 0, 1)
    assert rebuilt == (fresh[2], fresh[0], fresh[1])


def test_order_value_at_refuses_a_non_permutation() -> None:
    """Test positions that repeat one are refused with `ValueError`.

    Ports MOGA-VM
    test_decisions.py::test_order_domain_rejects_a_coordinate_that_is_not_a_permutation.
    """
    with pytest.raises(ValueError):
        _order("a", "b").value_at((0, 0))


# ===========================================================================
# Strided runs and domains
# ===========================================================================


@pytest.mark.parametrize("bounds", [(64, 64), (10, 5)], ids=["empty", "backwards"])
def test_strided_run_refuses_an_empty_run(bounds: tuple[int, int]) -> None:
    """Test a run with `stop <= start` is refused.

    Ports MOGA-VM test_decisions.py::test_address_interval_rejects_an_empty_run.
    """
    with pytest.raises(StepDomainError):
        StridedRun(*bounds)


def test_strided_run_refuses_a_zero_stride() -> None:
    """Test a stride of zero is refused.

    Ports MOGA-VM test_decisions.py::test_address_interval_rejects_a_non_positive_step.
    """
    with pytest.raises(StepDomainError):
        StridedRun(0, 10, 0)


def test_strided_run_refuses_a_negative_stride() -> None:
    """Test a negative stride is refused.

    Ports MOGA-VM test_decisions.py::test_address_interval_rejects_a_non_positive_step.
    """
    with pytest.raises(StepDomainError):
        StridedRun(0, 10, -1)


@pytest.mark.parametrize("bounds", [(True, 5), (0, True)], ids=["start", "stop"])
def test_strided_run_refuses_a_boolean_bound(bounds: tuple[Any, Any]) -> None:
    """Test a `bool` bound raises `TypeError`."""
    with pytest.raises(TypeError):
        StridedRun(*bounds)


def test_strided_run_has_a_stride_of_one_by_default() -> None:
    """Test a run without a stride steps by one and counts its integers."""
    run = StridedRun(5, 9)

    assert (run.start, run.stop, run.stride, run.width) == (5, 9, 1, 4)


def test_strided_run_width_counts_the_strides() -> None:
    """Test a run of `[64, 200)` by 64 holds 64, 128 and 192."""
    assert StridedRun(64, 200, 64).width == 3


@pytest.mark.parametrize(
    ("value", "admitted"),
    [
        (64, True),
        (128, True),
        (192, True),
        (65, False),
        (0, False),
        (200, False),
        (True, False),
    ],
    ids=["64", "128", "192", "unaligned", "below", "stop", "boolean"],
)
def test_strided_run_admits_only_its_stride(value: Any, admitted: bool) -> None:
    """Test a run admits the integers `start + k * stride` below `stop`.

    Ports MOGA-VM
    test_decisions.py::test_strided_address_interval_admits_only_aligned_addresses.
    """
    assert StridedRun(64, 200, 64).admits(value) is admitted


def test_strided_domain_flattens_its_runs() -> None:
    """Test the runs' integers are indexed in order as one space.

    Ports MOGA-VM
    test_decisions.py::test_address_domain_flattens_disjoint_runs_into_one_index_space.
    """
    domain = _addresses((0, 3), (100, 105))

    assert domain.cardinality == 8
    assert [domain.value_at(index) for index in range(8)] == [
        0,
        1,
        2,
        100,
        101,
        102,
        103,
        104,
    ]


@pytest.mark.parametrize(
    "value", [3, 99, 105, True], ids=["gap_start", "gap_end", "past_stop", "boolean"]
)
def test_strided_domain_admits_nothing_between_runs_nor_a_boolean(value: Any) -> None:
    """Test a gap, the far side of the last run and `True` are not admitted.

    Ports MOGA-VM
    test_decisions.py::test_address_domain_admits_nothing_between_its_runs.
    """
    assert not _addresses((0, 3), (100, 105)).admits(value)


def test_strided_value_at_past_the_end_raises_index_error() -> None:
    """Test an index past the last integer raises `IndexError`.

    Ports MOGA-VM test_decisions.py::test_address_domain_index_out_of_range_raises.
    """
    domain = _addresses((0, 3))

    with pytest.raises(IndexError):
        domain.value_at(3)


def test_strided_coordinate_of_a_gap_raises_value_error() -> None:
    """Test an integer between the runs has no coordinate."""
    with pytest.raises(ValueError):
        _addresses((0, 3), (100, 105)).coordinate_of(3)


@pytest.mark.parametrize(
    "runs",
    [[(0, 10), (5, 15)], [(100, 105), (0, 3)]],
    ids=["overlapping", "unsorted"],
)
def test_strided_domain_refuses_unordered_runs(runs: list[tuple[int, int]]) -> None:
    """Test runs must be disjoint and ascending.

    Ports MOGA-VM
    test_decisions.py::test_address_domain_rejects_overlapping_or_unsorted_runs.
    """
    with pytest.raises(StepDomainError):
        _addresses(*runs)


def test_strided_domain_refuses_no_runs() -> None:
    """Test a domain of no run is refused.

    Ports MOGA-VM test_decisions.py::test_address_domain_rejects_an_empty_union.
    """
    with pytest.raises(StepDomainError):
        StridedDomain(())


def test_strided_domain_indexes_only_its_strides() -> None:
    """Test strided runs count and index only their aligned integers.

    Ports MOGA-VM
    test_decisions.py::test_strided_address_domain_counts_and_indexes_only_aligned_addresses.
    """
    domain = StridedDomain((StridedRun(64, 200, 64), StridedRun(512, 600, 32)))

    assert domain.cardinality == 6
    assert [domain.value_at(index) for index in range(6)] == [
        64,
        128,
        192,
        512,
        544,
        576,
    ]


@pytest.mark.parametrize(
    "index", [0, 2, 3, 7], ids=["first", "last_of_run", "second_run", "last"]
)
def test_strided_coordinates_round_trip(index: int) -> None:
    """Test an index gives an integer that gives the index back.

    Ports MOGA-VM
    test_decisions.py::test_address_domain_round_trips_an_address_through_its_coordinate.
    """
    domain = _addresses((0, 3), (100, 105))

    assert domain.coordinate_of(domain.value_at(index)) == index


def test_strided_domain_returns_the_runs_it_was_given() -> None:
    """Test the runs read back are the objects given."""
    first, second = StridedRun(0, 3), StridedRun(10, 12)

    domain = StridedDomain((first, second))

    assert domain.runs[0] is first
    assert domain.runs[1] is second


# ===========================================================================
# Recorder
# ===========================================================================


def test_recorder_records_the_step_and_its_answer() -> None:
    """Test a recorded step holds what was asked and what was answered.

    Ports MOGA-VM
    test_oracle.py::test_recording_oracle_records_what_was_asked_and_what_was_answered.
    """
    recorder = Recorder(RandomOracle(seed=5))
    domain = _choice(10, 20, 30)

    value = recorder.decide_dynamic(_OPTION, _SUBJECT, domain)

    assert len(recorder.trace) == 1
    (step,) = recorder.trace.steps
    assert step.kind == _OPTION
    assert step.subject is _SUBJECT
    assert step.decision is None
    assert step.cardinality == 3
    assert value in (10, 20, 30)
    assert step.coordinate == domain.coordinate_of(value)
    assert step.value == value


def test_recorder_returns_the_objects_of_the_domain() -> None:
    """Test the value answered is the domain's own object at the coordinate."""
    options = (_Opaque(), _Opaque())
    recorder = Recorder(_Constant(1))

    value = recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(options))

    assert value is options[1]


def test_an_answer_outside_the_domain_is_refused_and_not_recorded() -> None:
    """Test an off-domain answer raises and leaves the trace empty.

    Ports MOGA-VM
    test_oracle.py::test_recording_oracle_rejects_an_answer_outside_the_offered_domain.
    """
    recorder = Recorder(_Constant(999))

    with pytest.raises(InadmissibleAnswerError):
        recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(10, 20, 30))

    assert len(recorder.trace) == 0


def test_the_trace_keeps_its_prefix_after_the_oracle_fails() -> None:
    """Test the steps answered before an oracle's exception stay recorded."""

    class Failing:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, step: Any) -> int:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second")
            return 0

    recorder = Recorder(Failing())
    recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(1, 2))

    with pytest.raises(RuntimeError, match="second"):
        recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(1, 2))

    assert len(recorder.trace) == 1


def test_a_recorder_over_no_space_refuses_to_decide_a_name() -> None:
    """Test `decide(name)` needs a space or a configuration."""
    recorder = Recorder(_Constant(0))

    with pytest.raises(TraceError):
        recorder.decide(Identifier("unroll"))


def test_a_recorder_refuses_a_space_and_a_configuration_together() -> None:
    """Test a recorder cannot both ask a space and realize a configuration."""
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space)

    with pytest.raises(ValueError):
        Recorder(_Constant(0), space=tiling.space, configuration=configuration)


def test_finish_returns_the_trace_and_no_configuration_over_no_space() -> None:
    """Test a dynamic-only run finishes with its trace and `None`."""
    recorder = Recorder(_Constant(0))
    recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(1, 2))

    trace, configuration = recorder.finish()

    assert len(trace) == 1
    assert configuration is None


def test_decide_answers_a_variable_with_its_value() -> None:
    """Test a static step over a variable returns the value at the coordinate."""
    tiling = build_tiling_space()
    recorder = Recorder(_Constant(0), space=tiling.space)

    value = recorder.decide(tiling.unroll.name)

    assert value in (1, 2, 4)
    assert recorder.trace.steps[0].subject == tiling.unroll.name


def test_decide_answers_a_choice_with_the_chosen_alternative_object() -> None:
    """Test a choice's value is the `Alternative` object itself."""
    tiling = build_tiling_space()
    recorder = Recorder(_Constant(0), space=tiling.space)
    recorder.decide(tiling.unroll.name)

    value = recorder.decide(tiling.layout.name)

    assert value is tiling.tiled


def test_decide_refuses_a_decision_under_an_undecided_choice() -> None:
    """Test a variable under a choice not yet decided is pending: `TraceError`."""
    tiling = build_tiling_space()
    recorder = Recorder(_Constant(0), space=tiling.space)

    with pytest.raises(TraceError):
        recorder.decide(tiling.tile.name)


def test_decide_refuses_a_decision_asked_twice() -> None:
    """Test a decision answers once: `TraceError` the second time."""
    tiling = build_tiling_space()
    recorder = Recorder(_Constant(0), space=tiling.space)
    recorder.decide(tiling.unroll.name)

    with pytest.raises(TraceError):
        recorder.decide(tiling.unroll.name)


def test_realizing_asks_the_oracle_for_unassigned_decisions() -> None:
    """Test a realizing run answers assigned decisions itself and asks for the rest.

    Ports MOGA-VM test_extraction.py::test_realization_refuses_a_drifted_domain.
    """
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space, {tiling.unroll.name: 2})
    recorder = Recorder(_Scripted([1]), configuration=configuration)

    unroll = recorder.decide(tiling.unroll.name)
    layout = recorder.decide(tiling.layout.name)

    assert unroll == 2
    assert layout is tiling.flat


def test_realizing_refuses_to_finish_with_unasked_decisions() -> None:
    """Test a realizing run never asking an assigned decision cannot finish.

    Ports MOGA-VM
    test_extraction.py::test_strict_lowering_refuses_an_unconsumed_assignment.
    """
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space, {tiling.unroll.name: 2})
    recorder = Recorder(_Forbidden(), configuration=configuration)

    with pytest.raises(TraceError):
        recorder.finish()


def test_realizing_finishes_once_every_assigned_decision_was_asked() -> None:
    """Test asking every assigned decision lets the run finish."""
    tiling = build_tiling_space()
    configuration = Configuration(tiling.space, {tiling.unroll.name: 2})
    recorder = Recorder(_Forbidden(), configuration=configuration)
    recorder.decide(tiling.unroll.name)

    trace, result = recorder.finish()

    assert len(trace) == 1
    assert result is not None
    assert result.value(tiling.unroll.name) == 2


# ===========================================================================
# Oracles
# ===========================================================================


def test_shipped_oracles_are_search_oracles_by_protocol() -> None:
    """Test an oracle is any object with `decide`: a plain class conforms."""
    assert isinstance(_Constant(0), SearchOracle)


@pytest.mark.parametrize(
    "build",
    [
        lambda: RandomOracle(seed=0),
        lambda: ReplayOracle(Trace()),
        ExhaustiveOracle,
    ],
    ids=["random", "replay", "exhaustive"],
)
def test_shipped_oracles_are_search_oracles(build: Callable[[], object]) -> None:
    """Test each oracle fhy-core ships conforms to `SearchOracle`.

    Ports MOGA-VM test_oracle.py::test_uniform_oracle_is_a_search_oracle.
    """
    assert isinstance(build(), SearchOracle)


def test_random_oracle_reports_its_seed() -> None:
    """Test `RandomOracle(seed=n).seed` is `n`."""
    assert RandomOracle(seed=7).seed == 7


def test_random_oracle_without_a_seed_reports_the_seed_it_drew() -> None:
    """Test an unseeded oracle exposes a seed in `[0, 2**64)` to reproduce its run."""
    seed = RandomOracle(seed=None).seed

    assert isinstance(seed, int)
    assert 0 <= seed < _MASK


def test_random_oracle_draws_from_the_rng_given() -> None:
    """Test `RandomOracle(rng=r).rng` is `r`."""
    rng = Rng(3)

    assert RandomOracle(rng=rng).rng is rng


def test_random_oracle_refuses_a_seed_and_an_rng() -> None:
    """Test two randomness sources are refused.

    Ports MOGA-VM
    test_oracle.py::test_uniform_oracle_refuses_a_seed_and_a_generator_together.
    """
    with pytest.raises(ValueError):
        RandomOracle(seed=1, rng=Rng(2))


def test_random_oracle_reproduces_a_stream_from_its_seed() -> None:
    """Test the same seed gives the same answers across all domain shapes.

    Ports MOGA-VM
    test_oracle.py::test_uniform_oracle_reproduces_a_whole_stream_from_its_seed.
    """
    stream = [
        (_OPTION, _choice(10, 20, 30)),
        (_ADDRESS, _addresses((0, 64))),
        (_WALK_ORDER, _order("i", "j", "k")),
        (_ADDRESS, _addresses((0, 64))),
    ]

    first, first_values = _record(RandomOracle(seed=1234), stream)
    second, second_values = _record(RandomOracle(seed=1234), stream)

    assert first == second
    assert first_values == second_values


def test_random_oracle_differs_across_seeds() -> None:
    """Test two seeds disagree on 8 draws from a 4096-wide domain.

    Ports MOGA-VM test_oracle.py::test_uniform_oracle_differs_across_seeds.
    """
    stream = [(_ADDRESS, _addresses((0, 4096)))] * 8

    first, _ = _record(RandomOracle(seed=1), stream)
    second, _ = _record(RandomOracle(seed=2), stream)

    assert first.coordinates != second.coordinates


def test_random_oracle_covers_every_choice() -> None:
    """Test 200 draws from a 3-wide domain hit every choice.

    Ports MOGA-VM test_oracle.py::test_uniform_oracle_covers_every_admissible_choice.
    """
    _, values = _record(RandomOracle(seed=7), [(_OPTION, _choice(10, 20, 30))] * 200)

    assert set(values) == {10, 20, 30}


def test_random_oracle_reaches_every_run() -> None:
    """Test both runs of a split strided domain are drawn from.

    Ports MOGA-VM
    test_oracle.py::test_uniform_oracle_reaches_both_runs_of_a_split_address_domain.
    """
    stream = [(_ADDRESS, _addresses((0, 4), (1000, 1004)))] * 200

    _, values = _record(RandomOracle(seed=11), stream)

    assert set(values) == {0, 1, 2, 3, 1000, 1001, 1002, 1003}


def test_random_oracle_draws_every_permutation() -> None:
    """Test both orderings of a 2-element order domain are drawn.

    Ports MOGA-VM
    test_oracle.py::test_uniform_oracle_draws_every_permutation_of_a_small_order_domain.
    """
    trace, _ = _record(RandomOracle(seed=3), [(_WALK_ORDER, _order("i", "j"))] * 50)

    assert set(trace.coordinates) == {(0, 1), (1, 0)}


def test_random_oracle_draws_choice_coordinates_from_the_seeds_stream() -> None:
    """Test a seeded oracle's choice answers follow `Rng(seed).below(n)`."""
    stream = [(_OPTION, _choice("a", "b", "c"))] * 8

    _, values = _record(RandomOracle(seed=42), stream)

    assert values == ["c", "a", "a", "b", "a", "c", "a", "c"]


def test_random_oracle_draws_an_order_from_a_shuffle_of_the_positions() -> None:
    """Test an order answer is the shuffle of `[0, 1, 2, 3]` by the seed's `Rng`."""
    domain = OrderDomain(("a", "b", "c", "d"))

    trace, values = _record(RandomOracle(seed=0), [(_WALK_ORDER, domain)])

    assert trace.coordinates == ((2, 0, 1, 3),)
    assert values == [("c", "a", "b", "d")]


def _all_kinds_stream() -> list[tuple[str, Any]]:
    """Return a stream asking a choice, an address and an order."""
    return [
        (_OPTION, _choice(10, 20, 30)),
        (_ADDRESS, _addresses((0, 64))),
        (_WALK_ORDER, _order("i", "j", "k")),
    ]


def test_replay_reproduces_a_stream() -> None:
    """Test replaying a recorded stream gives the same answers and is exhausted.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_reproduces_a_recorded_stream_exactly.
    """
    stream = _all_kinds_stream()
    trace, original = _record(RandomOracle(seed=99), stream)
    replay = ReplayOracle(trace)

    _, replayed = _record(replay, stream)

    assert replayed == original
    assert replay.is_exhausted
    replay.finish()


def test_replay_oracle_is_not_exhausted_before_its_stream_is_asked() -> None:
    """Test a replay with steps left is not exhausted."""
    trace, _ = _record(RandomOracle(seed=99), _all_kinds_stream())

    assert not ReplayOracle(trace).is_exhausted


def test_replay_refuses_another_shape() -> None:
    """Test a stream asking a different shape of domain is refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_a_stream_that_asks_a_different_decision.
    """
    trace, _ = _record(RandomOracle(seed=1), [(_OPTION, _choice(10, 20, 30))])

    with pytest.raises(ReplayMismatchError):
        _record(ReplayOracle(trace), [(_OPTION, _addresses((0, 3)))])


def test_replay_refuses_another_size() -> None:
    """Test the same step over a domain of another size is refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_the_same_decision_over_a_different_domain.
    """
    trace, _ = _record(RandomOracle(seed=1), [(_ADDRESS, _addresses((0, 64)))])

    with pytest.raises(ReplayMismatchError):
        _record(ReplayOracle(trace), [(_ADDRESS, _addresses((0, 32)))])


def test_replay_refuses_a_moved_run_of_the_same_size() -> None:
    """Test a run that moved but kept its width is refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_the_same_decision_over_a_different_domain.
    """
    trace, _ = _record(RandomOracle(seed=1), [(_ADDRESS, _addresses((0, 64)))])

    with pytest.raises(ReplayMismatchError):
        _record(ReplayOracle(trace), [(_ADDRESS, _addresses((64, 128)))])


def test_replay_refuses_reordered_plain_choices() -> None:
    """Test the same plain choices in another order are refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_the_same_decision_over_a_different_domain.
    """
    trace, _ = _record(RandomOracle(seed=1), [(_OPTION, _choice("a", "b", "c"))])

    with pytest.raises(ReplayMismatchError):
        _record(ReplayOracle(trace), [(_OPTION, _choice("c", "b", "a"))])


def test_replay_refuses_a_longer_stream() -> None:
    """Test asking beyond the recorded steps is refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_a_stream_longer_than_the_recorded_point.
    """
    trace, _ = _record(RandomOracle(seed=1), [(_OPTION, _choice(10, 20))])
    replay = ReplayOracle(trace)
    recorder = Recorder(replay)
    recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(10, 20))

    with pytest.raises(ReplayMismatchError):
        recorder.decide_dynamic(_OPTION, _SUBJECT, _choice(10, 20))


def test_replay_finish_refuses_a_shorter_stream() -> None:
    """Test finishing a replay that left recorded steps unasked is refused.

    Ports MOGA-VM
    test_oracle.py::test_replay_oracle_refuses_a_stream_longer_than_the_recorded_point.
    """
    trace, _ = _record(
        RandomOracle(seed=1), [(_OPTION, _choice(10, 20)), (_OPTION, _choice(10, 20))]
    )
    replay = ReplayOracle(trace)
    Recorder(replay).decide_dynamic(_OPTION, _SUBJECT, _choice(10, 20))

    with pytest.raises(ReplayMismatchError):
        replay.finish()


def test_replay_does_not_compare_a_dynamic_steps_subject() -> None:
    """Test a replay asking another subject for the same step is answered."""
    stream = [(_OPTION, _choice(10, 20, 30))]
    trace, original = _record(RandomOracle(seed=8), stream)

    _, replayed = _record(ReplayOracle(trace), stream, subject=Identifier("other"))

    assert replayed == original


@pytest.mark.parametrize(
    "make",
    [Identifier, lambda _name: _Opaque()],
    ids=["identifiers", "opaque_objects"],
)
def test_replay_answers_options_rebuilt_fresh_with_the_same_size(
    make: Callable[[str], object],
) -> None:
    """Test freshly built identifiers or objects replay by position, not by identity."""
    recorded = [(_OPTION, ChoiceDomain((make("p"), make("q"), make("r"))))]
    trace, _ = _record(RandomOracle(seed=21), recorded)
    fresh = ChoiceDomain((make("p"), make("q"), make("r")))

    _, replayed = _record(ReplayOracle(trace), [(_OPTION, fresh)])

    (index,) = trace.coordinates
    assert isinstance(index, int)
    assert replayed == [fresh.value_at(index)]


def test_replay_answers_repeated_subjects_in_ask_order() -> None:
    """Test steps sharing a subject replay positionally.

    Ports MOGA-VM
    test_extraction.py::test_same_key_axes_are_consumed_first_in_first_out.
    """
    stream = [
        (_ADDRESS, _addresses((0, 1000))),
        (_ADDRESS, _addresses((0, 1000))),
        (_ADDRESS, _addresses((0, 1000))),
    ]
    trace, original = _record(RandomOracle(seed=4), stream)

    _, replayed = _record(ReplayOracle(trace), stream)

    assert replayed == original
    assert len(set(original)) == 3


def test_exhaustive_oracle_visits_every_path_in_lexicographic_order() -> None:
    """Test successive runs take the 6 paths of sizes 2 and 3 in coordinate order."""
    oracle = ExhaustiveOracle()
    paths = []
    while True:
        trace, _ = _record(
            oracle, [(_OPTION, _choice("x", "y")), (_OPTION, _choice(1, 2, 3))]
        )
        paths.append(trace.coordinates)
        if not oracle.advance():
            break

    assert paths == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


# ===========================================================================
# Traces
# ===========================================================================


def test_traversed_cardinality_multiplies_the_steps() -> None:
    """Test the product of the domains' sizes is the cardinality traversed.

    Ports MOGA-VM
    test_decisions.py::test_search_point_traversed_cardinality_multiplies_along_the_path.
    """
    trace, _ = _record(
        _Constant(0),
        [
            (_OPTION, _choice(1, 2, 3)),
            (_OPTION, _choice(1, 2, 3, 4)),
            (_OPTION, _choice(1, 2, 3, 4, 5)),
        ],
    )

    assert trace.traversed_cardinality == 60


def test_empty_trace_counts_one_and_displays_zero_steps() -> None:
    """Test an empty trace's product is 1 and it reads "0 steps".

    Ports MOGA-VM
    test_decisions.py::test_empty_search_point_has_a_traversed_cardinality_of_one.
    """
    trace = Trace()

    assert len(trace) == 0
    assert trace.traversed_cardinality == 1
    assert str(trace) == "0 steps"


def test_trace_displays_its_steps_by_kind_in_first_seen_order() -> None:
    """Test the text counts steps per kind, in the order the kinds first occur."""
    choice = (_OPTION, _choice(1, 2))
    address = (_ADDRESS, _addresses((0, 8)))
    trace, _ = _record(_Constant(0), [choice, address, choice, address, address])

    assert str(trace) == f"5 steps (2 {_OPTION}, 3 {_ADDRESS})"


def test_of_kind_keeps_ask_order() -> None:
    """Test `of_kind` returns the steps of one kind in the order asked.

    Ports MOGA-VM test_decisions.py::test_search_point_groups_by_kind_in_ask_order.
    """
    first, second, third = Identifier("a"), Identifier("b"), Identifier("c")
    recorder = Recorder(_Constant(0))
    recorder.decide_dynamic(_OPTION, first, _choice(1, 2))
    recorder.decide_dynamic(_ADDRESS, second, _addresses((0, 8)))
    recorder.decide_dynamic(_OPTION, third, _choice(1, 2))

    options = recorder.trace.of_kind(_OPTION)

    assert [step.subject for step in options] == [first, third]
    assert recorder.trace.of_kind("moga.cir.unseen") == ()


def test_trace_coordinates_are_a_bare_tuple() -> None:
    """Test `coordinates` holds only the answers: an int or a tuple for an order.

    Ports MOGA-VM test_decisions.py::test_search_point_coordinates_are_a_bare_vector.
    """
    trace, _ = _record(
        _Scripted([1, (1, 2, 0)]),
        [(_OPTION, _choice("a", "b")), (_WALK_ORDER, _order("i", "j", "k"))],
    )

    assert trace.coordinates == (1, (1, 2, 0))


def test_trace_coordinates_serialize_tagged_by_their_shape() -> None:
    """Test a choice's coordinate is written as an index and an order's as positions.

    Ports MOGA-VM
    test_observers.py::test_coordinates_serialize_choice_and_order_decisions_faithfully.
    """
    trace, _ = _record(
        _Scripted([1, (1, 2, 0)]),
        [(_OPTION, _choice("a", "b")), (_WALK_ORDER, _order("i", "j", "k"))],
    )

    steps = json.loads(trace.to_json())["steps"]

    assert [step["coordinate"] for step in steps] == [
        {"index": 1},
        {"order": [1, 2, 0]},
    ]


def test_trace_steps_are_written_with_their_fields() -> None:
    """Test a step's JSON holds kind, subject, decision, domain, coordinate, value."""
    trace, _ = _record(_Constant(0), [(_OPTION, _choice("a", "b"))])

    (step,) = json.loads(trace.to_json())["steps"]

    assert sorted(step) == [
        "coordinate",
        "decision",
        "domain",
        "kind",
        "subject",
        "value",
    ]
    assert step["kind"] == _OPTION
    assert step["decision"] is None


def test_an_opaque_value_is_written_null_and_read_back_as_none() -> None:
    """Test a step answered with an `eq=False` object loses its value in text."""
    trace, _ = _record(_Constant(0), [(_OPTION, ChoiceDomain((_Opaque(), _Opaque())))])

    (step,) = json.loads(trace.to_json())["steps"]
    restored = Trace.from_json(trace.to_json())

    assert step["value"] is None
    assert restored.steps[0].value is None


def test_a_trace_recorded_in_process_returns_the_object_answered() -> None:
    """Test `step.value` is the very object the recorder returned."""
    options = (_Opaque(), _Opaque())
    recorder = Recorder(_Constant(1))

    answered = recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(options))

    assert recorder.trace.steps[0].value is answered
    assert answered is options[1]


def test_traces_from_one_seed_and_domains_are_equal_and_hash_alike() -> None:
    """Test `==` and `hash` are structural over the steps."""
    stream = [(_OPTION, _choice(1, 2, 3)), (_ADDRESS, _addresses((0, 64)))]

    first, _ = _record(RandomOracle(seed=6), stream)
    second, _ = _record(RandomOracle(seed=6), stream)

    assert first == second
    assert hash(first) == hash(second)


def test_traces_with_different_coordinates_differ() -> None:
    """Test two traces over one domain answered differently are unequal."""
    stream = [(_OPTION, _choice(1, 2, 3))]

    first, _ = _record(_Constant(0), stream)
    second, _ = _record(_Constant(1), stream)

    assert first != second


def test_trace_is_serializable() -> None:
    """Test a `Trace` is a `Serializable`."""
    assert isinstance(Trace(), Serializable)


def test_trace_round_trips_through_its_json() -> None:
    """Test decoding a trace's text gives an equal trace of plain values."""
    trace, _ = _record(
        RandomOracle(seed=2),
        [(_OPTION, _choice("a", "b", "c")), (_ADDRESS, _addresses((0, 64)))],
    )

    assert Trace.from_json(trace.to_json()) == trace


def test_a_trace_replays_from_its_json() -> None:
    """Test a trace read back from text answers the same stream.

    Analog of MOGA-VM
    test_search_strategy_integration.py::test_a_trace_row_replays_to_the_same_committed_addresses.
    """
    stream = [(_OPTION, _choice("a", "b", "c")), (_ADDRESS, _addresses((0, 64)))]
    trace, original = _record(RandomOracle(seed=31), stream)
    restored = Trace.from_json(trace.to_json())
    replay = ReplayOracle(restored)

    _, replayed = _record(replay, stream)

    assert replayed == original
    assert replay.is_exhausted


# ===========================================================================
# Space methods
# ===========================================================================


def test_sample_returns_a_complete_configuration_and_its_trace() -> None:
    """Test `Space.sample` answers every active decision."""
    tiling = build_tiling_space()

    configuration, trace = tiling.space.sample(RandomOracle(seed=3))

    assert configuration.is_complete()
    assert len(trace) == len(configuration.entries)


def test_sample_is_reproducible_from_the_seed() -> None:
    """Test one seed over one space gives one configuration and trace."""
    tiling = build_tiling_space()

    first, first_trace = tiling.space.sample(RandomOracle(seed=11))
    second, second_trace = tiling.space.sample(RandomOracle(seed=11))

    assert first.key() == second.key()
    assert first_trace == second_trace


def test_sample_refuses_a_variable_with_an_unbounded_domain() -> None:
    """Test a step over an unbounded natural cannot be offered: `NotEnumerableError`."""
    space = Space(
        variables=(Variable(param=create_natural_param(), name=Identifier("n")),)
    )

    with pytest.raises(NotEnumerableError):
        space.sample(RandomOracle(seed=0))


def test_sample_uniform_returns_a_complete_configuration() -> None:
    """Test `Space.sample_uniform` draws a complete configuration and its trace."""
    tiling = build_tiling_space(unroll_on_tiled_only=True)

    configuration, trace = tiling.space.sample_uniform(Rng(5))

    assert configuration.is_complete()
    assert len(trace) == len(configuration.entries)


def test_replay_rebuilds_the_sampled_configuration() -> None:
    """Test replaying a trace gives a configuration with the sampled key."""
    tiling = build_tiling_space()
    configuration, trace = tiling.space.sample(RandomOracle(seed=8))

    replayed = tiling.space.replay(trace)

    assert replayed.key() == configuration.key()


def test_replay_accepts_static_steps_in_another_order() -> None:
    """Test a trace whose steps are not in decision order still replays."""
    tiling = build_tiling_space()
    configuration, trace = tiling.space.sample(_Constant(0))

    replayed = tiling.space.replay(Trace(reversed(trace.steps)))

    assert replayed.key() == configuration.key()


def test_trace_replays_onto_a_relabeled_space() -> None:
    """Test a trace recorded over one space replays into a fresh copy with an equal key.

    Analog of MOGA-VM
    test_random_search.py::test_a_recorded_point_replays_onto_a_freshly_built_module.
    """
    recorded, relabeled = build_tiling_space(), build_tiling_space()
    configuration, trace = recorded.space.sample(RandomOracle(seed=13))

    replayed = relabeled.space.replay(trace)

    assert replayed.space is relabeled.space
    assert replayed.key() == configuration.key()


def test_configuration_trace_replays_to_the_same_configuration() -> None:
    """Test `Configuration.trace()` then `Space.replay` round trips."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    replayed = tiling.space.replay(configuration.trace())

    assert replayed.key() == configuration.key()


def test_configuration_trace_names_each_decision_with_its_coordinate() -> None:
    """Test the trace of a complete configuration has a step per decision."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    trace = configuration.trace()

    subjects = {step.subject for step in trace}
    assert subjects == {tiling.unroll.name, tiling.layout.name, tiling.tile.name}
    layout_step = next(step for step in trace if step.subject == tiling.layout.name)
    assert layout_step.coordinate == 0
    assert layout_step.cardinality == 2


def test_replay_refuses_a_trace_missing_a_step() -> None:
    """Test an active decision with no step is refused."""
    tiling = build_tiling_space()
    _, trace = tiling.space.sample(_Constant(0))

    with pytest.raises(ReplayMismatchError):
        tiling.space.replay(Trace(trace.steps[:-1]))


def test_replay_refuses_a_trace_with_two_steps_for_one_decision() -> None:
    """Test a decision answered twice is refused."""
    tiling = build_tiling_space()
    _, trace = tiling.space.sample(_Constant(0))

    with pytest.raises(ReplayMismatchError):
        tiling.space.replay(Trace((*trace.steps, trace.steps[0])))


def test_cardinality_of_a_choice_sums_its_alternatives_products() -> None:
    """Test a choice of a 3-valued alternative and an empty one counts 4."""
    holder = make_alternative("a", (make_variable("v", 1, 2, 3),))
    space = Space(choices=(make_choice("c", holder, make_alternative("b")),))

    assert space.cardinality() == Cardinality(CardinalityKind.EXACT, 4, None)


def test_cardinality_of_independent_variables_multiplies() -> None:
    """Test variables over 2 and 3 values count 6."""
    space = Space(variables=(make_variable("x", 1, 2), make_variable("y", 1, 2, 3)))

    assert space.cardinality() == Cardinality(CardinalityKind.EXACT, 6, None)


def test_cardinality_of_the_tiling_space_is_nine() -> None:
    """Test 3 unrolls times (2 tiles or a flat layout) count 9."""
    assert build_tiling_space().space.cardinality() == Cardinality(
        CardinalityKind.EXACT, 9, None
    )


def test_cardinality_of_an_unbounded_variable_names_it() -> None:
    """Test a variable over unbounded naturals makes the count unbounded."""
    unbounded = Variable(param=create_natural_param(), name=Identifier("n"))
    space = Space(variables=(unbounded,))

    cardinality = space.cardinality()

    assert cardinality.kind is CardinalityKind.UNBOUNDED
    assert cardinality.count is None
    assert cardinality.decision == unbounded.name


def test_enumerate_yields_each_configuration_once() -> None:
    """Test enumeration yields `count` distinct complete configurations."""
    tiling = build_tiling_space()

    configurations = list(tiling.space.enumerate())

    assert len(configurations) == 9
    assert len({configuration.key() for configuration in configurations}) == 9
    assert all(configuration.is_complete() for configuration in configurations)


def test_mutate_changes_a_complete_configuration() -> None:
    """Test a mutation returns a configuration with another key."""
    tiling = build_tiling_space()
    configuration = build_complete_configuration(tiling)

    mutated, trace = tiling.space.mutate(configuration, Rng(3))

    assert mutated.is_complete()
    assert mutated.key() != configuration.key()
    assert len(trace) == len(mutated.entries)


def test_mutate_refuses_an_incomplete_configuration() -> None:
    """Test mutating a configuration that leaves a decision open is a `TraceError`."""
    tiling = build_tiling_space()

    with pytest.raises(TraceError):
        tiling.space.mutate(Configuration(tiling.space), Rng(3))


# ===========================================================================
# Cardinality
# ===========================================================================


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (CardinalityKind.EXACT, "exact"),
        (CardinalityKind.AT_LEAST, "at_least"),
        (CardinalityKind.UNBOUNDED, "unbounded"),
        (CardinalityKind.UNKNOWN, "unknown"),
    ],
    ids=["exact", "at_least", "unbounded", "unknown"],
)
def test_cardinality_kind_is_a_string_enum(member: CardinalityKind, value: str) -> None:
    """Test each kind compares equal to its text."""
    assert member == value


def test_cardinality_is_a_frozen_dataclass() -> None:
    """Test a `Cardinality` holds its fields and refuses assignment."""
    cardinality = Cardinality(CardinalityKind.EXACT, 3, None)

    assert is_dataclass(cardinality)
    with pytest.raises(FrozenInstanceError):
        cardinality.count = 4  # type: ignore[misc]
