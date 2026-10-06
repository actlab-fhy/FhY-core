"""Tests of what the binding adds around the Rust traces, oracles and domains.

The class structure and freezing, the objects the domains and traces keep,
the public `Trace` registered under its type id, pickling, `repr` and
cyclic garbage collection through the objects the new classes hold.
"""

import gc
import pickle
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core import search_space as public
from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    ChoiceDomain,
    OrderDomain,
    RandomOracle,
    Recorder,
    Rng,
    StridedDomain,
    StridedRun,
    Trace,
)
from fhy_core.serialization import (
    Serializable,
    deserialize_value,
    serialize_value,
)

_SUBJECT = Identifier("subject")
_OPTION = "moga.cir.option"


class _Opaque:
    """An object whose `==` is identity, as MOGA-VM's options are."""


class _First:
    """An oracle answering the first coordinate of every step."""

    def decide(self, step: Any) -> int:
        return 0


def _recorded_trace() -> Trace:
    """Return a trace of one choice step and one strided step."""
    recorder = Recorder(RandomOracle(seed=4))
    recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(("a", "b", "c")))
    recorder.decide_dynamic(
        "moga.cir.address", _SUBJECT, StridedDomain((StridedRun(0, 64),))
    )
    return recorder.trace


# ===========================================================================
# Class structure
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    [
        "Rng",
        "ChoiceDomain",
        "OrderDomain",
        "StridedRun",
        "StridedDomain",
        "PendingStep",
        "RandomOracle",
        "ReplayOracle",
        "ExhaustiveOracle",
        "Recorder",
        "TraceStep",
    ],
)
def test_new_classes_are_the_rust_classes_exported_as_they_are(name: str) -> None:
    """Test each new class is its `_rs` class, in module `fhy_core._rs`."""
    exported = getattr(public, name)

    assert exported is getattr(_rs, name)
    assert exported.__module__ == "fhy_core._rs"


def test_trace_is_a_thin_subclass_of_its_rust_class() -> None:
    """Test the public `Trace` subclasses `_rs.Trace` and is `Serializable`."""
    assert issubclass(Trace, _rs.Trace)
    assert issubclass(Trace, Serializable)
    assert Trace is not _rs.Trace


@pytest.mark.parametrize(
    "build",
    [
        lambda: Rng(0),
        lambda: ChoiceDomain((1, 2)),
        lambda: OrderDomain((1, 2)),
        lambda: StridedRun(0, 4),
        lambda: StridedDomain((StridedRun(0, 4),)),
        lambda: RandomOracle(seed=0),
        lambda: Recorder(_First()),
        _recorded_trace,
        lambda: _recorded_trace().steps[0],
    ],
    ids=[
        "rng",
        "choice_domain",
        "order_domain",
        "strided_run",
        "strided_domain",
        "random_oracle",
        "recorder",
        "trace",
        "trace_step",
    ],
)
def test_new_values_are_frozen(build: Callable[[], Any]) -> None:
    """Test no new class takes a new attribute."""
    value = build()

    with pytest.raises(AttributeError):
        value.extra = 1


def test_a_recorded_trace_is_an_instance_of_the_public_class() -> None:
    """Test the binding builds `Trace`, not `_rs.Trace`, for the traces it returns."""
    assert type(_recorded_trace()) is Trace


def test_a_trace_read_from_text_is_an_instance_of_the_class_it_was_read_by() -> None:
    """Test `Trace.from_json` returns a `Trace`."""
    assert type(Trace.from_json(_recorded_trace().to_json())) is Trace


# ===========================================================================
# Kept objects
# ===========================================================================


def test_choice_domain_keeps_each_choice_object() -> None:
    """Test `choices` holds the objects given, by identity."""
    first, second = _Opaque(), _Opaque()

    domain = ChoiceDomain((first, second))

    assert domain.choices[0] is first
    assert domain.choices[1] is second


def test_order_domain_keeps_each_element_object() -> None:
    """Test `elements` holds the objects given, by identity."""
    first, second = _Opaque(), _Opaque()

    domain = OrderDomain((first, second))

    assert domain.elements[0] is first
    assert domain.elements[1] is second


def test_a_trace_step_recorded_here_keeps_the_object_answered() -> None:
    """Test `step.value` is the object the recorder returned, by identity."""
    options = (_Opaque(), _Opaque())
    recorder = Recorder(_First())

    answered = recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain(options))

    assert answered is options[0]
    assert recorder.trace.steps[0].value is answered


def test_a_trace_step_keeps_its_subject_object() -> None:
    """Test `step.subject` is the identifier the step was asked about."""
    recorder = Recorder(_First())
    recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((1, 2)))

    assert recorder.trace.steps[0].subject is _SUBJECT


def test_replay_oracle_keeps_its_trace() -> None:
    """Test `ReplayOracle.trace` is the trace given."""
    trace = _recorded_trace()

    assert public.ReplayOracle(trace).trace is trace


def test_a_trace_keeps_the_step_objects_it_was_built_from() -> None:
    """Test `Trace(steps).steps` holds the same `TraceStep` objects."""
    source = _recorded_trace()

    rebuilt = Trace(source.steps)

    assert rebuilt.steps[0] is source.steps[0]
    assert rebuilt.steps[1] is source.steps[1]


def test_trace_refuses_a_step_that_is_no_trace_step() -> None:
    """Test a non-`TraceStep` among the steps raises `TypeError`."""
    with pytest.raises(TypeError):
        Trace(("not a step",))


# ===========================================================================
# Serialization
# ===========================================================================


def test_trace_serializes_under_its_type_id() -> None:
    """Test `Trace` is registered under `search_space.trace`."""
    assert Trace().get_serialization_class_type_id() == "search_space.trace"


def test_trace_payload_dict_round_trips() -> None:
    """Test decoding a trace's V2 dict gives an equal `Trace`."""
    trace = _recorded_trace()

    decoded = Trace.deserialize_from_dict(trace.serialize_to_dict())

    assert type(decoded) is Trace
    assert decoded == trace
    assert decoded.serialize_to_dict() == trace.serialize_to_dict()


def test_trace_round_trips_through_the_serialization_registry() -> None:
    """Test `deserialize_value` finds `Trace` by its type id."""
    trace = _recorded_trace()

    decoded = deserialize_value(serialize_value(trace))

    assert type(decoded) is Trace
    assert decoded == trace


# ===========================================================================
# Pickling
# ===========================================================================


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_trace_pickles_to_an_equal_trace(protocol: int) -> None:
    """Test a trace pickles, under every protocol, to an equal trace."""
    trace = _recorded_trace()

    restored = pickle.loads(pickle.dumps(trace, protocol=protocol))

    assert type(restored) is Trace
    assert restored == trace
    assert hash(restored) == hash(trace)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_rng_pickled_mid_stream_continues_the_stream(protocol: int) -> None:
    """Test a pickled `Rng` goes on from where the original stood."""
    rng = Rng(5)
    for _ in range(3):
        rng.next_u64()

    restored = pickle.loads(pickle.dumps(rng, protocol=protocol))

    assert restored.seed == 5
    assert [restored.next_u64() for _ in range(4)] == [rng.next_u64() for _ in range(4)]


# ===========================================================================
# repr
# ===========================================================================


@pytest.mark.parametrize(
    ("build", "name"),
    [
        (lambda: Rng(1), "Rng"),
        (lambda: ChoiceDomain((1, 2)), "ChoiceDomain"),
        (lambda: OrderDomain((1, 2)), "OrderDomain"),
        (lambda: StridedRun(0, 4), "StridedRun"),
        (lambda: StridedDomain((StridedRun(0, 4),)), "StridedDomain"),
        (_recorded_trace, "Trace"),
        (lambda: _recorded_trace().steps[0], "TraceStep"),
    ],
    ids=[
        "rng",
        "choice_domain",
        "order_domain",
        "strided_run",
        "strided_domain",
        "trace",
        "trace_step",
    ],
)
def test_repr_names_the_class(build: Callable[[], Any], name: str) -> None:
    """Test each new class's `repr` mentions its class name."""
    assert name in repr(build())


def test_strided_run_repr_shows_its_bounds() -> None:
    """Test a run's `repr` shows where it starts and stops."""
    text = repr(StridedRun(64, 200, 64))

    assert "64" in text
    assert "200" in text


# ===========================================================================
# Cyclic garbage collection
# ===========================================================================


def _collects(build: Callable[[], object]) -> bool:
    """Return whether the cycle `build` makes is freed by `gc.collect()`."""
    watched = weakref.ref(build())
    gc.collect()
    return watched() is None


class _CyclicOracle:
    """An oracle holding the recorder that holds it."""

    def decide(self, step: Any) -> int:
        return 0


def test_cycle_through_a_python_oracle_held_by_a_recorder_is_collected() -> None:
    """Test an oracle whose attribute is the recorder holding it is freed."""

    def build() -> object:
        oracle = _CyclicOracle()
        oracle.recorder = Recorder(oracle)  # type: ignore[attr-defined]
        return oracle

    assert _collects(build)


def test_cycle_through_a_value_a_trace_keeps_is_collected() -> None:
    """Test an answered object whose attribute is its own trace is freed."""

    def build() -> object:
        option = _Opaque()
        recorder = Recorder(_First())
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((option,)))
        option.trace = recorder.trace  # type: ignore[attr-defined]
        return option

    assert _collects(build)


def test_cycle_through_a_choice_domain_is_collected() -> None:
    """Test a choice whose attribute is the domain holding it is freed."""

    def build() -> object:
        option = _Opaque()
        option.domain = ChoiceDomain((option,))  # type: ignore[attr-defined]
        return option

    assert _collects(build)


def test_cycle_through_a_replay_oracle_held_by_its_trace_value_is_collected() -> None:
    """Test a value of the trace a replay oracle holds can hold that oracle."""

    def build() -> object:
        option = _Opaque()
        recorder = Recorder(_First())
        recorder.decide_dynamic(_OPTION, _SUBJECT, ChoiceDomain((option,)))
        option.replay = public.ReplayOracle(recorder.trace)  # type: ignore[attr-defined]
        return option

    assert _collects(build)
