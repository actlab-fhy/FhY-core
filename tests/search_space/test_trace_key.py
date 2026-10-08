"""The interface suite of `Trace.key` and `TraceKey`.

A trace key is what a run's answers say, apart from the subjects minted for
that run: two runs that answered alike over the same domains have equal keys,
also across alpha-renamed spaces, while two runs that placed an address
differently have different keys. The suite covers the key's value
semantics, its accessors, its refusals and its pickle.
"""

import pickle
from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    ChoiceDomain,
    ConfigurationKey,
    OrderDomain,
    RandomOracle,
    Recorder,
    StridedDomain,
    StridedRun,
    Trace,
    TraceKey,
)
from fhy_core.serialization import DeserializationValueError
from fhy_core.traits import FrozenMixin

from .conftest import (
    build_complete_configuration,
    build_dynamic_trace,
    build_tiling_space,
)


class _Answers:
    """An oracle answering the given coordinates, one per step, in order."""

    def __init__(self, *coordinates: Any) -> None:
        self._coordinates = iter(coordinates)

    def decide(self, step: Any) -> Any:
        return next(self._coordinates)


def _single_step(kind: str, domain: Any, coordinate: Any) -> Trace:
    """Return the trace of one dynamic step of `kind` answered `coordinate`."""
    recorder = Recorder(_Answers(coordinate))
    recorder.decide_dynamic(kind, Identifier("subject"), domain)
    return recorder.trace


# ===========================================================================
# Trace.key
# ===========================================================================


def test_a_trace_has_a_trace_key() -> None:
    """Test `Trace.key()` returns a `TraceKey`."""
    key = build_dynamic_trace().key()

    assert type(key) is TraceKey
    assert isinstance(key, FrozenMixin)


def test_the_key_of_the_empty_trace_is_empty() -> None:
    """Test a trace with no step has a key of length 0 and no coordinate."""
    key = Trace().key()

    assert len(key) == 0
    assert key.coordinates == ()
    assert key == Trace().key()


def test_a_key_has_the_traces_length_and_coordinates() -> None:
    """Test `len` and `coordinates` of a key are the trace's own."""
    trace = build_dynamic_trace(option=2, address=17)

    key = trace.key()

    assert len(key) == len(trace) == 2
    assert key.coordinates == trace.coordinates == (2, 17)


def test_a_key_holds_an_order_coordinate_as_a_tuple() -> None:
    """Test the coordinate of an order step is the tuple of positions."""
    elements = tuple(Identifier(name) for name in ("p", "q", "r"))
    recorder = Recorder(_Answers((2, 0, 1)))
    recorder.decide_dynamic("tests.order", Identifier("subject"), OrderDomain(elements))

    key = recorder.trace.key()

    assert key.coordinates == ((2, 0, 1),)


def test_runs_with_fresh_subjects_and_the_same_answers_have_one_key() -> None:
    """Test the traces differ, the keys are equal and hash alike."""
    first = build_dynamic_trace(Identifier("first"))
    second = build_dynamic_trace(Identifier("second"))

    assert first != second
    assert first.key() == second.key()
    assert hash(first.key()) == hash(second.key())
    assert not first.key() != second.key()
    assert len({first.key(), second.key()}) == 1
    assert {first.key(): "measured"}[second.key()] == "measured"


def test_a_different_dynamic_coordinate_gives_a_different_key() -> None:
    """Test a run that answered another coordinate has another key."""
    base = build_dynamic_trace(option=1, address=5).key()

    assert build_dynamic_trace(option=0, address=5).key() != base
    assert build_dynamic_trace(option=1, address=6).key() != base
    assert len({base, build_dynamic_trace(address=6).key()}) == 2


def test_a_different_step_kind_gives_a_different_key() -> None:
    """Test the same answer to the same domain under another kind differs."""
    domain = ChoiceDomain(("a", "b", "c"))

    assert (
        _single_step("tests.one", domain, 1).key()
        != _single_step("tests.two", domain, 1).key()
    )
    assert (
        _single_step("tests.one", domain, 1).key()
        == _single_step("tests.one", domain, 1).key()
    )


def test_a_different_domain_gives_a_different_key() -> None:
    """Test the same coordinate over another domain is another key."""
    three = _single_step("tests.one", ChoiceDomain(("a", "b", "c")), 1)
    two = _single_step("tests.one", ChoiceDomain(("a", "b")), 1)
    renamed = _single_step("tests.one", ChoiceDomain(("a", "z", "c")), 1)

    assert three.key() != two.key()
    assert three.key() != renamed.key()
    assert (
        three.key() == _single_step("tests.one", ChoiceDomain(("a", "b", "c")), 1).key()
    )


def test_a_different_address_layout_gives_a_different_key() -> None:
    """Test runs that placed an address in another layout have another key."""
    one_run = StridedDomain((StridedRun(0, 64),))
    two_runs = StridedDomain((StridedRun(0, 32), StridedRun(64, 96)))

    assert (
        _single_step("tests.address", one_run, 5).key()
        != _single_step("tests.address", two_runs, 5).key()
    )


def test_the_order_of_the_steps_matters() -> None:
    """Test the same two answers in the other order are another key."""
    domain = ChoiceDomain(("a", "b", "c"))

    def run(first: int, second: int) -> Trace:
        recorder = Recorder(_Answers(first, second))
        recorder.decide_dynamic("tests.one", Identifier("subject"), domain)
        recorder.decide_dynamic("tests.one", Identifier("subject"), domain)
        return recorder.trace

    assert run(0, 2).key() != run(2, 0).key()
    assert run(0, 2).key() == run(0, 2).key()


def test_a_key_is_not_equal_to_another_type() -> None:
    """Test a trace key compares unequal to a trace, a tuple and a configuration key."""
    trace = build_dynamic_trace()
    key = trace.key()
    configuration_key = build_complete_configuration(build_tiling_space()).key()

    assert key != trace
    assert key != trace.coordinates
    assert key != 3
    assert key != configuration_key
    assert configuration_key != key
    assert len({key, configuration_key}) == 2


# ===========================================================================
# Static steps
# ===========================================================================


def test_corresponding_samples_of_relabeled_spaces_have_one_key() -> None:
    """Test one seed over two alpha-renamed spaces gives equal keys, unequal traces."""
    left, right = build_tiling_space(), build_tiling_space()

    _, left_trace = left.space.sample(RandomOracle(seed=21))
    _, right_trace = right.space.sample(RandomOracle(seed=21))

    assert left_trace != right_trace
    assert left_trace.key() == right_trace.key()
    assert hash(left_trace.key()) == hash(right_trace.key())
    assert len(left_trace.key()) == len(left_trace)


def test_a_replayed_trace_keeps_its_key() -> None:
    """Test the trace of a replay over a relabeled space has the original's key."""
    recorded, relabeled = build_tiling_space(), build_tiling_space()
    configuration, trace = recorded.space.sample(RandomOracle(seed=8))

    replayed = relabeled.space.replay(trace)

    assert replayed.trace().key() == configuration.trace().key()


def test_samples_that_chose_otherwise_have_different_keys() -> None:
    """Test configurations that differ in a static answer have different trace keys."""
    tiling = build_tiling_space()
    low = build_complete_configuration(tiling, tile=4)
    high = build_complete_configuration(tiling, tile=8)

    assert low.trace().key() != high.trace().key()
    assert (
        low.trace().key() == build_complete_configuration(tiling, tile=4).trace().key()
    )


# ===========================================================================
# Value semantics
# ===========================================================================


def test_trace_key_has_no_constructor() -> None:
    """Test a key is only made by `Trace.key`."""
    with pytest.raises(TypeError):
        TraceKey()


def test_a_key_is_frozen() -> None:
    """Test no attribute can be set or deleted."""
    key: Any = build_dynamic_trace().key()

    with pytest.raises(AttributeError):
        key.extra = 1
    with pytest.raises(AttributeError):
        key.coordinates = ()


def test_a_key_repr_names_its_class() -> None:
    """Test a key's `repr` names `TraceKey`."""
    assert repr(build_dynamic_trace().key()).startswith("TraceKey(")


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_trace_key_pickles_to_an_equal_key(protocol: int) -> None:
    """Test a key pickles, under every protocol, to an equal key of one hash."""
    key = build_dynamic_trace().key()

    restored = pickle.loads(pickle.dumps(key, protocol=protocol))

    assert type(restored) is TraceKey
    assert restored is not key
    assert restored == key
    assert hash(restored) == hash(key)
    assert restored.coordinates == key.coordinates


def test_a_pickled_key_keeps_telling_runs_apart() -> None:
    """Test a restored key equals its own run's key and no other."""
    key = build_dynamic_trace(option=1).key()
    other = build_dynamic_trace(option=2).key()

    restored = pickle.loads(pickle.dumps(key))

    assert restored == build_dynamic_trace(Identifier("fresh")).key()
    assert restored != other
    assert {restored: "measured"}[key] == "measured"


def test_a_trace_key_wire_text_is_not_a_configuration_key() -> None:
    """Test the wire text of a trace key is refused as a configuration key."""
    key: Any = build_dynamic_trace().key()
    _, (text,) = key.__reduce__()

    with pytest.raises(DeserializationValueError):
        ConfigurationKey._from_wire(text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param('{"entries": []}', id="configuration_key_shape"),
        pytest.param('{"steps": "none"}', id="steps_not_a_list"),
        pytest.param("not json", id="not_json"),
    ],
)
def test_a_trace_key_refuses_a_wire_text_of_another_shape(text: str) -> None:
    """Test `_from_wire` raises `DeserializationValueError` for another shape."""
    with pytest.raises(DeserializationValueError):
        TraceKey._from_wire(text)
