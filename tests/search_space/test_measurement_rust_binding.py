"""Tests of what the binding adds around the Rust objectives and measurements.

The class structure and freezing, the public classes registered under their
type ids, the V2 payloads and their refusals, pickling, and `repr`.
"""

import json
import math
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import Note
from fhy_core.search_space import (
    ConfigurationKey,
    Direction,
    Measurement,
    MeasurementStatus,
    Objective,
    TraceKey,
)
from fhy_core.serialization import (
    DeserializationValueError,
    MalformedPayloadError,
    Serializable,
    SerializationError,
    deserialize_value,
    serialize_value,
)
from fhy_core.traits import FrozenMixin, FrozenMutationError
from tests.v1 import writing_v1

from .conftest import (
    build_complete_configuration,
    build_dynamic_trace,
    build_tiling_space,
)


def _latency() -> Objective:
    """Return the minimized objective `latency_cycles`."""
    return Objective("latency_cycles", Direction.MINIMIZE)


def _throughput() -> Objective:
    """Return the maximized objective `throughput`."""
    return Objective("throughput", Direction.MAXIMIZE)


def _key() -> ConfigurationKey:
    """Return the key of the tiling space's complete configuration."""
    return build_complete_configuration(build_tiling_space()).key()


def _measurements() -> list[Measurement]:
    """Return a measurement of each status, one with notes."""
    key = _key()
    return [
        Measurement.ok(key, [(_latency(), 1532.0), (_throughput(), 0.5)]).with_notes(
            [Note("warm cache")]
        ),
        Measurement.infeasible(key, "rejected by validation"),
        Measurement.failed(key, "the simulator crashed"),
        Measurement.timeout(key),
    ]


_IDS = ["ok", "infeasible", "failed", "timeout"]


# ===========================================================================
# Class structure
# ===========================================================================


@pytest.mark.parametrize(
    ("public", "base", "type_id"),
    [
        (Objective, _rs.Objective, "search_space.objective"),
        (Measurement, _rs.Measurement, "search_space.measurement"),
    ],
    ids=["objective", "measurement"],
)
def test_public_classes_are_thin_registered_subclasses(
    public: type, base: type, type_id: str
) -> None:
    """Test each public class subclasses its `_rs` class and serializes by type id."""
    assert issubclass(public, base)
    assert issubclass(public, Serializable)
    assert issubclass(public, FrozenMixin)
    assert public is not base
    assert public.get_serialization_class_type_id() == type_id


def test_the_binding_builds_the_public_classes() -> None:
    """Test values and decoded objects are the public classes, not the bases."""
    measurement = _measurements()[0]

    assert type(measurement) is Measurement
    assert all(type(objective) is Objective for objective in measurement.values)
    assert type(Objective.from_json(_latency().to_json())) is Objective
    assert type(Measurement.from_json(measurement.to_json())) is Measurement


def test_the_status_and_key_are_the_public_types() -> None:
    """Test `status` is a `MeasurementStatus` member and `key` a `ConfigurationKey`."""
    measurement = _measurements()[1]

    assert measurement.status is MeasurementStatus.INFEASIBLE
    assert type(measurement.key) is ConfigurationKey


def test_the_key_of_a_run_is_a_trace_key() -> None:
    """Test `key` of a measurement built over a `TraceKey` is a `TraceKey`."""
    measurement = _trace_measurements()[1]

    assert type(measurement.key) is TraceKey
    assert measurement.key == _trace_key()


@pytest.mark.parametrize("index", range(5), ids=["objective", *_IDS])
def test_values_are_frozen(index: int) -> None:
    """Test no attribute can be set or deleted."""
    value: Any = [_latency(), *_measurements()][index]

    with pytest.raises(FrozenMutationError):
        value.extra = 1
    with pytest.raises(FrozenMutationError):
        del value.extra


# ===========================================================================
# Payloads
# ===========================================================================


def test_objective_payload_is_its_name_and_direction() -> None:
    """Test an objective's V2 dict."""
    assert _latency().serialize_to_dict() == {
        "name": "latency_cycles",
        "direction": "minimize",
    }


def test_measurement_payload_holds_the_key_status_values_and_notes() -> None:
    """Test a successful measurement's V2 text, the key tagged `configuration`."""
    measurement = _measurements()[0]

    payload = json.loads(measurement.to_json())

    assert payload == {
        "key": {"configuration": json.loads(_key_text())},
        "status": "ok",
        "values": [
            {
                "objective": {"name": "latency_cycles", "direction": "minimize"},
                "value": 1532.0,
            },
            {
                "objective": {"name": "throughput", "direction": "maximize"},
                "value": 0.5,
            },
        ],
        "notes": payload["notes"],
    }
    assert [note["message"] for note in payload["notes"]] == ["warm cache"]


def _key_text() -> str:
    """Return the V2 text of `_key()`, as its pickle writes it."""
    reduced: Any = _key().__reduce__()
    _, (text,) = reduced
    assert isinstance(text, str)
    return text


def _trace_key() -> TraceKey:
    """Return the key of a run of two dynamic steps."""
    return build_dynamic_trace().key()


def _trace_key_text() -> str:
    """Return the V2 text of `_trace_key()`, as its pickle writes it."""
    reduced: Any = _trace_key().__reduce__()
    _, (text,) = reduced
    assert isinstance(text, str)
    return text


def _trace_measurements() -> list[Measurement]:
    """Return a measurement of each status over a trace key, one with notes."""
    key = _trace_key()
    return [
        Measurement.ok(key, [(_latency(), 1532.0), (_throughput(), 0.5)]).with_notes(
            [Note("warm cache")]
        ),
        Measurement.infeasible(key, "rejected by validation"),
        Measurement.failed(key, "the simulator crashed"),
        Measurement.timeout(key),
    ]


def test_a_trace_key_measurement_writes_its_key_tagged_trace() -> None:
    """Test a measurement of a run writes `"key": {"trace": <the key>}`."""
    measurement = _trace_measurements()[0]

    payload = measurement.serialize_to_dict()

    assert payload["key"] == {"trace": json.loads(_trace_key_text())}
    assert json.loads(measurement.to_json())["key"] == {
        "trace": json.loads(_trace_key_text())
    }


def test_the_trace_key_payload_lists_its_steps() -> None:
    """Test the wire form of a trace key is its steps, each with four parts."""
    steps = json.loads(_trace_key_text())["steps"]

    assert len(steps) == 2
    for step in steps:
        assert set(step) == {"kind", "decision", "domain", "coordinate"}
    assert [step["coordinate"] for step in steps] == [{"index": 1}, {"index": 5}]
    assert [step["decision"] for step in steps] == [None, None]


@pytest.mark.parametrize(
    "round_trip",
    [
        lambda measurement: Measurement.deserialize_from_dict(
            measurement.serialize_to_dict()
        ),
        lambda measurement: Measurement.from_json(measurement.to_json()),
    ],
    ids=["dict", "text"],
)
@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_a_trace_key_measurement_round_trips_through_its_payloads(
    index: int, round_trip: Callable[[Measurement], Measurement]
) -> None:
    """Test the V2 dict and text of a run's measurement decode to its trace key."""
    measurement = _trace_measurements()[index]

    decoded = round_trip(measurement)

    assert type(decoded) is Measurement
    assert type(decoded.key) is TraceKey
    assert decoded.key == measurement.key
    assert hash(decoded.key) == hash(measurement.key)
    assert decoded.serialize_to_dict() == measurement.serialize_to_dict()
    assert decoded.values == measurement.values


def test_a_trace_key_measurement_round_trips_through_the_registry() -> None:
    """Test `deserialize_value` finds a run's measurement and its trace key."""
    measurement = _trace_measurements()[0]

    decoded: Any = deserialize_value(serialize_value(measurement))

    assert type(decoded.key) is TraceKey
    assert decoded.key == measurement.key


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_a_trace_key_measurement_pickles_with_its_key(
    index: int, protocol: int
) -> None:
    """Test a run's measurement pickles to one with the same trace key."""
    measurement = _trace_measurements()[index]

    restored = pickle.loads(pickle.dumps(measurement, protocol=protocol))

    assert type(restored.key) is TraceKey
    assert restored.key == measurement.key
    assert restored.serialize_to_dict() == measurement.serialize_to_dict()


@pytest.mark.parametrize(
    "key",
    [
        {"trace": {"steps": "none"}},
        {"trace": {"entries": []}},
        {"entries": []},
        {"configuration": {"steps": []}},
        {"other": {}},
        {"configuration": {"entries": []}, "trace": {"steps": []}},
    ],
    ids=[
        "steps_not_a_list",
        "configuration_shape_under_trace",
        "untagged_configuration_key",
        "trace_shape_under_configuration",
        "unknown_tag",
        "two_tags",
    ],
)
def test_measurement_decoding_refuses_a_malformed_key(key: Any) -> None:
    """Test a key that is not one tagged key is a `DeserializationValueError`."""
    data = _payload_with(key=key)

    with pytest.raises(DeserializationValueError):
        Measurement.deserialize_from_dict(data)


@pytest.mark.parametrize(
    ("index", "status"),
    [
        (1, {"infeasible": {"reason": "rejected by validation"}}),
        (2, {"failed": {"reason": "the simulator crashed"}}),
        (3, "timeout"),
    ],
    ids=_IDS[1:],
)
def test_a_failing_status_is_written_tagged_with_no_values(
    index: int, status: Any
) -> None:
    """Test a failing measurement's status shape and its empty values."""
    payload = _measurements()[index].serialize_to_dict()

    assert payload["status"] == status
    assert payload["values"] == []


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_measurement_round_trips_through_its_payloads(index: int) -> None:
    """Test the V2 dict and text decode to a measurement writing the same payload."""
    measurement = _measurements()[index]

    from_dict = Measurement.deserialize_from_dict(measurement.serialize_to_dict())
    from_text = Measurement.from_json(measurement.to_json())

    for decoded in (from_dict, from_text):
        assert decoded.serialize_to_dict() == measurement.serialize_to_dict()
        assert decoded.key == measurement.key
        assert decoded.values == measurement.values


@pytest.mark.parametrize(
    "build",
    [_latency, lambda: _measurements()[0]],
    ids=["objective", "measurement"],
)
def test_values_round_trip_through_the_serialization_registry(
    build: Callable[[], Any],
) -> None:
    """Test `deserialize_value` finds each class by its type id."""
    value = build()
    decoded: Any = deserialize_value(serialize_value(value))

    assert type(decoded) is type(value)
    assert decoded.serialize_to_dict() == value.serialize_to_dict()


def test_writing_v1_is_refused() -> None:
    """Test the classes have no V1 form."""
    with writing_v1(), pytest.raises(SerializationError, match="V1"):
        _latency().serialize_to_dict()
    with writing_v1(), pytest.raises(SerializationError, match="V1"):
        _measurements()[3].serialize_to_dict()


@pytest.mark.parametrize(
    "data",
    [
        {"name": "", "direction": "minimize"},
        {"name": "n", "direction": "lower"},
        {"name": "n"},
    ],
    ids=["empty_name", "unknown_direction", "no_direction"],
)
def test_objective_decoding_refuses_a_malformed_payload(data: Any) -> None:
    """Test an objective payload outside the rules is a `DeserializationValueError`."""
    with pytest.raises(DeserializationValueError):
        Objective.deserialize_from_dict(data)


def _payload_with(**changes: Any) -> dict[str, Any]:
    """Return the successful measurement's payload with `changes` applied."""
    payload = _measurements()[0].serialize_to_dict()
    payload.update(changes)
    return payload


@pytest.mark.parametrize(
    "changes",
    [
        {"values": []},
        {"status": {"failed": {"reason": "r"}}},
        {
            "values": [
                {"objective": {"name": "a", "direction": "minimize"}, "value": 1.0},
                {"objective": {"name": "a", "direction": "minimize"}, "value": 2.0},
            ]
        },
        {
            "values": [
                {"objective": {"name": "a", "direction": "minimize"}, "value": math.nan}
            ]
        },
        {"status": "crashed"},
    ],
    ids=["no_values", "values_on_failure", "repeated", "nan", "unknown_status"],
)
def test_measurement_decoding_refuses_a_malformed_payload(
    changes: dict[str, Any],
) -> None:
    """Test a measurement payload outside the rules is a `DeserializationValueError`."""
    data = _payload_with(**changes)

    with pytest.raises(DeserializationValueError):
        Measurement.deserialize_from_dict(data)


def test_measurement_decoding_refuses_text_that_is_no_json() -> None:
    """Test text that is no JSON is a `MalformedPayloadError`."""
    with pytest.raises(MalformedPayloadError):
        Measurement.from_json("{")


# ===========================================================================
# Pickling
# ===========================================================================


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_objective_pickles_to_an_equal_objective(protocol: int) -> None:
    """Test an objective pickles, under every protocol, to an equal objective."""
    restored = pickle.loads(pickle.dumps(_latency(), protocol=protocol))

    assert type(restored) is Objective
    assert restored == _latency()
    assert hash(restored) == hash(_latency())


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_measurement_pickles_to_a_measurement_of_the_same_payload(
    index: int, protocol: int
) -> None:
    """Test a measurement pickles to one with the same key, status and values."""
    measurement = _measurements()[index]

    restored = pickle.loads(pickle.dumps(measurement, protocol=protocol))

    assert type(restored) is Measurement
    assert restored.key == measurement.key
    assert restored.status is measurement.status
    assert restored.values == measurement.values
    assert restored.serialize_to_dict() == measurement.serialize_to_dict()


# ===========================================================================
# repr
# ===========================================================================


@pytest.mark.parametrize("index", range(4), ids=_IDS)
def test_measurement_repr_names_its_status(index: int) -> None:
    """Test a measurement's `repr` names the class and its status."""
    measurement = _measurements()[index]

    text = repr(measurement)

    assert "Measurement" in text
    assert str(measurement.status) in text
