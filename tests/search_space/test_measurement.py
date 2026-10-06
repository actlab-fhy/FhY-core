"""The interface suite of objectives and measurements.

`Direction` and `MeasurementStatus`, `Objective` and its `compare` (NaN
included, F-SS-023), the constructors of `Measurement` and every refusal,
its values, notes and `dominates`, and the `Measurer` protocol. The tests
ported from MOGA-VM's `tests/cir/lowering/search/test_records.py` carry the
name of the test they port in their docstrings.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from fhy_core.diagnostic import Note
from fhy_core.identifier import Identifier
from fhy_core.search_space import (
    Configuration,
    ConfigurationKey,
    Direction,
    Measurement,
    MeasurementError,
    MeasurementStatus,
    Measurer,
    Objective,
    SearchSpaceError,
)

from .conftest import build_complete_configuration, build_tiling_space


def _latency() -> Objective:
    """Return the minimized objective `latency`."""
    return Objective("latency", Direction.MINIMIZE)


def _throughput() -> Objective:
    """Return the maximized objective `throughput`."""
    return Objective("throughput", Direction.MAXIMIZE)


def _power() -> Objective:
    """Return the reported objective `power`."""
    return Objective("power", Direction.REPORT)


def _key(tile: int = 4) -> ConfigurationKey:
    """Return the key of the tiling space's complete configuration with `tile`."""
    return build_complete_configuration(build_tiling_space(), tile).key()


def _measure(latency: float, throughput: float, power: float = 0.0) -> Measurement:
    """Return a successful measurement of the three objectives."""
    return Measurement.ok(
        _key(), {_latency(): latency, _throughput(): throughput, _power(): power}
    )


def _running_best(objective: Objective, values: Sequence[float]) -> float:
    """Return the value a running best keeps, taking a value only when better."""
    best = values[0]
    for value in values[1:]:
        if objective.compare(value, best) == 1:
            best = value
    return best


# ===========================================================================
# Direction and MeasurementStatus
# ===========================================================================


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (Direction.MINIMIZE, "minimize"),
        (Direction.MAXIMIZE, "maximize"),
        (Direction.REPORT, "report"),
        (MeasurementStatus.OK, "ok"),
        (MeasurementStatus.INFEASIBLE, "infeasible"),
        (MeasurementStatus.FAILED, "failed"),
        (MeasurementStatus.TIMEOUT, "timeout"),
    ],
)
def test_enum_members_are_their_text(member: str, value: str) -> None:
    """Test each `Direction` and `MeasurementStatus` member equals its text."""
    assert member == value


def test_measurement_error_is_a_search_space_error() -> None:
    """Test `MeasurementError` derives from `SearchSpaceError`."""
    assert issubclass(MeasurementError, SearchSpaceError)


# ===========================================================================
# Objective
# ===========================================================================


def test_objective_keeps_its_name_and_direction() -> None:
    """Test an objective's name and its `Direction` member."""
    objective = Objective("latency_cycles", Direction.MINIMIZE)

    assert objective.name == "latency_cycles"
    assert objective.direction is Direction.MINIMIZE


def test_objective_takes_a_direction_by_its_text() -> None:
    """Test a direction given as its value reads as the member."""
    objective = Objective("throughput", "maximize")

    assert objective.direction is Direction.MAXIMIZE


def test_objective_refuses_an_empty_name() -> None:
    """Test an empty name raises `MeasurementError`."""
    with pytest.raises(MeasurementError, match="an objective needs a name"):
        Objective("", Direction.MINIMIZE)


def test_objective_refuses_an_unknown_direction() -> None:
    """Test a string naming no direction raises `ValueError`."""
    with pytest.raises(ValueError):
        Objective("latency", "lower")


@pytest.mark.parametrize(
    ("name", "direction"),
    [(3, Direction.MINIMIZE), ("latency", 1), ("latency", None)],
    ids=["name", "direction", "no_direction"],
)
def test_objective_refuses_arguments_of_the_wrong_type(
    name: Any, direction: Any
) -> None:
    """Test a name that is no `str` or a direction that is no text is a `TypeError`."""
    with pytest.raises(TypeError):
        Objective(name, direction)


def test_objectives_compare_and_hash_structurally() -> None:
    """Test objectives with one name and direction are equal, and key a dict alike."""
    left, right = Objective("bytes", "minimize"), Objective("bytes", "minimize")

    assert left == right
    assert hash(left) == hash(right)
    assert {left: 1}[right] == 1
    assert left != Objective("bytes", "maximize")
    assert left != Objective("bytes_moved", "minimize")
    assert left != "bytes"


def test_objective_repr_names_its_parts() -> None:
    """Test `repr` names the class, the name and the direction."""
    text = repr(Objective("latency", Direction.MINIMIZE))

    assert "Objective" in text
    assert "latency" in text
    assert "minimize" in text


def test_minimize_prefers_the_lower_value() -> None:
    """Test a minimized objective ranks the lower value better.

    Ports MOGA-VM
    test_records.py::test_best_returns_the_lowest_scored_feasible_record_with_earliest_tie_break
    (the comparison only).
    """
    assert _latency().compare(1.0, 2.0) == 1
    assert _latency().compare(2.0, 1.0) == -1
    assert _latency().compare(2.0, 2.0) == 0


def test_maximize_prefers_the_higher_value() -> None:
    """Test a maximized objective ranks the higher value better, `int`s included."""
    assert _throughput().compare(2, 1) == 1
    assert _throughput().compare(1.0, 2) == -1
    assert _throughput().compare(0.0, -0.0) == 0


def test_report_is_never_compared() -> None:
    """Test a reported objective's comparison is `None`."""
    assert _power().compare(1.0, 2.0) is None
    assert _power().compare(math.nan, 2.0) is None


@pytest.mark.parametrize("build", [_latency, _throughput], ids=["min", "max"])
@pytest.mark.parametrize("number", [0.0, -1e308, 1e308, math.inf, -math.inf])
def test_a_nan_loses_to_every_number(
    build: Callable[[], Objective], number: float
) -> None:
    """Test a NaN is worse than any number in either direction (F-SS-023)."""
    objective = build()

    assert objective.compare(math.nan, number) == -1
    assert objective.compare(number, math.nan) == 1


def test_two_nans_tie() -> None:
    """Test two NaNs compare equal."""
    assert _latency().compare(math.nan, math.nan) == 0


@pytest.mark.parametrize(
    ("build", "values", "best"),
    [
        (_latency, [math.nan, 3.0, 1.0, 2.0], 1.0),
        (_latency, [4.0, math.nan, 2.0], 2.0),
        (_throughput, [math.nan, math.nan, 1.0, 3.0], 3.0),
    ],
    ids=["min_after_nan", "min_between", "max_after_nans"],
)
def test_a_nan_never_blocks_a_later_best(
    build: Callable[[], Objective], values: list[float], best: float
) -> None:
    """Test a running best is not stuck on a NaN that came first (F-SS-023)."""
    assert _running_best(build(), values) == best


@pytest.mark.parametrize("value", [True, "1.0", None, [1.0]])
def test_compare_refuses_a_value_that_is_no_number(value: Any) -> None:
    """Test a `bool`, a string, `None` or a list is no value: `TypeError`."""
    with pytest.raises(TypeError):
        _latency().compare(value, 1.0)


# ===========================================================================
# Measurement construction
# ===========================================================================


def test_an_ok_measurement_keeps_its_values() -> None:
    """Test a successful measurement keeps its key, status and values in order.

    Ports MOGA-VM test_records.py::test_a_score_on_a_lowered_record_is_valid.
    """
    key = _key()

    measurement = Measurement.ok(key, [(_throughput(), 7.5), (_latency(), 1532)])

    assert measurement.key == key
    assert measurement.status is MeasurementStatus.OK
    assert measurement.reason is None
    assert measurement.is_ok()
    assert list(measurement.values.items()) == [
        (_throughput(), 7.5),
        (_latency(), 1532.0),
    ]
    assert type(measurement.values[_latency()]) is float
    assert measurement.value(_latency()) == 1532.0
    assert measurement.value("throughput") == 7.5
    assert measurement.value("area") is None
    assert measurement.notes == ()


def test_an_ok_measurement_takes_a_mapping() -> None:
    """Test values given as a mapping keep its order."""
    measurement = Measurement.ok(_key(), {_latency(): 2.0, _power(): 1.0})

    assert list(measurement.values) == [_latency(), _power()]


@pytest.mark.parametrize(
    ("build", "status", "reason"),
    [
        (
            lambda key: Measurement.infeasible(key, "rejected by placement"),
            MeasurementStatus.INFEASIBLE,
            "rejected by placement",
        ),
        (
            lambda key: Measurement.failed(key, "the simulator crashed"),
            MeasurementStatus.FAILED,
            "the simulator crashed",
        ),
        (Measurement.timeout, MeasurementStatus.TIMEOUT, None),
    ],
    ids=["infeasible", "failed", "timeout"],
)
def test_a_failed_measurement_holds_no_values(
    build: Any, status: MeasurementStatus, reason: str | None
) -> None:
    """Test a measurement that did not succeed has no value.

    Ports MOGA-VM test_records.py::test_a_score_on_an_infeasible_record_is_invalid.
    """
    measurement = build(_key())

    assert measurement.status is status
    assert measurement.reason == reason
    assert not measurement.is_ok()
    assert measurement.values == {}
    assert measurement.value(_latency()) is None


def test_an_ok_measurement_refuses_no_values() -> None:
    """Test a successful measurement needs a value."""
    with pytest.raises(MeasurementError, match="at least one value"):
        Measurement.ok(_key(), {})


def test_an_ok_measurement_refuses_a_repeated_objective() -> None:
    """Test two values for one objective name, in any direction, are refused."""
    with pytest.raises(MeasurementError, match='"latency" has two values'):
        Measurement.ok(
            _key(), [(_latency(), 1.0), (Objective("latency", "maximize"), 2.0)]
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_an_ok_measurement_refuses_a_non_finite_value(value: float) -> None:
    """Test a NaN or an infinity is refused, naming the objective (F-SS-023)."""
    with pytest.raises(MeasurementError, match='"power" is not finite'):
        Measurement.ok(_key(), [(_latency(), 1.0), (_power(), value)])


@pytest.mark.parametrize("value", [True, "1.0", None, 1j])
def test_an_ok_measurement_refuses_a_value_that_is_no_number(value: Any) -> None:
    """Test a `bool` or a non-number is no value: `TypeError`."""
    with pytest.raises(TypeError):
        Measurement.ok(_key(), {_latency(): value})


@pytest.mark.parametrize(
    "values",
    [
        lambda: {"latency": 1.0},
        lambda: [(_latency(),)],
        lambda: [_latency()],
        lambda: 3,
    ],
    ids=["name_key", "short_pair", "bare_objective", "not_iterable"],
)
def test_an_ok_measurement_refuses_values_of_another_shape(
    values: Callable[[], Any],
) -> None:
    """Test values that are no mapping or `(Objective, number)` pairs: `TypeError`."""
    with pytest.raises(TypeError):
        Measurement.ok(_key(), values())


@pytest.mark.parametrize(
    "build",
    [
        lambda key: Measurement.ok(key, {_latency(): 1.0}),
        lambda key: Measurement.infeasible(key, "r"),
        lambda key: Measurement.failed(key, "r"),
        Measurement.timeout,
    ],
    ids=["ok", "infeasible", "failed", "timeout"],
)
def test_every_constructor_refuses_a_key_that_is_no_configuration_key(
    build: Any,
) -> None:
    """Test a `Configuration` in place of its key is a `TypeError`."""
    configuration: Configuration = build_complete_configuration(build_tiling_space())

    with pytest.raises(TypeError):
        build(configuration)


@pytest.mark.parametrize("reason", [None, 3], ids=["none", "int"])
def test_a_failure_refuses_a_reason_that_is_no_text(reason: Any) -> None:
    """Test a reason that is no `str` is a `TypeError`."""
    with pytest.raises(TypeError):
        Measurement.failed(_key(), reason)


def test_measurement_has_no_constructor() -> None:
    """Test `Measurement(...)` is refused: the class methods build one."""
    with pytest.raises(TypeError):
        Measurement()


def test_a_negative_zero_is_kept_as_zero() -> None:
    """Test `-0.0` is held as `0.0`."""
    measurement = Measurement.ok(_key(), {_latency(): -0.0})

    value = measurement.value(_latency())
    assert value is not None
    assert math.copysign(1.0, value) == 1.0


def test_value_matches_an_objective_exactly_and_a_name_by_name() -> None:
    """Test `value(objective)` needs the same direction; `value(name)` does not."""
    measurement = Measurement.ok(_key(), {_latency(): 3.0})

    assert measurement.value(_latency()) == 3.0
    assert measurement.value(Objective("latency", Direction.MAXIMIZE)) is None
    assert measurement.value("latency") == 3.0


def test_values_is_a_copy() -> None:
    """Test changing the `values` dict leaves the measurement as it was."""
    measurement = Measurement.ok(_key(), {_latency(): 3.0})

    measurement.values[_latency()] = 9.0
    measurement.values.clear()

    assert measurement.values == {_latency(): 3.0}


def test_value_refuses_an_argument_that_is_no_objective_or_name() -> None:
    """Test `value(3)` is a `TypeError`."""
    with pytest.raises(TypeError):
        _measure(1.0, 2.0).value(3)  # type: ignore[arg-type]


# ===========================================================================
# Notes and equality
# ===========================================================================


def test_with_notes_replaces_the_notes() -> None:
    """Test `with_notes` returns a measurement with the notes given, in order."""
    original = Measurement.failed(_key(), "crashed").with_notes([Note("old")])

    replaced = original.with_notes((Note("first"), Note("second")))

    assert [note.message for note in replaced.notes] == ["first", "second"]
    assert [note.message for note in original.notes] == ["old"]
    assert replaced.status is MeasurementStatus.FAILED
    assert type(replaced) is Measurement


def test_with_notes_refuses_what_is_no_note() -> None:
    """Test a note that is no `Note` is a `TypeError`."""
    with pytest.raises(TypeError):
        _measure(1.0, 2.0).with_notes(["text"])  # type: ignore[list-item]


def test_measurements_compare_by_identity() -> None:
    """Test two measurements built alike are unequal: `==` is identity."""
    first, second = _measure(1.0, 2.0), _measure(1.0, 2.0)

    assert first is not second
    assert first != second
    assert first in {first}
    assert len({first, second}) == 2


# ===========================================================================
# Dominance
# ===========================================================================


@pytest.mark.parametrize(
    ("better", "worse"),
    [((5.0, 2.0), (10.0, 2.0)), ((10.0, 3.0), (10.0, 2.0)), ((5.0, 3.0), (10.0, 2.0))],
    ids=["latency", "throughput", "both"],
)
def test_a_better_measurement_dominates(
    better: tuple[float, float], worse: tuple[float, float]
) -> None:
    """Test better on one objective and as good on the others dominates, not back."""
    assert _measure(*better).dominates(_measure(*worse))
    assert not _measure(*worse).dominates(_measure(*better))


def test_a_trade_off_and_ties_do_not_dominate() -> None:
    """Test a trade-off, a tie and a reported difference dominate neither way."""
    pairs = [
        (_measure(5.0, 1.0), _measure(10.0, 2.0)),
        (_measure(5.0, 2.0), _measure(5.0, 2.0)),
        (_measure(5.0, 2.0, 1.0), _measure(5.0, 2.0, 100.0)),
    ]

    for left, right in pairs:
        assert not left.dominates(right)
        assert not right.dominates(left)


def test_dominance_matches_objectives_in_any_order() -> None:
    """Test values given in another order are matched by objective."""
    better = Measurement.ok(_key(), [(_latency(), 5.0), (_throughput(), 2.0)])
    worse = Measurement.ok(_key(8), [(_throughput(), 2.0), (_latency(), 10.0)])

    assert better.dominates(worse)


def test_dominance_refuses_different_objectives() -> None:
    """Test measurements over other objectives are not compared."""
    left = Measurement.ok(_key(), {_latency(): 1.0})
    right = Measurement.ok(_key(), {Objective("latency", "maximize"): 1.0})

    with pytest.raises(MeasurementError, match="different objectives"):
        left.dominates(right)


def test_dominance_refuses_a_measurement_that_did_not_succeed() -> None:
    """Test a failed measurement is not compared, in either position."""
    failed = Measurement.timeout(_key())

    with pytest.raises(MeasurementError, match="only successful measurements"):
        _measure(1.0, 2.0).dominates(failed)
    with pytest.raises(MeasurementError, match="only successful measurements"):
        failed.dominates(_measure(1.0, 2.0))


def test_dominance_refuses_an_argument_that_is_no_measurement() -> None:
    """Test `dominates(3)` is a `TypeError`."""
    with pytest.raises(TypeError):
        _measure(1.0, 2.0).dominates(3)  # type: ignore[arg-type]


# ===========================================================================
# Measurer
# ===========================================================================


class _TileMeasurer:
    """Measures a configuration's latency as its tile; a flat layout is infeasible."""

    def __init__(self, tile: Identifier) -> None:
        self.objectives = (_latency(),)
        self._tile = tile

    def measure(self, key: ConfigurationKey, subject: Any) -> Measurement:
        tile = subject.value(self._tile)
        if tile is None:
            return Measurement.infeasible(key, "the flat layout")
        return Measurement.ok(key, {_latency(): float(tile)})


def test_any_object_with_objectives_and_measure_is_a_measurer() -> None:
    """Test the `Measurer` protocol is structural."""
    assert isinstance(_TileMeasurer(Identifier("tile")), Measurer)
    assert not isinstance(object(), Measurer)


def test_a_measurer_ranks_the_configurations_of_a_space() -> None:
    """Test measuring every configuration ranks the smallest tile best."""
    tiling = build_tiling_space()
    measurer = _TileMeasurer(tiling.tile.name)

    measurements = [
        measurer.measure(configuration.key(), configuration)
        for configuration in tiling.space.enumerate()
    ]

    feasible = [m for m in measurements if m.is_ok()]
    assert len(feasible) == 6
    assert len(measurements) - len(feasible) == 3
    latencies = [m.values[_latency()] for m in feasible]
    assert _running_best(_latency(), latencies) == 4.0
    assert sorted(latencies) == [4.0, 4.0, 4.0, 8.0, 8.0, 8.0]
