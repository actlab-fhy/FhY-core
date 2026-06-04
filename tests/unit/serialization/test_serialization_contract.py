"""Tests of the reusable serialization test contract.

The contract helpers collapse the per-class round-trip / structure-rejection
battery -- previously copy-pasted for every ``Serializable`` -- into a single
parametrized form: a migrated class contributes only representative instances
and a few malformed dicts, not four near-identical test functions.
"""

import pytest

from fhy_core.provenance import Position, Span
from fhy_core.serialization import (
    DeserializationDictStructureError,
    Serializable,
    SerializedDict,
)

from .conftest import (
    assert_dict_round_trip,
    assert_rejects_malformed_dict,
    assert_round_trips_in_all_formats,
)


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(Position(2, 8), id="position"),
        pytest.param(Span(0, 3, Position(1, 1), Position(1, 4)), id="span_full"),
        pytest.param(Span(), id="span_unknown"),
    ],
)
def test_serializable_round_trips_in_all_formats(instance: Serializable) -> None:
    """Test representative serializables round-trip through every format."""
    assert_round_trips_in_all_formats(instance)


@pytest.mark.parametrize(
    "instance",
    [
        pytest.param(Position(5, 9), id="position"),
        pytest.param(Span(start_offset=0, end_offset=10), id="span"),
    ],
)
def test_serializable_dict_round_trip(instance: Serializable) -> None:
    """Test representative serializables survive a dict round trip."""
    assert_dict_round_trip(instance)


@pytest.mark.parametrize(
    "cls,data",
    [
        pytest.param(Position, {"line": 1}, id="position_missing_key"),
        pytest.param(Position, {"line": True, "column": 1}, id="position_bool"),
        pytest.param(Position, {"line": 1, "column": 2, "z": 3}, id="position_extra"),
        pytest.param(Span, {"start_offset": 0}, id="span_missing_keys"),
    ],
)
def test_serializable_rejects_malformed_dict(
    cls: type[Serializable], data: SerializedDict
) -> None:
    """Test malformed dicts are rejected with a structure error."""
    assert_rejects_malformed_dict(
        cls, data, expected_error=DeserializationDictStructureError
    )
