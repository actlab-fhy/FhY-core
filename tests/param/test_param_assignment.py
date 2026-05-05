"""Tests for `ParamAssignment` construction and serialization."""

from typing import Any

import pytest

from fhy_core.param import (
    IntParam,
    ParamAssignment,
    PermParam,
    RealParam,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializationFormat,
    serialize_registry_wrapped_value,
)

# =============================================================================
# Construction & accessors
# =============================================================================


def test_real_param_is_unset_until_assignment(default_real_param: RealParam) -> None:
    """Test a `RealParam` exposes no value until an assignment is created."""
    assignment = default_real_param.assign(1.0)
    assert isinstance(default_real_param, RealParam)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.param is default_real_param
    assert not hasattr(default_real_param, "is_value_set")


def test_assignment_is_value_set_after_assign(default_real_param: RealParam) -> None:
    """Test `ParamAssignment.is_value_set` returns ``True`` after assignment."""
    assignment = default_real_param.assign(1.0)
    assert assignment.is_value_set()


def test_param_no_longer_exposes_get_value_attribute(
    default_real_param: RealParam,
) -> None:
    """Test `Param` no longer exposes a direct `get_value` attribute."""
    with pytest.raises(AttributeError):
        default_real_param.get_value()  # type: ignore[attr-defined]  # test: removed


def test_assignment_value_property_returns_assigned_value(
    default_real_param: RealParam,
) -> None:
    """Test `ParamAssignment.value` returns the value handed to `assign`."""
    assignment = default_real_param.assign(1.0)
    assert assignment.value == 1.0


def test_real_param_with_value_creates_initialized_assignment() -> None:
    """Test `RealParam.with_value` returns a value-set assignment."""
    param = RealParam.with_value(1.0)
    assert param.is_value_set()
    assert param.value == 1.0


def test_real_param_with_value_rejects_invalid_value() -> None:
    """Test `RealParam.with_value` raises for an invalid value."""
    with pytest.raises(ValueError):
        RealParam.with_value("invalid")


def test_int_param_with_value_creates_initialized_assignment() -> None:
    """Test `IntParam.with_value` returns a value-set assignment."""
    param = IntParam.with_value(1)
    assert param.is_value_set()
    assert param.value == 1


def test_int_param_with_value_rejects_invalid_value() -> None:
    """Test `IntParam.with_value` raises for an invalid value."""
    with pytest.raises(ValueError):
        IntParam.with_value(1.2)  # type: ignore[arg-type]  # test: invalid input


def test_param_assign_creates_immutable_assignment() -> None:
    """Test `Param.assign` returns an immutable `ParamAssignment`."""
    param = IntParam.with_lower_bound(0)
    assignment = param.assign(3)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.value == 3
    assert assignment.param is param


def test_assignment_materialize_returns_underlying_param() -> None:
    """Test `ParamAssignment.materialize` returns the underlying parameter."""
    assignment = RealParam.with_upper_bound(2.0).assign(1.5)
    bound_param = assignment.materialize()
    assert bound_param is assignment.param


def test_repeated_assigns_share_param_definition_and_record_value(
    default_real_param: RealParam,
) -> None:
    """Test repeated `assign` calls share the param definition and record values."""
    assignment_1 = default_real_param.assign(1.0)
    assignment_2 = default_real_param.assign(1.0)
    assert assignment_1.value == 1.0
    assert assignment_2.value == 1.0
    assert assignment_1.param is default_real_param
    assert assignment_2.param is default_real_param


# =============================================================================
# Serialization round-trip
# =============================================================================


def test_assignment_round_trips_through_serialize_to_dict() -> None:
    """Test `ParamAssignment` round-trips through `serialize_to_dict`."""
    assignment = IntParam.with_lower_bound(0).assign(3)
    dictionary = assignment.serialize_to_dict()

    restored: ParamAssignment[Any] = ParamAssignment.deserialize_from_dict(dictionary)
    assert restored.value == assignment.value
    assert restored.param.is_structurally_equivalent(assignment.param)
    assert restored.serialize_to_dict() == dictionary


def test_assignment_round_trips_through_json_and_binary_serialization() -> None:
    """Test `ParamAssignment` round-trips through JSON and binary serialization."""
    assignment = PermParam(["n", "c", "h", "w"]).assign(["n", "c", "h", "w"])

    json_payload = assignment.serialize(SerializationFormat.JSON)
    from_json: ParamAssignment[Any] = ParamAssignment.deserialize(
        json_payload, SerializationFormat.JSON
    )
    assert isinstance(from_json.param, PermParam)
    assert from_json.value == ("n", "c", "h", "w")

    binary_payload = assignment.serialize(SerializationFormat.BINARY)
    from_binary: ParamAssignment[Any] = ParamAssignment.deserialize(
        binary_payload, SerializationFormat.BINARY
    )
    assert isinstance(from_binary.param, PermParam)
    assert from_binary.value == ("n", "c", "h", "w")


def test_assignment_deserialize_rejects_value_invalid_for_param() -> None:
    """Test assignment deserialization fails when payload value violates constraints."""
    param = RealParam.with_lower_bound(0.0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": serialize_registry_wrapped_value(-1.0),
    }
    with pytest.raises(DeserializationValueError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


# =============================================================================
# Serialization — exception wrapping
# =============================================================================


def test_assignment_deserialize_wraps_value_field_structure_error_as_value_error() -> (
    None
):
    """Test malformed wrapped ``value`` surfaces as `DeserializationValueError`.

    The wrapped registry raises `DeserializationDictStructureError` for a
    payload missing ``__type__``/``__data__``; `ParamAssignment.deserialize_from_dict`
    must re-wrap that as `DeserializationValueError` so callers see a single
    failure type for value-side problems.
    """
    param = IntParam.with_lower_bound(0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": {"not": "a wrapped value"},
    }
    with pytest.raises(DeserializationValueError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_wraps_value_field_value_error_as_value_error() -> None:
    """Test a wrapped-value validation failure surfaces as `DeserializationValueError`.

    A wrapped tuple payload whose ``__data__`` items are not serialized dicts
    causes `deserialize_registry_wrapped_value` to raise
    `DeserializationValueError`; `ParamAssignment.deserialize_from_dict` must
    surface that under the same error type rather than letting it propagate
    raw.
    """
    param = IntParam.with_lower_bound(0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": {"__type__": "builtins.tuple", "__data__": [42]},
    }
    with pytest.raises(DeserializationValueError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_missing_param_field() -> None:
    """Test a payload missing the ``param`` field is rejected as malformed."""
    payload = {"value": serialize_registry_wrapped_value(1)}
    with pytest.raises(DeserializationDictStructureError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_missing_value_field() -> None:
    """Test a payload missing the ``value`` field is rejected as malformed."""
    payload = {"param": IntParam().serialize_to_dict()}
    with pytest.raises(DeserializationDictStructureError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_with_param_not_serialized_dict() -> (
    None
):
    """Test a payload whose ``param`` is not a serialized dict is rejected."""
    payload = {"param": "not-a-dict", "value": serialize_registry_wrapped_value(1)}
    with pytest.raises(DeserializationDictStructureError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_with_value_not_serialized_dict() -> (
    None
):
    """Test a payload whose ``value`` is not a serialized dict is rejected."""
    payload = {"param": IntParam().serialize_to_dict(), "value": 42}
    with pytest.raises(DeserializationDictStructureError):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape
