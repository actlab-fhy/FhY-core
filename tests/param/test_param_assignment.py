"""Tests for `ParamAssignment` construction and serialization."""

from typing import Any

import pytest

from fhy_core.param import (
    Param,
    ParamAssignment,
    ParamError,
    PermutationDomain,
    RealDomain,
    create_integer_param,
    create_integer_param_with_lower_bound,
    create_permutation_param,
    create_real_param,
    create_real_param_with_lower_bound,
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


def test_real_param_is_unset_until_assignment(
    default_real_param: Param[str | float],
) -> None:
    """Test a real param exposes no value until an assignment is created."""
    assignment = default_real_param.assign(1.0)

    assert isinstance(default_real_param.domain, RealDomain)
    assert isinstance(assignment, ParamAssignment)
    assert assignment.param is default_real_param
    assert not hasattr(default_real_param, "is_value_set")


def test_assignment_is_value_set_after_assign(
    default_real_param: Param[str | float],
) -> None:
    """Test `ParamAssignment.is_value_set` returns ``True`` after assignment."""
    assignment = default_real_param.assign(1.0)

    assert assignment.is_value_set()


def test_param_no_longer_exposes_get_value_attribute(
    default_real_param: Param[str | float],
) -> None:
    """Test `Param` no longer exposes a direct `get_value` attribute."""
    with pytest.raises(AttributeError, match="get_value"):
        default_real_param.get_value()  # type: ignore[attr-defined]  # test: removed


def test_assignment_value_property_returns_assigned_value(
    default_real_param: Param[str | float],
) -> None:
    """Test `ParamAssignment.value` returns the value handed to `assign`."""
    assignment = default_real_param.assign(1.0)

    assert assignment.value == 1.0


def test_real_param_with_value_creates_initialized_assignment() -> None:
    """Test `create_real_param().assign(v)` returns a value-set assignment."""
    param = create_real_param().assign(1.0)

    assert param.is_value_set()
    assert param.value == 1.0


def test_real_param_with_value_rejects_invalid_value() -> None:
    """Test assigning an invalid value to a real param raises `ParamError`."""
    with pytest.raises(ParamError, match="is not admissible"):
        create_real_param().assign("invalid")


def test_int_param_with_value_creates_initialized_assignment() -> None:
    """Test `create_integer_param().assign(v)` returns a value-set assignment."""
    param = create_integer_param().assign(1)

    assert param.is_value_set()
    assert param.value == 1


def test_int_param_with_value_rejects_invalid_value() -> None:
    """Test assigning a float to an integer param raises `ParamError`."""
    with pytest.raises(ParamError, match="is not admissible"):
        create_integer_param().assign(1.2)  # type: ignore[arg-type]  # test: invalid input


def test_param_assign_creates_immutable_assignment() -> None:
    """Test `Param.assign` returns an immutable `ParamAssignment`."""
    param = create_integer_param_with_lower_bound(0)

    assignment = param.assign(3)

    assert isinstance(assignment, ParamAssignment)
    assert assignment.value == 3
    assert assignment.param is param


def test_repeated_assigns_share_param_definition_and_record_value(
    default_real_param: Param[str | float],
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
    assignment = create_integer_param_with_lower_bound(0).assign(3)
    dictionary = assignment.serialize_to_dict()

    restored: ParamAssignment[Any] = ParamAssignment.deserialize_from_dict(dictionary)
    assert restored.value == assignment.value
    assert restored.param.is_structurally_equivalent(assignment.param)
    assert restored.serialize_to_dict() == dictionary


def test_assignment_round_trips_through_json_and_binary_serialization() -> None:
    """Test `ParamAssignment` round-trips through JSON and binary serialization."""
    assignment = create_permutation_param(["n", "c", "h", "w"]).assign(
        ["n", "c", "h", "w"]  # type: ignore[arg-type]  # test: list as permutation value
    )

    json_payload = assignment.serialize(SerializationFormat.JSON)
    from_json: ParamAssignment[Any] = ParamAssignment.deserialize(
        json_payload, SerializationFormat.JSON
    )
    assert isinstance(from_json.param.domain, PermutationDomain)
    assert from_json.value == ("n", "c", "h", "w")

    binary_payload = assignment.serialize(SerializationFormat.BINARY)
    from_binary: ParamAssignment[Any] = ParamAssignment.deserialize(
        binary_payload, SerializationFormat.BINARY
    )
    assert isinstance(from_binary.param.domain, PermutationDomain)
    assert from_binary.value == ("n", "c", "h", "w")


def test_assignment_deserialize_rejects_value_invalid_for_param() -> None:
    """Test assignment deserialization fails when payload value violates constraints."""
    param = create_real_param_with_lower_bound(0.0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": serialize_registry_wrapped_value(-1.0),
    }

    with pytest.raises(DeserializationValueError, match="violates constraint"):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


# =============================================================================
# Serialization - exception wrapping
# =============================================================================


def test_assignment_deserialize_rejects_value_field_with_wrong_shaped_dict() -> None:
    """Test a wrong-shaped wrapped ``value`` dict is rejected as a structure error.

    The ``value`` codec defers to the wrapped registry, which raises
    `DeserializationDictStructureError` for a dict missing
    ``__type__``/``__data__``. The derived engine surfaces that subtype
    directly (both subtypes are in the serialization error hierarchy).
    """
    param = create_integer_param_with_lower_bound(0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": {"not": "a wrapped value"},
    }

    with pytest.raises(
        DeserializationDictStructureError, match='deserializing to "Serializable"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_wraps_value_field_value_error_as_value_error() -> None:
    """Test a wrapped-value validation failure surfaces as `DeserializationValueError`.

    A wrapped tuple payload whose ``__data__`` items are not serialized dicts
    causes `deserialize_registry_wrapped_value` to raise
    `DeserializationValueError`; `ParamAssignment.deserialize_from_dict` must
    surface that under the same error type rather than letting it propagate
    raw.
    """
    param = create_integer_param_with_lower_bound(0)
    payload = {
        "param": param.serialize_to_dict(),
        "value": {"__type__": "builtins.tuple", "__data__": [42]},
    }

    with pytest.raises(
        DeserializationValueError, match='deserializing to "ParamAssignment"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_missing_param_field() -> None:
    """Test a payload missing the ``param`` field is rejected as malformed."""
    payload = {"value": serialize_registry_wrapped_value(1)}

    with pytest.raises(
        DeserializationDictStructureError, match='deserializing to "ParamAssignment"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_missing_value_field() -> None:
    """Test a payload missing the ``value`` field is rejected as malformed."""
    payload = {"param": create_integer_param().serialize_to_dict()}

    with pytest.raises(
        DeserializationDictStructureError, match='deserializing to "ParamAssignment"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_with_param_not_serialized_dict() -> (
    None
):
    """Test a payload whose ``param`` is not a serialized dict is rejected."""
    payload = {"param": "not-a-dict", "value": serialize_registry_wrapped_value(1)}

    with pytest.raises(
        DeserializationDictStructureError, match='deserializing to "ParamAssignment"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape


def test_assignment_deserialize_rejects_payload_with_value_not_serialized_dict() -> (
    None
):
    """Test a payload whose ``value`` is not a serialized dict is rejected.

    A non-dict ``value`` (here ``42``) makes the wrapped-value decode raise a
    ``TypeError``, which the field codec surfaces as `DeserializationValueError`.
    """
    payload = {"param": create_integer_param().serialize_to_dict(), "value": 42}

    with pytest.raises(
        DeserializationValueError, match='deserializing to "ParamAssignment"'
    ):
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape
