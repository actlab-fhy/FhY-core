"""Serialization round-trips and deserialization-error coverage.

The TypeGuard validators (`_is_valid_*_constraint_data`) are reached
through the public ``deserialize_from_dict`` API, so the parametrized
``rejects_malformed_payload`` tests here drive the long
``and ... and ...`` chain mutations.
"""

from typing import Any, Callable, cast

import pytest

from fhy_core.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    IdentifierExpression,
    LiteralExpression,
)
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializedDict,
)

from .conftest import SerializableEqualHashable, mock_identifier

SetConstraintType = type[Constraint]


# =============================================================================
# Equation constraint round-trip
# =============================================================================


def test_equation_constraint_round_trip_dict_serialization() -> None:
    """Test an `EquationConstraint` round-trips through dict serialization."""
    x = mock_identifier("x", 0)
    expression = BinaryExpression(
        BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
    )
    constraint = EquationConstraint(x, expression)
    expected = {
        "__type__": "equation_constraint",
        "__data__": {
            "variable": x.serialize_to_dict(),
            "expression": expression.serialize_to_dict(),
        },
    }
    assert constraint.serialize_to_dict() == expected
    rebuilt = EquationConstraint.deserialize_from_dict(constraint.serialize_to_dict())
    assert isinstance(rebuilt, EquationConstraint)
    assert rebuilt.variable == x
    assert rebuilt.convert_to_expression().is_structurally_equivalent(expression)


# =============================================================================
# Set constraint round-trips
# =============================================================================


_SET_KINDS = [
    pytest.param(InSetConstraint, "valid_values", id="in_set"),
    pytest.param(NotInSetConstraint, "invalid_values", id="not_in_set"),
]


@pytest.mark.parametrize("factory, _field", _SET_KINDS)
def test_set_constraint_round_trip_dict_serialization(
    factory: SetConstraintType, _field: str
) -> None:
    """Test a set constraint round-trips through dict serialization."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})  # type: ignore[call-arg]
    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())
    assert isinstance(rebuilt, factory)
    assert rebuilt.variable == x
    for member in (1, 2):
        assert rebuilt.is_satisfied(member) == constraint.is_satisfied(member)
    assert rebuilt.is_satisfied(99) == constraint.is_satisfied(99)


@pytest.mark.parametrize("factory, _field", _SET_KINDS)
def test_set_constraint_round_trip_with_serializable_member(
    factory: SetConstraintType, _field: str
) -> None:
    """Test serializable+hashable members survive a round trip."""
    x = mock_identifier("x", 0)
    member = SerializableEqualHashable(7)
    constraint = factory(x, {member})  # type: ignore[call-arg]
    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())
    assert rebuilt.is_satisfied(member) == constraint.is_satisfied(member)
    assert rebuilt.is_satisfied(
        SerializableEqualHashable(8)
    ) == constraint.is_satisfied(SerializableEqualHashable(8))


@pytest.mark.parametrize("factory, _field", _SET_KINDS)
def test_set_constraint_round_trip_with_nested_collection_member(
    factory: SetConstraintType, _field: str
) -> None:
    """Test nested tuple/frozenset members round-trip and stay membership-equivalent."""
    x = mock_identifier("x", 0)
    nested_member = (1, (2, 3), frozenset({4, 5}))
    constraint = factory(x, [nested_member])  # type: ignore[call-arg]
    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())
    assert rebuilt.is_satisfied(nested_member) == constraint.is_satisfied(nested_member)


def test_in_set_constraint_serialized_values_are_repr_sorted() -> None:
    """Test serialized members are emitted in repr-sorted order for determinism."""
    constraint = InSetConstraint(mock_identifier("x", 0), {3, 1, 2})
    payload_data = cast(dict[str, Any], constraint.serialize_to_dict()["__data__"])
    serialized_values: list[Any] = payload_data["valid_values"]
    assert serialized_values == sorted(serialized_values, key=repr)


# =============================================================================
# Structural payload errors
# =============================================================================


def _drop(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda d: {k: v for k, v in d.items() if k != key}


def _replace(key: str, value: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda d: {**d, key: value}


@pytest.fixture
def equation_payload() -> dict[str, Any]:
    x = mock_identifier("x", 0)
    return {
        "variable": x.serialize_to_dict(),
        "expression": LiteralExpression(True).serialize_to_dict(),
    }


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_drop("variable"), id="missing_variable"),
        pytest.param(_replace("variable", 42), id="variable_not_a_dict"),
        pytest.param(_drop("expression"), id="missing_expression"),
        pytest.param(_replace("expression", [1, 2, 3]), id="expression_not_a_dict"),
    ],
)
def test_equation_constraint_rejects_malformed_payload(
    equation_payload: dict[str, Any],
    mutate: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Test malformed `EquationConstraint` payloads raise structure errors."""
    with pytest.raises(DeserializationDictStructureError):
        EquationConstraint.deserialize_data_from_dict(mutate(equation_payload))


@pytest.fixture(
    params=[
        pytest.param((InSetConstraint, "valid_values"), id="in_set"),
        pytest.param((NotInSetConstraint, "invalid_values"), id="not_in_set"),
    ]
)
def set_payload_with_field(
    request: pytest.FixtureRequest,
) -> tuple[type[Constraint], str, dict[str, Any]]:
    """Yield the factory, field name, and serialized payload for each set kind."""
    factory, field = request.param
    constraint = factory(mock_identifier("x", 0), {1, 2})
    return factory, field, constraint.serialize_to_dict()["__data__"]


@pytest.mark.parametrize(
    "mutate_template",
    [
        pytest.param(("drop", "variable"), id="missing_variable"),
        pytest.param(("replace", "variable", "scalar"), id="variable_not_a_dict"),
        pytest.param(("drop", "<field>"), id="missing_values_field"),
        pytest.param(
            ("replace", "<field>", "not-a-list"), id="values_field_not_a_list"
        ),
        pytest.param(("replace", "<field>", [42]), id="values_field_contains_non_dict"),
    ],
)
def test_set_constraint_rejects_malformed_payload(
    set_payload_with_field: tuple[type[Constraint], str, dict[str, Any]],
    mutate_template: tuple[str, ...],
) -> None:
    """Test malformed set-constraint payloads raise structure errors."""
    factory, field, payload = set_payload_with_field
    op, *args = mutate_template
    resolved_args = [field if a == "<field>" else a for a in args]
    if op == "drop":
        bad = _drop(resolved_args[0])(payload)
    else:
        bad = _replace(resolved_args[0], resolved_args[1])(payload)
    with pytest.raises(DeserializationDictStructureError):
        factory.deserialize_data_from_dict(bad)


# =============================================================================
# Member-deserializer error propagation
# =============================================================================


@pytest.mark.parametrize("factory, field", _SET_KINDS)
def test_set_member_deserializer_rewraps_dict_structure_error(
    factory: type[Constraint], field: str
) -> None:
    """Test the member deserializer re-wraps a wrapped-member structure error.

    A wrapped member missing ``__type__`` raises a structure error from
    ``deserialize_registry_wrapped_value``; the constraint deserializer
    must catch it and re-raise as ``DeserializationValueError``.

    The except-clause mutation that swaps ``DeserializationDictStructureError``
    out of the catch tuple lets the inner error escape unchanged.
    """
    x = mock_identifier("x", 0)
    bad_payload: SerializedDict = {
        "variable": x.serialize_to_dict(),
        field: [{"not_a_wrapped": "value"}],
    }
    with pytest.raises(DeserializationValueError):
        factory.deserialize_data_from_dict(bad_payload)


@pytest.mark.parametrize("factory, field", _SET_KINDS)
def test_set_member_deserializer_rewraps_value_error_with_field_name(
    factory: type[Constraint], field: str
) -> None:
    """Test the member deserializer embeds the field name in re-wrapped errors.

    The inner ``DeserializationValueError`` must be re-raised with the
    field name in the message. The mutation that drops
    ``DeserializationValueError`` from the catch tuple lets the inner
    exception bypass the field-name re-wrapping.
    """
    x = mock_identifier("x", 0)
    bad_payload: SerializedDict = {
        "variable": x.serialize_to_dict(),
        field: [
            {
                "__type__": "tests.serializable_equal_hashable",
                "__data__": "not-a-dict",
            }
        ],
    }
    with pytest.raises(DeserializationValueError) as exc_info:
        factory.deserialize_data_from_dict(bad_payload)
    assert field in str(exc_info.value)


@pytest.mark.parametrize("factory, _field", _SET_KINDS)
def test_set_constraint_deserialization_tolerates_extra_unknown_fields(
    factory: type[Constraint], _field: str
) -> None:
    """Test deserialization silently ignores unknown extra fields.

    The TypeGuard requires the *known* keys to be present and well-shaped
    but does not assert the absence of additional keys; deserialization
    picks just the known ones. Documents the forward-compatibility
    contract — older readers can consume payloads written by a newer
    schema with added fields.
    """
    constraint = factory(mock_identifier("x", 0), {1, 2})  # type: ignore[call-arg]
    payload = constraint.serialize_to_dict()
    cast(dict[str, Any], payload["__data__"])["unknown_future_field"] = "ignore me"
    rebuilt = factory.deserialize_from_dict(payload)
    for member in (1, 2):
        assert rebuilt.is_satisfied(member) == constraint.is_satisfied(member)


@pytest.mark.parametrize("factory, field", _SET_KINDS)
def test_set_member_deserializer_rejects_none_after_deserialization(
    factory: type[Constraint], field: str
) -> None:
    """Test deserialized members are revalidated and ``None`` is rejected."""
    x = mock_identifier("x", 0)
    bad_payload: SerializedDict = {
        "variable": x.serialize_to_dict(),
        field: [{"__type__": "builtins.NoneType", "__data__": None}],
    }
    with pytest.raises((DeserializationValueError, ValueError)):
        factory.deserialize_data_from_dict(bad_payload)
