"""Serialization round-trips and deserialization-error coverage."""

from collections.abc import Callable
from typing import Any, cast

import pytest

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializationFormat,
    SerializedDict,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    EquationConstraint,
    InSetConstraint,
    NotInSetConstraint,
)
from fhy_core.symbolic.expression import (
    BinaryOperation,
    LiteralExpression,
    make_binary_expression,
)

from .conftest import (
    SET_KINDS,
    HashCollidingMember,
    SerializableEqualHashable,
    mock_identifier,
)

SetConstraintType = type[Constraint]


# =============================================================================
# Equation constraint round-trip
# =============================================================================


def test_equation_constraint_round_trip_dict_serialization() -> None:
    """Test an `EquationConstraint` round-trips through dict serialization.

    The serialized shape drops the ``variable`` field the old API carried:
    an ``EquationConstraint``'s only data is its wrapped expression.
    """
    x = mock_identifier("x", 0)
    expression = make_binary_expression(BinaryOperation.EQUAL, x, True)
    constraint = EquationConstraint(expression)
    expected = {
        "__type__": "equation_constraint",
        "__data__": {
            "expression": expression.serialize_to_dict(),
        },
    }

    assert constraint.serialize_to_dict() == expected
    rebuilt = EquationConstraint.deserialize_from_dict(constraint.serialize_to_dict())
    assert isinstance(rebuilt, EquationConstraint)
    assert rebuilt.get_free_identifiers() == frozenset({x})
    assert rebuilt.convert_to_expression().is_structurally_equivalent(expression)


def test_equation_constraint_round_trip_preserves_structural_equivalence() -> None:
    """Test a round-tripped `EquationConstraint` stays structurally equivalent."""
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))

    rebuilt = EquationConstraint.deserialize_from_dict(constraint.serialize_to_dict())

    assert rebuilt.is_structurally_equivalent(constraint)


# =============================================================================
# Set constraint round-trips
# =============================================================================


_SET_KINDS_WITH_FIELD = [
    pytest.param(InSetConstraint, "values", id="in_set"),
    pytest.param(NotInSetConstraint, "values", id="not_in_set"),
]


@pytest.mark.parametrize("factory, _field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_round_trip_dict_serialization(
    factory: SetConstraintType, _field: str
) -> None:
    """Test a set constraint round-trips through dict serialization."""
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})  # type: ignore[call-arg]

    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())

    assert isinstance(rebuilt, factory)
    assert rebuilt.variable == x
    for member in (1, 2, 99):
        assert rebuilt.is_satisfied_with_bindings(
            {x: member}
        ) == constraint.is_satisfied_with_bindings({x: member})


@pytest.mark.parametrize("factory, _field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_serialized_payload_uses_the_unified_values_key(
    factory: SetConstraintType, _field: str
) -> None:
    """Test the wire payload carries exactly `variable` and `values`.

    Both set-constraint kinds are implemented by one shared base with a
    single ``values`` field, so the serialized shape has to reflect that
    for both kinds rather than the retired ``valid_values``/
    ``invalid_values`` split.
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2})  # type: ignore[call-arg]

    payload = cast(dict[str, Any], constraint.serialize_to_dict()["__data__"])

    assert set(payload) == {"variable", "values"}


@pytest.mark.parametrize("factory, _field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_round_trip_with_serializable_member(
    factory: SetConstraintType, _field: str
) -> None:
    """Test serializable+hashable members survive a round trip."""
    x = mock_identifier("x", 0)
    member = SerializableEqualHashable(7)
    constraint = factory(x, {member})  # type: ignore[call-arg]

    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())

    assert rebuilt.is_satisfied_with_bindings(
        {x: member}
    ) == constraint.is_satisfied_with_bindings({x: member})
    other = SerializableEqualHashable(8)
    assert rebuilt.is_satisfied_with_bindings(
        {x: other}
    ) == constraint.is_satisfied_with_bindings({x: other})


@pytest.mark.parametrize("factory, _field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_round_trip_with_nested_collection_member(
    factory: SetConstraintType, _field: str
) -> None:
    """Test nested tuple/frozenset members round-trip and stay membership-equivalent."""
    x = mock_identifier("x", 0)
    nested_member = (1, (2, 3), frozenset({4, 5}))
    constraint = factory(x, [nested_member])  # type: ignore[call-arg]

    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())

    assert rebuilt.is_satisfied_with_bindings(
        {x: nested_member}
    ) == constraint.is_satisfied_with_bindings({x: nested_member})


@pytest.mark.parametrize("factory", SET_KINDS)
def test_set_constraint_round_trip_preserves_type_strict_distinct_members(
    factory: SetConstraintType,
) -> None:
    """Test ``[True, 1, 1.0]`` survives a round trip with all three members distinct."""
    x = mock_identifier("x", 0)
    constraint = factory(x, [True, 1, 1.0])  # type: ignore[call-arg]

    rebuilt = type(constraint).deserialize_from_dict(constraint.serialize_to_dict())

    in_set = factory is InSetConstraint
    assert rebuilt.is_satisfied_with_bindings({x: True}) is in_set
    assert rebuilt.is_satisfied_with_bindings({x: 1}) is in_set
    assert rebuilt.is_satisfied_with_bindings({x: 1.0}) is in_set


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_serialization_keeps_bool_int_and_float_distinct(
    factory: SetConstraintType, field: str
) -> None:
    """Test ``[True, 1, 1.0]`` serializes to three distinct member entries."""
    constraint = factory(
        mock_identifier("x", 0),  # type: ignore[call-arg]
        [True, 1, 1.0],
    )

    payload = cast(dict[str, Any], constraint.serialize_to_dict()["__data__"])
    serialized = payload[field]

    assert len(serialized) == 3


def _read_wire_members(constraint: Constraint, field: str) -> list[Any]:
    """Return the serialized member list a set constraint puts on the wire."""
    payload_data = cast(dict[str, Any], constraint.serialize_to_dict()["__data__"])
    members: list[Any] = payload_data[field]
    return members


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_serialized_values_are_repr_sorted(
    factory: SetConstraintType, field: str
) -> None:
    """Test serialized members are emitted in repr-sorted order for determinism.

    The members are chosen so repr-sorted order (``10, 2, 33, 4`` --
    lexicographic on the rendered digits) is not the numeric order and is
    not the order the normalized member set iterates in, so emitting the
    set as it happens to iterate would produce a different list.
    """
    constraint = factory(mock_identifier("x", 0), {10, 2, 33, 4})  # type: ignore[call-arg]

    serialized_values = _read_wire_members(constraint, field)

    assert [member["__data__"] for member in serialized_values] == [10, 2, 33, 4]
    assert serialized_values == sorted(serialized_values, key=repr)


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_wire_order_is_independent_of_construction_order(
    factory: SetConstraintType, field: str
) -> None:
    """Test two constraints over the same members serialize to one byte-identical list.

    The members collide on hash, so the two constraints provably store
    them in different orders. Determinism of the wire form therefore has
    to come from sorting at encode time rather than from the stored order
    happening to agree.
    """
    x = mock_identifier("x", 0)
    members = [HashCollidingMember(1), HashCollidingMember(2)]
    left = factory(x, list(members))  # type: ignore[call-arg]
    right = factory(x, list(reversed(members)))  # type: ignore[call-arg]

    assert getattr(left, field) != getattr(right, field), (
        "the two constraints must store their members in different orders "
        "for this test to say anything about encode-time ordering"
    )
    assert _read_wire_members(left, field) == _read_wire_members(right, field)


# =============================================================================
# Structural payload errors
# =============================================================================


def _drop(key: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda d: {k: v for k, v in d.items() if k != key}


def _replace(key: str, value: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda d: {**d, key: value}


@pytest.fixture
def equation_payload() -> dict[str, Any]:
    """Return a well-formed serialized `EquationConstraint` data payload.

    The payload carries only ``expression``: the old ``variable`` field is
    gone from the shape entirely.
    """
    return {
        "expression": LiteralExpression(True).serialize_to_dict(),
    }


@pytest.mark.parametrize(
    "mutate",
    [
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


def test_equation_constraint_rejects_a_payload_carrying_the_old_variable_field() -> (
    None
):
    """Test a payload carrying the retired `variable` field is rejected.

    The derived deserialization path enforces an exact key set, so a
    payload shaped like the old two-field form is a structure error, not
    a silently-ignored extra key.
    """
    x = mock_identifier("x", 0)
    payload = {
        "variable": x.serialize_to_dict(),
        "expression": LiteralExpression(True).serialize_to_dict(),
    }

    with pytest.raises(DeserializationDictStructureError):
        EquationConstraint.deserialize_data_from_dict(payload)


@pytest.fixture(
    params=[
        pytest.param((InSetConstraint, "values"), id="in_set"),
        pytest.param((NotInSetConstraint, "values"), id="not_in_set"),
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


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
def test_set_member_deserializer_rewraps_dict_structure_error(
    factory: type[Constraint], field: str
) -> None:
    """Test a wrapped-member structure error is re-raised as a value error."""
    x = mock_identifier("x", 0)
    bad_payload: SerializedDict = {
        "variable": x.serialize_to_dict(),
        field: [{"not_a_wrapped": "value"}],
    }

    with pytest.raises(DeserializationValueError):
        factory.deserialize_data_from_dict(bad_payload)


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
def test_set_member_deserializer_rewraps_value_error_with_field_name(
    factory: type[Constraint], field: str
) -> None:
    """Test the member deserializer embeds the field name in re-wrapped errors."""
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


@pytest.mark.parametrize("factory, _field", _SET_KINDS_WITH_FIELD)
def test_set_constraint_deserialization_rejects_extra_unknown_fields(
    factory: type[Constraint], _field: str
) -> None:
    """Test deserialization rejects unknown extra fields.

    The derived deserialization path enforces an exact key set, so an unknown
    extra field is a structure error rather than being silently ignored.
    """
    constraint = factory(mock_identifier("x", 0), {1, 2})  # type: ignore[call-arg]
    payload = constraint.serialize_to_dict()
    cast(dict[str, Any], payload["__data__"])["unknown_future_field"] = "reject me"

    with pytest.raises(DeserializationDictStructureError):
        factory.deserialize_from_dict(payload)


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
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


@pytest.mark.parametrize("factory, field", _SET_KINDS_WITH_FIELD)
@pytest.mark.parametrize("fmt", list(SerializationFormat))
def test_set_constraint_round_trips_through_every_format(
    factory: SetConstraintType, field: str, fmt: SerializationFormat
) -> None:
    """Test a set constraint round-trips through DICT, JSON, and BINARY.

    The type-strict member set is derived from the public member field
    rather than carried on the wire, so a restored constraint has to
    re-derive it and decide exactly as the original does.
    """
    x = mock_identifier("x", 0)
    constraint = factory(x, {1, 2, 3})  # type: ignore[call-arg]

    payload = constraint.serialize(fmt)
    restored = type(constraint).deserialize(payload, fmt)

    assert set(getattr(restored, field)) == {1, 2, 3}
    for probe in (1, 2, 3, 99):
        assert restored.is_satisfied_with_bindings(
            {x: probe}
        ) == constraint.is_satisfied_with_bindings({x: probe})
