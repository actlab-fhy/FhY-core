"""Tests for `ParamAssignment` construction and serialization."""

from typing import Any

import pytest
from immutabledict import immutabledict

from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    SerializationFormat,
    serialize_registry_wrapped_value,
)
from fhy_core.symbolic.constraint import EquationConstraint, NotInSetConstraint
from fhy_core.symbolic.expression import (
    IdentifierExpression,
    LiteralExpression,
    NonBooleanLogicalOperandError,
    piecewise,
)
from fhy_core.symbolic.param import (
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

from .conftest import build_case_condition_constraint, mock_identifier

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
# Constraint-outcome error messages
# =============================================================================


def test_assignment_reports_violation_for_genuinely_violated_constraint() -> None:
    """Test a decided violation raises the ``violates`` message."""
    param = create_integer_param_with_lower_bound(0)

    with pytest.raises(ParamError, match="violates constraint"):
        param.assign(-1)


def test_assignment_reports_could_not_verify_for_undecided_constraint() -> None:
    """Test an undecided constraint raises the ``could not be verified`` message.

    The constraint is dependent on a second identifier that `assign` is
    given no `bindings` for, so it stays undecided. The assigned value is
    admissible, so the failure must surface as an undecided (``could not
    be verified``) error rather than a violation.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    undecided_constraint = EquationConstraint(
        IdentifierExpression(x) < IdentifierExpression(y)
    )
    param = create_integer_param(name=x, constraints=[undecided_constraint])

    with pytest.raises(ParamError, match="could not be verified against constraint"):
        param.assign(3)


def test_assignment_undecided_message_is_distinct_from_violation_message() -> None:
    """Test the undecided error does not use the ``violates`` wording."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    undecided_constraint = EquationConstraint(
        IdentifierExpression(x) < IdentifierExpression(y)
    )
    param = create_integer_param(name=x, constraints=[undecided_constraint])

    with pytest.raises(ParamError) as exc_info:
        param.assign(3)

    message = str(exc_info.value)
    assert "could not be verified" in message
    assert "violates constraint" not in message


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


def test_dependent_assignment_round_trips_through_dict_serialization() -> None:
    """Test an assignment proven via bindings survives a serialization round-trip.

    The bindings that proved the dependent constraint satisfied at
    ``assign`` time are not part of the serialized state, so deserialization
    cannot re-prove satisfaction; it must accept the absence of a provable
    violation rather than reject the assignment as unverifiable.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    dependent = EquationConstraint(IdentifierExpression(x) < IdentifierExpression(y))
    param = create_integer_param(name=x, constraints=[dependent])
    assignment = param.assign(3, bindings={y: 5})

    dictionary = assignment.serialize_to_dict()
    restored: ParamAssignment[Any] = ParamAssignment.deserialize_from_dict(dictionary)

    assert restored.value == 3
    assert restored.param.is_structurally_equivalent(param)
    assert restored.serialize_to_dict() == dictionary


def test_dependent_assignment_round_trips_when_bridge_fails_without_bindings() -> None:
    """Test round-tripping survives a constraint the bridge cannot lower alone.

    The dependent constraint divides by ``x - 5``; substituting the assigned
    value ``5`` for ``x`` alone, with no binding for the other free
    identifier, drives the expression bridge to a complex-infinity failure
    it cannot lift back into an expression. That bridge failure must count
    as an undecided remainder, the same as any other constraint
    deserialization cannot fully resolve, rather than escaping the
    round-trip as a raw bridge exception.
    """
    x = mock_identifier("x", 1)
    y = mock_identifier("y", 2)
    xe, ye = IdentifierExpression(x), IdentifierExpression(y)
    guarded = piecewise((ye > 0, LiteralExpression(1) / (xe - 5)), otherwise=1) > 0
    dependent = EquationConstraint(guarded)
    param = create_integer_param(name=x, constraints=[dependent])
    assignment = param.assign(5, bindings={y: -1})

    dictionary = assignment.serialize_to_dict()
    restored: ParamAssignment[Any] = ParamAssignment.deserialize_from_dict(dictionary)

    assert restored.value == 5
    assert restored.param.is_structurally_equivalent(param)
    assert restored.serialize_to_dict() == dictionary


def test_deserialization_rejects_a_value_that_provably_violates() -> None:
    """Test a tampered payload whose value violates a decidable constraint is rejected.

    Tolerating an undecided dependent constraint must not open the door to
    provably invalid payloads: a value that a decidable constraint rejects
    still fails deserialization.
    """
    x = mock_identifier("x", 1)
    constrained = create_integer_param(name=x, constraints=[NotInSetConstraint(x, {9})])
    tampered = constrained.assign(3).serialize_to_dict()
    donor = create_integer_param(name=mock_identifier("d", 2)).assign(9)
    tampered["value"] = donor.serialize_to_dict()["value"]

    with pytest.raises(DeserializationValueError, match="violates constraint"):
        ParamAssignment.deserialize_from_dict(tampered)


def test_permutation_validate_value_normalizes_list_before_constraint_check() -> None:
    """Test `validate_value` normalizes a list to a tuple before checking constraints.

    A permutation param stores set constraints over tuple members; an
    un-normalized list value would be unhashable. ``validate_value`` normalizes
    first so it agrees with ``is_constraints_satisfied`` and raises only
    ``ParamError``, never ``TypeError``.
    """
    var = mock_identifier("p", 0)
    param = create_permutation_param([1, 2, 3], name=var).add_constraint(
        NotInSetConstraint(var, {(3, 2, 1)})
    )

    # A permitted permutation given as a list validates (normalized to a tuple).
    param.validate_value([1, 2, 3])
    assert param.is_constraints_satisfied([1, 2, 3])

    # A forbidden permutation given as a list raises ParamError, not TypeError.
    with pytest.raises(ParamError):
        param.validate_value([3, 2, 1])


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


# =============================================================================
# Direct construction normalizes like `Param.assign`
# =============================================================================


def test_direct_construction_stores_the_domain_canonical_value() -> None:
    """Test constructing with a mutable value stores the canonical form."""
    param = create_permutation_param([1, 2, 3])
    members: Any = [1, 2, 3]

    assignment = ParamAssignment(param, members)

    assert assignment.value == (1, 2, 3)
    assert isinstance(assignment.value, tuple)


def test_direct_construction_equals_the_assign_form_of_the_same_binding() -> None:
    """Test the constructor and `assign` produce equivalent assignments."""
    param = create_permutation_param([1, 2, 3])
    members: Any = [1, 2, 3]

    constructed = ParamAssignment(param, members)
    assigned = param.assign(members)

    assert constructed.is_structurally_equivalent(assigned)
    assert assigned.is_structurally_equivalent(constructed)


def test_direct_construction_does_not_alias_a_mutable_argument() -> None:
    """Test mutating the argument afterward cannot invalidate the assignment."""
    param = create_permutation_param([1, 2, 3])
    members: Any = [1, 2, 3]

    assignment = ParamAssignment(param, members)
    members.append(4)

    assert assignment.value == (1, 2, 3)


def test_direct_construction_with_a_mutable_value_serializes() -> None:
    """Test a directly constructed assignment round-trips through JSON."""
    param = create_permutation_param(["n", "c", "h", "w"])
    members: Any = ["n", "c", "h", "w"]
    assignment = ParamAssignment(param, members)

    payload = assignment.serialize(SerializationFormat.JSON)
    restored: ParamAssignment[Any] = ParamAssignment.deserialize(
        payload, SerializationFormat.JSON
    )

    assert restored.value == ("n", "c", "h", "w")
    assert restored.is_structurally_equivalent(assignment)


def test_construct_from_fields_stores_the_domain_canonical_value() -> None:
    """Test `construct_from_fields` with a list value stores the canonical tuple."""
    param = create_permutation_param([1, 2, 3])
    members: Any = [1, 2, 3]

    from_fields = ParamAssignment.construct_from_fields(
        {"param": param, "value": members}
    )
    constructed = ParamAssignment(param, members)

    assert from_fields.value == (1, 2, 3)
    assert from_fields.is_structurally_equivalent(constructed)


def test_construct_from_fields_accepts_an_immutabledict() -> None:
    """Test `construct_from_fields` accepts an `immutabledict` field mapping."""
    param = create_permutation_param([1, 2, 3])
    members: Any = [1, 2, 3]

    from_fields = ParamAssignment.construct_from_fields(
        immutabledict({"param": param, "value": members})
    )
    constructed = ParamAssignment(param, members)

    assert from_fields.is_structurally_equivalent(constructed)


# =============================================================================
# An ill-typed constraint is refused, not accepted as undecided
# =============================================================================


def _create_param_conditioned_on_its_own_value() -> Param[int]:
    """Create `x` whose constraint takes `x` itself as a case condition.

    Binding any integer value to `x` puts a number in the case condition.
    """
    x = mock_identifier("x", 1)
    return create_integer_param(
        name=x, constraints=[build_case_condition_constraint(IdentifierExpression(x))]
    )


def test_direct_construction_raises_for_a_number_in_a_case_condition() -> None:
    """Test constructing an assignment that binds a number into a condition raises."""
    param = _create_param_conditioned_on_its_own_value()

    with pytest.raises(NonBooleanLogicalOperandError):
        ParamAssignment(param, 3)


def test_deserialization_refuses_a_number_in_a_case_condition() -> None:
    """Test deserialization refuses an ill-typed payload rather than accepting it.

    Deserialization accepts a constraint it cannot decide, since the
    bindings that proved a dependent constraint are not serialized. An
    ill-typed constraint is not undecided: binding the value puts a number
    in a case condition, so the payload is refused, as a
    `DeserializationValueError` caused by the typed error.
    """
    param = _create_param_conditioned_on_its_own_value()
    payload = {
        "param": param.serialize_to_dict(),
        "value": serialize_registry_wrapped_value(3),
    }

    with pytest.raises(DeserializationValueError) as excinfo:
        ParamAssignment.deserialize_from_dict(payload)  # type: ignore[arg-type]  # test: dict shape

    assert isinstance(excinfo.value.__cause__, NonBooleanLogicalOperandError)
