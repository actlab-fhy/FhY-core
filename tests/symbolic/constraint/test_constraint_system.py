"""Tests for `ConstraintSystem` and `create_constraint_system`."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    FieldCodec,
    SerializationFormat,
    SerializationValueError,
    make_field_codec,
)
from fhy_core.symbolic.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    MissingSymbolTypeError,
    NotInSetConstraint,
    create_constraint_system,
)
from fhy_core.symbolic.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    logical_or,
    make_binary_expression,
)
from fhy_core.symbolic.symbol_type import SymbolType
from fhy_core.term import (
    compared_as_reference,
    compared_as_value,
    excluded_from_equivalence,
)
from fhy_core.traits import Frozen, FrozenMutationError
from fhy_core.utils.override import override

from .conftest import SerializableEqualHashable, mock_identifier

# =============================================================================
# Test helper: a fully callback-driven constraint
# =============================================================================

# ``_ProbeConstraint`` is never serialized; this codec only exists to satisfy
# the schema-derived serialization engine's field-inference pass, which has no
# built-in support for a ``Callable``-typed field.
_UNUSED_CALLBACK_CODEC: FieldCodec = make_field_codec(
    lambda _value: None, lambda _data: lambda *_a, **_k: None
)


@dataclass(frozen=True, eq=False)
class _ProbeConstraint(Constraint):
    """Constraint whose evaluation is fully driven by an injected callback.

    Lets a test observe which members were evaluated, in what order, and
    with what bindings snapshot, without depending on any other kind's
    ``repr`` (whose mock-identifier form is not controllable) for canonical
    ordering: ``label`` drives a deterministic, test-controlled ``repr``.
    """

    variable: Identifier = field(metadata=compared_as_reference())
    label: str = field(metadata=compared_as_value())
    # Annotated with ``Mapping[Identifier, Any]`` (rather than the
    # ``ConstraintBindings`` alias) because this is a dataclass field: the
    # serialization/frozen engines resolve field annotations from this
    # module's own globals, and ``ConstraintBindings`` embeds a
    # module-relative forward reference that only resolves against
    # ``fhy_core.symbolic.constraint``'s namespace.
    on_evaluate_with_bindings: Callable[
        [Mapping[Identifier, Any]], ConstraintOutcome
    ] = field(
        metadata={
            **excluded_from_equivalence(),
            "serialize_codec": _UNUSED_CALLBACK_CODEC,
        }
    )

    @override
    def get_free_identifiers(self) -> frozenset[Identifier]:
        return frozenset({self.variable})

    @override
    def evaluate_with_bindings(self, bindings: ConstraintBindings) -> ConstraintOutcome:
        return self.on_evaluate_with_bindings(bindings)

    @override
    def convert_to_expression(self) -> Expression:
        return LiteralExpression(True)

    @override
    def __repr__(self) -> str:
        return f"_ProbeConstraint({self.label})"

    @override
    def __str__(self) -> str:
        return self.label


# =============================================================================
# Factory / construction
# =============================================================================


def test_create_constraint_system_with_no_arguments_is_empty() -> None:
    """Test the factory with no arguments produces an empty system."""
    system = create_constraint_system()

    assert system.constraints == ()


def test_create_constraint_system_rejects_non_constraint_element() -> None:
    """Test a non-`Constraint` argument raises `ConstraintError`."""
    with pytest.raises(ConstraintError):
        create_constraint_system("not-a-constraint")  # type: ignore[arg-type]


def test_create_constraint_system_rejects_nested_constraint_system() -> None:
    """Test a nested `ConstraintSystem` argument raises `ConstraintError`."""
    x = mock_identifier("x", 0)
    inner = create_constraint_system(InSetConstraint(x, {1, 2}))

    with pytest.raises(ConstraintError):
        create_constraint_system(inner)  # type: ignore[arg-type]


def test_create_constraint_system_retains_duplicate_constraints() -> None:
    """Test conjunction retains duplicate members rather than deduplicating."""
    x = mock_identifier("x", 0)
    member = InSetConstraint(x, {1, 2})

    system = create_constraint_system(member, member)

    assert len(system.constraints) == 2


def test_constraint_system_materializes_a_one_shot_iterator_input() -> None:
    """Test a one-shot iterator input is retained rather than consumed by validation."""
    x = mock_identifier("x", 0)
    first = InSetConstraint(x, {1, 2})
    second = InSetConstraint(x, {2, 3})

    system = ConstraintSystem(iter([first, second]))  # type: ignore[arg-type]

    assert len(system.constraints) == 2


def test_constraint_system_materializes_a_generator_input() -> None:
    """Test a generator input is retained rather than consumed by validation."""
    x = mock_identifier("x", 0)
    members = (InSetConstraint(x, {1, 2}), InSetConstraint(x, {2, 3}))

    system = ConstraintSystem(  # type: ignore[arg-type]
        member for member in members
    )

    assert len(system.constraints) == 2


@pytest.mark.z3
def test_constraint_system_from_a_generator_is_not_vacuously_satisfiable() -> None:
    """Test a generator-built contradictory system is not silently emptied.

    An emptied system answers SATISFIED vacuously, which is the visible
    symptom of dropping the members: ``x in {1}`` and ``x in {2}`` have no
    joint witness and must report VIOLATED.
    """
    x = mock_identifier("x", 0)
    members = (InSetConstraint(x, {1}), InSetConstraint(x, {2}))

    system = ConstraintSystem(  # type: ignore[arg-type]
        member for member in members
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


# =============================================================================
# Canonical ordering and structural equivalence
# =============================================================================


def test_create_constraint_system_orders_members_same_regardless_of_order() -> None:
    """Test the same members in different input order produce the same tuple."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    a = InSetConstraint(x, {1, 2})
    b = InSetConstraint(y, {3, 4})

    system_ab = create_constraint_system(a, b)
    system_ba = create_constraint_system(b, a)

    assert system_ab.constraints == system_ba.constraints


def test_constraint_system_structural_equivalence_ignores_construction_order() -> None:
    """Test independently built, differently ordered systems are equivalent."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    left = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )
    right = create_constraint_system(
        InSetConstraint(y, {3, 4}), InSetConstraint(x, {1, 2})
    )

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_constraint_system_structural_equivalence_false_for_different_members() -> None:
    """Test systems with different member constraints are not equivalent."""
    x = mock_identifier("x", 0)
    left = create_constraint_system(InSetConstraint(x, {1, 2}))
    right = create_constraint_system(InSetConstraint(x, {1, 2, 3}))

    assert not left.is_structurally_equivalent(right)


def test_constraint_system_ordering_is_stable_for_repr_colliding_members() -> None:
    """Test members whose `repr` forms collide still canonicalize deterministically.

    ``InSetConstraint(x, {"5"})`` and ``InSetConstraint(x, {5})`` are not
    structurally equivalent (membership is type-strict), so a canonical
    order keyed on a form that conflates them leaves the two construction
    orders in different member orders.
    """
    x = mock_identifier("x", 0)
    string_member = InSetConstraint(x, {"5"})
    integer_member = InSetConstraint(x, {5})

    left = create_constraint_system(string_member, integer_member)
    right = create_constraint_system(integer_member, string_member)

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_constraint_system_ordering_is_constant_on_equivalent_literal_forms() -> None:
    """Test pairwise-equivalent members canonicalize into the same order.

    ``LiteralExpression`` treats the integer-grammar string ``"5"`` and the
    integer ``5`` as structurally equivalent, so two systems whose members
    are pairwise equivalent must order those members identically and
    compare equivalent themselves.
    """
    n = mock_identifier("n", 0)
    left = create_constraint_system(
        EquationConstraint(LiteralExpression("5")),
        EquationConstraint(LiteralExpression(4)),
    )
    right = create_constraint_system(
        EquationConstraint(LiteralExpression(5)),
        EquationConstraint(LiteralExpression(4)),
    )

    assert left.is_structurally_equivalent(right)
    assert right.is_structurally_equivalent(left)


def test_constraint_system_canonicalizes_alike_for_independent_identifiers() -> None:
    """Test independently constructed, content-identical systems canonicalize alike.

    Two separate ``mock_identifier`` calls with the same name and id are
    distinct ``Mock`` objects, but ``ConstraintSystem`` orders its members
    by a canonical key that reads an identifier through its ``id``. Two
    independently built systems over "the same" identifiers, given in
    different construction order, therefore canonicalize to the same
    member order.
    """
    x1, y1 = mock_identifier("x", 0), mock_identifier("y", 1)
    x2, y2 = mock_identifier("x", 0), mock_identifier("y", 1)
    left = create_constraint_system(
        InSetConstraint(x1, {1, 2}), InSetConstraint(y1, {3, 4})
    )
    right = create_constraint_system(
        InSetConstraint(y2, {3, 4}), InSetConstraint(x2, {1, 2})
    )

    assert [repr(c) for c in left.constraints] == [repr(c) for c in right.constraints]
    assert left.convert_to_expression().is_structurally_equivalent(
        right.convert_to_expression()
    )


# =============================================================================
# Frozen contract
# =============================================================================


def test_constraint_system_implements_frozen_protocol_and_is_frozen() -> None:
    """Test a constructed system satisfies `Frozen` and reports `is_frozen`."""
    system = create_constraint_system()

    assert isinstance(system, Frozen)
    assert system.is_frozen


def test_constraint_system_rejects_arbitrary_attribute_assignment() -> None:
    """Test setattr on a frozen system raises `FrozenMutationError`."""
    system = create_constraint_system()

    with pytest.raises(FrozenMutationError):
        system.arbitrary_probe = "mutation"


# =============================================================================
# Identity equality
# =============================================================================


def test_constraint_system_equality_is_identity_not_structure() -> None:
    """Test two structurally equivalent systems stay distinct values.

    The class documents identity equality as a caller-visible contract:
    equivalent systems are distinct dict keys and distinct set members, and
    ``is_structurally_equivalent`` is the value-equality operation. Both
    systems are built from the same two member instances in opposite
    order, so canonical ordering makes their ``constraints`` tuples
    identical -- structural equality would collapse them, and only
    identity equality keeps them apart.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    first_member = InSetConstraint(x, {1, 2})
    second_member = InSetConstraint(y, {3, 4})
    first = create_constraint_system(first_member, second_member)
    second = create_constraint_system(second_member, first_member)

    assert first.is_structurally_equivalent(second)
    assert first.constraints == second.constraints
    assert first != second
    assert len({first, second}) == 2
    assert len({first: "first", second: "second"}) == 2


# =============================================================================
# repr / str
# =============================================================================


def test_constraint_system_repr_includes_class_name_and_member_reprs() -> None:
    """Test `repr` includes the class name and each member's repr."""
    x = mock_identifier("x", 0)
    member = InSetConstraint(x, {1, 2})
    system = create_constraint_system(member)

    rendered = repr(system)

    assert "ConstraintSystem" in rendered
    assert repr(member) in rendered


def test_constraint_system_str_includes_member_str_forms() -> None:
    """Test `str` includes each member's `str` form."""
    x = mock_identifier("x", 0)
    member = InSetConstraint(x, {1, 2})
    system = create_constraint_system(member)

    assert str(member) in str(system)


def test_constraint_system_str_joins_multiple_members_as_a_conjunction() -> None:
    """Test `str` separates members with the conjunction the system denotes.

    A system is the logical AND of its members, so the separator is part
    of the rendered meaning; a single-member system never shows it.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    first_member = InSetConstraint(x, {1, 2})
    second_member = InSetConstraint(y, {3, 4})
    system = create_constraint_system(first_member, second_member)

    assert system.constraints == (first_member, second_member)
    assert str(system) == f"{first_member} and {second_member}"


def test_constraint_system_str_empty_system_denotes_trivially_true() -> None:
    """Test the empty system's `str` denotes the trivially true conjunction."""
    system = create_constraint_system()

    assert str(system) == "True"


# =============================================================================
# `get_free_identifiers`
# =============================================================================


def test_get_free_identifiers_unions_every_members_free_identifiers() -> None:
    """Test the system's free identifiers union every member's."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}),
        EquationConstraint(
            make_binary_expression(BinaryOperation.LESS, y, z),
        ),
    )

    assert system.get_free_identifiers() == frozenset({x, y, z})


def test_get_free_identifiers_empty_system_is_empty() -> None:
    """Test the empty system's free identifiers is the empty set."""
    system = create_constraint_system()

    assert system.get_free_identifiers() == frozenset()


# =============================================================================
# `evaluate_with_bindings` / `is_satisfied_with_bindings`
# =============================================================================


def test_evaluate_with_bindings_empty_system_is_vacuously_satisfied() -> None:
    """Test the empty system is vacuously SATISFIED under any bindings."""
    system = create_constraint_system()

    assert system.evaluate_with_bindings({}) is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_all_members_satisfied_is_satisfied() -> None:
    """Test the conjunction is SATISFIED when every member is SATISFIED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )

    outcome = system.evaluate_with_bindings({x: 1, y: 3})

    assert outcome is ConstraintOutcome.SATISFIED


def test_evaluate_with_bindings_violated_dominates_undecided() -> None:
    """Test a VIOLATED member dominates a co-occurring UNDECIDED member."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )

    outcome = system.evaluate_with_bindings({x: 99})

    assert outcome is ConstraintOutcome.VIOLATED


def test_evaluate_with_bindings_undecided_without_any_violation() -> None:
    """Test the conjunction is UNDECIDED with no VIOLATED member but not all decided."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )

    outcome = system.evaluate_with_bindings({x: 1})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_evaluate_with_bindings_stops_at_first_violation() -> None:
    """Test evaluation stops after the first VIOLATED member in canonical order."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    calls: list[str] = []

    def violate(_bindings: ConstraintBindings) -> ConstraintOutcome:
        calls.append("first")
        return ConstraintOutcome.VIOLATED

    def satisfy(_bindings: ConstraintBindings) -> ConstraintOutcome:
        calls.append("second")
        return ConstraintOutcome.SATISFIED

    first = _ProbeConstraint(x, "a", violate)
    second = _ProbeConstraint(y, "b", satisfy)
    system = create_constraint_system(first, second)

    outcome = system.evaluate_with_bindings({})

    assert outcome is ConstraintOutcome.VIOLATED
    assert calls == ["first"]


def test_evaluate_with_bindings_uses_a_stable_snapshot_of_the_mapping() -> None:
    """Test a caller mutating the bindings mid-call does not affect later members."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    bindings: dict[Any, Any] = {x: 1, y: 3}
    seen_by_second: list[Any] = []

    def mutate_then_satisfy(_bindings: ConstraintBindings) -> ConstraintOutcome:
        bindings[y] = 999
        return ConstraintOutcome.SATISFIED

    def record_y_and_satisfy(seen_bindings: ConstraintBindings) -> ConstraintOutcome:
        seen_by_second.append(seen_bindings.get(y))
        return ConstraintOutcome.SATISFIED

    first = _ProbeConstraint(x, "a", mutate_then_satisfy)
    second = _ProbeConstraint(y, "b", record_y_and_satisfy)
    system = create_constraint_system(first, second)

    system.evaluate_with_bindings(bindings)

    assert seen_by_second == [3]


def test_is_satisfied_with_bindings_folds_undecided_to_false() -> None:
    """Test an UNDECIDED conjunction outcome maps to `False`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )

    assert system.evaluate_with_bindings({x: 1}) is ConstraintOutcome.UNDECIDED
    assert system.is_satisfied_with_bindings({x: 1}) is False


def test_is_satisfied_with_bindings_true_when_all_members_satisfied() -> None:
    """Test a SATISFIED conjunction outcome maps to `True`."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    assert system.is_satisfied_with_bindings({x: 1}) is True


def test_evaluate_with_bindings_rejects_a_value_outside_the_declared_union() -> None:
    """Test an off-union binding value surfaces as a `ConstraintError`.

    ``ConstraintBindings`` declares ``Expression | LiteralType``; a value
    in neither arm must be reported as a domain error naming the
    identifier rather than escaping as an expression-pass failure.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))
    )

    with pytest.raises(ConstraintError) as exception_info:
        system.evaluate_with_bindings({x: None})  # type: ignore[dict-item]

    assert repr(x) in str(exception_info.value)


def test_evaluate_with_bindings_propagates_type_error_for_unhashable_value() -> None:
    """Test an unhashable bound value propagates `TypeError` from a member."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    with pytest.raises(TypeError):
        system.evaluate_with_bindings({x: [1, 2]})  # type: ignore[dict-item]


# =============================================================================
# `convert_to_expression`
# =============================================================================


def test_convert_to_expression_empty_system_is_literal_true() -> None:
    """Test the empty system converts to the literal `True`."""
    system = create_constraint_system()

    expression = system.convert_to_expression()

    assert isinstance(expression, LiteralExpression)
    assert expression.value is True


def test_convert_to_expression_single_member_is_unwrapped() -> None:
    """Test a single-member system's expression is that member's own expression."""
    x = mock_identifier("x", 0)
    member = InSetConstraint(x, {1, 2})
    system = create_constraint_system(member)

    expression = system.convert_to_expression()

    assert expression.is_structurally_equivalent(member.convert_to_expression())


def test_convert_to_expression_multi_member_is_a_logical_and() -> None:
    """Test a multi-member system's expression is a top-level LOGICAL_AND."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), InSetConstraint(y, {3, 4})
    )

    expression = system.convert_to_expression()

    assert isinstance(expression, BinaryExpression)
    assert expression.operation is BinaryOperation.LOGICAL_AND


def test_convert_to_expression_propagates_constraint_error_from_a_member() -> None:
    """Test a member's `ConstraintError` during conversion propagates."""
    x = mock_identifier("x", 0)
    member = InSetConstraint(x, {SerializableEqualHashable(1)})
    system = create_constraint_system(member)

    with pytest.raises(ConstraintError):
        system.convert_to_expression()


# =============================================================================
# Serialization
# =============================================================================


def test_serialize_to_dict_uses_the_pinned_type_id() -> None:
    """Test the wrapped envelope uses the pinned `constraint_system` type id."""
    system = create_constraint_system()

    payload = system.serialize_to_dict()

    assert payload["__type__"] == "constraint_system"


def test_round_trip_dict_preserves_mixed_kind_members() -> None:
    """Test a mixed-kind system round-trips through dict serialization."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), EquationConstraint(LiteralExpression(True))
    )

    rebuilt = ConstraintSystem.deserialize_from_dict(system.serialize_to_dict())

    assert isinstance(rebuilt, ConstraintSystem)
    assert rebuilt.is_structurally_equivalent(system)


def test_round_trip_through_json_format() -> None:
    """Test a system round-trips through the JSON serialization format."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    rebuilt = ConstraintSystem.deserialize(
        system.serialize(SerializationFormat.JSON), SerializationFormat.JSON
    )

    assert rebuilt.is_structurally_equivalent(system)


def test_round_trip_through_binary_format() -> None:
    """Test a system round-trips through the binary serialization format."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    rebuilt = ConstraintSystem.deserialize(
        system.serialize(SerializationFormat.BINARY), SerializationFormat.BINARY
    )

    assert rebuilt.is_structurally_equivalent(system)


def test_empty_system_round_trips_through_dict_serialization() -> None:
    """Test the empty system round-trips and stays empty."""
    system = create_constraint_system()

    rebuilt = ConstraintSystem.deserialize_from_dict(system.serialize_to_dict())

    assert rebuilt.constraints == ()


def test_wire_members_are_emitted_in_canonical_order() -> None:
    """Test serialized members are emitted in the same order as `constraints`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), EquationConstraint(LiteralExpression(True))
    )

    payload_data = cast(dict[str, Any], system.serialize_to_dict()["__data__"])
    wrapped_members = cast(list[dict[str, Any]], payload_data["constraints"])

    assert [member["__type__"] for member in wrapped_members] == [
        constraint.get_serialization_class_type_id()
        for constraint in system.constraints
    ]


def test_deserialize_data_from_dict_rejects_missing_constraints_field() -> None:
    """Test a missing `constraints` field raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        ConstraintSystem.deserialize_data_from_dict({})


def test_deserialize_data_from_dict_rejects_non_list_constraints_field() -> None:
    """Test a non-list `constraints` field raises a structure error."""
    with pytest.raises(DeserializationDictStructureError):
        ConstraintSystem.deserialize_data_from_dict({"constraints": "not-a-list"})


def test_deserialize_data_from_dict_rejects_malformed_member_entry() -> None:
    """Test a malformed member entry raises a deserialization error."""
    with pytest.raises((DeserializationValueError, DeserializationDictStructureError)):
        ConstraintSystem.deserialize_data_from_dict(
            {"constraints": [{"not_a_wrapped_value": "value"}]}
        )


def test_json_serialization_rejects_a_nan_member() -> None:
    """Test a NaN-valued member propagates the module's NaN-rejection contract."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {float("nan")}))

    with pytest.raises(SerializationValueError):
        system.to_json()


# =============================================================================
# `check_satisfiability` / `check_satisfiability_with_bindings` (z3-backed)
# =============================================================================


@pytest.mark.z3
def test_check_satisfiability_is_satisfied_for_a_satisfiable_system() -> None:
    """Test a satisfiable multi-variable system reports SATISFIED."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_is_violated_for_an_unsatisfiable_strict_cycle() -> None:
    """Test a strict cyclic ordering constraint (x<y<z<x) is unsatisfiable."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    z = mock_identifier("z", 2)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, z)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, z, x)),
    )

    outcome = system.check_satisfiability(
        {x: SymbolType.INT, y: SymbolType.INT, z: SymbolType.INT}
    )

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_satisfiability_mixed_set_and_equation_system_is_satisfiable() -> None:
    """Test a mixed set-and-equation system is satisfiable without rewriting."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2, 3}),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_mixed_set_and_equation_system_is_unsatisfiable() -> None:
    """Test a mixed set-and-equation system can be jointly unsatisfiable."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2, 3}),
        EquationConstraint(make_binary_expression(BinaryOperation.GREATER, x, 100)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_satisfiability_needs_sorts_only_for_the_lowered_conjunction() -> None:
    """Test `symbol_types` need not cover an identifier an empty member set drops.

    An empty `InSetConstraint` reports its `variable` as part of the
    system's scope (`get_free_identifiers`), but its lowered expression
    (`LiteralExpression(False)`) references no identifier at all, so the
    lowered conjunction can have strictly fewer free identifiers than
    `get_free_identifiers()`; `symbol_types` is keyed on what is lowered.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, set()),
        EquationConstraint(make_binary_expression(BinaryOperation.GREATER, y, 0)),
    )

    assert system.get_free_identifiers() == frozenset({x, y})
    assert system.convert_to_expression().get_free_identifiers() == frozenset({y})

    outcome = system.check_satisfiability({y: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


def _extract_reported_missing_names(error: MissingSymbolTypeError) -> str:
    """Return the identifier listing a `MissingSymbolTypeError` message carries.

    The listing is everything after the message's final ``": "``. Reading
    it out separately keeps an assertion off the fixed prefix, which
    already contains several of the single-character name hints the tests
    use and would otherwise satisfy a substring match no matter which
    identifier the error actually reported.
    """
    return str(error).rpartition(": ")[2].rstrip(".")


@pytest.mark.z3
def test_check_satisfiability_raises_missing_symbol_type_error() -> None:
    """Test a missing `symbol_types` entry raises `MissingSymbolTypeError` naming it."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )

    with pytest.raises(MissingSymbolTypeError) as exception_info:
        system.check_satisfiability({x: SymbolType.INT})

    assert _extract_reported_missing_names(exception_info.value) == y.name_hint


@pytest.mark.z3
def test_check_satisfiability_with_bindings_raises_missing_symbol_type_error() -> None:
    """Test a missing `symbol_types` entry for a residual free identifier raises.

    Mirrors ``test_check_satisfiability_raises_missing_symbol_type_error``
    for the bindings-aware entry point: the entry is missing for an
    identifier left free after substitution, not for a bound one.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )

    with pytest.raises(MissingSymbolTypeError) as exception_info:
        system.check_satisfiability_with_bindings({x: 5}, {})

    assert _extract_reported_missing_names(exception_info.value) == y.name_hint


@pytest.mark.z3
def test_check_satisfiability_reports_every_missing_identifier_in_sorted_order() -> (
    None
):
    """Test two missing entries are both reported, ordered by name hint.

    The identifiers are declared so that their ``id`` order (which drives
    the free-identifier set's iteration order) is the reverse of their
    name-hint order, so a listing that skipped the sort would come out
    ``b, a``.
    """
    b = mock_identifier("b", 0)
    a = mock_identifier("a", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, b, a))
    )

    with pytest.raises(MissingSymbolTypeError) as exception_info:
        system.check_satisfiability({})

    assert _extract_reported_missing_names(exception_info.value) == (
        f"{a.name_hint}, {b.name_hint}"
    )


def test_evaluate_with_bindings_missing_value_binding_degrades_to_undecided() -> None:
    """Test a missing value binding is UNDECIDED, contrasting a missing symbol type.

    Uses the same system and missing identifier as
    ``test_check_satisfiability_raises_missing_symbol_type_error``: a
    missing VALUE binding here degrades gracefully to UNDECIDED, unlike a
    missing SYMBOL TYPE on the Z3-backed satisfiability methods, which raises.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )

    outcome = system.evaluate_with_bindings({x: 1})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_with_bindings_rejects_an_off_union_binding_value() -> (
    None
):
    """Test an off-union binding value raises before the conjunction is lowered.

    The bindings are coerced into a substitution environment ahead of the
    solver, so a value in neither arm of ``Expression | LiteralType`` has
    to be reported as a domain error naming the identifier rather than
    escaping as an expression-pass failure.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))
    )

    with pytest.raises(ConstraintError) as exception_info:
        system.check_satisfiability_with_bindings(
            {x: None},  # type: ignore[dict-item]
            {x: SymbolType.INT},
        )

    assert repr(x) in str(exception_info.value)


def test_check_satisfiability_propagates_constraint_error_from_a_member() -> None:
    """Test a member that cannot be lowered raises `ConstraintError`.

    ``check_satisfiability`` reaches the member through
    ``convert_to_expression``, so a member whose value set holds a
    non-literal must surface the documented ``ConstraintError`` rather
    than an outcome.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        InSetConstraint(x, {SerializableEqualHashable(1)})
    )

    with pytest.raises(ConstraintError):
        system.check_satisfiability({x: SymbolType.INT})


def test_check_satisfiability_with_bindings_propagates_constraint_error() -> None:
    """Test the bindings entry point raises `ConstraintError` for the same member.

    ``check_satisfiability_with_bindings`` calls ``convert_to_expression``
    exactly as ``check_satisfiability`` does, so it shares the documented
    ``ConstraintError`` failure mode.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        InSetConstraint(x, {SerializableEqualHashable(1)})
    )

    with pytest.raises(ConstraintError):
        system.check_satisfiability_with_bindings({}, {x: SymbolType.INT})


def test_check_satisfiability_empty_system_does_not_invoke_the_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the empty system short-circuits to SATISFIED without calling z3."""

    def _fail_if_called(*args: object, **kwargs: object) -> bool | None:
        raise AssertionError("check_expression_satisfiability must not be called")

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fail_if_called
    )
    system = create_constraint_system()

    outcome = system.check_satisfiability({})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_matches_the_documented_example() -> None:
    """Test `{x: 5}` on `{x<y, y<3}` is VIOLATED after substitution."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, 3)),
    )

    outcome = system.check_satisfiability_with_bindings({x: 5}, {y: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_satisfiable_after_substitution() -> None:
    """Test a partial assignment can leave the residual system satisfiable."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, 30)),
    )

    outcome = system.check_satisfiability_with_bindings({x: 5}, {y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_with_closed_conjunction_needs_no_symbol_types() -> None:
    """Test a system whose expression has no free identifiers needs no symbol types."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(EquationConstraint(LiteralExpression(True)))

    outcome = system.check_satisfiability({})

    assert outcome is ConstraintOutcome.SATISFIED


def test_check_satisfiability_with_bindings_empty_system_does_not_invoke_the_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the empty system short-circuits to SATISFIED without calling z3.

    Mirrors ``test_check_satisfiability_empty_system_does_not_invoke_the_solver``
    for the bindings-aware entry point.
    """

    def _fail_if_called(*args: object, **kwargs: object) -> bool | None:
        raise AssertionError("check_expression_satisfiability must not be called")

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fail_if_called
    )
    system = create_constraint_system()

    outcome = system.check_satisfiability_with_bindings({}, {})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_constants_only_needs_no_symbol_types() -> (
    None
):
    """Test a constants-only system needs no symbol types with empty bindings.

    Mirrors ``test_check_satisfiability_with_closed_conjunction_needs_no_symbol_types``
    for the bindings-aware entry point: with no bindings supplied, the
    residual is the same closed expression, which needs no symbol types.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(EquationConstraint(LiteralExpression(True)))

    outcome = system.check_satisfiability_with_bindings({}, {})

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# `evaluate_with_bindings` / `check_satisfiability_with_bindings` cross-path
# agreement (z3-backed)
# =============================================================================


@pytest.mark.z3
def test_evaluate_and_check_satisfiability_with_bindings_do_not_contradict() -> None:
    """Test the two bindings-aware APIs never reach opposite decided outcomes.

    For a chained binding such as ``{x: y, y: 5}`` on ``x < 5``,
    ``evaluate_with_bindings`` and ``check_satisfiability_with_bindings``
    (which substitutes through the always-simultaneous IR-level
    ``Expression.substitute``) must agree: neither reports VIOLATED
    while the other reports SATISFIED or UNDECIDED on the residual
    ``y < 5``.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 5))
    )
    bindings: ConstraintBindings = {x: IdentifierExpression(y), y: 5}

    evaluate_outcome = system.evaluate_with_bindings(bindings)
    satisfiability_outcome = system.check_satisfiability_with_bindings(
        bindings, {y: SymbolType.INT}
    )

    assert evaluate_outcome is not ConstraintOutcome.VIOLATED
    assert satisfiability_outcome is not ConstraintOutcome.VIOLATED


# =============================================================================
# Bool set-member sort ambiguity (z3-backed)
# =============================================================================


@pytest.mark.z3
def test_check_satisfiability_bool_member_ambiguity_is_undecided_not_violated() -> None:
    """Test a bool-ambiguous system that is type-strictly satisfiable reports UNDECIDED.

    ``x`` typed ``INT`` with ``x in {1}`` and ``x not in {True}`` is
    satisfied by the concrete witness ``x = 1`` under the package's
    type-strict membership semantics (``evaluate_with_bindings`` agrees).
    The Z3 bridge cannot preserve that distinction when lowering the bool
    member (Z3 coerces ``BoolVal(True)`` against an ``Int`` sort to the
    integer ``1``), so ``check_satisfiability`` reports UNDECIDED rather
    than the provably wrong VIOLATED.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        InSetConstraint(x, {1}), NotInSetConstraint(x, {True})
    )

    assert system.evaluate_with_bindings({x: 1}) is ConstraintOutcome.SATISFIED
    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_bool_member_under_int_sort_is_not_satisfied() -> None:
    """Test a bool member under a non-bool sort is never reported SATISFIED.

    No admissible ``int`` value type-strictly equals ``True``, so a
    system whose only constraint is ``x in {True}`` under an ``INT`` sort
    must not be reported SATISFIED; Z3's coercion of ``True`` to the
    integer ``1`` would otherwise let ``x = 1`` spuriously satisfy it.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {True}))

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is not ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_bool_member_under_bool_sort_is_unaffected() -> None:
    """Test a bool member under a `BOOL`-sorted variable still decides soundly.

    The sort-ambiguity guard must be specific to non-``BOOL`` sorts; a
    variable that is itself ``BOOL``-typed has no coercion ambiguity and
    should still be decided.
    """
    b = mock_identifier("b", 0)
    system = create_constraint_system(InSetConstraint(b, {True, False}))

    outcome = system.check_satisfiability({b: SymbolType.BOOL})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_bindings_bool_member_colliding_int_is_undecided() -> None:
    """Test a bool-ambiguous variable bound to a colliding int stays UNDECIDED.

    ``x != True`` is type-strictly SATISFIED for any bound int (``1`` is
    never ``True``), but Z3 lowers ``True`` to the integer ``1``, so
    binding ``x`` to exactly ``1`` would otherwise let the sort coercion
    report the provably-wrong VIOLATED. The ambiguity guard does not
    special-case bound identifiers, precisely to catch this collision.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(NotInSetConstraint(x, {True}))

    assert system.evaluate_with_bindings({x: 1}) is ConstraintOutcome.SATISFIED
    outcome = system.check_satisfiability_with_bindings({x: 1}, {})

    assert outcome is ConstraintOutcome.UNDECIDED


# =============================================================================
# Bool literal sort ambiguity in an `EquationConstraint` (z3-backed)
# =============================================================================


@pytest.mark.z3
def test_check_satisfiability_equation_bool_literal_ambiguity_is_undecided() -> None:
    """Test an equation's bool literal against a non-bool variable is UNDECIDED.

    ``x == True`` with ``x`` typed ``INT`` has no honest ``INT`` witness:
    the only value that actually satisfies it is the ``bool`` ``True``
    itself (``evaluate_with_bindings({x: True})`` is SATISFIED), which is
    not an ``int`` at all, and every genuine ``int`` -- including ``1``,
    the value Z3's coercion identifies ``True`` with -- provably VIOLATES
    it under the package's type-strict semantics
    (``evaluate_with_bindings({x: 1})`` is VIOLATED). The Z3 bridge does
    not see this distinction: it lowers the bool literal ``True`` to the
    integer ``1`` and would spuriously decide the system SATISFIED at
    ``x = 1``. Both satisfiability APIs must report UNDECIDED instead of
    that provably wrong decided outcome.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
        ),
    )
    system = create_constraint_system(constraint)

    assert constraint.evaluate_with_bindings({x: 1}) is ConstraintOutcome.VIOLATED
    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.SATISFIED

    outcome = system.check_satisfiability({x: SymbolType.INT})
    outcome_with_bindings = system.check_satisfiability_with_bindings(
        {}, {x: SymbolType.INT}
    )

    assert outcome is ConstraintOutcome.UNDECIDED
    assert outcome_with_bindings is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_equation_bool_literal_under_bool_sort_is_unaffected() -> (
    None
):
    """Test the guard exempts an equation whose only free identifier is BOOL-typed.

    The sort-ambiguity guard must be specific to non-``BOOL`` sorts; a
    variable that is itself ``BOOL``-typed has no coercion ambiguity, so
    ``x == True`` under a ``BOOL`` sort should still be decided.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
        ),
    )
    system = create_constraint_system(constraint)

    outcome = system.check_satisfiability({x: SymbolType.BOOL})
    outcome_with_bindings = system.check_satisfiability_with_bindings(
        {}, {x: SymbolType.BOOL}
    )

    assert outcome is ConstraintOutcome.SATISFIED
    assert outcome_with_bindings is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_non_bool_member_under_bool_sort_is_undecided() -> None:
    """Test a non-bool member against a `BOOL`-sorted variable is UNDECIDED.

    The coercion hazard is symmetric: ``x in {1}`` with ``x`` typed
    ``BOOL`` lowers to ``Bool('x') == IntVal(1)``, which Z3 decides by
    coercing the boolean to an integer. Type-strictly no boolean value
    satisfies the constraint (``evaluate_with_bindings`` VIOLATES both
    ``True`` and ``False``), so a decided SATISFIED would be provably
    wrong.
    """
    x = mock_identifier("x", 0)
    constraint = InSetConstraint(x, {1})
    system = create_constraint_system(constraint)

    assert constraint.evaluate_with_bindings({x: True}) is ConstraintOutcome.VIOLATED
    assert constraint.evaluate_with_bindings({x: False}) is ConstraintOutcome.VIOLATED

    outcome = system.check_satisfiability({x: SymbolType.BOOL})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_closed_bool_versus_int_comparison_is_undecided() -> None:
    """Test a closed bool-against-int comparison is UNDECIDED, not SATISFIED.

    ``True == 1`` has no free identifiers at all, so no symbol type can
    exempt it. Type-strictly the comparison is false
    (``evaluate_with_bindings`` VIOLATES it), while Z3 coerces the boolean
    to the integer ``1`` and would decide it SATISFIED.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        make_binary_expression(
            BinaryOperation.EQUAL, LiteralExpression(True), LiteralExpression(1)
        ),
    )
    system = create_constraint_system(constraint)

    assert constraint.evaluate_with_bindings({}) is ConstraintOutcome.VIOLATED

    outcome = system.check_satisfiability({})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_bool_value_against_int_member() -> None:
    """Test a bool binding value against an int set member is UNDECIDED.

    The binding value is lifted into the lowered expression exactly like a
    set member is, so ``y not in {1}`` under ``{y: True}`` lowers to
    ``BoolVal(True) != IntVal(1)`` and hits the same coercion. Type-strictly
    the constraint is SATISFIED (``True`` is not the integer ``1``), so the
    coerced VIOLATED is provably wrong.
    """
    y = mock_identifier("y", 0)
    system = create_constraint_system(NotInSetConstraint(y, {1}))

    assert system.evaluate_with_bindings({y: True}) is ConstraintOutcome.SATISFIED

    outcome = system.check_satisfiability_with_bindings({y: True}, {})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_bool_value_against_int_membership() -> None:
    """Test a bool binding value against a permitted int member is UNDECIDED.

    Mirrors the forbidden-set case with the opposite polarity: ``y in {1}``
    under ``{y: True}`` is type-strictly VIOLATED, while Z3's coercion of
    ``BoolVal(True)`` to the integer ``1`` would decide it SATISFIED.
    """
    y = mock_identifier("y", 0)
    system = create_constraint_system(InSetConstraint(y, {1}))

    assert system.evaluate_with_bindings({y: True}) is ConstraintOutcome.VIOLATED

    outcome = system.check_satisfiability_with_bindings({y: True}, {})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_set_ambiguity_survives_equation_branch() -> None:
    """Test the pre-existing set-constraint bool-ambiguity guard still fires.

    Guards against a regression where extending the ambiguity check to
    ``EquationConstraint`` could shadow or short-circuit the original
    ``InSetConstraint``/``NotInSetConstraint`` branch: a bool-ambiguous set
    constraint alongside an ordinary (non-bool-literal) equation
    constraint must still be reported UNDECIDED.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {True}),
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, 5)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


# =============================================================================
# Bool sort-hazard precision: soundly lowered bool literals stay decidable
# =============================================================================


@pytest.mark.z3
def test_check_satisfiability_bool_literal_under_a_logical_operator_is_decided() -> (
    None
):
    """Test a bool literal used as a logical operand does not trigger the guard.

    ``false || x > 5`` lowers the bool literal into ``z3.Or``, which takes
    boolean operands and performs no integer coercion, so the system stays
    decidable even though ``x`` is ``INT``-typed.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            logical_or(
                LiteralExpression(False),
                make_binary_expression(BinaryOperation.GREATER, x, 5),
            ),
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_sound_bool_literal_does_not_poison_other_members() -> (
    None
):
    """Test a soundly lowered bool literal leaves the rest of the system decidable.

    The hazard is per-site, not per-system: a member whose bool literal is
    consumed by a logical operator must not degrade an otherwise decidable
    conjunction to UNDECIDED.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(
            logical_or(
                LiteralExpression(False),
                make_binary_expression(BinaryOperation.GREATER, x, 5),
            ),
        ),
        InSetConstraint(y, {1, 2}),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_bare_bool_literal_equation_is_decided() -> None:
    """Test a bare bool literal that is compared against nothing stays decidable.

    ``EquationConstraint(LiteralExpression(True))`` lowers to a lone
    ``BoolVal`` with no coercing operator above it, so no hazard exists
    even alongside an ``INT``-sorted member constraint on the same variable.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(LiteralExpression(True)),
        InSetConstraint(x, {1, 2}),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_bool_binding_matching_bool_literal() -> (
    None
):
    """Test binding a bool variable to a bool value leaves the residual decidable.

    ``x == True`` under ``{x: True}`` substitutes to ``True == True``,
    a comparison of two booleans that Z3 lowers without coercion, so the
    residual is decided rather than degraded to UNDECIDED.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
            ),
        )
    )

    assert system.evaluate_with_bindings({x: True}) is ConstraintOutcome.SATISFIED

    outcome = system.check_satisfiability_with_bindings({x: True}, {})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_piecewise_with_uniform_bool_branches_is_decided() -> None:
    """Test a piecewise whose branches are all Boolean lowers faithfully.

    ``z3.If`` forces its arms to one sort; arms that already agree need no
    coercion, so the bool literals in the branches are not a hazard.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            Expression.piecewise(
                (
                    make_binary_expression(BinaryOperation.GREATER, x, 5),
                    LiteralExpression(True),
                ),
                otherwise=LiteralExpression(False),
            ),
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_piecewise_with_mixed_branches_is_undecided() -> None:
    """Test a piecewise mixing a Boolean and a numeric branch is UNDECIDED.

    ``z3.If`` coerces the Boolean arm to an integer to match its numeric
    sibling, which is the same hazard a mixed comparison carries.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.EQUAL,
                IdentifierExpression(x),
                Expression.piecewise(
                    (
                        make_binary_expression(BinaryOperation.GREATER, x, 5),
                        LiteralExpression(True),
                    ),
                    otherwise=LiteralExpression(0),
                ),
            ),
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


# =============================================================================
# Missing symbol types raise ahead of the bool sort-hazard screen (z3-backed)
# =============================================================================


@pytest.mark.z3
def test_check_satisfiability_missing_symbol_type_raises_despite_bool_hazard() -> None:
    """Test the missing-symbol-type precondition raises even for a hazardous system.

    Omitting ``symbol_types`` is exactly what makes the sort hazard fire,
    so the documented ``MissingSymbolTypeError`` must not be downgraded to
    ``UNDECIDED`` by the screen running first.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {True, 2}))

    with pytest.raises(MissingSymbolTypeError, match=x.name_hint):
        system.check_satisfiability({})


@pytest.mark.z3
def test_check_satisfiability_with_bindings_missing_symbol_type_raises_on_hazard() -> (
    None
):
    """Test the bindings entry point also raises ahead of the sort-hazard screen.

    Mirrors ``test_check_satisfiability_missing_symbol_type_raises_despite_bool_hazard``
    for the bindings-aware entry point, with the hazardous identifier left
    unbound so it survives into the residual.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {True, 2}))

    with pytest.raises(MissingSymbolTypeError, match=x.name_hint):
        system.check_satisfiability_with_bindings({}, {})


# =============================================================================
# `timeout_milliseconds` passthrough (solver seam replaced)
# =============================================================================


def test_check_satisfiability_forwards_timeout_milliseconds_to_the_solver_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_satisfiability` forwards `timeout_milliseconds` to the solver."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2, 3}))
    captured: dict[str, Any] = {}

    def _fake_check(
        expression: Expression,
        symbol_types: dict[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> bool | None:
        captured["timeout_milliseconds"] = timeout_milliseconds
        return True

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fake_check
    )

    outcome = system.check_satisfiability(
        {x: SymbolType.INT}, timeout_milliseconds=2500
    )

    assert outcome is ConstraintOutcome.SATISFIED
    assert captured["timeout_milliseconds"] == 2500


def test_check_satisfiability_with_bindings_forwards_timeout_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_satisfiability_with_bindings` forwards `timeout_milliseconds`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )
    captured: dict[str, Any] = {}

    def _fake_check(
        expression: Expression,
        symbol_types: dict[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> bool | None:
        captured["timeout_milliseconds"] = timeout_milliseconds
        return True

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fake_check
    )

    outcome = system.check_satisfiability_with_bindings(
        {x: 1}, {y: SymbolType.INT}, timeout_milliseconds=1000
    )

    assert outcome is ConstraintOutcome.SATISFIED
    assert captured["timeout_milliseconds"] == 1000


# =============================================================================
# Solver `unknown` result (`None`) maps to `UNDECIDED`
# =============================================================================


def test_check_satisfiability_solver_unknown_result_is_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_satisfiability` maps a solver `None` (unknown) result to UNDECIDED.

    Complements the `timeout_milliseconds` passthrough tests above, which
    patch the same seam but always return `True`: this covers the
    `_decide_satisfiability` branch where the solver itself is
    inconclusive.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2, 3}))

    def _fake_check(
        expression: Expression,
        symbol_types: dict[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> bool | None:
        return None

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fake_check
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_with_bindings_solver_unknown_result_is_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_satisfiability_with_bindings` maps solver `None` to UNDECIDED.

    Mirrors `test_check_satisfiability_solver_unknown_result_is_undecided`
    for the bindings-aware entry point.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, y))
    )

    def _fake_check(
        expression: Expression,
        symbol_types: dict[Identifier, SymbolType],
        *,
        timeout_milliseconds: int | None = None,
    ) -> bool | None:
        return None

    monkeypatch.setattr(
        "fhy_core.symbolic.constraint.check_expression_satisfiability", _fake_check
    )

    outcome = system.check_satisfiability_with_bindings({x: 1}, {y: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


# =============================================================================
# Reporting the cause of an undecided outcome
# =============================================================================

_CONSTRAINT_LOGGER = "fhy_core.symbolic.constraint"


def _find_records(
    caplog: pytest.LogCaptureFixture, level: int
) -> list[logging.LogRecord]:
    """Return the constraint module's records emitted at exactly ``level``."""
    return [
        record
        for record in caplog.records
        if record.levelno == level and record.name == _CONSTRAINT_LOGGER
    ]


def test_check_satisfiability_logs_warning_for_the_bool_coercion_hazard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the bool sort-hazard screen reports what it refused to lower.

    Returning UNDECIDED here is a capability gap in the Z3 bridge, not
    ordinary partial evaluation, so it warns -- and it names the offending
    node and the identifier sort involved, because a caller who cannot see
    either has no way to tell this apart from a solver timeout and will
    reach for ``timeout_milliseconds``, which cannot help.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {True}))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the hazardous node"
    message = warnings[0].getMessage()
    assert "check_satisfiability" in message
    assert repr(x) in message
    assert SymbolType.INT.name in message


def test_check_satisfiability_with_bindings_logs_warning_for_the_bool_coercion_hazard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the residual screened after substitution reports the same way."""
    y = mock_identifier("y", 0)
    system = create_constraint_system(NotInSetConstraint(y, {1}))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.check_satisfiability_with_bindings({y: True}, {})

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the hazardous node"
    message = warnings[0].getMessage()
    assert "check_satisfiability_with_bindings" in message


@pytest.mark.z3
def test_check_satisfiability_logs_nothing_when_the_solver_decides(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a decided satisfiability check emits no record at any level."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED
    assert not _find_records(caplog, logging.WARNING)
    assert not _find_records(caplog, logging.DEBUG)


def test_system_evaluate_with_bindings_logs_debug_naming_the_undecided_member(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the system reports which member left the conjunction undecided.

    A system-level UNDECIDED that names no member forces the caller to
    re-check every constraint by hand to find the one that could not be
    decided.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    undecided_member = InSetConstraint(y, {1, 2})
    system = create_constraint_system(InSetConstraint(x, {1, 2}), undecided_member)

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.evaluate_with_bindings({x: 1})

    assert outcome is ConstraintOutcome.UNDECIDED
    messages = [record.getMessage() for record in _find_records(caplog, logging.DEBUG)]
    assert any(
        "ConstraintSystem" in message and repr(undecided_member) in message
        for message in messages
    ), "expected a DEBUG record from the system naming the undecided member"


def test_system_evaluate_with_bindings_logs_nothing_when_every_member_decides(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a fully decided system emits no record at any level."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(InSetConstraint(x, {1, 2}))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.evaluate_with_bindings({x: 1})

    assert outcome is ConstraintOutcome.SATISFIED
    assert not _find_records(caplog, logging.DEBUG)
    assert not _find_records(caplog, logging.WARNING)


# =============================================================================
# `timeout_milliseconds` validation on the paths that skip the solver
# =============================================================================

_INVALID_TIMEOUTS = [
    pytest.param(-5, id="negative"),
    pytest.param(0, id="zero"),
]
"""Values ``timeout_milliseconds`` must reject; only ``None`` or positive pass."""


def _build_bool_hazard_system(variable: Identifier) -> ConstraintSystem:
    """Return a system whose lowering trips the Boolean-coercion screen."""
    return create_constraint_system(InSetConstraint(variable, {True}))


@pytest.mark.parametrize("timeout_milliseconds", _INVALID_TIMEOUTS)
def test_check_satisfiability_rejects_invalid_timeout_for_an_empty_system(
    timeout_milliseconds: int,
) -> None:
    """Test the empty system validates ``timeout_milliseconds`` before returning.

    The empty system is vacuously satisfiable and never reaches the
    solver, but the documented contract promises the raise
    unconditionally, so the argument cannot be accepted here while the
    ordinary path rejects it.
    """
    system = create_constraint_system()

    with pytest.raises(ValueError):
        system.check_satisfiability({}, timeout_milliseconds=timeout_milliseconds)


@pytest.mark.parametrize("timeout_milliseconds", _INVALID_TIMEOUTS)
def test_check_satisfiability_rejects_invalid_timeout_for_a_hazardous_system(
    timeout_milliseconds: int,
) -> None:
    """Test the bool-coercion early return still validates ``timeout_milliseconds``."""
    x = mock_identifier("x", 0)
    system = _build_bool_hazard_system(x)

    with pytest.raises(ValueError):
        system.check_satisfiability(
            {x: SymbolType.INT}, timeout_milliseconds=timeout_milliseconds
        )


@pytest.mark.parametrize("timeout_milliseconds", _INVALID_TIMEOUTS)
def test_check_satisfiability_with_bindings_rejects_invalid_timeout_when_empty(
    timeout_milliseconds: int,
) -> None:
    """Test the empty-system early return validates ``timeout_milliseconds``."""
    system = create_constraint_system()

    with pytest.raises(ValueError):
        system.check_satisfiability_with_bindings(
            {}, {}, timeout_milliseconds=timeout_milliseconds
        )


@pytest.mark.parametrize("timeout_milliseconds", _INVALID_TIMEOUTS)
def test_check_satisfiability_with_bindings_rejects_invalid_timeout_when_hazardous(
    timeout_milliseconds: int,
) -> None:
    """Test the bool-coercion early return validates ``timeout_milliseconds``."""
    y = mock_identifier("y", 0)
    system = create_constraint_system(NotInSetConstraint(y, {1}))

    with pytest.raises(ValueError):
        system.check_satisfiability_with_bindings(
            {y: True}, {}, timeout_milliseconds=timeout_milliseconds
        )


@pytest.mark.parametrize(
    "timeout_milliseconds", [pytest.param(None, id="none"), pytest.param(1, id="one")]
)
def test_check_satisfiability_accepts_a_valid_timeout_for_an_empty_system(
    timeout_milliseconds: int | None,
) -> None:
    """Test validation does not disturb the vacuous outcome for accepted values."""
    system = create_constraint_system()

    outcome = system.check_satisfiability({}, timeout_milliseconds=timeout_milliseconds)

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# Division-by-possibly-zero hazard screen (new; contains audit finding C1)
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.DIVIDE, BinaryOperation.FLOOR_DIVIDE, BinaryOperation.MODULO],
    ids=["divide", "floor_divide", "modulo"],
)
def test_check_satisfiability_division_by_non_literal_divisor_is_undecided(
    operation: BinaryOperation,
) -> None:
    """Test DIVIDE/FLOOR_DIVIDE/MODULO by a non-literal divisor is UNDECIDED.

    The divisor `x` could be zero for some assignment, and the solver
    seam's satisfiability encoding for division is unsound around a zero
    divisor (audit finding C1); the screen refuses to hand such an
    expression to the solver rather than report a decided outcome the
    lowering cannot support.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.NOT_EQUAL, make_binary_expression(operation, x, x), 1
            )
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.REAL})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_division_by_literal_zero_is_undecided() -> None:
    """Test a literal-zero divisor is UNDECIDED, not a decided outcome.

    A literal ``0`` divisor is provably hazardous (not merely possibly
    zero), so it is screened the same as any other non-nonzero-literal
    divisor.
    """
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.EQUAL,
                make_binary_expression(BinaryOperation.DIVIDE, 1, 0),
                1,
            )
        )
    )

    outcome = system.check_satisfiability({})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_with_bindings_division_hazard_screens_the_residual() -> (
    None
):
    """Test the division screen also covers the post-substitution residual."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.NOT_EQUAL,
                make_binary_expression(BinaryOperation.DIVIDE, x, y),
                1,
            )
        )
    )

    outcome = system.check_satisfiability_with_bindings({x: 5}, {y: SymbolType.REAL})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_logs_warning_for_the_division_hazard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the division screen reports what it refused to lower."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.NOT_EQUAL,
                make_binary_expression(BinaryOperation.DIVIDE, x, x),
                1,
            )
        )
    )

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.check_satisfiability({x: SymbolType.REAL})

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the hazardous division node"
    assert "check_satisfiability" in warnings[0].getMessage()


@pytest.mark.z3
def test_check_satisfiability_modulo_by_nonzero_literal_stays_decided() -> None:
    """Test MODULO by a nonzero literal divisor is the non-triggering neighbor.

    Contrasts the division-hazard screen: a nonzero *literal* divisor is
    provably never zero, so the expression is handed to the solver and
    decided normally.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.EQUAL,
                make_binary_expression(BinaryOperation.MODULO, x, 2),
                0,
            )
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_divide_by_nonzero_literal_stays_decided() -> None:
    """Test DIVIDE by a nonzero literal divisor also stays decided."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(
            make_binary_expression(
                BinaryOperation.GREATER,
                make_binary_expression(BinaryOperation.DIVIDE, x, 2),
                0,
            )
        )
    )

    outcome = system.check_satisfiability({x: SymbolType.REAL})

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# Int/float `EQUAL`/`NOT_EQUAL` sort-mixing hazard screen (new; closes C6's float arm)
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    [BinaryOperation.EQUAL, BinaryOperation.NOT_EQUAL],
    ids=["equal", "not_equal"],
)
def test_check_satisfiability_int_identifier_against_float_literal_is_undecided(
    operation: BinaryOperation,
) -> None:
    """Test EQUAL/NOT_EQUAL mixing an INT-sorted identifier and a float literal.

    Z3's ``ToReal`` rationalization of the INT-sorted operand collapses
    this package's type-strict int/float distinction (no ``int`` is ever
    ``==`` a ``float`` under ``evaluate_with_bindings``), so the screen
    refuses to hand the comparison to the solver.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(operation, x, 1.5))
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_satisfiability_logs_warning_for_the_int_float_equality_hazard(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the int/float equality screen reports what it refused to lower."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.EQUAL, x, 1.5))
    )

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the hazardous node"
    message = warnings[0].getMessage()
    assert "check_satisfiability" in message
    assert repr(x) in message


def test_check_satisfiability_with_bindings_int_float_hazard_screens_residual() -> None:
    """Test the residual after substitution is screened the same way.

    Binding ``y`` to a float-bucket value against an ``INT``-sorted ``x``
    in ``x == y`` leaves the same hazardous shape in the residual.
    """
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.EQUAL, x, y))
    )

    outcome = system.check_satisfiability_with_bindings({y: 1.5}, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


@pytest.mark.z3
def test_check_satisfiability_int_identifier_equal_to_int_literal_stays_decided() -> (
    None
):
    """Test EQUAL between an INT-sorted identifier and an int-bucket literal decides.

    Contrasts the hazard: the literal ``5`` is not float-bucketed, so no
    sort-mixing hazard exists and the comparison is handed to the solver.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.EQUAL, x, 5))
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_real_identifier_eq_float_literal_stays_decided() -> None:
    """Test EQUAL between a REAL-sorted identifier and a float literal decides.

    Contrasts the hazard: the identifier is not INT-sorted, so no
    sort-mixing hazard exists.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.EQUAL, x, 1.5))
    )

    outcome = system.check_satisfiability({x: SymbolType.REAL})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_int_identifier_lt_float_literal_not_screened() -> None:
    """Test `<` between an INT-sorted identifier and a float literal is NOT screened.

    Ordering comparisons are mathematically meaningful across a mixed
    int/float sort in a way `EQUAL`/`NOT_EQUAL` is not, so the screen
    deliberately does not cover them.
    """
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 1.5))
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


# =============================================================================
# `check_implication` (new; system-level entailment seam)
# =============================================================================


@pytest.mark.z3
def test_check_implication_proven_entailment_is_satisfied() -> None:
    """Test a provably-entailed consequent reports SATISFIED.

    Every assignment satisfying ``x in {1, 2}`` also satisfies
    ``x in {1, 2, 3}``.
    """
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {1, 2}))
    consequent = create_constraint_system(InSetConstraint(x, {1, 2, 3}))

    outcome = antecedent.check_implication(consequent, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_implication_reflexive_system_implies_itself() -> None:
    """Test a system trivially implies itself."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))
    )

    outcome = system.check_implication(system, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_implication_counterexample_is_violated() -> None:
    """Test a disprovable implication reports VIOLATED.

    ``x = 1`` satisfies ``x in {1, 2}`` but not ``x in {2, 3}``, so the
    implication does not hold for every assignment.
    """
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {1, 2}))
    consequent = create_constraint_system(InSetConstraint(x, {2, 3}))

    outcome = antecedent.check_implication(consequent, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_implication_transitive_ordering_is_satisfied() -> None:
    """Test `x < 10` implies `x < 100` for every INT assignment."""
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))
    )
    consequent = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 100))
    )

    outcome = antecedent.check_implication(consequent, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


def test_check_implication_bool_coercion_hazard_is_undecided() -> None:
    """Test a hazardous antecedent reports UNDECIDED instead of a decided outcome.

    ``x in {True}`` under an ``INT`` sort trips the bool-coercion screen
    exactly as it does for `check_satisfiability`.
    """
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {True}))
    consequent = create_constraint_system(InSetConstraint(x, {1}))

    outcome = antecedent.check_implication(consequent, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


def test_check_implication_logs_warning_for_a_hazardous_side(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the screen on `check_implication` names the hazardous node."""
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {True}))
    consequent = create_constraint_system(InSetConstraint(x, {1}))

    with caplog.at_level(logging.DEBUG, logger=_CONSTRAINT_LOGGER):
        outcome = antecedent.check_implication(consequent, {x: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED
    warnings = _find_records(caplog, logging.WARNING)
    assert warnings, "expected a WARNING naming the hazardous node"
    assert "check_implication" in warnings[0].getMessage()


@pytest.mark.z3
def test_check_implication_missing_symbol_type_raises_for_either_side() -> None:
    """Test a missing `symbol_types` entry for either side raises."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    antecedent = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, x, 10))
    )
    consequent = create_constraint_system(
        EquationConstraint(make_binary_expression(BinaryOperation.LESS, y, 10))
    )

    with pytest.raises(MissingSymbolTypeError, match=y.name_hint):
        antecedent.check_implication(consequent, {x: SymbolType.INT})


def test_check_implication_propagates_constraint_error_from_a_member() -> None:
    """Test a member that cannot be lowered raises `ConstraintError`."""
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {1}))
    consequent = create_constraint_system(
        InSetConstraint(x, {SerializableEqualHashable(1)})
    )

    with pytest.raises(ConstraintError):
        antecedent.check_implication(consequent, {x: SymbolType.INT})


@pytest.mark.parametrize("timeout_milliseconds", [-5, 0], ids=["negative", "zero"])
def test_check_implication_rejects_invalid_timeout_even_for_a_hazardous_pair(
    timeout_milliseconds: int,
) -> None:
    """Test timeout validation precedes the hazard screen's early return.

    Mirrors `check_satisfiability`'s ordering: an inadmissible bound is
    rejected even when the pair would otherwise be reported UNDECIDED
    without consulting the solver at all.
    """
    x = mock_identifier("x", 0)
    antecedent = create_constraint_system(InSetConstraint(x, {True}))
    consequent = create_constraint_system(InSetConstraint(x, {1}))

    with pytest.raises(ValueError):
        antecedent.check_implication(
            consequent, {x: SymbolType.INT}, timeout_milliseconds=timeout_milliseconds
        )
