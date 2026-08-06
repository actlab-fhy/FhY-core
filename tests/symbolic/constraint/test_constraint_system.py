"""Tests for `ConstraintSystem` and `create_constraint_system`."""

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
    def evaluate(self, value: Any) -> ConstraintOutcome:
        return self.on_evaluate_with_bindings({self.variable: value})

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


def test_constraint_system_canonicalizes_alike_for_independent_identifiers() -> None:
    """Test independently constructed, content-identical systems canonicalize alike.

    Two separate ``mock_identifier`` calls with the same name and id are
    distinct ``Mock`` objects, but ``ConstraintSystem`` orders its members
    by ``repr``. A deterministic, content-based mock ``repr`` is required
    for two independently built systems over "the same" identifiers, given
    in different construction order, to canonicalize to the same member
    order.
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
        InSetConstraint(x, {1, 2}), EquationConstraint(y, IdentifierExpression(z))
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
        InSetConstraint(x, {1, 2}), EquationConstraint(y, LiteralExpression(True))
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


def test_wire_members_are_emitted_in_repr_sorted_order() -> None:
    """Test serialized members are emitted in the same order as `constraints`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2}), EquationConstraint(y, LiteralExpression(True))
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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(y, make_binary_expression(BinaryOperation.LESS, y, z)),
        EquationConstraint(z, make_binary_expression(BinaryOperation.LESS, z, x)),
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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_mixed_set_and_equation_system_is_unsatisfiable() -> None:
    """Test a mixed set-and-equation system can be jointly unsatisfiable."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(
        InSetConstraint(x, {1, 2, 3}),
        EquationConstraint(x, make_binary_expression(BinaryOperation.GREATER, x, 100)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_satisfiability_raises_missing_symbol_type_error() -> None:
    """Test a missing `symbol_types` entry raises `MissingSymbolTypeError` naming it."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
    )

    with pytest.raises(MissingSymbolTypeError, match=y.name_hint):
        system.check_satisfiability({x: SymbolType.INT})


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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
    )

    with pytest.raises(MissingSymbolTypeError, match=y.name_hint):
        system.check_satisfiability_with_bindings({x: 5}, {})


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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
    )

    outcome = system.evaluate_with_bindings({x: 1})

    assert outcome is ConstraintOutcome.UNDECIDED


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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(y, make_binary_expression(BinaryOperation.LESS, y, 3)),
    )

    outcome = system.check_satisfiability_with_bindings({x: 5}, {y: SymbolType.INT})

    assert outcome is ConstraintOutcome.VIOLATED


@pytest.mark.z3
def test_check_satisfiability_with_bindings_satisfiable_after_substitution() -> None:
    """Test a partial assignment can leave the residual system satisfiable."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
        EquationConstraint(y, make_binary_expression(BinaryOperation.LESS, y, 30)),
    )

    outcome = system.check_satisfiability_with_bindings({x: 5}, {y: SymbolType.INT})

    assert outcome is ConstraintOutcome.SATISFIED


@pytest.mark.z3
def test_check_satisfiability_with_closed_conjunction_needs_no_symbol_types() -> None:
    """Test a system whose expression has no free identifiers needs no symbol types."""
    x = mock_identifier("x", 0)
    system = create_constraint_system(EquationConstraint(x, LiteralExpression(True)))

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
    system = create_constraint_system(EquationConstraint(x, LiteralExpression(True)))

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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, 5))
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
    itself (``constraint.evaluate(True)`` is SATISFIED), which is not an
    ``int`` at all, and every genuine ``int`` -- including ``1``, the
    value Z3's coercion identifies ``True`` with -- provably VIOLATES it
    under the package's type-strict semantics (``constraint.evaluate(1)``
    is VIOLATED). The Z3 bridge does not see this distinction: it lowers
    the bool literal ``True`` to the integer ``1`` and would spuriously
    decide the system SATISFIED at ``x = 1``, contradicting
    ``evaluate(1)``. Both satisfiability APIs must report UNDECIDED
    instead of that provably wrong decided outcome.
    """
    x = mock_identifier("x", 0)
    constraint = EquationConstraint(
        x,
        make_binary_expression(
            BinaryOperation.EQUAL, IdentifierExpression(x), LiteralExpression(True)
        ),
    )
    system = create_constraint_system(constraint)

    assert constraint.evaluate(1) is ConstraintOutcome.VIOLATED
    assert constraint.evaluate(True) is ConstraintOutcome.SATISFIED

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
        x,
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
        EquationConstraint(y, make_binary_expression(BinaryOperation.LESS, y, 5)),
    )

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    assert outcome is ConstraintOutcome.UNDECIDED


# =============================================================================
# `timeout_milliseconds` passthrough (z3-backed)
# =============================================================================


@pytest.mark.z3
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


@pytest.mark.z3
def test_check_satisfiability_with_bindings_forwards_timeout_milliseconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test `check_satisfiability_with_bindings` forwards `timeout_milliseconds`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
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


@pytest.mark.z3
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


@pytest.mark.z3
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
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
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
