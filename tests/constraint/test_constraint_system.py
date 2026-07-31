"""Tests for `ConstraintSystem` and `create_constraint_system`."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st

from fhy_core.constraint import (
    Constraint,
    ConstraintBindings,
    ConstraintError,
    ConstraintOutcome,
    ConstraintSystem,
    EquationConstraint,
    InSetConstraint,
    create_constraint_system,
)
from fhy_core.expression import (
    BinaryExpression,
    BinaryOperation,
    Expression,
    IdentifierExpression,
    LiteralExpression,
    make_binary_expression,
)
from fhy_core.identifier import Identifier
from fhy_core.serialization import (
    DeserializationDictStructureError,
    DeserializationValueError,
    FieldCodec,
    SerializationFormat,
    SerializationValueError,
    make_field_codec,
)
from fhy_core.symbol_type import SymbolType
from fhy_core.traits import Frozen, FrozenMutationError
from fhy_core.traits.derived_equivalence import (
    compared_as_reference,
    compared_as_value,
    excluded_from_equivalence,
)
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
    # ``fhy_core.constraint``'s namespace.
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
def test_check_satisfiability_raises_key_error_for_missing_symbol_type() -> None:
    """Test a missing `symbol_types` entry propagates `KeyError`."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    system = create_constraint_system(
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y))
    )

    with pytest.raises(KeyError):
        system.check_satisfiability({x: SymbolType.INT})


def test_check_satisfiability_empty_system_does_not_invoke_the_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the empty system short-circuits to SATISFIED without calling z3."""

    def _fail_if_called(*args: object, **kwargs: object) -> bool | None:
        raise AssertionError("does_expression_imply must not be called")

    monkeypatch.setattr("fhy_core.constraint.does_expression_imply", _fail_if_called)
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


# =============================================================================
# Property-based tests
# =============================================================================


@pytest.mark.property
@given(  # type: ignore[untyped-decorator]
    x_value=st.integers(min_value=-5, max_value=10),
    y_value=st.integers(min_value=-5, max_value=10),
)
def test_evaluate_with_bindings_matches_fold_of_member_outcomes(
    x_value: int, y_value: int
) -> None:
    """Test the conjunction outcome matches folding each member's own outcome."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    members: tuple[Constraint, ...] = (
        InSetConstraint(x, {1, 2, 3, 4}),
        InSetConstraint(y, {0, 1, 2}),
        EquationConstraint(x, make_binary_expression(BinaryOperation.LESS, x, y)),
    )
    system = create_constraint_system(*members)
    bindings = {x: x_value, y: y_value}

    outcome = system.evaluate_with_bindings(bindings)

    member_outcomes = [member.evaluate_with_bindings(bindings) for member in members]
    if any(o is ConstraintOutcome.VIOLATED for o in member_outcomes):
        expected = ConstraintOutcome.VIOLATED
    elif all(o is ConstraintOutcome.SATISFIED for o in member_outcomes):
        expected = ConstraintOutcome.SATISFIED
    else:
        expected = ConstraintOutcome.UNDECIDED
    assert outcome is expected


@pytest.mark.z3
@pytest.mark.property
@given(  # type: ignore[untyped-decorator]
    threshold=st.integers(min_value=0, max_value=10)
)
def test_check_satisfiability_matches_brute_force_enumeration(threshold: int) -> None:
    """Test z3-backed satisfiability agrees with brute-force enumeration."""
    x = mock_identifier("x", 0)
    y = mock_identifier("y", 1)
    domain = tuple(range(6))
    link_expression = make_binary_expression(
        BinaryOperation.EQUAL,
        make_binary_expression(BinaryOperation.ADD, x, threshold),
        y,
    )
    system = create_constraint_system(
        InSetConstraint(x, set(domain)),
        InSetConstraint(y, set(domain)),
        EquationConstraint(x, link_expression),
    )

    brute_force_satisfiable = any(a + threshold == b for a in domain for b in domain)

    outcome = system.check_satisfiability({x: SymbolType.INT, y: SymbolType.INT})

    expected = (
        ConstraintOutcome.SATISFIED
        if brute_force_satisfiable
        else ConstraintOutcome.VIOLATED
    )
    assert outcome is expected
