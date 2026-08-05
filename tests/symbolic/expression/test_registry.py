"""Tests for the process-wide expression-IR registry.

The registry holds three kinds of entries:

- ``RegisteredFunction``: expression-bodied function with declared
  parameter and result sorts.
- ``NativeFunction``: Python-backed function with declared
  parameter and result sorts.
- ``NativeConstant``: named Python literal value with a declared
  sort.

Lookup helpers (``get_registered_entry``, ``get_registered_entries``,
``is_entry_registered``) widen to ``RegisteredEntry``; callers that
need to distinguish kinds use ``isinstance``.
"""

import dataclasses
import math

import pytest

from fhy_core.symbolic.expression import (
    BUILTIN_CONSTANTS,
    EntryLookupError,
    EntryRegistrationError,
    FunctionSort,
    IdentifierExpression,
    LiteralExpression,
    NativeConstant,
    NativeFunction,
    RegisteredFunction,
    get_registered_entries,
    get_registered_entry,
    is_entry_registered,
    register_function,
    register_native_constant,
    register_native_function,
)

from .conftest import mock_identifier

# =============================================================================
# register_function: happy paths
# =============================================================================


def test_register_function_stores_name_parameters_and_body(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` records the supplied name, parameters, and body."""
    parameter = mock_identifier("x", 0)
    body = IdentifierExpression(parameter)

    registered = register_function(
        "test_identity",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=body,
    )

    assert registered.name == "test_identity"
    assert registered.parameters == (parameter,)
    assert registered.body is body


def test_register_function_records_parameter_sorts_and_result_sort(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` records the declared sorts on the stored entry."""
    parameter = mock_identifier("x", 0)
    registered = register_function(
        "test_sort_fields",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.NAT],
        result_sort=FunctionSort.INT,
        body=IdentifierExpression(parameter),
    )

    assert registered.parameter_sorts == (FunctionSort.NAT,)
    assert registered.result_sort == FunctionSort.INT


def test_register_function_returns_a_registered_function_instance(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` returns a ``RegisteredFunction``."""
    parameter = mock_identifier("x", 0)

    registered = register_function(
        "test_returns_instance",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    assert isinstance(registered, RegisteredFunction)


def test_register_function_with_multiple_parameters_records_order(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` preserves the parameter order on registration."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)
    c = mock_identifier("c", 2)

    registered = register_function(
        "test_three_params",
        parameters=[a, b, c],
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(a) + b + c,
    )

    assert registered.parameters == (a, b, c)


def test_registered_function_dataclass_is_frozen(
    function_registry_snapshot: None,
) -> None:
    """Test ``RegisteredFunction`` instances reject attribute mutation."""
    parameter = mock_identifier("x", 0)
    registered = register_function(
        "test_frozen",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        registered.name = "renamed"  # type: ignore[misc]


# =============================================================================
# register_function: rejection paths
# =============================================================================


def test_register_function_rejects_duplicate_name(
    function_registry_snapshot: None,
) -> None:
    """Test re-registering an existing name raises ``EntryRegistrationError``."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_duplicate",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    with pytest.raises(EntryRegistrationError, match="test_duplicate"):
        register_function(
            "test_duplicate",
            parameters=[parameter],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(parameter),
        )


def test_register_function_rejects_captured_free_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test a body referencing identifiers outside the parameter list is rejected."""
    parameter = mock_identifier("x", 0)
    captured = mock_identifier("y", 1)

    with pytest.raises(EntryRegistrationError, match="test_captured"):
        register_function(
            "test_captured",
            parameters=[parameter],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(parameter) + captured,
        )


def test_register_function_lists_multiple_captured_identifiers_in_sorted_order(
    function_registry_snapshot: None,
) -> None:
    """Test the captured-identifier error message lists every name, sorted.

    Pins the sort and the comma-join so that error messages are stable
    when a body captures more than one free identifier.
    """
    parameter = mock_identifier("a", 0)
    captured_z = mock_identifier("z", 1)
    captured_y = mock_identifier("y", 2)
    captured_x = mock_identifier("x", 3)

    with pytest.raises(EntryRegistrationError, match=r"x, y, z"):
        register_function(
            "test_captured_multiple",
            parameters=[parameter],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(parameter) + captured_z + captured_y + captured_x,
        )


def test_register_function_accepts_subset_of_parameters_used_in_body(
    function_registry_snapshot: None,
) -> None:
    """Test parameters declared but unused in the body are still accepted.

    Free-identifier validation is a subset relation: every body identifier
    must be a declared parameter, but declared parameters may go unused.
    """
    used = mock_identifier("a", 0)
    unused = mock_identifier("b", 1)

    registered = register_function(
        "test_unused_param",
        parameters=[used, unused],
        parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(used),
    )

    assert registered.parameters == (used, unused)


def test_register_function_accepts_literal_only_body(
    function_registry_snapshot: None,
) -> None:
    """Test a body with no identifiers is accepted (vacuously satisfies subset)."""
    registered = register_function(
        "test_literal_only_body",
        parameters=[],
        parameter_sorts=[],
        result_sort=FunctionSort.REAL,
        body=LiteralExpression(0),
    )

    assert registered.body.is_structurally_equivalent(LiteralExpression(0))


def test_register_function_rejects_sort_arity_mismatch(
    function_registry_snapshot: None,
) -> None:
    """Test a parameter / parameter-sort length mismatch is rejected."""
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)

    with pytest.raises(EntryRegistrationError, match="test_sort_arity_mismatch"):
        register_function(
            "test_sort_arity_mismatch",
            parameters=[a, b],
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(a) + b,
        )


def test_register_function_rejects_body_result_sort_mismatch(
    function_registry_snapshot: None,
) -> None:
    """Test a body whose type is incompatible with ``result_sort`` is rejected.

    Here the body synthesizes a boolean type (``a < b``), but the declared
    ``result_sort`` is ``REAL``: registration must fail because the boolean
    core data type is not compatible with the ``REAL`` sort.
    """
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)

    with pytest.raises(EntryRegistrationError, match="test_body_result_mismatch"):
        register_function(
            "test_body_result_mismatch",
            parameters=[a, b],
            parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(a) < b,
        )


def test_register_function_rolls_back_placeholder_on_body_check_failure(
    function_registry_snapshot: None,
) -> None:
    """Test a failed body type-check removes the placeholder registration.

    ``register_function`` inserts a placeholder entry under the
    declared sorts before running the body type-check (so a
    self-recursive body can resolve its own call). When the body
    check fails, the placeholder must be removed; otherwise a
    subsequent ``register_function`` call with the same name would
    fail with a spurious "already registered" error.
    """
    a = mock_identifier("a", 0)
    b = mock_identifier("b", 1)

    with pytest.raises(EntryRegistrationError):
        register_function(
            "test_rollback",
            parameters=[a, b],
            parameter_sorts=[FunctionSort.REAL, FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            body=IdentifierExpression(a) < b,
        )

    assert not is_entry_registered("test_rollback")
    with pytest.raises(EntryLookupError):
        get_registered_entry("test_rollback")


def test_register_function_accepts_body_referencing_registered_constant(
    function_registry_snapshot: None,
) -> None:
    """Test a body identifier whose name matches a registered constant is not captured.

    ``pi`` is registered as a built-in constant; an identifier named
    ``pi`` in a body must therefore be treated as a constant reference,
    not a captured free identifier.
    """
    x = mock_identifier("x", 0)
    pi = mock_identifier("pi", 1)

    # Should not raise: ``pi`` matches the registered native constant.
    registered = register_function(
        "test_body_uses_pi",
        parameters=[x],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(x) * pi,
    )

    assert registered.name == "test_body_uses_pi"


# =============================================================================
# Lookup, listing, and presence (RegisteredFunction)
# =============================================================================


def test_get_registered_entry_returns_previously_registered_entry(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_entry`` returns the same record as registration."""
    parameter = mock_identifier("x", 0)
    expected = register_function(
        "test_lookup",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    fetched = get_registered_entry("test_lookup")

    assert fetched is expected or fetched == expected


def test_get_registered_entry_raises_for_unknown_name(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_entry`` raises ``EntryLookupError`` for unknowns."""
    with pytest.raises(EntryLookupError, match="never_registered"):
        get_registered_entry("never_registered")


def test_is_entry_registered_true_after_registration(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_entry_registered`` returns True for a registered name."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_is_registered_true",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    assert is_entry_registered("test_is_registered_true") is True


def test_is_entry_registered_false_for_unknown_name(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_entry_registered`` returns False for an unknown name."""
    assert is_entry_registered("never_registered") is False


def test_get_registered_entries_includes_registered_entry(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_entries`` snapshots include newly registered entries."""
    parameter = mock_identifier("x", 0)
    expected = register_function(
        "test_snapshot_includes",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    snapshot = get_registered_entries()

    assert "test_snapshot_includes" in snapshot
    fetched = snapshot["test_snapshot_includes"]
    assert fetched == expected or fetched is expected


def test_get_registered_entries_returns_immutable_snapshot(
    function_registry_snapshot: None,
) -> None:
    """Test mutating the snapshot does not affect the registry."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_snapshot_immutable",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    snapshot = get_registered_entries()
    with pytest.raises(TypeError):
        snapshot["test_snapshot_immutable"] = None  # type: ignore[index]


# =============================================================================
# Registry isolation (the conftest fixture)
# =============================================================================


def test_function_registry_snapshot_restores_state_after_test_a(
    function_registry_snapshot: None,
) -> None:
    """Test the snapshot fixture leaves the registry clean for the sibling test."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_isolation_marker",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    assert is_entry_registered("test_isolation_marker") is True


def test_function_registry_snapshot_restores_state_after_test_b(
    function_registry_snapshot: None,
) -> None:
    """Test the marker registered in the sibling test does not leak here."""
    assert is_entry_registered("test_isolation_marker") is False


# =============================================================================
# NativeFunction dataclass
# =============================================================================


def test_native_function_constructs_with_declared_sorts_and_implementation() -> None:
    """Test ``NativeFunction`` stores the supplied fields."""
    native = NativeFunction(
        name="test_native_fn",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    assert native.name == "test_native_fn"
    assert native.parameter_sorts == (FunctionSort.REAL,)
    assert native.result_sort == FunctionSort.REAL
    assert native.implementation is math.sqrt


def test_native_function_dataclass_is_frozen() -> None:
    """Test ``NativeFunction`` instances reject attribute mutation."""
    native = NativeFunction(
        name="test_native_frozen",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        native.name = "renamed"  # type: ignore[misc]


def test_native_function_equality_compares_all_fields() -> None:
    """Test two ``NativeFunction`` with the same fields compare equal."""
    a = NativeFunction(
        name="test_eq",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )
    b = NativeFunction(
        name="test_eq",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    assert a == b


def test_native_function_inequality_when_name_differs() -> None:
    """Test two ``NativeFunction`` with different names compare unequal."""
    a = NativeFunction(
        name="a",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )
    b = NativeFunction(
        name="b",
        parameter_sorts=(FunctionSort.REAL,),
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    assert a != b


# =============================================================================
# NativeConstant dataclass
# =============================================================================


def test_native_constant_constructs_with_name_sort_and_value() -> None:
    """Test ``NativeConstant`` stores the supplied fields."""
    constant = NativeConstant(name="test_const", sort=FunctionSort.REAL, value=2.5)

    assert constant.name == "test_const"
    assert constant.sort == FunctionSort.REAL
    assert constant.value == 2.5


def test_native_constant_dataclass_is_frozen() -> None:
    """Test ``NativeConstant`` instances reject attribute mutation."""
    constant = NativeConstant(
        name="test_const_frozen", sort=FunctionSort.REAL, value=1.0
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        constant.name = "renamed"  # type: ignore[misc]


def test_native_constant_equality_compares_all_fields() -> None:
    """Test two ``NativeConstant`` with the same fields compare equal."""
    a = NativeConstant(name="c", sort=FunctionSort.REAL, value=1.0)
    b = NativeConstant(name="c", sort=FunctionSort.REAL, value=1.0)

    assert a == b


def test_native_constant_inequality_when_value_differs() -> None:
    """Test two ``NativeConstant`` with different values compare unequal."""
    a = NativeConstant(name="c", sort=FunctionSort.REAL, value=1.0)
    b = NativeConstant(name="c", sort=FunctionSort.REAL, value=2.0)

    assert a != b


# =============================================================================
# register_native_function
# =============================================================================


def test_register_native_function_stores_supplied_fields(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_native_function`` records the supplied fields."""
    registered = register_native_function(
        "test_native_register",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    assert registered.name == "test_native_register"
    assert registered.parameter_sorts == (FunctionSort.REAL,)
    assert registered.result_sort == FunctionSort.REAL
    assert registered.implementation is math.sqrt


def test_register_native_function_returns_native_function_instance(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_native_function`` returns a ``NativeFunction``."""
    registered = register_native_function(
        "test_native_instance",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.exp,
    )

    assert isinstance(registered, NativeFunction)


def test_register_native_function_rejects_duplicate_name(
    function_registry_snapshot: None,
) -> None:
    """Test re-registering an existing name raises ``EntryRegistrationError``."""
    register_native_function(
        "test_native_dup",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    with pytest.raises(EntryRegistrationError, match="test_native_dup"):
        register_native_function(
            "test_native_dup",
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            implementation=math.exp,
        )


def test_register_native_function_rejects_collision_with_registered_function(
    function_registry_snapshot: None,
) -> None:
    """Test a native registration cannot reuse an expression-bodied function name."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_native_collides_with_function",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    with pytest.raises(
        EntryRegistrationError, match="test_native_collides_with_function"
    ):
        register_native_function(
            "test_native_collides_with_function",
            parameter_sorts=[FunctionSort.REAL],
            result_sort=FunctionSort.REAL,
            implementation=math.sqrt,
        )


# =============================================================================
# register_native_constant
# =============================================================================


def test_register_native_constant_stores_supplied_fields(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_native_constant`` records the supplied fields."""
    registered = register_native_constant(
        "test_const_register", sort=FunctionSort.REAL, value=math.pi
    )

    assert registered.name == "test_const_register"
    assert registered.sort == FunctionSort.REAL
    assert registered.value == math.pi


def test_register_native_constant_returns_native_constant_instance(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_native_constant`` returns a ``NativeConstant``."""
    registered = register_native_constant(
        "test_const_instance", sort=FunctionSort.REAL, value=1.0
    )

    assert isinstance(registered, NativeConstant)


def test_register_native_constant_rejects_duplicate_name(
    function_registry_snapshot: None,
) -> None:
    """Test re-registering an existing constant name raises."""
    register_native_constant("test_const_dup", sort=FunctionSort.REAL, value=1.0)

    with pytest.raises(EntryRegistrationError, match="test_const_dup"):
        register_native_constant("test_const_dup", sort=FunctionSort.REAL, value=2.0)


def test_register_native_constant_rejects_sort_value_incompatibility(
    function_registry_snapshot: None,
) -> None:
    """Test a constant whose value is incompatible with the sort is rejected."""
    with pytest.raises(EntryRegistrationError, match="test_const_sort_mismatch"):
        register_native_constant(
            "test_const_sort_mismatch",
            sort=FunctionSort.BOOL,
            value=1.5,
        )


def test_register_native_constant_rejects_bool_for_int_sort(
    function_registry_snapshot: None,
) -> None:
    """Test a constant value of ``True`` is rejected for ``INT`` sort.

    Pins down the strict-``bool`` rule against a permissive
    ``isinstance(value, int)`` registration check.
    """
    with pytest.raises(EntryRegistrationError, match="test_const_bool_for_int"):
        register_native_constant(
            "test_const_bool_for_int",
            sort=FunctionSort.INT,
            value=True,
        )


def test_register_native_constant_rejects_collision_with_function(
    function_registry_snapshot: None,
) -> None:
    """Test a constant registration cannot reuse an existing function name."""
    parameter = mock_identifier("x", 0)
    register_function(
        "test_const_collides_with_function",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )

    with pytest.raises(
        EntryRegistrationError, match="test_const_collides_with_function"
    ):
        register_native_constant(
            "test_const_collides_with_function",
            sort=FunctionSort.REAL,
            value=1.0,
        )


# =============================================================================
# Widened lookup helpers (RegisteredEntry union)
# =============================================================================


def test_get_registered_entry_returns_native_function_when_registered(
    function_registry_snapshot: None,
) -> None:
    """Test the lookup helper returns the stored ``NativeFunction`` instance."""
    native = register_native_function(
        "test_lookup_native",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )

    fetched = get_registered_entry("test_lookup_native")

    assert fetched is native or fetched == native


def test_get_registered_entry_returns_native_constant_when_registered(
    function_registry_snapshot: None,
) -> None:
    """Test the lookup helper returns the stored ``NativeConstant`` instance."""
    constant = register_native_constant(
        "test_lookup_const", sort=FunctionSort.REAL, value=1.0
    )

    fetched = get_registered_entry("test_lookup_const")

    assert fetched is constant or fetched == constant


def test_is_entry_registered_true_for_native_function(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_entry_registered`` returns True for a registered native function."""
    register_native_function(
        "test_present_native",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.exp,
    )

    assert is_entry_registered("test_present_native") is True


def test_is_entry_registered_true_for_native_constant(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_entry_registered`` returns True for a registered constant."""
    register_native_constant("test_present_const", sort=FunctionSort.REAL, value=math.e)

    assert is_entry_registered("test_present_const") is True


def test_get_registered_entries_snapshot_includes_all_entry_kinds(
    function_registry_snapshot: None,
) -> None:
    """Test the snapshot mapping holds the union of all three entry kinds."""
    parameter = mock_identifier("x", 0)
    expression_function = register_function(
        "test_snapshot_function",
        parameters=[parameter],
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        body=IdentifierExpression(parameter),
    )
    native_function = register_native_function(
        "test_snapshot_native",
        parameter_sorts=[FunctionSort.REAL],
        result_sort=FunctionSort.REAL,
        implementation=math.sqrt,
    )
    constant = register_native_constant(
        "test_snapshot_const", sort=FunctionSort.REAL, value=1.0
    )

    snapshot = get_registered_entries()

    assert snapshot.get("test_snapshot_function") == expression_function
    assert snapshot.get("test_snapshot_native") == native_function
    assert snapshot.get("test_snapshot_const") == constant


# =============================================================================
# Built-in constants registered at import time
# =============================================================================


def test_pi_is_registered_at_import_time() -> None:
    """Test the ``pi`` constant is in the registry at package-import time."""
    assert is_entry_registered("pi") is True


def test_pi_lookup_returns_a_native_constant() -> None:
    """Test looking up ``pi`` returns a ``NativeConstant`` carrying ``math.pi``."""
    fetched = get_registered_entry("pi")

    assert isinstance(fetched, NativeConstant)
    assert fetched.value == math.pi


def test_builtin_constants_mapping_covers_seeded_constants() -> None:
    """Test ``BUILTIN_CONSTANTS`` exposes the canonical seeded constants."""
    assert set(BUILTIN_CONSTANTS.keys()) >= {"pi", "e", "inf", "nan"}
