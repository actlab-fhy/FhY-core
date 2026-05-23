"""Tests for the process-wide function registry."""

import dataclasses

import pytest

from fhy_core.expression import (
    FunctionLookupError,
    FunctionRegistrationError,
    IdentifierExpression,
    LiteralExpression,
    RegisteredFunction,
    get_registered_function,
    get_registered_functions,
    is_function_registered,
    register_function,
)
from fhy_core.identifier import Identifier

# =============================================================================
# register_function: happy paths
# =============================================================================


def test_register_function_stores_name_parameters_and_body(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` records the supplied name, parameters, and body."""
    parameter = Identifier("x")
    body = IdentifierExpression(parameter)

    registered = register_function("test_identity", parameters=[parameter], body=body)

    assert registered.name == "test_identity"
    assert registered.parameters == (parameter,)
    assert registered.body is body


def test_register_function_returns_a_registered_function_instance(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` returns a ``RegisteredFunction``."""
    parameter = Identifier("x")

    registered = register_function(
        "test_returns_instance",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    assert isinstance(registered, RegisteredFunction)


def test_register_function_with_multiple_parameters_records_order(
    function_registry_snapshot: None,
) -> None:
    """Test ``register_function`` preserves the parameter order on registration."""
    a = Identifier("a")
    b = Identifier("b")
    c = Identifier("c")

    registered = register_function(
        "test_three_params",
        parameters=[a, b, c],
        body=IdentifierExpression(a)
        + IdentifierExpression(b)
        + IdentifierExpression(c),
    )

    assert registered.parameters == (a, b, c)


def test_registered_function_dataclass_is_frozen(
    function_registry_snapshot: None,
) -> None:
    """Test ``RegisteredFunction`` instances reject attribute mutation."""
    parameter = Identifier("x")
    registered = register_function(
        "test_frozen",
        parameters=[parameter],
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
    """Test re-registering an existing name raises ``FunctionRegistrationError``."""
    parameter = Identifier("x")
    register_function(
        "test_duplicate",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    with pytest.raises(FunctionRegistrationError, match="test_duplicate"):
        register_function(
            "test_duplicate",
            parameters=[parameter],
            body=IdentifierExpression(parameter),
        )


def test_register_function_rejects_captured_free_identifier(
    function_registry_snapshot: None,
) -> None:
    """Test a body referencing identifiers outside the parameter list is rejected."""
    parameter = Identifier("x")
    captured = Identifier("y")

    with pytest.raises(FunctionRegistrationError):
        register_function(
            "test_captured",
            parameters=[parameter],
            body=IdentifierExpression(parameter) + IdentifierExpression(captured),
        )


def test_register_function_accepts_subset_of_parameters_used_in_body(
    function_registry_snapshot: None,
) -> None:
    """Test parameters declared but unused in the body are still accepted.

    Free-identifier validation is a subset relation: every body identifier
    must be a declared parameter, but declared parameters may go unused.
    """
    used = Identifier("a")
    unused = Identifier("b")

    registered = register_function(
        "test_unused_param",
        parameters=[used, unused],
        body=IdentifierExpression(used),
    )

    assert registered.parameters == (used, unused)


def test_register_function_accepts_literal_only_body(
    function_registry_snapshot: None,
) -> None:
    """Test a body with no identifiers is accepted (vacuously satisfies subset)."""
    registered = register_function(
        "test_literal_only_body", parameters=[], body=LiteralExpression(0)
    )

    assert registered.body.is_structurally_equivalent(LiteralExpression(0))


# =============================================================================
# Lookup, listing, and presence
# =============================================================================


def test_get_registered_function_returns_previously_registered_entry(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_function`` returns the same record as registration."""
    parameter = Identifier("x")
    expected = register_function(
        "test_lookup",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    fetched = get_registered_function("test_lookup")

    assert fetched is expected or fetched == expected


def test_get_registered_function_raises_for_unknown_name(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_function`` raises ``FunctionLookupError`` for unknowns."""
    with pytest.raises(FunctionLookupError, match="never_registered"):
        get_registered_function("never_registered")


def test_is_function_registered_true_after_registration(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_function_registered`` returns True for a registered name."""
    parameter = Identifier("x")
    register_function(
        "test_is_registered_true",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    assert is_function_registered("test_is_registered_true") is True


def test_is_function_registered_false_for_unknown_name(
    function_registry_snapshot: None,
) -> None:
    """Test ``is_function_registered`` returns False for an unknown name."""
    assert is_function_registered("never_registered") is False


def test_get_registered_functions_includes_registered_entry(
    function_registry_snapshot: None,
) -> None:
    """Test ``get_registered_functions`` snapshots include newly registered entries."""
    parameter = Identifier("x")
    expected = register_function(
        "test_snapshot_includes",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    snapshot = get_registered_functions()

    assert "test_snapshot_includes" in snapshot
    fetched = snapshot["test_snapshot_includes"]
    assert fetched == expected or fetched is expected


def test_get_registered_functions_returns_immutable_snapshot(
    function_registry_snapshot: None,
) -> None:
    """Test mutating the snapshot does not affect the registry."""
    parameter = Identifier("x")
    register_function(
        "test_snapshot_immutable",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    snapshot = get_registered_functions()
    with pytest.raises(TypeError):
        snapshot["test_snapshot_immutable"] = None  # type: ignore[index]


# =============================================================================
# Registry isolation (the conftest fixture)
# =============================================================================


def test_function_registry_snapshot_restores_state_after_test_a(
    function_registry_snapshot: None,
) -> None:
    """Test the snapshot fixture leaves the registry clean for the sibling test."""
    parameter = Identifier("x")
    register_function(
        "test_isolation_marker",
        parameters=[parameter],
        body=IdentifierExpression(parameter),
    )

    assert is_function_registered("test_isolation_marker") is True


def test_function_registry_snapshot_restores_state_after_test_b(
    function_registry_snapshot: None,
) -> None:
    """Test the marker registered in the sibling test does not leak here."""
    assert is_function_registered("test_isolation_marker") is False
