"""Shared helpers for the `tests/expression` sub-package."""

from collections.abc import Iterator

import pytest

from fhy_core.expression import registry as _registry
from fhy_core.expression.passes.type_checker import ExpressionTypeChecker
from fhy_core.identifier import Identifier
from fhy_core.types import Type, TypeQualifier

from ..conftest import mock_identifier  # noqa: F401  # re-exported below

__all__ = [
    "mock_identifier",
    "make_identifier_checker",
    "make_single_type_checker",
]


@pytest.fixture()
def function_registry_snapshot() -> Iterator[None]:
    """Snapshot the process-wide function registry around the test.

    Captures the registry's contents before the test runs, then restores
    them after the test completes. Tests that mutate the registry
    request this fixture explicitly. Built-in registrations
    (``max``, ``min``) survive across tests because they are in the
    snapshot.
    """
    snapshot = dict(_registry.get_registered_functions())
    try:
        yield
    finally:
        _registry._set_registry_state_for_tests(snapshot)


def _unexpected_lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
    raise AssertionError(f"Unexpected identifier lookup: {identifier}")


def make_identifier_checker(
    bindings: dict[Identifier, tuple[Type, TypeQualifier]],
) -> ExpressionTypeChecker:
    """Build an `ExpressionTypeChecker` whose lookup is driven by `bindings`.

    Unknown identifiers raise `AssertionError` so unintended lookups surface
    as test failures rather than silent defaults.

    """

    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        if identifier in bindings:
            return bindings[identifier]
        return _unexpected_lookup(identifier)

    return ExpressionTypeChecker(lookup)


def make_single_type_checker(
    result_type: Type,
    qualifier: TypeQualifier = TypeQualifier.PARAM,
) -> ExpressionTypeChecker:
    """Build an `ExpressionTypeChecker` that returns `(result_type, qualifier)`
    for every identifier lookup.
    """
    constant_result: tuple[Type, TypeQualifier] = (result_type, qualifier)

    def lookup(_: Identifier) -> tuple[Type, TypeQualifier]:
        return constant_result

    return ExpressionTypeChecker(lookup)
