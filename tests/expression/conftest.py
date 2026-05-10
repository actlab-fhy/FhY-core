"""Shared helpers for the `tests/expression` sub-package."""

from fhy_core.expression.passes.type_checker import ExpressionTypeChecker
from fhy_core.identifier import Identifier
from fhy_core.types import Type, TypeQualifier

from ..conftest import mock_identifier  # noqa: F401  # re-exported below

__all__ = [
    "mock_identifier",
    "make_identifier_checker",
    "make_single_type_checker",
]


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
