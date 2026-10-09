"""Shared helpers for the `tests/types/checking` sub-package."""

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression import registry as _registry
from fhy_core.types import Type, TypeQualifier
from fhy_core.types.checking import ExpressionTypeChecker

from ...conftest import mock_identifier  # re-exported below

__all__ = [
    "make_identifier_checker",
    "make_single_type_checker",
    "mock_identifier",
]


def make_identifier_checker(
    bindings: dict[Identifier, tuple[Type, TypeQualifier]],
) -> ExpressionTypeChecker:
    """Build an `ExpressionTypeChecker` whose lookup is driven by `bindings`.

    Unknown identifiers raise `KeyError` so the type checker can fall
    back to the registered-constant resolver and, failing that, frame
    the failure as an "identifier is not bound" type error.

    """

    def lookup(identifier: Identifier) -> tuple[Type, TypeQualifier]:
        if identifier in bindings:
            return bindings[identifier]
        raise KeyError(identifier.name_hint)

    return ExpressionTypeChecker(
        lookup, resolve_call_target=_registry.get_registered_entry
    )


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

    return ExpressionTypeChecker(
        lookup, resolve_call_target=_registry.get_registered_entry
    )
