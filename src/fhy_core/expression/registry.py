"""Process-wide registry of pure functions over the expression IR.

A registered function is named, positional, and has a body expression
written in the existing expression IR. Because the IR contains only
pure node kinds, every registered function is pure by construction; the
registry does not need a separate purity check.

The registry is process-wide. Tests that need isolation should request
the ``function_registry_snapshot`` fixture defined in
``tests/expression/conftest.py`` rather than mutating the registry
directly.
"""

__all__ = [
    "FunctionLookupError",
    "FunctionRegistrationError",
    "RegisteredFunction",
    "get_registered_function",
    "get_registered_functions",
    "is_function_registered",
    "register_function",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock

from frozendict import frozendict

from fhy_core.error import register_error
from fhy_core.identifier import Identifier

from .core import Expression
from .passes.basic import collect_identifiers


@register_error
class FunctionRegistrationError(RuntimeError):
    """Raised when a function cannot be registered.

    Possible causes:

    - A function is already registered under the same name.
    - The body references free identifiers that are not declared as
      parameters.
    """


@register_error
class FunctionLookupError(KeyError):
    """Raised when a function is requested but not registered."""


@dataclass(frozen=True)
class RegisteredFunction:
    """A named pure function over the expression IR.

    Attributes:
        name: Registry key. Used at call sites and in error messages.
        parameters: Ordered formal-parameter identifiers. Inlining
            substitutes these with the call's argument expressions.
        body: Expression tree that uses only the parameter identifiers
            (and possibly references other registered functions). The
            body's free identifiers are a subset of ``parameters``.

    """

    name: str
    parameters: tuple[Identifier, ...]
    body: Expression


_REGISTRY: dict[str, RegisteredFunction] = {}
_REGISTRY_LOCK = Lock()


def register_function(
    name: str,
    parameters: Sequence[Identifier],
    body: Expression,
) -> RegisteredFunction:
    """Register a pure function in the process-wide registry.

    The body's free identifiers (those reachable from ``body`` via the
    AST walk used by ``collect_identifiers``) must be a subset of
    ``parameters``. If any free identifier in the body is not declared
    as a parameter, registration fails.

    Args:
        name: Unique registry key.
        parameters: Positional parameter identifiers, in declared order.
        body: Body expression. May reference other already-registered
            functions via ``CallExpression`` (the inliner unfolds them
            at call time).

    Returns:
        The newly stored ``RegisteredFunction``.

    Raises:
        FunctionRegistrationError: On duplicate registration or on
            captured free identifiers.

    """
    parameter_tuple = tuple(parameters)
    _reject_captured_free_identifiers(name, parameter_tuple, body)

    with _REGISTRY_LOCK:
        if name in _REGISTRY:
            raise FunctionRegistrationError(
                f"A function is already registered under the name {name!r}."
            )
        registered = RegisteredFunction(name, parameter_tuple, body)
        _REGISTRY[name] = registered
        return registered


def _reject_captured_free_identifiers(
    name: str,
    parameters: tuple[Identifier, ...],
    body: Expression,
) -> None:
    declared = set(parameters)
    captured = collect_identifiers(body) - declared
    if not captured:
        return
    captured_names = ", ".join(sorted(identifier.name_hint for identifier in captured))
    raise FunctionRegistrationError(
        f"Function {name!r} body references identifiers not in its "
        f"parameters: {captured_names}."
    )


def get_registered_function(name: str) -> RegisteredFunction:
    """Return the function registered under ``name``.

    Args:
        name: Registry key.

    Returns:
        The stored ``RegisteredFunction``.

    Raises:
        FunctionLookupError: If no function is registered under that
            name.

    """
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(name)
    if registered is None:
        raise FunctionLookupError(f"No function is registered under the name {name!r}.")
    return registered


def get_registered_functions() -> Mapping[str, RegisteredFunction]:
    """Return an immutable snapshot of the current registry."""
    with _REGISTRY_LOCK:
        return frozendict(_REGISTRY)


def is_function_registered(name: str) -> bool:
    """Return whether a function is registered under ``name``."""
    with _REGISTRY_LOCK:
        return name in _REGISTRY


def _set_registry_state_for_tests(
    state: Mapping[str, RegisteredFunction],
) -> None:
    """Replace the registry contents with ``state``.

    Intended for the test-isolation fixture in
    ``tests/expression/conftest.py``: the fixture snapshots the registry
    before each test and calls this hook to restore the snapshot after
    the test runs. The leading underscore marks this as private; it
    should not be used outside test infrastructure.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(state)
