"""Process-wide storage and read-side accessors for the registry.

Holds the mutable dictionary that maps registry keys to
:class:`RegisteredEntry` instances, plus the lookup, presence-check,
and snapshot helpers that consumers use to read the registry. The
mutating ``register_*`` helpers live in
:mod:`fhy_core.symbolic.expression.registry.api`.

Every :class:`NativeConstant` additionally owns one canonical
:class:`Identifier`, minted when the constant is registered and held
here beside the entry. That identifier is the only expression-level
reference that denotes the constant: the backend bridges, the
evaluator, and the type checker all resolve a constant reference by
identifier identity, so an unrelated identifier that merely shares a
constant's ``name_hint`` is an ordinary free variable.
"""

__all__ = [
    "get_native_constant_identifier",
    "get_registered_entries",
    "get_registered_entry",
    "is_entry_registered",
    "set_registry_state_for_tests",
    "try_get_native_constant_for_identifier",
]

from collections.abc import Mapping
from threading import Lock

from immutabledict import immutabledict

from fhy_core.identifier import Identifier

from ..errors import EntryLookupError, EntryRegistrationError
from .entries import NativeConstant, RegisteredEntry

_REGISTRY: dict[str, RegisteredEntry] = {}
_NATIVE_CONSTANT_IDENTIFIERS: dict[str, Identifier] = {}
_NATIVE_CONSTANTS_BY_IDENTIFIER: dict[Identifier, NativeConstant] = {}
_REGISTRY_LOCK = Lock()


def get_registered_entry(name: str) -> RegisteredEntry:
    """Return the entry registered under ``name``.

    The return type widens to ``RegisteredEntry`` (the union of
    :class:`RegisteredFunction`, :class:`NativeFunction`, and
    :class:`NativeConstant`). Callers that need to distinguish kinds
    use ``isinstance``.

    Args:
        name: Registry key.

    Returns:
        The stored entry.

    Raises:
        EntryLookupError: If no entry is registered under ``name``.

    """
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(name)
    if registered is None:
        raise EntryLookupError(f"No entry is registered under the name {name!r}.")
    return registered


def get_registered_entries() -> Mapping[str, RegisteredEntry]:
    """Return an immutable snapshot of the current registry."""
    with _REGISTRY_LOCK:
        return immutabledict(_REGISTRY)


def is_entry_registered(name: str) -> bool:
    """Return whether any entry is registered under ``name``."""
    with _REGISTRY_LOCK:
        return name in _REGISTRY


def get_native_constant_identifier(name: str) -> Identifier:
    """Return the canonical identifier that denotes the constant ``name``.

    Each native constant owns exactly one :class:`Identifier`, minted
    when the constant is registered. Wrapping that identifier in an
    ``IdentifierExpression`` is how an expression refers to the
    constant, and every resolution path recognizes it by identity.

    Args:
        name: Registry key of the native constant.

    Returns:
        The constant's canonical identifier.

    Raises:
        EntryLookupError: If no native constant is registered under
            ``name``.

    """
    with _REGISTRY_LOCK:
        identifier = _NATIVE_CONSTANT_IDENTIFIERS.get(name)
    if identifier is None:
        raise EntryLookupError(
            f"No native constant is registered under the name {name!r}."
        )
    return identifier


def try_get_native_constant_for_identifier(
    identifier: Identifier,
) -> NativeConstant | None:
    """Return the constant ``identifier`` denotes, or ``None``.

    Resolution is by identifier identity: only the canonical identifier
    the registry minted for a constant resolves to it. An identifier
    that shares a constant's ``name_hint`` but not its id is an ordinary
    free variable, and so is an identifier whose name matches a
    registered function rather than a constant.

    Args:
        identifier: Identifier to resolve.

    Returns:
        The native constant the identifier denotes, or ``None`` when it
        denotes no constant.

    """
    with _REGISTRY_LOCK:
        return _NATIVE_CONSTANTS_BY_IDENTIFIER.get(identifier)


def set_registry_state_for_tests(
    state: Mapping[str, RegisteredEntry],
) -> None:
    """Replace the registry contents with ``state``.

    Intended for the test-isolation fixture in ``tests/conftest.py``:
    the fixture snapshots the registry before each test and calls this
    hook to restore the snapshot after the test runs. The
    ``_for_tests`` suffix marks this as a test-only seam; production
    code must not call it.

    Canonical constant identifiers are pruned to match: a constant
    entry that ``state`` does not carry loses its identifier, so a
    constant registered inside the test stops resolving once the
    snapshot is restored.
    """
    with _REGISTRY_LOCK:
        retained_identifiers = {
            name: identifier
            for name, identifier in _NATIVE_CONSTANT_IDENTIFIERS.items()
            if state.get(name) is _NATIVE_CONSTANTS_BY_IDENTIFIER[identifier]
        }
        retained_constants = {
            identifier: _NATIVE_CONSTANTS_BY_IDENTIFIER[identifier]
            for identifier in retained_identifiers.values()
        }
        _REGISTRY.clear()
        _REGISTRY.update(state)
        _NATIVE_CONSTANT_IDENTIFIERS.clear()
        _NATIVE_CONSTANT_IDENTIFIERS.update(retained_identifiers)
        _NATIVE_CONSTANTS_BY_IDENTIFIER.clear()
        _NATIVE_CONSTANTS_BY_IDENTIFIER.update(retained_constants)


def _claim_registry_name(name: str, entry: RegisteredEntry) -> None:
    """Insert ``entry`` under ``name``, with ``_REGISTRY_LOCK`` already held.

    Raises:
        EntryRegistrationError: If ``name`` is already registered.

    """
    if name in _REGISTRY:
        raise EntryRegistrationError(f"A name is already registered: {name!r}.")
    _REGISTRY[name] = entry


def _insert_unique_entry(name: str, entry: RegisteredEntry) -> None:
    """Insert ``entry`` under ``name`` if the name is free; otherwise raise."""
    with _REGISTRY_LOCK:
        _claim_registry_name(name, entry)


def _insert_unique_native_constant(entry: NativeConstant) -> None:
    """Insert ``entry`` and mint the canonical identifier that denotes it.

    The entry and its identifier are published under one lock, so no
    reader observes a registered constant that has no identifier yet.

    Raises:
        EntryRegistrationError: If the entry's name is already
            registered.

    """
    identifier = Identifier(entry.name)
    with _REGISTRY_LOCK:
        _claim_registry_name(entry.name, entry)
        _NATIVE_CONSTANT_IDENTIFIERS[entry.name] = identifier
        _NATIVE_CONSTANTS_BY_IDENTIFIER[identifier] = entry


def _registered_constant_identifiers() -> frozenset[Identifier]:
    """Return the canonical identifier of every registered native constant."""
    with _REGISTRY_LOCK:
        return frozenset(_NATIVE_CONSTANT_IDENTIFIERS.values())
