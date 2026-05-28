"""Process-wide storage and read-side accessors for the registry.

Holds the mutable dictionary that maps registry keys to
:class:`RegisteredEntry` instances, plus the lookup, presence-check,
and snapshot helpers that consumers use to read the registry. The
mutating ``register_*`` helpers live in
:mod:`fhy_core.expression.registry.api`.
"""

__all__ = [
    "get_registered_entries",
    "get_registered_entry",
    "is_entry_registered",
    "set_registry_state_for_tests",
]

from collections.abc import Mapping
from threading import Lock

from frozendict import frozendict

from ..errors import EntryLookupError, EntryRegistrationError
from .entries import NativeConstant, RegisteredEntry

_REGISTRY: dict[str, RegisteredEntry] = {}
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
        return frozendict(_REGISTRY)


def is_entry_registered(name: str) -> bool:
    """Return whether any entry is registered under ``name``."""
    with _REGISTRY_LOCK:
        return name in _REGISTRY


def set_registry_state_for_tests(
    state: Mapping[str, RegisteredEntry],
) -> None:
    """Replace the registry contents with ``state``.

    Intended for the test-isolation fixture in
    ``tests/expression/conftest.py``: the fixture snapshots the
    registry before each test and calls this hook to restore the
    snapshot after the test runs. The ``_for_tests`` suffix marks
    this as a test-only seam; production code must not call it.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(state)


def _insert_unique_entry(name: str, entry: RegisteredEntry) -> None:
    """Insert ``entry`` under ``name`` if the name is free; otherwise raise."""
    with _REGISTRY_LOCK:
        if name in _REGISTRY:
            raise EntryRegistrationError(f"A name is already registered: {name!r}.")
        _REGISTRY[name] = entry


def _remove_entry(name: str) -> None:
    """Remove ``name`` from the registry if present."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(name, None)


def _registered_constant_names() -> set[str]:
    """Return the set of names that resolve to a :class:`NativeConstant`."""
    with _REGISTRY_LOCK:
        return {
            name
            for name, entry in _REGISTRY.items()
            if isinstance(entry, NativeConstant)
        }
