"""Process-wide storage and read-side accessors for the registry.

Holds the mutable dictionary that maps registry keys to
:class:`RegisteredEntry` instances, plus the lookup, presence-check,
and snapshot helpers that consumers use to read the registry. The
mutating ``register_*`` helpers live in
:mod:`fhy_core.symbolic.expression.registry.api`.

This module also tracks which registered functions have a body check
deferred pending a forward-referenced name:
:func:`fhy_core.symbolic.expression.registry.register_function` records
a deferral here when the body checker reports one, and consults it to
re-run the check once the missing name is registered.
"""

__all__ = [
    "get_registered_entries",
    "get_registered_entry",
    "is_entry_registered",
    "set_registry_state_for_tests",
]

from collections.abc import Mapping
from threading import Lock

from immutabledict import immutabledict

from ..errors import EntryLookupError, EntryRegistrationError
from .entries import NativeConstant, RegisteredEntry

_REGISTRY: dict[str, RegisteredEntry] = {}
_REGISTRY_LOCK = Lock()

# Maps a deferred entry's name to the name of the not-yet-registered call
# target its body is waiting on. An entry has at most one pending
# deferral at a time: the body checker aborts at the first unresolved
# call, so a body that forward-references several names is re-filed
# under each one in turn as its predecessors are registered.
_DEFERRED_BODY_CHECKS: dict[str, str] = {}


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


def set_registry_state_for_tests(
    state: Mapping[str, RegisteredEntry],
) -> None:
    """Replace the registry contents with ``state``.

    Also clears any pending deferred body-check state, so a test that
    triggers a deferral never leaks it into the next test.

    Intended for the test-isolation fixture in
    ``tests/symbolic/expression/conftest.py``: the fixture snapshots
    the registry before each test and calls this hook to restore the
    snapshot after the test runs. The ``_for_tests`` suffix marks
    this as a test-only seam; production code must not call it.
    """
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
        _REGISTRY.update(state)
        _DEFERRED_BODY_CHECKS.clear()


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


def _record_deferred_body_check(entry_name: str, missing_name: str) -> None:
    """Record that ``entry_name``'s body check is pending ``missing_name``."""
    with _REGISTRY_LOCK:
        _DEFERRED_BODY_CHECKS[entry_name] = missing_name


def _clear_deferred_body_check(entry_name: str) -> None:
    """Remove any pending body-check deferral recorded for ``entry_name``."""
    with _REGISTRY_LOCK:
        _DEFERRED_BODY_CHECKS.pop(entry_name, None)


def _pop_entries_deferred_on(missing_name: str) -> tuple[str, ...]:
    """Return and clear the entry names currently deferred on ``missing_name``."""
    with _REGISTRY_LOCK:
        deferred = tuple(
            entry_name
            for entry_name, waiting_on in _DEFERRED_BODY_CHECKS.items()
            if waiting_on == missing_name
        )
        for entry_name in deferred:
            del _DEFERRED_BODY_CHECKS[entry_name]
    return deferred
