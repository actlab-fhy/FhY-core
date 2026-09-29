"""The read-side accessors of the registry.

The registry is the Rust core's owned ``FunctionRegistry``, which the
extension keeps in its module state for this process-wide API (decision
N-S7-2 of ``docs/design/python-switch.md``). The functions here are the
extension's own, re-exported:

- :func:`get_registered_entry` returns the entry registered, or built in,
  under a name, or raises :class:`EntryLookupError`;
- :func:`get_registered_entries` returns an immutable snapshot of every
  entry: the built-ins in catalogue order (the constants, the composed
  functions, then the native functions), then the user entries in
  registration order;
- :func:`is_entry_registered` says whether a name resolves;
- :func:`get_native_constant_identifier` returns the identifier that
  denotes a constant, and :func:`try_get_native_constant_for_identifier`
  the constant an identifier denotes, or ``None``.

A lookup resolves a name through the built-in catalogue first, then the
user registry (N-S7-3), and returns the entry's single object, the one its
registration returned. Every :class:`NativeConstant` owns one canonical
:class:`Identifier`: a built-in constant's has a fixed reserved id, and a
user constant's is minted when it is registered. That identifier is the
only expression-level reference that denotes the constant: the backend
bridges, the evaluator, and the type checker all resolve a constant
reference by identifier identity, so an unrelated identifier that merely
shares a constant's ``name_hint`` is an ordinary free variable.

The built-ins are no state, and they cannot be removed. A lookup never
takes a lock across a call into Python, and a registration swaps in a new
state whole, so a lookup sees the registry before or after a concurrent
registration, never between.
"""

__all__ = [
    "get_native_constant_identifier",
    "get_registered_entries",
    "get_registered_entry",
    "is_entry_registered",
    "try_get_native_constant_for_identifier",
]

from fhy_core import _rs

get_registered_entry = _rs.get_registered_entry
get_registered_entries = _rs.get_registered_entries
is_entry_registered = _rs.is_entry_registered
get_native_constant_identifier = _rs.get_native_constant_identifier
try_get_native_constant_for_identifier = _rs.try_get_native_constant_for_identifier
