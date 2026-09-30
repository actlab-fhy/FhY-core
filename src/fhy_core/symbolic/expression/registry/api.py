"""Public registration API for the expression registry.

Exposes ``register_function`` / ``register_native_function`` /
``register_native_constant``, the write-side surface that mutates the
process-wide registry, and ``try_get_registered_result_sort``. They are the
Rust extension's functions (``fhy_core._rs``), re-exported; each
registration returns the entry object that later lookups return.

Registration records an entry; it does not type-check it. Checking a
function body against its declared sorts needs the IR type system, which
sits above this package in the dependency graph, so callers who want that
answer ask the type-checking layer for it explicitly. A body is therefore
free to call a function registered later: nothing here inspects the call
target, and the sweep in the type-checking layer runs once registration
is complete.

Registration refuses, with :class:`EntryRegistrationError` and the Rust
core's text:

- a name that is empty, a built-in function's (``function name `max` is a
  built-in function``) or a built-in constant's (``"pi" is the name of a
  built-in constant``), or already registered (``"f" is already
  registered``);
- a function whose parameter sorts differ in number from its parameters,
  whose parameters repeat one, or whose body refers to identifiers that are
  neither parameters nor constants registered so far or built in
  (``function "f" captures identifiers that are not its parameters: x,
  y``);
- a native function whose implementation cannot take its arity;
- a constant whose value its sort does not accept.
"""

__all__ = [
    "register_function",
    "register_native_constant",
    "register_native_function",
    "try_get_registered_result_sort",
]

from fhy_core import _rs

register_function = _rs.register_function
register_native_function = _rs.register_native_function
register_native_constant = _rs.register_native_constant
try_get_registered_result_sort = _rs.try_get_registered_result_sort
