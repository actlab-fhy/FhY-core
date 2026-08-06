"""Process-wide registry of expression-IR entries.

The registry holds three kinds of entries:

- :class:`RegisteredFunction`: a pure function whose body is an
  expression in the IR. Inlining substitutes the parameter identifiers
  with the call's argument expressions.
- :class:`NativeFunction`: a pure function whose body is a Python
  callable. The evaluator folds literal-argument calls; the inliner
  passes them through.
- :class:`NativeConstant`: a named literal value. The evaluator
  substitutes identifier references whose name matches the constant.

All three kinds expose declared sorts that drive the call-site type
checker, decoupling type inference from body inspection.

Registration records an entry; it does not type-check it. Checking a
registered function's body against its declared result sort is an
explicit call into the type-checking layer above, which keeps the IR
type system out of this package's dependencies.

The registry is process-wide. Tests that need isolation should request
the ``function_registry_snapshot`` fixture defined in ``tests/conftest.py``
rather than mutating the registry directly.
"""

__all__ = [
    "CallTargetResolver",
    "EntryLookupError",
    "EntryRegistrationError",
    "NativeConstant",
    "NativeFunction",
    "RegisteredEntry",
    "RegisteredFunction",
    "get_registered_entries",
    "get_registered_entry",
    "is_entry_registered",
    "register_function",
    "register_native_constant",
    "register_native_function",
    "set_registry_state_for_tests",
]

from ..errors import EntryLookupError, EntryRegistrationError
from .api import (
    register_function,
    register_native_constant,
    register_native_function,
)
from .entries import (
    CallTargetResolver,
    NativeConstant,
    NativeFunction,
    RegisteredEntry,
    RegisteredFunction,
)
from .storage import (
    get_registered_entries,
    get_registered_entry,
    is_entry_registered,
    set_registry_state_for_tests,
)
