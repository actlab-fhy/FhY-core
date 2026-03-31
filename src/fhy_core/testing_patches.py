"""Testing patches for FhY core."""

__all__ = [
    "fail_fast_structural_equivalence",
    "compare_identifiers_by_name_hint",
]

import contextlib
import functools
import inspect
import sys


@contextlib.contextmanager
def fail_fast_structural_equivalence():
    """Patch the structural equivalence methods to fail fast."""
    original_methods = []

    def wrap_method(cls, method_name):
        orig = getattr(cls, method_name)

        @functools.wraps(orig)
        def wrapped(self, *args, **kwargs):
            result = orig(self, *args, **kwargs)
            if result is False:
                raise AssertionError(
                    f"{cls.__name__}.{method_name} returned False\n"
                    f"self={self}\nargs={args}\nkwargs={kwargs}"
                )
            return result

        original_methods.append((cls, method_name, orig))
        setattr(cls, method_name, wrapped)

    for module in list(sys.modules.values()):
        if not module:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "is_structurally_equivalent"):
                try:
                    wrap_method(obj, "is_structurally_equivalent")
                except Exception:
                    pass

    try:
        yield
    finally:
        for cls, method_name, orig in reversed(original_methods):
            setattr(cls, method_name, orig)


@contextlib.contextmanager
def compare_identifiers_by_name_hint(identifier_cls):
    """Patch the identifier equality method to compare by name hint."""
    original_eq = identifier_cls.__eq__

    def patched_eq(self, other):
        if not isinstance(other, identifier_cls):
            return False
        return self.name_hint == other.name_hint

    identifier_cls.__eq__ = patched_eq
    try:
        yield
    finally:
        identifier_cls.__eq__ = original_eq
