"""Testing patches for FhY core."""

__all__ = [
    "fail_fast_structural_equivalence",
    "compare_identifiers_by_name_hint",
]

import contextlib
import functools
import inspect
import sys
from typing import Any, Generator

from fhy_core.identifier import Identifier
from fhy_core.trait import StructuralEquivalence


@contextlib.contextmanager
def fail_fast_structural_equivalence() -> Generator[None, None, None]:
    """Patch the structural equivalence methods to fail fast."""
    original_methods = []

    def wrap_method(cls: type[StructuralEquivalence], method_name: str) -> None:
        orig = getattr(cls, method_name)

        @functools.wraps(orig)
        def wrapped(self: StructuralEquivalence, *args: Any, **kwargs: Any) -> Any:
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
            if isinstance(obj, StructuralEquivalence):
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
def compare_identifiers_by_name_hint() -> Generator[None, None, None]:
    """Patch the identifier equality method to compare by name hint."""
    original_eq = Identifier.__eq__

    def patched_eq(self: Identifier, other: object) -> bool:
        if not isinstance(other, Identifier):
            return False
        return self.name_hint == other.name_hint

    Identifier.__eq__ = patched_eq  # type: ignore
    try:
        yield
    finally:
        Identifier.__eq__ = original_eq  # type: ignore
