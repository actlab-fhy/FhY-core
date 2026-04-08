"""Testing patches for FhY core."""

__all__ = [
    "fail_fast_structural_equivalence",
    "deterministic_identifiers_by_name_hint",
]

import contextlib
import functools
import inspect
import sys
from contextlib import ContextDecorator
from typing import Any, Generator

from fhy_core.identifier import Identifier
from fhy_core.trait import StructuralEquivalence


@contextlib.contextmanager
def fail_fast_structural_equivalence() -> Generator[None, None, None]:
    """Patch the structural equivalence methods to fail fast."""
    original_methods: list[tuple[type[StructuralEquivalence], str, Any]] = []
    seen_classes: set[type[Any]] = set()

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
        if not module or not hasattr(module, "__name__"):
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj in seen_classes or obj.__module__ != module.__name__:
                continue
            try:
                if not issubclass(obj, StructuralEquivalence):
                    continue
            except TypeError:
                continue
            try:
                wrap_method(obj, "is_structurally_equivalent")
            except AttributeError:
                continue
            seen_classes.add(obj)

    try:
        yield
    finally:
        for cls, method_name, orig in reversed(original_methods):
            setattr(cls, method_name, orig)


class _DeterministicIdentifiersByNameHint(ContextDecorator):
    """Test patch that assigns deterministic IDs from `name_hint`.

    This is intended for tests that compare object graphs containing
    internally-created identifiers, including when those identifiers are used as
    dictionary keys or set members. Within the patch, all identifiers sharing
    the same `name_hint` receive the same ID, so equality and hashing remain
    consistent.

    Note:
        This is only safe when every semantically distinct identifier created
        within the patched scope has a unique `name_hint`.

    """

    _active_count: int
    _name_hint_to_id: dict[str, int]
    _original_init: Any
    _patched_init_func: Any

    def __init__(self) -> None:
        self._active_count = 0
        self._name_hint_to_id = {}
        self._original_init = None
        self._patched_init_func = None

    def __call__(self, func: Any = None) -> Any:
        if func is None:
            return self
        return super().__call__(func)

    def __enter__(self) -> "_DeterministicIdentifiersByNameHint":
        if self._active_count == 0:
            self._name_hint_to_id = {}
            self._original_init = Identifier.__init__
            self._patched_init_func = self._make_patched_init()
            Identifier.__init__ = self._patched_init_func  # type: ignore
        self._active_count += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self._active_count -= 1
        if self._active_count == 0:
            Identifier.__init__ = self._original_init  # type: ignore
            self._name_hint_to_id = {}
            self._original_init = None
            self._patched_init_func = None

    def _make_patched_init(self) -> Any:
        def patched_init(identifier: Identifier, name_hint: str) -> None:
            with Identifier._id_lock:
                identifier_id = self._name_hint_to_id.get(name_hint)
                if identifier_id is None:
                    identifier_id = Identifier._next_id
                    self._name_hint_to_id[name_hint] = identifier_id
                    Identifier._next_id += 1
            identifier._id = identifier_id
            identifier._name_hint = name_hint

        return patched_init


deterministic_identifiers_by_name_hint = _DeterministicIdentifiersByNameHint()
