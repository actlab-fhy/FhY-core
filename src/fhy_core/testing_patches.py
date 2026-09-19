"""Testing patches for FhY core."""

from fhy_core.utils.override import override

__all__ = [
    "deterministic_identifiers_by_name_hint",
    "fail_fast_structural_equivalence",
]

import contextlib
import functools
import sys
from collections.abc import Callable, Generator
from contextlib import ContextDecorator
from threading import RLock
from typing import Any, TypeVar, overload

from fhy_core.identifier import Identifier
from fhy_core.traits import StructuralEquivalence

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., Any])

_PATCHED_IDENTIFIER_DUNDERS = ("__new__", "__init__")
_ABSENT = object()


@contextlib.contextmanager
def fail_fast_structural_equivalence() -> Generator[None, None, None]:
    """Patch the structural equivalence methods to fail fast."""
    original_methods: list[tuple[type[StructuralEquivalence], str, Any]] = []
    seen_classes: set[type[Any]] = set()

    def wrap_method(cls: type[StructuralEquivalence], method_name: str) -> None:
        if method_name not in cls.__dict__:
            return

        orig = cls.__dict__[method_name]

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
        if module is None or not hasattr(module, "__name__"):
            continue

        # Iterate __dict__ directly rather than via inspect.getmembers so
        # we never trigger ``__getattr__``-only lazy attributes (notably
        # ``typing.io`` / ``typing.re``, which emit DeprecationWarnings on
        # access and are slated for removal in Python 3.13). Class
        # definitions live in __dict__; lazy aliases do not.
        module_dict = getattr(module, "__dict__", None)
        if module_dict is None:
            continue
        for obj in module_dict.values():
            if not isinstance(obj, type):
                continue
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


def _find_name_hint(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Return the name hint an ``Identifier(...)`` call passes, or ``_ABSENT``."""
    if args:
        return args[0]
    return kwargs.get("name_hint", _ABSENT)


class _DeterministicIdentifiersByNameHint(ContextDecorator):
    """Scope that assigns deterministic IDs from `name_hint`.

    This is intended for tests that compare object graphs containing
    internally-created identifiers, including when those identifiers are used as
    dictionary keys or set members. Within the scope, constructing an
    `Identifier` with a `name_hint` seen before in the scope returns the
    identifier constructed first with it, so equality and hashing remain
    consistent, and each new `name_hint` receives a fresh ID from the real
    counter. Deserialization, unpickling, and copying are unaffected. Scopes
    nest: the identifiers are shared until the outermost scope exits, which
    restores `Identifier` exactly and forgets them. The scope works as a
    context manager and as a decorator, with or without a call.

    The scope replaces `Identifier.__new__` and `Identifier.__init__` while
    active, so it applies to every thread.

    Note:
        This is only safe when every semantically distinct identifier created
        within the scope has a unique `name_hint`.

    """

    _lock: RLock
    _active_count: int
    _identifiers_by_name_hint: dict[Any, Identifier]
    _own_identifier_dunders: dict[str, Any]

    def __init__(self) -> None:
        self._lock = RLock()
        self._active_count = 0
        self._identifiers_by_name_hint = {}
        self._own_identifier_dunders = {}

    @overload
    def __call__(self, func: None = None) -> "_DeterministicIdentifiersByNameHint": ...

    @overload
    def __call__(self, func: _FunctionT) -> _FunctionT: ...

    @override
    def __call__(
        self, func: _FunctionT | None = None
    ) -> "_FunctionT | _DeterministicIdentifiersByNameHint":
        if func is None:
            return self
        return super().__call__(func)

    def __enter__(self) -> "_DeterministicIdentifiersByNameHint":
        with self._lock:
            if self._active_count == 0:
                self._patch_identifier()
            self._active_count += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        with self._lock:
            if self._active_count == 0:
                raise RuntimeError(
                    "deterministic_identifiers_by_name_hint exited without "
                    "a matching entry"
                )
            self._active_count -= 1
            if self._active_count == 0:
                self._restore_identifier()

    def _patch_identifier(self) -> None:
        """Make `Identifier` construction return one instance per name hint."""
        self._own_identifier_dunders = {
            name: Identifier.__dict__.get(name, _ABSENT)
            for name in _PATCHED_IDENTIFIER_DUNDERS
        }
        original_new: Callable[..., Identifier] = Identifier.__new__
        original_init: Callable[..., None] = Identifier.__init__

        def construct_once_per_name_hint(
            cls: type[Identifier], *args: Any, **kwargs: Any
        ) -> Identifier:
            # A call without a name hint is not a construction to share:
            # deserialization (and so unpickling and copying) allocates with
            # `cls.__new__(cls)`, and a missing argument must still raise.
            name_hint = _find_name_hint(args, kwargs)
            if name_hint is _ABSENT:
                return original_new(cls, *args, **kwargs)
            with self._lock:
                identifier = self._identifiers_by_name_hint.get(name_hint)
                if identifier is None:
                    identifier = original_new(cls, *args, **kwargs)
                    original_init(identifier, *args, **kwargs)
                    self._identifiers_by_name_hint[name_hint] = identifier
            return identifier

        def initialize_unless_shared(
            identifier: Identifier, *args: Any, **kwargs: Any
        ) -> None:
            # The shared instance was initialized when first constructed.
            name_hint = _find_name_hint(args, kwargs)
            if self._identifiers_by_name_hint.get(name_hint) is identifier:
                return
            original_init(identifier, *args, **kwargs)

        replacements = {
            "__new__": staticmethod(construct_once_per_name_hint),
            "__init__": initialize_unless_shared,
        }
        for name, replacement in replacements.items():
            setattr(Identifier, name, replacement)

    def _restore_identifier(self) -> None:
        """Restore `Identifier`'s own namespace and forget the shared instances."""
        for name, own_value in self._own_identifier_dunders.items():
            if own_value is _ABSENT:
                delattr(Identifier, name)
            else:
                setattr(Identifier, name, own_value)
        self._own_identifier_dunders = {}
        self._identifiers_by_name_hint.clear()


deterministic_identifiers_by_name_hint = _DeterministicIdentifiersByNameHint()
