"""`Interned` trait and mixin."""

__all__ = ["Interned", "InternedMixin"]

from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import wraps
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, runtime_checkable

from .frozen import Frozen
from .verifiable import Verifiable

_K = TypeVar("_K", bound=Hashable)
_K_co = TypeVar("_K_co", bound=Hashable, covariant=True)
_I = TypeVar("_I", bound="InternedMixin[Any]")


@runtime_checkable
class Interned(Protocol[_K_co]):
    """Protocol for objects that register a canonical instance by key."""

    def get_intern_key(self) -> _K_co:
        """Return the stable key used to look up the canonical instance."""


class InternedMixin(ABC, Generic[_K]):
    """Mixin that registers initialized instances into a family-local registry.

    Each direct subclass family gets its own registry. Descendants share that
    registry so a base class can look up canonical instances created by any of
    its concrete subclasses.

    If an instance also implements `Verifiable` and/or `Frozen`, the mixin
    verifies and deep-freezes it before registration.
    """

    _interned_instances: ClassVar[dict[Hashable, "InternedMixin[Any]"] | None] = None
    _intern_init_depth: int

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if cls._should_allocate_intern_registry():
            cls._interned_instances = {}

        init_method = cls.__dict__.get("__init__")
        if init_method is None:
            return
        if getattr(init_method, "__interned_registry_wrapper__", False):
            return

        @wraps(init_method)
        def _wrapped_init(
            self: "InternedMixin[_K]", *args: Any, **init_kwargs: Any
        ) -> None:
            init_depth = getattr(self, "_intern_init_depth", 0)
            object.__setattr__(self, "_intern_init_depth", init_depth + 1)
            init_completed = False
            try:
                init_method(self, *args, **init_kwargs)
                init_completed = True
            finally:
                new_depth = self._intern_init_depth - 1
                object.__setattr__(self, "_intern_init_depth", new_depth)
                if init_completed and new_depth == 0:
                    self.register_interned_instance()

        setattr(_wrapped_init, "__interned_registry_wrapper__", True)
        setattr(cls, "__init__", _wrapped_init)

    @classmethod
    def _should_allocate_intern_registry(cls) -> bool:
        for base in cls.__mro__[1:]:
            base_registry = base.__dict__.get("_interned_instances")
            if base_registry is not None:
                return False
        return True

    @classmethod
    def _get_interned_registry(cls) -> dict[Hashable, "InternedMixin[Any]"]:
        for base in cls.__mro__:
            registry = base.__dict__.get("_interned_instances")
            if registry is not None:
                return cast(dict[Hashable, "InternedMixin[Any]"], registry)
        raise RuntimeError(f"{cls.__name__} does not have an intern registry.")

    @classmethod
    def _register_interned_instance(cls, instance: "InternedMixin[Any]") -> None:
        registry = cls._get_interned_registry()
        key = instance.get_intern_key()
        existing_instance = registry.get(key)
        if existing_instance is None or existing_instance is instance:
            registry[key] = instance

    def _finalize_interned_instance(self) -> None:
        if isinstance(self, Verifiable):
            self.verify()
        if isinstance(self, Frozen) and not self.is_frozen:
            self.freeze(deep=True)

    def register_interned_instance(self) -> None:
        """Finalize and register this instance as the canonical value for its key."""
        self._finalize_interned_instance()
        type(self)._register_interned_instance(self)

    @classmethod
    def get_interned(cls: type[_I], key: _K) -> _I | None:
        """Return the canonical instance for `key`, if one has been registered."""
        instance = cls._get_interned_registry().get(key)
        if instance is None or not isinstance(instance, cls):
            return None
        return instance

    @classmethod
    def require_interned(cls: type[_I], key: _K) -> _I:
        """Return the canonical instance for `key` or raise `KeyError`."""
        instance = cls.get_interned(key)
        if instance is None:
            raise KeyError(f'No registered "{cls.__name__}" instance for key {key!r}.')
        return instance

    @classmethod
    def clear_interned_registry(cls) -> None:
        """Clear this family's canonical-instance registry."""
        cls._get_interned_registry().clear()

    @abstractmethod
    def get_intern_key(self) -> _K:
        """Return the stable key used to register this canonical instance."""
