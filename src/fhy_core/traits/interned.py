"""`Interned` trait and mixin."""

from fhy_core.utils.override import override

__all__ = ["Interned", "InternedMixin"]

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Hashable
from functools import wraps
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, runtime_checkable

from fhy_core.logger import get_logger

from .frozen import Frozen
from .verifiable import Verifiable

_LOGGER = get_logger(__name__)

_K = TypeVar("_K", bound=Hashable)
_K_co = TypeVar("_K_co", bound=Hashable, covariant=True)
_I = TypeVar("_I", bound="InternedMixin[Any]")


@runtime_checkable
class Interned(Protocol[_K_co]):
    """Protocol for objects that register a canonical instance by key."""

    def get_intern_key(self) -> _K_co:
        """Return the stable key used to look up the canonical instance."""


def _warn_interned_metadata_ignored(
    payload: "InternedMixin[Any]", canonical: "InternedMixin[Any]"
) -> None:
    """Log equality-excluded fields whose payload value the canonical ignores."""
    for field_definition in dataclasses.fields(cast(Any, payload)):
        if field_definition.compare:
            continue
        payload_value = getattr(payload, field_definition.name)
        canonical_value = getattr(canonical, field_definition.name)
        if payload_value != canonical_value:
            _LOGGER.warning(
                "%s %r already canonical; keeping %s=%r and ignoring payload %r.",
                type(payload).__name__,
                payload.get_intern_key(),
                field_definition.name,
                canonical_value,
                payload_value,
            )


class InternedMixin(Generic[_K], ABC):
    """Mixin that registers initialized instances into a family-local registry.

    Each direct subclass family gets its own registry. Descendants share that
    registry so a base class can look up canonical instances created by any of
    its concrete subclasses.

    If an instance also implements `Verifiable` and/or `Frozen`, the mixin
    verifies and freezes it before registration.

    Thread safety:
        The registry is not guarded by a lock. Concurrent construction of
        interned instances from multiple threads may race on
        ``register_interned_instance``, causing one of two otherwise-equivalent
        instances to shadow the other in the registry. Single-threaded use is
        assumed; synchronize externally if instances are constructed
        concurrently.
    """

    _interned_instances: ClassVar[dict[Hashable, "InternedMixin[Any]"] | None] = None
    _intern_init_depth: int

    @override
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
        else:
            _LOGGER.debug(
                "shadowing new instance of %s for key %r (canonical instance retained)",
                type(instance).__name__,
                key,
            )

    def _finalize_interned_instance(self) -> None:
        if isinstance(self, Verifiable):
            self.verify().raise_if_failed()
        if isinstance(self, Frozen) and not self.is_frozen:
            self.freeze()

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
        else:
            return instance

    @classmethod
    def require_interned(cls: type[_I], key: _K) -> _I:
        """Return the canonical instance for `key` or raise `KeyError`."""
        instance = cls.get_interned(key)
        if instance is None:
            raise KeyError(f'No registered "{cls.__name__}" instance for key {key!r}.')
        else:
            return instance

    @classmethod
    def clear_interned_registry(cls) -> None:
        """Clear this family's canonical-instance registry.

        Note:
            Subclasses that ship module-level canonical default instances
            do not have those defaults automatically re-registered. Call
            :meth:`register_default_instances` after clearing to restore
            the canonical mapping.
        """
        _LOGGER.debug("clearing intern registry for %s", cls.__name__)
        cls._get_interned_registry().clear()

    @classmethod
    def register_default_instances(cls) -> None:
        """Re-register the canonical default instances shipped by this class.

        Override on subclasses that bind module-level canonical defaults
        (e.g. ``ValueDomain.DATA_DOMAIN``, ``OpAttribute.COMMUTATIVE``) so
        callers can restore them after :meth:`clear_interned_registry`.
        The base implementation is a no-op for classes without defaults.
        """

    @classmethod
    def construct_from_fields(cls: type[_I], fields: dict[str, Any]) -> _I:
        """Reconstruct an interned instance, returning the canonical entry.

        Serves as the serialization reconstruction hook (see
        ``fhy_core.serialization``) for interned dataclasses: builds the
        instance from ``fields``, then returns the canonical instance for its
        intern key -- the freshly built one when the key is new, otherwise the
        pre-existing canonical. Equality-excluded fields
        (``field(compare=False)``) whose payload value differs from the
        canonical's are logged as ignored.

        Note that reconstruction always fully constructs (and verifies) a new
        instance: when the intern key already exists the freshly built instance
        is a throwaway that loses the registration race and is discarded, so its
        ``verify()`` still runs and may raise on otherwise-discardable data.

        Args:
            fields: Decoded field values, one entry per dataclass field.

        Returns:
            The canonical instance for the reconstructed object's intern key.
        """
        instance = cls(**fields)
        canonical = cls.get_interned(instance.get_intern_key())
        if canonical is None:
            return instance
        if canonical is not instance:
            _warn_interned_metadata_ignored(instance, canonical)
        return canonical

    @abstractmethod
    def get_intern_key(self) -> _K:
        """Return the stable key used to register this canonical instance."""
