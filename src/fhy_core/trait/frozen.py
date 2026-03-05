"""`Frozen` trait and mixin."""

__all__ = [
    "Frozen",
    "FrozenMixin",
    "FrozenMutationError",
    "FrozenValidationError",
]

from abc import ABC
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from frozendict import frozendict

from fhy_core.error import register_error

_FROZEN_FLAG = "_fhy_core_is_frozen"
_IMMUTABLE_ATOMS = (str, bytes, int, float, complex, bool, type(None))


@register_error
class FrozenMutationError(RuntimeError):
    """Raised when mutating a frozen object."""


@register_error
class FrozenValidationError(RuntimeError):
    """Raised when runtime frozen verification fails."""


@runtime_checkable
class Frozen(Protocol):
    """Protocol for objects that support runtime freezing."""

    @property
    def is_frozen(self) -> bool: ...

    def freeze(self, *, deep: bool = False) -> None: ...

    def assert_frozen(self, *, deep: bool = False, strict: bool = False) -> None: ...

    def assert_write_protected(self) -> None: ...


class FrozenMixin(ABC):
    """Mixin that enforces runtime immutability after `freeze()`."""

    @property
    def is_frozen(self) -> bool:
        return self._is_marked_frozen()

    def freeze(self, *, deep: bool = False) -> None:
        """Freeze this object and optionally deep-freeze its instance state."""
        if deep:
            for attribute_name, attribute_value in self._iter_instance_state_items():
                if attribute_name == _FROZEN_FLAG:
                    continue
                frozen_value = self._freeze_value(attribute_value, set())
                object.__setattr__(self, attribute_name, frozen_value)
        object.__setattr__(self, _FROZEN_FLAG, True)

    def assert_frozen(self, *, deep: bool = False, strict: bool = False) -> None:
        """Assert that this object is frozen.

        Args:
            deep: Whether to verify deep immutability for the instance state.
            strict: Whether unknown nested object types are treated as mutable.

        Raises:
            FrozenValidationError: If frozen constraints are violated.

        """
        if not self.is_frozen:
            raise FrozenValidationError(f"{type(self).__name__} is not frozen.")
        self.assert_write_protected()
        if deep and not self._is_deeply_immutable(self, set(), strict=strict):
            raise FrozenValidationError(
                f'"{type(self).__name__}" failed deep immutability validation.'
            )

    def assert_write_protected(self) -> None:
        """Assert that normal `setattr` attempts are blocked for this object."""
        if not self.is_frozen:
            raise FrozenValidationError(
                f"{type(self).__name__} is not frozen, so write protection is inactive."
            )
        candidate_name: str | None = None
        for attribute_name, _ in self._iter_instance_state_items():
            if attribute_name == _FROZEN_FLAG:
                continue
            candidate_name = attribute_name
            break
        if candidate_name is None:
            return
        try:
            setattr(self, candidate_name, getattr(self, candidate_name))
        except FrozenMutationError:
            return
        except Exception as exc:
            raise FrozenValidationError(
                f'"{type(self).__name__}" write-protection probe failed '
                f"unexpectedly: {exc}"
            ) from exc
        raise FrozenValidationError(
            f'"{type(self).__name__}" accepted mutation while frozen.'
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if name != _FROZEN_FLAG and self._is_marked_frozen():
            raise FrozenMutationError(
                f'Cannot modify "{name}" on frozen {type(self).__name__}.'
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name != _FROZEN_FLAG and self._is_marked_frozen():
            raise FrozenMutationError(
                f'Cannot delete "{name}" on frozen {type(self).__name__}.'
            )
        object.__delattr__(self, name)

    @staticmethod
    def _freeze_value(value: Any, seen: set[int]) -> Any:
        value_id = id(value)
        if value_id in seen:
            return value
        seen.add(value_id)

        if isinstance(value, list):
            return tuple(FrozenMixin._freeze_value(item, seen) for item in value)
        elif isinstance(value, tuple):
            return tuple(FrozenMixin._freeze_value(item, seen) for item in value)
        elif isinstance(value, set):
            return frozenset(FrozenMixin._freeze_value(item, seen) for item in value)
        elif isinstance(value, frozenset):
            return frozenset(FrozenMixin._freeze_value(item, seen) for item in value)
        elif isinstance(value, dict):
            frozen_dict = {
                FrozenMixin._freeze_value(key, seen): FrozenMixin._freeze_value(
                    item_value, seen
                )
                for key, item_value in value.items()
            }
            return frozendict(frozen_dict)
        elif isinstance(value, MappingProxyType):
            frozen_dict = {
                FrozenMixin._freeze_value(key, seen): FrozenMixin._freeze_value(
                    item_value, seen
                )
                for key, item_value in value.items()
            }
            return frozendict(frozen_dict)
        elif isinstance(value, bytearray):
            return bytes(value)
        elif isinstance(value, FrozenMixin):
            value.freeze(deep=True)
            return value
        else:
            return value

    @staticmethod
    def _is_deeply_immutable(value: Any, seen: set[int], *, strict: bool) -> bool:
        value_id = id(value)
        if value_id in seen:
            return True
        seen.add(value_id)

        if isinstance(value, _IMMUTABLE_ATOMS):
            return True
        elif isinstance(value, tuple):
            return all(
                FrozenMixin._is_deeply_immutable(item, seen, strict=strict)
                for item in value
            )
        elif isinstance(value, frozenset):
            return all(
                FrozenMixin._is_deeply_immutable(item, seen, strict=strict)
                for item in value
            )
        elif isinstance(value, (frozendict, MappingProxyType)):
            return all(
                FrozenMixin._is_deeply_immutable(key, seen, strict=strict)
                and FrozenMixin._is_deeply_immutable(item_value, seen, strict=strict)
                for key, item_value in value.items()
            )
        elif isinstance(value, FrozenMixin):
            if not value.is_frozen:
                return False
            return all(
                FrozenMixin._is_deeply_immutable(item_value, seen, strict=strict)
                for _, item_value in value._iter_instance_state_items()
                if _ != _FROZEN_FLAG
            )
        elif isinstance(value, (list, set, dict, bytearray)):
            return False
        elif strict and FrozenMixin._has_instance_state(value):
            return False
        else:
            return True

    def _iter_instance_state_items(self) -> list[tuple[str, Any]]:
        state: dict[str, Any] = {}
        instance_dict = getattr(self, "__dict__", None)
        if isinstance(instance_dict, dict):
            state.update(instance_dict)

        for klass in type(self).mro():
            slots = getattr(klass, "__slots__", ())
            if isinstance(slots, str):
                slot_names = (slots,)
            else:
                slot_names = tuple(slots)
            for slot_name in slot_names:
                if slot_name in {"__dict__", "__weakref__"}:
                    continue
                if slot_name in state:
                    continue
                try:
                    state[slot_name] = object.__getattribute__(self, slot_name)
                except AttributeError:
                    continue

        return list(state.items())

    def _is_marked_frozen(self) -> bool:
        try:
            return bool(object.__getattribute__(self, _FROZEN_FLAG))
        except AttributeError:
            return False

    @staticmethod
    def _has_instance_state(value: Any) -> bool:
        if hasattr(value, "__dict__"):
            return True
        for klass in type(value).mro():
            slots = getattr(klass, "__slots__", ())
            if isinstance(slots, str):
                if slots not in {"__dict__", "__weakref__"}:
                    return True
            else:
                for slot_name in slots:
                    if slot_name not in {"__dict__", "__weakref__"}:
                        return True
        return False
