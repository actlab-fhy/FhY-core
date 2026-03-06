"""`PartialOrderable`/`Orderable` traits and mixins."""

__all__ = [
    "Orderable",
    "OrderableMixin",
    "PartialOrderable",
    "PartialOrderableMixin",
]

from dataclasses import is_dataclass
from types import NotImplementedType
from typing import Protocol, runtime_checkable


@runtime_checkable
class PartialOrderable(Protocol):
    """Protocol for objects that define partial ordering semantics."""

    @property
    def supports_partial_ordering(self) -> bool: ...

    def __lt__(self, other: object) -> bool | NotImplementedType: ...


@runtime_checkable
class Orderable(PartialOrderable, Protocol):
    """Protocol for objects that define total ordering semantics."""

    @property
    def supports_ordering(self) -> bool: ...


class PartialOrderableMixin:
    """Mixin for objects that define partial ordering semantics."""

    @property
    def supports_partial_ordering(self) -> bool:
        if self._is_native_ordered_dataclass():
            return True
        type_lt = getattr(type(self), "__lt__")
        object_lt = getattr(object, "__lt__")
        mixin_lt = getattr(PartialOrderableMixin, "__lt__")
        return type_lt is not object_lt and type_lt is not mixin_lt

    def __lt__(self, other: object) -> bool | NotImplementedType:
        return NotImplemented

    def _is_native_ordered_dataclass(self) -> bool:
        if not is_dataclass(self):
            return False
        params = getattr(type(self), "__dataclass_params__", None)
        return bool(params and params.order)


class OrderableMixin(PartialOrderableMixin):
    """Mixin for objects with total ordering semantics."""

    @property
    def supports_ordering(self) -> bool:
        return True
