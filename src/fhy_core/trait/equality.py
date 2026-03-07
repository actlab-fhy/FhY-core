"""`PartialEqual`/`Equal` traits and mixins."""

__all__ = ["Equal", "EqualMixin", "PartialEqual", "PartialEqualMixin"]

from dataclasses import is_dataclass
from types import NotImplementedType
from typing import Protocol, runtime_checkable


@runtime_checkable
class PartialEqual(Protocol):
    """Protocol for objects that define `==` semantics."""

    __hash__ = None

    @property
    def supports_partial_equality(self) -> bool: ...

    def __eq__(self, other: object) -> bool | NotImplementedType: ...


@runtime_checkable
class Equal(PartialEqual, Protocol):
    """Protocol for objects with total equality semantics."""

    @property
    def supports_equality(self) -> bool: ...


class PartialEqualMixin:
    """Mixin for objects that define `==` semantics."""

    __hash__ = None

    @property
    def supports_partial_equality(self) -> bool:
        if self._is_native_equatable_dataclass():
            return True
        type_eq = getattr(type(self), "__eq__")
        return type_eq is not object.__eq__ and type_eq is not PartialEqualMixin.__eq__

    def __eq__(self, other: object) -> bool | NotImplementedType:
        return NotImplemented

    def _is_native_equatable_dataclass(self) -> bool:
        if not is_dataclass(self):
            return False
        params = getattr(type(self), "__dataclass_params__", None)
        return bool(params and params.eq)


class EqualMixin(PartialEqualMixin):
    """Mixin for objects with total equality semantics."""

    @property
    def supports_equality(self) -> bool:
        return True
