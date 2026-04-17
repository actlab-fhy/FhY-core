"""`HasType` trait and mixin."""

__all__ = ["HasType", "HasTypeMixin"]

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

_TypeT_co = TypeVar("_TypeT_co", covariant=True)


@runtime_checkable
class HasType(Protocol[_TypeT_co]):
    """Protocol for values and operations that carry a type."""

    def get_type(self) -> _TypeT_co: ...


class HasTypeMixin(ABC, Generic[_TypeT_co]):
    """Mixin for values and operations that carry a type."""

    @abstractmethod
    def get_type(self) -> _TypeT_co: ...
