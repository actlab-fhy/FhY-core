"""`HasType` trait."""

__all__ = ["HasType"]

from typing import Protocol, TypeVar, runtime_checkable

_TypeT_co = TypeVar("_TypeT_co", covariant=True)


@runtime_checkable
class HasType(Protocol[_TypeT_co]):
    """Protocol for values and operations that carry a type."""

    def get_type(self) -> _TypeT_co:
        """Return the object's type."""
