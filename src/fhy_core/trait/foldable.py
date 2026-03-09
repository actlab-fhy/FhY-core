"""`Foldable` trait and mixin."""

__all__ = ["Foldable", "FoldableMixin"]

from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar, runtime_checkable

_FoldResultT_co = TypeVar("_FoldResultT_co", covariant=True)


@runtime_checkable
class Foldable(Protocol[_FoldResultT_co]):
    """Protocol for objects that can constant-fold to a result."""

    def fold(self) -> _FoldResultT_co | None:
        """Return the folded result, or `None` if folding is not possible."""


class FoldableMixin(ABC, Generic[_FoldResultT_co]):
    """Mixin for objects that can constant-fold to a result."""

    @abstractmethod
    def fold(self) -> _FoldResultT_co | None:
        """Return the folded result, or `None` if folding is not possible."""
