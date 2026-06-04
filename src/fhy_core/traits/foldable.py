"""`Foldable` trait."""

__all__ = ["Foldable"]

from typing import Protocol, TypeVar, runtime_checkable

_FoldResultT_co = TypeVar("_FoldResultT_co", covariant=True)


@runtime_checkable
class Foldable(Protocol[_FoldResultT_co]):
    """Protocol for objects that can constant-fold to a result."""

    def fold(self) -> _FoldResultT_co | None:
        """Return the folded result, or `None` if folding is not possible."""
