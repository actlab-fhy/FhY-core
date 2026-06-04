"""`Canonicalizable` trait."""

__all__ = ["Canonicalizable"]

from typing import Protocol, runtime_checkable


@runtime_checkable
class Canonicalizable(Protocol):
    """Protocol for objects that can canonicalize their local representation."""

    def canonicalize(self) -> bool:
        """Canonicalize in place and return if a change was applied."""
