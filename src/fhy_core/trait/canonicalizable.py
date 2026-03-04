"""`Canonicalizable` trait and mixin."""

from __future__ import annotations

__all__ = ["Canonicalizable", "CanonicalizableMixin"]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Canonicalizable(Protocol):
    """Protocol for objects that can canonicalize their local representation."""

    def canonicalize(self) -> bool:
        """Canonicalize in place and return if a change was applied."""


class CanonicalizableMixin(ABC):
    """Mixin for objects that can canonicalize their local representation."""

    @abstractmethod
    def canonicalize(self) -> bool:
        """Canonicalize in place and return if a change was applied."""
