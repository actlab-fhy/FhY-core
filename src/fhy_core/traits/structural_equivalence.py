"""`StructuralEquivalence` trait and mixin."""

__all__ = ["StructuralEquivalence", "StructuralEquivalenceMixin"]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class StructuralEquivalence(Protocol):
    """Protocol for objects that support structural equivalence checks."""

    def is_structurally_equivalent(self, other: object) -> bool:
        """Return if `self` and `other` are equivalent by structure."""


class StructuralEquivalenceMixin(ABC):
    """Mixin for objects that support structural equivalence checks."""

    @abstractmethod
    def is_structurally_equivalent(self, other: object) -> bool:
        """Return if `self` and `other` are equivalent by structure."""
