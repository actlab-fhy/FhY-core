"""`StructuralEquivalence` trait."""

__all__ = ["StructuralEquivalence"]

from typing import Protocol, runtime_checkable


@runtime_checkable
class StructuralEquivalence(Protocol):
    """Protocol for objects that support structural equivalence checks."""

    def is_structurally_equivalent(self, other: object) -> bool:
        """Return if `self` and `other` are equivalent by structure."""
