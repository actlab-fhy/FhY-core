"""Lattice (order theory) utility."""

__all__ = ["Lattice"]

from typing import Generic, TypeVar

from fhy_core.trait.verifiable import VerifiableMixin, VerificationError
from fhy_core.utils.poset import PartiallyOrderedSet

T = TypeVar("T")


class Lattice(Generic[T], VerifiableMixin):
    """Lattice (order theory)."""

    _poset: PartiallyOrderedSet[T]

    def __init__(self) -> None:
        self._poset = PartiallyOrderedSet[T]()

    def __contains__(self, element: T) -> bool:
        return element in self._poset

    def add_element(self, element: T) -> None:
        """Add an element to the lattice.

        Args:
            element: The element to add.

        Raises:
            ValueError: If the element is already in the lattice.

        """
        self._poset.add_element(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements.

        Args:
            lower: The lesser element.
            upper: The greater element.

        Raises:
            ValueError: If either the lower or upper element is not in the
                lattice.
            RuntimeError: If the order relation is invalid.

        """
        self._poset.add_order(lower, upper)

    def is_lattice(self) -> bool:
        """Return True if the lattice is a valid lattice, False otherwise."""
        for x in self._poset:
            for y in self._poset:
                if not self.has_meet(x, y) or not self.has_join(x, y):
                    return False
        return True

    def verify(self) -> None:
        """Verify that this structure is a valid lattice.

        A valid lattice is a poset in which every pair of elements has both a
        unique greatest lower bound (meet) and a unique least upper bound
        (join).

        Raises:
            VerificationError: If some pair of elements lacks a unique meet or
                a unique join.

        """
        for x in self._poset:
            for y in self._poset:
                if self.get_meet(x, y) is None:
                    raise VerificationError(
                        f"Lattice has no unique meet for elements {x!r} and {y!r}."
                    )
                if self.get_join(x, y) is None:
                    raise VerificationError(
                        f"Lattice has no unique join for elements {x!r} and {y!r}."
                    )

    def has_meet(self, x: T, y: T) -> bool:
        """Check if two elements have a greatest lower bound.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            True if x and y have a greatest lower bound, False otherwise.

        """
        return self.get_meet(x, y) is not None

    def has_join(self, x: T, y: T) -> bool:
        """Check if two elements have a least upper bound.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            True if x and y have a least upper bound, False otherwise.

        """
        return self.get_join(x, y) is not None

    def get_least_upper_bound(self, x: T, y: T) -> T:
        """Get the least upper bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            The least upper bound of x and y.

        Raises:
            RuntimeError: If the least upper bound does not exist.

        """
        join = self.get_join(x, y)
        if join is None:
            raise RuntimeError(
                f"No least upper bound of {x} and {y} found for lattice."
            )
        return join

    def get_meet(self, x: T, y: T) -> T | None:
        """Get the greatest lower bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            The greatest lower bound of x and y, or None if none exists or if
            multiple incomparable maximal lower bounds exist.

        """
        lower_bounds = [
            z
            for z in self._poset
            if self._poset.is_less_than(z, x) and self._poset.is_less_than(z, y)
        ]
        maximal = [
            z
            for z in lower_bounds
            if not any(z != w and self._poset.is_less_than(z, w) for w in lower_bounds)
        ]
        if len(maximal) == 1:
            return maximal[0]
        return None

    def get_join(self, x: T, y: T) -> T | None:
        """Get the least upper bound of two elements.

        Args:
            x: The first element.
            y: The second element.

        Returns:
            The least upper bound of x and y, or None if none exists or if
            multiple incomparable minimal upper bounds exist.

        """
        upper_bounds = [
            z
            for z in self._poset
            if self._poset.is_greater_than(z, x) and self._poset.is_greater_than(z, y)
        ]
        minimal = [
            z
            for z in upper_bounds
            if not any(
                z != w and self._poset.is_greater_than(z, w) for w in upper_bounds
            )
        ]
        if len(minimal) == 1:
            return minimal[0]
        return None
