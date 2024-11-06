"""Partially ordered set (poset) utility."""

__all__ = ["PartiallyOrderedSet"]

from typing import Generic, Iterator, TypeVar

import networkx as nx  # type: ignore

T = TypeVar("T")


class PartiallyOrderedSet(Generic[T]):
    """A partially ordered set (poset)."""

    _graph: nx.DiGraph

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    def __contains__(self, element: T) -> bool:
        is_contain = self._graph.has_node(element)
        if not isinstance(is_contain, bool):
            raise TypeError(f"Expected bool, but got {type(is_contain)}")
        return is_contain

    def __iter__(self) -> Iterator[T]:
        return iter(nx.topological_sort(self._graph))

    def __len__(self) -> int:
        return len(self._graph.nodes)

    def add_element(self, element: T) -> None:
        """Add an element to the poset.

        Args:
            element: The element to add.

        Raises:
            ValueError: If the element is already a member of the poset.

        """
        self._check_element_not_in_poset(element)
        self._graph.add_node(element)

    def add_order(self, lower: T, upper: T) -> None:
        """Add an order relation between two elements.

        Args:
            lower: The lesser element.
            upper: The greater element.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.
            RuntimeError: If an order relation already exists between lower and upper.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        if nx.has_path(self._graph, upper, lower):
            raise RuntimeError(
                f"Expected no order between {lower} and {upper}, but found one."
            )
        self._graph.add_edge(lower, upper)

    def is_less_than(self, lower: T, upper: T) -> bool:
        """Check if one element is less than another.

        Args:
            lower: The postulated lesser element.
            upper: The postulated greater element.

        Returns:
            bool: True if lower is less than upper, False otherwise.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        has_path = nx.has_path(self._graph, lower, upper)
        if not isinstance(has_path, bool):
            raise TypeError(f"Expected bool, but got {type(has_path)}")
        return has_path

    def is_greater_than(self, lower: T, upper: T) -> bool:
        """Check if one element is greater than another.

        Args:
            lower: The postulated greater element.
            upper: The postulated lesser element.

        Returns:
            bool: True if lower is greater than upper, False otherwise.

        Raises:
            ValueError: If either lower or upper is not a member of the poset.

        """
        self._check_element_in_poset(lower)
        self._check_element_in_poset(upper)
        has_path = nx.has_path(self._graph, upper, lower)
        if not isinstance(has_path, bool):
            raise TypeError(f"Expected bool, but got {type(has_path)}")
        return has_path

    def _check_element_not_in_poset(self, element: T) -> None:
        if element in self:
            raise ValueError(
                f"Expected {element} to not be a member of the poset, but it is."
            )

    def _check_element_in_poset(self, element: T) -> None:
        if element not in self:
            raise ValueError(
                f"Expected {element} to be a member of the poset, but it is not."
            )
