"""Trait for IR nodes that can be visited by passes."""

__all__ = ["Visitable", "VisitableMixin"]

from abc import ABC
from typing import Any, Protocol


def _camel_to_snake(text: str) -> str:
    chars: list[str] = []
    for index, char in enumerate(text):
        if index > 0 and char.isupper():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


class _SupportsVisit(Protocol):
    def visit(self, node: "Visitable") -> Any: ...


class Visitable(ABC):
    """Interface for nodes that support pass visitor dispatch."""

    @classmethod
    def get_visit_method_suffix(cls) -> str:
        """Return visitor dispatch suffix for this node type."""
        return _camel_to_snake(cls.__name__)

    def accept(self, visitor: _SupportsVisit) -> Any:
        """Accept a visitor and return its visit result.

        Args:
            visitor: Visitor to accept, which must support visiting this node's type.

        Returns:
            The result of the visitor's visit method for this node.

        """
        return visitor.visit(self)


class VisitableMixin(Visitable):
    """Default `Visitable` behavior for IR node classes."""
