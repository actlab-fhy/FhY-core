"""Trait for IR nodes that can be visited by passes."""

__all__ = ["Visitable", "VisitableMixin"]

import re
from abc import ABC
from collections.abc import Sequence
from typing import Any, Protocol

_CAMEL_SPLIT_BEFORE_UPPER_LOWER = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_SPLIT_AT_LOWER_THEN_UPPER = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(text: str) -> str:
    """Convert a ``CamelCase`` or ``camelCase`` identifier to ``snake_case``.

    Acronyms collapse to a single segment: ``IOError`` becomes ``io_error``,
    ``XMLParser`` becomes ``xml_parser``, ``ABCExpression`` becomes
    ``abc_expression``.
    """
    text = _CAMEL_SPLIT_BEFORE_UPPER_LOWER.sub(r"\1_\2", text)
    text = _CAMEL_SPLIT_AT_LOWER_THEN_UPPER.sub(r"\1_\2", text)
    return text.lower()


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

    def get_visit_children(self) -> Sequence["Visitable"]:
        """Return child nodes for traversal-aware visitors.

        Nodes can override this to expose ordered children for automatic
        pre-order/post-order traversal. The default implementation reports no
        children.
        """
        return ()


class VisitableMixin(Visitable):
    """Default `Visitable` behavior for IR node classes."""
