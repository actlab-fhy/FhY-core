"""Trait for IR nodes that can be visited by passes.

``Visitable`` is the structural contract (a :class:`typing.Protocol`).
``VisitableMixin`` carries the default implementation that IR node classes
inherit.
"""

__all__ = ["Visitable", "VisitableMixin"]

import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from fhy_core.utils.override import override

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


@runtime_checkable
class Visitable(Protocol):
    """Protocol for nodes that support pass visitor dispatch."""

    @classmethod
    def get_visit_method_suffix(cls) -> str:
        """Return the visitor dispatch suffix for this node type."""
        ...

    def accept(self, visitor: _SupportsVisit) -> Any:
        """Accept a visitor and return its visit result."""
        ...

    def get_visit_children(self) -> Sequence["Visitable"]:
        """Return child nodes for traversal-aware visitors.

        Override to expose ordered children for automatic pre-order/post-order
        traversal; the default implementation reports no children.
        """
        ...


class VisitableMixin(Visitable):
    """Default ``Visitable`` behavior for IR node classes."""

    @classmethod
    @override
    def get_visit_method_suffix(cls) -> str:
        return _camel_to_snake(cls.__name__)

    @override
    def accept(self, visitor: _SupportsVisit) -> Any:
        return visitor.visit(self)

    @override
    def get_visit_children(self) -> Sequence["Visitable"]:
        return ()
