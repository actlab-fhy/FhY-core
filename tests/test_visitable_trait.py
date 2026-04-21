"""Tests the visitable trait and generic visitable pass."""

from __future__ import annotations

from fhy_core.pass_infrastructure import (
    AnalysisVisitablePass,
    TraversalOrder,
    VisitablePass,
)
from fhy_core.trait import VisitableMixin


class ToyNode(VisitableMixin):
    """Toy visitable node for test coverage."""


class ToyNodeConsumer(VisitablePass[ToyNode, int]):
    """Test pass that handles only ToyNode."""

    def get_noop_output(self, ir: ToyNode) -> int:
        _ = ir
        return 0

    def visit_toy_node(self, node: ToyNode) -> int:
        _ = node
        return 7


class ToyTreeNode(VisitableMixin):
    """Toy tree node that can optionally expose visit children."""

    name: str
    children: tuple["ToyTreeNode", ...]

    def __init__(self, name: str, children: tuple["ToyTreeNode", ...] = ()) -> None:
        self.name = name
        self.children = children

    def get_visit_children(self) -> tuple["ToyTreeNode", ...]:
        return self.children


class ToyOpaqueNode(VisitableMixin):
    """Toy node with a child reference but no child enumeration override."""

    name: str
    child: "ToyOpaqueNode | None"

    def __init__(self, name: str, child: "ToyOpaqueNode | None" = None) -> None:
        self.name = name
        self.child = child


class ToyTreePreOrderConsumer(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass that records preorder visit order."""

    order: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.PRE)
        self.order = []

    def visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.order.append(node.name)


class ToyTreePostOrderConsumer(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass that records postorder visit order."""

    order: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.POST)
        self.order = []

    def visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.order.append(node.name)


class ToyOpaqueConsumer(AnalysisVisitablePass[ToyOpaqueNode]):
    """Analysis pass for opaque nodes without child enumeration."""

    order: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.PRE)
        self.order = []

    def visit_toy_opaque_node(self, node: ToyOpaqueNode) -> None:
        self.order.append(node.name)


def test_visitable_default_kind_is_snake_case_class_name() -> None:
    """Test that the default visitor kind is snake case class name."""
    assert ToyNode.get_visit_method_suffix() == "toy_node"


def test_visitable_accept_uses_visitor_dispatch() -> None:
    """Test that `accept` routes through the visitor dispatch API."""
    visitor = ToyNodeConsumer()
    assert ToyNode().accept(visitor) == 7  # type: ignore[arg-type]  # test: narrow visitor type


def test_visitable_pass_execute_uses_run_pass_dispatch() -> None:
    """Test that execute dispatches through visit methods."""
    visitor = ToyNodeConsumer()
    result = visitor.execute(ToyNode())
    assert result.output == 7
    assert result.changed is True


def test_analysis_visitable_pass_supports_pre_order_traversal() -> None:
    """Test automatic child traversal in preorder."""
    tree = ToyTreeNode(
        "root",
        (
            ToyTreeNode("left", (ToyTreeNode("left_leaf"),)),
            ToyTreeNode("right"),
        ),
    )
    visitor = ToyTreePreOrderConsumer()
    result = visitor.execute(tree)

    assert result.output is None
    assert visitor.order == ["root", "left", "left_leaf", "right"]


def test_analysis_visitable_pass_supports_post_order_traversal() -> None:
    """Test automatic child traversal in postorder."""
    tree = ToyTreeNode(
        "root",
        (
            ToyTreeNode("left", (ToyTreeNode("left_leaf"),)),
            ToyTreeNode("right"),
        ),
    )
    visitor = ToyTreePostOrderConsumer()
    result = visitor.execute(tree)

    assert result.output is None
    assert visitor.order == ["left_leaf", "left", "right", "root"]


def test_analysis_visitable_pass_requires_node_child_enumeration() -> None:
    """Test that missing child enumeration prevents automatic recursion."""
    node = ToyOpaqueNode("root", ToyOpaqueNode("child"))
    visitor = ToyOpaqueConsumer()
    result = visitor.execute(node)

    assert result.output is None
    assert visitor.order == ["root"]
