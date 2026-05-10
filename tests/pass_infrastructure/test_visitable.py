"""Tests the visitable trait and generic visitable pass."""

from __future__ import annotations

import pytest

from fhy_core.pass_infrastructure import (
    AnalysisVisitablePass,
    PassExecutionError,
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


# ---------------------------------------------------------------------------
# AnalysisVisitablePass dispatches per-node before_visit_<suffix> /
# after_visit_<suffix> hooks; visit_unknown defaults.
# ---------------------------------------------------------------------------


class HookRecordingPreOrderConsumer(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass that records every dispatched hook in PRE order."""

    events: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.PRE)
        self.events = []

    def before_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"before:{node.name}")

    def visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"visit:{node.name}")

    def after_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"after:{node.name}")


class HookRecordingPostOrderConsumer(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass that records every dispatched hook in POST order."""

    events: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.POST)
        self.events = []

    def before_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"before:{node.name}")

    def visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"visit:{node.name}")

    def after_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"after:{node.name}")


def test_analysis_visitable_pass_dispatches_per_node_hooks_in_pre_order() -> None:
    """Test that PRE-order traversal calls before/visit/after for every node."""
    tree = ToyTreeNode("root", (ToyTreeNode("left"), ToyTreeNode("right")))
    visitor = HookRecordingPreOrderConsumer()

    visitor.execute(tree)

    assert visitor.events == [
        "before:root",
        "visit:root",
        "before:left",
        "visit:left",
        "after:left",
        "before:right",
        "visit:right",
        "after:right",
        "after:root",
    ]


def test_analysis_visitable_pass_dispatches_per_node_hooks_in_post_order() -> None:
    """Test POST-order: before, walk children, visit, after."""
    tree = ToyTreeNode("root", (ToyTreeNode("left"), ToyTreeNode("right")))
    visitor = HookRecordingPostOrderConsumer()

    visitor.execute(tree)

    assert visitor.events == [
        "before:root",
        "before:left",
        "visit:left",
        "after:left",
        "before:right",
        "visit:right",
        "after:right",
        "visit:root",
        "after:root",
    ]


class UnknownHookFallbackConsumer(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass that records `before_visit_unknown` / `after_visit_unknown`
    fallbacks (no dedicated `*_toy_tree_node` hooks)."""

    events: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.PRE)
        self.events = []

    def before_visit_unknown(self, node: ToyTreeNode) -> None:
        self.events.append(f"before-unknown:{node.name}")

    def after_visit_unknown(self, node: ToyTreeNode) -> None:
        self.events.append(f"after-unknown:{node.name}")


def test_analysis_visitable_pass_falls_back_to_unknown_pre_post_hooks() -> None:
    """Test that nodes without dedicated hooks fall back to `*_unknown`."""
    tree = ToyTreeNode("root", (ToyTreeNode("child"),))
    visitor = UnknownHookFallbackConsumer()

    visitor.execute(tree)

    assert visitor.events == [
        "before-unknown:root",
        "before-unknown:child",
        "after-unknown:child",
        "after-unknown:root",
    ]


# ---------------------------------------------------------------------------
# visit_unknown defaults differ between VisitablePass (raise) and
# AnalysisVisitablePass (no-op).
# ---------------------------------------------------------------------------


class UnhandledNode(VisitableMixin):
    """A node type for which the pass below has no `visit_<suffix>` method."""


class HandlesNothingVisitablePass(VisitablePass[UnhandledNode, int]):
    """VisitablePass with no `visit_<suffix>` for `UnhandledNode`."""

    def get_noop_output(self, ir: UnhandledNode) -> int:
        return 0


class HandlesNothingAnalysisPass(AnalysisVisitablePass[UnhandledNode]):
    """AnalysisVisitablePass with no `visit_<suffix>` for `UnhandledNode`."""


def test_visitable_pass_unknown_node_raises_not_implemented_error() -> None:
    """Test that VisitablePass.visit_unknown raises NotImplementedError by default.

    The unwrapped exception inside `run_pass`/`visit` is wrapped to
    `PassExecutionError` by `execute`. The original `NotImplementedError`
    is the cause and the message identifies the pass and node type.
    """
    visitor = HandlesNothingVisitablePass()

    with pytest.raises(PassExecutionError, match="UnhandledNode"):
        visitor.execute(UnhandledNode())


def test_analysis_visitable_pass_unknown_node_is_noop() -> None:
    """Test that AnalysisVisitablePass.visit_unknown is a no-op by default."""
    visitor = HandlesNothingAnalysisPass()

    result = visitor.execute(UnhandledNode())

    assert result.output is None


# ---------------------------------------------------------------------------
# walk's try/finally guarantees after_visit runs even when visit raises.
# ---------------------------------------------------------------------------


class RaisingVisitPass(AnalysisVisitablePass[ToyTreeNode]):
    """Analysis pass whose `visit_<suffix>` raises mid-walk."""

    events: list[str]

    def __init__(self) -> None:
        super().__init__(TraversalOrder.PRE)
        self.events = []

    def before_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"before:{node.name}")

    def visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"visit:{node.name}")
        if node.name == "boom":
            raise RuntimeError("visit-broken")

    def after_visit_toy_tree_node(self, node: ToyTreeNode) -> None:
        self.events.append(f"after:{node.name}")


def test_walk_runs_after_visit_when_visit_raises() -> None:
    """Test that `after_visit_<X>` fires even when `visit_<X>` raises.

    The underlying exception is wrapped to `PassExecutionError`.
    """
    tree = ToyTreeNode("root", (ToyTreeNode("boom"),))
    visitor = RaisingVisitPass()

    with pytest.raises(PassExecutionError, match="visit-broken"):
        visitor.execute(tree)

    # before:root, visit:root, before:boom, visit:boom (raises),
    # after:boom (try/finally), after:root (try/finally).
    assert visitor.events == [
        "before:root",
        "visit:root",
        "before:boom",
        "visit:boom",
        "after:boom",
        "after:root",
    ]
