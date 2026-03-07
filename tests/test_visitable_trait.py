"""Tests the visitable trait and generic visitable pass."""

from __future__ import annotations

from fhy_core.pass_infrastructure import VisitablePass
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


def test_visitable_default_kind_is_snake_case_class_name() -> None:
    """Test that the default visitor kind is snake case class name."""
    assert ToyNode.get_visit_method_suffix() == "toy_node"


def test_visitable_accept_uses_visitor_dispatch() -> None:
    """Test that `accept` routes through the visitor dispatch API."""
    visitor = ToyNodeConsumer()
    assert ToyNode().accept(visitor) == 7


def test_visitable_pass_execute_uses_run_pass_dispatch() -> None:
    """Test that execute dispatches through visit methods."""
    visitor = ToyNodeConsumer()
    result = visitor.execute(ToyNode())
    assert result.output == 7
    assert result.changed is True
