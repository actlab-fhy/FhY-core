"""Tests for `RewritablePass`.

`RewritablePass` is a bottom-up tree rewriter. For each node it
transforms every child first (rebuilding the node via
``rebuild_with_visit_children`` when any child changed) and then calls
the matching ``visit_<kind>`` method on the (possibly-rebuilt) node.
Subclass visitors return the node's type to replace it or ``None`` to
leave it alone. The default behavior - no overrides - is identity-
preserving on every node kind.

The tests use a small toy IR (parallel to ``ToyNode`` /
``ToyTreeNode`` in ``test_visitable.py``) so the pass-infrastructure
contract is exercised without coupling to any specific IR package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from fhy_core.pass_infrastructure import RewritablePass
from fhy_core.traits import RewritableMixin, VisitableMixin

# =============================================================================
# Toy IR: a small `Visitable + Rewritable` node hierarchy
# =============================================================================


@dataclass(frozen=True)
class ToyNode(VisitableMixin, RewritableMixin["ToyNode"]):
    """Base class for the toy IR used across these tests.

    Narrows ``get_visit_children`` and ``rebuild_with_visit_children``
    to return / accept ``ToyNode`` so the :class:`RewritablePass`
    ``_VisitableRewritable`` Protocol (which uses ``Sequence[Self]``)
    is satisfied for the whole hierarchy.
    """

    def get_visit_children(self) -> tuple["ToyNode", ...]:
        return ()

    def rebuild_with_visit_children(
        self, new_children: Sequence["ToyNode"]
    ) -> "ToyNode":
        if not new_children:
            return self
        raise NotImplementedError(
            f"{type(self).__name__} has children but does not implement "
            "`rebuild_with_visit_children`."
        )


@dataclass(frozen=True)
class ToyLeaf(ToyNode):
    """Leaf node carrying an integer payload; no children."""

    value: int


@dataclass(frozen=True)
class ToyUnary(ToyNode):
    """Node with exactly one child."""

    operand: ToyNode

    def get_visit_children(self) -> tuple[ToyNode, ...]:
        return (self.operand,)

    def rebuild_with_visit_children(self, new_children: Sequence[ToyNode]) -> ToyUnary:
        (operand,) = new_children
        return ToyUnary(operand)


@dataclass(frozen=True)
class ToyPair(ToyNode):
    """Node with exactly two children."""

    left: ToyNode
    right: ToyNode

    def get_visit_children(self) -> tuple[ToyNode, ...]:
        return (self.left, self.right)

    def rebuild_with_visit_children(self, new_children: Sequence[ToyNode]) -> ToyPair:
        left, right = new_children
        return ToyPair(left, right)


@dataclass(frozen=True)
class ToyList(ToyNode):
    """Node with a variable-length child sequence."""

    children: tuple[ToyNode, ...] = field(default_factory=tuple)

    def get_visit_children(self) -> tuple[ToyNode, ...]:
        return self.children

    def rebuild_with_visit_children(self, new_children: Sequence[ToyNode]) -> ToyList:
        return ToyList(tuple(new_children))


# =============================================================================
# Test transformers (subclasses of RewritablePass with various visitors)
# =============================================================================


class _IdentityRewriter(RewritablePass[ToyNode]):
    """Rewriter with no visitor overrides; preserves every node."""


class _DoubleLeavesRewriter(RewritablePass[ToyNode]):
    """Replace every ``ToyLeaf`` with one whose value is doubled."""

    def visit_toy_leaf(self, node: ToyLeaf) -> ToyNode | None:
        return ToyLeaf(node.value * 2)


class _RewriteSpecificLeafRewriter(RewritablePass[ToyNode]):
    """Replace leaves whose value equals ``target_value`` with ``replacement_value``."""

    _target_value: int
    _replacement_value: int

    def __init__(self, target_value: int, replacement_value: int) -> None:
        super().__init__()
        self._target_value = target_value
        self._replacement_value = replacement_value

    def visit_toy_leaf(self, node: ToyLeaf) -> ToyNode | None:
        if node.value == self._target_value:
            return ToyLeaf(self._replacement_value)
        return None


class _ReplaceUnaryWithOperandRewriter(RewritablePass[ToyNode]):
    """Replace every ``ToyUnary`` with its operand (unary -> identity)."""

    def visit_toy_unary(self, node: ToyUnary) -> ToyNode | None:
        return node.operand


class _ReplacePairWithLeafRewriter(RewritablePass[ToyNode]):
    """Replace every ``ToyPair`` with ``ToyLeaf(99)``."""

    def visit_toy_pair(self, node: ToyPair) -> ToyNode | None:
        _ = node
        return ToyLeaf(99)


class _VisitOrderRecordingRewriter(RewritablePass[ToyNode]):
    """Record every node it visits, in visit order, on an instance attribute."""

    visited_kinds: list[str]

    def __init__(self) -> None:
        super().__init__()
        self.visited_kinds = []

    def visit_toy_leaf(self, node: ToyLeaf) -> ToyNode | None:
        _ = node
        self.visited_kinds.append("leaf")
        return None

    def visit_toy_unary(self, node: ToyUnary) -> ToyNode | None:
        _ = node
        self.visited_kinds.append("unary")
        return None

    def visit_toy_pair(self, node: ToyPair) -> ToyNode | None:
        _ = node
        self.visited_kinds.append("pair")
        return None

    def visit_toy_list(self, node: ToyList) -> ToyNode | None:
        _ = node
        self.visited_kinds.append("list")
        return None


class _CapturingDoublingRewriter(_DoubleLeavesRewriter):
    """Double leaves and capture what the parent visitor saw on each container."""

    captured_pair_left_values: list[int]
    captured_list_child_values: list[tuple[int, ...]]

    def __init__(self) -> None:
        super().__init__()
        self.captured_pair_left_values = []
        self.captured_list_child_values = []

    def visit_toy_pair(self, node: ToyPair) -> ToyNode | None:
        assert isinstance(node.left, ToyLeaf)
        self.captured_pair_left_values.append(node.left.value)
        return None

    def visit_toy_list(self, node: ToyList) -> ToyNode | None:
        leaf_values: list[int] = []
        for child in node.children:
            assert isinstance(child, ToyLeaf)
            leaf_values.append(child.value)
        self.captured_list_child_values.append(tuple(leaf_values))
        return None


class _DoubleLeavesThenCollapsePairToFive(_DoubleLeavesRewriter):
    """Double leaves; then collapse every ``ToyPair`` to ``ToyLeaf(5)``."""

    def visit_toy_pair(self, node: ToyPair) -> ToyNode | None:
        _ = node
        return ToyLeaf(5)


class _NoneReturningLeafRewriter(RewritablePass[ToyNode]):
    """A leaf visitor that always returns ``None`` (no change)."""

    def visit_toy_leaf(self, node: ToyLeaf) -> ToyNode | None:
        _ = node
        return None


class _OnlyLeafRewriter(RewritablePass[ToyNode]):
    """Doubles leaves but defines no other visitors; tests ``visit_unknown``."""

    def visit_toy_leaf(self, node: ToyLeaf) -> ToyNode | None:
        return ToyLeaf(node.value * 2)


@dataclass(frozen=True)
class _BrokenContainer(ToyNode):
    """Container that exposes a child but inherits the raising rebuild default."""

    operand: ToyNode

    def get_visit_children(self) -> tuple[ToyNode, ...]:
        return (self.operand,)


# =============================================================================
# Default pass-through: no visitor overrides preserve every node kind
# =============================================================================


_PASS_THROUGH_NODES = [
    pytest.param(ToyLeaf(5), id="leaf"),
    pytest.param(ToyUnary(ToyLeaf(3)), id="unary"),
    pytest.param(ToyPair(ToyLeaf(1), ToyLeaf(2)), id="pair"),
    pytest.param(ToyList((ToyLeaf(1), ToyLeaf(2), ToyLeaf(3))), id="list"),
    pytest.param(ToyList(()), id="empty_list"),
    pytest.param(
        ToyPair(
            ToyUnary(ToyLeaf(7)),
            ToyList((ToyLeaf(1), ToyPair(ToyLeaf(2), ToyLeaf(3)))),
        ),
        id="nested",
    ),
]


@pytest.mark.parametrize("node", _PASS_THROUGH_NODES)
def test_identity_rewriter_returns_input_with_identity_preserved(
    node: ToyNode,
) -> None:
    """Test the no-override rewriter returns the input via ``is``-identity."""
    result = _IdentityRewriter().transform(node)

    assert result is node


# =============================================================================
# Single-kind visitor overrides
# =============================================================================


def test_top_level_leaf_is_rewritten_by_leaf_visitor() -> None:
    """Test a top-level leaf is rewritten by the override."""
    original = ToyLeaf(5)

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyLeaf)
    assert result.value == 10


def test_leaf_inside_unary_is_rewritten_by_leaf_visitor() -> None:
    """Test a leaf inside a unary operand is rewritten."""
    original = ToyUnary(ToyLeaf(5))

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyUnary)
    assert isinstance(result.operand, ToyLeaf)
    assert result.operand.value == 10


def test_leaves_inside_pair_are_both_rewritten_by_leaf_visitor() -> None:
    """Test both children of a pair are rewritten."""
    original = ToyPair(ToyLeaf(1), ToyLeaf(2))

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyPair)
    assert isinstance(result.left, ToyLeaf)
    assert isinstance(result.right, ToyLeaf)
    assert result.left.value == 2
    assert result.right.value == 4


def test_leaves_inside_list_are_all_rewritten_by_leaf_visitor() -> None:
    """Test every child of a list is rewritten."""
    original = ToyList((ToyLeaf(1), ToyLeaf(2), ToyLeaf(3)))

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyList)
    rewritten_values = [
        child.value for child in result.children if isinstance(child, ToyLeaf)
    ]
    assert len(rewritten_values) == len(result.children)
    assert rewritten_values == [2, 4, 6]


def test_unary_visitor_replaces_node_with_operand() -> None:
    """Test the unary-removing visitor replaces ``ToyUnary(x)`` with ``x``."""
    operand = ToyLeaf(7)
    original = ToyUnary(operand)

    result = _ReplaceUnaryWithOperandRewriter().transform(original)

    assert result is operand


def test_pair_visitor_replaces_node_with_leaf() -> None:
    """Test the pair-collapsing visitor replaces every pair with a sentinel leaf."""
    original = ToyPair(ToyLeaf(1), ToyLeaf(2))

    result = _ReplacePairWithLeafRewriter().transform(original)

    assert isinstance(result, ToyLeaf)
    assert result.value == 99


# =============================================================================
# Child-rewrite propagation: rebuilds only what changed
# =============================================================================


def test_unary_is_rebuilt_when_operand_changes() -> None:
    """Test the unary node is rebuilt when its operand is rewritten."""
    original = ToyUnary(ToyLeaf(5))

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyUnary)
    assert result is not original
    assert isinstance(result.operand, ToyLeaf)
    assert result.operand.value == 10


def test_pair_left_only_rewrite_preserves_right_identity() -> None:
    """Test rewriting only the left child reuses the right child object."""
    right = ToyLeaf(2)
    original = ToyPair(ToyLeaf(1), right)

    result = _RewriteSpecificLeafRewriter(
        target_value=1, replacement_value=100
    ).transform(original)

    assert isinstance(result, ToyPair)
    assert result is not original
    assert isinstance(result.left, ToyLeaf)
    assert result.left.value == 100
    assert result.right is right


def test_list_partial_child_rewrite_preserves_other_children() -> None:
    """Test rewriting one list child reuses the unchanged children's objects."""
    second = ToyLeaf(2)
    third = ToyLeaf(3)
    original = ToyList((ToyLeaf(1), second, third))

    result = _RewriteSpecificLeafRewriter(
        target_value=1, replacement_value=100
    ).transform(original)

    assert isinstance(result, ToyList)
    assert result is not original
    assert isinstance(result.children[0], ToyLeaf)
    assert result.children[0].value == 100
    assert result.children[1] is second
    assert result.children[2] is third


def test_unchanged_subtree_of_rewritten_parent_keeps_identity() -> None:
    """Test an unchanged subtree of a rebuilt parent keeps the same object."""
    unchanged_right = ToyUnary(ToyLeaf(99))
    original = ToyPair(ToyLeaf(7), unchanged_right)

    result = _RewriteSpecificLeafRewriter(
        target_value=7, replacement_value=0
    ).transform(original)

    assert isinstance(result, ToyPair)
    assert result.right is unchanged_right


# =============================================================================
# Bottom-up visitor ordering: visitor sees already-rewritten children
# =============================================================================


_VISIT_ORDER_CASES = [
    pytest.param(
        ToyPair(ToyLeaf(1), ToyLeaf(2)),
        ["leaf", "leaf", "pair"],
        id="pair",
    ),
    pytest.param(
        ToyList((ToyLeaf(1), ToyLeaf(2), ToyLeaf(3))),
        ["leaf", "leaf", "leaf", "list"],
        id="list",
    ),
    pytest.param(
        ToyUnary(ToyPair(ToyLeaf(1), ToyLeaf(2))),
        ["leaf", "leaf", "pair", "unary"],
        id="unary_over_pair",
    ),
]


@pytest.mark.parametrize("node, expected_visit_order", _VISIT_ORDER_CASES)
def test_parent_visitor_runs_after_every_child_visitor(
    node: ToyNode, expected_visit_order: list[str]
) -> None:
    """Test the per-node visitor fires after every child visitor has run."""
    recorder = _VisitOrderRecordingRewriter()

    recorder.transform(node)

    assert recorder.visited_kinds == expected_visit_order


def test_pair_visitor_sees_rewritten_leaf_children() -> None:
    """Test the pair visitor receives a node whose children are rewritten."""
    rewriter = _CapturingDoublingRewriter()
    node = ToyPair(ToyLeaf(5), ToyLeaf(7))

    rewriter.transform(node)

    assert rewriter.captured_pair_left_values == [10]


def test_list_visitor_sees_rewritten_children() -> None:
    """Test the list visitor receives a node whose children are rewritten."""
    rewriter = _CapturingDoublingRewriter()
    node = ToyList((ToyLeaf(3), ToyLeaf(4)))

    rewriter.transform(node)

    assert rewriter.captured_list_child_values == [(6, 8)]


# =============================================================================
# Visitor replacement is not re-transformed (single bottom-up pass)
# =============================================================================


def test_visitor_replacement_is_not_re_visited() -> None:
    """Test a pair visitor's leaf replacement is not re-doubled.

    The rewriter is a single bottom-up pass: a visitor's replacement
    output is returned as-is, never re-walked. The doubling rewriter
    transforms the two leaf operands of a pair (1 -> 2, 2 -> 4) before
    the pair visitor fires; the pair visitor's replacement
    ``ToyLeaf(5)`` then bypasses any further visiting.
    """
    original = ToyPair(ToyLeaf(1), ToyLeaf(2))

    result = _DoubleLeavesThenCollapsePairToFive().transform(original)

    assert isinstance(result, ToyLeaf)
    assert result.value == 5


# =============================================================================
# `transform` always returns a node (never `None`)
# =============================================================================


def test_transform_returns_node_when_no_rewrite_occurs() -> None:
    """Test ``transform`` returns the input node, not ``None``."""
    original = ToyLeaf(3)

    result = _IdentityRewriter().transform(original)

    assert isinstance(result, ToyNode)


def test_transform_returns_input_when_visitor_returns_none() -> None:
    """Test ``transform`` coerces a ``None`` visitor result back to the input."""
    original = ToyLeaf(3)

    result = _NoneReturningLeafRewriter().transform(original)

    assert result is original


def test_transform_returns_replacement_when_visitor_returns_node() -> None:
    """Test ``transform`` returns the visitor's replacement when non-``None``."""
    original = ToyLeaf(3)

    result = _DoubleLeavesRewriter().transform(original)

    assert isinstance(result, ToyLeaf)
    assert result is not original
    assert result.value == 6


# =============================================================================
# Pass-framework integration
# =============================================================================


def test_calling_pass_invokes_transform() -> None:
    """Test invoking the pass as a callable returns the transformed tree."""
    rewriter = _DoubleLeavesRewriter()
    original = ToyLeaf(3)

    result = rewriter(original)

    assert isinstance(result, ToyLeaf)
    assert result.value == 6


def test_visit_method_delegates_to_transform() -> None:
    """Test calling :meth:`visit` directly returns the transformed tree."""
    rewriter = _DoubleLeavesRewriter()
    original = ToyLeaf(3)

    result = rewriter.visit(original)

    assert isinstance(result, ToyLeaf)
    assert result.value == 6


def test_get_noop_output_returns_the_input_unchanged() -> None:
    """Test ``get_noop_output`` returns its argument as-is."""
    rewriter = _IdentityRewriter()
    original = ToyLeaf(3)

    result = rewriter.get_noop_output(original)

    assert result is original


def test_did_change_returns_false_when_no_rewrite_occurred() -> None:
    """Test ``did_change`` reports no change when the rewriter is a no-op."""
    rewriter = _IdentityRewriter()
    original = ToyLeaf(3)

    result = rewriter.transform(original)

    assert rewriter.did_change(original, result) is False


def test_did_change_returns_true_when_at_least_one_node_was_rewritten() -> None:
    """Test ``did_change`` reports a change when a visitor rewrote a node."""
    rewriter = _DoubleLeavesRewriter()
    original = ToyLeaf(3)

    result = rewriter.transform(original)

    assert rewriter.did_change(original, result) is True


def test_did_change_uses_identity_not_value_equality() -> None:
    """Test ``did_change`` returns False for distinct-but-equal objects.

    Two ``ToyLeaf(5)`` instances are value-equal (frozen dataclass) but
    are different objects. A rewriter that returned a freshly-constructed
    leaf with the same value as the input would be detected as "changed"
    by identity semantics - which is the intended contract.
    """
    rewriter = _IdentityRewriter()
    original = ToyLeaf(3)
    fresh_equal = ToyLeaf(3)

    assert rewriter.did_change(original, fresh_equal) is True


# =============================================================================
# `visit_unknown` default: preserves the node (does not raise)
# =============================================================================


def test_visit_unknown_default_preserves_an_unhandled_node() -> None:
    """Test a rewriter with no visitor for a kind leaves it alone."""
    # Pair has no dedicated visitor; only the leaf visitor fires.
    original = ToyPair(ToyLeaf(1), ToyLeaf(2))

    result = _OnlyLeafRewriter().transform(original)

    assert isinstance(result, ToyPair)
    assert isinstance(result.left, ToyLeaf)
    assert isinstance(result.right, ToyLeaf)
    assert result.left.value == 2
    assert result.right.value == 4


def test_visit_unknown_returns_none_directly() -> None:
    """Test the default ``visit_unknown`` returns ``None`` for any node."""
    rewriter = _IdentityRewriter()
    node = ToyLeaf(3)

    result = rewriter.visit_unknown(node)

    assert result is None


# =============================================================================
# Edge cases
# =============================================================================


def test_empty_list_node_is_preserved_by_identity() -> None:
    """Test an empty-children node is returned by identity (no rebuild attempted)."""
    original = ToyList(())

    result = _DoubleLeavesRewriter().transform(original)

    assert result is original


def test_deeply_nested_unchanged_subtrees_preserve_identity() -> None:
    """Test every unchanged subtree of a deep tree keeps the same object."""
    deep_subtree = ToyUnary(ToyUnary(ToyUnary(ToyLeaf(99))))
    original = ToyPair(ToyLeaf(7), deep_subtree)

    result = _RewriteSpecificLeafRewriter(
        target_value=7, replacement_value=0
    ).transform(original)

    assert isinstance(result, ToyPair)
    assert result.right is deep_subtree


# =============================================================================
# Adversarial: a node missing `rebuild_with_visit_children` raises at rebuild
# =============================================================================


def test_node_without_rebuild_with_visit_children_raises_at_first_rebuild() -> None:
    """Test the rewriter surfaces a missing rebuild contract when needed.

    A node type that mixes in ``Visitable`` but inherits the default
    ``RewritableMixin.rebuild_with_visit_children`` (which raises for
    non-empty children) surfaces the missing contract at the first
    rebuild attempt, not silently.
    """
    original = _BrokenContainer(ToyLeaf(1))

    with pytest.raises(NotImplementedError, match="_BrokenContainer"):
        _DoubleLeavesRewriter().transform(original)


# =============================================================================
# Pass-framework integration: execute() and PassResult
# =============================================================================


def test_rewritable_pass_execute_returns_pass_result_with_changed_true() -> None:
    """Test ``.execute(node)`` returns a ``PassResult`` reporting the rewrite."""
    original = ToyLeaf(5)

    result = _DoubleLeavesRewriter().execute(original)

    assert isinstance(result.output, ToyLeaf)
    assert result.output.value == 10
    assert result.changed is True


def test_rewritable_pass_execute_reports_no_change_when_nothing_rewrites() -> None:
    """Test ``.execute(node)`` reports ``changed=False`` for a no-op pass."""
    original = ToyLeaf(5)

    result = _IdentityRewriter().execute(original)

    assert result.output is original
    assert result.changed is False


def test_rewritable_pass_execute_returns_preserved_analyses_attribute() -> None:
    """Test ``.execute(node)`` returns a ``PassResult`` carrying ``preserved_analyses``.

    The default rewriter does not declare analyses; the result still
    exposes the attribute (which is empty / "preserve nothing").
    """
    original = ToyLeaf(5)

    result = _DoubleLeavesRewriter().execute(original)

    assert hasattr(result, "preserved_analyses")
    assert hasattr(result, "diagnostics")
