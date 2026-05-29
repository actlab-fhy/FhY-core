"""Tests compiler IR traits."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from fhy_core.trait import (
    Canonicalizable,
    CanonicalizableMixin,
    Foldable,
    FoldableMixin,
    HasOperands,
    HasOperandsMixin,
    HasResults,
    HasResultsMixin,
    Rewritable,
    RewritableMixin,
    StructuralEquivalence,
    StructuralEquivalenceMixin,
)


@dataclass
class _OperandNode(HasOperandsMixin[int]):
    _operands: tuple[int, ...]

    def get_operands(self) -> tuple[int, ...]:
        return self._operands


@dataclass
class _ResultNode(HasResultsMixin[int]):
    _results: tuple[int, ...]

    def get_results(self) -> tuple[int, ...]:
        return self._results


@dataclass
class _FoldableNode(FoldableMixin[int]):
    folded_value: int | None

    def fold(self) -> int | None:
        return self.folded_value


@dataclass
class _CanonicalNode(CanonicalizableMixin):
    value: int

    def canonicalize(self) -> bool:
        if self.value < 0:
            self.value = -self.value
            return True
        return False


@dataclass
class _StructEqNode(StructuralEquivalenceMixin):
    opcode: str
    operands: tuple[int, ...]

    def is_structurally_equivalent(self, other: object) -> bool:
        return (
            isinstance(other, _StructEqNode)
            and self.opcode == other.opcode
            and self.operands == other.operands
        )


@dataclass
class _LeafNode(RewritableMixin[int]):
    """Rewritable leaf with no children; uses the default ``rebuild`` behavior."""


@dataclass
class _PairNode(RewritableMixin[int]):
    """Rewritable node with two integer children that destructures on rebuild."""

    left: int
    right: int

    def rebuild_with_visit_children(self, new_children: Sequence[int]) -> "_PairNode":
        left, right = new_children
        return _PairNode(left, right)


@dataclass
class _ListNode(RewritableMixin[int]):
    """Rewritable node with a variable-length child sequence."""

    children: tuple[int, ...] = field(default_factory=tuple)

    def rebuild_with_visit_children(self, new_children: Sequence[int]) -> "_ListNode":
        return _ListNode(tuple(new_children))


def test_has_operands_runtime_protocol() -> None:
    """Test `HasOperands` runtime protocol."""
    node = _OperandNode((1, 2))
    assert isinstance(node, HasOperands)


def test_has_operands_mixin_contract() -> None:
    """Test `HasOperandsMixin` contract."""
    node = _OperandNode((1, 2, 3))
    assert node.get_operands() == (1, 2, 3)


def test_has_results_runtime_protocol() -> None:
    """Test `HasResults` runtime protocol."""
    node = _ResultNode((7,))
    assert isinstance(node, HasResults)


def test_has_results_mixin_contract() -> None:
    """Test `HasResultsMixin` contract."""
    node = _ResultNode((7, 8))
    assert node.get_results() == (7, 8)


def test_foldable_runtime_protocol() -> None:
    """Test `Foldable` runtime protocol."""
    node = _FoldableNode(42)
    assert isinstance(node, Foldable)


def test_foldable_fold_returns_value() -> None:
    """Test `FoldableMixin.fold` returns a value when available."""
    node = _FoldableNode(42)
    assert node.fold() == 42


def test_foldable_fold_returns_none() -> None:
    """Test `FoldableMixin.fold` returns `None` when not foldable."""
    node = _FoldableNode(None)
    assert node.fold() is None


def test_canonicalizable_runtime_protocol() -> None:
    """Test `Canonicalizable` runtime protocol."""
    node = _CanonicalNode(-3)
    assert isinstance(node, Canonicalizable)


def test_canonicalizable_applies_change() -> None:
    """Test `CanonicalizableMixin.canonicalize` reports applied change."""
    node = _CanonicalNode(-3)
    assert node.canonicalize()


def test_canonicalizable_updates_value() -> None:
    """Test `CanonicalizableMixin.canonicalize` updates node state."""
    node = _CanonicalNode(-3)
    node.canonicalize()
    assert node.value == 3


def test_canonicalizable_reports_no_change() -> None:
    """Test `CanonicalizableMixin.canonicalize` reports no change."""
    node = _CanonicalNode(5)
    assert not node.canonicalize()


def test_structural_equivalence_runtime_protocol() -> None:
    """Test `StructuralEquivalence` runtime protocol."""
    node = _StructEqNode("add", (1, 2))
    assert isinstance(node, StructuralEquivalence)


def test_structural_equivalence_true_for_same_structure() -> None:
    """Test structural equivalence is true for identical structure."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("add", (1, 2))
    assert left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_opcode() -> None:
    """Test structural equivalence is false for different opcodes."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("mul", (1, 2))
    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_operands() -> None:
    """Test structural equivalence is false for different operands."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("add", (2, 3))
    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_type() -> None:
    """Test structural equivalence is false for different Python types."""
    node = _StructEqNode("add", (1, 2))
    assert not node.is_structurally_equivalent((1, 2))


def test_rewritable_runtime_protocol() -> None:
    """Test `Rewritable` runtime protocol."""
    node = _PairNode(1, 2)
    assert isinstance(node, Rewritable)


def test_rewritable_mixin_default_returns_self_for_empty_children() -> None:
    """Test the default `rebuild_with_visit_children` returns ``self`` on ``()``."""
    node = _LeafNode()

    rebuilt = node.rebuild_with_visit_children(())

    assert rebuilt is node


def test_rewritable_mixin_default_raises_for_non_empty_children() -> None:
    """Test the default `rebuild_with_visit_children` raises when children are passed.

    A leaf subclass that inherits the default but is handed children
    (which means the caller is treating it as a child-bearing node)
    fails loudly so the bug surfaces at the offending node, not later.
    """
    node = _LeafNode()

    with pytest.raises(NotImplementedError, match="_LeafNode"):
        node.rebuild_with_visit_children((1,))


def test_rewritable_mixin_override_returns_new_instance_with_new_children() -> None:
    """Test a subclass override constructs a fresh instance from ``new_children``."""
    original = _PairNode(1, 2)

    rebuilt = original.rebuild_with_visit_children((10, 20))

    assert isinstance(rebuilt, _PairNode)
    assert rebuilt is not original
    assert rebuilt.left == 10
    assert rebuilt.right == 20


def test_rewritable_mixin_override_preserves_original_node() -> None:
    """Test rebuild does not mutate the original node."""
    original = _PairNode(1, 2)

    original.rebuild_with_visit_children((10, 20))

    assert original.left == 1
    assert original.right == 2


def test_rewritable_mixin_override_handles_variable_length_children() -> None:
    """Test a subclass override accepts an arbitrary-length child sequence."""
    original = _ListNode((1, 2, 3))

    rebuilt = original.rebuild_with_visit_children((10, 20, 30, 40))

    assert isinstance(rebuilt, _ListNode)
    assert rebuilt.children == (10, 20, 30, 40)


def test_rewritable_mixin_override_handles_empty_children() -> None:
    """Test a variable-length override accepts the empty child sequence."""
    original = _ListNode((1, 2, 3))

    rebuilt = original.rebuild_with_visit_children(())

    assert isinstance(rebuilt, _ListNode)
    assert rebuilt.children == ()
