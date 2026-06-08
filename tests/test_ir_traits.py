"""Tests compiler IR traits."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

import fhy_core.traits as traits_package
from fhy_core.traits import (
    Canonicalizable,
    HasOperands,
    HasResults,
    Rewritable,
    RewritableMixin,
    StructuralEquivalence,
)


@dataclass
class _OperandNode(HasOperands[int]):
    _operands: tuple[int, ...]

    def get_operands(self) -> tuple[int, ...]:
        return self._operands


@dataclass
class _ResultNode(HasResults[int]):
    _results: tuple[int, ...]

    def get_results(self) -> tuple[int, ...]:
        return self._results


@dataclass
class _MutableCanonicalNode(Canonicalizable):
    """Mutable node: canonicalizes in place and returns ``None``."""

    value: int

    def canonicalize(self) -> "_MutableCanonicalNode | None":
        if self.value < 0:
            self.value = -self.value
        return None


@dataclass(frozen=True)
class _FrozenCanonicalNode(Canonicalizable):
    """Frozen node: returns a canonicalized copy, never mutating ``self``."""

    value: int

    def canonicalize(self) -> "_FrozenCanonicalNode":
        return _FrozenCanonicalNode(abs(self.value))


@dataclass
class _StructEqNode(StructuralEquivalence):
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


def test_has_operands_returns_operands() -> None:
    """Test `HasOperands` protocol method."""
    node = _OperandNode((1, 2, 3))
    assert node.get_operands() == (1, 2, 3)


def test_has_results_runtime_protocol() -> None:
    """Test `HasResults` runtime protocol."""
    node = _ResultNode((7,))
    assert isinstance(node, HasResults)


def test_has_results_returns_results() -> None:
    """Test `HasResults` protocol method."""
    node = _ResultNode((7, 8))
    assert node.get_results() == (7, 8)


def test_foldable_is_removed_from_public_api() -> None:
    """Test `Foldable` is no longer exported by the traits package."""
    assert not hasattr(traits_package, "Foldable")
    assert "Foldable" not in traits_package.__all__


def test_canonicalizable_runtime_protocol() -> None:
    """Test `Canonicalizable` runtime protocol for mutable and frozen nodes."""
    assert isinstance(_MutableCanonicalNode(-3), Canonicalizable)
    assert isinstance(_FrozenCanonicalNode(-3), Canonicalizable)


def test_mutable_canonicalize_mutates_in_place_and_returns_none() -> None:
    """Test a mutable node canonicalizes in place and returns `None`."""
    node = _MutableCanonicalNode(-3)

    result = node.canonicalize()

    assert result is None
    assert node.value == 3


def test_frozen_canonicalize_returns_copy_without_mutating_self() -> None:
    """Test a frozen node returns a new canonicalized copy, leaving self intact."""
    node = _FrozenCanonicalNode(-3)

    result = node.canonicalize()

    assert result is not node
    assert result.value == 3
    assert node.value == -3


def test_frozen_canonicalize_of_canonical_node_returns_equal_object() -> None:
    """Test canonicalizing an already-canonical frozen node yields an equal node."""
    node = _FrozenCanonicalNode(5)

    assert node.canonicalize() == _FrozenCanonicalNode(5)


def test_canonicalize_caller_idiom_yields_canonical_form_for_both() -> None:
    """Test the uniform `obj.canonicalize() or obj` idiom across mutability.

    Mutable objects return ``None`` (so the idiom falls back to the mutated
    receiver); frozen objects return the canonical copy. Either way the
    expression evaluates to the canonical form.
    """
    mutable = _MutableCanonicalNode(-3)
    frozen = _FrozenCanonicalNode(-3)

    assert (mutable.canonicalize() or mutable).value == 3
    assert (frozen.canonicalize() or frozen).value == 3


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
