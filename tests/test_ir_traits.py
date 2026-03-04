"""Tests compiler IR traits."""

from dataclasses import dataclass

from fhy_core.trait import (
    Canonicalizable,
    CanonicalizableMixin,
    Foldable,
    FoldableMixin,
    HasOperands,
    HasOperandsMixin,
    HasResults,
    HasResultsMixin,
    StructuralEquivalence,
    StructuralEquivalenceMixin,
)


@dataclass
class _OperandNode(HasOperandsMixin[int]):
    _operands: tuple[int, ...]

    @property
    def operands(self) -> tuple[int, ...]:
        return self._operands


@dataclass
class _ResultNode(HasResultsMixin[int]):
    _results: tuple[int, ...]

    @property
    def results(self) -> tuple[int, ...]:
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


def test_has_operands_runtime_protocol():
    """Test `HasOperands` runtime protocol."""
    node = _OperandNode((1, 2))
    assert isinstance(node, HasOperands)


def test_has_operands_mixin_contract():
    """Test `HasOperandsMixin` contract."""
    node = _OperandNode((1, 2, 3))
    assert node.operands == (1, 2, 3)


def test_has_results_runtime_protocol():
    """Test `HasResults` runtime protocol."""
    node = _ResultNode((7,))
    assert isinstance(node, HasResults)


def test_has_results_mixin_contract():
    """Test `HasResultsMixin` contract."""
    node = _ResultNode((7, 8))
    assert node.results == (7, 8)


def test_foldable_runtime_protocol():
    """Test `Foldable` runtime protocol."""
    node = _FoldableNode(42)
    assert isinstance(node, Foldable)


def test_foldable_fold_returns_value():
    """Test `FoldableMixin.fold` returns a value when available."""
    node = _FoldableNode(42)
    assert node.fold() == 42


def test_foldable_fold_returns_none():
    """Test `FoldableMixin.fold` returns `None` when not foldable."""
    node = _FoldableNode(None)
    assert node.fold() is None


def test_canonicalizable_runtime_protocol():
    """Test `Canonicalizable` runtime protocol."""
    node = _CanonicalNode(-3)
    assert isinstance(node, Canonicalizable)


def test_canonicalizable_applies_change():
    """Test `CanonicalizableMixin.canonicalize` reports applied change."""
    node = _CanonicalNode(-3)
    assert node.canonicalize()


def test_canonicalizable_updates_value():
    """Test `CanonicalizableMixin.canonicalize` updates node state."""
    node = _CanonicalNode(-3)
    node.canonicalize()
    assert node.value == 3


def test_canonicalizable_reports_no_change():
    """Test `CanonicalizableMixin.canonicalize` reports no change."""
    node = _CanonicalNode(5)
    assert not node.canonicalize()


def test_structural_equivalence_runtime_protocol():
    """Test `StructuralEquivalence` runtime protocol."""
    node = _StructEqNode("add", (1, 2))
    assert isinstance(node, StructuralEquivalence)


def test_structural_equivalence_true_for_same_structure():
    """Test structural equivalence is true for identical structure."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("add", (1, 2))
    assert left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_opcode():
    """Test structural equivalence is false for different opcodes."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("mul", (1, 2))
    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_operands():
    """Test structural equivalence is false for different operands."""
    left = _StructEqNode("add", (1, 2))
    right = _StructEqNode("add", (2, 3))
    assert not left.is_structurally_equivalent(right)


def test_structural_equivalence_false_for_different_type():
    """Test structural equivalence is false for different Python types."""
    node = _StructEqNode("add", (1, 2))
    assert not node.is_structurally_equivalent((1, 2))
