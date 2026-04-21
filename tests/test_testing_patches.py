"""Tests the testing patches."""

import pytest

from fhy_core.expression.core import LiteralExpression
from fhy_core.identifier import Identifier
from fhy_core.testing_patches import (
    deterministic_identifiers_by_name_hint,
    fail_fast_structural_equivalence,
)
from fhy_core.trait import StructuralEquivalenceMixin


class _TestClass1(StructuralEquivalenceMixin):
    num: int

    def __init__(self, num: int) -> None:
        self.num = num

    def is_structurally_equivalent(self, other: object) -> bool:
        assert isinstance(other, _TestClass1)
        return self.num == other.num


class _TestClass2(StructuralEquivalenceMixin):
    test: _TestClass1

    def __init__(self, test: _TestClass1) -> None:
        self.test = test

    def is_structurally_equivalent(self, other: object) -> bool:
        assert isinstance(other, _TestClass2)
        return self.test.is_structurally_equivalent(other.test)


_TestClass1Alias = _TestClass1


@deterministic_identifiers_by_name_hint
def test_deterministic_identifiers_by_name_hint() -> None:
    """Test deterministic identifiers support hashed containers in tests."""
    identifier_a = Identifier("a")
    identifier_a_2 = Identifier("a")
    identifier_b = Identifier("b")

    assert identifier_a == identifier_a_2
    assert hash(identifier_a) == hash(identifier_a_2)
    assert identifier_a != identifier_b

    identifier_set = {identifier_a, identifier_b}
    identifier_dict = {identifier_a: "left", identifier_b: "right"}

    assert identifier_a_2 in identifier_set
    assert identifier_dict[identifier_a_2] == "left"


def test_deterministic_identifiers_by_name_hint_with_block() -> None:
    """Test deterministic identifiers patch supports `with` usage."""
    with deterministic_identifiers_by_name_hint:
        identifier_a = Identifier("a")
        identifier_a_2 = Identifier("a")

        assert identifier_a == identifier_a_2
        assert hash(identifier_a) == hash(identifier_a_2)


def test_fail_fast_structural_equivalence() -> None:
    """Test the fail fast structural equivalence patch works."""

    test_class1_a = _TestClass1(1)
    test_class1_b = _TestClass1(1)
    test_class1_c = _TestClass1(2)

    test_class2_a = _TestClass2(test_class1_a)
    test_class2_b = _TestClass2(test_class1_b)
    test_class2_c = _TestClass2(test_class1_c)

    assert test_class1_a.is_structurally_equivalent(test_class1_b)
    assert not test_class1_a.is_structurally_equivalent(test_class1_c)
    assert test_class2_a.is_structurally_equivalent(test_class2_b)
    assert not test_class2_a.is_structurally_equivalent(test_class2_c)

    with pytest.raises(AssertionError):
        with fail_fast_structural_equivalence():
            test_class1_a.is_structurally_equivalent(test_class1_c)


def test_fail_fast_structural_equivalence_wraps_each_class_once() -> None:
    """Test structural-equivalence patching does not double-wrap aliased classes."""
    original_method = _TestClass1.is_structurally_equivalent
    alias_method = _TestClass1Alias.is_structurally_equivalent

    assert original_method is alias_method

    with fail_fast_structural_equivalence():
        patched_method = _TestClass1.is_structurally_equivalent
        assert patched_method is _TestClass1Alias.is_structurally_equivalent
        assert getattr(patched_method, "__wrapped__", None) is original_method
        assert getattr(original_method, "__wrapped__", None) is None

    assert _TestClass1.is_structurally_equivalent is original_method


def test_fail_fast_structural_equivalence_restores_inherited_methods() -> None:
    """Inherited structural-equivalence methods should be restored after patching."""
    original_method = LiteralExpression.is_structurally_equivalent

    with pytest.raises(AssertionError):
        with fail_fast_structural_equivalence():
            LiteralExpression(1).is_structurally_equivalent(LiteralExpression(2))

    restored_method = LiteralExpression.is_structurally_equivalent
    assert restored_method is original_method
    assert getattr(restored_method, "__wrapped__", None) is None


def test_fail_fast_structural_equivalence_restores_methods_on_body_exception() -> None:
    """Test that patched methods are restored when the body raises."""
    original_method = _TestClass1.is_structurally_equivalent

    with pytest.raises(RuntimeError):
        with fail_fast_structural_equivalence():
            raise RuntimeError("user-raised")

    assert _TestClass1.is_structurally_equivalent is original_method


def test_deterministic_identifiers_by_name_hint_restores_on_body_exception() -> None:
    """Test that Identifier.__init__ is restored when the body raises."""
    original_init = Identifier.__init__

    with pytest.raises(RuntimeError):
        with deterministic_identifiers_by_name_hint:
            raise RuntimeError("user-raised")

    assert Identifier.__init__ is original_init


def test_deterministic_identifiers_by_name_hint_supports_nested_usage() -> None:
    """Test nested entries reuse the patch and only restore at the outermost exit."""
    original_init = Identifier.__init__

    with deterministic_identifiers_by_name_hint:
        outer_patched_init = Identifier.__init__
        assert outer_patched_init is not original_init

        with deterministic_identifiers_by_name_hint:
            inner_a = Identifier("shared")
            inner_b = Identifier("shared")
            assert inner_a == inner_b
            assert Identifier.__init__ is outer_patched_init

        # Inner exit must NOT restore the original while the outer scope is alive.
        assert Identifier.__init__ is outer_patched_init

        outer_a = Identifier("shared")
        outer_b = Identifier("shared")
        assert outer_a == outer_b

    assert Identifier.__init__ is original_init
