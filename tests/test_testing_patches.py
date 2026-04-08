"""Tests the testing patches."""

import pytest
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
        return self.num == other.num


class _TestClass2(StructuralEquivalenceMixin):
    test: _TestClass1

    def __init__(self, test: _TestClass1) -> None:
        self.test = test

    def is_structurally_equivalent(self, other: object) -> bool:
        return self.test.is_structurally_equivalent(other.test)


_TestClass1Alias = _TestClass1


@deterministic_identifiers_by_name_hint
def test_deterministic_identifiers_by_name_hint():
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


def test_deterministic_identifiers_by_name_hint_with_block():
    """Test deterministic identifiers patch supports `with` usage."""
    with deterministic_identifiers_by_name_hint:
        identifier_a = Identifier("a")
        identifier_a_2 = Identifier("a")

        assert identifier_a == identifier_a_2
        assert hash(identifier_a) == hash(identifier_a_2)


def test_fail_fast_structural_equivalence():
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
