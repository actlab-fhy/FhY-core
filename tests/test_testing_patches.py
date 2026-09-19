"""Tests the testing patches."""

import copy
import pickle
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from fhy_core.identifier import Identifier
from fhy_core.symbolic.expression.core import LiteralExpression
from fhy_core.testing_patches import (
    deterministic_identifiers_by_name_hint,
    fail_fast_structural_equivalence,
)
from fhy_core.traits import StructuralEquivalence
from fhy_core.utils.override import override


class _TestClass1(StructuralEquivalence):
    num: int

    def __init__(self, num: int) -> None:
        self.num = num

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        assert isinstance(other, _TestClass1)
        return self.num == other.num


class _TestClass2(StructuralEquivalence):
    test: _TestClass1

    def __init__(self, test: _TestClass1) -> None:
        self.test = test

    @override
    def is_structurally_equivalent(self, other: object) -> bool:
        assert isinstance(other, _TestClass2)
        return self.test.is_structurally_equivalent(other.test)


_TestClass1Alias = _TestClass1

_ABSENT = object()


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


@deterministic_identifiers_by_name_hint()
def _construct_two_identifiers_sharing_a_name_hint() -> tuple[Identifier, Identifier]:
    """Construct two identifiers named ``shared`` under the called decorator."""
    return Identifier("shared"), Identifier("shared")


def _capture_identifier_construction_dunders() -> dict[str, object]:
    """Return the `__new__`/`__init__` entries of `Identifier`'s own namespace."""
    return {
        name: Identifier.__dict__.get(name, _ABSENT) for name in ("__new__", "__init__")
    }


def _assert_same_dunders(expected: dict[str, object]) -> None:
    """Assert `Identifier`'s own `__new__`/`__init__` entries are exactly these."""
    actual = _capture_identifier_construction_dunders()
    assert all(actual[name] is expected[name] for name in expected)


def test_deterministic_identifiers_by_name_hint_as_a_called_decorator() -> None:
    """Test the patch also decorates when called with no arguments."""
    first, second = _construct_two_identifiers_sharing_a_name_hint()

    assert first == second


def test_deterministic_identifiers_by_name_hint_returns_one_instance_per_hint() -> None:
    """Test a repeated name hint, positional or keyword, gets the same instance."""
    with deterministic_identifiers_by_name_hint:
        first = Identifier("shared")
        second = Identifier("shared")
        by_keyword = Identifier(name_hint="shared")

    assert second is first
    assert by_keyword is first
    assert first.is_frozen


def test_deterministic_identifiers_by_name_hint_draws_fresh_ids_per_name_hint() -> None:
    """Test a new name hint takes the next real id and a repeat takes none."""
    base = Identifier("anchor").id

    with deterministic_identifiers_by_name_hint:
        a = Identifier("a")
        b = Identifier("b")
        a_again = Identifier("a")
    after = Identifier("after")

    assert (a.id, b.id, a_again.id, after.id) == (
        base + 1,
        base + 2,
        base + 1,
        base + 3,
    )


def test_deterministic_identifiers_by_name_hint_restores_on_body_exception() -> None:
    """Test identifiers get distinct ids again after the body raises."""
    with pytest.raises(RuntimeError):
        with deterministic_identifiers_by_name_hint:
            inside_first = Identifier("shared")
            inside_second = Identifier("shared")
            raise RuntimeError("user-raised")

    after_first = Identifier("shared")
    after_second = Identifier("shared")

    assert inside_first == inside_second
    assert after_first != after_second


def test_deterministic_identifiers_by_name_hint_supports_nested_usage() -> None:
    """Test nested entries share one table and only end at the outermost exit."""
    with deterministic_identifiers_by_name_hint:
        outer_before = Identifier("shared")
        with deterministic_identifiers_by_name_hint:
            inner = Identifier("shared")
        # The inner exit must not end the patch while the outer scope is alive.
        outer_after = Identifier("shared")
    after_exit = Identifier("shared")

    assert inner is outer_before
    assert outer_after is outer_before
    assert after_exit != outer_before


def test_deterministic_identifiers_by_name_hint_forgets_hints_after_exit() -> None:
    """Test a later patch gives a name hint seen by an earlier one a fresh id."""
    with deterministic_identifiers_by_name_hint:
        first_patch = Identifier("shared")
    with deterministic_identifiers_by_name_hint:
        second_patch = Identifier("shared")

    assert second_patch != first_patch


def test_deterministic_identifiers_by_name_hint_restores_the_class_exactly() -> None:
    """Test the outermost exit leaves `Identifier`'s own namespace as it was."""
    original = _capture_identifier_construction_dunders()

    with deterministic_identifiers_by_name_hint:
        with deterministic_identifiers_by_name_hint:
            pass
        patched = _capture_identifier_construction_dunders()

    assert any(patched[name] is not original[name] for name in original)
    _assert_same_dunders(original)


def test_deterministic_identifiers_by_name_hint_restores_the_class_on_raise() -> None:
    """Test the class is restored exactly when the body raises."""
    original = _capture_identifier_construction_dunders()

    with pytest.raises(RuntimeError):
        with deterministic_identifiers_by_name_hint:
            raise RuntimeError("user-raised")

    _assert_same_dunders(original)


def test_deterministic_identifiers_by_name_hint_keeps_deserialized_ids() -> None:
    """Test deserialization inside the patch keeps the payload's id."""
    far_id = Identifier("anchor").id + 1000

    with deterministic_identifiers_by_name_hint:
        constructed = Identifier("shared")
        deserialized = Identifier.deserialize_from_dict(
            {"id": far_id, "name_hint": "shared"}
        )

    assert (deserialized.id, deserialized.name_hint) == (far_id, "shared")
    assert deserialized.is_frozen
    assert deserialized != constructed


@pytest.mark.parametrize(
    "duplicate",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_deterministic_identifiers_by_name_hint_keeps_copies_working(
    duplicate: Callable[[Identifier], Identifier],
) -> None:
    """Test copying and unpickling inside the patch give equal frozen copies."""
    with deterministic_identifiers_by_name_hint:
        original = Identifier("shared")
        duplicated = duplicate(original)

    assert duplicated == original
    assert duplicated.name_hint == "shared"
    assert duplicated.is_frozen


def test_deterministic_identifiers_by_name_hint_still_requires_a_name_hint() -> None:
    """Test constructing without a name hint inside the patch raises `TypeError`."""
    with deterministic_identifiers_by_name_hint:
        with pytest.raises(TypeError):
            Identifier()  # type: ignore[call-arg]  # test: invalid input


def test_deterministic_identifiers_by_name_hint_is_thread_safe() -> None:
    """Test threads racing to construct one name hint all get one instance."""
    # Every thread starts at once, and the interpreter switches threads as
    # often as it can, so the first constructions of the name hint overlap.
    original_switch_interval = sys.getswitchinterval()
    thread_count = 16
    barrier = threading.Barrier(thread_count)

    def construct_after_barrier(_: int) -> Identifier:
        barrier.wait()
        return Identifier("shared")

    sys.setswitchinterval(1e-6)
    try:
        with (
            deterministic_identifiers_by_name_hint,
            ThreadPoolExecutor(max_workers=thread_count) as executor,
        ):
            identifiers = list(
                executor.map(construct_after_barrier, range(thread_count))
            )
    finally:
        sys.setswitchinterval(original_switch_interval)

    assert len({id(identifier) for identifier in identifiers}) == 1
