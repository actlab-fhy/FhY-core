"""Tests the general utilities."""

from typing import Any

import pytest
from fhy_core.utils import (
    Lattice,
    PartiallyOrderedSet,
    Stack,
    invert_dict,
    invert_frozen_dict,
)
from frozendict import frozendict


def test_invert_dict():
    """Test that the dictionary inversion works."""
    test_dict = {"a": 1, "b": 2, "c": 3}
    inverted_dict = invert_dict(test_dict)
    assert inverted_dict == {1: "a", 2: "b", 3: "c"}


def test_invert_frozen_dict():
    """Test that the frozen dictionary inversion works."""
    test_dict = {"a": 1, "b": 2, "c": 3}
    frozen_dict = frozendict(test_dict)
    inverted_dict = invert_frozen_dict(frozen_dict)
    assert inverted_dict == {1: "a", 2: "b", 3: "c"}


@pytest.fixture()
def empty_lattice():
    """Return an empty lattice."""
    lattice = Lattice[Any]()
    return lattice


@pytest.fixture()
def singleton_lattice():
    """Uses the Lattice class internals to create a lattice with one element.

    lattice: ({1}, <=)

    """
    lattice = Lattice[int]()
    lattice._poset.add_element(1)
    return lattice


@pytest.fixture()
def two_element_lattice():
    """Uses the Lattice class internals to create a lattice with two elements.

    lattice: ({1, 2}, <=)

    """
    lattice = Lattice[int]()
    lattice._poset.add_element(1)
    lattice._poset.add_element(2)
    lattice._poset.add_order(1, 2)
    return lattice


def test_empty_lattice_contains_no_elements(empty_lattice: Lattice[Any]):
    """Test that an empty lattice contains no elements."""
    assert 1 not in empty_lattice


def test_singleton_lattice_contains_element(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice contains the element."""
    assert 1 in singleton_lattice


def test_add_element_to_empty_lattice(empty_lattice: Lattice[Any]):
    """Test that an element can be added to an empty lattice."""
    empty_lattice.add_element(1)
    assert 1 in empty_lattice


def test_add_duplicate_element_to_lattice(singleton_lattice: Lattice[int]):
    """Test that adding a duplicate element to a lattice raises an error."""
    with pytest.raises(ValueError):
        singleton_lattice.add_element(1)


def test_singleton_lattice_meet_is_element(singleton_lattice: Lattice[int]):
    """Test that the meet of a singleton lattice is the element itself."""
    assert singleton_lattice.get_meet(1, 1) == 1


def test_singleton_lattice_has_meet(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice has a meet."""
    assert singleton_lattice.has_meet(1, 1) is True


def test_singleton_lattice_join_is_element(singleton_lattice: Lattice[int]):
    """Test that the join of a singleton lattice is the element itself."""
    assert singleton_lattice.get_join(1, 1) == 1


def test_singleton_lattice_has_join(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice has a join."""
    assert singleton_lattice.has_join(1, 1) is True


def test_singleton_lattice_get_least_upper_bound(singleton_lattice: Lattice[int]):
    """Test that the least upper bound of a singleton lattice is the element."""
    assert singleton_lattice.get_least_upper_bound(1, 1) == 1


def test_two_element_lattice_meet(two_element_lattice: Lattice[int]):
    """Test that the meet of a two element lattice is the lower element."""
    assert two_element_lattice.get_meet(1, 1) == 1
    assert two_element_lattice.get_meet(2, 2) == 2
    assert two_element_lattice.get_meet(1, 2) == 1


def test_two_element_lattice_has_meet(two_element_lattice: Lattice[int]):
    """Test that a two element lattice has a meet."""
    assert two_element_lattice.has_meet(1, 1) is True
    assert two_element_lattice.has_meet(2, 2) is True
    assert two_element_lattice.has_meet(1, 2) is True


def test_two_element_lattice_join(two_element_lattice: Lattice[int]):
    """Test that the join of a two element lattice is the upper element."""
    assert two_element_lattice.get_join(1, 1) == 1
    assert two_element_lattice.get_join(2, 2) == 2
    assert two_element_lattice.get_join(1, 2) == 2


def test_two_element_lattice_has_join(two_element_lattice: Lattice[int]):
    """Test that a two element lattice has a join."""
    assert two_element_lattice.has_join(1, 1) is True
    assert two_element_lattice.has_join(2, 2) is True
    assert two_element_lattice.has_join(1, 2) is True


def test_two_element_lattice_get_least_upper_bound(two_element_lattice: Lattice[int]):
    """Test that the least upper bound of a two element lattice is the element."""
    assert two_element_lattice.get_least_upper_bound(1, 1) == 1
    assert two_element_lattice.get_least_upper_bound(2, 2) == 2
    assert two_element_lattice.get_least_upper_bound(1, 2) == 2


def test_empty_lattice_meet(empty_lattice: Lattice[Any]):
    """Test that the meet of an empty lattice is None."""
    assert empty_lattice.get_meet(1, 1) is None


def test_empty_lattice_join(empty_lattice: Lattice[Any]):
    """Test that the join of an empty lattice is None."""
    assert empty_lattice.get_join(1, 1) is None


def test_add_order_to_lattice():
    """Test that an order can be added to a lattice."""
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_order(1, 2)
    assert lattice.get_meet(1, 2) == 1
    assert lattice.get_join(1, 2) == 2


def test_add_invalid_order_to_lattice():
    """Test that adding an invalid order to a lattice raises an error."""
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_order(1, 2)
    with pytest.raises(RuntimeError):
        lattice.add_order(2, 1)


def test_empty_lattice_is_lattice(empty_lattice: Lattice[Any]):
    """Test that an empty lattice is a lattice."""
    assert empty_lattice.is_lattice() is True


def test_singleton_lattice_is_lattice(singleton_lattice: Lattice[int]):
    """Test that a singleton lattice is a lattice."""
    assert singleton_lattice.is_lattice() is True


def test_two_element_lattice_is_lattice(two_element_lattice: Lattice[int]):
    """Test that a two element lattice is a lattice."""
    assert two_element_lattice.is_lattice() is True


@pytest.fixture()
def positive_integer_lattice():
    UPPER_BOUND = 10
    lattice = Lattice[int]()
    for i in range(1, UPPER_BOUND + 1):
        lattice.add_element(i)
    for i in range(1, UPPER_BOUND):
        lattice.add_order(i, i + 1)
    return lattice


@pytest.fixture()
def subsets_of_xyz_lattice():
    lattice = Lattice[str]()
    lattice.add_element("0")  # empty set
    lattice.add_element("x")
    lattice.add_element("y")
    lattice.add_element("z")
    lattice.add_element("xy")
    lattice.add_element("xz")
    lattice.add_element("yz")
    lattice.add_element("xyz")
    lattice.add_order("0", "x")
    lattice.add_order("0", "y")
    lattice.add_order("0", "z")
    lattice.add_order("x", "xy")
    lattice.add_order("x", "xz")
    lattice.add_order("y", "xy")
    lattice.add_order("y", "yz")
    lattice.add_order("z", "xz")
    lattice.add_order("z", "yz")
    lattice.add_order("xy", "xyz")
    lattice.add_order("xz", "xyz")
    lattice.add_order("yz", "xyz")
    return lattice


def test_positive_integer_lattice_meet(positive_integer_lattice: Lattice[int]):
    """Test that the meet of a positive integer lattice is the minimum of the
    two elements.
    """
    assert positive_integer_lattice.get_meet(3, 5) == 3
    assert positive_integer_lattice.get_meet(4, 6) == 4


def test_positive_integer_lattice_join(positive_integer_lattice: Lattice[int]):
    """Test that the join of a positive integer lattice is the maximum of the
    two elements.
    """
    assert positive_integer_lattice.get_join(3, 5) == 5
    assert positive_integer_lattice.get_join(4, 6) == 6


def test_positive_integer_lattice_is_lattice(positive_integer_lattice: Lattice[int]):
    """Test that a positive integer lattice is a lattice."""
    assert positive_integer_lattice.is_lattice() is True


def test_subsets_of_xyz_lattice_meet(subsets_of_xyz_lattice: Lattice[str]):
    """Test that the meet of subsets of XYZ lattice is the intersection of the
    two sets.
    """
    assert subsets_of_xyz_lattice.get_meet("x", "y") == "0"
    assert subsets_of_xyz_lattice.get_meet("x", "xy") == "x"
    assert subsets_of_xyz_lattice.get_meet("x", "z") == "0"
    assert subsets_of_xyz_lattice.get_meet("xz", "yz") == "z"
    assert subsets_of_xyz_lattice.get_meet("xy", "xyz") == "xy"


def test_subsets_of_xyz_lattice_join(subsets_of_xyz_lattice: Lattice[str]):
    """Test that the join of subsets of XYZ lattice is the union of the two sets."""
    assert subsets_of_xyz_lattice.get_join("x", "y") == "xy"
    assert subsets_of_xyz_lattice.get_join("x", "xy") == "xy"
    assert subsets_of_xyz_lattice.get_join("x", "z") == "xz"
    assert subsets_of_xyz_lattice.get_join("xz", "yz") == "xyz"
    assert subsets_of_xyz_lattice.get_join("xy", "xyz") == "xyz"


def test_subsets_of_xyz_lattice_is_lattice(subsets_of_xyz_lattice: Lattice[str]):
    """Test that subsets of XYZ lattice is a lattice."""
    assert subsets_of_xyz_lattice.is_lattice() is True


@pytest.fixture()
def basic_non_lattice_poset():
    lattice = Lattice[int]()
    lattice.add_element(1)
    lattice.add_element(2)
    lattice.add_element(3)
    lattice.add_element(4)
    lattice.add_order(1, 3)
    lattice.add_order(1, 4)
    lattice.add_order(2, 3)
    lattice.add_order(2, 4)
    return lattice


def test_basic_non_lattice_poset(basic_non_lattice_poset: Lattice[int]):
    """Test that a basic non-lattice poset is not a lattice."""
    assert basic_non_lattice_poset.is_lattice() is False


def test_basic_non_lattice_has_no_least_upper_bound(
    basic_non_lattice_poset: Lattice[int],
):
    """Test that a basic non-lattice poset has no least upper bound for certain
    elements.
    """
    with pytest.raises(RuntimeError):
        basic_non_lattice_poset.get_least_upper_bound(3, 4)


@pytest.fixture
def basic_poset():
    """Uses the PartiallyOrderedSet class internals to create a poset with two
    elements.

    poset: ({1, 2}, <=)

    """
    poset = PartiallyOrderedSet[int]()
    poset._graph.add_node(1)
    poset._graph.add_node(2)
    poset._graph.add_edge(1, 2)
    return poset


def test_empty_poset_length():
    """Test that an empty poset's length is correctly calculated."""
    poset = PartiallyOrderedSet[int]()

    assert len(poset) == 0


def test_poset_length(basic_poset):
    """Test that the poset's length is correctly calculated."""
    assert len(basic_poset) == 2


def test_element_not_in_empty_poset():
    """Test that an element is not in an empty poset."""
    poset = PartiallyOrderedSet[int]()

    assert 1 not in poset


def test_element_in_poset(basic_poset):
    """Test that an element is in the poset."""
    assert 1 in basic_poset


def test_poset_add_element():
    """Test that elements are correctly added to the poset."""
    poset = PartiallyOrderedSet[int]()
    poset.add_element(1)

    assert len(poset) == 1


def test_poset_add_duplicate_element():
    """Test that adding a duplicate element raises an error."""
    poset = PartiallyOrderedSet[int]()
    poset.add_element(1)

    with pytest.raises(ValueError):
        poset.add_element(1)


def test_poset_is_less_than(basic_poset):
    """Test that the is_less_than method correctly determines if one element is less
    than another.
    """
    assert basic_poset.is_less_than(1, 2)


def test_poset_is_less_than_with_invalid_lower_element(basic_poset):
    """Test that is_less_than raises an error when the lower element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.is_less_than(3, 2)


def test_poset_is_less_than_with_invalid_upper_element(basic_poset):
    """Test that is_less_than raises an error when the upper element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.is_less_than(1, 3)


def test_poset_is_greater_than(basic_poset):
    """Test that the is_greater_than method correctly determines if one element is
    greater than another.
    """
    assert basic_poset.is_greater_than(2, 1)


def test_poset_is_greater_than_with_invalid_lower_element(basic_poset):
    """Test that is_greater_than raises an error when the lower element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.is_greater_than(3, 2)


def test_poset_is_greater_than_with_invalid_upper_element(basic_poset):
    """Test that is_greater_than raises an error when the upper element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.is_greater_than(1, 3)


def test_poset_add_order(basic_poset):
    """Test that order relations are correctly added to the poset."""
    basic_poset.add_element(3)
    basic_poset.add_order(2, 3)

    assert basic_poset.is_less_than(2, 3)
    assert basic_poset.is_greater_than(3, 2)


def test_poset_add_order_with_invalid_lower_element(basic_poset):
    """Test that add_order raises an error when the lower element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.add_order(3, 2)


def test_poset_add_order_with_invalid_upper_element(basic_poset):
    """Test that add_order raises an error when the upper element is not in the
    poset.
    """
    with pytest.raises(ValueError):
        basic_poset.add_order(1, 3)


def test_poset_invalid_order():
    """Test that an invalid order raises an error."""
    poset = PartiallyOrderedSet[int]()
    poset.add_element(1)
    poset.add_element(2)
    poset.add_order(1, 2)

    with pytest.raises(RuntimeError):
        poset.add_order(2, 1)


def test_poset_iter(basic_poset):
    """Test that the poset can be iterated over."""
    assert list(basic_poset) == [1, 2]
    assert list(basic_poset) == [1, 2], "Expected the poset to be iterable \
multiple times."


@pytest.fixture
def text_stack() -> Stack:
    """Use the Stack class internals to create a stack with two elements.

    |- stack - |
    |  "test"  |
    |  "fhy"   |
    | -------- |

    """
    stack = Stack[str]()
    stack._stack.append("fhy")
    stack._stack.append("test")

    return stack


def test_empty_stack_length():
    """Test that an empty stack's length is correctly calculated."""
    stack = Stack[str]()
    assert len(stack) == 0


def test_stack_length(text_stack):
    """Test that the stack's length is correctly calculated."""
    assert len(text_stack) == 2


def test_stack_push():
    """Test that elements are correctly pushed to the stack."""
    stack = Stack[str]()
    stack.push("fhy")
    assert len(stack) == 1
    stack.push("test")
    assert len(stack) == 2


def test_stack_peek(text_stack):
    """Test that peek method reveals the correct element and does not mutate the
    stack.
    """
    current: str = text_stack.peek()
    assert current == "test"
    current_length: int = len(text_stack)
    assert current_length == 2, f"Expected the stack to be unchanged after peek, \
but the length changed to {current_length}."


def test_stack_peek_error():
    """Test that stack raises an IndexError when peek is called on an empty stack."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.peek()


def test_stack_pop(text_stack):
    """Test that pop method removes the correct element from the stack."""
    first = text_stack.pop()
    assert first == "test"
    assert len(text_stack) == 1

    second = text_stack.pop()
    assert second == "fhy"
    assert len(text_stack) == 0


def test_stack_pop_error():
    """Test that stack raises an IndexError when pop is called on an empty stack."""
    stack = Stack[str]()

    with pytest.raises(IndexError):
        stack.pop()


def test_stack_clear(text_stack):
    """Test that clear method removes all elements from the stack."""
    text_stack.clear()
    assert len(text_stack) == 0


def test_stack_iter(text_stack):
    """Test that we can iterate over the stack and retrieve elements."""
    for element, expected_element in zip(text_stack, ("fhy", "test")):
        assert element == expected_element

    assert len(list(text_stack)) == 2, "Expected to be able to iterate over the \
stack again."


def test_stack_next(text_stack):
    """Test use of next on a generator of the stack to retrieve elements."""
    generator = (i for i in text_stack)

    assert next(generator) == "fhy"
    assert next(generator) == "test"
    with pytest.raises(StopIteration):
        next(generator)
