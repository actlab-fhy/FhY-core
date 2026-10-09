"""Tests for the Python interface over the Rust-backed poset and lattice.

``PartiallyOrderedSet`` and ``Lattice`` are thin subclasses of
``fhy_core._rs.PartiallyOrderedSet`` and ``fhy_core._rs.Lattice``, which keep
the element objects in a dict and run the order in the Rust core. These tests
cover what the binding adds around the core: the class structure, the elements'
Python hashing and equality, the key calls of ``iter_stable``, the ``verify``
report, and the exception classes and texts.
"""

import copy
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel, ValidationReport
from fhy_core.lattice import Lattice
from fhy_core.traits.verifiable import Verifiable, VerifiableMixin
from fhy_core.utils.override import override
from fhy_core.utils.poset import PartiallyOrderedSet


def _build_crown() -> Lattice[int]:
    """Return two minimal elements, 1 and 2, below two maximal ones, 3 and 4."""
    lattice: Lattice[int] = Lattice()
    for element in (1, 2, 3, 4):
        lattice.add_element(element)
    for lower, upper in ((1, 3), (1, 4), (2, 3), (2, 4)):
        lattice.add_order(lower, upper)
    return lattice


# =============================================================================
# Class structure
# =============================================================================


def test_poset_and_lattice_are_thin_subclasses_of_their_rust_classes() -> None:
    """Test the public classes subclass the extension's classes."""
    assert issubclass(PartiallyOrderedSet, _rs.PartiallyOrderedSet)
    assert issubclass(Lattice, _rs.Lattice)


def test_poset_and_lattice_subscript_at_run_time() -> None:
    """Test `X[T]()` builds an instance, as a generic class does."""
    assert isinstance(PartiallyOrderedSet[str](), PartiallyOrderedSet)
    assert isinstance(Lattice[int](), Lattice)


def test_lattice_is_a_verifiable_and_a_virtual_verifiable_mixin() -> None:
    """Test the lattice satisfies `Verifiable` and registers as the mixin."""
    lattice: Lattice[int] = Lattice()

    assert isinstance(lattice, Verifiable)
    assert isinstance(lattice, VerifiableMixin)


def test_subclasses_with_their_own_init_construct() -> None:
    """Test a subclass whose `__init__` takes arguments constructs."""

    class _Seeded(PartiallyOrderedSet[int]):
        def __init__(self, seeds: list[int]) -> None:
            super().__init__()
            for seed in seeds:
                self.add_element(seed)

    poset = _Seeded([3, 1])

    assert list(poset) == [3, 1]


# =============================================================================
# Elements
# =============================================================================


def test_elements_keep_their_python_hashing_and_equality() -> None:
    """Test `1`, `1.0` and `True` are one element, as dict keys are."""
    poset: PartiallyOrderedSet[Any] = PartiallyOrderedSet()
    poset.add_element(1)

    assert 1.0 in poset
    assert True in poset
    with pytest.raises(ValueError, match="already a member"):
        poset.add_element(1.0)


def test_an_unhashable_element_is_refused_and_is_no_member() -> None:
    """Test adding an unhashable element raises, and asking for one answers no."""
    poset: PartiallyOrderedSet[Any] = PartiallyOrderedSet()

    with pytest.raises(TypeError):
        poset.add_element([1])
    assert [1] not in poset


def test_ordering_and_asking_about_a_non_member_raise_value_error() -> None:
    """Test the non-member errors name the element, the first one first."""
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    poset.add_element(1)

    with pytest.raises(ValueError, match=r"^3 is not a member"):
        poset.add_order(3, 4)
    with pytest.raises(ValueError, match=r"^4 is not a member"):
        poset.is_less_than(1, 4)
    with pytest.raises(ValueError, match=r"^5 is not a member"):
        poset.is_greater_than(5, 1)


def test_a_cycle_raises_runtime_error_naming_both_elements() -> None:
    """Test reversing an order raises `RuntimeError` with the core's text."""
    poset: PartiallyOrderedSet[str] = PartiallyOrderedSet()
    for element in ("a", "b"):
        poset.add_element(element)
    poset.add_order("a", "b")

    with pytest.raises(RuntimeError, match=r"^ordering b below a would close a cycle$"):
        poset.add_order("b", "a")
    with pytest.raises(RuntimeError, match="would close a cycle"):
        poset.add_order("a", "a")


def test_the_order_is_reflexive_and_an_order_already_holding_is_accepted() -> None:
    """Test `is_less_than(x, x)` holds and a repeated order is no error."""
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    for element in (1, 2, 3):
        poset.add_element(element)
    poset.add_order(1, 2)
    poset.add_order(2, 3)
    poset.add_order(1, 3)

    assert poset.is_less_than(2, 2)
    assert poset.is_greater_than(3, 1)
    assert len(poset) == 3


# =============================================================================
# Iteration
# =============================================================================


def test_iteration_is_topological_with_insertion_order_breaking_ties() -> None:
    """Test ties come in the order the elements were added."""
    poset: PartiallyOrderedSet[str] = PartiallyOrderedSet()
    for element in ("d", "c", "b", "a"):
        poset.add_element(element)
    poset.add_order("a", "c")

    assert list(poset) == ["d", "b", "a", "c"]


def test_iter_stable_calls_the_key_once_per_element() -> None:
    """Test `iter_stable` calls `key` exactly once for each element."""
    poset: PartiallyOrderedSet[str] = PartiallyOrderedSet()
    for element in ("b", "c", "a"):
        poset.add_element(element)
    calls: list[str] = []

    def key(element: str) -> str:
        calls.append(element)
        return element

    order = list(poset.iter_stable(key=key))

    assert order == ["a", "b", "c"]
    assert sorted(calls) == ["a", "b", "c"]


def test_iter_stable_propagates_the_key_s_exception() -> None:
    """Test an exception from `key` reaches the caller unchanged."""
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    poset.add_element(1)
    failure = ArithmeticError("no key")

    def key(element: int) -> int:
        raise failure

    with pytest.raises(ArithmeticError) as raised:
        list(poset.iter_stable(key=key))
    assert raised.value is failure


# =============================================================================
# The lattice
# =============================================================================


def test_verify_reports_each_missing_bound_in_iteration_order() -> None:
    """Test `verify` reports one ERROR per missing bound, meet first."""
    report = _build_crown().verify()

    assert isinstance(report, ValidationReport)
    messages = [diagnostic.message.message for diagnostic in report.errors()]
    assert messages[:2] == [
        "lattice has no unique meet for elements 1 and 2",
        "lattice has no unique join for elements 1 and 2",
    ]
    assert len(messages) == 8
    assert all(
        diagnostic.level is DiagnosticLevel.ERROR for diagnostic in report.errors()
    )
    assert all(
        diagnostic.source == "fhy_core.lattice.Lattice.verify"
        for diagnostic in report.errors()
    )


def test_verify_writes_the_elements_reprs() -> None:
    """Test the report names the elements by their `repr`."""
    lattice: Lattice[str] = Lattice()
    for element in ("x", "y"):
        lattice.add_element(element)

    messages = [diagnostic.message.message for diagnostic in lattice.verify().errors()]

    assert "lattice has no unique meet for elements 'x' and 'y'" in messages


def test_get_least_upper_bound_without_a_join_raises_runtime_error() -> None:
    """Test the missing join raises `RuntimeError` with the core's text."""
    with pytest.raises(RuntimeError, match=r"^no least upper bound of 3 and 4"):
        _build_crown().get_least_upper_bound(3, 4)


def test_meet_and_join_return_the_element_objects() -> None:
    """Test a meet and a join are the very objects that were added."""
    bottom, top = ("bottom",), ("top",)
    lattice: Lattice[tuple[str]] = Lattice()
    lattice.add_element(bottom)
    lattice.add_element(top)
    lattice.add_order(bottom, top)

    assert lattice.get_meet(("bottom",), top) is bottom
    assert lattice.get_join(bottom, ("top",)) is top
    assert lattice.has_meet(bottom, top)
    assert lattice.has_join(bottom, top)


def test_a_non_member_is_refused_by_every_bound_query() -> None:
    """Test every meet or join query of a non-member raises `ValueError`."""
    lattice = _build_crown()

    for query in (
        lattice.get_meet,
        lattice.get_join,
        lattice.has_meet,
        lattice.has_join,
    ):
        with pytest.raises(ValueError, match="not a member"):
            query(1, 9)


# =============================================================================
# Pickling and copying
# =============================================================================


class _TaggedPoset(PartiallyOrderedSet[str]):
    """A poset subclass whose `__init__` takes an argument and keeps state."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


class _TaggedLattice(Lattice[int]):
    """A lattice subclass whose `__init__` takes an argument and keeps state."""

    def __init__(self, tag: str) -> None:
        super().__init__()
        self.tag = tag


def _pickle_round_trip(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


_COPIES: list[Callable[[Any], Any]] = [_pickle_round_trip, copy.copy, copy.deepcopy]
_COPY_IDS = ["pickle", "copy", "deepcopy"]


def _build_poset(poset: PartiallyOrderedSet[str]) -> PartiallyOrderedSet[str]:
    """Fill `poset` with `d` below `b` below `a`, and `c` beside them."""
    for element in ("d", "c", "b", "a"):
        poset.add_element(element)
    poset.add_order("d", "b")
    poset.add_order("b", "a")
    return poset


def _fill_crown(lattice: Lattice[int]) -> Lattice[int]:
    """Fill `lattice` with a bottom 0 below 1 and 2, below a top 3."""
    for element in (3, 1, 2, 0):
        lattice.add_element(element)
    for lower, upper in ((0, 1), (0, 2), (1, 3), (2, 3)):
        lattice.add_order(lower, upper)
    return lattice


@pytest.mark.parametrize("copy_function", _COPIES, ids=_COPY_IDS)
def test_a_poset_pickles_and_copies_with_its_order(
    copy_function: Callable[[Any], Any],
) -> None:
    """Test a poset copies with its elements' insertion order and its order."""
    original = _build_poset(PartiallyOrderedSet())

    rebuilt = copy_function(original)

    assert type(rebuilt) is PartiallyOrderedSet
    assert rebuilt is not original
    assert list(rebuilt) == list(original) == ["d", "c", "b", "a"]
    assert list(rebuilt.iter_stable(key=lambda _element: 0)) == list(original)
    assert rebuilt.is_less_than("d", "a")
    assert not rebuilt.is_less_than("c", "a")
    assert len(rebuilt) == 4


@pytest.mark.parametrize("copy_function", _COPIES, ids=_COPY_IDS)
def test_a_lattice_pickles_and_copies_with_its_meets_and_joins(
    copy_function: Callable[[Any], Any],
) -> None:
    """Test a lattice copies with its order, meets and joins."""
    original = _fill_crown(Lattice())

    rebuilt = copy_function(original)

    assert type(rebuilt) is Lattice
    assert rebuilt.is_lattice()
    assert rebuilt.get_meet(1, 2) == 0
    assert rebuilt.get_join(1, 2) == 3
    assert all(element in rebuilt for element in (0, 1, 2, 3))


@pytest.mark.parametrize("copy_function", _COPIES, ids=_COPY_IDS)
def test_a_subclass_copies_with_its_instance_state_without_calling_init(
    copy_function: Callable[[Any], Any],
) -> None:
    """Test a subclass whose `__init__` needs an argument copies, with its state."""
    poset = _build_poset(_TaggedPoset("p"))
    lattice = _fill_crown(_TaggedLattice("l"))

    rebuilt_poset = copy_function(poset)
    rebuilt_lattice = copy_function(lattice)

    assert type(rebuilt_poset) is _TaggedPoset
    assert rebuilt_poset.tag == "p"
    assert list(rebuilt_poset) == ["d", "c", "b", "a"]
    assert type(rebuilt_lattice) is _TaggedLattice
    assert rebuilt_lattice.tag == "l"
    assert rebuilt_lattice.get_join(1, 2) == 3


def test_a_copy_is_independent_of_the_original() -> None:
    """Test adding to a copy leaves the original unchanged."""
    original = _build_poset(PartiallyOrderedSet())

    rebuilt = copy.copy(original)
    rebuilt.add_element("e")
    rebuilt.add_order("c", "e")

    assert "e" not in original
    assert len(original) == 4


def test_a_deep_copy_copies_the_elements() -> None:
    """Test a deep copy holds copies of the elements, and a shallow one the same."""
    element = frozenset({1})
    poset: PartiallyOrderedSet[Any] = PartiallyOrderedSet()
    poset.add_element(element)
    shared = [object()]

    class Holder:
        def __init__(self, value: list[object]) -> None:
            self.value = value

        @override
        def __hash__(self) -> int:
            return 7

        @override
        def __eq__(self, other: object) -> bool:
            return isinstance(other, Holder)

    holder = Holder(shared)
    poset.add_element(holder)

    deep = list(copy.deepcopy(poset))
    shallow = list(copy.copy(poset))

    assert shallow[1] is holder
    assert deep[1] is not holder
    assert deep[1].value is not shared


def test_setstate_refuses_a_state_it_did_not_write() -> None:
    """Test `__setstate__` of a malformed state raises `TypeError`."""
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()

    with pytest.raises(TypeError, match="__reduce__"):
        poset.__setstate__(("not", "a", "state", "at", "all"))


# =============================================================================
# Re-entrant reads and changes
# =============================================================================


def test_iter_stable_with_a_key_that_adds_an_element_iterates_it_too() -> None:
    """Test `iter_stable`'s key may add an element, which is iterated last.

    The key runs over a copy of the elements under no borrow; the element it
    adds has no rank, so it follows the ranked ones, which it is not ordered
    against, in insertion order.
    """
    poset: PartiallyOrderedSet[int] = PartiallyOrderedSet()
    for element in (1, 2, 3):
        poset.add_element(element)
    poset.add_order(1, 2)

    def key(element: int) -> int:
        if element == 2 and 99 not in poset:
            poset.add_element(99)
        return -element

    assert list(poset.iter_stable(key)) == [3, 1, 2, 99]
    assert len(poset) == 4


def test_an_element_whose_hash_reads_the_set_sees_it_before_the_insertion() -> None:
    """Test `add_element` hashes the element under no borrow.

    The hash reads the set, which holds the elements added before, and the
    element is then added.
    """
    poset: PartiallyOrderedSet[object] = PartiallyOrderedSet()
    poset.add_element(1)
    seen: list[list[object]] = []

    class NosyHash:
        @override
        def __hash__(self) -> int:
            seen.append(list(poset))
            return 1

    nosy = NosyHash()
    poset.add_element(nosy)

    assert seen[0] == [1]
    assert list(poset) == [1, nosy]
    assert nosy in poset


def test_an_element_whose_hash_adds_an_element_keeps_both_positions() -> None:
    """Test an element added by another's hash leaves both correctly placed.

    The inner element takes the position the outer one was going to take,
    and the outer one is recorded at the next, so each is ordered as itself.
    """
    lattice: Lattice[object] = Lattice()
    lattice.add_element("bottom")

    class AddsOnFirstHash:
        def __init__(self) -> None:
            self.hashed = False

        @override
        def __hash__(self) -> int:
            if not self.hashed:
                self.hashed = True
                lattice.add_element("inner")
            return 2

    outer = AddsOnFirstHash()
    lattice.add_element(outer)
    lattice.add_order("bottom", "inner")
    lattice.add_order("bottom", outer)

    assert lattice.get_meet("inner", outer) == "bottom"
    assert lattice.get_join("inner", outer) is None
    assert all(element in lattice for element in ("bottom", "inner", outer))


def test_orders_and_queries_compare_their_arguments_under_no_borrow() -> None:
    """Test an argument's `__eq__` may query the lattice from `add_order`.

    `probe` equals the member `member` without being it, so looking it up
    calls `__eq__`, which asks the lattice for a meet.
    """
    lattice: Lattice[object] = Lattice()
    calls: list[bool] = []

    class Key:
        @override
        def __hash__(self) -> int:
            return 3

        @override
        def __eq__(self, other: object) -> bool:
            calls.append(lattice.has_meet(1, 2))
            return isinstance(other, Key)

    member, probe = Key(), Key()
    for element in (1, 2, member):
        lattice.add_element(element)
    lattice.add_order(1, 2)
    lattice.add_order(1, probe)

    assert lattice.get_meet(2, probe) == 1
    assert calls
    assert all(calls)


def test_verify_writes_reprs_that_change_the_lattice() -> None:
    """Test `verify` builds its report after the borrow, so a `repr` may add."""
    lattice: Lattice[object] = Lattice()

    class Nosy:
        def __init__(self, name: str) -> None:
            self.name = name

        @override
        def __repr__(self) -> str:
            if "late" not in lattice:
                lattice.add_element("late")
            return self.name

    for element in (Nosy("a"), Nosy("b")):
        lattice.add_element(element)

    messages = [diagnostic.message.message for diagnostic in lattice.verify().errors()]

    assert "lattice has no unique meet for elements a and b" in messages
    assert "late" in lattice
