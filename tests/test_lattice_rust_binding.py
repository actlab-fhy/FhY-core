"""Tests for the Python interface over the Rust-backed poset and lattice.

``PartiallyOrderedSet`` and ``Lattice`` are thin subclasses of
``fhy_core._rs.PartiallyOrderedSet`` and ``fhy_core._rs.Lattice``, which keep
the element objects in a dict and run the order in the Rust core (S11a of
``docs/design/python-switch.md``). These tests cover what the binding adds
around the core: the class structure, the elements' Python hashing and
equality, the key calls of ``iter_stable``, the ``verify`` report, and the
exception classes and texts.
"""

from typing import Any

import pytest

from fhy_core import _rs
from fhy_core.diagnostic import DiagnosticLevel, ValidationReport
from fhy_core.lattice import Lattice
from fhy_core.traits.verifiable import Verifiable, VerifiableMixin
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
    """Test T-11: ties come in the order the elements were added."""
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
