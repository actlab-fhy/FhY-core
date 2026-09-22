"""Tests the Rust identifier id counter against the pure-Python counter.

The extension ``fhy_core._rs`` exposes the Rust backend's id counter as
``allocate_identifier_id`` and ``advance_identifier_counter_past``, which the
public ``Identifier`` draws from when the package selected the Rust backend.
The Rust counter is process-global and the Python counter is fresh per test,
so ids are compared relative to an anchor, never absolutely.
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest

import fhy_core
from fhy_core.identifier import (
    Identifier,
    _PythonIdCounter,  # the reference implementation of the Rust counter
)

_rs = pytest.importorskip("fhy_core._rs")


class _Counter(NamedTuple):
    """The two operations of one identifier id counter."""

    allocate: Callable[[], int]
    advance_past: Callable[[int], None]


_RUST_COUNTER = _Counter(
    _rs.allocate_identifier_id, _rs.advance_identifier_counter_past
)

# Allocate/advance steps whose relative ids are computed by hand in
# `test_counters_issue_the_same_relative_ids_for_the_same_operations`. Advance
# offsets land ahead of, behind, and exactly on the counter.
_OPERATION_SCRIPT: list[tuple[str, int]] = [
    ("allocate", 0),
    ("advance", 50),
    ("allocate", 0),
    ("advance", 1),
    ("allocate", 0),
    ("advance", 53),
    ("allocate", 0),
    ("advance", 200),
    ("advance", 100),
    ("allocate", 0),
]


def _create_python_counter() -> _Counter:
    """Return the operations of a fresh pure-Python counter."""
    python_counter = _PythonIdCounter()
    return _Counter(python_counter.allocate, python_counter.advance_past)


def _run_operation_script(counter: _Counter) -> list[int]:
    """Run the operation script and return the allocated ids relative to an anchor."""
    base = counter.allocate()
    relative_ids = []
    for operation, offset in _OPERATION_SCRIPT:
        if operation == "allocate":
            relative_ids.append(counter.allocate() - base)
        else:
            counter.advance_past(base + offset)
    return relative_ids


@pytest.fixture(params=["rust", "python"])
def counter(request: pytest.FixtureRequest) -> _Counter:
    """Return the Rust counter or a fresh pure-Python counter."""
    return _RUST_COUNTER if request.param == "rust" else _create_python_counter()


# =============================================================================
# Counter equivalence
# =============================================================================


def test_counters_issue_the_same_relative_ids_for_the_same_operations() -> None:
    """Test both counters issue identical ids, relative to an anchor, for one script."""
    rust_relative_ids = _run_operation_script(_RUST_COUNTER)
    python_relative_ids = _run_operation_script(_create_python_counter())

    assert rust_relative_ids == [1, 51, 52, 54, 201]
    assert python_relative_ids == rust_relative_ids


def test_counter_allocates_consecutive_ids(counter: _Counter) -> None:
    """Test successive allocations issue ids one apart."""
    first = counter.allocate()
    second = counter.allocate()

    assert second == first + 1


def test_counter_advance_makes_the_next_id_the_exact_successor(
    counter: _Counter,
) -> None:
    """Test advancing past an id ahead of the counter issues its successor next."""
    base = counter.allocate()

    counter.advance_past(base + 43)

    assert counter.allocate() == base + 44


def test_counter_advance_does_not_rewind(counter: _Counter) -> None:
    """Test advancing past an already-issued id leaves the counter where it was."""
    first = counter.allocate()
    last = counter.allocate()

    counter.advance_past(first)

    assert counter.allocate() == last + 1


def test_counter_advancing_past_the_largest_64_bit_id_raises_runtime_error(
    counter: _Counter,
) -> None:
    """Test advancing past `2**64 - 1` raises `RuntimeError`, counter unchanged."""
    base = counter.allocate()

    with pytest.raises(RuntimeError, match=r"^identifier id space exhausted$"):
        counter.advance_past(2**64 - 1)

    assert counter.allocate() == base + 1


def test_python_counter_issues_the_largest_id_then_fails() -> None:
    """Test the Python counter issues `2**64 - 2` and then refuses to allocate.

    The Rust counter is process-global, so its equivalent runs in a
    subprocess in the identifier tests.
    """
    python_counter = _create_python_counter()
    python_counter.advance_past(2**64 - 3)

    largest = python_counter.allocate()

    assert largest == 2**64 - 2
    with pytest.raises(RuntimeError, match="identifier id space exhausted"):
        python_counter.allocate()


# =============================================================================
# Backend wiring
# =============================================================================


@pytest.mark.skipif(
    not fhy_core.RUST_BACKEND_SELECTED, reason="the Rust backend is not selected"
)
def test_public_identifier_draws_ids_from_the_rust_counter_when_selected() -> None:
    """Test the public class shares the Rust counter on the Rust backend."""
    first = Identifier("first")
    allocated = _RUST_COUNTER.allocate()
    second = Identifier("second")

    assert (allocated, second.id) == (first.id + 1, first.id + 2)


@pytest.mark.skipif(
    fhy_core.RUST_BACKEND_SELECTED, reason="the Rust backend is selected"
)
def test_public_identifier_leaves_the_rust_counter_alone_when_unselected() -> None:
    """Test the public class never draws from the Rust counter on the Python backend."""
    before = _RUST_COUNTER.allocate()
    Identifier("public")
    Identifier.deserialize_from_dict({"id": before + 1000, "name_hint": "far"})
    after = _RUST_COUNTER.allocate()

    assert after == before + 1
