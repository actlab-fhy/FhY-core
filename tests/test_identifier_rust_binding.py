"""Tests the Rust identifier id counter the public ``Identifier`` draws from.

The extension ``fhy_core._rs`` exposes its process-global id counter as
``allocate_identifier_id`` and ``advance_identifier_counter_past``. Other
tests in the process advance the same counter, so ids are compared relative
to an anchor, never absolutely.
"""

import re

import pytest

from fhy_core import _rs
from fhy_core.identifier import Identifier

from .conftest import run_counter_operations

# Allocate (`None`) and advance steps, followed by the runner's final
# allocation, whose relative ids are computed by hand in
# `test_counter_issues_the_expected_relative_ids_for_a_script`. Advance
# offsets land ahead of, behind, and exactly on the counter.
_OPERATION_SCRIPT: list[int | None] = [None, 50, None, 1, None, 53, None, 200, 100]


# =============================================================================
# Counter semantics
# =============================================================================


def test_counter_issues_the_expected_relative_ids_for_a_script() -> None:
    """Test a script of allocations and advances issues the ids worked out by hand."""
    relative_ids = run_counter_operations(
        _OPERATION_SCRIPT,
        _rs.allocate_identifier_id,
        _rs.advance_identifier_counter_past,
    )

    assert relative_ids == [1, 51, 52, 54, 201]


def test_counter_allocates_consecutive_ids() -> None:
    """Test successive allocations issue ids one apart."""
    first = _rs.allocate_identifier_id()
    second = _rs.allocate_identifier_id()

    assert second == first + 1


def test_counter_advance_makes_the_next_id_the_exact_successor() -> None:
    """Test advancing past an id ahead of the counter issues its successor next."""
    base = _rs.allocate_identifier_id()

    _rs.advance_identifier_counter_past(base + 43)

    assert _rs.allocate_identifier_id() == base + 44


def test_counter_advance_does_not_rewind() -> None:
    """Test advancing past an already-issued id leaves the counter where it was."""
    first = _rs.allocate_identifier_id()
    last = _rs.allocate_identifier_id()

    _rs.advance_identifier_counter_past(first)

    assert _rs.allocate_identifier_id() == last + 1


@pytest.mark.parametrize("identifier_id", [2**62, 2**63 - 1, 2**63, 2**64 - 1])
def test_counter_advancing_past_a_foreign_id_at_or_above_2_pow_62_raises(
    identifier_id: int,
) -> None:
    """Test advancing past an id in `[2**62, 2**64)` not issued here raises.

    It raises `OverflowError` naming both bounds, and leaves the counter
    unchanged.
    """
    base = _rs.allocate_identifier_id()

    with pytest.raises(
        OverflowError,
        match=(
            f"^identifier id {identifier_id} is out of range: a payload id must "
            "be below 4611686018427387904, or below 9223372036854775808 if this "
            "process issued it$"
        ),
    ):
        _rs.advance_identifier_counter_past(identifier_id)

    assert _rs.allocate_identifier_id() == base + 1


def test_next_identifier_id_is_the_id_the_counter_issues_next() -> None:
    """Test the counter's next id lies after the last id drawn from it."""
    drawn = _rs.allocate_identifier_id()
    peeked = _rs.next_identifier_id()

    assert drawn < peeked <= _rs.allocate_identifier_id()


@pytest.mark.parametrize(
    ("identifier_id", "message"),
    [
        (-1, "can't convert negative int to unsigned"),
        (2**64, "int too big to convert"),
        (2**70, "int too big to convert"),
    ],
)
def test_counter_advancing_past_an_id_outside_64_bits_raises_overflow_error(
    identifier_id: int, message: str
) -> None:
    """Test advancing past an id outside `[0, 2**64)` raises, counter unchanged.

    PyO3 rejects these before the Rust counter sees them, with its own
    messages.
    """
    base = _rs.allocate_identifier_id()

    with pytest.raises(OverflowError, match=f"^{re.escape(message)}"):
        _rs.advance_identifier_counter_past(identifier_id)

    assert _rs.allocate_identifier_id() == base + 1


# =============================================================================
# Wiring of the public class
# =============================================================================


def test_public_identifier_draws_ids_from_the_rust_counter() -> None:
    """Test the public class shares the Rust counter."""
    first = Identifier("first")
    allocated = _rs.allocate_identifier_id()
    second = Identifier("second")

    assert (allocated, second.id) == (first.id + 1, first.id + 2)


def test_public_identifier_deserialization_advances_the_rust_counter() -> None:
    """Test decoding an id ahead of the counter advances the Rust counter."""
    base = _rs.allocate_identifier_id()

    Identifier.deserialize_from_dict({"id": base + 1000, "name_hint": "far"})

    assert _rs.allocate_identifier_id() == base + 1001
