"""Hypothesis property tests of the Rust id counter against a model.

The Rust counter runs a generated sequence of allocations and advances
against its process-global state. The model is the counter's contract: an
allocation issues the next id and moves past it, and advancing past an id
moves the counter just past that id unless it is already further. Relative
to an anchor allocated just before the sequence, both must issue the same
ids.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core import _rs

from .conftest import run_counter_operations

pytestmark = pytest.mark.property

# `None` allocates; an integer advances the counter past the id that far past
# the anchor.
_operation_sequences = st.lists(
    st.none() | st.integers(min_value=0, max_value=500), max_size=20
)


class _ModelCounter:
    """The counter's contract over ids relative to an anchor of `0`."""

    def __init__(self) -> None:
        self._next_id = 0

    def allocate(self) -> int:
        """Issue the next id and move past it."""
        identifier_id = self._next_id
        self._next_id += 1
        return identifier_id

    def advance_past(self, identifier_id: int) -> None:
        """Move past `identifier_id`, never back."""
        self._next_id = max(self._next_id, identifier_id + 1)


@given(operations=_operation_sequences)
def test_counter_issues_the_model_ids_for_any_operation_sequence(
    operations: list[int | None],
) -> None:
    """Test the Rust counter agrees with the model on every relative id."""
    model = _ModelCounter()

    rust_relative_ids = run_counter_operations(
        operations, _rs.allocate_identifier_id, _rs.advance_identifier_counter_past
    )
    model_relative_ids = run_counter_operations(
        operations, model.allocate, model.advance_past
    )

    assert rust_relative_ids == model_relative_ids
