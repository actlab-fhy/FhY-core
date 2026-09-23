"""Hypothesis property tests comparing the Rust and pure-Python id counters.

Both counters run the same generated sequence of allocations and advances,
the Rust one against its process-global state and the Python one from a
fresh start, and must issue the same ids relative to an anchor allocated just
before the sequence.
"""

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.identifier import (
    _PythonIdCounter,  # the reference implementation of the Rust counter
)

from .conftest import run_counter_operations

_rs = pytest.importorskip("fhy_core._rs")

pytestmark = pytest.mark.property

# `None` allocates; an integer advances the counter past the id that far past
# the anchor.
_operation_sequences = st.lists(
    st.none() | st.integers(min_value=0, max_value=500), max_size=20
)


@given(operations=_operation_sequences)
def test_counters_issue_the_same_relative_ids_for_any_operation_sequence(
    operations: list[int | None],
) -> None:
    """Test both counters agree on every id for one allocate/advance sequence."""
    python_counter = _PythonIdCounter()

    rust_relative_ids = run_counter_operations(
        operations, _rs.allocate_identifier_id, _rs.advance_identifier_counter_past
    )
    python_relative_ids = run_counter_operations(
        operations, python_counter.allocate, python_counter.advance_past
    )

    assert python_relative_ids == rust_relative_ids
