"""Hypothesis property tests for `PassManager` run-order and fixpoint termination.

Kept intentionally small: a handful of toy identity passes over a
`list[int]` IR, registered once at import time, cover the run-order law;
one toy decrementing pass covers fixpoint termination. Both laws are
independent of `PassManager`'s internals: the first compares against the
sequence the test itself chose to add, the second against the arithmetic
bound on how many iterations a strictly-decreasing pass needs to reach a
fixed point.
"""

from collections.abc import Callable

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given
from hypothesis import strategies as st

from fhy_core.pass_infrastructure import (
    CompilerPass,
    FixpointGroupRecord,
    FixpointPassGroup,
    PassManager,
    PassRunRecord,
    register_pass,
)
from fhy_core.utils.override import override

from ..conftest import mock_identifier

pytestmark = pytest.mark.property

_TOY_PASS_MODULE = "fhy_core_tests.pass_infrastructure.test_manager_properties"


_RecordingPassBuilder = Callable[[list[str]], CompilerPass[list[int], list[int]]]


def _build_recording_pass(index: int) -> tuple[str, _RecordingPassBuilder]:
    """Register one toy identity pass and return its name and constructor.

    The recorder is injected per instance (not a class attribute), so every
    Hypothesis example gets an independent recording even though the pass
    class itself is registered once, at import time. Returning the class as
    a plain `Callable[[list[str]], CompilerPass[...]]` (rather than
    `type[CompilerPass[...]]`) keeps its constructor signature visible to
    the type checker, which erases a dynamically-built subclass's `__init__`
    back to the base class's no-argument one otherwise.
    """
    name = f"{_TOY_PASS_MODULE}.recording_pass_{index}"

    @register_pass(
        name, f"Toy identity pass {index} that records its own name when run."
    )
    class _RecordingPass(CompilerPass[list[int], list[int]]):
        def __init__(self, recorder: list[str]) -> None:
            super().__init__()
            self._recorder = recorder

        @override
        def get_noop_output(self, ir: list[int]) -> list[int]:
            return ir

        @override
        def run_pass(self, ir: list[int]) -> list[int]:
            self._recorder.append(self.get_pass_name())
            return ir

    return name, _RecordingPass


_RECORDING_PASSES: tuple[tuple[str, _RecordingPassBuilder], ...] = tuple(
    _build_recording_pass(index) for index in range(5)
)


@register_pass(
    f"{_TOY_PASS_MODULE}.decrement_positive",
    "Toy pass: decrements every positive int in the list by one.",
)
class _DecrementPositivePass(CompilerPass[list[int], list[int]]):
    """Decrements every positive element by one; non-positive elements are unchanged."""

    @override
    def get_noop_output(self, ir: list[int]) -> list[int]:
        return ir

    @override
    def run_pass(self, ir: list[int]) -> list[int]:
        return [value - 1 if value > 0 else value for value in ir]


@given(
    indices=st.lists(
        st.integers(min_value=0, max_value=len(_RECORDING_PASSES) - 1),
        min_size=1,
        max_size=5,
    )
)
def test_pass_manager_runs_passes_in_registration_order(indices: list[int]) -> None:
    """Test PassManager.run executes passes in the exact order they were added.

    Oracle: the sequence of pass names the test itself chose to add, compared
    against both an independently recorded run order (appended by each pass
    instance) and the manager's own PassRunRecord order.
    """
    recorder: list[str] = []
    manager: PassManager[list[int]] = PassManager()
    expected_names = [_RECORDING_PASSES[index][0] for index in indices]
    for index in indices:
        _, build_pass = _RECORDING_PASSES[index]
        manager.add_pass(build_pass(recorder))

    result = manager.run([0])

    assert recorder == expected_names
    run_names = [
        record.pass_name
        for record in result.records
        if isinstance(record, PassRunRecord)
    ]
    assert run_names == expected_names


@given(values=st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=5))
def test_fixpoint_group_of_decrementing_pass_terminates_with_matching_iterations(
    values: list[int],
) -> None:
    """Test a fixpoint group of a strictly-decrementing pass converges within budget.

    Oracle: the pass decrements every positive element by exactly one per
    iteration, so the list reaches an all-non-positive fixed point after at
    most `max(values, default=0)` changing iterations, plus one confirming
    iteration where nothing changes; `max_iterations` is sized with margin
    above that bound so the group is guaranteed to converge rather than
    raise `PassExecutionError`.
    """
    max_iterations = max(values, default=0) + 2
    manager: PassManager[list[int]] = PassManager()
    group: FixpointPassGroup[list[int]] = FixpointPassGroup(
        name=mock_identifier("decrement-to-fixpoint", 30_000),
        max_iterations=max_iterations,
    )
    group.add_pass(_DecrementPositivePass())
    manager.add_fixpoint_group(group)

    result = manager.run(list(values))

    assert result.output == [0 for _ in values]
    fixpoint_records = [
        record for record in result.records if isinstance(record, FixpointGroupRecord)
    ]
    assert len(fixpoint_records) == 1
    record = fixpoint_records[0]
    assert record.converged is True
    assert record.iterations == len(record.iteration_records)
    assert record.iterations <= max_iterations
