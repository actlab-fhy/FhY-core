"""Tests the pass manager infrastructure."""

import gc
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    Analysis,
    AnalysisManager,
    CompilerPass,
    FixpointGroupRecord,
    FixpointIterationRecord,
    FixpointPassGroup,
    PassExecutionError,
    PassManager,
    PassRunRecord,
    PreservedAnalyses,
    register_pass,
)
from fhy_core.traits import FrozenMixin, PartialEqual
from fhy_core.utils.override import override

_MANAGER_LOGGER = "fhy_core.pass_infrastructure.manager"


@dataclass
class Box(FrozenMixin):
    """Simple immutable IR for pass manager tests."""

    value: int

    def __post_init__(self) -> None:
        self.freeze()


@dataclass
class MutableBox:
    """Simple mutable IR used to assert no analysis caching."""

    value: int


class BoxDoubleAnalysis(Analysis[Box, int]):
    """Analysis that doubles the box value and tracks run count."""

    runs = 0

    @override
    def run(self, ir: Box) -> int:
        type(self).runs += 1
        return ir.value * 2


class BoxParityAnalysis(Analysis[Box, int]):
    """Analysis that computes parity and tracks run count."""

    runs = 0

    @override
    def run(self, ir: Box) -> int:
        type(self).runs += 1
        return ir.value % 2


class MutableBoxDoubleAnalysis(Analysis[MutableBox, int]):
    """Analysis for mutable IR cache behavior tests."""

    runs = 0

    @override
    def run(self, ir: MutableBox) -> int:
        type(self).runs += 1
        return ir.value * 2


def test_pass_manager_runs_passes_in_order() -> None:
    """Test that pass manager runs passes in insertion order."""

    @register_pass("tests.pm.add_one", "Add one to the Box value.")
    class AddOnePass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

    @register_pass("tests.pm.double", "Double the Box value.")
    class DoublePass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value * 2)

    manager = PassManager[Box]()
    manager.add_pass(AddOnePass())
    manager.add_pass(DoublePass())

    result = manager.run(Box(3))

    assert result.output == Box(8)
    assert len(result.records) == 2


def test_pass_manager_applies_analysis_preservation_and_invalidation() -> None:
    """Test that analysis cache transfer respects preserved analyses."""
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0
    observed: list[tuple[int, int]] = []

    @register_pass("tests.pm.read_both", "Read both analyses of the IR.")
    class ReadBothPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            observed.append(
                (
                    self.get_analysis(BoxDoubleAnalysis, ir),
                    self.get_analysis(BoxParityAnalysis, ir),
                )
            )
            return ir

    @register_pass(
        "tests.pm.preserve_double_only",
        "Change IR while preserving only the double analysis.",
    )
    class PreserveDoubleOnlyPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            _ = (input_ir, output, changed)
            return PreservedAnalyses.none().preserve(
                BoxDoubleAnalysis.get_analysis_name()
            )

    manager = PassManager[Box]()
    manager.add_pass(ReadBothPass())
    manager.add_pass(PreserveDoubleOnlyPass())
    manager.add_pass(ReadBothPass())

    result = manager.run(Box(2))

    # The preserved double result is carried to the new box, although it
    # doubles the old value; the parity is computed afresh for the new box.
    assert result.output == Box(3)
    assert observed == [(4, 0), (4, 1)]
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 2


def test_analysis_manager_does_not_cache_non_frozen_ir() -> None:
    """Test that a pipeline run skips caching for non-frozen IR."""
    MutableBoxDoubleAnalysis.runs = 0
    observed: list[int] = []

    @register_pass("tests.pm.read_mutable_twice", "Read an analysis of mutable IR.")
    class ReadTwicePass(CompilerPass[MutableBox, MutableBox]):
        @override
        def run_pass(self, ir: MutableBox) -> MutableBox:
            observed.append(self.get_analysis(MutableBoxDoubleAnalysis, ir))
            observed.append(self.get_analysis(MutableBoxDoubleAnalysis, ir))
            return ir

    manager = PassManager[MutableBox]()
    manager.add_pass(ReadTwicePass())
    manager.run(MutableBox(3))

    assert observed == [6, 6]
    assert MutableBoxDoubleAnalysis.runs == 2


def test_analysis_manager_does_not_block_ir_from_garbage_collection() -> None:
    """Test that caching an analysis result does not pin the IR after the run.

    A run's cache holds each cached node until the run ends, and releases
    it then. If the cache outlived the run, the weakref below would still
    resolve after `gc.collect()`.
    """
    BoxDoubleAnalysis.runs = 0

    @register_pass("tests.pm.read_then_replace", "Read an analysis, return new IR.")
    class ReadThenReplacePass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return Box(ir.value + 1)

    manager = PassManager[Box]()
    manager.add_pass(ReadThenReplacePass())
    ir = Box(3)
    weak_ir = weakref.ref(ir)

    manager.run(ir)
    assert weak_ir() is not None  # ir still alive while caller holds it

    del ir
    gc.collect()

    assert weak_ir() is None


def test_analysis_identifier_is_unique() -> None:
    """Test analyses expose unique identifier keys."""
    assert (
        BoxDoubleAnalysis.get_analysis_name() != BoxParityAnalysis.get_analysis_name()
    )


def test_pass_manager_fixpoint_group_converges() -> None:
    """Test that a fixpoint group converges and records iterations."""

    @register_pass("tests.pm.decrement_to_zero", "Decrement value toward zero.")
    class DecrementToZeroPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return max(ir - 1, 0)

    manager = PassManager[int]()
    fixpoint_group = FixpointPassGroup[int](
        name=Identifier("decrement-group"), max_iterations=10
    )
    fixpoint_group.add_pass(DecrementToZeroPass())
    manager.add_fixpoint_group(fixpoint_group)

    result = manager.run(3)

    assert result.output == 0
    assert len(result.records) == 1
    record = result.records[0]
    assert isinstance(record, FixpointGroupRecord)
    assert isinstance(record, PartialEqual)
    assert record.supports_partial_equality is True
    assert record.converged is True
    assert record.iterations == 4
    assert record.iteration_records
    assert isinstance(record.iteration_records[0], PartialEqual)
    assert record.iteration_records[0].supports_partial_equality is True


def test_pass_manager_fixpoint_group_raises_on_non_convergence() -> None:
    """Test that non-convergent fixpoint groups raise PassExecutionError."""

    @register_pass("tests.pm.flip_bit", "Flip a bit forever.")
    class FlipBitPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return 1 - ir

    manager = PassManager[int]()
    fixpoint_group = FixpointPassGroup[int](
        name=Identifier("flip-group"),
        max_iterations=3,
        fail_on_non_convergence=True,
    )
    fixpoint_group.add_pass(FlipBitPass())
    manager.add_fixpoint_group(fixpoint_group)

    with pytest.raises(PassExecutionError) as excinfo:
        manager.run(0)

    assert str(excinfo.value) == (
        'fixpoint group "flip-group" did not converge (max iterations: 3)'
    )
    assert excinfo.value.pass_name is None
    assert excinfo.value.hook is None
    (record,) = excinfo.value.records
    assert isinstance(record, FixpointGroupRecord)
    assert record.group_name is fixpoint_group.name
    assert record.converged is False
    assert record.iterations == 3


def test_fixpoint_group_configuration_is_read_only() -> None:
    """Test fixpoint group configuration fields are read-only after init."""
    group = FixpointPassGroup[int](name=Identifier("cfg"), max_iterations=2)

    with pytest.raises(AttributeError):
        group.max_iterations = 10  # type: ignore[misc]
    with pytest.raises(AttributeError):
        group.fail_on_non_convergence = False  # type: ignore[misc]


def test_pass_manager_configuration_is_read_only() -> None:
    """Test pass manager configuration fields are read-only after init."""
    manager = PassManager[int](name=Identifier("pipeline"))

    with pytest.raises(AttributeError):
        setattr(manager, "name", Identifier("other"))  # noqa: B010


def test_get_analysis_runs_uncached_when_pass_is_standalone() -> None:
    """Test that get_analysis computes fresh each call when no manager is bound."""
    BoxDoubleAnalysis.runs = 0

    @register_pass("tests.pm.standalone_get_analysis", "Reads analysis standalone.")
    class ReadAnalysisPass(CompilerPass[Box, Box]):
        observed: ClassVar[list[int]] = []

        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            ReadAnalysisPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            ReadAnalysisPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    pass_ = ReadAnalysisPass()
    pass_.execute(Box(5))

    # Two calls -> two runs (no cache available when standalone).
    assert ReadAnalysisPass.observed == [10, 10]
    assert BoxDoubleAnalysis.runs == 2


def test_get_analysis_uses_cache_when_bound_by_pass_manager() -> None:
    """Test that get_analysis hits the manager's cache for duplicate requests."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.cached_get_analysis", "Reads analysis twice under a manager."
    )
    class TwiceReadPass(CompilerPass[Box, Box]):
        observed: ClassVar[list[int]] = []

        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            TwiceReadPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            TwiceReadPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    manager = PassManager[Box]()
    manager.add_pass(TwiceReadPass())
    manager.run(Box(5))

    assert TwiceReadPass.observed == [10, 10]
    assert BoxDoubleAnalysis.runs == 1


def test_get_analysis_reuses_cache_across_preserving_passes() -> None:
    """Test that get_analysis reuses the cache across passes that preserve it."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.compute_analysis", "Triggers the analysis in its first run."
    )
    class ComputeAnalysisPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir  # identity - preserves all by default (no change)

    @register_pass("tests.pm.read_analysis_again", "Reads the analysis a second time.")
    class ReadAgainPass(CompilerPass[Box, Box]):
        observed: ClassVar[list[int]] = []

        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            ReadAgainPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    manager = PassManager[Box]()
    manager.add_pass(ComputeAnalysisPass())
    manager.add_pass(ReadAgainPass())
    manager.run(Box(5))

    assert ReadAgainPass.observed == [10]
    # One run across both passes, because the first pass didn't change the IR.
    assert BoxDoubleAnalysis.runs == 1


def test_get_analysis_recomputes_after_non_preserving_pass() -> None:
    """Test that get_analysis recomputes when an earlier pass did not preserve it."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.seed_analysis", "Computes the analysis before the mutating pass."
    )
    class SeedAnalysisPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir

    @register_pass(
        "tests.pm.mutate_without_preserve",
        "Changes the IR and preserves no analyses (default).",
    )
    class MutateNoPreservePass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

    @register_pass(
        "tests.pm.reread_after_mutation", "Re-reads the analysis after mutation."
    )
    class RereadPass(CompilerPass[Box, Box]):
        observed: ClassVar[list[int]] = []

        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            RereadPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    manager = PassManager[Box]()
    manager.add_pass(SeedAnalysisPass())
    manager.add_pass(MutateNoPreservePass())
    manager.add_pass(RereadPass())
    manager.run(Box(5))

    # First run computed on Box(5) -> 10. Mutation invalidated it.
    # Second run computed on Box(6) -> 12.
    assert RereadPass.observed == [12]
    assert BoxDoubleAnalysis.runs == 2


def test_get_analysis_manager_is_the_hook_view_of_the_run_cache() -> None:
    """Test that `get_analysis_manager` returns the running hook's cache view.

    Outside a run it returns `None`; during a hook it returns an
    `AnalysisManager` whose `get` reads the same cache as `get_analysis`.
    """
    BoxDoubleAnalysis.runs = 0
    observed: list[int] = []

    @register_pass(
        "tests.pm.public_bind_accessors", "Identity pass for accessor testing."
    )
    class AccessorPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            analysis_manager = self.get_analysis_manager()
            assert isinstance(analysis_manager, AnalysisManager)
            observed.append(analysis_manager.get(BoxDoubleAnalysis, ir))
            observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    compiler_pass = AccessorPass()
    assert compiler_pass.get_analysis_manager() is None

    manager = PassManager[Box]()
    manager.add_pass(compiler_pass)
    manager.run(Box(3))

    assert observed == [6, 6]
    assert BoxDoubleAnalysis.runs == 1
    assert compiler_pass.get_analysis_manager() is None


def test_get_analysis_works_inside_fixpoint_group() -> None:
    """Test that get_analysis is available to passes within a fixpoint group."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.fixpoint_reader",
        "Reads analysis and decrements until fixed-point.",
    )
    class FixpointReaderPass(CompilerPass[Box, Box]):
        observed: ClassVar[list[int]] = []

        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            FixpointReaderPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return Box(max(ir.value - 1, 0))

    manager = PassManager[Box]()
    fixpoint_group = FixpointPassGroup[Box](
        name=Identifier("read-then-decrement"), max_iterations=10
    )
    fixpoint_group.add_pass(FixpointReaderPass())
    manager.add_fixpoint_group(fixpoint_group)
    manager.run(Box(2))

    # Iteration 1: Box(2) -> observes 4 -> returns Box(1)
    # Iteration 2: Box(1) -> observes 2 -> returns Box(0)
    # Iteration 3: Box(0) -> observes 0 -> returns Box(0); converged.
    assert FixpointReaderPass.observed == [4, 2, 0]


def test_get_analysis_restores_pass_state_between_runs() -> None:
    """Test that a manager-bound pass has no dangling analysis manager after the run."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.state_restoration", "Identity pass that reads an analysis."
    )
    class StateCheckPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir

    compiler_pass = StateCheckPass()
    manager = PassManager[Box]()
    manager.add_pass(compiler_pass)
    manager.run(Box(5))

    # After the run the pass should no longer be bound to the manager.
    assert compiler_pass.get_analysis_manager() is None

    # Using the pass standalone afterward must fall back to uncached execution.
    compiler_pass.execute(Box(7))
    compiler_pass.execute(Box(7))
    # Standalone re-runs (no cache) -> total runs == 1 (managed) + 2 (standalone) = 3.
    assert BoxDoubleAnalysis.runs == 3


def test_pass_manager_records_support_partial_equal_traits() -> None:
    """Test pass-manager records satisfy `PartialEqual` protocol."""

    @register_pass("tests.pm.partial_equal_record", "Identity pass for records.")
    class IdentityPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    manager = PassManager[int]()
    manager.add_pass(IdentityPass())
    result = manager.run(1)
    run_record = result.records[0]

    assert isinstance(result, PartialEqual)
    assert result.supports_partial_equality is True
    assert isinstance(run_record, PartialEqual)
    assert run_record.supports_partial_equality is True


# ---------------------------------------------------------------------------
# add_* methods return None (no fluent chaining anywhere).
# ---------------------------------------------------------------------------


def test_pass_manager_add_pass_returns_none() -> None:
    """Test that PassManager.add_pass returns None (no chaining)."""

    @register_pass("tests.pm.no_chain_a", "Identity pass for chaining test A.")
    class _NoChainA(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    manager = PassManager[int]()
    # The assertion pins the runtime contract that the return annotation
    # states for type checkers.
    assert manager.add_pass(_NoChainA()) is None


def test_pass_manager_add_fixpoint_group_returns_none() -> None:
    """Test that PassManager.add_fixpoint_group returns None (no chaining)."""
    manager = PassManager[int]()
    group = FixpointPassGroup[int](name=Identifier("no-chain-group"), max_iterations=1)

    assert manager.add_fixpoint_group(group) is None


def test_fixpoint_pass_group_add_pass_returns_none() -> None:
    """Test that FixpointPassGroup.add_pass returns None (no chaining)."""

    @register_pass("tests.pm.no_chain_group_pass", "Identity pass for group chaining.")
    class _NoChainGroupPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    group = FixpointPassGroup[int](name=Identifier("no-chain-group"), max_iterations=1)

    assert group.add_pass(_NoChainGroupPass()) is None


# ---------------------------------------------------------------------------
# Analysis subclass __init__ signature is validated at class creation.
# ---------------------------------------------------------------------------


def test_analysis_subclass_with_no_arg_init_is_accepted() -> None:
    """Test that an Analysis subclass with `__init__(self)` is accepted."""

    class _NoArgAnalysis(Analysis[int, int]):
        def __init__(self) -> None:
            super().__init__()

        @override
        def run(self, ir: int) -> int:
            return ir

    assert _NoArgAnalysis().run(5) == 5


def test_analysis_subclass_with_default_init_is_accepted() -> None:
    """Test that an Analysis subclass with no explicit `__init__` is accepted."""

    class _DefaultInitAnalysis(Analysis[int, int]):
        @override
        def run(self, ir: int) -> int:
            return ir

    assert _DefaultInitAnalysis().run(5) == 5


def test_analysis_subclass_with_required_positional_arg_is_rejected() -> None:
    """Test that an Analysis subclass requiring positional args is rejected at
    class creation time."""
    with pytest.raises(TypeError, match="no-arg"):

        class _BadAnalysis(Analysis[int, int]):
            def __init__(self, scale: int) -> None:
                super().__init__()
                self.scale = scale

            @override
            def run(self, ir: int) -> int:
                return ir * self.scale


def test_analysis_subclass_with_optional_only_args_is_accepted() -> None:
    """Test that an Analysis subclass with only optional/keyword args is accepted."""

    class _OptionalOnlyAnalysis(Analysis[int, int]):
        def __init__(self, *, scale: int = 2) -> None:
            super().__init__()
            self.scale = scale

        @override
        def run(self, ir: int) -> int:
            return ir * self.scale

    assert _OptionalOnlyAnalysis().run(3) == 6


# ---------------------------------------------------------------------------
# PassRunRecord stores PreservedAnalyses directly.
# ---------------------------------------------------------------------------


def test_pass_run_record_stores_preserved_analyses_directly() -> None:
    """Test that `PassRunRecord.preserved_analyses` is a `PreservedAnalyses`."""

    @register_pass(
        "tests.pm.record_schema_preserve_all",
        "Identity pass; preserves all analyses by default for unchanged IR.",
    )
    class IdentityPreserveAllPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    manager = PassManager[int]()
    manager.add_pass(IdentityPreserveAllPass())
    result = manager.run(1)
    record = result.records[0]
    assert isinstance(record, PassRunRecord)

    assert isinstance(record.preserved_analyses, PreservedAnalyses)
    assert record.preserved_analyses.preserve_all is True
    assert record.skipped is False


def test_pass_run_record_carries_specific_preservation_set() -> None:
    """Test that a pass preserving one analysis surfaces that name in the record."""
    name_to_preserve = BoxDoubleAnalysis.get_analysis_name()

    @register_pass(
        "tests.pm.record_schema_preserve_specific",
        "Changes IR while preserving only the double analysis.",
    )
    class PreserveSpecificPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            return PreservedAnalyses.none().preserve(name_to_preserve)

    manager = PassManager[Box]()
    manager.add_pass(PreserveSpecificPass())
    result = manager.run(Box(0))
    record = result.records[0]
    assert isinstance(record, PassRunRecord)

    assert isinstance(record.preserved_analyses, PreservedAnalyses)
    assert record.preserved_analyses.preserve_all is False
    assert record.preserved_analyses.is_preserved(name_to_preserve) is True


# ---------------------------------------------------------------------------
# FixpointGroupRecord.iterations is a property derived from
# iteration_records.
# ---------------------------------------------------------------------------


def test_fixpoint_group_record_iterations_matches_records_length() -> None:
    """Test that `record.iterations == len(record.iteration_records)`."""
    record = FixpointGroupRecord(
        group_name=Identifier("group"),
        iteration_records=(
            FixpointIterationRecord(1, True, ()),
            FixpointIterationRecord(2, True, ()),
            FixpointIterationRecord(3, False, ()),
        ),
        converged=True,
    )

    assert record.iterations == len(record.iteration_records)
    assert record.iterations == 3


def test_fixpoint_group_record_iterations_cannot_be_set() -> None:
    """Test that `iterations` is read-only after construction."""
    record = FixpointGroupRecord(
        group_name=Identifier("group"),
        iteration_records=(),
        converged=False,
    )

    with pytest.raises(AttributeError):
        record.iterations = 42  # type: ignore[misc]


# ---------------------------------------------------------------------------
# A run computes analyses uncached for IR it cannot cache.
# ---------------------------------------------------------------------------


def test_analysis_of_ir_that_is_not_frozen_yet_is_computed_uncached() -> None:
    """Test that a run computes analyses uncached for IR it cannot cache.

    Only a `Frozen` IR that is frozen is cached, so a `FrozenMixin` object
    that is not frozen yet is analyzed afresh on every request.
    """
    BoxDoubleAnalysis.runs = 0

    @dataclass
    class UnfrozenBox(FrozenMixin):
        value: int

    @register_pass("tests.pm.read_unfrozen_twice", "Read an analysis twice.")
    class ReadTwicePass(CompilerPass[Any, Any]):
        @override
        def run_pass(self, ir: Any) -> Any:
            self.get_analysis(BoxDoubleAnalysis, ir)
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir

    manager = PassManager[Any]()
    manager.add_pass(ReadTwicePass())
    manager.run(UnfrozenBox(4))

    # Two calls -> two runs because the IR could not be cached.
    assert BoxDoubleAnalysis.runs == 2


# ---------------------------------------------------------------------------
# AnalysisManager public surface (clear / invalidate / transfer).
# ---------------------------------------------------------------------------


def _build_reading_pass(
    name: str, *analyses: type[Analysis[Box, int]]
) -> CompilerPass[Box, Box]:
    """Return a registered identity pass that reads ``analyses`` of its input."""

    @register_pass(name, f"Reads {len(analyses)} analyses of its input.")
    class _ReadingPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            for analysis in analyses:
                self.get_analysis(analysis, ir)
            return ir

    return _ReadingPass()


def _build_replacing_pass(
    name: str, preserved: PreservedAnalyses
) -> CompilerPass[Box, Box]:
    """Return a registered pass that returns a new box and preserves ``preserved``."""

    @register_pass(name, "Returns a new box, preserving the given analyses.")
    class _ReplacingPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            return preserved

    return _ReplacingPass()


def _build_returning_pass(
    name: str, preserved: PreservedAnalyses
) -> CompilerPass[Box, Box]:
    """Return a registered pass that returns its input and preserves ``preserved``."""

    @register_pass(name, "Returns its input, preserving the given analyses.")
    class _ReturningPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

        @override
        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            return preserved

    return _ReturningPass()


def test_each_run_starts_with_an_empty_cache() -> None:
    """Test that two runs of one pipeline over one IR each compute the analysis.

    The cache lives for one run.
    """
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(_build_reading_pass("tests.pm.cache.clear", BoxDoubleAnalysis))
    ir = Box(3)

    manager.run(ir)
    assert BoxDoubleAnalysis.runs == 1

    manager.run(ir)
    assert BoxDoubleAnalysis.runs == 2


def test_pass_returning_its_input_keeps_every_result_even_when_preserving_none() -> (
    None
):
    """Test that a node never loses its own cached results within a run.

    The transfer is merge-only, so a pass that returns its input keeps the
    input's results whatever it preserves.
    """
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(
        _build_reading_pass(
            "tests.pm.cache.none.seed", BoxDoubleAnalysis, BoxParityAnalysis
        )
    )
    manager.add_pass(
        _build_returning_pass("tests.pm.cache.none.keep", PreservedAnalyses.none())
    )
    manager.add_pass(
        _build_reading_pass(
            "tests.pm.cache.none.reread", BoxDoubleAnalysis, BoxParityAnalysis
        )
    )

    manager.run(Box(3))

    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 1


def test_pass_returning_its_input_and_preserving_all_keeps_every_result() -> None:
    """Test that a pass returning its input and preserving all keeps the results."""
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(_build_reading_pass("tests.pm.cache.all.seed", BoxDoubleAnalysis))
    manager.add_pass(
        _build_returning_pass("tests.pm.cache.all.keep", PreservedAnalyses.all())
    )
    manager.add_pass(
        _build_reading_pass("tests.pm.cache.all.reread", BoxDoubleAnalysis)
    )

    manager.run(Box(3))

    assert BoxDoubleAnalysis.runs == 1


def test_changing_pass_carries_only_its_preserved_results() -> None:
    """Test that a pass that changes the IR carries only the preserved results."""
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(
        _build_reading_pass(
            "tests.pm.cache.some.seed", BoxDoubleAnalysis, BoxParityAnalysis
        )
    )
    manager.add_pass(
        _build_replacing_pass(
            "tests.pm.cache.some.replace",
            PreservedAnalyses.none().preserve(BoxDoubleAnalysis.get_analysis_name()),
        )
    )
    manager.add_pass(
        _build_reading_pass(
            "tests.pm.cache.some.reread", BoxDoubleAnalysis, BoxParityAnalysis
        )
    )

    manager.run(Box(3))

    # Double was preserved -> still 1 run total.
    # Parity was not -> recomputed for the new box.
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 2


def test_changing_pass_preserving_all_carries_every_result() -> None:
    """Test that the results move to a changed output that preserves all."""
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(_build_reading_pass("tests.pm.cache.move.seed", BoxDoubleAnalysis))
    manager.add_pass(
        _build_replacing_pass("tests.pm.cache.move.replace", PreservedAnalyses.all())
    )
    manager.add_pass(
        _build_reading_pass("tests.pm.cache.move.reread", BoxDoubleAnalysis)
    )

    manager.run(Box(3))

    # The cached entry transferred; no recomputation despite the different IR.
    assert BoxDoubleAnalysis.runs == 1


def test_changing_pass_preserving_none_carries_no_result() -> None:
    """Test that a changed output that preserves nothing is analyzed afresh."""
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    manager.add_pass(_build_reading_pass("tests.pm.cache.drop.seed", BoxDoubleAnalysis))
    manager.add_pass(
        _build_replacing_pass("tests.pm.cache.drop.replace", PreservedAnalyses.none())
    )
    manager.add_pass(
        _build_reading_pass("tests.pm.cache.drop.reread", BoxDoubleAnalysis)
    )

    manager.run(Box(3))

    assert BoxDoubleAnalysis.runs == 2


# ---------------------------------------------------------------------------
# The analysis view of a hook (a hook's context carries the run's cache).
# ---------------------------------------------------------------------------


def test_get_analysis_manager_raises_in_a_hook_without_a_context() -> None:
    """Test that `did_change` cannot reach the run's analyses.

    The core runs `did_change` and `get_preserved_analyses` without a
    context, so the view is refused there, and the hook fails.
    """

    @register_pass("tests.pm.bind_rejects_none", "Reads the view in did_change.")
    class _ViewInDidChangePass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

        @override
        def did_change(self, input_ir: Box, output: Box) -> bool:
            self.get_analysis_manager()
            return False

    with pytest.raises(PassExecutionError) as excinfo:
        _ViewInDidChangePass().execute(Box(0))

    assert excinfo.value.hook == "did_change"
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "did_change" in str(excinfo.value.__cause__)


def test_retained_analysis_manager_raises_after_its_hook() -> None:
    """Test that a view kept past its hook raises instead of reaching the cache."""
    retained: list[AnalysisManager[Box]] = []

    @register_pass("tests.pm.unbind", "Keeps the view of its hook.")
    class _RetainingPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            analysis_manager = self.get_analysis_manager()
            assert analysis_manager is not None
            retained.append(analysis_manager)
            return ir

    manager = PassManager[Box]()
    manager.add_pass(_RetainingPass())
    manager.run(Box(1))

    with pytest.raises(RuntimeError, match="expired"):
        retained[0].get(BoxDoubleAnalysis, Box(1))


def test_get_analysis_manager_is_none_outside_a_run() -> None:
    """Test that a pass outside a run has no analysis view, before and after runs."""

    @register_pass("tests.pm.unbind_idempotent", "Identity pass for the view.")
    class _ViewlessPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

    compiler_pass = _ViewlessPass()
    assert compiler_pass.get_analysis_manager() is None

    compiler_pass.execute(Box(0))

    assert compiler_pass.get_analysis_manager() is None


def test_concurrent_runs_of_one_pipeline_are_independent() -> None:
    """Test concurrent runs do not crash or produce inconsistent results.

    Each run builds its own pipeline and cache, so runs of one manager
    from many threads see only their own results.
    """
    irs = [Box(i) for i in range(64)]

    @register_pass("tests.pm.concurrent_reader", "Reads an analysis many times.")
    class _ConcurrentReaderPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            for _ in range(20):
                assert self.get_analysis(BoxDoubleAnalysis, ir) == ir.value * 2
            return Box(self.get_analysis(BoxDoubleAnalysis, ir))

    manager = PassManager[Box]()
    manager.add_pass(_ConcurrentReaderPass())

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(ir: Box) -> int:
        try:
            value = manager.run(ir).output.value
            assert value == ir.value * 2
            return value
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)
            raise

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(worker, irs * 4))

    assert errors == []


@pytest.mark.slow
def test_concurrent_runs_survive_garbage_collection_of_their_ir() -> None:
    """Test concurrent runs and collection of their IR do not corrupt state."""

    @register_pass("tests.pm.concurrent_gc_reader", "Reads an analysis of new IR.")
    class _GcReaderPass(CompilerPass[Box, Box]):
        @override
        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return Box(ir.value + 1)

    manager = PassManager[Box]()
    manager.add_pass(_GcReaderPass())

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(seed: int) -> None:
        try:
            for offset in range(50):
                ir = Box(seed * 100 + offset)
                manager.run(ir)
                # Drop the local reference; let the IR be collected.
                del ir
            gc.collect()
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)
            raise

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(32)))

    gc.collect()
    assert errors == []


def test_run_emits_pipeline_lifecycle_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that PassManager.run emits INFO start/end + DEBUG per-pass boundary."""

    @register_pass("tests.pm.logging.identity", "Identity pass for logging.")
    class IdentityPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return ir

    manager = PassManager[int](name=Identifier("logging-pipeline"))
    manager.add_pass(IdentityPass())

    with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
        manager.run(7)

    start_messages = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and "starting" in record.getMessage()
    ]
    end_messages = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and "finished" in record.getMessage()
    ]
    pass_boundary = [
        record
        for record in caplog.records
        if record.levelno == logging.DEBUG
        and record.name == _MANAGER_LOGGER
        and "pass tests.pm.logging.identity finished" in record.getMessage()
    ]
    assert start_messages, "expected pipeline-start INFO record"
    assert end_messages, "expected pipeline-end INFO record"
    assert pass_boundary, "expected per-pass DEBUG boundary record"


def test_fixpoint_non_convergence_logs_error_before_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test ERROR log is emitted before raising on non-convergence."""

    @register_pass("tests.pm.logging.flip", "Flip bit forever for logging.")
    class FlipBitLoggingPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return 1 - ir

    manager = PassManager[int]()
    group = FixpointPassGroup[int](
        name=Identifier("logging-flip-group"),
        max_iterations=2,
        fail_on_non_convergence=True,
    )
    group.add_pass(FlipBitLoggingPass())
    manager.add_fixpoint_group(group)

    with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
        with pytest.raises(PassExecutionError):
            manager.run(0)

    error_records = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR and "did not converge" in record.getMessage()
    ]
    assert error_records, "expected ERROR record before PassExecutionError raise"


def test_fixpoint_convergence_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test INFO log on fixpoint convergence."""

    @register_pass("tests.pm.logging.decrement", "Decrement-to-zero for logging.")
    class DecrementLoggingPass(CompilerPass[int, int]):
        @override
        def get_noop_output(self, ir: int) -> int:
            return ir

        @override
        def run_pass(self, ir: int) -> int:
            return max(ir - 1, 0)

    manager = PassManager[int]()
    group = FixpointPassGroup[int](
        name=Identifier("logging-dec-group"), max_iterations=10
    )
    group.add_pass(DecrementLoggingPass())
    manager.add_fixpoint_group(group)

    with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
        manager.run(2)

    finished_records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO
        and "logging-dec-group finished" in record.getMessage()
        and "converged=True" in record.getMessage()
    ]
    assert finished_records, "expected INFO record on fixpoint convergence"


def test_analysis_cache_logs_no_hit_or_miss_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the run's cache logs no hit or miss lines.

    The cache is the Rust core's, which does not log; the pipeline and
    pass lines are still logged.
    """

    class HitMissAnalysis(Analysis[Box, int]):
        @override
        def run(self, ir: Box) -> int:
            return ir.value

    manager = PassManager[Box]()
    manager.add_pass(
        _build_reading_pass(
            "tests.pm.logging.hit_miss", HitMissAnalysis, HitMissAnalysis
        )
    )

    with caplog.at_level(logging.DEBUG):
        manager.run(Box(value=11))

    messages = [record.getMessage() for record in caplog.records]
    assert not any("cache hit" in message for message in messages)
    assert not any("cache miss" in message for message in messages)
    assert any("pass tests.pm.logging.hit_miss finished" in m for m in messages)
