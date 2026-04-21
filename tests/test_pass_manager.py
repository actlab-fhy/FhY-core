"""Tests the pass manager infrastructure."""

import gc
from dataclasses import dataclass

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    Analysis,
    CompilerPass,
    FixpointGroupRecord,
    FixpointPassGroup,
    PassExecutionError,
    PassManager,
    PreservedAnalyses,
    register_pass,
)
from fhy_core.trait import FrozenMixin, PartialEqual


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

    def run(self, ir: Box) -> int:
        type(self).runs += 1
        return ir.value * 2


class BoxParityAnalysis(Analysis[Box, int]):
    """Analysis that computes parity and tracks run count."""

    runs = 0

    def run(self, ir: Box) -> int:
        type(self).runs += 1
        return ir.value % 2


class MutableBoxDoubleAnalysis(Analysis[MutableBox, int]):
    """Analysis for mutable IR cache behavior tests."""

    runs = 0

    def run(self, ir: MutableBox) -> int:
        type(self).runs += 1
        return ir.value * 2


def test_pass_manager_runs_passes_in_order() -> None:
    """Test that pass manager runs passes in insertion order."""

    @register_pass("tests.pm.add_one", "Add one to the Box value.")
    class AddOnePass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

    @register_pass("tests.pm.double", "Double the Box value.")
    class DoublePass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value * 2)

    manager = PassManager[Box]()
    manager.add_pass(AddOnePass()).add_pass(DoublePass())

    result = manager.run(Box(3))

    assert result.output == Box(8)
    assert len(result.records) == 2


def test_pass_manager_applies_analysis_preservation_and_invalidation() -> None:
    """Test that analysis cache transfer respects preserved analyses."""
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0

    @register_pass(
        "tests.pm.preserve_double_only",
        "Change IR while preserving only the double analysis.",
    )
    class PreserveDoubleOnlyPass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

        def get_preserved_analyses(
            self, input_ir: Box, output: Box, *, changed: bool
        ) -> PreservedAnalyses:
            _ = (input_ir, output, changed)
            return PreservedAnalyses.none().preserve(
                BoxDoubleAnalysis.get_analysis_name()
            )

    manager = PassManager[Box]()
    manager.add_pass(PreserveDoubleOnlyPass())
    input_ir = Box(2)

    assert manager.analysis_manager.get(BoxDoubleAnalysis, input_ir) == 4
    assert manager.analysis_manager.get(BoxParityAnalysis, input_ir) == 0
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 1

    result = manager.run(input_ir)

    assert manager.analysis_manager.get(BoxDoubleAnalysis, result.output) == 4
    assert manager.analysis_manager.get(BoxParityAnalysis, result.output) == 1
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 2


def test_analysis_manager_does_not_cache_non_frozen_ir() -> None:
    """Test that analysis manager skips caching for non-frozen IR."""
    MutableBoxDoubleAnalysis.runs = 0
    manager = PassManager[MutableBox]()
    ir = MutableBox(3)

    assert manager.analysis_manager.get(MutableBoxDoubleAnalysis, ir) == 6
    assert manager.analysis_manager.get(MutableBoxDoubleAnalysis, ir) == 6
    assert MutableBoxDoubleAnalysis.runs == 2


def test_analysis_manager_evicts_cache_after_ir_collection() -> None:
    """Test that cached analysis entries are evicted after IR collection."""
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    ir = Box(3)

    assert manager.analysis_manager.get(BoxDoubleAnalysis, ir) == 6
    assert BoxDoubleAnalysis.runs == 1

    ir_id = id(ir)
    assert ir_id in manager.analysis_manager._cache
    del ir
    gc.collect()

    assert ir_id not in manager.analysis_manager._cache
    assert ir_id not in manager.analysis_manager._finalizers


def test_analysis_identifier_is_unique() -> None:
    """Test analyses expose unique identifier keys."""
    assert (
        BoxDoubleAnalysis.get_analysis_name() != BoxParityAnalysis.get_analysis_name()
    )


def test_pass_manager_fixpoint_group_converges() -> None:
    """Test that a fixpoint group converges and records iterations."""

    @register_pass("tests.pm.decrement_to_zero", "Decrement value toward zero.")
    class DecrementToZeroPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

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
        def get_noop_output(self, ir: int) -> int:
            return ir

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

    with pytest.raises(PassExecutionError):
        manager.run(0)


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
        setattr(manager, "name", Identifier("other"))
    with pytest.raises(AttributeError):
        setattr(manager, "analysis_manager", manager.analysis_manager)


def test_get_analysis_runs_uncached_when_pass_is_standalone() -> None:
    """Test that get_analysis computes fresh each call when no manager is bound."""
    BoxDoubleAnalysis.runs = 0

    @register_pass("tests.pm.standalone_get_analysis", "Reads analysis standalone.")
    class ReadAnalysisPass(CompilerPass[Box, Box]):
        observed: list[int] = []

        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            ReadAnalysisPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            ReadAnalysisPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    pass_ = ReadAnalysisPass()
    pass_.execute(Box(5))

    # Two calls → two runs (no cache available when standalone).
    assert ReadAnalysisPass.observed == [10, 10]
    assert BoxDoubleAnalysis.runs == 2


def test_get_analysis_uses_cache_when_bound_by_pass_manager() -> None:
    """Test that get_analysis hits the manager's cache for duplicate requests."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.cached_get_analysis", "Reads analysis twice under a manager."
    )
    class TwiceReadPass(CompilerPass[Box, Box]):
        observed: list[int] = []

        def get_noop_output(self, ir: Box) -> Box:
            return ir

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
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir  # identity — preserves all by default (no change)

    @register_pass("tests.pm.read_analysis_again", "Reads the analysis a second time.")
    class ReadAgainPass(CompilerPass[Box, Box]):
        observed: list[int] = []

        def get_noop_output(self, ir: Box) -> Box:
            return ir

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
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            self.get_analysis(BoxDoubleAnalysis, ir)
            return ir

    @register_pass(
        "tests.pm.mutate_without_preserve",
        "Changes the IR and preserves no analyses (default).",
    )
    class MutateNoPreservePass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            return Box(ir.value + 1)

    @register_pass(
        "tests.pm.reread_after_mutation", "Re-reads the analysis after mutation."
    )
    class RereadPass(CompilerPass[Box, Box]):
        observed: list[int] = []

        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            RereadPass.observed.append(self.get_analysis(BoxDoubleAnalysis, ir))
            return ir

    manager = PassManager[Box]()
    manager.add_pass(SeedAnalysisPass())
    manager.add_pass(MutateNoPreservePass())
    manager.add_pass(RereadPass())
    manager.run(Box(5))

    # First run computed on Box(5) → 10. Mutation invalidated it.
    # Second run computed on Box(6) → 12.
    assert RereadPass.observed == [12]
    assert BoxDoubleAnalysis.runs == 2


def test_bind_and_get_analysis_manager_are_public_accessors() -> None:
    """Test that bind_analysis_manager / get_analysis_manager expose the binding."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.public_bind_accessors", "Identity pass for accessor testing."
    )
    class AccessorPass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        def run_pass(self, ir: Box) -> Box:
            return ir

    compiler_pass = AccessorPass()
    assert compiler_pass.get_analysis_manager() is None

    manager = PassManager[Box]()
    compiler_pass.bind_analysis_manager(manager.analysis_manager)
    assert compiler_pass.get_analysis_manager() is manager.analysis_manager

    # The pass now sees the manager's cache on get_analysis calls for the
    # same IR instance (cache is keyed on object identity).
    ir = Box(3)
    compiler_pass.get_analysis(BoxDoubleAnalysis, ir)
    compiler_pass.get_analysis(BoxDoubleAnalysis, ir)
    assert BoxDoubleAnalysis.runs == 1

    compiler_pass.bind_analysis_manager(None)
    assert compiler_pass.get_analysis_manager() is None


def test_get_analysis_works_inside_fixpoint_group() -> None:
    """Test that get_analysis is available to passes within a fixpoint group."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.fixpoint_reader",
        "Reads analysis and decrements until fixed-point.",
    )
    class FixpointReaderPass(CompilerPass[Box, Box]):
        observed: list[int] = []

        def get_noop_output(self, ir: Box) -> Box:
            return ir

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

    # Iteration 1: Box(2) → observes 4 → returns Box(1)
    # Iteration 2: Box(1) → observes 2 → returns Box(0)
    # Iteration 3: Box(0) → observes 0 → returns Box(0); converged.
    assert FixpointReaderPass.observed == [4, 2, 0]


def test_get_analysis_restores_pass_state_between_runs() -> None:
    """Test that a manager-bound pass has no dangling analysis manager after the run."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.state_restoration", "Identity pass that reads an analysis."
    )
    class StateCheckPass(CompilerPass[Box, Box]):
        def get_noop_output(self, ir: Box) -> Box:
            return ir

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
    # Standalone re-runs (no cache) → total runs == 1 (managed) + 2 (standalone) = 3.
    assert BoxDoubleAnalysis.runs == 3


def test_pass_manager_records_support_partial_equal_traits() -> None:
    """Test pass-manager records satisfy `PartialEqual` protocol."""

    @register_pass("tests.pm.partial_equal_record", "Identity pass for records.")
    class IdentityPass(CompilerPass[int, int]):
        def get_noop_output(self, ir: int) -> int:
            return ir

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
