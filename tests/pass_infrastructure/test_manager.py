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


def test_analysis_manager_does_not_block_ir_from_garbage_collection() -> None:
    """Test that caching an analysis result does not pin the IR in memory.

    Public observation of the eviction-via-weakref contract: if the
    cache held a strong reference to the IR, the weakref below would
    still resolve after `gc.collect()`. The fact that it returns None
    confirms the cache is not pinning the IR.
    """
    BoxDoubleAnalysis.runs = 0
    manager = PassManager[Box]()
    ir = Box(3)
    weak_ir = weakref.ref(ir)

    manager.analysis_manager.get(BoxDoubleAnalysis, ir)
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
        setattr(manager, "name", Identifier("other"))  # noqa: B010
    with pytest.raises(AttributeError):
        setattr(manager, "analysis_manager", manager.analysis_manager)  # noqa: B010


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


def test_bind_and_get_analysis_manager_are_public_accessors() -> None:
    """Test that bind / unbind / get_analysis_manager expose the binding."""
    BoxDoubleAnalysis.runs = 0

    @register_pass(
        "tests.pm.public_bind_accessors", "Identity pass for accessor testing."
    )
    class AccessorPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
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

    compiler_pass.unbind_analysis_manager()
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
    # The assertion pins the runtime contract that mypy already enforces
    # statically; the `type: ignore` is intentional.
    assert manager.add_pass(_NoChainA()) is None  # type: ignore[func-returns-value]


def test_pass_manager_add_fixpoint_group_returns_none() -> None:
    """Test that PassManager.add_fixpoint_group returns None (no chaining)."""
    manager = PassManager[int]()
    group = FixpointPassGroup[int](name=Identifier("no-chain-group"), max_iterations=1)

    assert manager.add_fixpoint_group(group) is None  # type: ignore[func-returns-value]


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

    assert group.add_pass(_NoChainGroupPass()) is None  # type: ignore[func-returns-value]


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
# AnalysisManager falls back to uncached execution when the underlying
# `weakref.finalize` call raises (no probe ref needed).
# ---------------------------------------------------------------------------


def test_analysis_manager_falls_back_when_weakref_finalize_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the manager runs uncached when finalizer registration fails.

    Simulates a non-weakreffable IR by patching the manager-module `weakref`
    so that `weakref.finalize` raises `TypeError`. The contract: rather than
    propagating the error, `AnalysisManager.get` falls back to uncached
    execution.
    """
    BoxDoubleAnalysis.runs = 0

    def _raise_type_error(*_: Any, **__: Any) -> None:
        raise TypeError("simulated non-weakreffable IR")

    monkeypatch.setattr(
        "fhy_core.pass_infrastructure.manager.weakref.finalize", _raise_type_error
    )

    manager = AnalysisManager[Box]()
    ir = Box(4)

    assert manager.get(BoxDoubleAnalysis, ir) == 8
    assert manager.get(BoxDoubleAnalysis, ir) == 8
    # Two calls -> two runs because the IR could not be cached.
    assert BoxDoubleAnalysis.runs == 2


# ---------------------------------------------------------------------------
# AnalysisManager public surface (clear / invalidate / transfer).
# ---------------------------------------------------------------------------


def test_analysis_manager_clear_drops_cached_entries() -> None:
    """Test that `clear(ir)` removes cached entries for that IR."""
    BoxDoubleAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    ir = Box(3)

    assert manager.get(BoxDoubleAnalysis, ir) == 6
    assert BoxDoubleAnalysis.runs == 1

    manager.clear(ir)

    assert manager.get(BoxDoubleAnalysis, ir) == 6
    assert BoxDoubleAnalysis.runs == 2


def test_analysis_manager_invalidate_with_none_drops_everything() -> None:
    """Test that `invalidate(ir, PreservedAnalyses.none())` drops every entry."""
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    ir = Box(3)

    manager.get(BoxDoubleAnalysis, ir)
    manager.get(BoxParityAnalysis, ir)
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 1

    manager.invalidate(ir, PreservedAnalyses.none())

    manager.get(BoxDoubleAnalysis, ir)
    manager.get(BoxParityAnalysis, ir)
    assert BoxDoubleAnalysis.runs == 2
    assert BoxParityAnalysis.runs == 2


def test_analysis_manager_invalidate_with_preserve_all_keeps_everything() -> None:
    """Test that `invalidate(ir, PreservedAnalyses.all())` is a no-op."""
    BoxDoubleAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    ir = Box(3)

    manager.get(BoxDoubleAnalysis, ir)
    assert BoxDoubleAnalysis.runs == 1

    manager.invalidate(ir, PreservedAnalyses.all())

    manager.get(BoxDoubleAnalysis, ir)
    assert BoxDoubleAnalysis.runs == 1


def test_analysis_manager_invalidate_keeps_only_preserved_entries() -> None:
    """Test that `invalidate` retains preserved entries and drops the rest."""
    BoxDoubleAnalysis.runs = 0
    BoxParityAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    ir = Box(3)

    manager.get(BoxDoubleAnalysis, ir)
    manager.get(BoxParityAnalysis, ir)

    preserved = PreservedAnalyses.none().preserve(BoxDoubleAnalysis.get_analysis_name())
    manager.invalidate(ir, preserved)

    manager.get(BoxDoubleAnalysis, ir)
    manager.get(BoxParityAnalysis, ir)
    # Double was preserved -> still 1 run total.
    # Parity was dropped -> recomputed on the second `get`.
    assert BoxDoubleAnalysis.runs == 1
    assert BoxParityAnalysis.runs == 2


def test_analysis_manager_transfer_moves_preserved_entries() -> None:
    """Test that `transfer` moves preserved entries from one IR to another."""
    BoxDoubleAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    from_ir = Box(3)
    to_ir = Box(4)

    manager.get(BoxDoubleAnalysis, from_ir)
    assert BoxDoubleAnalysis.runs == 1

    manager.transfer(from_ir, to_ir, PreservedAnalyses.all())

    manager.get(BoxDoubleAnalysis, to_ir)
    # The cached entry transferred; no recomputation despite the different IR.
    assert BoxDoubleAnalysis.runs == 1


def test_analysis_manager_transfer_drops_when_not_preserved() -> None:
    """Test that `transfer` with `PreservedAnalyses.none()` drops the entry."""
    BoxDoubleAnalysis.runs = 0
    manager = AnalysisManager[Box]()
    from_ir = Box(3)
    to_ir = Box(4)

    manager.get(BoxDoubleAnalysis, from_ir)
    assert BoxDoubleAnalysis.runs == 1

    manager.transfer(from_ir, to_ir, PreservedAnalyses.none())

    manager.get(BoxDoubleAnalysis, to_ir)
    assert BoxDoubleAnalysis.runs == 2


# ---------------------------------------------------------------------------
# bind / unbind split. The `None` overload is gone.
# ---------------------------------------------------------------------------


def test_bind_analysis_manager_rejects_none() -> None:
    """Test that `bind_analysis_manager(None)` is no longer accepted."""

    @register_pass("tests.pm.bind_rejects_none", "Identity pass for bind/unbind tests.")
    class _BindRejectsNonePass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

    compiler_pass = _BindRejectsNonePass()

    with pytest.raises(TypeError):
        compiler_pass.bind_analysis_manager(None)  # type: ignore[arg-type]


def test_unbind_analysis_manager_clears_binding() -> None:
    """Test that `unbind_analysis_manager()` returns the pass to standalone mode."""

    @register_pass("tests.pm.unbind", "Identity pass for unbind testing.")
    class _UnbindPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

    compiler_pass = _UnbindPass()
    manager = PassManager[Box]()
    compiler_pass.bind_analysis_manager(manager.analysis_manager)
    assert compiler_pass.get_analysis_manager() is manager.analysis_manager

    compiler_pass.unbind_analysis_manager()

    assert compiler_pass.get_analysis_manager() is None


def test_unbind_analysis_manager_is_idempotent() -> None:
    """Test that calling `unbind_analysis_manager()` when unbound is a no-op."""

    @register_pass("tests.pm.unbind_idempotent", "Identity pass for idempotent unbind.")
    class _UnbindIdemPass(CompilerPass[Box, Box]):
        @override
        def get_noop_output(self, ir: Box) -> Box:
            return ir

        @override
        def run_pass(self, ir: Box) -> Box:
            return ir

    compiler_pass = _UnbindIdemPass()
    compiler_pass.unbind_analysis_manager()  # already unbound; must not raise

    assert compiler_pass.get_analysis_manager() is None


def test_analysis_manager_get_is_safe_under_concurrent_callers() -> None:
    """Test concurrent ``get`` calls do not crash or produce inconsistent results."""
    manager: AnalysisManager[Box] = AnalysisManager()
    irs = [Box(i) for i in range(64)]

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(ir: Box) -> int:
        try:
            for _ in range(20):
                manager.get(BoxDoubleAnalysis, ir)
                manager.invalidate(ir, PreservedAnalyses.none())
                manager.get(BoxDoubleAnalysis, ir)
            value = manager.get(BoxDoubleAnalysis, ir)
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
def test_analysis_manager_survives_concurrent_get_and_gc_eviction() -> None:
    """Test concurrent ``get`` and finalizer-driven eviction do not corrupt state."""
    manager: AnalysisManager[Box] = AnalysisManager()

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(seed: int) -> None:
        try:
            for offset in range(50):
                ir = Box(seed * 100 + offset)
                manager.get(BoxDoubleAnalysis, ir)
                # Drop the local reference; let the finalizer fire on GC.
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
        and record.funcName == "_run_fixpoint_group"
        and "finished" in record.getMessage()
        and "converged=True" in record.getMessage()
    ]
    assert finished_records, "expected INFO record on fixpoint convergence"


def test_analysis_manager_logs_cache_hit_and_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test AnalysisManager.get emits DEBUG records on hit and miss."""

    class HitMissAnalysis(Analysis[Box, int]):
        @override
        def run(self, ir: Box) -> int:
            return ir.value

    am = AnalysisManager[Box]()
    box = Box(value=11)

    with caplog.at_level(logging.DEBUG, logger=_MANAGER_LOGGER):
        am.get(HitMissAnalysis, box)
        am.get(HitMissAnalysis, box)

    miss_records = [
        record for record in caplog.records if "cache miss" in record.getMessage()
    ]
    hit_records = [
        record for record in caplog.records if "cache hit" in record.getMessage()
    ]
    assert miss_records, "expected cache miss DEBUG record"
    assert hit_records, "expected cache hit DEBUG record"
