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
