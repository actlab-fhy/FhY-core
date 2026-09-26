"""Benchmarks of the pass infrastructure's hot paths.

They measure the pass API before and after it switches to the Rust core
(S6 of ``docs/design/python-switch.md``). The IR is the frozen `Box` of
``conftest.py``, and "the pipeline" is five passes that change, keep and
analyze it; the expression row runs over the expression benchmarks' deep
tree with the pattern benchmarks' four rules.

The benchmarks call only API whose meaning S6 keeps, and assert no result
that S6 changes. The calls whose spelling S6 changes sit in helpers marked
with their decisions: :func:`_warm_cache` reads a cached analysis
(D-S6-8), and :func:`_run_count` counts a pipeline's pass runs (N-S6-1).
"""

from typing import Any

import pytest

from fhy_core.identifier import Identifier
from fhy_core.pass_infrastructure import (
    CompilerPass,
    FixpointGroupRecord,
    FixpointPassGroup,
    PassExecutionError,
    PassManager,
    PassManagerResult,
    ValidationManager,
    VerificationRegistry,
    run_verification,
)
from fhy_core.symbolic.expression import Expression, RewriteRuleApplier
from fhy_core.symbolic.expression.pprint import ExpressionPrettyFormatter
from fhy_core.utils.override import override

from .conftest import (
    PIPELINE_INCREMENT,
    REPORT_DIAGNOSTIC_COUNT,
    SATURATION_VALUE,
    Benchmark,
    BoundedBoxVerifier,
    Box,
    BoxValueAnalysis,
    CopyReadingPass,
    CountingBoxValueAnalysis,
    EveryHookPass,
    FailingPass,
    IdentityPass,
    NonNegativeBoxVerifier,
    ReportingPass,
    ReportingValidator,
    SaturatingIncrementPass,
    SkippedPass,
    TwiceVerifiedBox,
    VerifiedBox,
    VerifiedNode,
    build_pipeline,
)
from .test_expression import (
    _DEEP_TREE_DEPTH,
    _DEEP_TREE_IDENTIFIER_COUNT,
    _build_deep_tree,
    _build_identifiers,
)
from .test_pattern import _build_rules

pytestmark = pytest.mark.benchmark(group="pass_infrastructure")

# How many copies of the five-pass pipeline the long pipeline chains.
_LONG_PIPELINE_REPETITIONS = 10
# How many passes the preserving pipeline chains.
_PRESERVING_PASS_COUNT = 5
# How many validators the benchmarked validation manager runs.
_VALIDATOR_COUNT = 10


class _CacheHitPass(CompilerPass[Box, Box]):
    """Pass that warms the cache with an analysis, then benchmarks reading it."""

    _benchmark: Benchmark
    result: int | None

    def __init__(self, benchmark: Benchmark) -> None:
        super().__init__()
        self._benchmark = benchmark
        self.result = None

    @override
    def get_noop_output(self, ir: Box) -> Box:
        return ir

    @override
    def run_pass(self, ir: Box) -> Box:
        self.get_analysis(BoxValueAnalysis, ir)
        self.result = self._benchmark(self.get_analysis, BoxValueAnalysis, ir)
        return ir


def _warm_cache(benchmark: Benchmark, box: Box) -> int | None:
    """Benchmark a second ``get_analysis`` of one analysis inside one pass run.

    D-S6-8: the analysis cache lives for one pipeline run and is reached
    only from inside a pass, so the cache hit is timed there; before S6 it
    was timed as ``AnalysisManager.get`` on a standalone manager.
    """
    cache_hit_pass = _CacheHitPass(benchmark)
    manager = PassManager[Box]()
    manager.add_pass(cache_hit_pass)
    manager.run(box)
    return cache_hit_pass.result


def _run_count(result: PassManagerResult[Any]) -> int:
    """Return how many pass runs `result` records, none of them skipped.

    N-S6-1: ``result.run_count()`` since S6, which replaces the global run
    counters; before S6, the number of pass records, which is the run count
    when no pass skips.
    """
    return result.run_count()


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


def test_compiler_pass_execute(benchmark: Benchmark, box: Box) -> None:
    """Benchmark executing a pass that returns its input unchanged."""
    result = benchmark(IdentityPass().execute, box)
    assert not result.changed


def test_compiler_pass_call(benchmark: Benchmark, box: Box) -> None:
    """Benchmark calling a pass that returns its input unchanged."""
    assert benchmark(IdentityPass(), box) is box


def test_compiler_pass_execute_with_every_hook_overridden(
    benchmark: Benchmark, box: Box
) -> None:
    """Benchmark executing an identity pass whose seven hooks are all Python."""
    result = benchmark(EveryHookPass().execute, box)
    assert not result.changed


def test_compiler_pass_execute_skipped(benchmark: Benchmark, box: Box) -> None:
    """Benchmark executing a pass whose ``should_run`` is false."""
    result = benchmark(SkippedPass().execute, box)
    assert result.output is box
    assert not result.changed


def test_compiler_pass_execute_failing(benchmark: Benchmark, box: Box) -> None:
    """Benchmark executing a pass whose run raises, up to the caught error."""
    failing_pass = FailingPass()

    def execute_failing() -> PassExecutionError:
        try:
            failing_pass.execute(box)
        except PassExecutionError as error:
            return error
        raise AssertionError("the pass run did not fail")

    assert isinstance(benchmark(execute_failing).__cause__, ValueError)


def test_compiler_pass_report_of_100_diagnostics(
    benchmark: Benchmark, box: Box
) -> None:
    """Benchmark executing a pass that reports 100 diagnostics."""
    result = benchmark(ReportingPass(REPORT_DIAGNOSTIC_COUNT).execute, box)
    assert len(result.diagnostics) == REPORT_DIAGNOSTIC_COUNT


def test_compiler_pass_create(benchmark: Benchmark) -> None:
    """Benchmark building a registered pass by its name."""
    assert isinstance(
        benchmark(CompilerPass.create, "benchmarks.identity"), IdentityPass
    )


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_pass_manager_run_of_5_passes(
    benchmark: Benchmark, pass_manager: PassManager[Box], box: Box
) -> None:
    """Benchmark running a five-pass pipeline over a small IR."""
    result = benchmark(pass_manager.run, box)
    assert result.output.value == box.value + PIPELINE_INCREMENT


def test_pass_manager_run_of_50_passes(benchmark: Benchmark, box: Box) -> None:
    """Benchmark running ten copies of the five-pass pipeline."""
    manager = build_pipeline(_LONG_PIPELINE_REPETITIONS)
    result = benchmark(manager.run, box)
    assert (
        result.output.value
        == box.value + _LONG_PIPELINE_REPETITIONS * PIPELINE_INCREMENT
    )
    assert _run_count(result) == 5 * _LONG_PIPELINE_REPETITIONS


def test_pass_manager_fixpoint_group_of_10_iterations(
    benchmark: Benchmark, box: Box
) -> None:
    """Benchmark a fixpoint group that converges on its tenth iteration."""
    group = FixpointPassGroup[Box](Identifier("saturate"))
    group.add_pass(SaturatingIncrementPass())
    manager = PassManager[Box]()
    manager.add_fixpoint_group(group)
    result = benchmark(manager.run, box)
    (record,) = result.records
    assert isinstance(record, FixpointGroupRecord)
    assert record.converged
    assert record.iterations == SATURATION_VALUE + 1
    assert _run_count(result) == SATURATION_VALUE + 1


def test_pass_manager_run_with_verification(benchmark: Benchmark) -> None:
    """Benchmark the five-pass pipeline over a box with a verification pass."""
    manager = build_pipeline(1)
    box = VerifiedBox(0)
    result = benchmark(manager.run, box)
    assert result.output == VerifiedBox(box.value + PIPELINE_INCREMENT)


def test_analysis_manager_cache_hit(benchmark: Benchmark, box: Box) -> None:
    """Benchmark reading an analysis the pipeline has already cached."""
    assert _warm_cache(benchmark, box) == box.value


def test_analysis_preserved_across_5_passes(benchmark: Benchmark, box: Box) -> None:
    """Benchmark one analysis computed once and read by five preserving passes."""
    manager = PassManager[Box]()
    for _ in range(_PRESERVING_PASS_COUNT):
        manager.add_pass(CopyReadingPass())
    CountingBoxValueAnalysis.runs = 0
    manager.run(box)
    assert CountingBoxValueAnalysis.runs == 1
    result = benchmark(manager.run, box)
    assert result.output == box


# ---------------------------------------------------------------------------
# Validation and verification
# ---------------------------------------------------------------------------


def test_validation_manager_validate_of_10_validators(
    benchmark: Benchmark, box: Box
) -> None:
    """Benchmark ten validators, every other one reporting a warning."""
    manager = ValidationManager[Box]()
    for index in range(_VALIDATOR_COUNT):
        manager.add(ReportingValidator(is_reporting=index % 2 == 0))
    report = benchmark(manager.validate, box)
    assert len(report.diagnostics) == _VALIDATOR_COUNT // 2
    assert len(report.records) == _VALIDATOR_COUNT


def test_run_verification(benchmark: Benchmark) -> None:
    """Benchmark verifying a box with two registered verification passes."""
    report = benchmark(run_verification, TwiceVerifiedBox(0))
    assert not report.has_errors()
    assert len(report.records) == 2  # noqa: PLR2004


def test_verification_registry_get_passes_for(benchmark: Benchmark) -> None:
    """Benchmark the lookup of two verification passes over a two-level MRO."""
    passes = benchmark(VerificationRegistry.get_passes_for, TwiceVerifiedBox)
    assert passes == (NonNegativeBoxVerifier, BoundedBoxVerifier)


def test_verification_registry_register_again(benchmark: Benchmark) -> None:
    """Benchmark re-registering a registered pair, which changes nothing."""
    benchmark(VerificationRegistry.register, VerifiedBox, NonNegativeBoxVerifier)
    assert VerificationRegistry.get_passes_for(VerifiedBox) == (NonNegativeBoxVerifier,)


def test_verifiable_mixin_verify(benchmark: Benchmark) -> None:
    """Benchmark the default `verify` of a node with one registered pass."""
    report = benchmark(VerifiedNode(0).verify)
    assert not report.has_errors()
    assert len(report.records) == 1


# ---------------------------------------------------------------------------
# A mixed pipeline over expressions
# ---------------------------------------------------------------------------


class _ExpressionIdentityPass(CompilerPass[Expression, Expression]):
    """Python pass that returns its expression unchanged."""

    @override
    def get_noop_output(self, ir: Expression) -> Expression:
        return ir

    @override
    def run_pass(self, ir: Expression) -> Expression:
        return ir


def test_mixed_pipeline_over_a_deep_expression(benchmark: Benchmark) -> None:
    """Benchmark the rule applier, a Python pass and the formatter on a deep tree."""
    tree = _build_deep_tree(
        _build_identifiers(_DEEP_TREE_IDENTIFIER_COUNT, "v"), _DEEP_TREE_DEPTH
    )
    manager = PassManager[Any]()
    manager.add_pass(RewriteRuleApplier(_build_rules()))
    manager.add_pass(_ExpressionIdentityPass())
    manager.add_pass(ExpressionPrettyFormatter())
    result = benchmark(manager.run, tree)
    assert isinstance(result.output, str)
