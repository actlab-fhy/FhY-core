"""Benchmarks of the pass infrastructure's hot paths."""

import pytest

from fhy_core.pass_infrastructure import AnalysisManager, PassManager

from .conftest import Benchmark, Box, BoxValueAnalysis, IdentityPass

pytestmark = pytest.mark.benchmark(group="pass_infrastructure")

# The value the benchmarked pipeline's two increment passes leave in a box.
_PIPELINE_INCREMENT = 2


def test_compiler_pass_execute(benchmark: Benchmark, box: Box) -> None:
    """Benchmark executing a pass that returns its input unchanged."""
    result = benchmark(IdentityPass().execute, box)
    assert not result.changed


def test_pass_manager_run_of_5_passes(
    benchmark: Benchmark, pass_manager: PassManager[Box], box: Box
) -> None:
    """Benchmark running a five-pass pipeline over a small IR."""
    result = benchmark(pass_manager.run, box)
    assert result.output.value == box.value + _PIPELINE_INCREMENT


def test_analysis_manager_cache_hit(
    benchmark: Benchmark, warm_analysis_manager: AnalysisManager[Box], box: Box
) -> None:
    """Benchmark getting an analysis the manager has already cached."""
    assert benchmark(warm_analysis_manager.get, BoxValueAnalysis, box) == box.value
