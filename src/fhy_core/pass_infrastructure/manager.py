"""Pass manager and analysis manager infrastructure."""

from __future__ import annotations

__all__ = [
    "Analysis",
    "AnalysisManager",
    "FixpointGroupRecord",
    "FixpointIterationRecord",
    "FixpointPassGroup",
    "PassManager",
    "PassManagerResult",
    "PassRunRecord",
]

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, cast

from fhy_core.identifier import Identifier
from fhy_core.trait import Frozen, FrozenMixin, HasIdentifierMixin

from .core import CompilerPass, PassExecutionError, PassResult, PreservedAnalyses

_IRType = TypeVar("_IRType")
_AnalysisResultT = TypeVar("_AnalysisResultT")


class Analysis(ABC, Generic[_IRType, _AnalysisResultT]):
    """Base class for reusable analyses cached by the pass manager."""

    _analysis_name: ClassVar[Identifier | None] = None

    @classmethod
    def get_analysis_name(cls) -> Identifier:
        """Return the unique identifier for this analysis type."""
        if "_analysis_name" not in cls.__dict__ or cls._analysis_name is None:
            cls._analysis_name = Identifier(f"{cls.__module__}.{cls.__qualname__}")
        return cls._analysis_name

    @abstractmethod
    def run(self, ir: _IRType) -> _AnalysisResultT:
        """Compute analysis results for IR.

        Args:
            ir: The IR to analyze.

        Returns:
            The analysis result for IR.

        """


class AnalysisManager(Generic[_IRType]):
    """Caches analysis results and applies preservation/invalidation rules."""

    _cache: dict[int, dict[Identifier, Any]]
    _finalizers: dict[int, Any]

    def __init__(self) -> None:
        self._cache = {}
        self._finalizers = {}

    def get(
        self, analysis_type: type[Analysis[_IRType, _AnalysisResultT]], ir: _IRType
    ) -> _AnalysisResultT:
        """Get analysis results for IR, computing and caching when necessary.

        Args:
            analysis_type: The type of analysis to retrieve.
            ir: The IR to analyze.

        Returns:
            The analysis result for IR.

        """
        if not self._is_cacheable_ir(ir):
            self._drop_cached_ir(id(ir))
            return analysis_type().run(ir)

        ir_id = id(ir)
        analysis_name = analysis_type.get_analysis_name()
        bucket = self._cache.get(ir_id)
        if bucket is None:
            if not self._register_finalizer(ir, ir_id):
                return analysis_type().run(ir)
            bucket = {}
            self._cache[ir_id] = bucket
        if analysis_name in bucket:
            return cast(_AnalysisResultT, bucket[analysis_name])
        result = analysis_type().run(ir)
        bucket[analysis_name] = result
        return result

    def clear(self, ir: _IRType) -> None:
        """Clear all cached analyses for IR.

        Args:
            ir: The IR to clear analyses for.

        """
        self._drop_cached_ir(id(ir))

    def invalidate(self, ir: _IRType, preserved: PreservedAnalyses) -> None:
        """Invalidate non-preserved analyses for IR.

        Args:
            ir: The IR to invalidate analyses for.
            preserved: The analyses to preserve.

        """
        if not self._is_cacheable_ir(ir):
            self._drop_cached_ir(id(ir))
            return

        ir_id = id(ir)
        bucket = self._cache.get(ir_id)
        if bucket is None:
            return
        if preserved.preserves_all:
            return
        analyses_to_drop = [
            analysis_name
            for analysis_name in bucket
            if not preserved.is_preserved(analysis_name)
        ]
        for analysis_name in analyses_to_drop:
            del bucket[analysis_name]
        if not bucket:
            self._drop_cached_ir(ir_id)

    def transfer(
        self,
        from_ir: _IRType,
        to_ir: _IRType,
        preserved: PreservedAnalyses,
    ) -> None:
        """Transfer preserved analysis results across an IR replacement.

        Args:
            from_ir: The original IR being replaced.
            to_ir: The new IR replacing from_ir.
            preserved: The analyses to preserve across the replacement.

        """
        from_cacheable = self._is_cacheable_ir(from_ir)
        to_cacheable = self._is_cacheable_ir(to_ir)
        if not from_cacheable and not to_cacheable:
            return
        if not from_cacheable:
            self._drop_cached_ir(id(to_ir))
            return

        from_id = id(from_ir)
        to_id = id(to_ir)
        if from_id == to_id:
            self.invalidate(from_ir, preserved)
            return

        from_bucket = self._cache.pop(from_id, {})
        self._drop_finalizer(from_id)
        self._drop_cached_ir(to_id)
        if not from_bucket:
            return
        if not to_cacheable:
            return

        if preserved.preserves_all:
            if not self._register_finalizer(to_ir, to_id):
                return
            self._cache[to_id] = dict(from_bucket)
            return

        kept = {
            analysis_name: result
            for analysis_name, result in from_bucket.items()
            if preserved.is_preserved(analysis_name)
        }
        if kept:
            if not self._register_finalizer(to_ir, to_id):
                return
            self._cache[to_id] = kept

    @staticmethod
    def _is_cacheable_ir(ir: object) -> bool:
        return isinstance(ir, Frozen) and ir.is_frozen

    @staticmethod
    def _evict_cached_ir(
        manager_ref: "weakref.ReferenceType[AnalysisManager[Any]]", ir_id: int
    ) -> None:
        manager = manager_ref()
        if manager is None:
            return
        manager._cache.pop(ir_id, None)
        manager._finalizers.pop(ir_id, None)

    def _register_finalizer(self, ir: object, ir_id: int) -> bool:
        if ir_id in self._finalizers:
            return True
        try:
            weakref.ref(ir)
        except TypeError:
            return False
        self._finalizers[ir_id] = weakref.finalize(
            ir, AnalysisManager._evict_cached_ir, weakref.ref(self), ir_id
        )
        return True

    def _drop_finalizer(self, ir_id: int) -> None:
        finalizer = self._finalizers.pop(ir_id, None)
        if finalizer is not None and finalizer.alive:
            finalizer.detach()

    def _drop_cached_ir(self, ir_id: int) -> None:
        self._cache.pop(ir_id, None)
        self._drop_finalizer(ir_id)


@dataclass(frozen=True)
class PassRunRecord(Frozen):
    """Execution record for one pass run."""

    pass_name: str
    changed: bool
    diagnostics: tuple[Any, ...]
    preserved_analyses: tuple[Identifier, ...]
    preserves_all_analyses: bool


@dataclass(frozen=True)
class FixpointIterationRecord(FrozenMixin):
    """Execution record for one fixpoint iteration."""

    iteration: int
    changed: bool
    pass_runs: tuple[PassRunRecord, ...]


@dataclass(frozen=True)
class FixpointGroupRecord(FrozenMixin):
    """Execution record for a fixpoint group."""

    group_name: Identifier
    iterations: int
    converged: bool
    iteration_records: tuple[FixpointIterationRecord, ...]


@dataclass(frozen=True)
class PassManagerResult(FrozenMixin, Generic[_IRType]):
    """Overall pass manager execution result."""

    output: _IRType
    records: tuple[PassRunRecord | FixpointGroupRecord, ...]


class FixpointPassGroup(HasIdentifierMixin, Generic[_IRType]):
    """A repeatedly executed pass sequence until fixpoint or iteration budget."""

    _passes: list[CompilerPass[_IRType, _IRType]]
    _identifier: Identifier
    _max_iterations: int
    _fail_on_non_convergence: bool

    def __init__(
        self,
        name: Identifier,
        *,
        max_iterations: int = 10,
        fail_on_non_convergence: bool = True,
    ) -> None:
        if max_iterations < 1:
            raise ValueError('"max_iterations" must be >= 1.')
        self._identifier = name
        self._max_iterations = max_iterations
        self._fail_on_non_convergence = fail_on_non_convergence
        self._passes = []

    @property
    def identifier(self) -> Identifier:
        return self._identifier

    @property
    def name(self) -> Identifier:
        return self._identifier

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def fail_on_non_convergence(self) -> bool:
        return self._fail_on_non_convergence

    @property
    def passes(self) -> tuple[CompilerPass[_IRType, _IRType], ...]:
        return tuple(self._passes)

    def add_pass(
        self, compiler_pass: CompilerPass[_IRType, _IRType]
    ) -> "FixpointPassGroup[_IRType]":
        """Append a pass to the fixpoint group.

        Args:
            compiler_pass: The pass to add.

        Returns:
            This fixpoint group, for chaining.

        """
        self._passes.append(compiler_pass)
        return self


class PassManager(HasIdentifierMixin, Generic[_IRType]):
    """Ordered pass pipeline manager with analysis preservation/invalidation."""

    _items: list[CompilerPass[_IRType, _IRType] | FixpointPassGroup[_IRType]]
    _analysis_manager: AnalysisManager[_IRType]
    _identifier: Identifier

    def __init__(self, name: Identifier | None = None) -> None:
        self._identifier = name if name is not None else Identifier("pipeline")
        self._items = []
        self._analysis_manager = AnalysisManager()

    @property
    def identifier(self) -> Identifier:
        return self._identifier

    @property
    def name(self) -> Identifier:
        return self._identifier

    @property
    def analysis_manager(self) -> AnalysisManager[_IRType]:
        return self._analysis_manager

    def add_pass(
        self, compiler_pass: CompilerPass[_IRType, _IRType]
    ) -> "PassManager[_IRType]":
        """Append one pass to the pipeline.

        Args:
            compiler_pass: The pass to add.

        Returns:
            This pass manager, for chaining.

        """
        self._items.append(compiler_pass)
        return self

    def add_fixpoint_group(
        self,
        group: FixpointPassGroup[_IRType],
    ) -> "PassManager[_IRType]":
        """Append one fixpoint group to the pipeline.

        Args:
            group: The fixpoint group to add.

        Returns:
            This pass manager, for chaining.

        """
        self._items.append(group)
        return self

    def run(self, ir: _IRType) -> PassManagerResult[_IRType]:
        """Run the pass pipeline over the IR.

        Args:
            ir: IR to optimize.

        Returns:
            The overall pass manager result, including the final IR and execution
            records.

        Raises:
            PassExecutionError: If any pass raises an error during execution, or if
                a fixpoint group fails to converge within its iteration budget.

        """
        current = ir
        records: list[PassRunRecord | FixpointGroupRecord] = []

        for item in self._items:
            if isinstance(item, FixpointPassGroup):
                current, record = self._run_fixpoint_group(item, current)
                records.append(record)
                continue

            result = item.execute(current)
            run_record = self._make_pass_run_record(item, result)
            self.analysis_manager.transfer(
                current, result.output, result.preserved_analyses
            )
            current = result.output
            records.append(run_record)

        return PassManagerResult(output=current, records=tuple(records))

    def _run_fixpoint_group(
        self, group: FixpointPassGroup[_IRType], ir: _IRType
    ) -> tuple[_IRType, FixpointGroupRecord]:
        current = ir
        iteration_records: list[FixpointIterationRecord] = []
        converged = False

        for iteration in range(1, group.max_iterations + 1):
            changed_any = False
            pass_runs: list[PassRunRecord] = []
            for compiler_pass in group.passes:
                result = compiler_pass.execute(current)
                pass_run = self._make_pass_run_record(compiler_pass, result)
                self.analysis_manager.transfer(
                    current, result.output, result.preserved_analyses
                )
                current = result.output
                pass_runs.append(pass_run)
                changed_any = changed_any or result.changed

            iteration_records.append(
                FixpointIterationRecord(
                    iteration,
                    changed_any,
                    tuple(pass_runs),
                )
            )
            if not changed_any:
                converged = True
                break

        if not converged and group.fail_on_non_convergence:
            raise PassExecutionError(
                f'Fixpoint group "{group.name}" did not converge in '
                f"{group.max_iterations} iterations."
            )

        return current, FixpointGroupRecord(
            group.name,
            len(iteration_records),
            converged,
            tuple(iteration_records),
        )

    @staticmethod
    def _make_pass_run_record(
        compiler_pass: CompilerPass[_IRType, _IRType], result: PassResult[_IRType]
    ) -> PassRunRecord:
        preserved = result.preserved_analyses
        return PassRunRecord(
            compiler_pass.get_pass_name(),
            result.changed,
            result.diagnostics,
            tuple(
                sorted(preserved.analysis_names, key=lambda identifier: identifier.id)
            ),
            preserved.preserves_all,
        )
