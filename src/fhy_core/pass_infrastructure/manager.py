"""Pass pipelines, fixpoint groups, analyses and the records of a run.

Backed by the Rust implementation (``fhy_core._rs``), with the Rust core's
semantics:

- A `PassManager` run feeds each item the previous item's output, and
  caches analysis results per IR node for the length of the run: a pass's
  output gains the results its input had for the analyses the pass
  preserves, unless it has its own. Only a frozen ``Frozen`` IR is cached.
- A run verifies its input and every output a pass reports as changed,
  blaming the pass, with the verification passes registered for the IR's
  type, unless ``set_verifier`` sets another verifier or turns it off.
- A failure raises with the records of the work completed before it.
- `AnalysisManager` is the view of a run's analyses that
  ``CompilerPass.get_analysis_manager()`` returns during a hook; it cannot
  be constructed, and expires when the hook returns.
"""

from fhy_core.utils.override import override

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

import inspect
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from fhy_core import _rs
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.traits import FrozenMixin, PartialEqualMixin
from fhy_core.utils.self import Self

from .core import CompilerPass, PassExecutionError

if TYPE_CHECKING:
    from .validation import ValidationManager

_LOGGER = get_logger(__name__)

_IRType = TypeVar("_IRType")
_AnalysisResultT = TypeVar("_AnalysisResultT")


class Analysis(_rs.AnalysisBase, ABC, Generic[_IRType, _AnalysisResultT]):
    """Base class for reusable analyses cached by the pass manager.

    Backed by the Rust implementation: ``fhy_core._rs.AnalysisBase``. A
    pipeline run caches an analysis's result per IR node under
    `get_analysis_name`. Subclasses must support no-argument construction,
    since the cache instantiates analyses with ``analysis_type()``. Optional
    keyword arguments with defaults are fine; required positional arguments
    are rejected at class-creation time with ``TypeError``.
    """

    _analysis_name: ClassVar[Identifier | None] = None
    _analysis_name_lock: ClassVar[Lock] = Lock()

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        signature = inspect.signature(cls.__init__)
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if parameter.default is inspect.Parameter.empty and parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError(
                    f'Analysis subclass "{cls.__qualname__}" requires a no-arg '
                    f'constructor; parameter "{parameter.name}" has no default. '
                    f"The analysis cache instantiates analyses with no arguments."
                )

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_analysis_name(cls) -> Identifier:
        """Return the unique identifier for this analysis type."""
        if "_analysis_name" in cls.__dict__ and cls._analysis_name is not None:
            return cls._analysis_name

        with Analysis._analysis_name_lock:
            if "_analysis_name" not in cls.__dict__ or cls._analysis_name is None:
                analysis_name = Identifier(f"{cls.__module__}.{cls.__qualname__}")
                cls._analysis_name = analysis_name
                return analysis_name
            return cls._analysis_name

    @abstractmethod
    def run(self, ir: _IRType) -> _AnalysisResultT:
        """Compute analysis results for IR.

        Args:
            ir: The IR to analyze.

        Returns:
            The analysis result for IR.

        """


AnalysisManager = _rs.AnalysisManager
"""The analyses of one pass run, as one hook sees them.

``CompilerPass.get_analysis_manager()`` returns one during a hook of a pass;
``get(analysis_type, ir)`` returns an analysis result as the pass's
``get_analysis`` does. It has no constructor, and raises ``RuntimeError``
once its hook returned.
"""


class PassRunRecord(_rs.PassRunRecord, PartialEqualMixin):
    """Execution record for one pass run.

    Backed by the Rust implementation: ``fhy_core._rs.PassRunRecord``.
    Records are immutable, compare, hash and print as frozen dataclasses
    do, and pickle as a call of their class.

    Attributes:
        pass_name: The name of the pass that ran.
        changed: Whether the run changed the IR.
        diagnostics: The run's diagnostics, in emission order.
        preserved_analyses: The analyses the run left valid.
        skipped: Whether the pass skipped the run.

    """

    __slots__ = ()
    __match_args__ = ("pass_name", "changed", "diagnostics", "preserved_analyses")


FrozenMixin.register(PassRunRecord)
PassRunRecord._register_public_class()


class FixpointIterationRecord(_rs.FixpointIterationRecord, PartialEqualMixin):
    """Execution record for one fixpoint iteration.

    Attributes:
        iteration: The iteration's 1-based number.
        changed: Whether any pass changed the IR in the iteration.
        pass_runs: The records of the iteration's pass runs.

    """

    __slots__ = ()
    __match_args__ = ("iteration", "changed", "pass_runs")


FrozenMixin.register(FixpointIterationRecord)
FixpointIterationRecord._register_public_class()


class FixpointGroupRecord(_rs.FixpointGroupRecord, PartialEqualMixin):
    """Execution record for a fixpoint group.

    Attributes:
        group_name: The group's name.
        iteration_records: The records of the iterations run; after a
            failure inside the group, the last lists only the pass runs
            that completed.
        converged: Whether an iteration changed nothing.
        iterations: The number of iterations run.

    """

    __slots__ = ()
    __match_args__ = ("group_name", "iteration_records", "converged")


FrozenMixin.register(FixpointGroupRecord)
FixpointGroupRecord._register_public_class()


class PassManagerResult(_rs.PassManagerResult, PartialEqualMixin, Generic[_IRType]):
    """Overall pass manager execution result.

    Attributes:
        output: The final IR.
        records: One record per pipeline item.

    ``pass_runs()`` returns the record of every pass run, with each group's
    in place of the group, and ``run_count()`` the number of runs that were
    not skipped.
    """

    __slots__ = ()
    __match_args__ = ("output", "records")

    if TYPE_CHECKING:
        # The stub cannot make `_rs.PassManagerResult` generic in the IR type.
        def __new__(
            cls,
            output: _IRType,
            records: Iterable[PassRunRecord | FixpointGroupRecord],
        ) -> Self:
            """Return the result of a run."""
            ...

        @property
        @override
        def output(self) -> _IRType:
            """The final IR."""
            ...


FrozenMixin.register(PassManagerResult)
PassManagerResult._register_public_class()


class FixpointPassGroup(_rs.FixpointPassGroup, Generic[_IRType]):
    """A pass sequence repeated until an iteration changes nothing.

    Backed by the Rust implementation: ``fhy_core._rs.FixpointPassGroup``.
    ``FixpointPassGroup(name, *, max_iterations=10,
    fail_on_non_convergence=True)``; ``max_iterations`` below 1 raises
    ``ValueError``. A pipeline reads the group's passes when it runs, so a
    pass added after the group was added to a pipeline runs too.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @override
        def add_pass(self, compiler_pass: CompilerPass[_IRType, _IRType]) -> None:
            """Append a pass to the group."""
            ...

        @property
        @override
        def passes(self) -> tuple[CompilerPass[_IRType, _IRType], ...]:
            """The group's passes, in order."""
            ...


class PassManager(_rs.PassManager, Generic[_IRType]):
    """Ordered pass pipeline over one IR type.

    Backed by the Rust implementation: ``fhy_core._rs.PassManager``. Each
    run builds the Rust pipeline from the current passes and groups, with
    its own analysis cache. It logs INFO lines when it starts and finishes,
    and DEBUG lines per item.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @override
        def add_pass(self, compiler_pass: CompilerPass[_IRType, _IRType]) -> None:
            """Append one pass to the pipeline."""
            ...

        @override
        def add_fixpoint_group(self, group: FixpointPassGroup[_IRType]) -> None:
            """Append one fixpoint group to the pipeline."""
            ...

        @override
        def set_verifier(self, verifier: "ValidationManager[_IRType] | None") -> None:
            """Verify every run's IR with ``verifier``, or verify nothing."""
            ...

    @override
    def run(self, ir: _IRType) -> PassManagerResult[_IRType]:
        """Run the pass pipeline over the IR.

        Args:
            ir: IR to optimize.

        Returns:
            The overall pass manager result, including the final IR and execution
            records.

        Raises:
            PassValidationError: If a validation hook fails, or the verifier
                rejects the input or a changed output.
            PassExecutionError: If any other hook fails, or if a fixpoint group
                fails to converge within its iteration budget.

        """
        start = time.perf_counter()
        _LOGGER.info(
            "%s starting (items=%d, input id=%d)",
            self.name,
            self._item_count(),
            id(ir),
        )
        try:
            result: PassManagerResult[_IRType] = super().run(ir)
        except PassExecutionError as error:
            if error.pass_name is None:
                _LOGGER.error("%s (elapsed=%.2fms); raising", error, _elapsed_ms(start))
            raise
        if _LOGGER.isEnabledFor(logging.INFO):
            _log_records(result.records)
        _LOGGER.info(
            "%s finished (output id=%d, elapsed=%.2fms)",
            self.name,
            id(result.output),
            _elapsed_ms(start),
        )
        return result


def _elapsed_ms(start: float) -> float:
    """Return the milliseconds since ``start``, a ``perf_counter`` reading."""
    return (time.perf_counter() - start) * 1000.0


def _log_pass_run(record: PassRunRecord) -> None:
    """Log the DEBUG line of one pass run's record."""
    _LOGGER.debug(
        "pass %s finished "
        "(changed=%s, skipped=%s, diagnostics=%d, preserve_all=%s, preserved=%d)",
        record.pass_name,
        record.changed,
        record.skipped,
        len(record.diagnostics),
        record.preserved_analyses.preserve_all,
        len(record.preserved_analyses.analysis_names),
    )


def _log_fixpoint_group(record: FixpointGroupRecord) -> None:
    """Log one fixpoint group's record: its iterations at DEBUG, its end at INFO."""
    for iteration in record.iteration_records:
        for pass_run in iteration.pass_runs:
            _log_pass_run(pass_run)
        _LOGGER.debug(
            "%s iteration %d (changed_any=%s)",
            record.group_name,
            iteration.iteration,
            iteration.changed,
        )
    _LOGGER.info(
        "%s finished (converged=%s, iterations=%d)",
        record.group_name,
        record.converged,
        record.iterations,
    )


def _log_records(records: Iterable[PassRunRecord | FixpointGroupRecord]) -> None:
    """Log the lines of a run's records, in pipeline order."""
    for record in records:
        if isinstance(record, FixpointGroupRecord):
            _log_fixpoint_group(record)
        else:
            _log_pass_run(record)
