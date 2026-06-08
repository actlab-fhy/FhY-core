"""Core compiler pass abstractions and registration."""

from fhy_core.utils.override import override

__all__ = [
    "AnalysisVisitablePass",
    "CompilerPass",
    "PassExecutionError",
    "PassInfo",
    "PassRegistrationError",
    "PassResult",
    "PassValidationError",
    "PreservedAnalyses",
    "RewritablePass",
    "TraversalOrder",
    "VisitablePass",
    "register_pass",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from immutabledict import immutabledict

from fhy_core.diagnostic import Diagnostic, DiagnosticLevel, Note
from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.traits import FrozenMixin, PartialEqualMixin, Visitable
from fhy_core.utils.enum import StrEnum
from fhy_core.utils.self import Self

if TYPE_CHECKING:
    from fhy_core.diagnostic import ValidationReport

    from .manager import Analysis, AnalysisManager

_PassInputT = TypeVar("_PassInputT")
_PassOutputT = TypeVar("_PassOutputT")
_PassClassT = TypeVar("_PassClassT", bound=type["CompilerPass[Any, Any]"])
_VisitableNodeT = TypeVar("_VisitableNodeT", bound=Visitable)
_AnalysisIRT = TypeVar("_AnalysisIRT")
_AnalysisResultT = TypeVar("_AnalysisResultT")


_DIAGNOSTIC_TO_LOGGING_LEVEL: immutabledict[DiagnosticLevel, int] = immutabledict(
    {
        DiagnosticLevel.ERROR: logging.ERROR,
        DiagnosticLevel.WARNING: logging.WARNING,
        DiagnosticLevel.INFO: logging.INFO,
    }
)


class TraversalOrder(StrEnum):
    """Traversal order for automatic visitable analysis passes."""

    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class PreservedAnalyses(FrozenMixin, PartialEqualMixin):
    """Set of analyses preserved by a pass run."""

    preserve_all: bool = field(default=False)
    analysis_names: frozenset[Identifier] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.preserve_all and self.analysis_names:
            raise ValueError(
                "PreservedAnalyses: analysis_names must be empty when "
                "preserve_all=True."
            )

    @classmethod
    def all(cls) -> "PreservedAnalyses":
        """Create a preserved set representing all analyses."""
        return cls(preserve_all=True)

    @classmethod
    def none(cls) -> "PreservedAnalyses":
        """Create a preserved set representing no analyses."""
        return cls(preserve_all=False)

    def preserve(self, analysis_name: Identifier) -> "PreservedAnalyses":
        """Mark one analysis as preserved."""
        if self.preserve_all:
            return self
        if analysis_name in self.analysis_names:
            return self
        return PreservedAnalyses(
            preserve_all=False,
            analysis_names=self.analysis_names | {analysis_name},
        )

    def is_preserved(self, analysis_name: Identifier) -> bool:
        """Return whether an analysis is preserved."""
        return self.preserve_all or analysis_name in self.analysis_names


@dataclass(frozen=True)
class PassInfo(FrozenMixin, PartialEqualMixin):
    """Registered pass metadata."""

    name: str
    description: str
    pass_type: type["CompilerPass[Any, Any]"]


@register_error
class PassRegistrationError(RuntimeError):
    """Pass registration failure."""


@register_error
class PassValidationError(RuntimeError):
    """Pass validation failure.

    Carries the underlying :class:`ValidationReport` when raised by the
    auto-verification hook (see :class:`VerificationAnalysis`).
    ``report`` is ``None`` when the error originated from a user's
    ``validate_input`` / ``validate_output`` raising something other
    than auto-verification.
    """

    _report: "ValidationReport[Any] | None"

    def __init__(
        self,
        message: str = "",
        *,
        report: "ValidationReport[Any] | None" = None,
    ) -> None:
        super().__init__(message)
        self._report = report

    @property
    def report(self) -> "ValidationReport[Any] | None":
        """The attached :class:`ValidationReport`, or ``None`` if none was set."""
        return self._report


@register_error
class PassExecutionError(RuntimeError):
    """Pass execution failure."""


@dataclass(frozen=True)
class PassResult(FrozenMixin, PartialEqualMixin, Generic[_PassOutputT]):
    """Result of a pass execution."""

    output: _PassOutputT
    changed: bool
    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)
    preserved_analyses: PreservedAnalyses = field(
        default_factory=PreservedAnalyses.none
    )


class CompilerPass(ABC, Generic[_PassInputT, _PassOutputT]):
    """Base class for standardized compiler passes."""

    _registry: ClassVar[dict[str, PassInfo]] = {}
    _run_counts: ClassVar[dict[str, int]] = {}
    _total_run_count: ClassVar[int] = 0
    _registry_lock: ClassVar[Lock] = Lock()

    _pass_name: ClassVar[str | None] = None
    _pass_description: ClassVar[str] = ""
    _auto_verify: ClassVar[bool] = True
    """When True, ``_guarded_validate_input`` and ``_guarded_validate_output``
    additionally run :class:`VerificationAnalysis` against the input and
    output IR and raise :class:`PassValidationError` (carrying the
    :class:`ValidationReport`) if it reports errors. When False, the
    auto-verify hooks are skipped entirely. Verification pass classes
    registered through ``register_verification`` are stamped to ``False``
    automatically to prevent recursive verification."""
    _diagnostics: list[Diagnostic]
    _analysis_manager: "AnalysisManager[Any] | None"

    def __init__(self) -> None:
        self._diagnostics = []
        self._analysis_manager = None

    @classmethod
    def get_pass_name(cls) -> str:
        """Return a stable pass name used for registration and reporting."""
        return cls._pass_name or cls.__name__

    @classmethod
    def get_pass_description(cls) -> str:
        """Return a stable pass description used for discovery/reporting."""
        return cls._pass_description or (cls.__doc__ or cls.__name__)

    @staticmethod
    def get_registered_passes() -> Mapping[str, PassInfo]:
        """Return all registered pass metadata entries."""
        with CompilerPass._registry_lock:
            return dict(CompilerPass._registry)

    @classmethod
    def get_run_count(cls) -> int:
        """Return execution count for this pass class."""
        with CompilerPass._registry_lock:
            return CompilerPass._run_counts.get(cls.get_pass_name(), 0)

    @staticmethod
    def get_total_run_count() -> int:
        """Return total executions across all pass classes."""
        with CompilerPass._registry_lock:
            return CompilerPass._total_run_count

    @staticmethod
    def create(pass_name: str, *args: Any, **kwargs: Any) -> "CompilerPass[Any, Any]":
        """Create an instance from the global pass registry."""
        with CompilerPass._registry_lock:
            pass_info = CompilerPass._registry.get(pass_name)
        if pass_info is None:
            raise PassRegistrationError(f'Unknown pass "{pass_name}".')
        return pass_info.pass_type(*args, **kwargs)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Return diagnostics emitted during the most recent run."""
        return tuple(self._diagnostics)

    def __call__(self, ir: _PassInputT) -> _PassOutputT:
        """Run the pass and return only the output IR.

        Use :meth:`execute` instead when diagnostics, the changed flag, or
        preserved analyses are needed; this convenience form discards them.
        """
        return self.execute(ir).output

    def execute(self, ir: _PassInputT) -> PassResult[_PassOutputT]:
        """Execute the pass with validation and standardized error handling.

        Every user-overridable lifecycle method is executed inside a guard
        with a method-specific error contract:

        - ``validate_input`` / ``validate_output``: any unexpected exception
          is converted to ``PassValidationError`` after emitting an ERROR
          diagnostic. ``PassValidationError`` raised directly by user code
          passes through unchanged.
        - ``should_run`` / ``get_noop_output`` / ``run_pass`` /
          ``did_change`` / ``get_preserved_analyses``: any unexpected
          exception is converted to ``PassExecutionError`` after emitting
          an ERROR diagnostic. ``PassExecutionError`` raised directly by
          user code passes through unchanged.

        Run counters (``get_run_count`` / ``get_total_run_count``) increment
        once per ``execute(ir)`` invocation where ``should_run(ir)`` returned
        True. The counter increments before ``run_pass`` is invoked, so a
        pass that raises during ``run_pass``, ``validate_output``,
        ``did_change``, or ``get_preserved_analyses`` still counts toward
        its run total. Skipped runs (``should_run`` returned False) do not
        count.
        """
        pass_logger = self._get_pass_logger()
        pass_logger.debug(
            "entering (prospective run #%d, input type=%s)",
            self.get_run_count() + 1,
            type(ir).__name__,
        )

        self._diagnostics = []
        self._guarded_validate_input(ir)

        if not self._guarded_should_run(ir):
            pass_logger.debug("skipped: should_run returned False")
            noop_output = self._guarded_get_noop_output(ir)
            preserved_skip = self._guarded_get_preserved_analyses(
                ir, noop_output, changed=False
            )
            return PassResult(
                noop_output,
                False,
                diagnostics=tuple(self._diagnostics),
                preserved_analyses=preserved_skip,
            )

        self._record_run()
        try:
            output = self.run_pass(ir)
        except (PassValidationError, PassExecutionError):
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed with {type(exc).__name__}: {exc}'
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassExecutionError(message) from exc

        self._guarded_validate_output(ir, output)
        changed = self._guarded_did_change(ir, output)
        preserved = self._guarded_get_preserved_analyses(ir, output, changed=changed)
        pass_logger.debug(
            "finished (changed=%s, output type=%s, diagnostics=%d)",
            changed,
            type(output).__name__,
            len(self._diagnostics),
        )
        return PassResult(
            output=output,
            changed=changed,
            diagnostics=tuple(self._diagnostics),
            preserved_analyses=preserved,
        )

    def _guarded_validate_input(self, ir: _PassInputT) -> None:
        try:
            self.validate_input(ir)
        except PassValidationError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed validate_input with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassValidationError(message) from exc

        if type(self)._auto_verify:
            self._auto_verify_input(ir)

    def _guarded_validate_output(
        self, input_ir: _PassInputT, output: _PassOutputT
    ) -> None:
        try:
            self.validate_output(input_ir, output)
        except PassValidationError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed validate_output with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassValidationError(message) from exc

        if type(self)._auto_verify:
            self._auto_verify_output(output)

    def _auto_verify_input(self, ir: _PassInputT) -> None:
        self._auto_verify_ir(ir, "rejected input IR")

    def _auto_verify_output(self, output: _PassOutputT) -> None:
        self._auto_verify_ir(output, "produced invalid output IR")

    def _auto_verify_ir(self, ir: object, phase_summary: str) -> None:
        # Lazy import: verification.py imports CompilerPass from this module.
        from .verification import VerificationAnalysis  # noqa: PLC0415

        report = self.get_analysis(VerificationAnalysis, ir)
        if not report.has_errors():
            return
        summary = (
            f'Pass "{self.get_pass_name()}" {phase_summary}: '
            f"verification reported {len(report.errors())} error(s)."
        )
        self.report(DiagnosticLevel.ERROR, summary, detail=report.format())
        raise PassValidationError(summary, report=report)

    def _guarded_should_run(self, ir: _PassInputT) -> bool:
        try:
            return self.should_run(ir)
        except PassExecutionError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed should_run with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassExecutionError(message) from exc

    def _guarded_get_noop_output(self, ir: _PassInputT) -> _PassOutputT:
        try:
            return self.get_noop_output(ir)
        except PassExecutionError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed get_noop_output with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassExecutionError(message) from exc

    def _guarded_did_change(self, input_ir: _PassInputT, output: _PassOutputT) -> bool:
        try:
            return self.did_change(input_ir, output)
        except PassExecutionError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed did_change with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassExecutionError(message) from exc

    def _guarded_get_preserved_analyses(
        self, input_ir: _PassInputT, output: _PassOutputT, *, changed: bool
    ) -> PreservedAnalyses:
        try:
            return self.get_preserved_analyses(input_ir, output, changed=changed)
        except PassExecutionError:
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed get_preserved_analyses '
                f"with {type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message, exc_info=exc)
            raise PassExecutionError(message) from exc

    def get_analysis_manager(self) -> "AnalysisManager[Any] | None":
        """Return the analysis manager currently bound to this pass, if any."""
        return self._analysis_manager

    def bind_analysis_manager(self, analysis_manager: "AnalysisManager[Any]") -> None:
        """Attach an analysis manager to this pass.

        Typically called by :class:`PassManager` before executing this pass,
        so that :meth:`get_analysis` resolves against the cache. Use
        :meth:`unbind_analysis_manager` to detach.
        """
        if analysis_manager is None:
            raise TypeError(
                "bind_analysis_manager requires a non-None AnalysisManager; "
                "use unbind_analysis_manager() to detach."
            )
        self._analysis_manager = analysis_manager

    def unbind_analysis_manager(self) -> None:
        """Detach the analysis manager, returning the pass to standalone mode.

        After this call, :meth:`get_analysis` recomputes results on every
        call. Calling this method when no manager is bound is a no-op.
        """
        self._analysis_manager = None

    def get_analysis(
        self,
        analysis_type: "type[Analysis[_AnalysisIRT, _AnalysisResultT]]",
        ir: _AnalysisIRT,
    ) -> _AnalysisResultT:
        """Obtain an analysis result for ``ir``.

        When this pass is executed under a :class:`PassManager`, results are
        fetched from the manager's :class:`AnalysisManager`, which caches them
        and preserves/invalidates across passes based on each pass's
        ``get_preserved_analyses`` return value. When the pass is executed
        standalone (no manager has been bound), the analysis is computed fresh
        on every call.

        Args:
            analysis_type: The analysis class to obtain results for.
            ir: The IR instance to analyze. Typically the same IR being passed
                to the current pass, but sub-IR is also accepted.

        Returns:
            The analysis result, cached when possible.

        """
        if self._analysis_manager is None:
            return analysis_type().run(ir)
        return self._analysis_manager.get(analysis_type, ir)

    def report(
        self,
        level: DiagnosticLevel,
        message: str | Note,
        detail: str | None = None,
        *,
        exc_info: BaseException | bool | None = None,
    ) -> None:
        """Emit a diagnostic for this pass execution.

        The diagnostic is recorded on the in-memory list (returned via
        :attr:`diagnostics` and ``PassResult.diagnostics``) and additionally
        emitted on a per-pass-class logger named
        ``<core-module>.<pass-name>``. The logging level mirrors the
        diagnostic level (ERROR/WARNING/INFO). When ``detail`` is provided
        it is appended to the log message as ``" | detail: <detail>"``;
        the in-memory record stores the detail separately on the
        :class:`Diagnostic`. When ``exc_info`` is provided, it is
        forwarded to the underlying log call so the originating traceback
        is preserved alongside the structured record; the in-memory
        diagnostic does not carry it.
        """
        note = message if isinstance(message, Note) else Note(message)
        self._diagnostics.append(
            Diagnostic(
                level=level,
                message=note,
                source=self.get_pass_name(),
                detail=detail,
            )
        )

        log_message = note.message
        if detail is not None:
            log_message = f"{log_message} | detail: {detail}"
        self._get_pass_logger().log(
            _DIAGNOSTIC_TO_LOGGING_LEVEL[level], log_message, exc_info=exc_info
        )

    @classmethod
    def _get_pass_logger(cls) -> logging.Logger:
        return get_logger(__name__).getChild(cls.get_pass_name())

    def validate_input(self, ir: _PassInputT) -> None:
        """Validate input IR before execution."""
        if ir is None:
            message = f'Pass "{self.get_pass_name()}" does not accept None input.'
            self.report(DiagnosticLevel.ERROR, message)
            raise PassValidationError(message)

    def should_run(self, ir: _PassInputT) -> bool:
        """Return whether this pass should run for the input IR."""
        return True

    @abstractmethod
    def get_noop_output(self, ir: _PassInputT) -> _PassOutputT:
        """Return output when pass execution is skipped."""

    @abstractmethod
    def run_pass(self, ir: _PassInputT) -> _PassOutputT:
        """Run the pass over IR after validation."""

    def validate_output(self, input_ir: _PassInputT, output: _PassOutputT) -> None:
        """Validate output after execution."""

    def did_change(self, input_ir: _PassInputT, output: _PassOutputT) -> bool:
        """Return whether output differs from input.

        This method prefers value semantics (`!=`) when supported by the input/output
        types, and falls back to identity semantics if value comparison fails.
        """
        try:
            return bool(cast(Any, input_ir) != output)
        except Exception:
            return input_ir is not output

    def get_preserved_analyses(
        self, input_ir: _PassInputT, output: _PassOutputT, *, changed: bool
    ) -> PreservedAnalyses:
        """Return analyses preserved by this pass run.

        By default, unchanged passes preserve all analyses; changed passes preserve
        none.
        """
        _ = (input_ir, output)
        if changed:
            return PreservedAnalyses.none()
        return PreservedAnalyses.all()

    def _record_run(self) -> None:
        pass_name = self.get_pass_name()
        with CompilerPass._registry_lock:
            CompilerPass._total_run_count += 1
            CompilerPass._run_counts[pass_name] = (
                CompilerPass._run_counts.get(pass_name, 0) + 1
            )


class VisitablePass(CompilerPass[_VisitableNodeT, _PassOutputT], ABC):
    """Compiler pass with convention-based visitor dispatch.

    Visitor method naming convention:
        Subclasses implement per-node-type visitor methods named
        ``visit_<suffix>``, where ``<suffix>`` is produced by
        ``Visitable.get_visit_method_suffix()``. By default, that suffix is the
        node class name converted from ``CamelCase`` to ``snake_case``. For
        example, a node class named ``BinaryExpression`` dispatches to
        ``visit_binary_expression``, and a node named ``IntLiteral`` dispatches
        to ``visit_int_literal``. A node type may override
        ``get_visit_method_suffix()`` to customize this mapping.

        When no matching ``visit_<suffix>`` method is defined on the pass,
        dispatch falls back to ``visit_unknown``, which by default raises
        ``NotImplementedError``. Subclasses may override ``visit_unknown`` to
        provide a generic handler.
    """

    _VISIT_METHOD_PREFIX: ClassVar[str] = "visit_"

    @override
    def run_pass(self, ir: _VisitableNodeT) -> _PassOutputT:
        return self.visit(ir)

    def visit(self, node: _VisitableNodeT) -> _PassOutputT:
        """Visit a node by resolving `visit_<node_kind>` dynamically.

        Args:
            node: Node to visit.

        Returns:
            Result of visiting the node.

        """
        method_name = (
            f"{self._VISIT_METHOD_PREFIX}{type(node).get_visit_method_suffix()}"
        )
        candidate = getattr(self, method_name, None)
        if candidate is None or not callable(candidate):
            return self.visit_unknown(node)
        method = cast(Callable[[_VisitableNodeT], _PassOutputT], candidate)
        return method(node)

    def visit_unknown(self, node: _VisitableNodeT) -> _PassOutputT:
        """Handle node types without a dedicated visitor method."""
        raise NotImplementedError(
            f'"{self.get_pass_name()}" does not implement "{type(node).__name__}"'
            " handling."
        )


class AnalysisVisitablePass(VisitablePass[_VisitableNodeT, None], ABC):
    """Analysis-only visitable pass with optional automatic traversal.

    Per-node pre/post hook convention:
        In addition to ``visit_<suffix>`` dispatch inherited from
        ``VisitablePass``, this class dispatches per-node
        ``before_visit_<suffix>`` and ``after_visit_<suffix>`` hooks around
        the walk of every node (both the root and each descendant), using
        the same ``<suffix>`` convention as ``visit_<suffix>``. The pre-hook
        runs before the node's visit method and any child traversal; the
        post-hook runs after both have completed, regardless of traversal
        order. When a hook method is not defined for a given node type,
        dispatch falls back to ``before_visit_unknown`` /
        ``after_visit_unknown`` (both no-ops by default). This enables
        subclasses to inject node-type-specific pre/post processing (e.g.,
        pushing/popping a scope for a ``FunctionDefinition``) independent
        of traversal order, without overriding the walk itself.

    Unknown-node handling:
        Unlike ``VisitablePass.visit_unknown`` (which raises
        ``NotImplementedError``), ``AnalysisVisitablePass.visit_unknown`` is
        a no-op by default. This lets analysis passes quietly skip node
        types they do not care about during a full-tree walk. Override
        ``visit_unknown`` if strict handling is required.
    """

    _BEFORE_VISIT_METHOD_PREFIX: ClassVar[str] = "before_visit_"
    _AFTER_VISIT_METHOD_PREFIX: ClassVar[str] = "after_visit_"

    _traversal_order: TraversalOrder

    def __init__(
        self, traversal_order: TraversalOrder | str = TraversalOrder.PRE
    ) -> None:
        super().__init__()
        self._traversal_order = TraversalOrder(traversal_order)

    @property
    def traversal_order(self) -> TraversalOrder:
        """Return the traversal order used to walk the IR."""
        return self._traversal_order

    @override
    def run_pass(self, ir: _VisitableNodeT) -> None:
        self.walk(ir)

    @override
    def get_noop_output(self, ir: _VisitableNodeT) -> None:
        _ = ir

    @override
    def did_change(self, input_ir: _VisitableNodeT, output: None) -> bool:
        return False

    def walk(self, node: _VisitableNodeT) -> None:
        """Visit a node and, when provided, recursively visit its children.

        Each walked node is bracketed by dispatch-based ``before_visit_*`` and
        ``after_visit_*`` hooks. The pre-hook runs before any visit or child
        traversal for the node, and the post-hook runs after both the node's
        visit method and its child traversal have completed, regardless of
        traversal order. See the class docstring for dispatch details.

        Precondition: the visit graph rooted at ``node`` must be finite and
        acyclic. ``walk`` performs no cycle detection; cyclic or
        DAG-with-shared-subtree IRs cause unbounded recursion until Python
        raises ``RecursionError``. FhY IRs are tree-shaped by convention.

        Args:
            node: Node to visit.

        """
        self.before_visit(node)
        try:
            if self._traversal_order == TraversalOrder.PRE:
                self.visit(node)
                self.walk_children(node)
            else:
                self.walk_children(node)
                self.visit(node)
        finally:
            self.after_visit(node)

    def walk_children(self, node: _VisitableNodeT) -> None:
        """Visit all children declared by the node.

        Args:
            node: Node whose children to visit.

        """
        for child in self.get_visit_children(node):
            self.walk(child)

    def before_visit(self, node: _VisitableNodeT) -> None:
        """Dispatch the pre-visit hook for ``node``.

        Resolves ``before_visit_<suffix>`` using the same naming convention as
        ``visit``. Falls back to ``before_visit_unknown`` when no dedicated
        method is defined.

        Args:
            node: Node about to be walked.

        """
        method_name = (
            f"{self._BEFORE_VISIT_METHOD_PREFIX}{type(node).get_visit_method_suffix()}"
        )
        candidate = getattr(self, method_name, None)
        if candidate is None or not callable(candidate):
            self.before_visit_unknown(node)
            return
        method = cast(Callable[[_VisitableNodeT], None], candidate)
        method(node)

    def after_visit(self, node: _VisitableNodeT) -> None:
        """Dispatch the post-visit hook for ``node``.

        Resolves ``after_visit_<suffix>`` using the same naming convention as
        ``visit``. Falls back to ``after_visit_unknown`` when no dedicated
        method is defined.

        Args:
            node: Node that was just walked.

        """
        method_name = (
            f"{self._AFTER_VISIT_METHOD_PREFIX}{type(node).get_visit_method_suffix()}"
        )
        candidate = getattr(self, method_name, None)
        if candidate is None or not callable(candidate):
            self.after_visit_unknown(node)
            return
        method = cast(Callable[[_VisitableNodeT], None], candidate)
        method(node)

    def before_visit_unknown(self, node: _VisitableNodeT) -> None:
        """Default pre-visit handler for node types without a dedicated hook."""
        _ = node

    def after_visit_unknown(self, node: _VisitableNodeT) -> None:
        """Default post-visit handler for node types without a dedicated hook."""
        _ = node

    def get_visit_children(self, node: _VisitableNodeT) -> Sequence[_VisitableNodeT]:
        """Return children for automatic traversal.

        By default, this uses optional node-provided child enumeration via
        `Visitable.get_visit_children()`. If a node does not override that
        method, no child recursion is performed for that node and traversal
        must be done manually in visit methods.
        """
        return cast(Sequence[_VisitableNodeT], node.get_visit_children())

    @override
    def visit_unknown(self, node: _VisitableNodeT) -> None: ...


@runtime_checkable
class _VisitableRewritable(Protocol):
    """Internal protocol combining `Visitable` + `Rewritable[Self]`.

    The bound on :class:`RewritablePass`'s node type. A node satisfies
    this protocol when it provides both:

    - the :class:`~fhy_core.traits.Visitable` API
      (``get_visit_method_suffix``, ``get_visit_children``), and
    - a ``rebuild_with_visit_children(new_children: Sequence[Self]) -> Self``
      method (the :class:`~fhy_core.traits.Rewritable` API specialized
      to the node's own type).
    """

    @classmethod
    def get_visit_method_suffix(cls) -> str: ...

    def get_visit_children(self) -> Sequence[Self]: ...

    def rebuild_with_visit_children(self, new_children: Sequence[Self]) -> Self: ...


_RewritableNodeT = TypeVar("_RewritableNodeT", bound=_VisitableRewritable)


class RewritablePass(
    CompilerPass[_RewritableNodeT, _RewritableNodeT], ABC, Generic[_RewritableNodeT]
):
    """Compiler pass that rewrites a tree of ``Visitable + Rewritable`` nodes.

    Subclasses override per-node ``visit_<kind>`` methods to rewrite a
    node:

    - Return ``None`` to leave the node unchanged.
    - Return a node of the same kind to replace it.

    The framework handles the recursive walk. For each node, it first
    transforms every child via :meth:`transform`. If any child changed,
    it rebuilds the node (via
    :meth:`~fhy_core.traits.RewritableMixin.rebuild_with_visit_children`)
    with the new children before calling ``visit_<kind>`` on the result.
    Each visitor therefore sees a node whose children are already in
    their post-transform form.

    The public entry point is :meth:`transform`, which returns the
    fully-rewritten tree (identity-preserved when nothing changed).
    Invoking the pass as a callable (``rewriter(node)``) routes through
    the standard pass pipeline and ends up calling :meth:`transform`.

    Node-type requirement:
        The node type must satisfy both the
        :class:`~fhy_core.traits.Visitable` and
        :class:`~fhy_core.traits.Rewritable` protocols. The type bound
        enforces this at ``RewritablePass[_NodeT]`` specialization
        time; nodes mixing in
        :class:`~fhy_core.traits.VisitableMixin` and
        :class:`~fhy_core.traits.RewritableMixin` (parameterized at
        their own type) qualify naturally.
    """

    _VISIT_METHOD_PREFIX: ClassVar[str] = "visit_"

    @override
    def run_pass(self, ir: _RewritableNodeT) -> _RewritableNodeT:
        """Pass-framework entry: forwards to :meth:`transform`."""
        return self.transform(ir)

    @override
    def get_noop_output(self, ir: _RewritableNodeT) -> _RewritableNodeT:
        """Return the input unchanged when the pass is skipped."""
        return ir

    @override
    def did_change(self, input_ir: _RewritableNodeT, output: _RewritableNodeT) -> bool:
        """Identity-based change detection (``output is not input_ir``).

        The rewriter promises identity preservation on no-op rewrites,
        so object identity is the precise signal.
        """
        return input_ir is not output

    def visit(self, node: _RewritableNodeT) -> _RewritableNodeT:
        """Convenience entry: delegates to :meth:`transform`."""
        return self.transform(node)

    def transform(self, node: _RewritableNodeT) -> _RewritableNodeT:
        """Rewrite ``node`` and return either it or a new tree.

        Identity-preserving: when no node in the tree is rewritten, the
        input ``node`` is returned unchanged (same object identity).

        Args:
            node: Root node to rewrite.

        Returns:
            The rewritten tree, or the input unchanged.
        """
        rewritten = self._transform_optional(node)
        return node if rewritten is None else rewritten

    def visit_unknown(self, node: _RewritableNodeT) -> _RewritableNodeT | None:
        """Default per-node handler when no ``visit_<kind>`` is defined.

        Returns ``None`` (no rewrite). The rewriter framework treats
        ``None`` as "leave this node alone, but still propagate any
        child rewrites." This differs from
        :meth:`VisitablePass.visit_unknown` (which raises): the
        rewriter's semantics are that an unhandled node is preserved,
        not an error.

        This lenient default is intentional for the common
        *selectively-transformative* pattern (e.g. an inliner that
        rewrites :class:`CallExpression` only, or an evaluator that
        rewrites literal-argument calls and constant references). Such
        passes intentionally do not enumerate every node kind in the
        IR; the lenient default lets them rely on the framework's
        recursive walk to preserve everything else.

        Subclasses that genuinely require *exhaustive* coverage (e.g.
        a verifier that must explicitly accept or reject every node
        kind) override this method to raise. When new node kinds are
        later added to the IR, an exhaustive subclass will fail loudly
        the first time it encounters the new kind, forcing the author
        to extend its visitor set.
        """
        _ = node
        return None

    def _transform_optional(self, node: _RewritableNodeT) -> _RewritableNodeT | None:
        """Transform ``node``; return ``None`` when no node was rewritten."""
        rebuilt = self._rebuild_with_transformed_children(node)
        node_to_visit = rebuilt if rebuilt is not None else node
        per_node = self._dispatch_per_node_visitor(node_to_visit)
        if per_node is not None:
            return per_node
        return rebuilt

    def _rebuild_with_transformed_children(
        self, node: _RewritableNodeT
    ) -> _RewritableNodeT | None:
        """Rebuild ``node`` with transformed children, or ``None``."""
        original_children = tuple(node.get_visit_children())
        if not original_children:
            return None
        transformed_children = tuple(
            self._transform_optional(child) for child in original_children
        )
        if all(child is None for child in transformed_children):
            return None
        merged_children = tuple(
            transformed if transformed is not None else original
            for transformed, original in zip(
                transformed_children, original_children, strict=True
            )
        )
        return node.rebuild_with_visit_children(merged_children)

    def _dispatch_per_node_visitor(
        self, node: _RewritableNodeT
    ) -> _RewritableNodeT | None:
        method_name = (
            f"{self._VISIT_METHOD_PREFIX}{type(node).get_visit_method_suffix()}"
        )
        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            return self.visit_unknown(node)
        return method(node)  # type: ignore[no-any-return]


def register_pass(name: str, description: str) -> Callable[[_PassClassT], _PassClassT]:
    """Register a concrete pass class with explicit metadata.

    Args:
        name: Stable pass name for registration and reporting.
        description: Human-readable pass description for discovery/reporting.

    Raises:
        PassRegistrationError: If the pass class is invalid or the name is already
            registered by a different class.

    """
    if not name.strip():
        raise PassRegistrationError("Pass name cannot be empty.")
    if not description.strip():
        raise PassRegistrationError("Pass description cannot be empty.")

    def _decorator(pass_cls: _PassClassT) -> _PassClassT:
        if not issubclass(pass_cls, CompilerPass):
            raise PassRegistrationError(
                f"Cannot register non-CompilerPass type: {pass_cls.__qualname__}."
            )
        with CompilerPass._registry_lock:
            existing = CompilerPass._registry.get(name)
            if existing is not None:
                if existing.pass_type is not pass_cls:
                    raise PassRegistrationError(
                        f'Pass name "{name}" is already registered by '
                        f"{existing.pass_type.__qualname__} "
                        f"with description {existing.description!r}."
                    )
                if existing.description != description:
                    raise PassRegistrationError(
                        f'Pass name "{name}" is already registered by '
                        f"{pass_cls.__qualname__} with description "
                        f"{existing.description!r}; refusing to overwrite "
                        f"with new description {description!r}."
                    )
                get_logger(__name__).debug(
                    "%s already registered as %s (idempotent re-registration)",
                    name,
                    pass_cls.__qualname__,
                )
                return pass_cls

            pass_cls._pass_name = name
            pass_cls._pass_description = description
            CompilerPass._registry[name] = PassInfo(name, description, pass_cls)
            CompilerPass._run_counts.setdefault(name, 0)
            get_logger(__name__).debug(
                "registered %s -> %s", name, pass_cls.__qualname__
            )

        return pass_cls

    return _decorator
