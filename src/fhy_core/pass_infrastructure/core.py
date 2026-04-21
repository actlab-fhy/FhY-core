"""Core compiler pass abstractions and registration."""

__all__ = [
    "AnalysisVisitablePass",
    "CompilerPass",
    "DiagnosticLevel",
    "PassDiagnostic",
    "PassExecutionError",
    "PassInfo",
    "PassRegistrationError",
    "PassResult",
    "PassValidationError",
    "PreservedAnalyses",
    "TraversalOrder",
    "VisitablePass",
    "register_pass",
]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Mapping,
    TypeVar,
    cast,
)

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.provenance import Note
from fhy_core.trait import FrozenMixin, PartialEqualMixin, Visitable
from fhy_core.utils.enum import StrEnum

if TYPE_CHECKING:
    from .manager import Analysis, AnalysisManager

_PassInputT = TypeVar("_PassInputT")
_PassOutputT = TypeVar("_PassOutputT")
_PassClassT = TypeVar("_PassClassT", bound=type["CompilerPass[Any, Any]"])
_VisitableNodeT = TypeVar("_VisitableNodeT", bound=Visitable)
_AnalysisIRT = TypeVar("_AnalysisIRT")
_AnalysisResultT = TypeVar("_AnalysisResultT")


class DiagnosticLevel(StrEnum):
    """Diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class TraversalOrder(StrEnum):
    """Traversal order for automatic visitable analysis passes."""

    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class PassDiagnostic(FrozenMixin, PartialEqualMixin):
    """Structured diagnostic emitted by a pass."""

    level: DiagnosticLevel
    message: Note
    pass_name: str
    detail: str | None = None

    @property
    def message_text(self) -> str:
        return self.message.message


@dataclass(frozen=True)
class PreservedAnalyses(FrozenMixin, PartialEqualMixin):
    """Set of analyses preserved by a pass run."""

    preserve_all: bool = field(default=False)
    analysis_names: frozenset[Identifier] = field(default_factory=frozenset)

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
    """Pass validation failure."""


@register_error
class PassExecutionError(RuntimeError):
    """Pass execution failure."""


@dataclass(frozen=True)
class PassResult(FrozenMixin, PartialEqualMixin, Generic[_PassOutputT]):
    """Result of a pass execution."""

    output: _PassOutputT
    changed: bool
    diagnostics: tuple[PassDiagnostic, ...] = field(default_factory=tuple)
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
    _diagnostics: list[PassDiagnostic]
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
    def diagnostics(self) -> tuple[PassDiagnostic, ...]:
        """Return diagnostics emitted during the most recent run."""
        return tuple(self._diagnostics)

    def __call__(self, ir: _PassInputT) -> _PassOutputT:
        return self.execute(ir).output

    def execute(self, ir: _PassInputT) -> PassResult[_PassOutputT]:
        """Execute the pass with validation and standardized error handling."""
        self._diagnostics = []
        self.validate_input(ir)

        self._record_run()
        if not self.should_run(ir):
            output = self.get_noop_output(ir)
            preserved = self.get_preserved_analyses(ir, output, changed=False)
            return PassResult(
                output,
                False,
                diagnostics=tuple(self._diagnostics),
                preserved_analyses=preserved,
            )

        try:
            output = self.run_pass(ir)
        except (PassValidationError, PassExecutionError):
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed with {type(exc).__name__}: {exc}'
            )
            self.report(DiagnosticLevel.ERROR, message)
            raise PassExecutionError(message) from exc

        self.validate_output(ir, output)
        changed = self.did_change(ir, output)
        preserved = self.get_preserved_analyses(ir, output, changed=changed)
        return PassResult(
            output=output,
            changed=changed,
            diagnostics=tuple(self._diagnostics),
            preserved_analyses=preserved,
        )

    def get_analysis_manager(self) -> "AnalysisManager[Any] | None":
        """Return the analysis manager currently bound to this pass, if any."""
        return self._analysis_manager

    def bind_analysis_manager(
        self, analysis_manager: "AnalysisManager[Any] | None"
    ) -> None:
        """Attach (or detach, with ``None``) an analysis manager to this pass.

        Typically called by :class:`PassManager` before and after executing
        this pass, so that :meth:`get_analysis` resolves against the cache.
        Passing ``None`` returns the pass to standalone mode, in which
        :meth:`get_analysis` recomputes results every call.
        """
        self._analysis_manager = analysis_manager

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
        self, level: DiagnosticLevel, message: str | Note, detail: str | None = None
    ) -> None:
        """Emit a diagnostic for this pass execution."""
        note = message if isinstance(message, Note) else Note(message)
        self._diagnostics.append(
            PassDiagnostic(
                level=level,
                message=note,
                pass_name=self.get_pass_name(),
                detail=detail,
            )
        )

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
        return self._traversal_order

    def run_pass(self, ir: _VisitableNodeT) -> None:
        self.walk(ir)

    def get_noop_output(self, ir: _VisitableNodeT) -> None:
        _ = ir

    def did_change(self, input_ir: _VisitableNodeT, output: None) -> bool:
        return False

    def walk(self, node: _VisitableNodeT) -> None:
        """Visit a node and, when provided, recursively visit its children.

        Each walked node is bracketed by dispatch-based ``before_visit_*`` and
        ``after_visit_*`` hooks. The pre-hook runs before any visit or child
        traversal for the node, and the post-hook runs after both the node's
        visit method and its child traversal have completed, regardless of
        traversal order. See the class docstring for dispatch details.

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

    def visit_unknown(self, node: _VisitableNodeT) -> None: ...


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
            if existing is not None and existing.pass_type is not pass_cls:
                raise PassRegistrationError(
                    f'Pass name "{name}" is already registered by '
                    f"{existing.pass_type.__qualname__}."
                )

            pass_cls._pass_name = name
            pass_cls._pass_description = description
            CompilerPass._registry[name] = PassInfo(name, description, pass_cls)
            CompilerPass._run_counts.setdefault(name, 0)

        return pass_cls

    return _decorator
