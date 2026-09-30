"""Core compiler pass abstractions and registration.

The pass machinery is backed by the Rust implementation (``fhy_core._rs``),
with the Rust core's semantics:

- `CompilerPass` is a Python abstract class over ``_rs.CompilerPassBase``.
  A subclass implements the hooks under their Python names, and the Rust
  lifecycle drives them: ``validate_input``, ``should_run`` (and
  ``get_noop_output`` when it is false), ``run_pass``, ``validate_output``,
  ``did_change`` and ``get_preserved_analyses``. The hooks a class does not
  override run their defaults in Rust.
- A failing hook raises `PassValidationError` or `PassExecutionError` with
  the core's message, such as ``pass "X" failed in run_pass``, whose
  ``__cause__`` is the hook's exception; the error of a nested pass run
  nests in the outer one.
- A standalone `CompilerPass.execute` computes analyses afresh and never
  verifies; a `PassManager` caches analyses for one run and verifies its IR.
- Run statistics come from each run: `PassResult.skipped`,
  ``PassRunRecord.skipped`` and ``PassManagerResult.run_count()``.

The pass registry (`register_pass`, `CompilerPass.create`) stays a registry
of Python classes. `VisitablePass`, `AnalysisVisitablePass` and
`RewritablePass` are Python classes over `CompilerPass` whose walks run in
Python inside ``run_pass``.
"""

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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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

from fhy_core import _rs
from fhy_core.diagnostic import Diagnostic, DiagnosticLevel, Note
from fhy_core.error import register_error
from fhy_core.logger import get_logger
from fhy_core.traits import FrozenMixin, PartialEqualMixin, Visitable
from fhy_core.utils.enum import StrEnum
from fhy_core.utils.self import Self

if TYPE_CHECKING:
    from fhy_core.diagnostic import ValidationReport

    from .manager import Analysis, AnalysisManager, FixpointGroupRecord, PassRunRecord

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

# The hooks whose defaults the binding runs in Rust when a class does not
# override them, in the bit order of `CompilerPass._python_hooks`. Matches
# the Rust implementation: `fhy-core-py`'s `pass::compiler_pass::hook_bit`.
_HOOKS_WITH_RUST_DEFAULTS: tuple[str, ...] = (
    "validate_input",
    "should_run",
    "validate_output",
    "did_change",
    "get_preserved_analyses",
    "get_pass_name",
)


def _get_pass_logger_by_name(pass_name: str) -> logging.Logger:
    """Return the logger of the pass named ``pass_name``."""
    return get_logger(__name__).getChild(pass_name)


def _log_diagnostic(
    source: str,
    level: DiagnosticLevel,
    message: str,
    detail: str | None,
    exc_info: BaseException | bool | None,
) -> None:
    """Log a diagnostic on the logger of the pass ``source``.

    The binding calls this for the diagnostics it makes, such as a failed
    hook's, so they are logged as `CompilerPass.report` logs the ones a
    pass reports.
    """
    log_message = message if detail is None else f"{message} | detail: {detail}"
    _get_pass_logger_by_name(source).log(
        _DIAGNOSTIC_TO_LOGGING_LEVEL[level], log_message, exc_info=exc_info
    )


def _lifecycle_logger(pass_name: str) -> logging.Logger | None:
    """Return the logger of the pass ``pass_name`` if it logs DEBUG lines.

    The binding logs a run's lifecycle lines only when this returns one.
    """
    logger = _get_pass_logger_by_name(pass_name)
    return logger if logger.isEnabledFor(logging.DEBUG) else None


def _find_defining_class(cls: type, name: str) -> type | None:
    """Return the first class in ``cls``'s MRO whose namespace defines ``name``."""
    for base in cls.__mro__:
        if name in base.__dict__:
            return base
    return None


class TraversalOrder(StrEnum):
    """Traversal order for automatic visitable analysis passes."""

    PRE = "pre"
    POST = "post"


class PreservedAnalyses(_rs.PreservedAnalyses, PartialEqualMixin):
    """Set of analyses preserved by a pass run, keyed by analysis name.

    Backed by the Rust implementation: ``fhy_core._rs.PreservedAnalyses``
    holds the core's set. ``PreservedAnalyses(preserve_all=False,
    analysis_names=frozenset())`` builds a set; setting both raises
    ``ValueError``, a ``preserve_all`` that is not a ``bool`` or a name that
    is not an `Identifier` raises ``TypeError``. Sets are immutable,
    compare and hash by what they preserve, and pickle as a call of their
    class with their fields.

    Attributes:
        preserve_all: Whether every analysis is preserved.
        analysis_names: The names of the preserved analyses; empty when every
            analysis is preserved.

    """

    __slots__ = ()
    __match_args__ = ("preserve_all", "analysis_names")


FrozenMixin.register(PreservedAnalyses)
PreservedAnalyses._register_public_class()


@dataclass(frozen=True)
class PassInfo(FrozenMixin, PartialEqualMixin):
    """Registered pass metadata."""

    name: str
    description: str
    pass_type: type["CompilerPass[Any, Any]"]


@register_error
class PassRegistrationError(RuntimeError):
    """Pass registration failure."""


def _without_rust_error(state: dict[str, Any]) -> dict[str, Any]:
    """Return an exception's ``__dict__`` without the Rust error it keeps."""
    return {key: value for key, value in state.items() if key != "_rust_error"}


@register_error
class PassValidationError(RuntimeError):
    """Pass validation failure: a validation hook failed, or verification rejected IR.

    The binding raises it with the core's message, such as ``pass "X"
    failed in validate_input`` or ``verification rejected the output of
    pass "X" (errors: 2)``, and the failing hook's exception as the
    ``__cause__``. Code may raise it too.

    Attributes:
        report: The verifier's `ValidationReport` of a verification
            failure, or ``None``.
        pass_name: The name of the failing or blamed pass, or ``None``.
        hook: The name of the Python hook that failed, such as
            ``"validate_input"``, or ``None``.
        diagnostics: The failing run's diagnostics, ending with the error
            diagnostic that records the failure.
        records: The records of the pipeline work completed before the
            failure.

    """

    pass_name: str | None
    hook: str | None
    diagnostics: tuple[Diagnostic, ...]
    records: "tuple[PassRunRecord | FixpointGroupRecord, ...]"
    _report: "ValidationReport[Any] | None"

    def __init__(
        self,
        message: str = "",
        *,
        report: "ValidationReport[Any] | None" = None,
        pass_name: str | None = None,
        hook: str | None = None,
        diagnostics: Iterable[Diagnostic] = (),
        records: "Iterable[PassRunRecord | FixpointGroupRecord]" = (),
    ) -> None:
        super().__init__(message)
        self._report = report
        self.pass_name = pass_name
        self.hook = hook
        self.diagnostics = tuple(diagnostics)
        self.records = tuple(records)

    @property
    def report(self) -> "ValidationReport[Any] | None":
        """The attached :class:`ValidationReport`, or ``None`` if none was set."""
        return self._report

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), self.args, _without_rust_error(self.__dict__))


@register_error
class PassExecutionError(RuntimeError):
    """Pass execution failure: a hook other than a validation hook failed.

    It is also raised when a fixpoint group does not converge.
    The binding raises it with the core's message, such as ``pass "X"
    failed in run_pass`` or ``fixpoint group "g" did not converge (max
    iterations: 10)``, and the failing hook's exception as the
    ``__cause__``. Code may raise it too.

    Attributes:
        pass_name: The name of the failing pass, or ``None``.
        hook: The name of the Python hook that failed, such as
            ``"run_pass"``, or ``None``.
        diagnostics: The failing run's diagnostics, ending with the error
            diagnostic that records the failure.
        records: The records of the pipeline work completed before the
            failure, a partial fixpoint group record included.

    """

    pass_name: str | None
    hook: str | None
    diagnostics: tuple[Diagnostic, ...]
    records: "tuple[PassRunRecord | FixpointGroupRecord, ...]"

    def __init__(
        self,
        message: str = "",
        *,
        pass_name: str | None = None,
        hook: str | None = None,
        diagnostics: Iterable[Diagnostic] = (),
        records: "Iterable[PassRunRecord | FixpointGroupRecord]" = (),
    ) -> None:
        super().__init__(message)
        self.pass_name = pass_name
        self.hook = hook
        self.diagnostics = tuple(diagnostics)
        self.records = tuple(records)

    @override
    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), self.args, _without_rust_error(self.__dict__))


class PassResult(_rs.PassResult, PartialEqualMixin, Generic[_PassOutputT]):
    """Result of a pass execution.

    Backed by the Rust implementation: ``fhy_core._rs.PassResult``.
    ``PassResult(output, changed, diagnostics=(), preserved_analyses=none,
    skipped=False)``; results are immutable, compare, hash and print as
    frozen dataclasses do, and pickle as a call of their class.

    Attributes:
        output: The output IR, the object the pass returned.
        changed: Whether the run changed the IR.
        diagnostics: The run's diagnostics, in emission order.
        preserved_analyses: The analyses the run left valid.
        skipped: Whether the pass skipped the run: its output came from
            ``get_noop_output`` and ``run_pass`` was not called.

    """

    __slots__ = ()
    __match_args__ = ("output", "changed", "diagnostics", "preserved_analyses")

    if TYPE_CHECKING:
        # The stub cannot make `_rs.PassResult` generic in the output type.
        def __new__(
            cls,
            output: _PassOutputT,
            changed: bool,
            diagnostics: Iterable[Diagnostic] = (),
            preserved_analyses: PreservedAnalyses = ...,
            skipped: bool = False,
        ) -> Self:
            """Return the result of a run."""
            ...

        @property
        @override
        def output(self) -> _PassOutputT:
            """The output IR."""
            ...


FrozenMixin.register(PassResult)
PassResult._register_public_class()


class CompilerPass(_rs.CompilerPassBase, ABC, Generic[_PassInputT, _PassOutputT]):
    """Base class for standardized compiler passes.

    Backed by the Rust implementation: ``fhy_core._rs.CompilerPassBase``
    drives the hooks through the Rust core's lifecycle. A subclass
    implements `run_pass`, and may override the other hooks:

    1. `validate_input`, which accepts every input by default;
    2. `should_run`, true by default; when it is false, `get_noop_output`
       gives the run's output, the run is skipped and unchanged, and
       `get_preserved_analyses` is asked with ``changed=False``;
    3. `run_pass`;
    4. `validate_output`;
    5. `did_change`, by default ``input != output``, falling back to
       ``is not`` when ``!=`` raises;
    6. `get_preserved_analyses`, by default none when changed and all
       otherwise.

    The hooks a class does not override run in Rust, without a Python
    call. `report`, `get_analysis` and `get_analysis_manager` work in the
    hooks the core gives a context, all but `did_change` and
    `get_preserved_analyses`, where they raise ``RuntimeError``; outside a
    run, `report` records on the pass, and `get_analysis` computes afresh.
    """

    _registry: ClassVar[dict[str, PassInfo]] = {}
    _registry_lock: ClassVar[Lock] = Lock()

    _pass_name: ClassVar[str | None] = None
    _pass_description: ClassVar[str] = ""
    _python_hooks: ClassVar[int] = 0
    """The hooks of `_HOOKS_WITH_RUST_DEFAULTS` the class overrides, one bit
    each; the binding runs the others' defaults in Rust."""

    @override
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        hooks = 0
        for bit, name in enumerate(_HOOKS_WITH_RUST_DEFAULTS):
            if _find_defining_class(cls, name) is not CompilerPass:
                hooks |= 1 << bit
        cls._python_hooks = hooks

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

    @staticmethod
    def create(pass_name: str, *args: Any, **kwargs: Any) -> "CompilerPass[Any, Any]":
        """Create an instance from the global pass registry."""
        with CompilerPass._registry_lock:
            pass_info = CompilerPass._registry.get(pass_name)
        if pass_info is None:
            raise PassRegistrationError(f'Unknown pass "{pass_name}".')
        return pass_info.pass_type(*args, **kwargs)

    if TYPE_CHECKING:
        # The stub cannot make `_rs.CompilerPassBase` generic.
        @override
        def execute(self, ir: _PassInputT) -> PassResult[_PassOutputT]:
            """Run the pass over ``ir`` through its lifecycle."""
            ...

        @override
        def __call__(self, ir: _PassInputT) -> _PassOutputT:
            """Run the pass over ``ir`` and return only its output."""
            ...

        @override
        def get_analysis(
            self,
            analysis_type: "type[Analysis[_AnalysisIRT, _AnalysisResultT]]",
            ir: _AnalysisIRT,
        ) -> _AnalysisResultT:
            """Return the result of ``analysis_type`` for ``ir``."""
            ...

        @override
        def get_analysis_manager(self) -> "AnalysisManager[Any] | None":
            """Return the analyses of the running hook, or ``None``."""
            ...

    def report(
        self,
        level: DiagnosticLevel,
        message: str | Note,
        detail: str | None = None,
        *,
        exc_info: BaseException | bool | None = None,
    ) -> None:
        """Emit a diagnostic for this pass execution.

        During a hook, the diagnostic is recorded in the running pass's
        context, so it appears in the run's `PassResult` or error, as this
        object. Outside a run it is recorded on the pass, where the next run
        forgets it. It is also logged on the pass's logger,
        ``fhy_core.pass_infrastructure.core.<pass-name>``, at the matching
        level, with ``" | detail: <detail>"`` appended when ``detail`` is
        given and ``exc_info`` forwarded.

        Raises:
            RuntimeError: In `did_change` or `get_preserved_analyses`, which
                run without a context.

        """
        note = message if isinstance(message, Note) else Note(message)
        diagnostic = Diagnostic(
            level=level, message=note, source=self.get_pass_name(), detail=detail
        )
        self._record_diagnostic(diagnostic)
        _log_diagnostic(diagnostic.source, level, note.message, detail, exc_info)

    @classmethod
    def _get_pass_logger(cls) -> logging.Logger:
        return _get_pass_logger_by_name(cls.get_pass_name())

    def validate_input(self, ir: _PassInputT) -> None:
        """Validate input IR before execution; by default, accept it."""

    def should_run(self, ir: _PassInputT) -> bool:
        """Return whether this pass should run for the input IR."""
        return True

    def get_noop_output(self, ir: _PassInputT) -> _PassOutputT:
        """Return the output of a run `should_run` skips.

        A pass whose `should_run` can be false overrides this; the default
        raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f'Pass "{self.get_pass_name()}" does not define get_noop_output, '
            "which a skipped run needs."
        )

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
            get_logger(__name__).debug(
                "registered %s -> %s", name, pass_cls.__qualname__
            )

        return pass_cls

    return _decorator
