"""Generic compiler pass infrastructure."""

from __future__ import annotations

__all__ = [
    "CompilerPass",
    "DiagnosticLevel",
    "PassInfo",
    "PassDiagnostic",
    "PassExecutionError",
    "PassRegistrationError",
    "PassResult",
    "PassValidationError",
    "register_pass",
]

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, ClassVar, Generic, Mapping, TypeVar

from fhy_core.error import register_error
from fhy_core.provenance import Note
from fhy_core.utils.enum import StrEnum

PassInputT = TypeVar("PassInputT")
PassOutputT = TypeVar("PassOutputT")
PassClassT = TypeVar("PassClassT", bound=type["CompilerPass[Any, Any]"])


class DiagnosticLevel(StrEnum):
    """Diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class PassDiagnostic:
    """Structured diagnostic emitted by a pass."""

    level: DiagnosticLevel
    message: Note
    pass_name: str
    detail: str | None = None

    @property
    def message_text(self) -> str:
        """Return the raw message text for convenience."""
        return self.message.message


@dataclass(frozen=True)
class PassInfo:
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
class PassResult(Generic[PassOutputT]):
    """Result of a pass execution."""

    output: PassOutputT
    changed: bool
    diagnostics: tuple[PassDiagnostic, ...] = ()


class CompilerPass(ABC, Generic[PassInputT, PassOutputT]):
    """Base class for standardized compiler passes."""

    _registry: ClassVar[dict[str, PassInfo]] = {}
    _run_counts: ClassVar[dict[str, int]] = {}
    _total_run_count: ClassVar[int] = 0
    _registry_lock: ClassVar[Lock] = Lock()

    _pass_name: ClassVar[str | None] = None
    _pass_description: ClassVar[str] = ""
    _diagnostics: list[PassDiagnostic]

    def __init__(self) -> None:
        self._diagnostics = []

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
        """Create an instance from the global pass registry.

        Args:
            pass_name: The name of the pass to create.
            *args: Positional arguments to forward to the pass constructor.
            **kwargs: Keyword arguments to forward to the pass constructor.

        Returns:
            An instance of the requested pass.

        Raises:
            PassRegistrationError: If the pass name is not registered.

        """
        with CompilerPass._registry_lock:
            pass_info = CompilerPass._registry.get(pass_name)
        if pass_info is None:
            raise PassRegistrationError(f'Unknown pass "{pass_name}".')
        return pass_info.pass_type(*args, **kwargs)

    @property
    def diagnostics(self) -> tuple[PassDiagnostic, ...]:
        return tuple(self._diagnostics)

    def __call__(self, ir: PassInputT) -> PassOutputT:
        return self.execute(ir).output

    def execute(self, ir: PassInputT) -> PassResult[PassOutputT]:
        """Execute the pass with validation and standardized error handling.

        Args:
            ir: The input IR to process.

        Returns:
            A `PassResult` containing the output, change status, and diagnostics.

        Raises:
            PassValidationError: If input validation fails.
            PassExecutionError: If an error occurs during execution.

        """
        self._diagnostics = []
        self.validate_input(ir)

        self._record_run()
        if not self.should_run(ir):
            output = self.noop_output(ir)
            return PassResult(
                output,
                False,
                diagnostics=tuple(self._diagnostics),
            )

        try:
            output = self.run_pass(ir)
        except (PassValidationError, PassExecutionError):
            raise
        except Exception as exc:
            message = (
                f'Pass "{self.get_pass_name()}" failed with '
                f"{type(exc).__name__}: {exc}"
            )
            self.report(DiagnosticLevel.ERROR, message)
            raise PassExecutionError(message) from exc

        self.validate_output(ir, output)
        return PassResult(
            output=output,
            changed=self.did_change(ir, output),
            diagnostics=tuple(self._diagnostics),
        )

    def report(
        self, level: DiagnosticLevel, message: str | Note, detail: str | None = None
    ) -> None:
        """Emit a diagnostic for this pass execution.

        Args:
            level: The severity level of the diagnostic.
            message: The diagnostic message, either as a raw string or a Note.
            detail: Optional additional detail to include with the diagnostic.

        """
        note = message if isinstance(message, Note) else Note(message)
        self._diagnostics.append(
            PassDiagnostic(
                level,
                note,
                self.get_pass_name(),
                detail=detail,
            )
        )

    def validate_input(self, ir: PassInputT) -> None:
        """Validate input IR before execution.

        Args:
            ir: The input IR to validate.

        """
        if ir is None:
            message = f'Pass "{self.get_pass_name()}" does not accept None input.'
            self.report(DiagnosticLevel.ERROR, message)
            raise PassValidationError(message)

    def should_run(self, ir: PassInputT) -> bool:
        """Return whether this pass should run for the input IR."""
        return True

    def noop_output(self, ir: PassInputT) -> PassOutputT:
        """Return output when pass execution is skipped."""
        return ir  # type: ignore[return-value]

    @abstractmethod
    def run_pass(self, ir: PassInputT) -> PassOutputT:
        """Run the pass over IR after validation.

        Args:
            ir: The input IR to process.

        Returns:
            The output after processing.

        Raises:
            PassExecutionError: If an error occurs during execution.

        """

    def validate_output(self, input_ir: PassInputT, output: PassOutputT) -> None:
        """Validate output after execution.

        Args:
            input_ir: The original input IR.
            output: The output to validate.

        """

    def did_change(self, input_ir: PassInputT, output: PassOutputT) -> bool:
        """Return whether output differs from input."""
        return input_ir is not output

    def _record_run(self) -> None:
        pass_name = self.get_pass_name()
        with CompilerPass._registry_lock:
            CompilerPass._total_run_count += 1
            CompilerPass._run_counts[pass_name] = (
                CompilerPass._run_counts.get(pass_name, 0) + 1
            )


def register_pass(name: str, description: str) -> Callable[[PassClassT], PassClassT]:
    """Register a concrete pass class with explicit metadata.

    Args:
        name: A stable name for the pass, used for registration and reporting.
        description: A human-readable description of the pass for discovery
            and reporting.

    Returns:
        A class decorator that registers the pass.

    """
    if not name.strip():
        raise PassRegistrationError("Pass name cannot be empty.")
    if not description.strip():
        raise PassRegistrationError("Pass description cannot be empty.")

    def _decorator(pass_cls: PassClassT) -> PassClassT:
        if not issubclass(pass_cls, CompilerPass):
            raise PassRegistrationError(
                f"Cannot register non-CompilerPass type: {pass_cls.__qualname__}."
            )
        if inspect.isabstract(pass_cls):
            raise PassRegistrationError(
                f'Cannot register abstract pass "{pass_cls.__qualname__}".'
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
            CompilerPass._registry[name] = PassInfo(
                name,
                description,
                pass_cls,
            )
            CompilerPass._run_counts.setdefault(name, 0)

        return pass_cls

    return _decorator
