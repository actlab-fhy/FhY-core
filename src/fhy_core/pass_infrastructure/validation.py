"""Validation pipeline infrastructure built on `CompilerPass[IR, None]`.

`ValidationManager` is a sibling to `PassManager`: it sequences validation
passes and aggregates every diagnostic they emit into a single
`ValidationReport`.

Unlike `PassManager`, which is fail-fast and shaped around `IR -> IR`
transformations, `ValidationManager`:

- accepts any `CompilerPass[IR, Any]` whose purpose is to `report(...)`
  diagnostics (no transformation semantics are assumed),
- runs every validator even when earlier ones produce errors (collect-all),
- wraps an unexpected validator exception into an ERROR diagnostic and
  continues with the next validator.

"""

__all__ = [
    "ValidationFailedError",
    "ValidationManager",
    "ValidationReport",
]

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from fhy_core.error import register_error
from fhy_core.identifier import Identifier
from fhy_core.provenance import Note
from fhy_core.trait import FrozenMixin, HasIdentifierMixin, PartialEqualMixin

from .core import (
    CompilerPass,
    DiagnosticLevel,
    PassDiagnostic,
    PassExecutionError,
    PassValidationError,
)
from .manager import PassRunRecord

_IRType = TypeVar("_IRType")


@register_error
class ValidationFailedError(RuntimeError):
    """Raised when a validation pipeline produced one or more ERROR diagnostics.

    The associated :class:`ValidationReport` is available via
    :attr:`ValidationFailedError.report`. The exception message is the
    formatted report, so uncaught exceptions print a useful summary.
    """

    _report: "ValidationReport"

    def __init__(self, report: "ValidationReport") -> None:
        super().__init__(report.format())
        self._report = report

    @property
    def report(self) -> "ValidationReport":
        """Return the validation report that triggered this failure."""
        return self._report


@dataclass(frozen=True)
class ValidationReport(FrozenMixin, PartialEqualMixin):
    """Aggregated result of running a :class:`ValidationManager` pipeline.

    Attributes:
        diagnostics: Every diagnostic, across every validator, in the order
            each validator emitted them.
        records: Per-validator execution metadata, one entry per registered
            validator, in pipeline order.

    """

    diagnostics: tuple[PassDiagnostic, ...] = field(default_factory=tuple)
    records: tuple[PassRunRecord, ...] = field(default_factory=tuple)

    def errors(self) -> tuple[PassDiagnostic, ...]:
        """Return only the ERROR-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.ERROR)

    def warnings(self) -> tuple[PassDiagnostic, ...]:
        """Return only the WARNING-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.WARNING)

    def infos(self) -> tuple[PassDiagnostic, ...]:
        """Return only the INFO-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.level == DiagnosticLevel.INFO)

    def has_errors(self) -> bool:
        """Return True when at least one ERROR-level diagnostic is present."""
        return any(d.level == DiagnosticLevel.ERROR for d in self.diagnostics)

    def format(self) -> str:
        """Return a human-readable rendering of every diagnostic.

        Each diagnostic is rendered on its own line as
        ``[LEVEL] <pass-name>: <message>``; optional detail is appended on an
        indented continuation line.
        """
        if not self.diagnostics:
            return "No validation diagnostics."
        lines: list[str] = []
        for diagnostic in self.diagnostics:
            prefix = f"[{diagnostic.level.value.upper()}] {diagnostic.pass_name}: "
            body = diagnostic.message_text
            lines.append(f"{prefix}{body}")
            if diagnostic.detail:
                lines.append(f"    detail: {diagnostic.detail}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise :class:`ValidationFailedError` if any ERROR diagnostics exist.

        No-op when the report contains only warnings/infos or nothing at all.
        """
        if self.has_errors():
            raise ValidationFailedError(self)


class ValidationManager(HasIdentifierMixin, Generic[_IRType]):
    """Sequences validation passes and aggregates their diagnostics.

    Every registered validator runs against the input IR once, regardless of
    whether earlier validators produced ERROR diagnostics. If a validator's
    ``execute`` raises ``PassValidationError`` or ``PassExecutionError``, the
    already-emitted diagnostics are still captured and the pipeline proceeds
    to the next validator. Any other exception (an unexpected crash inside a
    validator) is itself turned into an ERROR diagnostic attributed to that
    validator, and the pipeline proceeds.

    """

    _validators: list[CompilerPass[_IRType, Any]]
    _identifier: Identifier

    def __init__(self, name: Identifier | None = None) -> None:
        self._identifier = (
            name if name is not None else Identifier("validation-pipeline")
        )
        self._validators = []

    def get_identifier(self) -> Identifier:
        return self._identifier

    @property
    def name(self) -> Identifier:
        return self._identifier

    @property
    def validators(self) -> tuple[CompilerPass[_IRType, Any], ...]:
        """Return the registered validators in pipeline order."""
        return tuple(self._validators)

    def add(self, validator: CompilerPass[_IRType, Any]) -> None:
        """Append a validator to the pipeline.

        Args:
            validator: The validation pass to register. Typically an
                :class:`~fhy_core.pass_infrastructure.core.AnalysisVisitablePass`
                subclass that calls ``self.report(...)`` to emit diagnostics.

        """
        self._validators.append(validator)

    def validate(self, ir: _IRType) -> ValidationReport:
        """Run every validator and return the aggregated report.

        Args:
            ir: The IR to validate.

        Returns:
            A :class:`ValidationReport` whose ``diagnostics`` are the
            concatenation of every validator's diagnostics, in pipeline
            order.

        """
        aggregated_diagnostics: list[PassDiagnostic] = []
        records: list[PassRunRecord] = []

        for validator in self._validators:
            diagnostics = self._run_single_validator(validator, ir)
            aggregated_diagnostics.extend(diagnostics)
            records.append(
                PassRunRecord(
                    pass_name=validator.get_pass_name(),
                    changed=False,
                    diagnostics=tuple(diagnostics),
                    preserved_analyses=(),
                    preserves_all_analyses=True,
                )
            )

        return ValidationReport(
            diagnostics=tuple(aggregated_diagnostics),
            records=tuple(records),
        )

    @staticmethod
    def _run_single_validator(
        validator: CompilerPass[_IRType, Any], ir: _IRType
    ) -> tuple[PassDiagnostic, ...]:
        """Execute one validator and return its captured diagnostics.

        Never raises. Converts an unexpected validator crash into a synthetic
        ERROR diagnostic so the pipeline can continue.
        """
        try:
            result = validator.execute(ir)
            return tuple(result.diagnostics)
        except (PassValidationError, PassExecutionError) as exc:
            captured = tuple(validator.diagnostics)
            if any(d.level == DiagnosticLevel.ERROR for d in captured):
                return captured
            # Infrastructure raised without emitting an ERROR; synthesize one
            # so the report accurately reflects the failure.
            return (
                *captured,
                PassDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    message=Note(
                        f'Validator "{validator.get_pass_name()}" raised '
                        f'"{type(exc).__name__} without reporting a diagnostic: '
                        f"{exc}"
                    ),
                    pass_name=validator.get_pass_name(),
                ),
            )
        except Exception as exc:  # noqa: BLE001 — defense-in-depth
            captured = tuple(validator.diagnostics)
            synthesized = PassDiagnostic(
                level=DiagnosticLevel.ERROR,
                message=Note(
                    f'Validator "{validator.get_pass_name()}" crashed with '
                    f"{type(exc).__name__}: {exc}"
                ),
                pass_name=validator.get_pass_name(),
            )
            return (*captured, synthesized)
