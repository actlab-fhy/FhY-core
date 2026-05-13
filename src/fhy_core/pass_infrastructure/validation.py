"""Validation pipeline infrastructure built on `CompilerPass[IR, None]`.

`ValidationManager` is a sibling to `PassManager`: it sequences validation
passes and aggregates every diagnostic they emit into a single
`ValidationReport[PassRunRecord]`.

Unlike `PassManager`, which is fail-fast and shaped around `IR -> IR`
transformations, `ValidationManager`:

- accepts any `CompilerPass[IR, Any]` whose purpose is to `report(...)`
  diagnostics (no transformation semantics are assumed),
- runs every validator even when earlier ones produce errors (collect-all),
- wraps an unexpected validator exception into an ERROR diagnostic and
  continues with the next validator.

"""

__all__ = [
    "ValidationManager",
]

from typing import Any, Generic, TypeVar

from fhy_core.diagnostic import (
    Diagnostic,
    DiagnosticLevel,
    Note,
    ValidationReport,
)
from fhy_core.identifier import Identifier
from fhy_core.logger import get_logger
from fhy_core.trait import HasIdentifierMixin

from .core import (
    CompilerPass,
    PassExecutionError,
    PassValidationError,
    PreservedAnalyses,
)
from .manager import PassRunRecord

_LOGGER = get_logger(__name__)

_IRType = TypeVar("_IRType")


class ValidationManager(HasIdentifierMixin, Generic[_IRType]):
    """Sequences validation passes and aggregates their diagnostics.

    Every registered validator runs against the input IR once, regardless of
    whether earlier validators produced ERROR diagnostics. If a validator's
    ``execute`` raises ``PassValidationError`` or ``PassExecutionError``, the
    already-emitted diagnostics are still captured and the pipeline proceeds
    to the next validator. Any other exception (an unexpected crash inside a
    validator) is itself turned into an ERROR diagnostic attributed to that
    validator, and the pipeline proceeds.

    Unlike :class:`PassManager`, ``ValidationManager`` does **not** provide
    an :class:`AnalysisManager` to its validators. Each validator runs
    standalone; ``self.get_analysis(...)`` inside a validator recomputes
    its result on every call. Validators that need to share an expensive
    analysis must either compute it themselves once and pass it through
    state, or be sequenced inside a :class:`PassManager` instead.

    """

    _validators: list[CompilerPass[_IRType, Any]]
    _name: Identifier

    def __init__(self, name: Identifier | None = None) -> None:
        self._name = name if name is not None else Identifier("validation-pipeline")
        self._validators = []

    def get_identifier(self) -> Identifier:
        return self._name

    @property
    def name(self) -> Identifier:
        return self._name

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

    def validate(self, ir: _IRType) -> ValidationReport[PassRunRecord]:
        """Run every validator and return the aggregated report.

        Args:
            ir: The IR to validate.

        Returns:
            A :class:`ValidationReport` whose ``diagnostics`` are the
            concatenation of every validator's diagnostics, in pipeline
            order, and whose ``records`` carry one
            :class:`PassRunRecord` per validator.

        """
        aggregated_diagnostics: list[Diagnostic] = []
        records: list[PassRunRecord] = []

        _LOGGER.info(
            "%s starting (validators=%d)",
            self._name,
            len(self._validators),
        )

        for validator in self._validators:
            diagnostics = self._run_single_validator(validator, ir)
            aggregated_diagnostics.extend(diagnostics)
            records.append(
                PassRunRecord(
                    pass_name=validator.get_pass_name(),
                    changed=False,
                    diagnostics=tuple(diagnostics),
                    preserved_analyses=PreservedAnalyses.all(),
                )
            )

        report: ValidationReport[PassRunRecord] = ValidationReport(
            diagnostics=tuple(aggregated_diagnostics),
            records=tuple(records),
        )
        error_count = 0
        warning_count = 0
        info_count = 0
        for diagnostic in report.diagnostics:
            if diagnostic.level == DiagnosticLevel.ERROR:
                error_count += 1
            elif diagnostic.level == DiagnosticLevel.WARNING:
                warning_count += 1
            elif diagnostic.level == DiagnosticLevel.INFO:
                info_count += 1
        _LOGGER.info(
            "%s finished (errors=%d, warnings=%d, infos=%d)",
            self._name,
            error_count,
            warning_count,
            info_count,
        )
        return report

    @staticmethod
    def _run_single_validator(
        validator: CompilerPass[_IRType, Any], ir: _IRType
    ) -> tuple[Diagnostic, ...]:
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
            type(validator)._get_pass_logger().error(
                "raised %s without reporting a diagnostic: %s",
                type(exc).__name__,
                exc,
                exc_info=exc,
            )
            return (
                *captured,
                Diagnostic(
                    level=DiagnosticLevel.ERROR,
                    message=Note(
                        f'Validator "{validator.get_pass_name()}" raised '
                        f'"{type(exc).__name__}" without reporting a diagnostic: '
                        f"{exc}"
                    ),
                    source=validator.get_pass_name(),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - defense-in-depth
            captured = tuple(validator.diagnostics)
            type(validator)._get_pass_logger().error(
                "validator crashed with %s: %s",
                type(exc).__name__,
                exc,
                exc_info=exc,
            )
            synthesized = Diagnostic(
                level=DiagnosticLevel.ERROR,
                message=Note(
                    f'Validator "{validator.get_pass_name()}" crashed with '
                    f"{type(exc).__name__}: {exc}"
                ),
                source=validator.get_pass_name(),
            )
            return (*captured, synthesized)
