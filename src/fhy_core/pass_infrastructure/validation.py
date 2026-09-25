"""Validators and the collect-all validation pipeline.

A `Validator` checks IR and reports every problem it finds as a
diagnostic; it raises only when it cannot finish its check. A
`ValidationManager` runs validators, and passes as checks, into one
`ValidationReport` of `ValidatorRecord`s. Both are backed by the Rust
implementation (``fhy_core._rs``), with the Rust core's semantics:

- every validator runs, whatever the earlier ones reported;
- a pass runs as a check: ``validate_input``, ``should_run`` (and
  ``get_noop_output``), ``run_pass``, then ``validate_output``; a hook that
  fails adds the error diagnostic ``pass "X" failed in <hook>: <cause>``;
- a `Validator` whose ``validate`` raises without reporting an error gains
  ``validator "X" failed without reporting an error: <cause>``.

Run on its own, a validation computes the analyses it requests afresh; as a
pipeline's verifier, its validators share the pipeline's analysis cache.
"""

from fhy_core.utils.override import override

__all__ = [
    "ValidationManager",
    "Validator",
    "ValidatorRecord",
]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from fhy_core import _rs
from fhy_core.diagnostic import Diagnostic, DiagnosticLevel, Note, ValidationReport
from fhy_core.logger import get_logger
from fhy_core.traits import FrozenMixin, PartialEqualMixin

from .core import CompilerPass, _log_diagnostic

if TYPE_CHECKING:
    from .manager import Analysis, AnalysisManager

_LOGGER = get_logger(__name__)

_IRType = TypeVar("_IRType")
_AnalysisIRT = TypeVar("_AnalysisIRT")
_AnalysisResultT = TypeVar("_AnalysisResultT")


class ValidatorRecord(_rs.ValidatorRecord, PartialEqualMixin):
    """The record of one validator in a `ValidationReport`.

    Backed by the Rust implementation: ``fhy_core._rs.ValidatorRecord``.
    Records are immutable, compare, hash and print as frozen dataclasses do,
    and pickle as a call of their class.

    Attributes:
        validator_name: The validator's name.
        failed: Whether the validator could not finish its check.
        diagnostics: The validator's diagnostics: its slice of the report's
            diagnostic objects.

    """

    __slots__ = ()
    __match_args__ = ("validator_name", "failed", "diagnostics")


FrozenMixin.register(ValidatorRecord)
ValidatorRecord._register_public_class()


class Validator(_rs.ValidatorBase, ABC, Generic[_IRType]):
    """A collect-all check over IR.

    Backed by the Rust implementation: ``fhy_core._rs.ValidatorBase``. A
    subclass implements `validate`, which reports each problem with
    `report`; it raises only when it cannot finish the check. `name`
    defaults to the class's ``__name__``, and is the source of the
    validator's diagnostics.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        """Return the validator's name."""
        return type(self).__name__

    @abstractmethod
    def validate(self, ir: _IRType) -> None:
        """Check ``ir``, reporting every problem found with `report`.

        Args:
            ir: The IR to check.

        """

    def report(
        self,
        level: DiagnosticLevel,
        message: str | Note,
        detail: str | None = None,
        *,
        exc_info: BaseException | bool | None = None,
    ) -> None:
        """Report a diagnostic of the running check, and log it.

        The diagnostic's source is `name`. It is logged on the logger
        ``fhy_core.pass_infrastructure.core.<name>``, as a pass's is.
        """
        note = message if isinstance(message, Note) else Note(message)
        diagnostic = Diagnostic(
            level=level, message=note, source=self.name, detail=detail
        )
        self._record_diagnostic(diagnostic)
        _log_diagnostic(diagnostic.source, level, note.message, detail, exc_info)

    if TYPE_CHECKING:
        # The stub cannot make `_rs.ValidatorBase` generic.
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
            """Return the analyses of the running check, or ``None``."""
            ...


class ValidationManager(_rs.ValidationManager, Generic[_IRType]):
    """Sequences validators and aggregates their diagnostics.

    Backed by the Rust implementation: ``fhy_core._rs.ValidationManager``.
    ``ValidationManager(name=None)`` names the pipeline
    ``validation-pipeline`` by default. `add` takes a `Validator` or a
    `CompilerPass`, and `validate` returns a `ValidationReport` of one
    `ValidatorRecord` per validator. It logs INFO lines when it starts and
    finishes.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @property
        @override
        def validators(
            self,
        ) -> tuple["Validator[_IRType] | CompilerPass[_IRType, Any]", ...]:
            """The validators, in pipeline order."""
            ...

        @override
        def add(
            self, validator: "Validator[_IRType] | CompilerPass[_IRType, Any]"
        ) -> None:
            """Append a validator to the pipeline."""
            ...

    @override
    def validate(self, ir: _IRType) -> ValidationReport[ValidatorRecord]:
        """Run every validator and return the aggregated report.

        Args:
            ir: The IR to validate.

        Returns:
            A :class:`ValidationReport` whose ``diagnostics`` are the
            concatenation of every validator's diagnostics, in pipeline
            order, and whose ``records`` carry one :class:`ValidatorRecord`
            per validator.

        """
        _LOGGER.info("%s starting (validators=%d)", self.name, len(self.validators))
        report: ValidationReport[ValidatorRecord] = super().validate(ir)
        if _LOGGER.isEnabledFor(logging.INFO):
            _LOGGER.info(
                "%s finished (errors=%d, warnings=%d, infos=%d)",
                self.name,
                len(report.errors()),
                len(report.warnings()),
                len(report.infos()),
            )
        return report
