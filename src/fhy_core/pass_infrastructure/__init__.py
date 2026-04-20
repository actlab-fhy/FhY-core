"""Compiler pass infrastructure package."""

from .core import (
    AnalysisVisitablePass,
    CompilerPass,
    DiagnosticLevel,
    PassDiagnostic,
    PassExecutionError,
    PassInfo,
    PassRegistrationError,
    PassResult,
    PassValidationError,
    PreservedAnalyses,
    TraversalOrder,
    VisitablePass,
    register_pass,
)
from .manager import (
    Analysis,
    AnalysisManager,
    FixpointGroupRecord,
    FixpointIterationRecord,
    FixpointPassGroup,
    PassManager,
    PassManagerResult,
    PassRunRecord,
)
from .validation import (
    ValidationFailedError,
    ValidationManager,
    ValidationReport,
)

__all__ = [
    "Analysis",
    "AnalysisManager",
    "AnalysisVisitablePass",
    "CompilerPass",
    "DiagnosticLevel",
    "FixpointGroupRecord",
    "FixpointIterationRecord",
    "FixpointPassGroup",
    "PassDiagnostic",
    "PassExecutionError",
    "PassInfo",
    "PassManager",
    "PassManagerResult",
    "PassRegistrationError",
    "PassResult",
    "PassRunRecord",
    "PassValidationError",
    "PreservedAnalyses",
    "TraversalOrder",
    "ValidationFailedError",
    "ValidationManager",
    "ValidationReport",
    "VisitablePass",
    "register_pass",
]
