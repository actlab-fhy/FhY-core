"""Compiler pass infrastructure package."""

from .core import (
    CompilerPass,
    DiagnosticLevel,
    PassDiagnostic,
    PassExecutionError,
    PassInfo,
    PassRegistrationError,
    PassResult,
    PassValidationError,
    PreservedAnalyses,
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

__all__ = [
    "Analysis",
    "AnalysisManager",
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
    "VisitablePass",
    "register_pass",
]
