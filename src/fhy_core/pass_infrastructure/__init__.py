"""Compiler pass infrastructure package."""

from .core import (
    AnalysisVisitablePass,
    CompilerPass,
    PassExecutionError,
    PassInfo,
    PassRegistrationError,
    PassResult,
    PassValidationError,
    PreservedAnalyses,
    RewritablePass,
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
from .validation import ValidationManager
from .verification import (
    VerificationAnalysis,
    VerificationRegistry,
    register_verification,
    run_verification,
)

__all__ = [
    "Analysis",
    "AnalysisManager",
    "AnalysisVisitablePass",
    "CompilerPass",
    "FixpointGroupRecord",
    "FixpointIterationRecord",
    "FixpointPassGroup",
    "PassExecutionError",
    "PassInfo",
    "PassManager",
    "PassManagerResult",
    "PassRegistrationError",
    "PassResult",
    "PassRunRecord",
    "PassValidationError",
    "PreservedAnalyses",
    "RewritablePass",
    "TraversalOrder",
    "ValidationManager",
    "VerificationAnalysis",
    "VerificationRegistry",
    "VisitablePass",
    "register_pass",
    "register_verification",
    "run_verification",
]
