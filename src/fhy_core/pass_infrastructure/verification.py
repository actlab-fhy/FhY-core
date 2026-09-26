"""Verification registry, analysis, and pipeline-verification helpers.

Exposes:

- :class:`VerificationRegistry`: type-keyed registry of verification
  pass classes.
- :class:`VerificationAnalysis`: an :class:`Analysis` that runs the
  registered passes for an IR and returns a :class:`ValidationReport`.
- :func:`register_verification`: decorator that registers a
  ``CompilerPass`` subclass as a verification pass.
- :func:`run_verification`: runs the registered passes for an IR; the
  default :meth:`fhy_core.traits.VerifiableMixin.verify` does the same.

Verification passes are ``CompilerPass`` subclasses whose only effect
is to ``report(...)`` structural diagnostics about an IR. They run as
checks, as a :class:`ValidationManager` runs them, collect-all. A
:class:`PassManager` verifies its input and every changed output with
the passes registered for the IR's type, unless its verifier is replaced;
a verification pass runs as a check, so it never verifies anything
itself.

The registry is the Rust core's ``fhy_core::pass::VerificationRegistry``,
held in the extension's module state (S14 of
``docs/design/python-switch.md``). It keys registrations by IR type and
looks them up along ``reversed(type(ir).__mro__)``. The classes here are
thin layers over ``fhy_core._rs``.
"""

from fhy_core.utils.override import override

__all__ = [
    "VerificationAnalysis",
    "VerificationRegistry",
    "register_verification",
    "run_verification",
]

from collections.abc import Callable
from typing import Any, TypeVar

from fhy_core import _rs
from fhy_core.diagnostic import ValidationReport
from fhy_core.logger import get_logger

from .core import CompilerPass, PassRegistrationError, register_pass
from .manager import Analysis

_LOGGER = get_logger(__name__)

_PassClassT = TypeVar("_PassClassT", bound=type[CompilerPass[Any, Any]])


class VerificationRegistry:
    """Registry of verification pass classes, keyed by IR type.

    A verification pass is a ``CompilerPass`` subclass that reports
    structural-invariant diagnostics about an IR. Registrations are
    keyed by IR ``type``; lookups walk the IR's MRO so subclasses
    inherit base-class verification passes.

    The class has no instances and holds no state: the one registry is
    the Rust core's, in the extension's module state, and every method is
    a classmethod over it. It keeps the registered types and classes
    alive.

    Registration is module-load-time by convention. Late registration
    during a pipeline run does not invalidate the :class:`VerificationAnalysis`
    results that run cached already.
    """

    @classmethod
    def register(
        cls,
        ir_type: type,
        pass_class: type[CompilerPass[Any, Any]],
    ) -> None:
        """Register one verification pass for an IR type.

        Idempotent: re-registering the same ``(ir_type, pass_class)``
        pair is a no-op. Distinct pass classes registered under the same
        IR type are appended in registration order.

        Args:
            ir_type: The IR type this verification pass applies to.
            pass_class: A ``CompilerPass`` subclass. Typically an
                :class:`AnalysisVisitablePass` subclass whose visitor
                methods call ``self.report(...)``.

        Raises:
            TypeError: If ``ir_type`` is not a type.
            PassRegistrationError: If ``pass_class`` is not a
                ``CompilerPass`` subclass.

        """
        is_new = _rs.register_verification_pass(ir_type, pass_class)
        _LOGGER.debug(
            "registered verification pass %s for %s"
            if is_new
            else "verification pass %s already registered for %s (idempotent)",
            pass_class.__qualname__,
            ir_type.__qualname__,
        )

    @classmethod
    def get_passes_for(cls, ir_type: type) -> tuple[type[CompilerPass[Any, Any]], ...]:
        """Return all verification passes applicable to ``ir_type``.

        Walks ``reversed(ir_type.__mro__)`` and concatenates the lists in
        base-to-derived order, so base-class passes appear first and
        subclass passes appear last. Duplicate pass classes (e.g.,
        registered against both a base and a subclass) appear exactly
        once, at their first position in that reverse-MRO traversal.

        Args:
            ir_type: The IR type to look up.

        Returns:
            A tuple of pass classes in execution order, possibly empty.

        Raises:
            TypeError: If ``ir_type`` is not a type.

        """
        return _rs.get_verification_passes_for(ir_type)


class VerificationAnalysis(Analysis[Any, ValidationReport[Any]]):
    """Analysis that runs the verification passes for an IR.

    Runs a new instance of each class
    :meth:`VerificationRegistry.get_passes_for(type(ir))` returns as a
    check against the IR, as a :class:`ValidationManager` would. The
    aggregated :class:`ValidationReport`, with one record per pass, is the
    analysis result. When no passes are registered for the IR's type (or
    any of its base classes), the report is empty.

    Has a stable ``analysis_name``, so a pipeline run caches its result
    per IR node, and carries it to a pass's output when the pass preserves
    it.
    """

    @override
    def run(self, ir: Any) -> ValidationReport[Any]:
        """Run every registered verification pass for ``type(ir)``.

        A pass class whose construction raises fails its check, and the
        other passes still run.

        Args:
            ir: The IR to verify.

        Returns:
            The aggregated :class:`ValidationReport`.

        """
        return _rs.run_verification(ir)


def register_verification(
    ir_type: type,
    name: str,
    description: str,
) -> Callable[[_PassClassT], _PassClassT]:
    """Register a ``CompilerPass`` subclass as a verification pass.

    The decorated class is:

    - added to the global pass registry under ``name`` and
      ``description``,
    - added to the :class:`VerificationRegistry` under ``ir_type``.

    Args:
        ir_type: The IR type this verification pass applies to.
        name: Stable pass name, with the same uniqueness and validity
            rules as :func:`register_pass`.
        description: Human-readable description, with the same rules as
            :func:`register_pass`.

    Returns:
        A decorator that takes a ``CompilerPass`` subclass and returns
        it after registration.

    Raises:
        PassRegistrationError: If the decorated class is not a
            ``CompilerPass`` subclass, if the name is empty or already
            registered to a different class, or if the description is
            empty or conflicts with an existing registration.

    """

    def _decorator(pass_class: _PassClassT) -> _PassClassT:
        if not (isinstance(pass_class, type) and issubclass(pass_class, CompilerPass)):
            raise PassRegistrationError(
                f"Cannot register non-CompilerPass type as a verification pass: "
                f"{getattr(pass_class, '__qualname__', repr(pass_class))}."
            )
        register_pass(name, description)(pass_class)
        VerificationRegistry.register(ir_type, pass_class)
        return pass_class

    return _decorator


def run_verification(ir: Any) -> ValidationReport[Any]:
    """Run the verification pipeline for ``ir`` and return the report.

    Computes the report afresh on every call. A pass class whose
    construction raises fails its check, and the other passes still run.

    Args:
        ir: The IR to verify.

    Returns:
        The aggregated :class:`ValidationReport`. Empty when no passes
        are registered for ``type(ir)`` or any of its base classes.

    """
    return _rs.run_verification(ir)
