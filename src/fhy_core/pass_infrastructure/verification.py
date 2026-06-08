"""Verification registry, analysis, and auto-verification helpers.

Exposes:

- :class:`VerificationRegistry`: type-keyed registry of verification
  pass classes.
- :class:`VerificationAnalysis`: an :class:`Analysis` that runs the
  registered pipeline for an IR and returns a
  :class:`ValidationReport`.
- :func:`register_verification`: decorator that registers a
  ``CompilerPass`` subclass as a verification pass.
- :func:`run_verification`: helper used by
  :meth:`fhy_core.traits.VerifiableMixin.verify` and by
  :class:`CompilerPass` auto-verification when no analysis manager is
  bound.

Verification passes are ``CompilerPass`` subclasses whose only effect
is to ``report(...)`` structural diagnostics about an IR. They run
through :class:`ValidationManager`, inheriting its collect-all and
synthetic-error-on-crash behavior.

:class:`VerificationRegistry` is a class-level singleton: all state
lives in ``ClassVar`` attributes, there are no instances, and every
method is a classmethod.
"""

from fhy_core.utils.override import override

__all__ = [
    "VerificationAnalysis",
    "VerificationRegistry",
    "register_verification",
    "run_verification",
]

from threading import Lock
from typing import Any, Callable, ClassVar, TypeVar

from fhy_core.diagnostic import ValidationReport
from fhy_core.logger import get_logger

from .core import CompilerPass, PassRegistrationError, register_pass
from .manager import Analysis
from .validation import ValidationManager

_LOGGER = get_logger(__name__)

_PassClassT = TypeVar("_PassClassT", bound=type[CompilerPass[Any, Any]])


class VerificationRegistry:
    """Class-level singleton registry of verification pass classes.

    A verification pass is a ``CompilerPass`` subclass that reports
    structural-invariant diagnostics about an IR. Registrations are
    keyed by IR ``type``; lookups walk the IR's MRO so subclasses
    inherit base-class verification passes.

    All state lives in ``ClassVar`` attributes; there are no instances.
    The class itself is the registry.

    Registration is module-load-time by convention. Late registration
    after cached :class:`VerificationAnalysis` results exist does not
    retroactively invalidate those caches; callers who need that must
    clear the relevant :class:`AnalysisManager` themselves.
    """

    _passes_by_type: ClassVar[dict[type, list[type[CompilerPass[Any, Any]]]]] = {}
    _lock: ClassVar[Lock] = Lock()

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
            PassRegistrationError: If ``pass_class`` is not a
                ``CompilerPass`` subclass.

        """
        if not (isinstance(pass_class, type) and issubclass(pass_class, CompilerPass)):
            raise PassRegistrationError(
                f"Cannot register non-CompilerPass type as a verification pass: "
                f"{getattr(pass_class, '__qualname__', repr(pass_class))}."
            )
        with cls._lock:
            bucket = cls._passes_by_type.setdefault(ir_type, [])
            if pass_class in bucket:
                _LOGGER.debug(
                    "verification pass %s already registered for %s (idempotent)",
                    pass_class.__qualname__,
                    ir_type.__qualname__,
                )
                return
            bucket.append(pass_class)
            _LOGGER.debug(
                "registered verification pass %s for %s",
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

        """
        with cls._lock:
            seen: set[type[CompilerPass[Any, Any]]] = set()
            ordered: list[type[CompilerPass[Any, Any]]] = []
            for mro_cls in reversed(ir_type.__mro__):
                for pass_cls in cls._passes_by_type.get(mro_cls, ()):
                    if pass_cls in seen:
                        continue
                    seen.add(pass_cls)
                    ordered.append(pass_cls)
            return tuple(ordered)


class VerificationAnalysis(Analysis[Any, ValidationReport[Any]]):
    """Analysis that runs the verification pipeline for an IR.

    Builds a fresh :class:`ValidationManager` from
    :meth:`VerificationRegistry.get_passes_for(type(ir))` and runs it
    against the IR. The aggregated :class:`ValidationReport` is the
    analysis result. When no passes are registered for the IR's type
    (or any of its base classes), the report is empty.

    Has a stable ``analysis_name``, so results compose with
    :class:`AnalysisManager`: cached per IR identity, invalidated on
    pass-reported changes unless the pass declares this analysis
    preserved.
    """

    @override
    def run(self, ir: Any) -> ValidationReport[Any]:
        """Run every registered verification pass for ``type(ir)``.

        Args:
            ir: The IR to verify.

        Returns:
            The aggregated :class:`ValidationReport`.

        """
        pass_classes = VerificationRegistry.get_passes_for(type(ir))
        if not pass_classes:
            return ValidationReport()
        manager: ValidationManager[Any] = ValidationManager()
        for pass_class in pass_classes:
            manager.add(pass_class())
        return manager.validate(ir)


def register_verification(
    ir_type: type,
    name: str,
    description: str,
) -> Callable[[_PassClassT], _PassClassT]:
    """Register a ``CompilerPass`` subclass as a verification pass.

    The decorated class is:

    - added to the global pass registry under ``name`` and
      ``description``,
    - added to the :class:`VerificationRegistry` under ``ir_type``,
    - assigned ``_auto_verify = False``.

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
        pass_class._auto_verify = False
        return pass_class

    return _decorator


def run_verification(ir: Any) -> ValidationReport[Any]:
    """Run the verification pipeline for ``ir`` and return the report.

    Bypasses any :class:`AnalysisManager` and recomputes the report on
    every call.

    Args:
        ir: The IR to verify.

    Returns:
        The aggregated :class:`ValidationReport`. Empty when no passes
        are registered for ``type(ir)`` or any of its base classes.

    """
    return VerificationAnalysis().run(ir)
