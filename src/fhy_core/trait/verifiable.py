"""`Verifiable` trait and mixin.

This module hosts three related but independent symbols:

- :class:`Verifiable`: the protocol describing self-verifiable objects.
  Implementations return a :class:`ValidationReport`.
- :class:`VerifiableMixin`: a concrete mixin that backs ``verify()`` with
  the verification registry by default.
- :class:`VerificationError`: a general-purpose exception for fail-fast
  structural-correctness helpers (currently the type-unification and
  dispatch helpers in ``fhy_core.types.dispatch``). It is *not* raised by
  :meth:`Verifiable.verify` or by auto-verification on ``CompilerPass``;
  those entry points speak in reports.
"""

__all__ = ["Verifiable", "VerifiableMixin", "VerificationError"]

from abc import ABC
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fhy_core.error import register_error

if TYPE_CHECKING:
    from fhy_core.diagnostic import ValidationReport


@register_error
class VerificationError(Exception):
    """Raised by fail-fast structural-correctness helpers.

    This exception type is used by helpers that fail-fast when they detect
    a malformed structure (currently the type-unification and dispatch
    helpers in ``fhy_core.types.dispatch``). It is *not* raised by
    :meth:`Verifiable.verify` or by :class:`CompilerPass` auto-verification:
    those entry points return or carry a :class:`ValidationReport`.
    """


@runtime_checkable
class Verifiable(Protocol):
    """Protocol for objects that can self-verify structural invariants.

    Implementations return the aggregated :class:`ValidationReport` rather
    than raising. Callers inspect the report (``has_errors`` / ``errors``
    / ``format``) or convert it to an exception via ``raise_if_failed``,
    which raises :class:`ValidationFailedError`.
    """

    def verify(self) -> "ValidationReport[Any]":
        """Verify structural invariants and return the diagnostic report.

        Returns:
            A :class:`ValidationReport` aggregating every diagnostic
            produced by the underlying verification pipeline. An empty
            report means the object satisfies all checked invariants.

        """


class VerifiableMixin(ABC):
    """Mixin for objects that can self-verify structural invariants.

    The default :meth:`verify` body runs every verification pass registered
    for ``type(self)`` via
    :func:`fhy_core.pass_infrastructure.register_verification` (walking
    MRO so base-class passes apply) and returns the aggregated
    :class:`ValidationReport`. Subclasses may override :meth:`verify` to
    provide their own implementation; the override takes precedence over
    the registry-backed default.

    Abstractness contract: subclasses cannot be instantiated unless they
    satisfy at least one of the following:

    - they override :meth:`verify` with their own implementation, or
    - at least one verification pass is registered for the class (walking
      its MRO) at the time of the first attempted instantiation.

    Enforcement happens in :meth:`__new__`: the first instantiation of a
    given subclass checks both conditions and, if neither holds, raises
    :class:`TypeError`. Successful instantiations cache the "ok" status on
    the class so subsequent instantiations skip the check entirely. A
    negative result is *not* cached: callers who register verification
    passes for the class after the first failed instantiation see the
    class become instantiable on the next attempt.

    Instantiating ``VerifiableMixin`` directly is rejected by the same
    check (no override and no registered passes).
    """

    _verifiable_instantiation_ok: bool = False

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiableMixin":
        """Construct an instance after checking the abstractness contract.

        See the class docstring for the full contract. The check is skipped
        when the class has already been validated once.

        Args:
            *args: Forwarded to ``object.__new__`` / the next class in the
                MRO.
            **kwargs: Forwarded to ``object.__new__`` / the next class in
                the MRO.

        Returns:
            The newly created instance.

        Raises:
            TypeError: If the class neither overrides :meth:`verify` nor
                has any verification passes registered for it (walking
                MRO).

        """
        _ = (args, kwargs)
        if cls.__dict__.get("_verifiable_instantiation_ok", False):
            return super().__new__(cls)

        if cls.verify is not VerifiableMixin.verify:
            cls._verifiable_instantiation_ok = True
            return super().__new__(cls)

        # Lazy import: pass_infrastructure imports from this module.
        from fhy_core.pass_infrastructure.verification import (  # noqa: PLC0415
            VerificationRegistry,
        )

        if VerificationRegistry.get_passes_for(cls):
            cls._verifiable_instantiation_ok = True
            return super().__new__(cls)

        raise TypeError(
            f"Cannot instantiate {cls.__name__}: no verification passes are "
            f"registered for it and verify() is not overridden. Register a "
            f"verification pass via @register_verification, or implement "
            f"verify() directly."
        )

    def verify(self) -> "ValidationReport[Any]":
        """Run every registered verification pass for ``type(self)``.

        The :meth:`__new__` abstractness check verified at instantiation
        time that either :meth:`verify` was overridden or at least one
        verification pass was registered for ``type(self)`` (walking its
        MRO). That guarantee is point-in-time; if the verification registry
        is mutated after the instance was constructed, this default
        implementation may legitimately run with no registered passes and
        return an empty aggregated report.

        Returns:
            The aggregated :class:`ValidationReport`. Callers inspect the
            diagnostics or call :meth:`ValidationReport.raise_if_failed`
            to convert errors into a :class:`ValidationFailedError`.

        """
        # Lazy import: pass_infrastructure imports from this module.
        from fhy_core.pass_infrastructure.verification import (  # noqa: PLC0415
            run_verification,
        )

        return run_verification(self)
