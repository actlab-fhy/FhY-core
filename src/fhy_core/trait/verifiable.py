"""`Verifiable` trait and mixin."""

from __future__ import annotations

__all__ = ["Verifiable", "VerifiableMixin", "VerificationError"]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from fhy_core.error import register_error


@register_error
class VerificationError(Exception):
    """Raised when verification detects an invalid structure."""


@runtime_checkable
class Verifiable(Protocol):
    """Protocol for objects that can self-verify structural invariants."""

    def verify(self) -> None:
        """Verify structural invariants.

        Raises:
            VerificationError: If structural verification fails.

        """


class VerifiableMixin(ABC):
    """Mixin for objects that can self-verify structural invariants."""

    @abstractmethod
    def verify(self) -> None:
        """Verify structural invariants.

        Raises:
            VerificationError: If structural verification fails.

        """
