"""`HasIdentifier` trait and mixin."""

__all__ = ["HasIdentifier", "HasIdentifierMixin"]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from fhy_core.identifier import Identifier


@runtime_checkable
class HasIdentifier(Protocol):
    """Protocol for objects that have a stable identifier."""

    @property
    def identifier(self) -> Identifier: ...


class HasIdentifierMixin(ABC):
    """Mixin for objects that have a stable identifier."""

    @property
    @abstractmethod
    def identifier(self) -> Identifier: ...
