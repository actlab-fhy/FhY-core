"""`HasIdentifier` trait."""

__all__ = ["HasIdentifier"]

from typing import Protocol, runtime_checkable

from fhy_core.identifier import Identifier


@runtime_checkable
class HasIdentifier(Protocol):
    """Protocol for objects that have a stable identifier."""

    def get_identifier(self) -> Identifier: ...
