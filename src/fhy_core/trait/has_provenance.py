"""`HasProvenance` trait and mixin."""

__all__ = ["HasProvenance", "HasProvenanceMixin"]

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from fhy_core.provenance import Provenance


@runtime_checkable
class HasProvenance(Protocol):
    """Tracks origin: source span, lowering steps, original node, etc."""

    def get_provenance(self) -> Provenance: ...


class HasProvenanceMixin(ABC):
    """Mixin for objects with provenance metadata."""

    @abstractmethod
    def get_provenance(self) -> Provenance: ...
