"""`HasProvenance` trait."""

__all__ = ["HasProvenance"]

from typing import Protocol, runtime_checkable

from fhy_core.provenance import Provenance


@runtime_checkable
class HasProvenance(Protocol):
    """Tracks origin: source span, lowering steps, original node, etc."""

    def get_provenance(self) -> Provenance: ...
