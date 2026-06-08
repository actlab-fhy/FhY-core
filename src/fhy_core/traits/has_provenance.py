"""`HasProvenance` trait."""

__all__ = ["HasProvenance"]

from typing import Protocol, runtime_checkable

from fhy_core.provenance import Provenance


@runtime_checkable
class HasProvenance(Protocol):
    """Protocol for objects that carry provenance information.

    Provenance records an object's origin: source span, lowering steps,
    original node, etc.
    """

    def get_provenance(self) -> Provenance: ...
