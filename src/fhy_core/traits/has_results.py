"""`HasResults` trait."""

__all__ = ["HasResults"]

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

_ResultT_co = TypeVar("_ResultT_co", covariant=True)


@runtime_checkable
class HasResults(Protocol[_ResultT_co]):
    """Protocol for operation-like objects with ordered results."""

    def get_results(self) -> Sequence[_ResultT_co]:
        """Return the object's results in order."""
