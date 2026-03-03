"""`HasResults` trait and mixin."""

from __future__ import annotations

__all__ = ["HasResults", "HasResultsMixin"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

_ResultT_co = TypeVar("_ResultT_co", covariant=True)


@runtime_checkable
class HasResults(Protocol[_ResultT_co]):
    """Protocol for operation-like objects with ordered results."""

    @property
    def results(self) -> Sequence[_ResultT_co]: ...


class HasResultsMixin(ABC, Generic[_ResultT_co]):
    """Mixin for operation-like objects with ordered results."""

    @property
    @abstractmethod
    def results(self) -> Sequence[_ResultT_co]: ...
