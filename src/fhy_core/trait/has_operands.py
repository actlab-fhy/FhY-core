"""`HasOperands` trait and mixin."""

__all__ = ["HasOperands", "HasOperandsMixin"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

_OperandT_co = TypeVar("_OperandT_co", covariant=True)


@runtime_checkable
class HasOperands(Protocol[_OperandT_co]):
    """Protocol for operation-like objects with ordered operands."""

    @property
    def operands(self) -> Sequence[_OperandT_co]: ...


class HasOperandsMixin(ABC, Generic[_OperandT_co]):
    """Mixin for operation-like objects with ordered operands."""

    @property
    @abstractmethod
    def operands(self) -> Sequence[_OperandT_co]: ...
