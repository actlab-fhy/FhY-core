"""`HasOperands` trait."""

__all__ = ["HasOperands"]

from collections.abc import Sequence
from typing import Protocol, TypeVar, runtime_checkable

_OperandT_co = TypeVar("_OperandT_co", covariant=True)


@runtime_checkable
class HasOperands(Protocol[_OperandT_co]):
    """Protocol for operation-like objects with ordered operands."""

    def get_operands(self) -> Sequence[_OperandT_co]:
        """Return the object's operands in order."""
